"""Validation et agrégation des captures téléphone E045.

Le payload en clair n'est pas stocké. Chaque succès signifie que l'application a
restitué le payload attendu et que son hash correspond au hash de contexte.

Le fichier d'entrée peut être enrichi progressivement. Les lignes invalides sont
isolées avec une raison; elles ne font jamais échouer ni contaminer les lignes
valides.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .resilient_experiment import atomic_write_json, atomic_write_text

SCHEMA = "e045-phone-captures-v1"

COLUMNS = (
    "capture_id",
    "image_sha256",
    "image_path",
    "experiment_id",
    "prompt_id",
    "device_id",
    "device_model",
    "os_name",
    "os_version",
    "scanner_app",
    "scanner_version",
    "captured_at_utc",
    "display_device",
    "display_width_px",
    "display_height_px",
    "display_brightness_pct",
    "ambient_lux",
    "distance_cm",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "attempt_index",
    "frames_observed",
    "time_to_decode_ms",
    "decoded_exact",
    "decoded_payload_sha256",
    "expected_payload_sha256",
    "notes",
)

REQUIRED = {
    "capture_id",
    "image_sha256",
    "device_id",
    "device_model",
    "scanner_app",
    "captured_at_utc",
    "attempt_index",
    "decoded_exact",
    "expected_payload_sha256",
}


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    normalized: dict[str, Any]
    reasons: tuple[str, ...]


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "oui", "y"}:
        return True
    if text in {"0", "false", "no", "non", "n", ""}:
        return False
    raise ValueError(f"booléen invalide: {value!r}")


def _float(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    number = float(text.replace(",", "."))
    if not math.isfinite(number):
        raise ValueError("valeur non finie")
    if minimum is not None and number < minimum:
        raise ValueError(f"valeur < {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"valeur > {maximum}")
    return number


def _int(value: Any, *, minimum: int | None = None, maximum: int | None = None) -> int | None:
    number = _float(value, minimum=minimum, maximum=maximum)
    if number is None:
        return None
    if not float(number).is_integer():
        raise ValueError("entier attendu")
    return int(number)


def _sha256(value: Any, *, required: bool = False) -> str | None:
    text = str(value).strip().lower()
    if not text:
        if required:
            raise ValueError("SHA-256 requis")
        return None
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError("SHA-256 invalide")
    return text


def _timestamp(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("timestamp requis")
    normalized = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timezone requise")
    return parsed.astimezone(UTC).isoformat()


def validate_row(row: Mapping[str, Any]) -> ValidationResult:
    reasons: list[str] = []
    normalized = {column: str(row.get(column, "")).strip() for column in COLUMNS}

    for field in REQUIRED:
        if not normalized.get(field):
            reasons.append(f"{field}: valeur requise")

    def apply(field: str, function, *args, **kwargs):
        try:
            normalized[field] = function(normalized.get(field), *args, **kwargs)
        except (TypeError, ValueError) as exc:
            reasons.append(f"{field}: {exc}")

    apply("capture_id", lambda value: str(uuid.UUID(str(value))))
    apply("image_sha256", _sha256, required=True)
    apply("expected_payload_sha256", _sha256, required=True)
    apply("decoded_payload_sha256", _sha256, required=False)
    apply("captured_at_utc", _timestamp)
    apply("attempt_index", _int, minimum=1, maximum=1000)
    apply("frames_observed", _int, minimum=0, maximum=100000)
    apply("time_to_decode_ms", _float, minimum=0.0, maximum=120000.0)
    apply("decoded_exact", _bool)
    apply("display_width_px", _int, minimum=1, maximum=20000)
    apply("display_height_px", _int, minimum=1, maximum=20000)
    apply("display_brightness_pct", _float, minimum=0.0, maximum=100.0)
    apply("ambient_lux", _float, minimum=0.0, maximum=200000.0)
    apply("distance_cm", _float, minimum=1.0, maximum=500.0)
    for field in ("yaw_deg", "pitch_deg", "roll_deg"):
        apply(field, _float, minimum=-180.0, maximum=180.0)

    if normalized.get("decoded_exact"):
        actual = normalized.get("decoded_payload_sha256")
        expected = normalized.get("expected_payload_sha256")
        if actual is None:
            reasons.append("decoded_payload_sha256: requis quand decoded_exact=1")
        elif actual != expected:
            reasons.append("decoded_exact: hash décodé différent du hash attendu")
    else:
        # Un échec peut laisser le champ vide ou contenir un mauvais hash pour audit.
        pass

    if not normalized.get("device_id"):
        reasons.append("device_id: valeur requise")
    if not normalized.get("scanner_app"):
        reasons.append("scanner_app: valeur requise")

    return ValidationResult(
        valid=not reasons,
        normalized=normalized,
        reasons=tuple(reasons),
    )


def wilson_interval(successes: int, attempts: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if attempts <= 0:
        return 0.0, 0.0
    p = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = (p + z * z / (2.0 * attempts)) / denominator
    radius = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * attempts)) / attempts)
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Iterable[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns or sorted({key for row in rows for key in row}))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(valid_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_image_device: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in valid_rows:
        scanner_key = f"{row['scanner_app']}@{row.get('scanner_version') or 'unknown'}"
        by_image_device[(row["image_sha256"], row["device_id"], scanner_key)].append(row)
        by_image[row["image_sha256"]].append(row)

    device_rows: list[dict[str, Any]] = []
    for (image_sha, device_id, scanner_key), rows in sorted(by_image_device.items()):
        attempts = len(rows)
        successes = sum(bool(row["decoded_exact"]) for row in rows)
        low, high = wilson_interval(successes, attempts)
        times = [
            float(row["time_to_decode_ms"])
            for row in rows
            if row.get("decoded_exact") and row.get("time_to_decode_ms") is not None
        ]
        device_rows.append(
            {
                "image_sha256": image_sha,
                "device_id": device_id,
                "device_model": rows[0].get("device_model"),
                "scanner": scanner_key,
                "attempts": attempts,
                "successes": successes,
                "success_rate": successes / attempts,
                "wilson_low_95": low,
                "wilson_high_95": high,
                "median_time_to_decode_ms": statistics.median(times) if times else None,
                "phone_pass_2_of_3": attempts >= 3 and successes >= 2,
            }
        )

    image_rows: list[dict[str, Any]] = []
    for image_sha, rows in sorted(by_image.items()):
        attempts = len(rows)
        successes = sum(bool(row["decoded_exact"]) for row in rows)
        low, high = wilson_interval(successes, attempts)
        device_groups = {
            (row["device_id"], row["scanner_app"], row.get("scanner_version") or "")
            for row in rows
        }
        per_device = [
            item
            for item in device_rows
            if item["image_sha256"] == image_sha
        ]
        image_rows.append(
            {
                "image_sha256": image_sha,
                "image_path": next((row.get("image_path") for row in rows if row.get("image_path")), ""),
                "experiment_id": next((row.get("experiment_id") for row in rows if row.get("experiment_id")), ""),
                "prompt_id": next((row.get("prompt_id") for row in rows if row.get("prompt_id")), ""),
                "attempts": attempts,
                "successes": successes,
                "success_rate": successes / attempts,
                "wilson_low_95": low,
                "wilson_high_95": high,
                "device_group_count": len(device_groups),
                "all_devices_pass_2_of_3": bool(per_device) and all(
                    bool(item["phone_pass_2_of_3"]) for item in per_device
                ),
                "minimum_device_success_rate": min(
                    (float(item["success_rate"]) for item in per_device),
                    default=0.0,
                ),
            }
        )
    return image_rows, device_rows


def import_captures(input_csv: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    seen_capture_ids: set[str] = set()

    with input_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"colonnes requises absentes: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            result = validate_row(row)
            if result.valid and result.normalized["capture_id"] in seen_capture_ids:
                result = ValidationResult(
                    valid=False,
                    normalized=result.normalized,
                    reasons=("capture_id dupliqué",),
                )
            if result.valid:
                seen_capture_ids.add(result.normalized["capture_id"])
                valid_rows.append(result.normalized)
            else:
                rejected_rows.append(
                    {
                        "line_number": line_number,
                        **{key: row.get(key, "") for key in COLUMNS},
                        "rejection_reasons": " | ".join(result.reasons),
                    }
                )

    image_rows, device_rows = aggregate_rows(valid_rows)
    _write_csv(output_dir / "phone-captures-valid.csv", valid_rows, COLUMNS)
    _write_csv(output_dir / "phone-captures-rejected.csv", rejected_rows)
    _write_csv(output_dir / "phone-labels-by-image.csv", image_rows)
    _write_csv(output_dir / "phone-labels-by-device.csv", device_rows)

    summary = {
        "schema": SCHEMA,
        "input_csv": str(input_csv),
        "input_sha256": hashlib.sha256(input_csv.read_bytes()).hexdigest(),
        "valid_capture_count": len(valid_rows),
        "rejected_capture_count": len(rejected_rows),
        "labeled_image_count": len(image_rows),
        "device_image_group_count": len(device_rows),
        "images_all_devices_pass_2_of_3": sum(
            bool(row["all_devices_pass_2_of_3"]) for row in image_rows
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(output_dir / "phone-label-summary.json", summary)
    return summary


def write_template(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(path, [], COLUMNS)
