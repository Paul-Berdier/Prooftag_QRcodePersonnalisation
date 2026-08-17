from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import textwrap
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _family(prompt_id: str) -> str:
    match = re.match(r"e026w_([^_]+)_", prompt_id)
    return match.group(1) if match else "other"


def _record_entry(record: Any, probability: float | None) -> dict[str, Any]:
    parameters = record.parameters
    metadata = record.metadata
    targets = record.targets
    return {
        "trial_id": record.trial_id,
        "prompt_id": record.prompt_id,
        "prompt_text": record.prompt_text,
        "prompt_family": _family(record.prompt_id),
        "method_id": str(metadata.get("method_id") or parameters.get("id") or "unknown"),
        "output_variant": parameters.get("output_variant"),
        "seed": _finite(metadata.get("seed")),
        "generation_run_id": metadata.get("generation_run_id"),
        "qr_success": _finite(targets.get("qr_success")),
        "qr_tolerance": _finite(targets.get("qr_tolerance")),
        "clip_aesthetic": _finite(targets.get("clip_aesthetic")),
        "clip_score": _finite(targets.get("clip_score")),
        "hpsv2_1": _finite(targets.get("hpsv2_1")),
        "saturation_risk": _finite(targets.get("saturation_risk")),
        "predicted_qr_probability": _finite(probability),
    }


def _round_robin_prompts(entries: Sequence[dict[str, Any]], limit: int) -> list[str]:
    families: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        prompt_id = str(entry["prompt_id"])
        family = str(entry["prompt_family"])
        if prompt_id not in families[family]:
            families[family].append(prompt_id)
    for prompts in families.values():
        prompts.sort()
    selected: list[str] = []
    while len(selected) < limit and any(families.values()):
        for family in sorted(families):
            if families[family] and len(selected) < limit:
                selected.append(families[family].pop(0))
    return selected


def select_advisor_gallery(
    records: Sequence[Any],
    *,
    validation_predictions: Sequence[Mapping[str, Any]] | None = None,
    comparison_method_ids: Sequence[str],
    comparison_prompt_count: int = 8,
    preferred_seed: int = 113_001,
    section_size: int = 8,
) -> list[dict[str, Any]]:
    """Select a compact, auditable visual sample from advisor records.

    The comparison section always uses the same requested seed for every method when
    available. Other sections keep at most one image per prompt so a prolific retry
    cannot dominate the visual report.
    """

    predictions = list(validation_predictions or [])
    if predictions and len(predictions) != len(records):
        raise ValueError("validation predictions must align one-to-one with advisor records")
    entries = [
        _record_entry(
            record,
            predictions[index].get("calibrated_probability") if predictions else None,
        )
        for index, record in enumerate(records)
    ]
    entries = [entry for entry in entries if entry.get("generation_run_id")]

    by_prompt_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_prompt_method[(entry["prompt_id"], entry["method_id"])].append(entry)

    eligible = []
    method_ids = tuple(comparison_method_ids)
    for prompt_id in sorted({entry["prompt_id"] for entry in entries}):
        if all((prompt_id, method_id) in by_prompt_method for method_id in method_ids):
            eligible.append(
                next(
                    entry
                    for entry in entries
                    if entry["prompt_id"] == prompt_id
                )
            )
    comparison_prompts = _round_robin_prompts(eligible, comparison_prompt_count)

    selected: list[dict[str, Any]] = []
    for prompt_id in comparison_prompts:
        for method_id in method_ids:
            candidates = by_prompt_method[(prompt_id, method_id)]
            candidates = sorted(
                candidates,
                key=lambda item: (
                    item.get("seed") == preferred_seed,
                    -(item.get("seed") or 0),
                ),
                reverse=True,
            )
            selected.append({**candidates[0], "section": "comparison"})

    def unique_prompts(
        candidates: Sequence[dict[str, Any]], section: str
    ) -> list[dict[str, Any]]:
        output = []
        seen = set()
        for candidate in candidates:
            if candidate["prompt_id"] in seen:
                continue
            seen.add(candidate["prompt_id"])
            output.append({**candidate, "section": section})
            if len(output) == section_size:
                break
        return output

    successful = sorted(
        (entry for entry in entries if (entry.get("qr_success") or 0) >= 0.5),
        key=lambda item: (
            item.get("hpsv2_1") or -math.inf,
            item.get("clip_aesthetic") or -math.inf,
            item.get("qr_tolerance") or -math.inf,
        ),
        reverse=True,
    )
    failures = sorted(
        (entry for entry in entries if (entry.get("qr_success") or 0) < 0.5),
        key=lambda item: (
            item.get("clip_aesthetic") or -math.inf,
            item.get("hpsv2_1") or -math.inf,
        ),
        reverse=True,
    )
    uncertain = sorted(
        (
            entry
            for entry in entries
            if entry.get("predicted_qr_probability") is not None
        ),
        key=lambda item: abs(item["predicted_qr_probability"] - 0.5),
    )
    selected.extend(unique_prompts(successful, "best_scannable"))
    selected.extend(unique_prompts(failures, "aesthetic_failures"))
    selected.extend(unique_prompts(uncertain, "uncertain"))
    return selected


def _safe_name(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip("-")[:80] or "item"


def download_advisor_gallery(
    entries: Sequence[Mapping[str, Any]],
    *,
    api_url: str,
    output_dir: Path,
    timeout: float = 30.0,
    fetcher: Callable[[str], bytes] | None = None,
) -> list[dict[str, Any]]:
    """Download final generation images without failing the model training on a 404."""

    output_dir.mkdir(parents=True, exist_ok=True)

    def fetch(run_id: str) -> bytes:
        request = Request(
            f"{api_url.rstrip('/')}/v1/generations/{run_id}/image",
            headers={"Accept": "image/png,image/*"},
        )
        with urlopen(request, timeout=timeout) as response:
            return response.read()

    fetch_image = fetcher or fetch
    downloaded: dict[str, tuple[str | None, str | None, str | None]] = {}
    result: list[dict[str, Any]] = []
    for index, source in enumerate(entries, start=1):
        entry = dict(source)
        run_id = str(entry.get("generation_run_id") or "")
        if run_id not in downloaded:
            filename = (
                f"{index:03d}-{_safe_name(entry.get('section'))}-"
                f"{_safe_name(entry.get('prompt_id'))}-"
                f"{_safe_name(entry.get('method_id'))}.png"
            )
            path = output_dir / filename
            try:
                image = Image.open(BytesIO(fetch_image(run_id))).convert("RGB")
                image.save(path, format="PNG", optimize=True)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                downloaded[run_id] = (str(path), None, digest)
            except Exception as exc:  # a missing old artifact must remain visible in the audit
                downloaded[run_id] = (None, f"{type(exc).__name__}: {exc}", None)
        local_path, error, digest = downloaded[run_id]
        entry["local_image"] = local_path
        entry["download_error"] = error
        entry["image_sha256"] = digest
        result.append(entry)
    return result


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _metric(value: Any, digits: int = 2) -> str:
    result = _finite(value)
    return "-" if result is None else f"{result:.{digits}f}"


def render_advisor_contact_sheet(
    entries: Sequence[Mapping[str, Any]],
    *,
    title: str,
    output_path: Path,
    columns: int = 4,
    tile_size: int = 256,
) -> Path:
    rows = max(1, math.ceil(len(entries) / columns))
    label_height = 116
    cell_width = tile_size + 24
    cell_height = tile_size + label_height + 20
    title_height = 64
    canvas = Image.new(
        "RGB",
        (columns * cell_width + 24, rows * cell_height + title_height + 16),
        "#0b1119",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 16), title, fill="#f4f7fb", font=_font(24, bold=True))

    for index, entry in enumerate(entries):
        row, column = divmod(index, columns)
        left = 12 + column * cell_width
        top = title_height + row * cell_height
        success = (entry.get("qr_success") or 0) >= 0.5
        border = "#36d399" if success else "#fb7185"
        draw.rounded_rectangle(
            (left, top, left + cell_width - 8, top + cell_height - 8),
            radius=12,
            fill="#111927",
            outline=border,
            width=3,
        )
        image_box = (left + 8, top + 8, left + 8 + tile_size, top + 8 + tile_size)
        local_image = entry.get("local_image")
        if local_image and Path(str(local_image)).is_file():
            with Image.open(str(local_image)) as source:
                preview = ImageOps.contain(source.convert("RGB"), (tile_size, tile_size))
            background = Image.new("RGB", (tile_size, tile_size), "white")
            background.paste(
                preview,
                ((tile_size - preview.width) // 2, (tile_size - preview.height) // 2),
            )
            canvas.paste(background, image_box[:2])
        else:
            draw.rectangle(image_box, fill="#1f2937")
            draw.multiline_text(
                (left + 24, top + tile_size // 2 - 12),
                "IMAGE INDISPONIBLE",
                fill="#fca5a5",
                font=_font(14, bold=True),
            )

        text_left = left + 10
        text_top = top + tile_size + 14
        lines = [
            textwrap.shorten(str(entry.get("prompt_id") or ""), width=36),
            textwrap.shorten(
                f"{entry.get('selection_profile') or 'candidate'} | "
                f"{entry.get('source_method_id') or entry.get('method_id') or ''}",
                width=36,
            ),
            f"seed {_metric(entry.get('seed'), 0)} | {entry.get('output_variant') or '-'} | "
            f"QR {'OK' if success else 'ECHEC'} "
            f"| tol {_metric(entry.get('qr_tolerance'))}",
            f"AES {_metric(entry.get('clip_aesthetic'))} | CLIP {_metric(entry.get('clip_score'))} "
            f"| HPS {_metric(entry.get('hpsv2_1'), 3)}",
            f"P(QR) {_metric(entry.get('predicted_qr_probability'), 3)} "
            f"| saturation {_metric(entry.get('saturation_risk'), 3)}",
        ]
        for offset, line in enumerate(lines):
            draw.text(
                (text_left, text_top + offset * 19),
                line,
                fill="#f8fafc" if offset < 2 else "#cbd5e1",
                font=_font(13, bold=offset == 0),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return output_path


def write_gallery_index(entries: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = []
    for entry in entries:
        row = dict(entry)
        local_image = row.get("local_image")
        if local_image:
            try:
                row["local_image"] = str(Path(str(local_image)).relative_to(output_dir))
            except ValueError:
                row["local_image"] = str(local_image)
        serializable.append(row)
    fields = sorted({key for entry in serializable for key in entry})
    with (output_dir / "gallery-index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(serializable)
    audit = {
        "selected": len(serializable),
        "downloaded": sum(not entry.get("download_error") for entry in serializable),
        "missing": sum(bool(entry.get("download_error")) for entry in serializable),
        "unique_generation_runs": len(
            {entry.get("generation_run_id") for entry in serializable}
        ),
        "unique_images_downloaded": len(
            {
                entry.get("image_sha256")
                for entry in serializable
                if not entry.get("download_error") and entry.get("image_sha256")
            }
        ),
        "duplicate_images_downloaded": (
            sum(not entry.get("download_error") for entry in serializable)
            - len(
                {
                    entry.get("image_sha256")
                    for entry in serializable
                    if not entry.get("download_error")
                    and entry.get("image_sha256")
                }
            )
        ),
        "sections": {
            section: sum(entry.get("section") == section for entry in serializable)
            for section in sorted({str(entry.get("section")) for entry in serializable})
        },
    }
    (output_dir / "gallery-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
