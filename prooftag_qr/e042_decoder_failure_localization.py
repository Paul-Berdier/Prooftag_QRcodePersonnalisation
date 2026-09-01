"""E042 — localize why E041 module improvements still fail real QR decoders.

E042 is a *diagnostic* experiment. It does not optimize a new QR and it does not
change E041's scientific verdict. It reuses a small preregistered subset of E041
Phase-A latents, re-decodes them once with the pinned VAE, then asks progressively
more constrained questions:

1. Does a detector see a QR in the artistic raster?
2. Does correcting only the quiet-zone geometry help?
3. Does scanner-native grayscale/Otsu/adaptive binarization help?
4. If the known 29x29 module grid is sampled and re-rendered canonically, do the
   inferred bits decode? This distinguishes bit errors from texture/grid detection.
5. Which QR functional sub-regions still contain bit errors (finder, separator,
   timing, alignment, format information, fixed dark module)?

QR-Verify remains authoritative for decode evidence. One-shot QR-Verify calls in
E042 are explicitly diagnostic. Any rescued candidates are rechecked with the
project's conservative repeated scorer. No result authorizes production or
cross-prompt generalization.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

EXPERIMENT = "e042-decoder-failure-localization-v1"
E041_REQUIRED_EXPERIMENT = "e041-gamma-functional-pattern-frontier-v1"
PAYLOAD = "https://ptag.io/t/e041"
ERROR_CORRECTION = "M"
QR_VERSION = 3
QR_MASK_PATTERN = 4
QR_MODULE_SIZE = 20
QR_PADDING_PX = 78
QR_CORE_MODULES = 29
QR_CORE_PX = QR_CORE_MODULES * QR_MODULE_SIZE
QR_CANVAS_PX = QR_CORE_PX + 2 * QR_PADDING_PX


@dataclass(frozen=True, slots=True)
class SelectedState:
    state_id: str
    gamma: float
    iteration: int

    @property
    def recipe(self) -> str:
        return f"e041_gamma_{int(self.gamma):04d}_r200_i08"

    @property
    def variant(self) -> str:
        return f"{self.recipe}__i{self.iteration:02d}"


# Frozen before E042 execution. These states sample the parent, the numerically
# interesting gamma=500 trajectory, and two higher-gamma projected trajectories.
SELECTED_STATES: tuple[SelectedState, ...] = (
    SelectedState("parent", 50.0, 0),
    SelectedState("g500_i01", 500.0, 1),
    SelectedState("g500_i02", 500.0, 2),
    SelectedState("g500_i04", 500.0, 4),
    SelectedState("g500_i08", 500.0, 8),
    SelectedState("g1000_i02", 1000.0, 2),
    SelectedState("g1000_i03", 1000.0, 3),
    SelectedState("g2000_i02", 2000.0, 2),
    SelectedState("g2000_i04", 2000.0, 4),
)

DIAGNOSTIC_QRVERIFY_VARIANTS: tuple[str, ...] = (
    "current-scan-ready",
    "raw-vae",
    "exact-qz-adaptive",
    "exact-qz-white",
    "otsu",
    "adaptive",
    "grid-mean-050",
    "grid-mean-best",
    "grid-center-best",
)


# ---------------------------------------------------------------------------
# Small I/O helpers
# ---------------------------------------------------------------------------

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        if value and not value.endswith("\n"):
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", optimize=False, compress_level=9)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_sha256(image: Image.Image) -> str:
    from .quality import image_sha256

    return image_sha256(image.convert("RGB"))


def _manifest(root: Path) -> list[dict[str, Any]]:
    excluded = {"e042-artifact-manifest.json"}
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Preconditions and E041 bindings
# ---------------------------------------------------------------------------

def _load_e041_verdict(e041_results_dir: Path) -> dict[str, Any]:
    path = e041_results_dir / "verdict.json"
    if not path.is_file():
        raise FileNotFoundError(f"E041 verdict missing: {path}")
    verdict = json.loads(path.read_text(encoding="utf-8"))
    if verdict.get("experiment") != E041_REQUIRED_EXPERIMENT:
        raise RuntimeError(f"unexpected E041 experiment: {verdict.get('experiment')!r}")
    if int(verdict.get("phase_a_checkpoint_count", 0)) != 54:
        raise RuntimeError("E042 requires the complete 54-checkpoint E041 Phase A")
    if verdict.get("generalization_authorized") is not False:
        raise RuntimeError("E042 expects E041 generalization to remain unauthorized")
    if verdict.get("production_ready") is not False:
        raise RuntimeError("E042 expects E041 production_ready=false")
    return verdict


def _load_e041_phase_a_rows(e041_results_dir: Path) -> dict[str, dict[str, Any]]:
    path = e041_results_dir / "phase-a-scoring/comparison.json"
    if not path.is_file():
        raise FileNotFoundError(f"E041 Phase-A comparison missing: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("variant")): dict(row) for row in rows}


def _state_paths(e041_results_dir: Path, state: SelectedState) -> tuple[Path, Path]:
    root = e041_results_dir / "phase-a-trajectories" / state.recipe
    return (
        root / "images" / f"iteration-{state.iteration:03d}.png",
        root / "latents" / f"iteration-{state.iteration:03d}.safetensors",
    )


def _plan_document(*, source_commit: str, e041_results_dir: Path) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "e041_results_dir": str(e041_results_dir),
        "payload": PAYLOAD,
        "qr": {
            "version": QR_VERSION,
            "mask_pattern": QR_MASK_PATTERN,
            "error_correction": ERROR_CORRECTION,
            "module_size_px": QR_MODULE_SIZE,
            "padding_px": QR_PADDING_PX,
            "core_modules": QR_CORE_MODULES,
            "core_px": QR_CORE_PX,
            "canvas_px": QR_CANVAS_PX,
        },
        "selected_states": [asdict(item) | {"recipe": item.recipe, "variant": item.variant} for item in SELECTED_STATES],
        "diagnostic_qrverify_variants": list(DIAGNOSTIC_QRVERIFY_VARIANTS),
        "diagnostic_only": True,
        "production_ready": False,
        "generalization_authorized": False,
    }


# ---------------------------------------------------------------------------
# Exact DiffQRCoder raster helpers
# ---------------------------------------------------------------------------

def _exact_adaptive_quiet_color(image: Image.Image, minimum_luminance: float = 0.90) -> tuple[int, int, int]:
    source = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask = np.ones(source.shape[:2], dtype=bool)
    p = QR_PADDING_PX
    mask[p : p + QR_CORE_PX, p : p + QR_CORE_PX] = False
    sampled = source[mask]
    color = np.median(sampled, axis=0) if sampled.size else np.array((255.0, 255.0, 255.0), dtype=np.float32)
    luminance = float(np.dot(color, np.array((0.299, 0.587, 0.114), dtype=np.float32))) / 255.0
    if luminance < minimum_luminance:
        blend = (minimum_luminance - luminance) / max(1e-6, 1.0 - luminance)
        color = color * (1.0 - blend) + 255.0 * blend
    return tuple(int(value) for value in np.rint(color).clip(0, 255))


def _restore_exact_quiet_zone(image: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    source = image.convert("RGB")
    if source.size != (QR_CANVAS_PX, QR_CANVAS_PX):
        raise ValueError(f"E042 expects {QR_CANVAS_PX}x{QR_CANVAS_PX}, got {source.size}")
    p = QR_PADDING_PX
    output = Image.new("RGB", source.size, color)
    output.paste(source.crop((p, p, p + QR_CORE_PX, p + QR_CORE_PX)), (p, p))
    return output


def _quiet_zone_overwrite_metrics(raw: Image.Image, current: Image.Image) -> dict[str, Any]:
    raw_a = np.asarray(raw.convert("RGB"), dtype=np.int16)
    cur_a = np.asarray(current.convert("RGB"), dtype=np.int16)
    if raw_a.shape != cur_a.shape or raw_a.shape[:2] != (QR_CANVAS_PX, QR_CANVAS_PX):
        raise ValueError("quiet-zone comparison requires equal 736px rasters")
    changed = np.any(raw_a != cur_a, axis=2)
    abs_delta = np.abs(raw_a - cur_a).mean(axis=2)
    p = QR_PADDING_PX
    core = np.zeros_like(changed)
    core[p : p + QR_CORE_PX, p : p + QR_CORE_PX] = True
    edge = np.zeros_like(changed)
    # The historical proportional 37-cell restoration starts near pixel 80 and
    # therefore may overwrite roughly two pixels inside the exact 78px crop.
    q = 2
    edge[p : p + q, p : p + QR_CORE_PX] = True
    edge[p + QR_CORE_PX - q : p + QR_CORE_PX, p : p + QR_CORE_PX] = True
    edge[p : p + QR_CORE_PX, p : p + q] = True
    edge[p : p + QR_CORE_PX, p + QR_CORE_PX - q : p + QR_CORE_PX] = True
    edge &= core
    return {
        "raw_current_mae_255": float(abs_delta.mean()),
        "changed_pixel_ratio_total": float(changed.mean()),
        "changed_pixel_ratio_inside_exact_core": float(changed[core].mean()),
        "changed_pixel_ratio_exact_core_edge_2px": float(changed[edge].mean()),
        "changed_pixels_inside_exact_core": int(changed[core].sum()),
        "exact_core_pixel_count": int(core.sum()),
        "legacy_quiet_zone_overwrites_exact_core": bool(changed[edge].mean() > 0.05),
    }


def _render_matrix(matrix: np.ndarray) -> Image.Image:
    if matrix.shape != (QR_CORE_MODULES, QR_CORE_MODULES):
        raise ValueError(f"expected {QR_CORE_MODULES}x{QR_CORE_MODULES} matrix, got {matrix.shape}")
    canvas = np.full((QR_CANVAS_PX, QR_CANVAS_PX), 255, dtype=np.uint8)
    p = QR_PADDING_PX
    for row in range(QR_CORE_MODULES):
        for col in range(QR_CORE_MODULES):
            if bool(matrix[row, col]):
                y0 = p + row * QR_MODULE_SIZE
                x0 = p + col * QR_MODULE_SIZE
                canvas[y0 : y0 + QR_MODULE_SIZE, x0 : x0 + QR_MODULE_SIZE] = 0
    return Image.fromarray(canvas, mode="L").convert("RGB")


def _exact_reference_image(blueprint: Any) -> Image.Image:
    border = int(blueprint.border)
    matrix = blueprint.matrix[border:-border, border:-border] if border else blueprint.matrix
    return _render_matrix(np.asarray(matrix, dtype=bool))


# ---------------------------------------------------------------------------
# Module sampling and QR sub-regions
# ---------------------------------------------------------------------------

def _sample_modules(image: Image.Image, center_fraction: float = 0.40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < center_fraction <= 1:
        raise ValueError("center_fraction must be in (0, 1]")
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if gray.shape != (QR_CANVAS_PX, QR_CANVAS_PX):
        raise ValueError("module sampling requires 736x736 raster")
    p = QR_PADDING_PX
    means = np.empty((QR_CORE_MODULES, QR_CORE_MODULES), dtype=np.float32)
    centers = np.empty_like(means)
    stds = np.empty_like(means)
    center_size = max(1, round(QR_MODULE_SIZE * center_fraction))
    center_offset = (QR_MODULE_SIZE - center_size) // 2
    for row in range(QR_CORE_MODULES):
        for col in range(QR_CORE_MODULES):
            y0 = p + row * QR_MODULE_SIZE
            x0 = p + col * QR_MODULE_SIZE
            region = gray[y0 : y0 + QR_MODULE_SIZE, x0 : x0 + QR_MODULE_SIZE]
            center = region[
                center_offset : center_offset + center_size,
                center_offset : center_offset + center_size,
            ]
            means[row, col] = float(region.mean())
            centers[row, col] = float(center.mean())
            stds[row, col] = float(region.std())
    return means, centers, stds


def _best_threshold(values: np.ndarray, target_dark: np.ndarray) -> tuple[float, int]:
    candidates = np.linspace(0.25, 0.75, 101, dtype=np.float32)
    scored: list[tuple[int, float, float]] = []
    for threshold in candidates:
        predicted = values < threshold
        errors = int(np.not_equal(predicted, target_dark).sum())
        scored.append((errors, abs(float(threshold) - 0.5), float(threshold)))
    errors, _, threshold = min(scored)
    return threshold, errors


def _region_masks(blueprint: Any) -> dict[str, np.ndarray]:
    from .qr import functional_pattern_mask

    border = int(blueprint.border)
    size = QR_CORE_MODULES
    functional = functional_pattern_mask(blueprint)
    functional = functional[border:-border, border:-border] if border else functional
    functional = np.asarray(functional, dtype=bool)
    if functional.shape != (size, size):
        raise ValueError(f"functional mask is {functional.shape}, expected {(size, size)}")

    finder = np.zeros((size, size), dtype=bool)
    separator = np.zeros_like(finder)
    timing = np.zeros_like(finder)
    alignment = np.zeros_like(finder)
    fmt = np.zeros_like(finder)
    fixed_dark = np.zeros_like(finder)

    finder[0:7, 0:7] = True
    finder[0:7, size - 7 : size] = True
    finder[size - 7 : size, 0:7] = True

    separator[7, 0:8] = True
    separator[0:8, 7] = True
    separator[7, size - 8 : size] = True
    separator[0:8, size - 8] = True
    separator[size - 8, 0:8] = True
    separator[size - 8 : size, 7] = True

    timing[6, 8 : size - 8] = True
    timing[8 : size - 8, 6] = True

    # Version 3 has one non-overlapping alignment pattern centred on (22,22).
    alignment[20:25, 20:25] = True

    # Exact QR format-information coordinates, matching qrcode.QRCode.setup_type_info.
    for i in range(15):
        if i < 6:
            row = i
        elif i < 8:
            row = i + 1
        else:
            row = size - 15 + i
        fmt[row, 8] = True

        if i < 8:
            col = size - i - 1
        elif i < 9:
            col = 15 - i
        else:
            col = 15 - i - 1
        fmt[8, col] = True

    fixed_dark[size - 8, 8] = True
    data = ~functional
    return {
        "finder": finder,
        "separator": separator,
        "timing": timing,
        "alignment": alignment,
        "format": fmt,
        "fixed_dark": fixed_dark,
        "functional_all": functional,
        "data": data,
    }


def _region_error_metrics(predicted: np.ndarray, target: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    errors = np.not_equal(predicted, target)
    output: dict[str, Any] = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        error_count = int(errors[mask].sum()) if count else 0
        output[f"{name}_module_count"] = count
        output[f"{name}_error_count"] = error_count
        output[f"{name}_error_rate"] = error_count / count if count else 0.0
    output["total_error_count"] = int(errors.sum())
    output["total_error_rate"] = float(errors.mean())
    return output


def _margin_metrics(values: np.ndarray, target_dark: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    margin = np.where(target_dark, 0.5 - values, values - 0.5)
    output: dict[str, Any] = {
        "margin_mean": float(margin.mean()),
        "margin_p10": float(np.quantile(margin, 0.10)),
        "margin_min": float(margin.min()),
        "ambiguous_ratio_abs_margin_lt_005": float((np.abs(margin) < 0.05).mean()),
    }
    for name in ("finder", "timing", "alignment", "format", "fixed_dark", "functional_all", "data"):
        mask = masks[name]
        if mask.any():
            output[f"{name}_margin_mean"] = float(margin[mask].mean())
            output[f"{name}_margin_min"] = float(margin[mask].min())
    return output


def _error_map(path: Path, predicted: np.ndarray, target: np.ndarray, masks: dict[str, np.ndarray], scale: int = 18) -> None:
    errors = np.not_equal(predicted, target)
    functional = masks["functional_all"]
    canvas = np.full((QR_CORE_MODULES, QR_CORE_MODULES, 3), 242, dtype=np.uint8)
    canvas[functional & ~errors] = (185, 215, 235)
    canvas[~functional & errors] = (235, 155, 35)
    canvas[functional & errors] = (190, 40, 40)
    image = Image.fromarray(canvas, mode="RGB").resize(
        (QR_CORE_MODULES * scale, QR_CORE_MODULES * scale), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(image)
    for i in range(QR_CORE_MODULES + 1):
        v = i * scale
        draw.line((0, v, image.width, v), fill=(130, 130, 130), width=1)
        draw.line((v, 0, v, image.height), fill=(130, 130, 130), width=1)
    _save_png(path, image)


# ---------------------------------------------------------------------------
# Scanner-native transforms and decoder diagnostics
# ---------------------------------------------------------------------------

def _threshold_variants(image: Image.Image) -> dict[str, Image.Image]:
    import cv2

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        5,
    )
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, blur_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {
        "otsu": Image.fromarray(otsu, mode="L").convert("RGB"),
        "adaptive": Image.fromarray(adaptive, mode="L").convert("RGB"),
        "blur-otsu": Image.fromarray(blur_otsu, mode="L").convert("RGB"),
    }


def _opencv_stage(image: Image.Image, expected_payload: str) -> dict[str, Any]:
    import cv2

    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    detected, points = detector.detect(bgr)
    value, decode_points, straight = detector.detectAndDecode(bgr)
    pts = points.tolist() if points is not None else None
    decode_pts = decode_points.tolist() if decode_points is not None else None
    return {
        "detected": bool(detected),
        "detect_points": pts,
        "decoded": bool(value),
        "exact_payload": bool(value == expected_payload),
        "decode_points": decode_pts,
        "straight_qr_available": bool(straight is not None and getattr(straight, "size", 0) > 0),
        "decoded_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest() if value else None,
    }


def _build_direct_decoders() -> dict[str, Any]:
    from .validation import OpenCVDecoder, PyzbarDecoder, WeChatQRCodeDecoder, ZXingCPPDecoder

    output: dict[str, Any] = {}
    for decoder_cls in (OpenCVDecoder, PyzbarDecoder, ZXingCPPDecoder, WeChatQRCodeDecoder):
        name = getattr(decoder_cls, "name", decoder_cls.__name__)
        try:
            output[name] = decoder_cls()
        except Exception as exc:
            output[name] = {"init_error": {"type": type(exc).__name__, "message": str(exc)[:500]}}
    return output


def _direct_decoders(
    image: Image.Image, expected_payload: str, decoders: dict[str, Any]
) -> dict[str, Any]:
    from .validation import decode_safely

    output: dict[str, Any] = {}
    for name, decoder in decoders.items():
        if isinstance(decoder, dict) and decoder.get("init_error"):
            output[name] = {
                "decoded": False,
                "exact_payload": False,
                "decoded_sha256": None,
                "error": decoder["init_error"],
            }
            continue
        decoded, error = decode_safely(decoder, image)
        output[name] = {
            "decoded": bool(decoded),
            "exact_payload": decoded == expected_payload,
            "decoded_sha256": hashlib.sha256(decoded.encode("utf-8")).hexdigest() if decoded else None,
            "error": error,
        }
    return output


def _qrverify_one_shot(decoder: Any, image: Image.Image, expected_payload: str) -> dict[str, Any]:
    attempts = decoder.decode_presets(image)
    rows = []
    for item in attempts:
        decoded = str(item.get("text") or "")
        rows.append(
            {
                "preset": str(item.get("preset") or ""),
                "decoded": bool(decoded),
                "exact_payload": decoded == expected_payload,
                "decoded_sha256": hashlib.sha256(decoded.encode("utf-8")).hexdigest() if decoded else None,
                "decoder_error": str(item.get("error"))[:500] if item.get("error") else None,
                "latency_ms": float(item.get("latency_ms") or 0.0),
            }
        )
    original = next((row for row in rows if row["preset"] == "original"), None)
    exact = [row["preset"] for row in rows if row["exact_payload"]]
    decoded = [row["preset"] for row in rows if row["decoded"]]
    return {
        "engine": getattr(decoder, "engine_version", "unknown"),
        "diagnostic_one_shot": True,
        "preset_count": len(rows),
        "exact_preset_count": len(exact),
        "decoded_preset_count": len(decoded),
        "original_exact": bool(original and original["exact_payload"]),
        "original_decoded": bool(original and original["decoded"]),
        "exact_presets": exact,
        "decoded_presets": decoded,
        "attempts": rows,
    }


# ---------------------------------------------------------------------------
# Phase 1: GPU VAE re-decode only
# ---------------------------------------------------------------------------

def run_decode_phase(*, e041_results_dir: Path, output_dir: Path, source_commit: str) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    from .e035_loss_fidelity import _decode_latent_tensor, _load_pipeline, _offload_diffusion_modules

    if not torch.cuda.is_available():
        raise RuntimeError("E042 decode phase requires CUDA for a short VAE-only re-decode")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source_commit must be a lowercase 40-character Git SHA")

    verdict = _load_e041_verdict(e041_results_dir)
    phase_rows = _load_e041_phase_a_rows(e041_results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "e041-verdict.json", verdict)
    if not (output_dir / "plan.json").is_file():
        _atomic_json(output_dir / "plan.json", _plan_document(source_commit=source_commit, e041_results_dir=e041_results_dir))

    decode_root = output_dir / "decode"
    complete_path = decode_root / "complete.json"
    if complete_path.is_file():
        document = json.loads(complete_path.read_text(encoding="utf-8"))
        if int(document.get("state_count", 0)) == len(SELECTED_STATES):
            return document

    pending: list[SelectedState] = []
    bindings: list[dict[str, Any]] = []
    for state in SELECTED_STATES:
        current_path, latent_path = _state_paths(e041_results_dir, state)
        if not current_path.is_file() or not latent_path.is_file():
            raise FileNotFoundError(f"E041 state missing: {state.state_id}: {current_path} / {latent_path}")
        state_root = decode_root / "states" / state.state_id
        raw_path = state_root / "raw-vae.png"
        if not raw_path.is_file():
            pending.append(state)
        bindings.append(
            {
                **asdict(state),
                "recipe": state.recipe,
                "variant": state.variant,
                "e041_current_image": str(current_path),
                "e041_latent": str(latent_path),
                "e041_phase_a_row": phase_rows.get(state.variant),
            }
        )

    backend = pipeline = None
    original_vae_dtype = None
    if pending:
        backend, pipeline = _load_pipeline()
        original_vae_dtype = next(pipeline.vae.parameters()).dtype
        checkpointing_was_enabled = bool(getattr(pipeline.vae, "is_gradient_checkpointing", False))
        enable_checkpointing = getattr(pipeline.vae, "enable_gradient_checkpointing", None)
        disable_checkpointing = getattr(pipeline.vae, "disable_gradient_checkpointing", None)
        try:
            with _offload_diffusion_modules(pipeline) as offloaded:
                try:
                    if not checkpointing_was_enabled and callable(enable_checkpointing):
                        enable_checkpointing()
                    pipeline.vae.requires_grad_(False).eval().to(dtype=torch.float32)
                    _atomic_json(
                        decode_root / "runtime.json",
                        {
                            "torch_version": torch.__version__,
                            "cuda_version": torch.version.cuda,
                            "device_name": torch.cuda.get_device_name(0),
                            "offloaded_modules": list(offloaded),
                            "vae_original_dtype": str(original_vae_dtype),
                            "vae_effective_dtype": str(next(pipeline.vae.parameters()).dtype),
                            "vae_scaling_factor": float(pipeline.vae.config.scaling_factor),
                            "source_commit": source_commit,
                        },
                    )
                    for state in pending:
                        current_path, latent_path = _state_paths(e041_results_dir, state)
                        state_root = decode_root / "states" / state.state_id
                        state_root.mkdir(parents=True, exist_ok=True)
                        latent = load_file(str(latent_path), device="cpu")["latent"].to(
                            device="cuda", dtype=torch.float32
                        )
                        with torch.no_grad():
                            decoded = _decode_latent_tensor(pipeline, latent).float()
                        raw = pipeline.image_processor.postprocess(
                            decoded.detach(), output_type="pil", do_denormalize=[True]
                        )[0].convert("RGB")
                        if raw.size != (QR_CANVAS_PX, QR_CANVAS_PX):
                            raise RuntimeError(f"{state.state_id}: VAE decode is {raw.size}, expected 736x736")
                        _save_png(state_root / "raw-vae.png", raw)
                        del decoded, latent
                        gc.collect()
                        torch.cuda.empty_cache()
                finally:
                    if original_vae_dtype is not None:
                        pipeline.vae.to(dtype=original_vae_dtype)
                    if not checkpointing_was_enabled and callable(disable_checkpointing):
                        disable_checkpointing()
                    gc.collect()
                    torch.cuda.empty_cache()
        finally:
            if pipeline is not None and original_vae_dtype is not None:
                try:
                    pipeline.vae.to(dtype=original_vae_dtype)
                except Exception:
                    pass
            if backend is not None:
                backend._pipeline = None
            del pipeline, backend
            gc.collect()
            torch.cuda.empty_cache()

    state_documents: list[dict[str, Any]] = []
    for binding in bindings:
        state = next(item for item in SELECTED_STATES if item.state_id == binding["state_id"])
        current_path, latent_path = _state_paths(e041_results_dir, state)
        state_root = decode_root / "states" / state.state_id
        raw = Image.open(state_root / "raw-vae.png").convert("RGB")
        current = Image.open(current_path).convert("RGB")
        shutil.copy2(current_path, state_root / "current-scan-ready.png")
        adaptive_color = _exact_adaptive_quiet_color(raw)
        exact_adaptive = _restore_exact_quiet_zone(raw, adaptive_color)
        exact_white = _restore_exact_quiet_zone(raw, (255, 255, 255))
        _save_png(state_root / "exact-qz-adaptive.png", exact_adaptive)
        _save_png(state_root / "exact-qz-white.png", exact_white)
        overwrite = _quiet_zone_overwrite_metrics(raw, current)
        document = {
            **binding,
            "raw_image_sha256": _image_sha256(raw),
            "current_image_sha256": _image_sha256(current),
            "latent_file_sha256": _sha256_file(latent_path),
            "exact_adaptive_quiet_color": list(adaptive_color),
            "quiet_zone_overwrite": overwrite,
            "files": {
                "raw-vae": str(state_root / "raw-vae.png"),
                "current-scan-ready": str(state_root / "current-scan-ready.png"),
                "exact-qz-adaptive": str(state_root / "exact-qz-adaptive.png"),
                "exact-qz-white": str(state_root / "exact-qz-white.png"),
            },
        }
        _atomic_json(state_root / "state.json", document)
        state_documents.append(document)

    complete = {
        "experiment": EXPERIMENT,
        "phase": "decode",
        "state_count": len(state_documents),
        "selected_states": [item["state_id"] for item in state_documents],
        "reused_e041_phase_a": True,
        "stage1_recomputed": False,
        "stage2_recomputed": False,
        "srmpgd_recomputed": False,
        "vae_redecode_only": True,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(decode_root / "selected-states.json", state_documents)
    _atomic_json(complete_path, complete)
    return complete


# ---------------------------------------------------------------------------
# Phase 2: CPU scanner diagnostics
# ---------------------------------------------------------------------------

def _flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            flat[key] = value
    return flat


def _state_diagnostics(
    *,
    state_doc: dict[str, Any],
    blueprint: Any,
    target: np.ndarray,
    masks: dict[str, np.ndarray],
    work_root: Path,
    qrverify_decoder: Any,
    direct_decoders: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state_id = str(state_doc["state_id"])
    source_root = Path(state_doc["files"]["raw-vae"]).parent
    state_root = work_root / "states" / state_id
    state_root.mkdir(parents=True, exist_ok=True)

    variants: dict[str, Image.Image] = {
        "current-scan-ready": Image.open(source_root / "current-scan-ready.png").convert("RGB"),
        "raw-vae": Image.open(source_root / "raw-vae.png").convert("RGB"),
        "exact-qz-adaptive": Image.open(source_root / "exact-qz-adaptive.png").convert("RGB"),
        "exact-qz-white": Image.open(source_root / "exact-qz-white.png").convert("RGB"),
    }
    thresholded = _threshold_variants(variants["exact-qz-adaptive"])
    variants.update(thresholded)

    means, centers, stds = _sample_modules(variants["raw-vae"], center_fraction=0.40)
    mean_best_t, mean_best_errors = _best_threshold(means, target)
    center_best_t, center_best_errors = _best_threshold(centers, target)
    mean_pred_050 = means < 0.5
    center_pred_050 = centers < 0.5
    mean_pred_best = means < mean_best_t
    center_pred_best = centers < center_best_t

    variants["grid-mean-050"] = _render_matrix(mean_pred_050)
    variants["grid-center-050"] = _render_matrix(center_pred_050)
    variants["grid-mean-best"] = _render_matrix(mean_pred_best)
    variants["grid-center-best"] = _render_matrix(center_pred_best)

    for name, image in variants.items():
        _save_png(state_root / f"{name}.png", image)

    _error_map(state_root / "module-error-map-mean050.png", mean_pred_050, target, masks)
    _error_map(state_root / "module-error-map-mean-best.png", mean_pred_best, target, masks)

    structure = {
        "mean_threshold_050": _region_error_metrics(mean_pred_050, target, masks),
        "center_threshold_050": _region_error_metrics(center_pred_050, target, masks),
        "mean_best": {
            "threshold": mean_best_t,
            "target_assisted": True,
            **_region_error_metrics(mean_pred_best, target, masks),
        },
        "center_best": {
            "threshold": center_best_t,
            "target_assisted": True,
            **_region_error_metrics(center_pred_best, target, masks),
        },
        "mean_margin": _margin_metrics(means, target, masks),
        "center_margin": _margin_metrics(centers, target, masks),
        "intra_module_std_mean": float(stds.mean()),
        "intra_module_std_p95": float(np.quantile(stds, 0.95)),
        "mean_best_error_count": mean_best_errors,
        "center_best_error_count": center_best_errors,
    }

    decoder_rows: list[dict[str, Any]] = []
    variant_results: dict[str, Any] = {}
    for name, image in variants.items():
        opencv = _opencv_stage(image, PAYLOAD)
        direct = _direct_decoders(image, PAYLOAD, direct_decoders)
        qrverify = None
        if name in DIAGNOSTIC_QRVERIFY_VARIANTS:
            qrverify = _qrverify_one_shot(qrverify_decoder, image, PAYLOAD)
        result = {
            "opencv_stage": opencv,
            "direct_decoders": direct,
            "qr_verify": qrverify,
        }
        variant_results[name] = result
        decoder_rows.append(
            {
                "state_id": state_id,
                "gamma": state_doc["gamma"],
                "iteration": state_doc["iteration"],
                "variant": name,
                "opencv_detected": opencv["detected"],
                "opencv_exact": opencv["exact_payload"],
                "zbar_exact": bool((direct.get("zbar") or {}).get("exact_payload")),
                "zxingcpp_exact": bool((direct.get("zxingcpp") or {}).get("exact_payload")),
                "wechat_exact": bool((direct.get("wechat_qrcode") or {}).get("exact_payload")),
                "qr_verify_one_shot_exact_presets": (
                    int(qrverify["exact_preset_count"]) if qrverify else None
                ),
                "qr_verify_one_shot_original_exact": (
                    bool(qrverify["original_exact"]) if qrverify else None
                ),
            }
        )

    document = {
        "state": state_doc,
        "structure": structure,
        "variants": variant_results,
    }
    _atomic_json(state_root / "diagnostic.json", document)
    return document, decoder_rows


def _select_conservative_rescues(state_docs: list[dict[str, Any]], limit: int = 3) -> list[tuple[str, str, int]]:
    candidates: list[tuple[int, str, str]] = []
    for document in state_docs:
        state_id = str(document["state"]["state_id"])
        for variant, result in document["variants"].items():
            qv = result.get("qr_verify")
            if not qv:
                continue
            exact = int(qv.get("exact_preset_count", 0))
            if exact > 0:
                candidates.append((exact, state_id, variant))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(state_id, variant, exact) for exact, state_id, variant in candidates[:limit]]


def _diagnostic_conclusion(
    state_docs: list[dict[str, Any]], decoder_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    def qv(state_doc: dict[str, Any], variant: str) -> int:
        item = state_doc["variants"].get(variant) or {}
        evidence = item.get("qr_verify") or {}
        return int(evidence.get("exact_preset_count", 0))

    qz_rescues = []
    bin_rescues = []
    grid_rescues = []
    for document in state_docs:
        state_id = str(document["state"]["state_id"])
        current = qv(document, "current-scan-ready")
        exact_best = max(qv(document, "exact-qz-adaptive"), qv(document, "exact-qz-white"))
        threshold_best = max(qv(document, "otsu"), qv(document, "adaptive"))
        grid_best = max(
            qv(document, "grid-mean-050"),
            qv(document, "grid-mean-best"),
            qv(document, "grid-center-best"),
        )
        if exact_best > current:
            qz_rescues.append({"state_id": state_id, "current": current, "exact_qz": exact_best})
        if threshold_best > exact_best:
            bin_rescues.append({"state_id": state_id, "exact_qz": exact_best, "thresholded": threshold_best})
        if grid_best > max(current, exact_best, threshold_best):
            grid_rescues.append({"state_id": state_id, "previous": max(current, exact_best, threshold_best), "grid": grid_best})

    overwrite_states = [
        str(document["state"]["state_id"])
        for document in state_docs
        if bool((document["state"].get("quiet_zone_overwrite") or {}).get("legacy_quiet_zone_overwrites_exact_core"))
    ]
    opencv_current_detected = sum(
        1
        for row in decoder_rows
        if row["variant"] == "current-scan-ready" and bool(row["opencv_detected"])
    )
    opencv_current_exact = sum(
        1
        for row in decoder_rows
        if row["variant"] == "current-scan-ready" and bool(row["opencv_exact"])
    )
    min_mean_best = min(
        int(document["structure"]["mean_best_error_count"]) for document in state_docs
    )
    min_center_best = min(
        int(document["structure"]["center_best_error_count"]) for document in state_docs
    )
    min_format_errors = min(
        int(document["structure"]["mean_best"]["format_error_count"]) for document in state_docs
    )
    min_data_errors = min(
        int(document["structure"]["mean_best"]["data_error_count"]) for document in state_docs
    )

    if qz_rescues:
        primary = "QUIET_ZONE_GEOMETRY_OR_CORE_EDGE_OVERWRITE"
    elif bin_rescues:
        primary = "BINARIZATION_AND_LOCAL_TEXTURE"
    elif grid_rescues:
        primary = "GRID_DETECTION_OR_INTRA_MODULE_TEXTURE"
    elif min_format_errors > 0:
        primary = "FORMAT_INFORMATION_BIT_ERRORS_PLUS_RESIDUAL_MODULE_ERRORS"
    else:
        primary = "RESIDUAL_MODULE_BITS_OR_ECC_NOT_CAPTURED_BY_CURRENT_LOSS"

    components: list[str] = []
    if overwrite_states:
        components.append("exact_diffqrcoder_quiet_zone_geometry")
    if bin_rescues:
        components.extend(("soft_threshold_ensemble", "intra_module_variance_penalty"))
    if grid_rescues:
        components.extend(("whole_cell_margin", "intra_module_variance_penalty", "grid_consistency"))
    if min_format_errors > 0:
        components.append("format_information_weighted_margin")
    if min_data_errors > 0:
        components.append("data_module_margin_with_ecc_awareness")
    components.extend(("finder_timing_geometry_diagnostics", "real_decoder_validation"))
    components = list(dict.fromkeys(components))

    return {
        "primary_blocker": primary,
        "quiet_zone_rescues": qz_rescues,
        "binarization_rescues": bin_rescues,
        "grid_reconstruction_rescues": grid_rescues,
        "legacy_quiet_zone_core_overwrite_states": overwrite_states,
        "opencv_current_detected_count": opencv_current_detected,
        "opencv_current_exact_count": opencv_current_exact,
        "selected_state_count": len(state_docs),
        "minimum_target_assisted_mean_module_errors": min_mean_best,
        "minimum_target_assisted_center_module_errors": min_center_best,
        "minimum_target_assisted_format_errors": min_format_errors,
        "minimum_target_assisted_data_errors": min_data_errors,
        "recommended_e043_loss_components": components,
        "interpretation_guard": (
            "E042 diagnostic transforms are not production candidates. Target-assisted threshold "
            "sweeps and canonical grid reconstructions only localize failure modes."
        ),
    }


def _contact_sheet(path: Path, items: list[tuple[str, Image.Image, str]], columns: int = 4) -> None:
    tile_w, tile_h = 340, 390
    rows = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (title, image, subtitle) in enumerate(items):
        row, col = divmod(index, columns)
        x0, y0 = col * tile_w, row * tile_h
        preview = image.convert("RGB").copy()
        preview.thumbnail((320, 300), Image.Resampling.LANCZOS)
        x = x0 + (tile_w - preview.width) // 2
        y = y0 + 60 + (300 - preview.height) // 2
        sheet.paste(preview, (x, y))
        draw.text((x0 + 10, y0 + 10), title, fill=(0, 0, 0))
        draw.text((x0 + 10, y0 + 30), subtitle[:52], fill=(70, 70, 70))
    _save_png(path, sheet)


def run_diagnose_phase(*, e041_results_dir: Path, output_dir: Path, source_commit: str) -> dict[str, Any]:
    from .qr import generate_diffqrcoder_qr
    from .validation import (
        ConservativeQRVerifyScorer,
        QRVerifyDecoder,
        canonical_conservative_qr_verify_evidence,
    )

    verdict_e041 = _load_e041_verdict(e041_results_dir)
    decode_complete = output_dir / "decode/complete.json"
    state_list_path = output_dir / "decode/selected-states.json"
    if not decode_complete.is_file() or not state_list_path.is_file():
        raise FileNotFoundError("E042 decode phase is incomplete; run --phase decode first")
    if (output_dir / "verdict.json").is_file():
        return json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
    if (output_dir / "diagnose").exists():
        raise RuntimeError("E042 diagnose directory exists without verdict; inspect before rerunning")

    blueprint = generate_diffqrcoder_qr(
        PAYLOAD,
        ERROR_CORRECTION,
        version=QR_VERSION,
        mask_pattern=QR_MASK_PATTERN,
        module_size=QR_MODULE_SIZE,
    )
    border = int(blueprint.border)
    target = blueprint.matrix[border:-border, border:-border] if border else blueprint.matrix
    target = np.asarray(target, dtype=bool)
    if target.shape != (QR_CORE_MODULES, QR_CORE_MODULES):
        raise RuntimeError(f"E042 expected QR core 29x29, got {target.shape}")
    masks = _region_masks(blueprint)

    tmp_root = Path(tempfile.mkdtemp(prefix="e042-diagnose-", dir=str(output_dir)))
    work_root = tmp_root / "diagnose"
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        exact_reference = _exact_reference_image(blueprint)
        _save_png(work_root / "exact-reference-736.png", exact_reference)

        qrverify_decoder = QRVerifyDecoder()
        direct_decoders = _build_direct_decoders()
        cache_dir = output_dir / "qr-verify-cache"
        scorer = ConservativeQRVerifyScorer(
            decoder=qrverify_decoder,
            repetitions=3,
            cache_dir=cache_dir,
        )
        try:
            preflight = scorer.score(exact_reference, PAYLOAD)
            preflight_evidence = canonical_conservative_qr_verify_evidence(preflight)
            _atomic_json(work_root / "qr-verify-preflight.json", preflight_evidence)
            if not preflight.direct_exact_all_repetitions or preflight.conservative_exact_presets != preflight.preset_count:
                raise RuntimeError(
                    "E042 QR-Verify preflight failed on exact binary reference: "
                    f"{preflight.conservative_exact_presets}/{preflight.preset_count}"
                )

            state_list = json.loads(state_list_path.read_text(encoding="utf-8"))
            state_docs: list[dict[str, Any]] = []
            decoder_rows: list[dict[str, Any]] = []
            for state_doc in state_list:
                document, rows = _state_diagnostics(
                    state_doc=state_doc,
                    blueprint=blueprint,
                    target=target,
                    masks=masks,
                    work_root=work_root,
                    qrverify_decoder=qrverify_decoder,
                    direct_decoders=direct_decoders,
                )
                state_docs.append(document)
                decoder_rows.extend(rows)

            _atomic_json(work_root / "state-diagnostics.json", state_docs)
            _atomic_json(work_root / "decoder-stage-matrix.json", decoder_rows)
            _write_csv(work_root / "decoder-stage-matrix.csv", [_flatten_for_csv(row) for row in decoder_rows])

            rescue_candidates = _select_conservative_rescues(state_docs, limit=3)
            rescue_evidence: list[dict[str, Any]] = []
            for state_id, variant, one_shot_exact in rescue_candidates:
                image_path = work_root / "states" / state_id / f"{variant}.png"
                score = scorer.score(Image.open(image_path).convert("RGB"), PAYLOAD)
                rescue_evidence.append(
                    {
                        "state_id": state_id,
                        "variant": variant,
                        "diagnostic_one_shot_exact_presets": one_shot_exact,
                        "conservative": canonical_conservative_qr_verify_evidence(score),
                    }
                )
            _atomic_json(work_root / "conservative-rescue-checks.json", rescue_evidence)

            conclusion = _diagnostic_conclusion(state_docs, decoder_rows)
            conclusion["conservative_rescue_checks"] = [
                {
                    "state_id": row["state_id"],
                    "variant": row["variant"],
                    "conservative_exact_presets": int(row["conservative"]["conservative_exact_presets"]),
                    "preset_count": int(row["conservative"]["preset_count"]),
                }
                for row in rescue_evidence
            ]
            _atomic_json(work_root / "diagnostic-conclusion.json", conclusion)

            representative = min(
                state_docs,
                key=lambda document: (
                    int(document["structure"]["mean_best_error_count"]),
                    abs(float(document["state"]["gamma"]) - 500.0),
                    int(document["state"]["iteration"]),
                ),
            )
            rep_id = str(representative["state"]["state_id"])
            rep_root = work_root / "states" / rep_id
            pipeline = work_root / "pipeline"
            pipeline.mkdir(parents=True, exist_ok=True)
            shutil.copy2(work_root / "exact-reference-736.png", pipeline / "01-exact-reference.png")
            sequence = [
                ("02-current-scan-ready.png", "current-scan-ready"),
                ("03-raw-vae.png", "raw-vae"),
                ("04-exact-qz.png", "exact-qz-adaptive"),
                ("05-otsu.png", "otsu"),
                ("06-adaptive.png", "adaptive"),
                ("07-grid-mean-050.png", "grid-mean-050"),
                ("08-grid-mean-best.png", "grid-mean-best"),
                ("09-grid-center-best.png", "grid-center-best"),
            ]
            for filename, variant in sequence:
                shutil.copy2(rep_root / f"{variant}.png", pipeline / filename)
            shutil.copy2(rep_root / "module-error-map-mean050.png", pipeline / "10-module-error-map-mean050.png")
            shutil.copy2(rep_root / "module-error-map-mean-best.png", pipeline / "11-module-error-map-mean-best.png")

            items: list[tuple[str, Image.Image, str]] = [
                ("Exact reference", exact_reference, "QR-Verify preflight 37/37"),
            ]
            rep_diag = representative["variants"]
            for title, variant in (
                ("Current", "current-scan-ready"),
                ("Raw VAE", "raw-vae"),
                ("Exact QZ", "exact-qz-adaptive"),
                ("Otsu", "otsu"),
                ("Adaptive", "adaptive"),
                ("Grid mean .50", "grid-mean-050"),
                ("Grid mean best", "grid-mean-best"),
                ("Grid center best", "grid-center-best"),
            ):
                qv = (rep_diag.get(variant) or {}).get("qr_verify") or {}
                items.append(
                    (
                        title,
                        Image.open(rep_root / f"{variant}.png").convert("RGB"),
                        f"QRV={qv.get('exact_preset_count', 'n/a')}/37",
                    )
                )
            _contact_sheet(pipeline / "decoder-localization-contact-sheet.png", items, columns=3)

            verdict = {
                "experiment": EXPERIMENT,
                "source_commit": source_commit,
                "e041_experiment": verdict_e041["experiment"],
                "e041_phase_a_checkpoint_count": verdict_e041["phase_a_checkpoint_count"],
                "selected_state_count": len(state_docs),
                "representative_state": rep_id,
                "qr_verify_preflight_exact_presets": int(preflight.conservative_exact_presets),
                "qr_verify_preflight_preset_count": int(preflight.preset_count),
                "primary_blocker": conclusion["primary_blocker"],
                "quiet_zone_rescue_count": len(conclusion["quiet_zone_rescues"]),
                "binarization_rescue_count": len(conclusion["binarization_rescues"]),
                "grid_reconstruction_rescue_count": len(conclusion["grid_reconstruction_rescues"]),
                "legacy_quiet_zone_core_overwrite_state_count": len(
                    conclusion["legacy_quiet_zone_core_overwrite_states"]
                ),
                "minimum_target_assisted_mean_module_errors": conclusion[
                    "minimum_target_assisted_mean_module_errors"
                ],
                "minimum_target_assisted_format_errors": conclusion[
                    "minimum_target_assisted_format_errors"
                ],
                "minimum_target_assisted_data_errors": conclusion[
                    "minimum_target_assisted_data_errors"
                ],
                "recommended_e043_loss_components": conclusion["recommended_e043_loss_components"],
                "diagnostic_only": True,
                "production_ready": False,
                "generalization_authorized": False,
                "next_action": "DESIGN_E043_FROM_DECODER_LOCALIZATION_NOT_FROM_MER_ALONE",
            }
            _atomic_json(work_root / "verdict.json", verdict)
            _atomic_text(
                work_root / "report.md",
                "\n".join(
                    [
                        "# E042 — decoder failure localization",
                        "",
                        f"- E041 checkpoints reused: **{len(state_docs)}** selected states",
                        "- Stage 1 / Stage 2 / SR-MPGD recomputed: **no**",
                        "- selected latents VAE re-decoded: **yes, diagnostic only**",
                        f"- QR-Verify exact binary preflight: **{preflight.conservative_exact_presets}/{preflight.preset_count}**",
                        f"- primary blocker: **{conclusion['primary_blocker']}**",
                        f"- exact quiet-zone rescues: **{len(conclusion['quiet_zone_rescues'])}**",
                        f"- binarization rescues: **{len(conclusion['binarization_rescues'])}**",
                        f"- grid reconstruction rescues: **{len(conclusion['grid_reconstruction_rescues'])}**",
                        f"- minimum target-assisted mean module errors: **{conclusion['minimum_target_assisted_mean_module_errors']}**",
                        f"- minimum target-assisted format errors: **{conclusion['minimum_target_assisted_format_errors']}**",
                        "- production/generalization: **no**",
                        "",
                        "## E043 components suggested by evidence",
                        *[f"- {item}" for item in conclusion["recommended_e043_loss_components"]],
                        "",
                    ]
                ),
            )
        finally:
            scorer.close()

        final_diagnose = output_dir / "diagnose"
        os.replace(work_root, final_diagnose)
        shutil.rmtree(tmp_root, ignore_errors=True)
        shutil.copy2(final_diagnose / "verdict.json", output_dir / "verdict.json")
        shutil.copy2(final_diagnose / "report.md", output_dir / "report.md")
        _atomic_json(output_dir / "e042-artifact-manifest.json", _manifest(output_dir))
        return json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
    except Exception:
        # Keep tmp_root for forensic inspection. The final diagnose/ directory is
        # never exposed unless the entire CPU phase completed successfully.
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("decode", "diagnose"), required=True)
    parser.add_argument("--e041-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.phase == "decode":
        result = run_decode_phase(
            e041_results_dir=args.e041_results_dir,
            output_dir=args.output_dir,
            source_commit=args.source_commit,
        )
    else:
        result = run_diagnose_phase(
            e041_results_dir=args.e041_results_dir,
            output_dir=args.output_dir,
            source_commit=args.source_commit,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
