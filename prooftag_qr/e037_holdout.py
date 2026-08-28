"""E037 — prospective 10-case mini-holdout for the frozen E036 global trust policy.

E037 does not tune the trust region. It freezes the E036 research winner exactly:

* official DiffQRCoder SRL at revision e24ea73ee2e13c7e6e87cb422e8b11784e70ae00;
* gamma = 1000 for every raw proposal;
* latent RMS radius = 0.050;
* LPIPS budget = 0.050;
* core MAE budget = 0.050;
* four recorded updates.

Ten prompts and seeds are preregistered in this module before the GPU run. Each case
regenerates a fresh DiffQRCoder Stage 1 + public SRPG Stage 2 parent, then applies only
the frozen E036 global trust policy. The runner writes side-by-side images, QR-Verify
evidence, aggregate metrics, a verdict, an integrity manifest and an archive.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import shutil
import tarfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .e035_loss_fidelity import (
    _atomic_json,
    _atomic_text,
    _image_sha256,
    _offload_diffusion_modules,
    _score_qr_verify,
)
from .e035_parent_artifact import LoadedParentArtifact, tensor_sha256
from .e035_parent_capture import (
    CaptureConfig,
    DEFAULT_NEGATIVE_PROMPT,
    UPSTREAM_REVISION,
    _settings_document,
)
from .e036_trust_region import (
    BRANCH_GLOBAL,
    DEFAULT_POLICIES,
    E036Config,
    _run_branch,
)

EXPERIMENT = "e037-prospective-global-trust-mini-holdout-v1"
PAYLOAD = "https://ptag.io/t/e037"
ERROR_CORRECTION = "M"
QR_VERSION = 3
QR_MASK_PATTERN = 4
QR_MODULE_SIZE = 20
QR_PADDING_PX = 78
CASE_COUNT = 10


@dataclass(frozen=True, slots=True)
class HoldoutCase:
    case_id: str
    seed: int
    prompt: str


# Frozen before execution. Do not edit after seeing E037 results.
HOLDOUT_CASES: tuple[HoldoutCase, ...] = (
    HoldoutCase(
        "courtyard",
        61001,
        "a Mediterranean courtyard with blue ceramic tiles, lemon trees and a stone fountain, warm editorial photograph",
    ),
    HoldoutCase(
        "station",
        61002,
        "a grand retro-futurist railway station with travelers, clocks, glass arches and glowing kiosks, cinematic photograph",
    ),
    HoldoutCase(
        "wine_cellar",
        61003,
        "an old French wine cellar with oak barrels, limestone walls and warm hanging lamps, atmospheric editorial photograph",
    ),
    HoldoutCase(
        "alpine_cabin",
        61004,
        "a cozy alpine cabin surrounded by fresh snow and pine trees at blue hour, realistic architectural photograph",
    ),
    HoldoutCase(
        "ramen_shop",
        61005,
        "a narrow Japanese ramen shop at night with wooden counters, steam and warm lantern light, documentary photograph",
    ),
    HoldoutCase(
        "botanical_library",
        61006,
        "a quiet botanical library filled with books, climbing plants and tall arched windows, natural light editorial photograph",
    ),
    HoldoutCase(
        "lighthouse",
        61007,
        "a white lighthouse on a rugged Atlantic cliff above rough sea and sea grass, dramatic landscape photograph",
    ),
    HoldoutCase(
        "workshop",
        61008,
        "a precise industrial workshop with metal tools, workbenches and soft skylight, clean documentary photograph",
    ),
    HoldoutCase(
        "paris_cafe",
        61009,
        "an elegant Paris cafe interior with marble tables, bentwood chairs and morning window light, editorial photograph",
    ),
    HoldoutCase(
        "vineyard",
        61010,
        "rows of vineyard vines at golden hour beside a small stone winery, refined wine brand campaign photograph",
    ),
)

GLOBAL_POLICY = next(policy for policy in DEFAULT_POLICIES if policy.name == BRANCH_GLOBAL)
FROZEN_CONFIG = E036Config()


def _assert_frozen_protocol() -> None:
    assert len(HOLDOUT_CASES) == CASE_COUNT
    assert len({case.case_id for case in HOLDOUT_CASES}) == CASE_COUNT
    assert len({case.seed for case in HOLDOUT_CASES}) == CASE_COUNT
    assert FROZEN_CONFIG.gamma == 1000.0
    assert FROZEN_CONFIG.max_iterations == 4
    assert FROZEN_CONFIG.lpips_weight == 0.01
    assert GLOBAL_POLICY.latent_radius_rms == 0.050
    assert GLOBAL_POLICY.lpips_budget == 0.050
    assert GLOBAL_POLICY.core_mae_budget == 0.050
    assert GLOBAL_POLICY.outside_active_mae_budget is None
    assert UPSTREAM_REVISION == "e24ea73ee2e13c7e6e87cb422e8b11784e70ae00"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", optimize=False, compress_level=9)


def _case_comparison(path: Path, label: str, parent: Image.Image, final: Image.Image) -> None:
    tile_w, tile_h = 420, 460
    sheet = Image.new("RGB", (tile_w * 2, tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(((f"{label} parent", parent), (f"{label} E037", final))):
        preview = image.convert("RGB").copy()
        preview.thumbnail((400, 390), Image.Resampling.LANCZOS)
        x0 = index * tile_w
        x = x0 + (tile_w - preview.width) // 2
        y = 48 + (390 - preview.height) // 2
        sheet.paste(preview, (x, y))
        draw.text((x0 + 12, 14), title, fill=(0, 0, 0))
    _save_png(path, sheet)


def _global_contact_sheet(path: Path, rows: list[tuple[str, Image.Image, Image.Image, str]]) -> None:
    # Two cases per row: parent/final | parent/final. This keeps ten cases readable.
    tile_w, tile_h = 360, 390
    columns = 4
    sheet_rows = math.ceil(len(rows) / 2)
    sheet = Image.new("RGB", (columns * tile_w, sheet_rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for case_index, (case_id, parent, final, subtitle) in enumerate(rows):
        row = case_index // 2
        case_slot = case_index % 2
        for pair_index, (kind, image) in enumerate((("Parent", parent), ("E037", final))):
            col = case_slot * 2 + pair_index
            x0, y0 = col * tile_w, row * tile_h
            preview = image.convert("RGB").copy()
            preview.thumbnail((340, 310), Image.Resampling.LANCZOS)
            x = x0 + (tile_w - preview.width) // 2
            y = y0 + 62 + (310 - preview.height) // 2
            sheet.paste(preview, (x, y))
            draw.text((x0 + 10, y0 + 8), f"{case_id} — {kind}", fill=(0, 0, 0))
            draw.text((x0 + 10, y0 + 28), subtitle, fill=(60, 60, 60))
    _save_png(path, sheet)


def _build_manifest(root: Path) -> list[dict[str, Any]]:
    excluded = {"e037-artifact-manifest.json"}
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _write_deterministic_archive(root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as handle:
                for path in sorted(root.rglob("*")):
                    if not path.is_file():
                        continue
                    arcname = f"{root.name}/{path.relative_to(root).as_posix()}"
                    info = handle.gettarinfo(str(path), arcname=arcname)
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mtime = 0
                    with path.open("rb") as stream:
                        handle.addfile(info, stream)


def _extract_exact(evidence: dict[str, Any], key: str) -> int:
    return int((evidence.get(key) or {}).get("conservative_exact_presets", 0))


def _load_e036_precondition(e036_results_dir: Path) -> dict[str, Any]:
    verdict_path = e036_results_dir / "verdict.json"
    if not verdict_path.is_file():
        raise FileNotFoundError(f"E036 verdict missing: {verdict_path}")
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    if verdict.get("research_winner") != BRANCH_GLOBAL:
        raise RuntimeError(
            "E037 requires the frozen E036 global trust winner; "
            f"got {verdict.get('research_winner')!r}"
        )
    if verdict.get("gamma") != 1000.0 or verdict.get("gamma_preserved") is not True:
        raise RuntimeError("E036 prerequisite does not prove gamma=1000 preservation")
    if verdict.get("decision") != "PREPARE_MINI_HOLDOUT_WITH_WINNER":
        raise RuntimeError(f"E036 did not authorize a mini-holdout: {verdict.get('decision')!r}")
    return verdict


def run_e037(
    *,
    output_dir: Path,
    e036_results_dir: Path,
    source_commit: str,
    skip_qr_verify: bool = False,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    from .config import Settings
    from .diffqrcoder_backend import UpstreamDiffQRCoderBackend
    from .qr import generate_diffqrcoder_qr
    from .schemas import GenerationRequest

    _assert_frozen_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("E037 requires an available CUDA GPU")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source_commit must be a lowercase 40-character Git SHA")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"E037 output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    e036_verdict = _load_e036_precondition(e036_results_dir)
    _atomic_json(output_dir / "e036-prerequisite-verdict.json", e036_verdict)
    _atomic_json(
        output_dir / "plan.json",
        {
            "experiment": EXPERIMENT,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_commit": source_commit,
            "payload": PAYLOAD,
            "error_correction": ERROR_CORRECTION,
            "qr": {
                "version": QR_VERSION,
                "mask_pattern": QR_MASK_PATTERN,
                "module_size": QR_MODULE_SIZE,
                "padding_px": QR_PADDING_PX,
            },
            "frozen_e036_policy": asdict(GLOBAL_POLICY),
            "frozen_e036_config": asdict(FROZEN_CONFIG),
            "cases": [asdict(case) for case in HOLDOUT_CASES],
            "case_count": CASE_COUNT,
            "preregistered_before_results": True,
            "production_ready": False,
            "automatic_expansion_authorized": False,
        },
    )

    capture_template = CaptureConfig(payload=PAYLOAD)
    base = Settings()
    settings = Settings.model_validate({**base.model_dump(), **_settings_document(capture_template)})
    if str(settings.diffqrcoder_revision) != UPSTREAM_REVISION:
        raise RuntimeError("runtime DiffQRCoder revision differs from E037 preregistration")
    backend = UpstreamDiffQRCoderBackend(settings)
    pipeline = backend._load()
    blueprint = generate_diffqrcoder_qr(
        PAYLOAD,
        ERROR_CORRECTION,
        version=QR_VERSION,
        mask_pattern=QR_MASK_PATTERN,
        module_size=QR_MODULE_SIZE,
    )

    parents: dict[str, LoadedParentArtifact] = {}
    parent_images: dict[str, Image.Image] = {}
    stage1_images: dict[str, Image.Image] = {}
    case_roots: dict[str, Path] = {}

    generation_started = time.perf_counter()
    for case in HOLDOUT_CASES:
        case_root = output_dir / "cases" / case.case_id
        parent_root = case_root / "parent"
        parent_root.mkdir(parents=True, exist_ok=True)
        case_roots[case.case_id] = case_root

        capture = CaptureConfig(payload=PAYLOAD, prompt=case.prompt, seed=case.seed)
        request = GenerationRequest(
            payload=PAYLOAD,
            prompt=case.prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            backend="controlnet",
            error_correction=ERROR_CORRECTION,
            seed=case.seed,
            steps=capture.steps,
            guidance_scale=capture.guidance_scale,
            controlnet_scale=capture.controlnet_scale,
            strength=capture.strength,
            max_attempts=1,
        )

        stage1 = backend.generate(request, blueprint, case.seed)
        stage2 = backend._run_stage2(stage1, blueprint, request, case.seed)
        state = backend.export_stage2_state()
        if state is None:
            raise RuntimeError(f"{case.case_id}: Stage 2 produced no exportable latent")
        latent = state["latent"].detach().cpu().contiguous()
        latent_hash = tensor_sha256(latent)
        if latent_hash != str(state.get("latent_sha256") or latent_hash):
            raise RuntimeError(f"{case.case_id}: Stage 2 latent hash mismatch")
        if stage1.size != (736, 736) or stage2.size != (736, 736):
            raise RuntimeError(f"{case.case_id}: expected 736x736 Stage 1 and Stage 2")

        stage1_path = parent_root / "stage1.png"
        stage2_path = parent_root / "stage2.png"
        latent_path = parent_root / "stage2-latent.safetensors"
        _save_png(stage1_path, stage1)
        _save_png(stage2_path, stage2)
        save_file({"latent": latent}, str(latent_path))

        metadata = {
            "case": asdict(case),
            "payload": PAYLOAD,
            "source_commit": source_commit,
            "diffqrcoder_revision": UPSTREAM_REVISION,
            "stage1_image_sha256": _image_sha256(stage1),
            "stage2_image_sha256": _image_sha256(stage2),
            "stage2_latent_tensor_sha256": latent_hash,
            "stage2_latent_file_sha256": _sha256_file(latent_path),
            "stage2_source_run_id": state.get("source_run_id"),
            "stage2_source_method_id": state.get("source_method_id"),
            "generation": {
                "stage1_steps": capture.steps,
                "stage1_guidance_scale": capture.guidance_scale,
                "stage1_controlnet_scale": capture.controlnet_scale,
                "srpg_steps": capture.srpg_steps,
                "srpg_controlnet_scale": capture.srpg_controlnet_scale,
                "srpg_qr_weight": capture.srpg_qr_weight,
                "srpg_perceptual_weight": capture.srpg_perceptual_weight,
                "srpg_eta": capture.srpg_eta,
                "srpg_seed_offset": capture.srpg_seed_offset,
                "stage2_initialization": capture.stage2_initialization,
                "stage2_target_mode": capture.stage2_target_mode,
            },
        }
        _atomic_json(parent_root / "metadata.json", metadata)
        parents[case.case_id] = LoadedParentArtifact(
            root=parent_root,
            image=stage2.copy(),
            latent=latent,
            metadata={"files": {"latent": {"tensor_sha256": latent_hash}}, "source": metadata},
        )
        parent_images[case.case_id] = stage2.copy()
        stage1_images[case.case_id] = stage1.copy()
        gc.collect()
        torch.cuda.empty_cache()

    original_vae_dtype = next(pipeline.vae.parameters()).dtype
    checkpointing_was_enabled = bool(getattr(pipeline.vae, "is_gradient_checkpointing", False))
    enable_checkpointing = getattr(pipeline.vae, "enable_gradient_checkpointing", None)
    disable_checkpointing = getattr(pipeline.vae, "disable_gradient_checkpointing", None)

    branch_results: dict[str, Any] = {}
    try:
        with _offload_diffusion_modules(pipeline) as offloaded:
            try:
                if not checkpointing_was_enabled and callable(enable_checkpointing):
                    enable_checkpointing()
                pipeline.vae.requires_grad_(False).eval().to(dtype=torch.float32)
                torch.cuda.reset_peak_memory_stats()
                _atomic_json(
                    output_dir / "runtime.json",
                    {
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "device_name": torch.cuda.get_device_name(0),
                        "source_commit": source_commit,
                        "offloaded_modules": list(offloaded),
                        "vae_original_dtype": str(original_vae_dtype),
                        "vae_effective_dtype": str(next(pipeline.vae.parameters()).dtype),
                        "diffqrcoder_revision": str(backend.settings.diffqrcoder_revision),
                        "parent_generation_duration_s": time.perf_counter() - generation_started,
                    },
                )
                for case in HOLDOUT_CASES:
                    result = _run_branch(
                        pipeline=pipeline,
                        parent=parents[case.case_id],
                        blueprint=blueprint,
                        policy=GLOBAL_POLICY,
                        config=FROZEN_CONFIG,
                        output_root=case_roots[case.case_id] / "refinement",
                    )
                    branch_results[case.case_id] = result
                    final_image = Image.open(result.final_image_path).convert("RGB")
                    _save_png(case_roots[case.case_id] / "final.png", final_image)
                    _case_comparison(
                        case_roots[case.case_id] / "comparison.png",
                        case.case_id,
                        parent_images[case.case_id],
                        final_image,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                pipeline.vae.to(dtype=original_vae_dtype)
                if not checkpointing_was_enabled and callable(disable_checkpointing):
                    disable_checkpointing()
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        if next(pipeline.vae.parameters()).dtype != original_vae_dtype:
            pipeline.vae.to(dtype=original_vae_dtype)
        gc.collect()
        torch.cuda.empty_cache()

    qr_images: dict[str, Image.Image] = {}
    for case in HOLDOUT_CASES:
        qr_images[f"{case.case_id}__parent"] = parent_images[case.case_id]
        qr_images[f"{case.case_id}__final"] = Image.open(
            branch_results[case.case_id].final_image_path
        ).convert("RGB")
    qr_verify = None if skip_qr_verify else _score_qr_verify(output_dir, PAYLOAD, qr_images)
    if skip_qr_verify:
        _atomic_json(output_dir / "qr-verify-evidence.json", {"skipped": True})

    rows: list[dict[str, Any]] = []
    contact_rows: list[tuple[str, Image.Image, Image.Image, str]] = []
    for case in HOLDOUT_CASES:
        result = branch_results[case.case_id]
        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        initial = trace[0]
        final = trace[-1]
        parent_exact = 0 if qr_verify is None else _extract_exact(qr_verify, f"{case.case_id}__parent")
        final_exact = 0 if qr_verify is None else _extract_exact(qr_verify, f"{case.case_id}__final")
        budget_safe = (
            float(final["latent_delta_rms"]) <= GLOBAL_POLICY.latent_radius_rms + 1e-9
            and float(final["lpips_loss"]) <= GLOBAL_POLICY.lpips_budget + 1e-9
            and float(final["core_mae"]) <= GLOBAL_POLICY.core_mae_budget + 1e-9
        )
        row = {
            "case_id": case.case_id,
            "seed": case.seed,
            "prompt": case.prompt,
            "parent_qr_verify_exact_presets": parent_exact,
            "final_qr_verify_exact_presets": final_exact,
            "qr_verify_exact_gain": final_exact - parent_exact,
            "parent_upstream_active_modules": int(initial["upstream_active_modules"]),
            "final_upstream_active_modules": int(final["upstream_active_modules"]),
            "parent_full_module_error_count": int(initial["full_module_error_count"]),
            "final_full_module_error_count": int(final["full_module_error_count"]),
            "full_module_error_reduction": int(initial["full_module_error_count"]) - int(final["full_module_error_count"]),
            "final_upstream_srl": float(final["upstream_srl"]),
            "final_lpips": float(final["lpips_loss"]),
            "final_core_mae": float(final["core_mae"]),
            "final_latent_delta_rms": float(final["latent_delta_rms"]),
            "visual_budget_safe": bool(budget_safe),
            "final_exact_success": final_exact >= 1,
            "refinement_exact_gain": final_exact > parent_exact,
        }
        rows.append(row)
        subtitle = (
            f"parent→final exact {parent_exact}→{final_exact}; "
            f"modules {row['parent_full_module_error_count']}→{row['final_full_module_error_count']}"
        )
        contact_rows.append(
            (
                case.case_id,
                parent_images[case.case_id],
                Image.open(result.final_image_path).convert("RGB"),
                subtitle,
            )
        )

    summary_csv = output_dir / "holdout-summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(output_dir / "holdout-summary.json", rows)

    safe_cases = sum(1 for row in rows if row["visual_budget_safe"])
    final_exact_cases = sum(1 for row in rows if row["final_exact_success"])
    exact_gain_cases = sum(1 for row in rows if row["refinement_exact_gain"])
    module_improved_cases = sum(1 for row in rows if row["full_module_error_reduction"] > 0)
    total_parent_exact = sum(int(row["parent_qr_verify_exact_presets"]) for row in rows)
    total_final_exact = sum(int(row["final_qr_verify_exact_presets"]) for row in rows)
    mean_module_reduction = sum(float(row["full_module_error_reduction"]) for row in rows) / len(rows)

    if safe_cases != CASE_COUNT:
        decision = "STOP_VISUAL_BUDGET_GENERALIZATION_FAILURE"
    elif final_exact_cases >= 5:
        decision = "GENERALIZES_PREPARE_E038_SCANNER_ROBUSTNESS"
    elif final_exact_cases >= 1 or module_improved_cases >= 7:
        decision = "PARTIAL_GENERALIZATION_PREPARE_E038_SCANNER_AWARE_TRUST"
    else:
        decision = "NO_GENERALIZATION_REVISIT_OBJECTIVE"

    verdict = {
        "experiment": EXPERIMENT,
        "source_commit": source_commit,
        "gamma": FROZEN_CONFIG.gamma,
        "gamma_preserved": True,
        "frozen_policy": asdict(GLOBAL_POLICY),
        "case_count": CASE_COUNT,
        "visual_budget_safe_cases": safe_cases,
        "final_exact_success_cases": final_exact_cases,
        "refinement_exact_gain_cases": exact_gain_cases,
        "module_improved_cases": module_improved_cases,
        "total_parent_conservative_exact_presets": total_parent_exact,
        "total_final_conservative_exact_presets": total_final_exact,
        "mean_full_module_error_reduction": mean_module_reduction,
        "generalization_pass": safe_cases == CASE_COUNT and final_exact_cases >= 5,
        "decision": decision,
        "production_ready": False,
        "automatic_expansion_authorized": False,
        "advisor_training_authorized": False,
    }
    _atomic_json(output_dir / "verdict.json", verdict)
    _global_contact_sheet(output_dir / "e037-final-contact-sheet.png", contact_rows)

    report = f"""# E037 — prospective mini-holdout of E036 global trust

- Cases: **{CASE_COUNT}** preregistered prompts/seeds.
- Gamma: **1000**, unchanged.
- Policy: E036 `e036_gamma1000_global_trust`, unchanged.
- Visual-budget-safe cases: **{safe_cases}/{CASE_COUNT}**.
- Cases with at least one conservative exact QR-Verify preset: **{final_exact_cases}/{CASE_COUNT}**.
- Cases with exact-preset gain over their own Stage-2 parent: **{exact_gain_cases}/{CASE_COUNT}**.
- Cases with fewer full-module errors: **{module_improved_cases}/{CASE_COUNT}**.
- Total conservative exact presets, parent → final: **{total_parent_exact} → {total_final_exact}**.
- Decision: **{decision}**.
- Production ready: **no**.

The contact sheet and each case directory contain the parent/final images side-by-side.
No E037 parameter was selected after observing holdout results.
"""
    _atomic_text(output_dir / "report.md", report)

    manifest = _build_manifest(output_dir)
    _atomic_json(output_dir / "e037-artifact-manifest.json", manifest)
    archive = output_dir.parent / f"{output_dir.name}.tar.gz"
    _write_deterministic_archive(output_dir, archive)
    _atomic_json(
        output_dir / "archive.json",
        {
            "path": str(archive),
            "sha256": _sha256_file(archive),
            "size_bytes": archive.stat().st_size,
            "note": "archive.json is intentionally written after the archive to avoid a recursive self-hash",
        },
    )
    return verdict


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--e036-results-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--skip-qr-verify", action="store_true")
    args = parser.parse_args()
    verdict = run_e037(
        output_dir=args.output_dir,
        e036_results_dir=args.e036_results_dir,
        source_commit=args.source_commit,
        skip_qr_verify=args.skip_qr_verify,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
