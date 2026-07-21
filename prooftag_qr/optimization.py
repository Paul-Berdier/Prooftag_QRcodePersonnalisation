from __future__ import annotations

import csv
import gc
import hashlib
import json
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .backends import ControlNetBackend
from .config import Settings
from .experiments import SRPGTrial, image_context_features
from .qr import generate_qr, module_error_rate
from .quality import image_change_metrics
from .quality_scoring import CLIPQualityScorer, project_embedding
from .schemas import GenerationRequest
from .srpg import run_srpg_controlnet_img2img
from .validation import QRValidator

STAGE2_SEED_STRIDE = 1_000_003


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    context_id: str
    axis: str
    prompt: str
    payload: str
    seed: int
    error_correction: str = "H"


def factorial_contexts() -> tuple[ExperimentContext, ...]:
    """Paired contexts that isolate prompt, seed and payload effects."""
    payload = "https://example.prooftag.test/t/e007-fixed"
    seed = 42
    prompts = {
        "botanical": (
            "premium botanical packaging, organic leaves and flowers, detailed illustration"
        ),
        "engraving": "monochrome botanical engraving, luxury label, elegant organic linework",
        "geometric": "premium geometric mosaic, blue and gold, intricate editorial illustration",
        "abstract": "colorful abstract paper cut artwork, flowing shapes, premium graphic design",
        "architecture": (
            "elegant art deco building facade, warm light, detailed architectural poster"
        ),
        "landscape": (
            "majestic waterfall in a lush forest, cinematic natural light, detailed painting"
        ),
    }
    contexts = [
        ExperimentContext(f"prompt-{name}", "prompt", prompt, payload, seed)
        for name, prompt in prompts.items()
    ]
    contexts.extend(
        ExperimentContext(
            f"seed-{value}",
            "seed",
            prompts["botanical"],
            payload,
            value,
        )
        for value in (314, 2026, 9001)
    )
    contexts.extend(
        ExperimentContext(
            f"payload-{suffix}",
            "payload",
            prompts["botanical"],
            f"https://example.prooftag.test/t/e007-{suffix}",
            seed,
        )
        for suffix in ("alpha", "bravo", "delta")
    )
    return tuple(contexts)


def holdout_contexts() -> tuple[ExperimentContext, ...]:
    return (
        ExperimentContext(
            "holdout-watercolor",
            "holdout",
            "delicate watercolor birds and flowers, luxury package design, soft paper texture",
            "https://example.prooftag.test/t/e007-holdout-1",
            71,
        ),
        ExperimentContext(
            "holdout-neon",
            "holdout",
            "futuristic neon city at night, cyan and magenta reflections, cinematic poster",
            "https://example.prooftag.test/t/e007-holdout-2",
            818,
        ),
        ExperimentContext(
            "holdout-ceramic",
            "holdout",
            "hand painted blue ceramic tiles, floral ornament, premium mediterranean design",
            "https://example.prooftag.test/t/e007-holdout-3",
            1618,
        ),
        ExperimentContext(
            "holdout-minimal",
            "holdout",
            "minimal japanese ink landscape, generous negative space, refined gallery print",
            "https://example.prooftag.test/t/e007-holdout-4",
            2718,
        ),
    )


def query_gpu_processes() -> list[dict[str, Any]]:
    """Return GPU compute processes visible before model loading."""
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    processes = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        pid, memory, name = [item.strip() for item in line.split(",", 2)]
        processes.append({"pid": int(pid), "used_memory_mib": float(memory), "name": name})
    return processes


def require_exclusive_gpu() -> None:
    processes = query_gpu_processes()
    if processes:
        details = ", ".join(
            f"PID {item['pid']}={item['used_memory_mib']:.0f} MiB ({item['name']})"
            for item in processes
        )
        raise RuntimeError(f"GPU not exclusive before E007 model loading: {details}")


class E007Experiment:
    def __init__(self, settings: Settings, experiment_name: str) -> None:
        self.settings = settings
        self.experiment_name = experiment_name
        self.run_dir = settings.data_dir / "parameter-search" / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.run_dir / "results.jsonl"
        self.backend = ControlNetBackend(settings)
        self.pipeline = self.backend._load()
        self.validator = QRValidator()
        self.quality = CLIPQualityScorer(settings.model_cache_dir, device="cpu")

    def rows(self) -> list[dict[str, Any]]:
        if not self.results_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.results_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def _append(self, row: dict[str, Any]) -> None:
        with self.results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _stage1_hash(trial: SRPGTrial) -> str:
        """Hash only inputs that can change the first diffusion.

        This guarantees that every Stage-2 candidate evaluated for an adaptive request starts
        from the exact same cached pixels, rather than merely a statistically equivalent rerun.
        """
        signature = {
            "steps": trial.base_steps,
            "strength": trial.base_strength,
            "guidance_scale": trial.base_guidance_scale,
            "controlnet_scale": trial.base_controlnet_scale,
            "negative_prompt_profile": trial.negative_prompt_profile,
        }
        encoded = json.dumps(signature, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()[:12]

    def _raw_path(self, phase: str, context: ExperimentContext, trial: SRPGTrial) -> Path:
        context_dir = self.run_dir / phase / context.context_id
        context_dir.mkdir(parents=True, exist_ok=True)
        return context_dir / f"raw-{self._stage1_hash(trial)}.png"

    def _generate_raw(
        self,
        phase: str,
        context: ExperimentContext,
        trial: SRPGTrial,
        blueprint: Any,
    ) -> Image.Image:
        raw_path = self._raw_path(phase, context, trial)
        if raw_path.exists():
            return Image.open(raw_path).convert("RGB")
        request = GenerationRequest(
            payload=context.payload,
            prompt=context.prompt,
            negative_prompt=trial.negative_prompt,
            backend="controlnet",
            error_correction=context.error_correction,
            seed=context.seed,
            steps=trial.base_steps,
            strength=trial.base_strength,
            guidance_scale=trial.base_guidance_scale,
            controlnet_scale=trial.base_controlnet_scale,
            max_attempts=1,
        )
        raw = self.backend.generate(request, blueprint, context.seed)
        raw.save(raw_path)
        blueprint.image.save(raw_path.with_name("qr-control.png"))
        return raw

    def execute(
        self,
        phase: str,
        context: ExperimentContext,
        trial: SRPGTrial,
    ) -> dict[str, Any]:
        key = f"{phase}:{context.context_id}:{trial.name}"
        existing = {row["key"]: row for row in self.rows()}
        if key in existing:
            return existing[key]
        started = time.perf_counter()
        result = None
        raw = None
        try:
            import torch

            blueprint = generate_qr(
                context.payload,
                context.error_correction,
                size=512,
            )
            raw = self._generate_raw(phase, context, trial, blueprint)
            raw_quality = self.quality.score(raw, context.prompt)
            raw_image_embedding, prompt_embedding = self.quality.embeddings(raw, context.prompt)
            context_features = image_context_features(raw, blueprint)
            context_features.update(
                {
                    f"prompt_clip_{index:02d}": value
                    for index, value in enumerate(project_embedding(prompt_embedding))
                }
            )
            context_features.update(
                {
                    f"raw_clip_{index:02d}": value
                    for index, value in enumerate(
                        project_embedding(raw_image_embedding, seed=20260722)
                    )
                }
            )
            generator = torch.Generator(device=self.settings.device).manual_seed(
                (
                    context.seed
                    + self.settings.srpg_seed_offset
                    + trial.stage2_seed_index * STAGE2_SEED_STRIDE
                )
                % (2**32)
            )
            result = run_srpg_controlnet_img2img(
                self.pipeline,
                raw,
                blueprint,
                prompt=context.prompt,
                negative_prompt=trial.negative_prompt,
                guidance_scale=trial.guidance_scale,
                generator=generator,
                config=trial.to_srpg_config(),
            )
            duration = time.perf_counter() - started
            records = self.validator.validate(result.image, context.payload)
            exact = sum(record.exact_payload_match for record in records)
            originals = [record for record in records if record.scenario == "original"]
            original_exact = sum(record.exact_payload_match for record in originals)
            quality = self.quality.score(result.image, context.prompt)
            change = image_change_metrics(result.image, raw)
            context_dir = self.run_dir / phase / context.context_id
            image_path = context_dir / f"{trial.name}.png"
            result.image.save(image_path)
            image_path.with_suffix(".validations.json").write_text(
                json.dumps([asdict(record) for record in records], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            with image_path.with_suffix(".steps.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(asdict(result.steps[0]).keys()))
                writer.writeheader()
                writer.writerows(asdict(step) for step in result.steps)
            row = {
                "key": key,
                "phase": phase,
                "context_id": context.context_id,
                "axis": context.axis,
                "trial": trial.name,
                "status": "ok",
                "timestamp": datetime.now(UTC).isoformat(),
                "prompt": context.prompt,
                "payload_hash": hashlib.sha256(context.payload.encode()).hexdigest(),
                "seed": context.seed,
                "error_correction": context.error_correction,
                "parameters": asdict(trial),
                "context_features": context_features,
                "qr_version": blueprint.version,
                "matrix_modules": int(blueprint.matrix.shape[0]),
                "pass_rate": exact / len(records),
                "strict_all": exact == len(records),
                "passed": exact,
                "validations": len(records),
                "original_pass_rate": original_exact / len(originals),
                "module_error_rate": module_error_rate(result.image, blueprint),
                "mean_absolute_change": change["mean_absolute_change"],
                "changed_pixel_ratio": change["changed_pixel_ratio"],
                "clip_similarity": quality.clip_similarity,
                "clip_score": quality.clip_score,
                "clip_aesthetic": quality.clip_aesthetic,
                "raw_clip_similarity": raw_quality.clip_similarity,
                "raw_clip_score": raw_quality.clip_score,
                "raw_clip_aesthetic": raw_quality.clip_aesthetic,
                "duration_seconds": duration,
                "peak_gpu_memory_mib": result.peak_gpu_memory_allocated_mib,
                "gradient_clip_rate": sum(step.gradient_clipped for step in result.steps)
                / len(result.steps),
                "internal_accepted": result.accepted,
                "internal_rejection_reason": result.rejection_reason,
                "image": str(image_path),
                "raw_image": str(self._raw_path(phase, context, trial)),
            }
        except Exception as exc:
            duration = time.perf_counter() - started
            row = {
                "key": key,
                "phase": phase,
                "context_id": context.context_id,
                "axis": context.axis,
                "trial": trial.name,
                "status": "error",
                "timestamp": datetime.now(UTC).isoformat(),
                "parameters": asdict(trial),
                "duration_seconds": duration,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            self._append(row)
            if exc.__class__.__name__ == "OutOfMemoryError":
                raise RuntimeError(
                    "E007 stopped on CUDA OOM; verify exclusive GPU ownership before resuming"
                ) from exc
            return row
        finally:
            del result
            del raw
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        self._append(row)
        return row
