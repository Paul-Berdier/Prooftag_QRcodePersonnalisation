from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from datetime import UTC, datetime

from PIL import Image

from . import metrics
from .artifacts import ArtifactStore
from .backends import GLOBAL_REPAIR_VARIANTS, GenerationBackend
from .config import Settings
from .domain import AttemptRecord, RunRecord
from .qr import generate_diffqrcoder_qr, generate_qr, module_error_rate
from .quality import image_change_metrics, image_quality_metrics
from .repository import RunRepository
from .schemas import GenerationRequest
from .validation import QRValidator, summarize_validation_records

logger = logging.getLogger(__name__)


class GenerationService:
    def __init__(
        self,
        settings: Settings,
        repository: RunRepository,
        artifact_store: ArtifactStore,
        backends: dict[str, GenerationBackend],
        validator: QRValidator,
    ):
        self.settings = settings
        self.repository = repository
        self.artifact_store = artifact_store
        self.backends = backends
        self.validator = validator
        self._generation_lock = threading.Lock()
        self._last_raw_candidate: Image.Image | None = None

    @property
    def last_raw_candidate(self) -> Image.Image | None:
        return self._last_raw_candidate.copy() if self._last_raw_candidate is not None else None

    @staticmethod
    def _matches_target_variant(variant_name: str, target_variant: str) -> bool:
        if target_variant == "latent":
            return variant_name.endswith("latent_srl")
        return variant_name == target_variant

    def generate(
        self,
        request: GenerationRequest,
        *,
        raw_candidate_override: Image.Image | None = None,
        target_variant: str | None = None,
        stage1_source_run_id: str | None = None,
    ) -> RunRecord:
        started = time.perf_counter()
        backend_name = request.backend or self.settings.default_backend
        best_variant = "raw"
        best_attempt = 1
        best_raw_candidate: Image.Image | None = None
        self._last_raw_candidate = None
        run = RunRecord(
            id=str(uuid.uuid4()),
            created_at=datetime.now(UTC),
            completed_at=None,
            status="running",
            backend=backend_name,
            prompt=request.prompt,
            payload_hash=hashlib.sha256(request.payload.encode()).hexdigest(),
            seed=request.seed,
            selection_mode="forced" if target_variant else "delivery",
            stage1_reused=raw_candidate_override is not None,
            stage1_source_run_id=stage1_source_run_id,
        )
        self.repository.save(run)
        metrics.ACTIVE_RUNS.inc()
        logger.info("generation_started", extra={"run_id": run.id, "backend": backend_name})

        try:
            blueprint = (
                generate_diffqrcoder_qr(
                    request.payload,
                    request.error_correction,
                    version=self.settings.diffqrcoder_qr_version,
                    mask_pattern=self.settings.diffqrcoder_qr_mask_pattern,
                    module_size=self.settings.diffqrcoder_qr_module_size,
                )
                if self.settings.diffqrcoder_upstream_enabled
                else generate_qr(request.payload, request.error_correction)
            )
            run.qr_version = blueprint.version
            backend = self.backends[backend_name]
            max_attempts = request.max_attempts or self.settings.max_attempts
            best = None
            best_records = []
            best_pass_rate = -1.0
            best_changed_pixel_ratio = float("inf")
            generation_ms = 0.0
            validation_ms = 0.0

            with self._generation_lock:
                for attempt in range(max_attempts):
                    run.attempts = attempt + 1
                    generation_started = time.perf_counter()
                    reuse_override = raw_candidate_override is not None and attempt == 0
                    raw_candidate = (
                        raw_candidate_override.copy()
                        if reuse_override
                        else backend.generate(request, blueprint, request.seed + attempt)
                    )
                    attempt_generation_ms = (time.perf_counter() - generation_started) * 1000
                    generation_ms += attempt_generation_ms
                    metrics.DURATION.labels(backend_name, "generation").observe(
                        attempt_generation_ms / 1000
                    )

                    attempt_best = None
                    attempt_best_records = []
                    attempt_best_pass_rate = -1.0
                    attempt_best_module_error_rate = 1.0
                    attempt_best_changed_pixel_ratio = float("inf")
                    attempt_best_mean_absolute_change = float("inf")
                    attempt_best_variant = "raw"
                    attempt_best_accepted = False
                    attempt_validation_ms = 0.0
                    attempt_accepted = False
                    target_found = target_variant is None
                    allow_global_repair = (
                        backend_name != "controlnet"
                        or not self.settings.regenerate_before_global_repair
                        or attempt + 1 == max_attempts
                    )

                    if self.settings.save_debug_artifacts:
                        self.artifact_store.save_variant(run.id, "stage1_raw", raw_candidate)

                    def validate_refinement_state(
                        image: Image.Image,
                        iteration: int,
                    ) -> dict[str, object]:
                        del iteration
                        validation_kwargs = (
                            backend.validation_kwargs("srmpgd")
                            if hasattr(backend, "validation_kwargs")
                            else {}
                        )
                        refinement_records = self.validator.validate(
                            image,
                            request.payload,
                            **validation_kwargs,
                        )
                        passed = sum(
                            item.exact_payload_match for item in refinement_records
                        )
                        total = len(refinement_records)
                        return {
                            "passed": passed,
                            "total": total,
                            "pass_rate": passed / total if total else 0.0,
                            "strict_all": total > 0 and passed == total,
                            **summarize_validation_records(refinement_records),
                        }

                    variant_iterator = iter(
                        backend.variants(
                            raw_candidate,
                            blueprint,
                            request=request,
                            seed=request.seed + attempt,
                            run_id=run.id,
                            attempt=attempt + 1,
                            research_mode=target_variant is not None,
                            validation_callback=validate_refinement_state,
                        )
                    )
                    while True:
                        variant_generation_started = time.perf_counter()
                        try:
                            variant_name, candidate = next(variant_iterator)
                        except StopIteration:
                            break
                        variant_generation_ms = (
                            time.perf_counter() - variant_generation_started
                        ) * 1000
                        generation_ms += variant_generation_ms
                        attempt_generation_ms += variant_generation_ms
                        metrics.DURATION.labels(backend_name, "variant_generation").observe(
                            variant_generation_ms / 1000
                        )
                        if target_variant and not self._matches_target_variant(
                            variant_name, target_variant
                        ):
                            continue
                        target_found = True
                        if variant_name in GLOBAL_REPAIR_VARIANTS and not allow_global_repair:
                            continue
                        validation_started = time.perf_counter()
                        validation_kwargs = (
                            backend.validation_kwargs(variant_name)
                            if hasattr(backend, "validation_kwargs")
                            else {}
                        )
                        records = self.validator.validate(
                            candidate,
                            request.payload,
                            **validation_kwargs,
                        )
                        variant_validation_ms = (time.perf_counter() - validation_started) * 1000
                        attempt_validation_ms += variant_validation_ms
                        validation_ms += variant_validation_ms
                        metrics.DURATION.labels(backend_name, "validation").observe(
                            variant_validation_ms / 1000
                        )
                        exact_count = sum(item.exact_payload_match for item in records)
                        pass_rate = exact_count / len(records) if records else 0.0
                        metric_blueprint = (
                            backend.module_blueprint(variant_name, blueprint)
                            if hasattr(backend, "module_blueprint")
                            else blueprint
                        )
                        variant_module_error_rate = module_error_rate(
                            candidate,
                            metric_blueprint,
                        )
                        variant_quality = {
                            **image_quality_metrics(candidate),
                            **image_change_metrics(candidate, raw_candidate),
                        }
                        for item in records:
                            outcome = (
                                "exact"
                                if item.exact_payload_match
                                else ("wrong_payload" if item.success else "not_detected")
                            )
                            metrics.VALIDATIONS.labels(item.decoder, item.scenario, outcome).inc()
                            metrics.VALIDATION_DURATION.labels(item.decoder).observe(
                                item.latency_ms / 1000
                            )
                        original_ok = all(
                            item.exact_payload_match
                            for item in records
                            if item.scenario == "original"
                        )
                        accepted = (
                            original_ok
                            and pass_rate >= self.settings.validation_min_pass_rate
                            and (
                                backend.candidate_guard_ok(variant_name)
                                if hasattr(backend, "candidate_guard_ok")
                                else True
                            )
                        )
                        validation_failures = [
                            {
                                "decoder": item.decoder,
                                "scenario": item.scenario,
                                "outcome": (
                                    "wrong_payload"
                                    if item.success and not item.exact_payload_match
                                    else "not_detected"
                                ),
                            }
                            for item in records
                            if not item.exact_payload_match
                        ]
                        if backend_name == "controlnet":
                            metrics.REPAIR_VARIANTS.labels(
                                variant_name, "accepted" if accepted else "rejected"
                            ).inc()
                            metrics.REPAIR_VARIANT_SCAN_PASS_RATE.labels(variant_name).set(
                                pass_rate
                            )
                            metrics.REPAIR_VARIANT_MODULE_ERROR_RATE.labels(variant_name).set(
                                variant_module_error_rate
                            )
                            for quality_name, quality_value in variant_quality.items():
                                metrics.REPAIR_VARIANT_IMAGE_QUALITY.labels(
                                    variant_name, quality_name
                                ).set(quality_value)
                            debug_variant = variant_name
                            for prefix in (
                                "guided_latent_",
                                "srpg_latent_",
                                "guided_",
                                "srpg_",
                                "latent_",
                            ):
                                debug_variant = debug_variant.removeprefix(prefix)
                            if self.settings.save_debug_artifacts and debug_variant in {
                                "raw",
                                "guided",
                                "srpg",
                                "srl",
                                "rounded_16",
                                "rounded_32",
                                "rounded_48",
                                "perceptual_16",
                                "perceptual_16_strong",
                                "perceptual_32",
                                "perceptual_32_strong",
                                "perceptual_32_wide",
                                "perceptual_48",
                                "perceptual_64",
                                "incorrect_80",
                                "incorrect_85",
                                "uncertain_16",
                                "uncertain_32",
                                "uncertain_48",
                                "uncertain_64",
                                "tonal_90",
                                "tonal_95",
                            }:
                                self.artifact_store.save_variant(run.id, variant_name, candidate)
                                self.artifact_store.save_variant(
                                    run.id,
                                    f"attempt_{attempt + 1}_{variant_name}",
                                    candidate,
                                )
                            logger.info(
                                "repair_variant_validated",
                                extra={
                                    "run_id": run.id,
                                    "backend": backend_name,
                                    "repair_variant": variant_name,
                                    "status": "accepted" if accepted else "rejected",
                                    "attempt": attempt + 1,
                                    "seed": request.seed + attempt,
                                    "scan_pass_rate": round(pass_rate, 6),
                                    "module_error_rate": round(variant_module_error_rate, 6),
                                    "exact_payload_match": original_ok,
                                    "quality_metrics": variant_quality,
                                    "validation_failures": validation_failures,
                                },
                            )
                        variant_changed_pixel_ratio = variant_quality["changed_pixel_ratio"]
                        variant_mean_absolute_change = variant_quality["mean_absolute_change"]
                        accepted_is_better = accepted and (
                            not attempt_best_accepted
                            or (
                                variant_mean_absolute_change,
                                variant_changed_pixel_ratio,
                            )
                            < (
                                attempt_best_mean_absolute_change,
                                attempt_best_changed_pixel_ratio,
                            )
                        )
                        rejected_is_better = (
                            not accepted
                            and not attempt_best_accepted
                            and (
                                pass_rate > attempt_best_pass_rate
                                or (
                                    pass_rate == attempt_best_pass_rate
                                    and (
                                        variant_module_error_rate,
                                        variant_mean_absolute_change,
                                    )
                                    < (
                                        attempt_best_module_error_rate,
                                        attempt_best_mean_absolute_change,
                                    )
                                )
                            )
                        )
                        if accepted_is_better or rejected_is_better:
                            attempt_best = candidate
                            attempt_best_records = records
                            attempt_best_pass_rate = pass_rate
                            attempt_best_module_error_rate = variant_module_error_rate
                            attempt_best_changed_pixel_ratio = variant_changed_pixel_ratio
                            attempt_best_mean_absolute_change = variant_mean_absolute_change
                            attempt_best_variant = variant_name
                            attempt_best_accepted = accepted
                        if accepted:
                            attempt_accepted = True
                        if target_variant:
                            break

                    if not target_found:
                        raise RuntimeError(
                            f"Requested laboratory variant '{target_variant}' was not produced"
                        )

                    if self.settings.save_debug_artifacts:
                        for artifact_name, artifact_image in backend.debug_artifacts().items():
                            self.artifact_store.save_variant(
                                run.id,
                                artifact_name,
                                artifact_image,
                            )
                            self.artifact_store.save_variant(
                                run.id,
                                f"attempt_{attempt + 1}_{artifact_name}",
                                artifact_image,
                            )

                    if (
                        attempt_accepted
                        or attempt_best_pass_rate > best_pass_rate
                        or (
                            attempt_best_pass_rate == best_pass_rate
                            and attempt_best_changed_pixel_ratio < best_changed_pixel_ratio
                        )
                    ):
                        best = attempt_best
                        best_records = attempt_best_records
                        best_pass_rate = attempt_best_pass_rate
                        best_changed_pixel_ratio = attempt_best_changed_pixel_ratio
                        best_variant = attempt_best_variant
                        best_attempt = attempt + 1
                        best_raw_candidate = raw_candidate.copy()
                    run.attempt_details.append(
                        AttemptRecord(
                            attempt=attempt + 1,
                            seed=request.seed + attempt,
                            generation_ms=attempt_generation_ms,
                            validation_ms=attempt_validation_ms,
                            scan_pass_rate=attempt_best_pass_rate,
                            module_error_rate=attempt_best_module_error_rate,
                            accepted=attempt_accepted,
                        )
                    )
                    if attempt_accepted:
                        break
                    if (
                        backend_name == "controlnet"
                        and not allow_global_repair
                        and attempt + 1 < max_attempts
                    ):
                        metrics.REGENERATIONS.labels("targeted_repair_exhausted").inc()
                        logger.info(
                            "generation_retry_scheduled",
                            extra={
                                "run_id": run.id,
                                "backend": backend_name,
                                "attempt": attempt + 1,
                                "seed": request.seed + attempt,
                                "repair_variant": attempt_best_variant,
                                "scan_pass_rate": round(attempt_best_pass_rate, 6),
                            },
                        )

            if best is None:
                raise RuntimeError("The backend did not produce an image")
            run.validations = best_records
            run.scan_pass_rate = best_pass_rate
            run.exact_payload_match = all(
                item.exact_payload_match for item in best_records if item.scenario == "original"
            )
            run.module_error_rate = module_error_rate(best, blueprint)
            run.quality_metrics = image_quality_metrics(best)
            if backend_name == "controlnet" and best_raw_candidate is not None:
                run.quality_metrics.update(
                    {
                        f"stage1_{name}": value
                        for name, value in image_change_metrics(
                            best,
                            best_raw_candidate,
                        ).items()
                    }
                )
                run.quality_metrics.update(backend.diagnostics())
            run.selected_variant = best_variant
            if backend_name == "controlnet":
                metrics.REPAIR_SELECTED.labels(best_variant).inc()
            run.image_path = self.artifact_store.save_image(run.id, best)
            run.generation_ms = generation_ms
            run.validation_ms = validation_ms
            run.status = (
                "accepted"
                if run.exact_payload_match
                and run.scan_pass_rate >= self.settings.validation_min_pass_rate
                else "rejected"
            )
            metrics.SCAN_PASS_RATE.observe(run.scan_pass_rate)
            metrics.MODULE_ERROR_RATE.observe(run.module_error_rate)
            for name, value in run.quality_metrics.items():
                metrics.IMAGE_QUALITY.labels(name).set(value)
        except Exception as exc:
            run.status = "error"
            run.error = f"{type(exc).__name__}: {exc}"
            logger.exception("generation_failed", extra={"run_id": run.id, "backend": backend_name})
        finally:
            self._last_raw_candidate = (
                best_raw_candidate.copy() if best_raw_candidate is not None else None
            )
            run.completed_at = datetime.now(UTC)
            run.total_ms = (time.perf_counter() - started) * 1000
            self.repository.save(run)
            metrics.ACTIVE_RUNS.dec()
            metrics.RUNS.labels(backend_name, run.status).inc()
            metrics.ATTEMPTS.observe(run.attempts)
            metrics.DURATION.labels(backend_name, "total").observe(run.total_ms / 1000)
            logger.info(
                "generation_completed",
                extra={
                    "run_id": run.id,
                    "backend": backend_name,
                    "status": run.status,
                    "duration_ms": round(run.total_ms, 2),
                    "attempt": best_attempt,
                    "attempts": run.attempts,
                    "repair_variant": best_variant if backend_name == "controlnet" else None,
                },
            )
        return run
