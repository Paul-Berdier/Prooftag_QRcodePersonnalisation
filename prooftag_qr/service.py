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
from .qr import (
    diffqrcoder_structure_metrics,
    generate_diffqrcoder_qr,
    generate_qr,
    module_error_rate,
)
from .quality import image_change_metrics, image_quality_metrics, image_sha256
from .repository import RunRepository
from .schemas import GenerationRequest
from .validation import QRValidator, compare_validation_to_reference

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
            reference_validation_cache: dict[tuple[str, tuple[int, int]], list] = {}

            def validate_against_reference(
                image: Image.Image,
                variant_name: str,
            ) -> tuple[list, dict[str, object]]:
                validation_kwargs = (
                    backend.validation_kwargs(variant_name)
                    if hasattr(backend, "validation_kwargs")
                    else {}
                )
                match_mode = str(validation_kwargs.get("match_mode", "exact"))
                cache_key = (match_mode, image.size)
                reference_records = reference_validation_cache.get(cache_key)
                if reference_records is None:
                    reference_image = blueprint.image.convert("RGB").resize(
                        image.size,
                        Image.Resampling.NEAREST,
                    )
                    reference_records = self.validator.validate(
                        reference_image,
                        request.payload,
                        **validation_kwargs,
                    )
                    reference_validation_cache[cache_key] = reference_records
                records = self.validator.validate(
                    image,
                    request.payload,
                    **validation_kwargs,
                )
                return records, compare_validation_to_reference(
                    records,
                    reference_records,
                )

            def measure_module_error(
                image: Image.Image,
                variant_name: str,
            ) -> float:
                if hasattr(backend, "measure_module_error"):
                    return backend.measure_module_error(
                        variant_name,
                        image,
                        blueprint,
                    )
                metric_blueprint = (
                    backend.module_blueprint(variant_name, blueprint)
                    if hasattr(backend, "module_blueprint")
                    else blueprint
                )
                return module_error_rate(image, metric_blueprint)

            max_attempts = request.max_attempts or self.settings.max_attempts
            best = None
            best_records = []
            best_validation_summary: dict[str, object] = {}
            best_pass_rate = -1.0
            best_module_error_rate = 1.0
            best_changed_pixel_ratio = float("inf")
            best_accepted = False
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
                    attempt_best_validation_summary: dict[str, object] = {}
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
                        _, summary = validate_against_reference(
                            image,
                            "srmpgd",
                        )
                        return {
                            "passed": summary["normalized_passed"],
                            "total": summary["normalized_total"],
                            "pass_rate": summary["normalized_pass_rate"],
                            "strict_all": summary["normalized_strict_all"],
                            "decoder_pass_rates": summary["decoder_pass_rates"],
                            "scenario_pass_rates": summary["scenario_pass_rates"],
                            "worst_decoder_pass_rate": summary[
                                "worst_decoder_pass_rate"
                            ],
                            "worst_scenario_pass_rate": summary[
                                "worst_scenario_pass_rate"
                            ],
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
                        records, validation_summary = validate_against_reference(
                            candidate,
                            variant_name,
                        )
                        variant_validation_ms = (time.perf_counter() - validation_started) * 1000
                        attempt_validation_ms += variant_validation_ms
                        validation_ms += variant_validation_ms
                        metrics.DURATION.labels(backend_name, "validation").observe(
                            variant_validation_ms / 1000
                        )
                        pass_rate = float(
                            validation_summary["normalized_pass_rate"]
                        )
                        variant_module_error_rate = measure_module_error(
                            candidate,
                            variant_name,
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
                        original_ok = bool(
                            validation_summary["original_strict_all"]
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
                            attempt_best_validation_summary = validation_summary
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
                        if hasattr(backend, "debug_metadata"):
                            for metadata_name, payload in backend.debug_metadata().items():
                                self.artifact_store.save_metadata(
                                    run.id,
                                    metadata_name,
                                    payload,
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
                        best_validation_summary = attempt_best_validation_summary
                        best_pass_rate = attempt_best_pass_rate
                        best_module_error_rate = attempt_best_module_error_rate
                        best_changed_pixel_ratio = attempt_best_changed_pixel_ratio
                        best_accepted = attempt_best_accepted
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
            run.exact_payload_match = bool(
                best_validation_summary.get("original_strict_all", False)
            )
            run.module_error_rate = best_module_error_rate
            original_decoder_records = [
                record for record in best_records if record.scenario == "original"
            ]
            wechat_original = next(
                (
                    record
                    for record in original_decoder_records
                    if record.decoder == "wechat_qrcode"
                ),
                None,
            )
            run.quality_metrics = {
                **image_quality_metrics(best),
                "validation_raw_pass_rate": float(
                    best_validation_summary.get("raw_pass_rate", 0.0)
                ),
                "validation_reference_pass_rate": float(
                    best_validation_summary.get("reference_pass_rate", 0.0)
                ),
                "validation_normalized_pass_rate": float(
                    best_validation_summary.get("normalized_pass_rate", 0.0)
                ),
                "validation_normalized_passed": float(
                    best_validation_summary.get("normalized_passed", 0)
                ),
                "validation_normalized_total": float(
                    best_validation_summary.get("normalized_total", 0)
                ),
                "validation_original_passed": float(
                    best_validation_summary.get("original_passed", 0)
                ),
                "validation_original_total": float(
                    best_validation_summary.get("original_total", 0)
                ),
                # Stable aliases with honest semantics. Historical validation_*
                # fields and RunRecord.scan_pass_rate remain for compatibility.
                "synthetic_robustness_raw_pass_rate": float(
                    best_validation_summary.get("raw_pass_rate", 0.0)
                ),
                "synthetic_robustness_normalized_pass_rate": float(
                    best_validation_summary.get("normalized_pass_rate", 0.0)
                ),
                "synthetic_original_decoder_pass_rate": float(
                    best_validation_summary.get("original_passed", 0)
                )
                / max(float(best_validation_summary.get("original_total", 0)), 1.0),
                "synthetic_decoder_count": float(len(original_decoder_records)),
                "synthetic_metric_is_physical": 0.0,
                "wechat_qrcode_available": float(wechat_original is not None),
                "wechat_qrcode_original_exact": float(
                    bool(wechat_original and wechat_original.exact_payload_match)
                ),
            }
            if hasattr(self.validator, "validate_phone_proxy"):
                phone_started = time.perf_counter()
                validation_kwargs = (
                    backend.validation_kwargs(best_variant)
                    if hasattr(backend, "validation_kwargs")
                    else {}
                )
                phone_records = self.validator.validate_phone_proxy(
                    best,
                    request.payload,
                    **validation_kwargs,
                )
                phone_reference = blueprint.image.convert("RGB").resize(
                    best.size,
                    Image.Resampling.NEAREST,
                )
                phone_reference_records = self.validator.validate_phone_proxy(
                    phone_reference,
                    request.payload,
                    **validation_kwargs,
                )
                phone_summary = compare_validation_to_reference(
                    phone_records,
                    phone_reference_records,
                )
                phone_elapsed_ms = (time.perf_counter() - phone_started) * 1000
                validation_ms += phone_elapsed_ms
                run.validations.extend(phone_records)
                run.quality_metrics.update(
                    {
                        "phone_proxy_raw_pass_rate": float(
                            phone_summary["raw_pass_rate"]
                        ),
                        "phone_proxy_reference_pass_rate": float(
                            phone_summary["reference_pass_rate"]
                        ),
                        "phone_proxy_normalized_pass_rate": float(
                            phone_summary["normalized_pass_rate"]
                        ),
                        "phone_proxy_normalized_passed": float(
                            phone_summary["normalized_passed"]
                        ),
                        "phone_proxy_normalized_total": float(
                            phone_summary["normalized_total"]
                        ),
                        "phone_proxy_calibration_only": 1.0,
                        "software_preprocessing_proxy_normalized_pass_rate": float(
                            phone_summary["normalized_pass_rate"]
                        ),
                        "software_preprocessing_proxy_is_mobile_decoder": 0.0,
                    }
                )
                for item in phone_records:
                    outcome = (
                        "exact"
                        if item.exact_payload_match
                        else ("wrong_payload" if item.success else "not_detected")
                    )
                    metrics.VALIDATIONS.labels(
                        item.decoder,
                        item.scenario,
                        outcome,
                    ).inc()
                    metrics.VALIDATION_DURATION.labels(item.decoder).observe(
                        item.latency_ms / 1000
                    )
            if self.settings.diffqrcoder_upstream_enabled:
                try:
                    run.quality_metrics.update(
                        {
                            f"structure_{name}": value
                            for name, value in diffqrcoder_structure_metrics(
                                best,
                                blueprint,
                                padding_px=self.settings.diffqrcoder_qr_padding_px,
                                module_size=self.settings.diffqrcoder_qr_module_size,
                            ).items()
                        }
                    )
                except ValueError as exc:
                    logger.warning(
                        "structure_metrics_skipped",
                        extra={"run_id": run.id, "reason": str(exc)},
                    )
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
                run.quality_metrics["selection_auto_mode"] = float(
                    target_variant is None
                )
                run.quality_metrics["selection_preserved_stage1"] = float(
                    target_variant is None and best_variant == "raw"
                )
            run.provenance = {
                "final_image_sha256": image_sha256(best),
                "qr_reference_sha256": image_sha256(
                    blueprint.image.convert("RGB").resize(
                        best.size,
                        Image.Resampling.NEAREST,
                    )
                ),
                "selected_variant": best_variant,
            }
            if best_raw_candidate is not None:
                run.provenance["stage1_image_sha256"] = image_sha256(
                    best_raw_candidate
                )
            if hasattr(backend, "provenance"):
                run.provenance.update(backend.provenance())
            run.selected_variant = best_variant
            if backend_name == "controlnet":
                metrics.REPAIR_SELECTED.labels(best_variant).inc()
            run.image_path = self.artifact_store.save_image(run.id, best)
            run.generation_ms = generation_ms
            run.validation_ms = validation_ms
            run.status = "accepted" if best_accepted else "rejected"
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
