import numpy as np
from PIL import Image

from prooftag_qr.artifacts import LocalArtifactStore
from prooftag_qr.backends import GenerationBackend, QRBackend
from prooftag_qr.config import Settings
from prooftag_qr.domain import ValidationRecord
from prooftag_qr.repository import RunRepository
from prooftag_qr.schemas import GenerationRequest
from prooftag_qr.service import GenerationService
from prooftag_qr.validation import OpenCVDecoder, QRValidator


class VariantBackend(GenerationBackend):
    def generate(self, request, blueprint, seed):
        noise = np.random.default_rng(seed).integers(0, 256, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(noise)

    def variants(self, candidate, blueprint, **kwargs):
        return [("raw", candidate), ("repaired", blueprint.image)]


class RegenerationBeforeGlobalBackend(GenerationBackend):
    def __init__(self, second_attempt_decodes: bool):
        self.calls = 0
        self.second_attempt_decodes = second_attempt_decodes

    def generate(self, request, blueprint, seed):
        self.calls += 1
        if self.second_attempt_decodes and self.calls == 2:
            return blueprint.image.copy()
        noise = np.random.default_rng(seed).integers(0, 256, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(noise)

    def variants(self, candidate, blueprint, **kwargs):
        return [
            ("raw", candidate),
            ("uncertain_16", candidate),
            ("centers_85", blueprint.image),
        ]


class TwoAcceptedVariantsBackend(GenerationBackend):
    def generate(self, request, blueprint, seed):
        return Image.new("RGB", blueprint.image.size, "black")

    def variants(self, candidate, blueprint, **kwargs):
        return [
            ("first_valid", Image.new("RGB", blueprint.image.size, "red")),
            ("less_changed_valid", Image.new("RGB", blueprint.image.size, "blue")),
        ]


class ForcedResearchBackend(GenerationBackend):
    def __init__(self):
        self.generate_calls = 0

    def generate(self, request, blueprint, seed):
        self.generate_calls += 1
        return Image.new("RGB", blueprint.image.size, "black")

    def variants(self, candidate, blueprint, **kwargs):
        yield "raw", candidate
        yield "srpg", Image.new("RGB", blueprint.image.size, "green")
        yield "centers_95", blueprint.image


class RefinementCallbackBackend(GenerationBackend):
    def __init__(self):
        self.summary = None

    def generate(self, request, blueprint, seed):
        return blueprint.image.copy()

    def variants(self, candidate, blueprint, **kwargs):
        callback = kwargs["validation_callback"]
        self.summary = callback(candidate, 0)
        yield "srmpgd", candidate


class AutoFallbackBackend(GenerationBackend):
    def generate(self, request, blueprint, seed):
        return Image.new("RGB", blueprint.image.size, "red")

    def variants(self, candidate, blueprint, **kwargs):
        yield "raw", candidate
        yield "srpg", Image.new("RGB", blueprint.image.size, "black")


class ReferenceLimitedValidator:
    def validate(self, image, expected_payload):
        readable = image.getpixel((0, 0)) != (0, 0, 0)
        return [
            ValidationRecord(
                decoder="fake",
                scenario="original",
                success=readable,
                exact_payload_match=readable,
                latency_ms=0.0,
                decoded_hash=None,
                parameters={},
            ),
            ValidationRecord(
                decoder="fake",
                scenario="unsupported_blur",
                success=False,
                exact_payload_match=False,
                latency_ms=0.0,
                decoded_hash=None,
                parameters={},
            ),
        ]


class ColorValidator:
    def validate(self, image, expected_payload):
        exact = image.getpixel((0, 0)) != (0, 0, 0)
        return [
            ValidationRecord(
                decoder="fake",
                scenario="original",
                success=exact,
                exact_payload_match=exact,
                latency_ms=0.0,
                decoded_hash=None,
                parameters={},
            )
        ]


def test_reference_pipeline_accepts_and_persists_run(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_cache_dir=tmp_path / "models",
        validation_min_pass_rate=0.7,
        max_attempts=1,
    )
    settings.ensure_directories()
    repository = RunRepository(settings.database_url)
    service = GenerationService(
        settings=settings,
        repository=repository,
        artifact_store=LocalArtifactStore(settings.artifacts_dir),
        backends={"qr": QRBackend()},
        validator=QRValidator(decoders=[OpenCVDecoder()]),
    )

    run = service.generate(
        GenerationRequest(
            payload="https://example.prooftag.test/t/service-test",
            backend="qr",
            seed=7,
            max_attempts=1,
        )
    )

    assert run.status == "accepted"
    assert run.exact_payload_match is True
    assert run.image_path is not None
    assert run.attempt_details[0].accepted
    assert repository.get(run.id) is not None
    assert repository.summary()["accepted_runs"] == 1


def test_pipeline_selects_a_valid_repair_without_regenerating(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_cache_dir=tmp_path / "models",
        validation_min_pass_rate=1.0,
        max_attempts=1,
        save_debug_artifacts=True,
    )
    settings.ensure_directories()
    repository = RunRepository(settings.database_url)
    service = GenerationService(
        settings=settings,
        repository=repository,
        artifact_store=LocalArtifactStore(settings.artifacts_dir),
        backends={"controlnet": VariantBackend()},
        validator=QRValidator(decoders=[OpenCVDecoder()]),
    )

    run = service.generate(
        GenerationRequest(
            payload="https://example.prooftag.test/t/adaptive-repair",
            backend="controlnet",
            seed=7,
            max_attempts=1,
        )
    )

    assert run.status == "accepted"
    assert run.attempts == 1
    assert run.scan_pass_rate == 1.0
    assert run.attempt_details[0].accepted
    assert (settings.artifacts_dir / run.id / "variants" / "raw.png").is_file()


def test_pipeline_regenerates_before_using_a_global_repair(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_cache_dir=tmp_path / "models",
        validation_min_pass_rate=1.0,
        max_attempts=3,
        regenerate_before_global_repair=True,
        save_debug_artifacts=True,
    )
    settings.ensure_directories()
    backend = RegenerationBeforeGlobalBackend(second_attempt_decodes=True)
    service = GenerationService(
        settings=settings,
        repository=RunRepository(settings.database_url),
        artifact_store=LocalArtifactStore(settings.artifacts_dir),
        backends={"controlnet": backend},
        validator=QRValidator(decoders=[OpenCVDecoder()]),
    )

    run = service.generate(
        GenerationRequest(
            payload="https://example.prooftag.test/t/regenerate-first",
            backend="controlnet",
            seed=42,
            max_attempts=3,
        )
    )

    variants = settings.artifacts_dir / run.id / "variants"
    assert run.status == "accepted"
    assert run.attempts == 2
    assert backend.calls == 2
    assert [attempt.accepted for attempt in run.attempt_details] == [False, True]
    assert (variants / "attempt_1_raw.png").is_file()
    assert (variants / "attempt_2_raw.png").is_file()


def test_pipeline_uses_global_repair_only_on_the_last_attempt(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_cache_dir=tmp_path / "models",
        validation_min_pass_rate=1.0,
        max_attempts=2,
        regenerate_before_global_repair=True,
    )
    settings.ensure_directories()
    backend = RegenerationBeforeGlobalBackend(second_attempt_decodes=False)
    service = GenerationService(
        settings=settings,
        repository=RunRepository(settings.database_url),
        artifact_store=LocalArtifactStore(settings.artifacts_dir),
        backends={"controlnet": backend},
        validator=QRValidator(decoders=[OpenCVDecoder()]),
    )

    run = service.generate(
        GenerationRequest(
            payload="https://example.prooftag.test/t/global-last",
            backend="controlnet",
            seed=7,
            max_attempts=2,
        )
    )

    assert run.status == "accepted"
    assert run.attempts == 2
    assert backend.calls == 2
    assert run.attempt_details[0].accepted is False
    assert run.attempt_details[1].accepted is True


def test_pipeline_selects_least_changed_image_among_all_valid_variants(tmp_path, monkeypatch):
    from prooftag_qr import service as service_module

    def change_metrics(image, reference):
        color = image.getpixel((0, 0))
        mean_change = 0.20 if color == (255, 0, 0) else 0.05
        return {
            "changed_pixel_ratio": mean_change,
            "mean_absolute_change": mean_change,
        }

    monkeypatch.setattr(service_module, "image_change_metrics", change_metrics)
    settings = Settings(
        data_dir=tmp_path / "data",
        model_cache_dir=tmp_path / "models",
        validation_min_pass_rate=1.0,
        max_attempts=1,
    )
    settings.ensure_directories()
    service = GenerationService(
        settings=settings,
        repository=RunRepository(settings.database_url),
        artifact_store=LocalArtifactStore(settings.artifacts_dir),
        backends={"controlnet": TwoAcceptedVariantsBackend()},
        validator=ColorValidator(),
    )

    run = service.generate(
        GenerationRequest(
            payload="https://example.prooftag.test/t/least-changed",
            backend="controlnet",
            max_attempts=1,
        )
    )

    assert run.status == "accepted"
    with Image.open(run.image_path) as selected:
        assert selected.getpixel((0, 0)) == (0, 0, 255)


def test_forced_laboratory_output_does_not_fall_back_to_qr_repair(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_cache_dir=tmp_path / "models",
        validation_min_pass_rate=1.0,
        max_attempts=1,
        save_debug_artifacts=True,
    )
    settings.ensure_directories()
    backend = ForcedResearchBackend()
    service = GenerationService(
        settings=settings,
        repository=RunRepository(settings.database_url),
        artifact_store=LocalArtifactStore(settings.artifacts_dir),
        backends={"controlnet": backend},
        validator=ColorValidator(),
    )

    run = service.generate(
        GenerationRequest(
            payload="https://example.prooftag.test/t/forced-srpg",
            backend="controlnet",
            max_attempts=1,
        ),
        target_variant="srpg",
    )

    assert run.status == "accepted"
    assert run.selection_mode == "forced"
    assert run.selected_variant == "srpg"
    assert backend.generate_calls == 1
    assert run.quality_metrics["stage1_mean_absolute_change"] > 0
    assert run.quality_metrics["stage1_changed_pixel_ratio"] > 0
    with Image.open(run.image_path) as selected:
        assert selected.getpixel((0, 0)) == (0, 128, 0)
    assert not (settings.artifacts_dir / run.id / "variants" / "centers_95.png").exists()


def test_service_supplies_full_decoder_summary_to_srmpgd(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_cache_dir=tmp_path / "models",
        validation_min_pass_rate=1.0,
        max_attempts=1,
    )
    settings.ensure_directories()
    backend = RefinementCallbackBackend()
    service = GenerationService(
        settings=settings,
        repository=RunRepository(settings.database_url),
        artifact_store=LocalArtifactStore(settings.artifacts_dir),
        backends={"controlnet": backend},
        validator=ColorValidator(),
    )

    run = service.generate(
        GenerationRequest(
            payload="https://example.prooftag.test/t/srmpgd-callback",
            backend="controlnet",
            max_attempts=1,
        ),
        target_variant="srmpgd",
    )

    assert run.status == "accepted"
    assert backend.summary == {
        "passed": 1,
        "total": 1,
        "pass_rate": 1.0,
        "strict_all": True,
        "decoder_pass_rates": {"fake": 1.0},
        "scenario_pass_rates": {"original": 1.0},
        "worst_decoder_pass_rate": 1.0,
        "worst_scenario_pass_rate": 1.0,
    }


def test_forced_laboratory_output_reuses_supplied_stage1_without_regeneration(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_cache_dir=tmp_path / "models",
        validation_min_pass_rate=1.0,
        max_attempts=1,
    )
    settings.ensure_directories()
    backend = ForcedResearchBackend()
    service = GenerationService(
        settings=settings,
        repository=RunRepository(settings.database_url),
        artifact_store=LocalArtifactStore(settings.artifacts_dir),
        backends={"controlnet": backend},
        validator=ColorValidator(),
    )
    shared_stage1 = Image.new("RGB", (512, 512), "red")

    run = service.generate(
        GenerationRequest(
            payload="https://example.prooftag.test/t/reused-stage1",
            backend="controlnet",
            max_attempts=1,
        ),
        raw_candidate_override=shared_stage1,
        target_variant="srpg",
        stage1_source_run_id="source-run",
    )

    assert backend.generate_calls == 0
    assert run.stage1_reused is True
    assert run.stage1_source_run_id == "source-run"
    assert run.selected_variant == "srpg"
    assert service.last_raw_candidate.getpixel((0, 0)) == (255, 0, 0)


def test_reference_limited_validation_is_normalized_in_the_service(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_cache_dir=tmp_path / "models",
        validation_min_pass_rate=1.0,
        max_attempts=1,
    )
    settings.ensure_directories()
    service = GenerationService(
        settings=settings,
        repository=RunRepository(settings.database_url),
        artifact_store=LocalArtifactStore(settings.artifacts_dir),
        backends={"controlnet": AutoFallbackBackend()},
        validator=ReferenceLimitedValidator(),
    )

    run = service.generate(
        GenerationRequest(
            payload="https://example.prooftag.test/t/normalized",
            backend="controlnet",
            max_attempts=1,
        ),
        target_variant="raw",
    )

    assert run.status == "accepted"
    assert run.scan_pass_rate == 1.0
    assert run.quality_metrics["validation_raw_pass_rate"] == 0.5
    assert run.quality_metrics["validation_reference_pass_rate"] == 0.5
    assert run.quality_metrics["validation_normalized_pass_rate"] == 1.0


def test_auto_mode_keeps_stage1_when_stage2_loses_scannability(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        model_cache_dir=tmp_path / "models",
        validation_min_pass_rate=1.0,
        max_attempts=1,
    )
    settings.ensure_directories()
    service = GenerationService(
        settings=settings,
        repository=RunRepository(settings.database_url),
        artifact_store=LocalArtifactStore(settings.artifacts_dir),
        backends={"controlnet": AutoFallbackBackend()},
        validator=ColorValidator(),
    )

    run = service.generate(
        GenerationRequest(
            payload="https://example.prooftag.test/t/auto-fallback",
            backend="controlnet",
            max_attempts=1,
        )
    )

    assert run.status == "accepted"
    assert run.selection_mode == "delivery"
    assert run.selected_variant == "raw"
    assert run.quality_metrics["selection_auto_mode"] == 1.0
    assert run.quality_metrics["selection_preserved_stage1"] == 1.0
    with Image.open(run.image_path) as selected:
        assert selected.getpixel((0, 0)) == (255, 0, 0)
