import numpy as np
from PIL import Image

from prooftag_qr.artifacts import LocalArtifactStore
from prooftag_qr.backends import GenerationBackend, QRBackend
from prooftag_qr.config import Settings
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
