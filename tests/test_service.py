from prooftag_qr.artifacts import LocalArtifactStore
from prooftag_qr.backends import QRBackend
from prooftag_qr.config import Settings
from prooftag_qr.repository import RunRepository
from prooftag_qr.schemas import GenerationRequest
from prooftag_qr.service import GenerationService
from prooftag_qr.validation import OpenCVDecoder, QRValidator


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
