from datetime import UTC, datetime

import pytest
from PIL import Image
from pydantic import ValidationError

import prooftag_qr.lab as lab_module
from prooftag_qr.artifacts import LocalArtifactStore
from prooftag_qr.config import Settings
from prooftag_qr.domain import RunRecord
from prooftag_qr.lab import LabService
from prooftag_qr.lab_repository import LabRepository
from prooftag_qr.quality_scoring import CLIPQualityScore
from prooftag_qr.repository import RunRepository
from prooftag_qr.schemas import LabCampaignCreate, LabMethod, LabToolConfig


def test_lab_rejects_conflicting_stage2_tools():
    with pytest.raises(ValidationError, match="cannot be enabled together"):
        LabToolConfig(srpg_enabled=True, guided_rediffusion_enabled=True)


def test_lab_limits_cartesian_campaign_size():
    methods = [
        LabMethod(id=f"m{index}", name=f"Method {index}") for index in range(11)
    ]
    with pytest.raises(ValidationError, match="limited to 500"):
        LabCampaignCreate(
            name="too large",
            payload="https://example.test",
            prompts=[
                {"id": f"p{index}", "text": f"Prompt {index}"} for index in range(50)
            ],
            seeds=[1],
            methods=methods,
        )


def test_lab_persists_cpu_clip_scores_without_using_the_generation_gpu(
    tmp_path,
    monkeypatch,
):
    settings = Settings(
        data_dir=tmp_path,
        model_cache_dir=tmp_path / "models",
        lab_clip_scoring_enabled=True,
        device="cpu",
    )
    run_repository = RunRepository(tmp_path / "runs.sqlite3")
    artifact_store = LocalArtifactStore(tmp_path / "artifacts")
    run = RunRecord(
        id="run-quality",
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        status="accepted",
        backend="qr",
        prompt="botanical ink",
        payload_hash="a" * 64,
        seed=7,
    )
    run.image_path = artifact_store.save_image(
        run.id,
        Image.new("RGB", (64, 64), "white"),
    )
    run_repository.save(run)

    class FakeScorer:
        def __init__(self, *args, **kwargs):
            assert kwargs["device"] == "cpu"

        def score(self, image, prompt):
            assert image.size == (64, 64)
            assert prompt == "botanical ink"
            return CLIPQualityScore(
                clip_similarity=0.42,
                clip_score=1.05,
                clip_aesthetic=6.4,
            )

    monkeypatch.setattr(lab_module, "CLIPQualityScorer", FakeScorer)
    service = LabService(
        base_settings=settings,
        run_repository=run_repository,
        lab_repository=LabRepository(run_repository.engine),
        artifact_store=artifact_store,
        validator=object(),
    )
    try:
        service._score_quality(run, run.prompt)
    finally:
        service.shutdown()

    stored = run_repository.get(run.id)
    assert stored is not None
    assert stored.quality_metrics["clip_similarity"] == pytest.approx(0.42)
    assert stored.quality_metrics["clip_score"] == pytest.approx(1.05)
    assert stored.quality_metrics["clip_aesthetic"] == pytest.approx(6.4)
