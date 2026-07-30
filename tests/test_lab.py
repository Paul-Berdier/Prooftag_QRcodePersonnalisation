import time
import uuid
from datetime import UTC, datetime

import pytest
from PIL import Image
from pydantic import ValidationError

import prooftag_qr.lab as lab_module
from prooftag_qr.artifacts import LocalArtifactStore
from prooftag_qr.config import Settings
from prooftag_qr.domain import RunRecord
from prooftag_qr.lab import LabService, laboratory_profiles
from prooftag_qr.lab_repository import LabRepository
from prooftag_qr.quality_scoring import CLIPQualityScore
from prooftag_qr.repository import RunRepository
from prooftag_qr.schemas import LabCampaignCreate, LabMethod, LabToolConfig


def test_lab_rejects_conflicting_stage2_tools():
    with pytest.raises(ValidationError, match="cannot be enabled together"):
        LabToolConfig(srpg_enabled=True, guided_rediffusion_enabled=True)


def test_lab_rejects_srmpgd_without_stage2_srpg():
    with pytest.raises(ValidationError, match="requires Stage 2 SRPG"):
        LabToolConfig(srmpgd_enabled=True)


def test_web_lab_exposes_only_the_pinned_diffqrcoder_chain():
    profiles = laboratory_profiles()

    assert [profile["id"] for profile in profiles] == [
        "qr_reference",
        "diffqrcoder_stage1",
        "diffqrcoder_srpg",
        "diffqrcoder_srmpgd",
        "diffqrcoder_binary_srpg",
    ]
    generated = [profile for profile in profiles if profile["backend"] == "controlnet"]
    assert all(
        profile["model"]["diffqrcoder_upstream_enabled"] is True
        for profile in generated
    )
    assert {profile["model"]["controlnet_model_id"] for profile in generated} == {
        "monster-labs/control_v1p_sd15_qrcode_monster"
    }
    srpg = next(profile for profile in profiles if profile["id"] == "diffqrcoder_srpg")
    binary = next(
        profile for profile in profiles if profile["id"] == "diffqrcoder_binary_srpg"
    )
    assert (
        srpg["tools"]["settings"]["diffqrcoder_stage2_target_mode"]
        == "qart_url_fragment"
    )
    assert (
        binary["tools"]["settings"]["diffqrcoder_stage2_target_mode"]
        == "binary_exact"
    )


def test_srmpgd_reuses_the_matching_srpg_stage2_cache_key(tmp_path):
    settings = Settings(data_dir=tmp_path, device="cpu")
    run_repository = RunRepository(tmp_path / "runs.sqlite3")
    service = LabService(
        base_settings=settings,
        run_repository=run_repository,
        lab_repository=LabRepository(run_repository.engine),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        validator=object(),
    )
    profiles = laboratory_profiles()
    srpg = LabMethod.model_validate(
        next(item for item in profiles if item["id"] == "diffqrcoder_srpg")
    )
    srmpgd = LabMethod.model_validate(
        next(item for item in profiles if item["id"] == "diffqrcoder_srmpgd")
    )
    try:
        srpg_key = service._stage2_cache_key(
            srpg,
            "blue courtyard",
            "easynegative",
            51001,
            "M",
            "https://ptag.io/t/cache",
        )
        srmpgd_key = service._stage2_cache_key(
            srmpgd,
            "blue courtyard",
            "easynegative",
            51001,
            "M",
            "https://ptag.io/t/cache",
        )
    finally:
        service.shutdown()

    assert srpg_key == srmpgd_key


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
        backend="controlnet",
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


def test_lab_reuses_the_exact_stage1_and_forces_each_research_output(
    tmp_path,
    monkeypatch,
):
    settings = Settings(
        data_dir=tmp_path,
        model_cache_dir=tmp_path / "models",
        lab_clip_scoring_enabled=False,
        device="cpu",
    )
    run_repository = RunRepository(tmp_path / "runs.sqlite3")
    artifact_store = LocalArtifactStore(tmp_path / "artifacts")
    service = LabService(
        base_settings=settings,
        run_repository=run_repository,
        lab_repository=LabRepository(run_repository.engine),
        artifact_store=artifact_store,
        validator=object(),
    )
    calls = []

    class FakeGenerationService:
        def __init__(self):
            self.backends = {}
            self.last_raw_candidate = None

        def generate(
            self,
            request,
            *,
            raw_candidate_override=None,
            target_variant=None,
            stage1_source_run_id=None,
        ):
            raw = (
                raw_candidate_override.copy()
                if raw_candidate_override is not None
                else Image.new("RGB", (512, 512), "red")
            )
            self.last_raw_candidate = raw.copy()
            image = (
                raw
                if target_variant == "raw"
                else Image.new("RGB", (512, 512), "green")
            )
            run = RunRecord(
                id=str(uuid.uuid4()),
                created_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                status="accepted",
                backend=request.backend,
                prompt=request.prompt,
                payload_hash="b" * 64,
                seed=request.seed,
                selected_variant=target_variant,
                selection_mode="forced",
                stage1_reused=raw_candidate_override is not None,
                stage1_source_run_id=stage1_source_run_id,
                attempts=1,
                scan_pass_rate=1.0,
                exact_payload_match=True,
                module_error_rate=0.0,
                generation_ms=1.0,
                validation_ms=1.0,
                total_ms=2.0,
            )
            run.image_path = artifact_store.save_image(run.id, image)
            run_repository.save(run)
            calls.append(
                {
                    "run_id": run.id,
                    "target": target_variant,
                    "reused": raw_candidate_override is not None,
                    "source": stage1_source_run_id,
                    "raw_pixel": raw.getpixel((0, 0)),
                }
            )
            return run

    monkeypatch.setattr(service, "_generation_service", lambda method: FakeGenerationService())
    request = LabCampaignCreate(
        name="paired research outputs",
        payload="https://example.test/t/paired",
        prompts=[{"id": "p1", "text": "green courtyard"}],
        seeds=[51001],
        methods=[
            LabMethod(
                id="raw",
                name="Stage 1",
                output_variant="raw",
                generation={
                    "steps": 40,
                    "guidance_scale": 7.5,
                    "controlnet_scale": 1.35,
                    "strength": 1.0,
                },
            ),
            LabMethod(
                id="srpg",
                name="Stage 2",
                output_variant="srpg",
                generation={
                    "steps": 40,
                    "guidance_scale": 7.5,
                    "controlnet_scale": 1.35,
                    "strength": 1.0,
                },
                tools=LabToolConfig(srpg_enabled=True),
            ),
        ],
    )
    campaign = service.create_campaign(request)
    try:
        for _ in range(100):
            stored = service.lab_repository.get_campaign(campaign["id"])
            if stored["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
    finally:
        service.shutdown()

    assert [call["target"] for call in calls] == ["raw", "srpg"]
    assert calls[0]["reused"] is False
    assert calls[1]["reused"] is True
    assert calls[1]["source"] == calls[0]["run_id"]
    assert calls[1]["raw_pixel"] == calls[0]["raw_pixel"] == (255, 0, 0)
    srpg_run = run_repository.get(calls[1]["run_id"])
    assert srpg_run is not None
    assert srpg_run.quality_metrics["srpg_requested_steps"] == 40
    assert srpg_run.quality_metrics["srpg_effective_steps"] == 40
    assert srpg_run.quality_metrics["srpg_restart_strength"] == 1.0
    assert srpg_run.quality_metrics["srpg_controlnet_scale"] == 1.35
    assert srpg_run.quality_metrics["srpg_qr_weight"] == 500.0
    assert srpg_run.quality_metrics["srpg_perceptual_weight"] == 3.0
    assert srpg_run.quality_metrics["srpg_functional_weight"] == 4.0
    assert srpg_run.quality_metrics["srpg_max_noise_delta_rms"] == 2.0
