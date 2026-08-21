import hashlib
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from prooftag_qr import quality_scoring as quality_scoring_module
from prooftag_qr.advisor import (
    AdvisorPrediction,
    ContextualParameterAdvisor,
    advisor_rank_key,
)
from prooftag_qr.experiments import (
    SRPGTrial,
    aggregate_confirmation,
    image_context_features,
    sample_e007_trial,
    select_delivery_candidate,
)
from prooftag_qr.optimization import E007Experiment, factorial_contexts, query_gpu_processes
from prooftag_qr.qr import generate_qr
from prooftag_qr.quality_scoring import (
    DEFAULT_AESTHETIC_WEIGHTS_SHA256,
    DEFAULT_AESTHETIC_WEIGHTS_URL,
    DEFAULT_CLIP_MODEL_REVISION,
    DEFAULT_HPS_CHECKPOINT_REVISION,
    DEFAULT_HPS_SOURCE_REVISION,
    CLIPQualityScorer,
    QualityProvenanceError,
    clip_score_from_similarity,
    project_embedding,
)
from prooftag_qr.srpg import SRPGConfig, _qr_improvement_is_acceptable, _validate_config


class FakeSampler:
    def __init__(self):
        self.names = set()

    def suggest_float(self, name, low, high, **kwargs):
        self.names.add(name)
        return (low + high) / 2

    def suggest_categorical(self, name, choices):
        self.names.add(name)
        return choices[len(choices) // 2]

    def suggest_int(self, name, low, high):
        self.names.add(name)
        return (low + high) // 2


def test_e007_samples_all_generation_and_robustness_dimensions():
    sampler = FakeSampler()
    trial = sample_e007_trial(sampler, name="sample")

    assert trial.base_steps in {8, 12, 16, 20, 24}
    assert trial.steps in {32, 40, 60, 80, 100, 120}
    assert 0.35 <= trial.dark_threshold <= trial.light_threshold <= 0.80
    assert trial.robust_blur_weight > 0
    assert trial.robust_blur_kernel in {3, 5, 7}
    assert trial.robust_downscale_weight > 0
    assert 0.50 <= trial.robust_downscale_factor <= 0.90
    assert trial.robust_brightness_weight > 0
    assert 0.60 <= trial.robust_brightness_low <= 0.90
    assert 1.10 <= trial.robust_brightness_high <= 1.40
    assert trial.robust_contrast_weight > 0
    assert 0.50 <= trial.robust_contrast_factor <= 0.90
    assert 0 <= trial.target_module_error_rate <= 0.08
    assert trial.negative_prompt
    config = trial.to_srpg_config()
    assert config.center_fraction == trial.center_fraction
    assert config.eta == trial.eta
    assert len(sampler.names) == 28


def test_factorial_design_changes_one_axis_at_a_time():
    contexts = factorial_contexts()
    prompt_contexts = [context for context in contexts if context.axis == "prompt"]
    seed_contexts = [context for context in contexts if context.axis == "seed"]
    payload_contexts = [context for context in contexts if context.axis == "payload"]

    assert len({context.payload for context in prompt_contexts}) == 1
    assert len({context.seed for context in prompt_contexts}) == 1
    assert len({context.prompt for context in seed_contexts}) == 1
    assert len({context.payload for context in seed_contexts}) == 1
    assert len({context.prompt for context in payload_contexts}) == 1
    assert len({context.seed for context in payload_contexts}) == 1


def test_reference_qr_has_safe_context_features():
    blueprint = generate_qr("https://example.prooftag.test/t/context", "H", size=128)
    features = image_context_features(blueprint.image, blueprint)

    assert features["raw_module_error_rate"] == 0
    assert features["raw_functional_error_rate"] == 0
    assert features["raw_module_margin_p10"] > 0
    assert features["matrix_modules"] == blueprint.matrix.shape[0]


def test_clip_score_and_projection_are_deterministic():
    embedding = np.linspace(-1, 1, 32, dtype=np.float32)

    assert clip_score_from_similarity(0.3) == pytest.approx(0.75)
    assert clip_score_from_similarity(-0.3) == 0
    assert project_embedding(embedding) == project_embedding(embedding)
    assert len(project_embedding(embedding)) == 16


def test_hpsv21_score_is_optional_and_uses_the_official_version(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    calls = []
    checkpoint = tmp_path / "hps-v2.1.pt"
    checkpoint.write_bytes(b"pinned HPS checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    fake_hps = SimpleNamespace(__path__=[])
    fake_img_score = SimpleNamespace(
        device="cuda",
        score=lambda image, prompt, cp, hps_version: (
            calls.append((image.size, prompt, cp, hps_version)) or [0.314]
        ),
    )
    monkeypatch.setitem(sys.modules, "hpsv2", fake_hps)
    monkeypatch.setitem(sys.modules, "hpsv2.img_score", fake_img_score)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=lambda **kwargs: (
                calls.append(("download", kwargs)) or str(checkpoint)
            )
        ),
    )
    monkeypatch.setattr(
        quality_scoring_module,
        "_installed_distribution_source",
        lambda package: ("1.2.0", DEFAULT_HPS_SOURCE_REVISION, None),
    )
    scorer = CLIPQualityScorer(
        tmp_path,
        hps_enabled=True,
        hps_checkpoint_sha256=checkpoint_sha256,
    )

    score = scorer._hps_score(Image.new("RGB", (16, 16)), "a blue vase")

    assert score == pytest.approx(0.314)
    assert calls[0] == (
        "download",
        {
            "repo_id": "xswu/HPSv2",
            "filename": "HPS_v2.1_compressed.pt",
            "revision": DEFAULT_HPS_CHECKPOINT_REVISION,
            "cache_dir": tmp_path / "huggingface" / "hub",
        },
    )
    assert calls[1] == ((16, 16), "a blue vase", str(checkpoint), "v2.1")
    assert fake_img_score.device == "cpu"
    assert scorer.provenance()["hpsv2_1"]["checkpoint_verified"] is True


def test_hps_provenance_mismatch_fails_before_inference(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "hpsv2", SimpleNamespace(__path__=[]))
    monkeypatch.setattr(
        quality_scoring_module,
        "_installed_distribution_source",
        lambda package: ("1.2.0", "0" * 40, None),
    )
    scorer = CLIPQualityScorer(tmp_path, hps_enabled=True)

    with pytest.raises(QualityProvenanceError, match="source revision mismatch"):
        scorer._hps_score(Image.new("RGB", (16, 16)), "a blue vase")


def test_quality_defaults_use_immutable_revisions_and_hashes(tmp_path):
    scorer = CLIPQualityScorer(tmp_path)

    assert scorer.model_revision == DEFAULT_CLIP_MODEL_REVISION
    assert DEFAULT_HPS_SOURCE_REVISION == "866735ecaae999fa714bd9edfa05aa2672669ee3"
    assert "/refs/heads/" not in DEFAULT_AESTHETIC_WEIGHTS_URL
    assert "/main/" not in DEFAULT_AESTHETIC_WEIGHTS_URL
    assert len(DEFAULT_AESTHETIC_WEIGHTS_SHA256) == 64


def test_clip_and_aesthetic_loader_verify_effective_artifacts(tmp_path, monkeypatch):
    calls = []
    expected_weights = b"verified aesthetic weights"
    expected_sha256 = hashlib.sha256(expected_weights).hexdigest()

    class FakeModel:
        config = SimpleNamespace(
            projection_dim=512,
            _commit_hash=DEFAULT_CLIP_MODEL_REVISION,
        )

        def requires_grad_(self, value):
            return self

        def eval(self):
            return self

        def to(self, device):
            return self

    class FakeLinear(FakeModel):
        def load_state_dict(self, state):
            calls.append(("state", state))

    class FakeCLIPModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            return FakeModel()

    class FakeCLIPProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("processor", model_id, kwargs))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(CLIPModel=FakeCLIPModel, CLIPProcessor=FakeCLIPProcessor),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            nn=SimpleNamespace(Linear=lambda dimensions, output: FakeLinear()),
            load=lambda *args, **kwargs: {"weight": "verified"},
        ),
    )
    monkeypatch.setattr(
        quality_scoring_module,
        "urlretrieve",
        lambda url, path: (calls.append(("download", url)), path.write_bytes(expected_weights)),
    )
    weight_path = tmp_path / "clip-quality" / "sa_0_4_vit_b_32_linear.pth"
    weight_path.parent.mkdir(parents=True)
    weight_path.write_bytes(b"corrupt")
    scorer = CLIPQualityScorer(
        tmp_path,
        aesthetic_weights_sha256=expected_sha256,
    )

    scorer._load()

    assert calls[0][2]["revision"] == DEFAULT_CLIP_MODEL_REVISION
    assert calls[1][2]["revision"] == DEFAULT_CLIP_MODEL_REVISION
    assert ("download", DEFAULT_AESTHETIC_WEIGHTS_URL) in calls
    assert weight_path.read_bytes() == expected_weights
    provenance = scorer.provenance()
    assert provenance["clip"]["revision_verified"] is True
    assert provenance["clip_aesthetic"]["sha256_verified"] is True


def test_clip_effective_revision_mismatch_fails_closed(tmp_path, monkeypatch):
    class FakeModel:
        config = SimpleNamespace(projection_dim=512, _commit_hash="0" * 40)

    loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeModel())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(CLIPModel=loader, CLIPProcessor=loader),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())

    with pytest.raises(QualityProvenanceError, match="CLIP model revision mismatch"):
        CLIPQualityScorer(tmp_path)._load()


def test_delivery_never_trades_scan_for_aesthetic():
    rejected_pretty = {
        "status": "ok",
        "strict_all": False,
        "clip_aesthetic": 9.0,
        "clip_score": 1.0,
    }
    strict_lower_quality = {
        "status": "ok",
        "strict_all": True,
        "clip_aesthetic": 5.0,
        "clip_score": 0.5,
        "mean_absolute_change": 0.1,
        "duration_seconds": 10,
    }

    selected = select_delivery_candidate([rejected_pretty, strict_lower_quality])
    assert selected is strict_lower_quality
    assert select_delivery_candidate([rejected_pretty]) is None


def test_stage1_cache_ignores_stage2_parameters_and_trial_name():
    first = SRPGTrial(name="first", source="test", steps=40, qr_weight=300.0)
    second = SRPGTrial(name="second", source="advisor", steps=120, qr_weight=1800.0)
    different_stage1 = SRPGTrial(
        name="third", source="advisor", base_steps=24, steps=120, qr_weight=1800.0
    )

    assert E007Experiment._stage1_hash(first) == E007Experiment._stage1_hash(second)
    assert E007Experiment._stage1_hash(first) != E007Experiment._stage1_hash(different_stage1)


def test_incomplete_confirmation_cannot_be_strict():
    rows = [
        {
            "trial": "partial",
            "status": "ok",
            "strict_all": True,
            "pass_rate": 1.0,
            "module_error_rate": 0.0,
            "mean_absolute_change": 0.1,
            "duration_seconds": 10,
        }
    ]

    aggregate = aggregate_confirmation(rows, expected_cases=4)[0]

    assert aggregate["complete"] is False
    assert aggregate["all_strict"] is False


def test_advisor_ranks_scan_lower_bound_before_quality():
    trial = SRPGTrial(name="trial", source="test")
    robust = AdvisorPrediction(trial, 0.95, 0.02, 5.0, 0.5)
    pretty = AdvisorPrediction(trial, 0.80, 0.01, 9.0, 1.0)

    assert advisor_rank_key(robust) < advisor_rank_key(pretty)


def test_contextual_advisor_fits_predicts_and_persists(tmp_path):
    rows = []
    for context_index in range(4):
        for trial_index in range(6):
            configuration = SRPGTrial(
                name=f"trial-{context_index}-{trial_index}",
                source="test",
                qr_weight=300.0 + 200.0 * trial_index,
                perceptual_weight=float(context_index),
            )
            scan_rate = min(1.0, 0.30 + 0.12 * trial_index + 0.02 * context_index)
            rows.append(
                {
                    "status": "ok",
                    "context_id": f"context-{context_index}",
                    "context_features": {
                        "qr_version": 5.0 + context_index,
                        "raw_entropy": 0.5 + context_index / 10,
                    },
                    "parameters": asdict(configuration),
                    "pass_rate": scan_rate,
                    "clip_aesthetic": 5.0 + 0.1 * context_index - 0.05 * trial_index,
                    "clip_score": 0.5 + 0.02 * trial_index,
                }
            )

    advisor = ContextualParameterAdvisor(trees=32)
    report = advisor.fit(rows)
    candidates = [
        SRPGTrial(name="weak", source="test", qr_weight=300.0),
        SRPGTrial(name="strong", source="test", qr_weight=1300.0),
    ]
    predictions = advisor.recommend(
        {"qr_version": 7.0, "raw_entropy": 0.7}, candidates, limit=2
    )
    model_path = tmp_path / "advisor.joblib"
    advisor.save(model_path)

    assert report["rows"] == 24
    assert report["contexts"] == 4
    assert len(predictions) == 2
    assert all(0.0 <= item.predicted_pass_rate <= 1.0 for item in predictions)
    assert model_path.exists()


def test_gpu_process_query_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(
        "prooftag_qr.optimization.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="123, 11450, python\n456, 8200, vllm\n"
        ),
    )

    processes = query_gpu_processes()

    assert processes[0]["pid"] == 123
    assert processes[1]["used_memory_mib"] == 8200


def test_srpg_rejects_invalid_robustness_and_eta():
    with pytest.raises(ValueError, match="robustness"):
        _validate_config(SRPGConfig(robust_blur_weight=-1))
    with pytest.raises(ValueError, match="eta"):
        _validate_config(SRPGConfig(eta=1.1))
    with pytest.raises(ValueError, match="blur_kernel"):
        _validate_config(SRPGConfig(robust_blur_kernel=4))
    with pytest.raises(ValueError, match="downscale_factor"):
        _validate_config(SRPGConfig(robust_downscale_factor=0))
    with pytest.raises(ValueError, match="at least one effective step"):
        _validate_config(SRPGConfig(steps=40, strength=0.01))


def test_srpg_accepts_preserved_zero_error_without_allowing_regression():
    assert _qr_improvement_is_acceptable(
        0.0,
        0.0,
        target_error=0.0,
        min_relative_improvement=0.1,
    )
    assert not _qr_improvement_is_acceptable(
        0.0,
        0.001,
        target_error=0.0,
        min_relative_improvement=0.1,
    )
    assert _qr_improvement_is_acceptable(
        0.1,
        0.08,
        target_error=0.0,
        min_relative_improvement=0.1,
    )
