import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import prooftag_qr.diffqrcoder_backend as backend_module
from prooftag_qr.config import Settings
from prooftag_qr.diffqrcoder_backend import (
    UpstreamDiffQRCoderBackend,
    _control_target_center_error_rate,
    _install_partial_schedule,
    _offload_unused_pipeline_modules_for_paper_srmpgd,
)
from prooftag_qr.qr import generate_diffqrcoder_qr
from prooftag_qr.quality import image_sha256
from prooftag_qr.schemas import GenerationRequest
from prooftag_qr.srmpgd import SRMPGDStep


def _reference_artwork(size: int = 736) -> Image.Image:
    x = np.linspace(0, 255, size, dtype=np.uint8)
    y = np.linspace(255, 0, size, dtype=np.uint8)
    red = np.broadcast_to(x, (size, size))
    green = np.broadcast_to(y[:, None], (size, size))
    blue = np.full((size, size), 127, dtype=np.uint8)
    return Image.fromarray(np.stack((red, green, blue), axis=2), mode="RGB")


def _srmpgd_step_zero() -> SRMPGDStep:
    return SRMPGDStep(
        iteration=0,
        elapsed_s=0.1,
        scanning_robust_loss=0.2,
        lpips_loss=0.0,
        objective=0.2,
        surrogate_module_error_rate=0.1,
        actual_module_error_rate=0.1,
        passed=1,
        total=2,
        pass_rate=0.5,
        strict_all=False,
        worst_decoder_pass_rate=0.5,
        worst_scenario_pass_rate=0.5,
        gradient_rms=None,
        image_gradient_rms=None,
        gradient_scale=32_768.0,
        next_step_rms=None,
        applied_step_rms=None,
        step_scale=None,
        latent_delta_rms=0.0,
        relative_module_improvement=0.0,
        mean_absolute_change=0.0,
        saturation_mean_increase=0.0,
        high_saturation_ratio_increase=0.0,
        rgb_clipped_channel_ratio_increase=0.0,
        aesthetic_guard_passed=True,
        qr_gain_sufficient=True,
        eligible_for_selection=True,
        base_scanning_loss=0.2,
        blur_scanning_loss=None,
        downscale_scanning_loss=None,
        brightness_scanning_loss=None,
        contrast_scanning_loss=None,
    )


def test_upstream_srmpgd_iteration_zero_receives_and_returns_exact_stage2_raster(
    monkeypatch,
):
    backend = UpstreamDiffQRCoderBackend(
        Settings(srpg_enabled=True, srmpgd_enabled=True, device="cpu")
    )
    stage2_image = _reference_artwork(64)
    stage2_hash = image_sha256(stage2_image)
    step = _srmpgd_step_zero()
    observed = {}

    def fake_run_srmpgd(_pipe, _latent, _blueprint, _config, **kwargs):
        observed["initial_image"] = kwargs["initial_image"]
        return SimpleNamespace(
            image=kwargs["initial_image"].copy(),
            latent=_latent,
            initial_redecoded_image=kwargs["initial_image"].copy(),
            steps=(step,),
            selected_iteration=0,
            stop_reason="initial_module_error_rate_above_limit",
            duration_s=0.1,
            initial_module_error_rate=0.1,
            final_module_error_rate=0.1,
        )

    monkeypatch.setattr(backend_module, "run_srmpgd", fake_run_srmpgd)
    fake_parameter = SimpleNamespace(dtype="float16")

    class FakeVAE:
        def parameters(self):
            return iter((fake_parameter,))

        def to(self, *, dtype):
            fake_parameter.dtype = dtype
            return self

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))
    result = backend._apply_srmpgd(
        SimpleNamespace(
            srpg=object(),
            unet=SimpleNamespace(dtype="float16"),
            vae=FakeVAE(),
        ),
        object(),
        stage2_image,
        SimpleNamespace(),
    )

    assert observed["initial_image"] is stage2_image
    assert image_sha256(result) == stage2_hash
    assert backend.diagnostics()["diffqrcoder_srmpgd_iteration_zero_exact"] == 1.0
    assert backend.debug_metadata()["srmpgd_stage2_image_sha256"] == stage2_hash
    assert backend.debug_metadata()["srmpgd_selected_image_sha256"] == stage2_hash
    assert backend.debug_metadata()["srmpgd_trace"]["initial_redecoded_image_sha256"] == stage2_hash
    assert backend.provenance()["srmpgd_stage2_image_sha256"] == stage2_hash
    assert backend.provenance()["srmpgd_selected_image_sha256"] == stage2_hash


def test_paper_srmpgd_offloads_diffusion_modules_and_skips_upstream_srpg(monkeypatch):
    class Device:
        def __init__(self, kind):
            self.type = kind

    class Parameter:
        def __init__(self, kind="cuda"):
            self.device = Device(kind)
            self.dtype = "float16"

    class Module:
        def __init__(self, name, kind="cuda"):
            self.name = name
            self.parameter = Parameter(kind)
            self.moves = []

        def parameters(self):
            return iter((self.parameter,))

        def to(self, *args, device=None, **_kwargs):
            target = device if device is not None else args[0]
            kind = target.type if hasattr(target, "type") else str(target).split(":")[0]
            self.parameter.device = Device(kind)
            self.moves.append(kind)
            return self

    unet = Module("unet")
    unet.dtype = "float16"
    controlnet = Module("controlnet")
    text_encoder = Module("text_encoder")
    vae = Module("vae")
    lpips = Module("lpips")
    modules = [unet, controlnet, text_encoder, vae, lpips]
    empty_cache_calls = []
    fake_torch = SimpleNamespace(
        float32="float32",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: empty_cache_calls.append(True),
            memory_allocated=lambda: sum(
                1024 for module in modules if module.parameter.device.type == "cuda"
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    pipe = SimpleNamespace(
        unet=unet,
        controlnet=controlnet,
        text_encoder=text_encoder,
        vae=vae,
    )
    backend = UpstreamDiffQRCoderBackend(
        Settings(
            srpg_enabled=True,
            srmpgd_enabled=True,
            srmpgd_protocol="paper_equations",
            device="cuda",
        )
    )
    stage2_image = _reference_artwork(64)
    step = _srmpgd_step_zero()

    def fake_run_srmpgd(current_pipe, latent, _blueprint, _config, **kwargs):
        assert current_pipe.unet.parameter.device.type == "cpu"
        assert current_pipe.controlnet.parameter.device.type == "cpu"
        assert current_pipe.text_encoder.parameter.device.type == "cpu"
        assert current_pipe.vae.parameter.device.type == "cuda"
        current_pipe._prooftag_srmpgd_lpips_vgg = lpips
        return SimpleNamespace(
            image=kwargs["initial_image"].copy(),
            latent=latent,
            initial_redecoded_image=kwargs["initial_image"].copy(),
            steps=(step,),
            selected_iteration=0,
            stop_reason="zero_latent_gradient_at_iteration_0",
            duration_s=0.1,
            initial_module_error_rate=0.1,
            final_module_error_rate=0.1,
        )

    monkeypatch.setattr(backend_module, "run_srmpgd", fake_run_srmpgd)

    result = backend._apply_srmpgd(
        pipe,
        object(),
        stage2_image,
        SimpleNamespace(),
    )

    assert image_sha256(result) == image_sha256(stage2_image)
    assert not hasattr(pipe, "srpg")
    assert unet.moves == ["cpu", "cuda"]
    assert controlnet.moves == ["cpu", "cuda"]
    assert text_encoder.moves == ["cpu", "cuda"]
    assert vae.moves == []
    assert lpips.moves == ["cpu"]
    assert len(empty_cache_calls) == 3
    assert backend.diagnostics()["diffqrcoder_srmpgd_offloaded_module_count"] == 3.0
    assert backend.diagnostics()["diffqrcoder_srmpgd_offloaded_gib"] > 0.0


def test_paper_srmpgd_offload_scope_restores_modules_after_failure(monkeypatch):
    class Device:
        type = "cuda"

    class Parameter:
        device = Device()

    class Module:
        def __init__(self):
            self.parameter = Parameter()
            self.moves = []

        def parameters(self):
            return iter((self.parameter,))

        def to(self, *args, device=None, **_kwargs):
            target = device if device is not None else args[0]
            kind = target.type if hasattr(target, "type") else str(target)
            self.parameter.device = SimpleNamespace(type=kind)
            self.moves.append(kind)
            return self

    unet = Module()
    upstream_srpg = Module()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: None,
            memory_allocated=lambda: 0,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with _offload_unused_pipeline_modules_for_paper_srmpgd(
            SimpleNamespace(unet=unet, srpg=upstream_srpg),
            lpips_net="vgg",
        ):
            raise RuntimeError("synthetic failure")

    assert unet.moves == ["cpu", "cuda"]
    assert upstream_srpg.moves == ["cpu", "cuda"]


def test_upstream_provenance_records_all_pinned_model_revisions():
    backend = UpstreamDiffQRCoderBackend(
        Settings(
            base_model_revision="base-revision",
            base_model_config_revision="config-revision",
            controlnet_model_revision="controlnet-revision",
        )
    )

    provenance = backend.provenance()

    assert provenance["base_model_revision"] == "base-revision"
    assert provenance["base_model_config_id"] == ("stable-diffusion-v1-5/stable-diffusion-v1-5")
    assert provenance["base_model_config_revision"] == "config-revision"
    assert provenance["controlnet_model_revision"] == "controlnet-revision"


def test_upstream_loader_passes_pinned_huggingface_revisions(monkeypatch):
    observed = {}

    class FakeComponent:
        def requires_grad_(self, _enabled):
            return self

        def eval(self):
            return self

    class FakeControlNetModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            observed["controlnet"] = (model_id, kwargs)
            return FakeComponent()

    class FakePipeline:
        @classmethod
        def from_single_file(cls, model_id, **kwargs):
            observed["pipeline"] = (model_id, kwargs)
            return SimpleNamespace(
                scheduler=SimpleNamespace(config={}),
                _callback_tensor_inputs=[],
                unet=FakeComponent(),
                controlnet=kwargs["controlnet"],
                vae=FakeComponent(),
                text_encoder=FakeComponent(),
                set_progress_bar_config=lambda **_: None,
                to=lambda *_: None,
            )

    class FakeScheduler:
        @classmethod
        def from_config(cls, config):
            return SimpleNamespace(config=config)

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        float16="float16",
    )
    monkeypatch.setattr(backend_module, "_patch_upstream_perceptual_gradient", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "diffqrcoder",
        SimpleNamespace(DiffQRCoderPipeline=FakePipeline),
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            ControlNetModel=FakeControlNetModel,
            DDIMScheduler=FakeScheduler,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            hf_hub_download=lambda **kwargs: (
                observed.setdefault("checkpoint_download", kwargs) and "C:/cache/model.safetensors"
            ),
            snapshot_download=lambda **kwargs: (
                observed.setdefault("config_download", kwargs) and "C:/cache/sd15-config"
            ),
        ),
    )
    backend = UpstreamDiffQRCoderBackend(
        Settings(
            device="cuda",
            base_model_id=("https://huggingface.co/example/model/resolve/main/model.safetensors"),
            base_model_revision="base-revision",
            base_model_config_revision="config-revision",
            controlnet_model_id="example/controlnet",
            controlnet_model_revision="controlnet-revision",
        )
    )

    backend._load()

    assert observed["controlnet"][1]["revision"] == "controlnet-revision"
    assert observed["checkpoint_download"]["revision"] == "base-revision"
    assert observed["config_download"]["revision"] == "config-revision"
    assert observed["pipeline"][0] == "C:/cache/model.safetensors"
    assert observed["pipeline"][1]["config"] == "C:/cache/sd15-config"
    assert "revision" not in observed["pipeline"][1]


def test_stage2_state_export_and_import_are_hash_verified():
    torch = pytest.importorskip("torch")
    latent = torch.arange(16, dtype=torch.float16).reshape(1, 1, 4, 4)
    image = Image.new("RGB", (16, 16), "navy")
    source = UpstreamDiffQRCoderBackend(Settings(device="cpu"))
    source._last_stage2_state = {
        "latent": latent,
        "image": image,
        "reference": image,
        "control": object(),
        "diagnostics": {"diffqrcoder_stage2_reused": 0.0},
    }

    state = source.export_stage2_state()
    assert state is not None
    state["source_run_id"] = "run-srpg"
    state["source_method_id"] = "diffqrcoder_srpg"

    target = UpstreamDiffQRCoderBackend(Settings(device="cpu"))
    target.import_stage2_state(state)
    assert target._stage2_override["latent_sha256"] == state["latent_sha256"]
    assert target._stage2_override["source_run_id"] == "run-srpg"

    corrupted = {**state, "latent": state["latent"] + 1}
    with pytest.raises(RuntimeError, match="latent hash mismatch"):
        target.import_stage2_state(corrupted)


def test_reused_qart_stage2_routes_original_qr_to_srmpgd(monkeypatch):
    torch = pytest.importorskip("torch")
    original = generate_diffqrcoder_qr(
        "https://pt.ag/t/original",
        "M",
        version=3,
        mask_pattern=4,
        module_size=20,
    )
    qart_proxy = generate_diffqrcoder_qr(
        "https://pt.ag/t/original#proxy",
        "M",
        version=3,
        mask_pattern=4,
        module_size=20,
    )
    stage2_image = _reference_artwork()
    latent = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    backend = UpstreamDiffQRCoderBackend(
        Settings(srpg_enabled=True, srmpgd_enabled=True, device="cpu")
    )
    backend._stage2_override = {
        "latent": latent,
        "latent_sha256": backend_module._tensor_sha256(latent),
        "source_run_id": "paired-stage2",
        "source_method_id": "diffqrcoder_paper_srpg",
        "image": stage2_image,
        "reference": stage2_image,
        "control": SimpleNamespace(
            image=qart_proxy.image,
            blueprint=qart_proxy,
            match_mode="canonical_url_without_fragment",
        ),
        "diagnostics": {},
    }
    observed = {}
    monkeypatch.setattr(
        backend,
        "_load",
        lambda: SimpleNamespace(unet=SimpleNamespace(dtype=torch.float32)),
    )

    def apply_srmpgd(_pipe, _latent, image, target, **_kwargs):
        observed["target"] = target
        return image

    monkeypatch.setattr(backend, "_apply_srmpgd", apply_srmpgd)
    monkeypatch.setattr(backend, "_record_divergence_guard", lambda *_args: None)

    backend._run_stage2(
        stage2_image,
        original,
        GenerationRequest(
            payload="https://pt.ag/t/original",
            prompt="paired target routing",
        ),
        51_001,
    )

    assert observed["target"] is original
    assert observed["target"] is not qart_proxy


def test_stage2_target_is_the_exact_binary_qr_not_a_visual_proxy():
    blueprint = generate_diffqrcoder_qr(
        "https://pt.ag/t/1",
        "M",
        version=3,
        mask_pattern=4,
        module_size=20,
    )
    reference = _reference_artwork()
    backend = UpstreamDiffQRCoderBackend(Settings(diffqrcoder_stage2_target_mode="binary_exact"))

    target = backend._stage2_target(
        reference,
        blueprint,
        "https://pt.ag/t/1",
    )

    assert target.image.size == blueprint.image.size
    assert target.match_mode == "exact"
    assert np.array_equal(
        np.asarray(target.image),
        np.asarray(blueprint.image.convert("RGB")),
    )
    assert not np.array_equal(np.asarray(target.image), np.asarray(reference))
    assert (
        _control_target_center_error_rate(
            target.image,
            blueprint,
            padding_px=78,
            module_size=20,
        )
        == 0.0
    )


def test_partial_stage2_schedule_avoids_custom_timesteps_and_keeps_stride():
    class Scheduler:
        num_inference_steps = None
        timesteps = []

        def set_timesteps(self, num_inference_steps, *args, **kwargs):
            self.num_inference_steps = num_inference_steps
            self.timesteps = list(range(num_inference_steps))

    class Pipeline:
        scheduler = Scheduler()

    pipe = Pipeline()
    original = pipe.scheduler.set_timesteps

    with _install_partial_schedule(
        pipe,
        base_steps=40,
        effective_steps=20,
    ):
        pipe.scheduler.set_timesteps(20, device="cuda")
        assert pipe.scheduler.num_inference_steps == 40
        assert pipe.scheduler.timesteps == list(range(20, 40))

    assert pipe.scheduler.set_timesteps == original


def test_color_guard_distinguishes_warning_from_destructive_divergence(monkeypatch):
    change = {
        "changed_pixel_ratio": 0.99,
        "mean_absolute_change": 0.36,
        "clipped_pixel_ratio_increase": 0.06,
        "rgb_clipped_channel_ratio_increase": 0.03,
        "saturation_mean_increase": 0.09,
        "high_saturation_ratio_increase": 0.06,
    }
    quality = {
        "clipped_pixel_ratio": 0.06,
        "rgb_clipped_channel_ratio": 0.03,
        "saturation_mean": 0.50,
        "saturation_p95": 0.95,
    }
    monkeypatch.setattr(backend_module, "image_change_metrics", lambda *_: change)
    monkeypatch.setattr(backend_module, "image_quality_metrics", lambda *_: quality)
    backend = UpstreamDiffQRCoderBackend(Settings())
    image = Image.new("RGB", (32, 32), "red")

    backend._record_divergence_guard(image, image)

    assert backend.diagnostics()["diffqrcoder_guard_warning"] == 1.0
    assert backend.diagnostics()["diffqrcoder_guard_diverged"] == 0.0
    assert backend.candidate_guard_ok("srpg") is True

    change["saturation_mean_increase"] = 0.25
    backend._record_divergence_guard(image, image)

    assert backend.diagnostics()["diffqrcoder_guard_diverged"] == 1.0
    assert backend.candidate_guard_ok("srpg") is False


def test_hard_color_guard_must_be_at_least_the_warning_threshold():
    with pytest.raises(ValueError, match="hard saturation mean guard"):
        Settings(
            diffqrcoder_guard_max_saturation_mean_increase=0.25,
            diffqrcoder_guard_hard_max_saturation_mean_increase=0.20,
        )


@pytest.mark.parametrize(
    ("selected_iteration", "expected_variant"),
    [(0, "srpg"), (2, "srmpgd")],
)
def test_srmpgd_iteration_zero_is_reported_as_srpg(selected_iteration, expected_variant):
    backend = UpstreamDiffQRCoderBackend(Settings(srpg_enabled=True, srmpgd_enabled=True))
    image = Image.new("RGB", (32, 32), "navy")

    def fake_stage2(*_args, **_kwargs):
        backend._srmpgd_selected_iteration = selected_iteration
        return image.copy()

    backend._run_stage2 = fake_stage2
    variants = list(
        backend.variants(
            image,
            SimpleNamespace(),
            request=SimpleNamespace(),
            seed=41,
        )
    )

    assert [name for name, _ in variants] == ["raw", expected_variant]
