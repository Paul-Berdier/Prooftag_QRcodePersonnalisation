from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from prooftag_qr.qr import generate_qr
from prooftag_qr.srpg import SRPGConfig, _predict_original_sample


def test_predict_original_sample_matches_ddim_epsilon_formula():
    torch = pytest.importorskip("torch")
    scheduler = SimpleNamespace(
        alphas_cumprod=torch.tensor([0.25, 0.81]),
        config=SimpleNamespace(
            prediction_type="epsilon",
            thresholding=False,
            clip_sample=False,
        ),
    )
    sample = torch.tensor([[[[0.8]]]])
    noise = torch.tensor([[[[0.2]]]])

    predicted = _predict_original_sample(scheduler, noise, torch.tensor(1), sample)

    expected = (sample - (1 - 0.81) ** 0.5 * noise) / 0.81**0.5
    assert torch.allclose(predicted, expected)


def test_srpg_loop_runs_guidance_inside_each_fake_ddim_step(monkeypatch):
    torch = pytest.importorskip("torch")
    from prooftag_qr import srpg

    class DDIMScheduler:
        order = 1

        def __init__(self):
            self.config = SimpleNamespace(
                prediction_type="epsilon",
                thresholding=False,
                clip_sample=False,
            )
            self.alphas_cumprod = torch.linspace(0.05, 0.95, 1000)
            self.timesteps = torch.tensor([900, 450])

        def set_timesteps(self, steps, device):
            self.timesteps = torch.linspace(900, 450, steps, device=device).long()

        def scale_model_input(self, sample, timestep):
            return sample

        def step(self, model_output, timestep, sample, **kwargs):
            return (sample - 0.1 * model_output,)

    class FakeComponent:
        dtype = torch.float32

        def requires_grad_(self, value):
            return self

        def eval(self):
            return self

    class FakeControlNet(FakeComponent):
        def __call__(self, sample, timestep, **kwargs):
            return ([torch.zeros_like(sample)], torch.zeros_like(sample))

    class FakeUNet(FakeComponent):
        def __call__(self, sample, timestep, **kwargs):
            return (sample * 0.25,)

    class FakeVAE(FakeComponent):
        config = SimpleNamespace(scaling_factor=1.0)

        def decode(self, latent, **kwargs):
            return (latent,)

    class FakeImageProcessor:
        def preprocess(self, image, height, width):
            array = np.asarray(image.resize((width, height)), dtype=np.float32) / 127.5 - 1
            return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

        def postprocess(self, image, **kwargs):
            array = (image[0].detach().clamp(-1, 1) / 2 + 0.5).permute(1, 2, 0).cpu().numpy()
            return [Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB")]

    class FakePipeline:
        _execution_device = torch.device("cpu")

        def __init__(self):
            self.scheduler = DDIMScheduler()
            self.vae = FakeVAE()
            self.unet = FakeUNet()
            self.controlnet = FakeControlNet()
            self.text_encoder = FakeComponent()
            self.image_processor = FakeImageProcessor()

        def encode_prompt(self, *args, **kwargs):
            embedding = torch.zeros((1, 1, 1))
            return embedding, embedding.clone()

        def prepare_control_image(self, **kwargs):
            batch = 2 if kwargs["do_classifier_free_guidance"] else 1
            return torch.zeros((batch, 3, 128, 128))

        def get_timesteps(self, steps, strength, device):
            return self.scheduler.timesteps, len(self.scheduler.timesteps)

        def prepare_latents(self, image, *args, **kwargs):
            return image.clone()

        def prepare_extra_step_kwargs(self, generator, eta):
            return {}

    class FakeLPIPS(torch.nn.Module):
        def forward(self, image, reference):
            return (image - reference).square().mean().reshape(1, 1, 1, 1)

    monkeypatch.setattr(srpg, "_load_lpips", lambda pipeline, device: FakeLPIPS())
    blueprint = generate_qr("https://example.prooftag.test/t/srpg-fake-loop", "H", size=128)

    callback_steps = []
    result = srpg.run_srpg_controlnet_img2img(
        FakePipeline(),
        blueprint.image,
        blueprint,
        prompt="test",
        negative_prompt=None,
        guidance_scale=7.5,
        generator=torch.Generator().manual_seed(7),
        config=SRPGConfig(
            steps=2,
            qr_weight=1.0,
            perceptual_weight=0.1,
            robust_blur_weight=0.2,
            robust_blur_kernel=5,
            robust_downscale_weight=0.2,
            robust_downscale_factor=0.60,
            robust_brightness_weight=0.2,
            robust_brightness_low=0.70,
            robust_brightness_high=1.30,
            robust_contrast_weight=0.2,
            robust_contrast_factor=0.60,
            min_relative_module_improvement=0.0,
            save_step_previews=True,
            preview_interval=1,
        ),
        preview_callback=lambda preview, step: callback_steps.append(
            (preview.index, step.index)
        ),
    )

    assert len(result.steps) == 2
    assert [preview.index for preview in result.previews] == [0, 1]
    assert all(preview.predicted_clean_image.size == (128, 128) for preview in result.previews)
    assert all(preview.active_module_map.size == (128, 128) for preview in result.previews)
    assert callback_steps == [(0, 0), (1, 1)]
    assert all(step.guidance_applied for step in result.steps)
    assert all(np.isfinite(step.gradient_rms) for step in result.steps)
    assert result.image.size == (128, 128)
    assert np.all(np.asarray(result.image)[0, 0] == 255)
