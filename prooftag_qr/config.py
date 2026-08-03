from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROOFTAG_QR_", env_file=".env", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"
    lab_clip_scoring_enabled: bool = False
    data_dir: Path = Path("data")
    model_cache_dir: Path = Path("models")
    default_backend: Literal["qr", "controlnet"] = "qr"
    base_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    controlnet_model_id: str = "DionTimmer/controlnet_qrcode-control_v1p_sd15"
    controlnet_model_subfolder: str = ""
    controlnet_conditioning_profile: Literal[
        "binary", "gray_quiet_zone", "nacholmo_extremes_25"
    ] = "binary"
    controlnet_pipeline_mode: Literal["text2img", "img2img"] = "img2img"
    diffqrcoder_upstream_enabled: bool = False
    diffqrcoder_revision: str = "e24ea73ee2e13c7e6e87cb422e8b11784e70ae00"
    diffqrcoder_qr_version: int = Field(default=3, ge=1, le=40)
    diffqrcoder_qr_mask_pattern: int = Field(default=4, ge=0, le=7)
    diffqrcoder_qr_module_size: int = Field(default=20, ge=4, le=64)
    diffqrcoder_qr_padding_px: int = Field(default=78, ge=0, le=256)
    diffqrcoder_control_guidance_start: float = Field(default=0.0, ge=0.0, le=1.0)
    diffqrcoder_control_guidance_end: float = Field(default=1.0, ge=0.0, le=1.0)
    diffqrcoder_stage2_initialization: Literal[
        "paper_stage1_noise", "public_random"
    ] = "paper_stage1_noise"
    diffqrcoder_stage2_strength: float = Field(default=1.0, gt=0.0, le=1.0)
    diffqrcoder_stage2_target_mode: Literal[
        "binary_exact", "qart_url_fragment"
    ] = "binary_exact"
    diffqrcoder_qart_executable: str = "/usr/local/bin/qart"
    diffqrcoder_qart_thresholds: tuple[int, ...] = (96, 112, 128, 144, 160)
    diffqrcoder_guard_max_changed_pixel_ratio: float = Field(
        default=0.995, gt=0.0, le=1.0
    )
    diffqrcoder_guard_max_mean_absolute_change: float = Field(
        default=0.35, gt=0.0, le=1.0
    )
    diffqrcoder_guard_max_clipped_pixel_ratio_increase: float = Field(
        default=0.05, ge=0.0, le=1.0
    )
    diffqrcoder_guard_max_rgb_clipped_channel_ratio_increase: float = Field(
        default=0.02, ge=0.0, le=1.0
    )
    diffqrcoder_guard_max_saturation_mean_increase: float = Field(
        default=0.08, ge=0.0, le=1.0
    )
    diffqrcoder_guard_max_high_saturation_ratio_increase: float = Field(
        default=0.05, ge=0.0, le=1.0
    )
    diffqrcoder_guard_hard_max_mean_absolute_change: float = Field(
        default=0.40, gt=0.0, le=1.0
    )
    diffqrcoder_guard_hard_max_clipped_pixel_ratio_increase: float = Field(
        default=0.20, ge=0.0, le=1.0
    )
    diffqrcoder_guard_hard_max_rgb_clipped_channel_ratio_increase: float = Field(
        default=0.25, ge=0.0, le=1.0
    )
    diffqrcoder_guard_hard_max_saturation_mean_increase: float = Field(
        default=0.20, ge=0.0, le=1.0
    )
    diffqrcoder_guard_hard_max_high_saturation_ratio_increase: float = Field(
        default=0.30, ge=0.0, le=1.0
    )
    device: str = "cuda"
    validation_min_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    max_attempts: int = Field(default=3, ge=1, le=20)
    regenerate_before_global_repair: bool = True
    guided_rediffusion_enabled: bool = False
    guided_rediffusion_steps: int = Field(default=8, ge=1, le=40)
    guided_rediffusion_strength: float = Field(default=0.30, ge=0.05, le=1.0)
    guided_rediffusion_controlnet_scale: float = Field(default=1.75, gt=0.0, le=3.0)
    guided_rediffusion_guide_center_scale: float = Field(default=0.45, ge=0.0, le=1.0)
    guided_rediffusion_guide_confidence_margin: float = Field(default=16.0, ge=0.0, lt=128.0)
    guided_rediffusion_mask_dilation_px: int = Field(default=4, ge=0, le=64)
    guided_rediffusion_mask_feather_px: int = Field(default=4, ge=0, le=64)
    guided_rediffusion_max_mean_absolute_change: float = Field(default=0.12, gt=0.0, le=1.0)
    guided_rediffusion_min_relative_module_improvement: float = Field(default=0.01, ge=0.0, le=1.0)
    guided_rediffusion_seed_offset: int = Field(default=1_000_003, ge=0, le=2**32 - 1)
    srpg_enabled: bool = False
    srpg_steps: int = Field(default=40, ge=1, le=100)
    srpg_strength: float = Field(default=1.0, gt=0.0, le=1.0)
    srpg_controlnet_scale: float = Field(default=1.35, gt=0.0, le=3.0)
    srpg_qr_weight: float = Field(default=500.0, gt=0.0, le=5000.0)
    srpg_perceptual_weight: float = Field(default=3.0, ge=0.0, le=100.0)
    srpg_functional_weight: float = Field(default=4.0, ge=1.0, le=100.0)
    srpg_center_fraction: float = Field(default=1 / 3, gt=0.0, le=1.0)
    srpg_dark_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    srpg_light_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    srpg_robust_blur_weight: float = Field(default=0.0, ge=0.0, le=10.0)
    srpg_robust_blur_kernel: int = Field(default=3, ge=1, le=15)
    srpg_robust_downscale_weight: float = Field(default=0.0, ge=0.0, le=10.0)
    srpg_robust_downscale_factor: float = Field(default=0.75, gt=0.0, le=1.0)
    srpg_robust_brightness_weight: float = Field(default=0.0, ge=0.0, le=10.0)
    srpg_robust_brightness_low: float = Field(default=0.75, gt=0.0, le=1.0)
    srpg_robust_brightness_high: float = Field(default=1.25, ge=1.0, le=2.0)
    srpg_robust_contrast_weight: float = Field(default=0.0, ge=0.0, le=10.0)
    srpg_robust_contrast_factor: float = Field(default=0.70, gt=0.0, le=1.0)
    srpg_target_module_error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    srpg_max_noise_delta_rms: float = Field(default=2.0, gt=0.0, le=100.0)
    srpg_eta: float = Field(default=0.0, ge=0.0, le=1.0)
    srpg_max_mean_absolute_change: float = Field(default=0.20, gt=0.0, le=1.0)
    srpg_min_relative_module_improvement: float = Field(default=0.10, ge=0.0, le=1.0)
    srpg_seed_offset: int = Field(default=2_000_003, ge=0, le=2**32 - 1)
    srpg_save_step_previews: bool = False
    srpg_preview_interval: int = Field(default=5, ge=1, le=100)
    srpg_latent_fusion_enabled: bool = False
    srpg_latent_fusion_channel: int = Field(default=1, ge=0, le=3)
    srpg_latent_fusion_alpha: float = Field(default=0.15, ge=0.0, le=1.0)
    srpg_latent_fusion_start: float = Field(default=0.0, ge=0.0, le=1.0)
    srpg_latent_fusion_end: float = Field(default=1.0, ge=0.0, le=1.0)
    srpg_quiet_zone_mode: Literal["none", "white", "adaptive_light"] = (
        "adaptive_light"
    )
    srpg_quiet_zone_minimum_luminance: float = Field(
        default=0.90, gt=0.0, le=1.0
    )
    srpg_functional_pattern_tone_factor: float = Field(
        default=0.0, ge=0.0, le=1.0
    )
    srmpgd_enabled: bool = False
    srmpgd_max_iterations: int = Field(default=20, ge=1, le=100)
    srmpgd_step_size: float = Field(default=1000.0, gt=0.0, le=100_000.0)
    srmpgd_lpips_weight: float = Field(default=0.01, ge=0.0, le=100.0)
    srmpgd_lpips_net: Literal["alex", "vgg", "squeeze"] = "vgg"
    srmpgd_crop_padding_px: int = Field(default=-1, ge=-1, le=256)
    srmpgd_dark_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    srmpgd_light_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    srmpgd_center_fraction: float = Field(default=1 / 3, gt=0.0, le=1.0)
    srmpgd_max_initial_module_error_rate: float = Field(
        default=0.10, ge=0.0, le=1.0
    )
    latent_refinement_enabled: bool = False
    latent_refinement_iterations: int = Field(default=8, ge=1, le=100)
    latent_refinement_learning_rate: float = Field(default=0.02, gt=0.0, le=10.0)
    latent_refinement_qr_weight: float = Field(default=1.0, gt=0.0, le=100.0)
    latent_refinement_preservation_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    latent_refinement_functional_weight: float = Field(default=4.0, ge=1.0, le=100.0)
    latent_refinement_target_module_error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latent_refinement_max_latent_delta: float = Field(default=0.10, gt=0.0, le=10.0)
    latent_refinement_max_mean_absolute_change: float = Field(default=0.08, gt=0.0, le=1.0)
    latent_refinement_min_relative_module_improvement: float = Field(default=0.01, ge=0.0, le=1.0)
    save_debug_artifacts: bool = False
    artifact_store: Literal["local", "s3"] = "local"
    database_backend: Literal["sqlite", "postgresql"] = "sqlite"
    database_host: str = "prooftag-qr-postgres"
    database_port: int = 5432
    database_name: str = "prooftag_qr"
    database_user: str = "prooftag_qr"
    database_password: str = ""
    database_url_override: str = ""
    s3_endpoint: str = "http://minio.data-core.svc.cluster.local:9000"
    s3_bucket: str = "prooftag-qr"
    s3_access_key: str = ""
    s3_secret_key: str = ""

    @model_validator(mode="after")
    def validate_guided_rediffusion_schedule(self) -> "Settings":
        scheduler_steps = ceil(self.guided_rediffusion_steps / self.guided_rediffusion_strength)
        if scheduler_steps > 100:
            raise ValueError(
                "guided rediffusion effective steps / strength cannot schedule over 100 steps"
            )
        if self.srpg_enabled and self.guided_rediffusion_enabled:
            raise ValueError("SRPG and legacy guided rediffusion cannot be enabled together")
        if (
            self.srpg_enabled
            and not self.diffqrcoder_upstream_enabled
            and self.controlnet_pipeline_mode != "img2img"
        ):
            raise ValueError("SRPG requires the img2img ControlNet pipeline")
        if self.srpg_enabled and self.srpg_effective_steps < 1:
            raise ValueError(
                "SRPG steps multiplied by strength must schedule at least one effective step"
            )
        if self.srmpgd_enabled and not self.srpg_enabled:
            raise ValueError("paper SR-MPGD requires Stage 2 SRPG and its exact clean latent")
        if self.srmpgd_dark_threshold > self.srmpgd_light_threshold:
            raise ValueError("SR-MPGD dark threshold cannot exceed light threshold")
        if self.srpg_dark_threshold > self.srpg_light_threshold:
            raise ValueError("SRPG dark threshold cannot exceed light threshold")
        if self.srpg_robust_blur_kernel % 2 == 0:
            raise ValueError("SRPG robust blur kernel must be odd")
        if self.srpg_latent_fusion_start > self.srpg_latent_fusion_end:
            raise ValueError("SRPG latent fusion start cannot exceed end")
        if self.diffqrcoder_control_guidance_start > self.diffqrcoder_control_guidance_end:
            raise ValueError("DiffQRCoder control guidance start cannot exceed end")
        guard_pairs = (
            (
                "mean absolute change",
                self.diffqrcoder_guard_max_mean_absolute_change,
                self.diffqrcoder_guard_hard_max_mean_absolute_change,
            ),
            (
                "clipped pixels",
                self.diffqrcoder_guard_max_clipped_pixel_ratio_increase,
                self.diffqrcoder_guard_hard_max_clipped_pixel_ratio_increase,
            ),
            (
                "RGB clipped channels",
                self.diffqrcoder_guard_max_rgb_clipped_channel_ratio_increase,
                self.diffqrcoder_guard_hard_max_rgb_clipped_channel_ratio_increase,
            ),
            (
                "saturation mean",
                self.diffqrcoder_guard_max_saturation_mean_increase,
                self.diffqrcoder_guard_hard_max_saturation_mean_increase,
            ),
            (
                "high saturation",
                self.diffqrcoder_guard_max_high_saturation_ratio_increase,
                self.diffqrcoder_guard_hard_max_high_saturation_ratio_increase,
            ),
        )
        for name, warning_threshold, hard_threshold in guard_pairs:
            if hard_threshold < warning_threshold:
                raise ValueError(
                    f"DiffQRCoder hard {name} guard cannot be lower than its warning"
                )
        return self

    @property
    def srpg_effective_steps(self) -> int:
        return min(self.srpg_steps, int(self.srpg_steps * self.srpg_strength))

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        if self.database_backend == "postgresql":
            return URL.create(
                drivername="postgresql+psycopg",
                username=self.database_user,
                password=self.database_password,
                host=self.database_host,
                port=self.database_port,
                database=self.database_name,
            ).render_as_string(hide_password=False)
        return f"sqlite:///{(self.data_dir / 'runs.sqlite3').resolve().as_posix()}"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
