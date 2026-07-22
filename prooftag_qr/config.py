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
        if self.srpg_enabled and self.controlnet_pipeline_mode != "img2img":
            raise ValueError("SRPG requires the img2img ControlNet pipeline")
        if self.srpg_dark_threshold > self.srpg_light_threshold:
            raise ValueError("SRPG dark threshold cannot exceed light threshold")
        if self.srpg_robust_blur_kernel % 2 == 0:
            raise ValueError("SRPG robust blur kernel must be odd")
        return self

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
