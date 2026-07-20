from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
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
    controlnet_pipeline_mode: Literal["text2img", "img2img"] = "img2img"
    device: str = "cuda"
    validation_min_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    max_attempts: int = Field(default=3, ge=1, le=20)
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
