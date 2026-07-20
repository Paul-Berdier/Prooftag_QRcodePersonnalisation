from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class ValidationRecord:
    decoder: str
    scenario: str
    success: bool
    exact_payload_match: bool
    latency_ms: float
    decoded_hash: str | None = None
    parameters: dict = field(default_factory=dict)


@dataclass(slots=True)
class AttemptRecord:
    attempt: int
    seed: int
    generation_ms: float
    validation_ms: float
    scan_pass_rate: float
    module_error_rate: float
    accepted: bool


@dataclass(slots=True)
class RunRecord:
    id: str
    created_at: datetime
    completed_at: datetime | None
    status: str
    backend: str
    prompt: str
    payload_hash: str
    seed: int
    attempts: int = 0
    image_path: str | None = None
    qr_version: int | None = None
    scan_pass_rate: float | None = None
    exact_payload_match: bool | None = None
    module_error_rate: float | None = None
    generation_ms: float | None = None
    validation_ms: float | None = None
    total_ms: float | None = None
    error: str | None = None
    validations: list[ValidationRecord] = field(default_factory=list)
    attempt_details: list[AttemptRecord] = field(default_factory=list)
    quality_metrics: dict[str, float] = field(default_factory=dict)
