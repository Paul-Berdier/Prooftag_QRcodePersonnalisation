from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    payload: str = Field(min_length=1, max_length=2048)
    prompt: str = Field(default="", max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    backend: Literal["qr", "controlnet"] | None = None
    error_correction: Literal["M", "Q", "H"] = "Q"
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    steps: int = Field(default=12, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=0, le=30)
    controlnet_scale: float = Field(default=1.35, ge=0, le=3)
    max_attempts: int | None = Field(default=None, ge=1, le=20)


class ValidationResult(BaseModel):
    decoder: str
    scenario: str
    success: bool
    exact_payload_match: bool
    latency_ms: float


class AttemptResult(BaseModel):
    attempt: int
    seed: int
    generation_ms: float
    validation_ms: float
    scan_pass_rate: float
    module_error_rate: float
    accepted: bool


class GenerationResponse(BaseModel):
    id: str
    created_at: datetime
    completed_at: datetime | None = None
    status: Literal["running", "accepted", "rejected", "error"]
    backend: str
    seed: int
    attempts: int
    image_url: str | None = None
    qr_version: int | None = None
    scan_pass_rate: float | None = None
    exact_payload_match: bool | None = None
    module_error_rate: float | None = None
    generation_ms: float | None = None
    validation_ms: float | None = None
    total_ms: float | None = None
    error: str | None = None
    validations: list[ValidationResult] = Field(default_factory=list)
    attempt_details: list[AttemptResult] = Field(default_factory=list)
    quality_metrics: dict[str, float] = Field(default_factory=dict)


class MetricsSummary(BaseModel):
    total_runs: int
    accepted_runs: int
    rejected_runs: int
    error_runs: int
    acceptance_rate: float
    mean_scan_pass_rate: float
    mean_module_error_rate: float
    p50_total_ms: float
    p95_total_ms: float


class PhysicalValidationRequest(BaseModel):
    decoded_payload: str | None = Field(default=None, max_length=2048)
    device: str = Field(min_length=1, max_length=100)
    operating_system: str = Field(default="", max_length=100)
    scanner: str = Field(default="native-camera", max_length=100)
    print_profile: str = Field(default="digital", max_length=100)
    material: str = Field(default="screen", max_length=100)
    size_mm: float | None = Field(default=None, gt=0, le=1000)
    lighting: str = Field(default="office", max_length=100)
    distance_cm: float | None = Field(default=None, ge=0, le=1000)
    angle_degrees: float | None = Field(default=None, ge=0, le=90)
    scan_latency_ms: float | None = Field(default=None, ge=0, le=120000)
    notes: str = Field(default="", max_length=1000)


class PhysicalValidationResponse(BaseModel):
    id: int
    run_id: str
    created_at: datetime
    device: str
    operating_system: str
    scanner: str
    print_profile: str
    material: str
    size_mm: float | None
    lighting: str
    distance_cm: float | None
    angle_degrees: float | None
    scan_latency_ms: float | None
    outcome: Literal["exact", "wrong_payload", "not_detected"]
    notes: str
