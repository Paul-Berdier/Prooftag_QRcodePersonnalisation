from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class GenerationRequest(BaseModel):
    payload: str = Field(min_length=1, max_length=2048)
    prompt: str = Field(default="", max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    backend: Literal["qr", "controlnet"] | None = None
    error_correction: Literal["M", "Q", "H"] = "H"
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    steps: int = Field(default=12, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=0, le=30)
    controlnet_scale: float = Field(default=1.35, ge=0, le=3)
    strength: float = Field(default=0.9, gt=0, le=1)
    max_attempts: int | None = Field(default=None, ge=1, le=20)


class ValidationResult(BaseModel):
    decoder: str
    scenario: str
    success: bool
    exact_payload_match: bool
    latency_ms: float
    parameters: dict[str, Any] = Field(default_factory=dict)


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
    selected_variant: str | None = None
    selection_mode: Literal["delivery", "forced"] = "delivery"
    stage1_reused: bool = False
    stage1_source_run_id: str | None = None
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
    provenance: dict[str, str] = Field(default_factory=dict)


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


class LabPrompt(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$", min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)


class LabToolConfig(BaseModel):
    srpg_enabled: bool = False
    srmpgd_enabled: bool = False
    guided_rediffusion_enabled: bool = False
    latent_refinement_enabled: bool = False
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_exclusive_stage2(self) -> "LabToolConfig":
        if self.srpg_enabled and self.guided_rediffusion_enabled:
            raise ValueError("SRPG and guided rediffusion cannot be enabled together")
        if self.guided_rediffusion_enabled or self.latent_refinement_enabled:
            raise ValueError(
                "the restarted DiffQRCoder lab only supports SRPG and SR-MPGD"
            )
        if self.srmpgd_enabled and not self.srpg_enabled:
            raise ValueError("paper SR-MPGD requires Stage 2 SRPG")
        return self


class LabMethod(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$", min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    backend: Literal["qr", "controlnet"] = "controlnet"
    enabled: bool = True
    output_variant: Literal["raw", "srpg", "srmpgd", "auto"] = "raw"
    reuse_stage1: bool = True
    generation: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)
    tools: LabToolConfig = Field(default_factory=LabToolConfig)

    @model_validator(mode="before")
    @classmethod
    def recover_output_variant_from_legacy_web_form(cls, value: Any) -> Any:
        """Repair the empty value emitted by a cached pre-auto HTML select."""
        if not isinstance(value, dict) or value.get("output_variant") not in {None, ""}:
            return value
        method_id = str(value.get("id", ""))
        tools = value.get("tools") or {}
        if method_id == "diffqrcoder_auto" or method_id.endswith("_auto"):
            output_variant = "auto"
        elif bool(tools.get("srmpgd_enabled")):
            output_variant = "srmpgd"
        elif bool(tools.get("srpg_enabled")):
            output_variant = "srpg"
        else:
            output_variant = "raw"
        return {**value, "output_variant": output_variant}


class LabCampaignCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    payload: str = Field(min_length=1, max_length=2048)
    error_correction: Literal["M", "Q", "H"] = "M"
    prompts: list[LabPrompt] = Field(min_length=1, max_length=50)
    seeds: list[int] = Field(min_length=1, max_length=20)
    methods: list[LabMethod] = Field(min_length=1, max_length=25)
    max_attempts: int = Field(default=1, ge=1, le=20)

    @model_validator(mode="after")
    def validate_campaign(self) -> "LabCampaignCreate":
        if len({item.id for item in self.prompts}) != len(self.prompts):
            raise ValueError("prompt ids must be unique")
        active = [item for item in self.methods if item.enabled]
        if not active:
            raise ValueError("at least one method must be enabled")
        if len({item.id for item in active}) != len(active):
            raise ValueError("enabled method ids must be unique")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if any(seed < 0 or seed > 2**32 - 1 for seed in self.seeds):
            raise ValueError("seeds must be between 0 and 2^32 - 1")
        if len(self.prompts) * len(self.seeds) * len(active) > 500:
            raise ValueError("a campaign is limited to 500 trials")
        return self


class LabRatingRequest(BaseModel):
    aesthetic_score: int | None = Field(default=None, ge=1, le=10)
    aesthetic_ok: bool | None = None
    human_scan_result: Literal["scannable", "not_scannable", "not_tested"] = "not_tested"
    human_scan_attempts: int = Field(default=0, ge=0, le=20)
    human_scan_successes: int = Field(default=0, ge=0, le=20)
    human_scan_device: str = Field(default="", max_length=200)
    prompt_fidelity_score: int | None = Field(default=None, ge=1, le=10)
    qr_discretion_score: int | None = Field(default=None, ge=1, le=10)
    overall_score: int | None = Field(default=None, ge=1, le=10)
    favorite: bool = False
    notes: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def normalize_repeated_phone_scan(self) -> "LabRatingRequest":
        if self.human_scan_successes > self.human_scan_attempts:
            raise ValueError("human scan successes cannot exceed attempts")
        if self.human_scan_attempts == 0:
            if self.human_scan_result == "scannable":
                self.human_scan_attempts = 1
                self.human_scan_successes = 1
            elif self.human_scan_result == "not_scannable":
                self.human_scan_attempts = 1
                self.human_scan_successes = 0
        else:
            self.human_scan_result = (
                "scannable"
                if self.human_scan_successes / self.human_scan_attempts >= 2 / 3
                else "not_scannable"
            )
        return self


class LabRatingResponse(LabRatingRequest):
    id: int
    trial_id: str
    created_at: datetime
    updated_at: datetime


class LabTrialResponse(BaseModel):
    id: str
    campaign_id: str
    created_at: datetime
    completed_at: datetime | None
    prompt_id: str
    method_id: str
    seed: int
    status: Literal["queued", "running", "accepted", "rejected", "error", "cancelled"]
    generation_run_id: str | None
    configuration: dict[str, Any]
    error: str | None
    generation: GenerationResponse | None = None
    rating: LabRatingResponse | None = None


class LabCampaignResponse(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    name: str
    status: Literal[
        "queued", "running", "completed", "completed_with_errors", "cancelled", "interrupted"
    ]
    payload_hash: str
    specification: dict[str, Any]
    total_trials: int
    completed_trials: int
    accepted_trials: int
    error: str | None
    trials: list[LabTrialResponse] = Field(default_factory=list)
