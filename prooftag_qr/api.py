from __future__ import annotations

import csv
import hashlib
import io
import re
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__, metrics
from .artifacts import build_artifact_store
from .backends import build_backends
from .config import get_settings
from .domain import RunRecord
from .lab import LabService, method_schema
from .lab_repository import LabRepository
from .logging import configure_logging
from .repository import RunRepository
from .runtime import runtime_info
from .schemas import (
    AttemptResult,
    GenerationRequest,
    GenerationResponse,
    LabCampaignCreate,
    LabCampaignResponse,
    LabRatingRequest,
    LabRatingResponse,
    LabTrialResponse,
    MetricsSummary,
    PhysicalValidationRequest,
    PhysicalValidationResponse,
    ValidationResult,
)
from .service import GenerationService
from .validation import QRValidator

settings = get_settings()
configure_logging(settings.log_level)
settings.ensure_directories()
repository = RunRepository(
    settings.database_url, create_schema=settings.database_backend == "sqlite"
)
artifact_store = build_artifact_store(settings)
service = GenerationService(
    settings=settings,
    repository=repository,
    artifact_store=artifact_store,
    backends=build_backends(settings),
    validator=QRValidator(),
)
lab_repository = LabRepository(repository.engine)
lab_service = LabService(
    base_settings=settings,
    run_repository=repository,
    lab_repository=lab_repository,
    artifact_store=artifact_store,
    validator=service.validator,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_directories()
    lab_service.start()
    try:
        yield
    finally:
        lab_service.shutdown()


app = FastAPI(
    title="Prooftag Aesthetic QR API",
    version=__version__,
    lifespan=lifespan,
)
static_dir = Path(__file__).with_name("lab_static")
app.mount("/lab-assets", StaticFiles(directory=static_dir), name="lab-assets")


def response_from_record(run: RunRecord) -> GenerationResponse:
    return GenerationResponse(
        id=run.id,
        created_at=run.created_at,
        completed_at=run.completed_at,
        status=run.status,
        backend=run.backend,
        seed=run.seed,
        selected_variant=run.selected_variant,
        selection_mode=run.selection_mode,
        stage1_reused=run.stage1_reused,
        stage1_source_run_id=run.stage1_source_run_id,
        attempts=run.attempts,
        image_url=f"/v1/generations/{run.id}/image" if run.image_path else None,
        qr_version=run.qr_version,
        scan_pass_rate=run.scan_pass_rate,
        exact_payload_match=run.exact_payload_match,
        module_error_rate=run.module_error_rate,
        generation_ms=run.generation_ms,
        validation_ms=run.validation_ms,
        total_ms=run.total_ms,
        error=run.error,
        validations=[
            ValidationResult(
                decoder=item.decoder,
                scenario=item.scenario,
                success=item.success,
                exact_payload_match=item.exact_payload_match,
                latency_ms=item.latency_ms,
                parameters=item.parameters,
            )
            for item in run.validations
        ],
        attempt_details=[AttemptResult(**asdict(item)) for item in run.attempt_details],
        quality_metrics=run.quality_metrics,
        provenance=run.provenance,
    )


def lab_trial_response(values: dict) -> LabTrialResponse:
    run_record = (
        repository.get(values["generation_run_id"]) if values.get("generation_run_id") else None
    )
    run = response_from_record(run_record) if run_record else None
    rating = lab_repository.get_rating(values["id"])
    return LabTrialResponse(
        **values,
        generation=run,
        rating=LabRatingResponse(**rating) if rating else None,
    )


def lab_campaign_response(values: dict, *, include_trials: bool) -> LabCampaignResponse:
    trials = (
        [lab_trial_response(item) for item in lab_repository.list_trials(values["id"])]
        if include_trials
        else []
    )
    return LabCampaignResponse(**values, trials=trials)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/lab")


@app.get("/lab", include_in_schema=False)
def laboratory_ui() -> FileResponse:
    return FileResponse(static_dir / "index.html", media_type="text/html")


@app.get("/healthz", tags=["operations"])
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/readyz", tags=["operations"])
def ready() -> dict:
    repository.ping()
    return {"status": "ready", "backend": settings.default_backend}


@app.get("/v1/runtime", tags=["operations"])
def runtime() -> dict:
    return runtime_info(settings)


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/generations", response_model=GenerationResponse, tags=["generations"])
def create_generation(request: GenerationRequest) -> GenerationResponse:
    run = service.generate(request)
    return response_from_record(run)


@app.get("/v1/generations", response_model=list[GenerationResponse], tags=["generations"])
def list_generations(limit: int = Query(default=100, ge=1, le=1000)) -> list[GenerationResponse]:
    return [response_from_record(run) for run in repository.list(limit)]


@app.get("/v1/generations/{run_id}", response_model=GenerationResponse, tags=["generations"])
def get_generation(run_id: str) -> GenerationResponse:
    run = repository.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Generation not found")
    return response_from_record(run)


@app.get("/v1/generations/{run_id}/image", tags=["generations"])
def get_image(run_id: str):
    run = repository.get(run_id)
    if not run or not run.image_path:
        raise HTTPException(status_code=404, detail="Image not found")
    if run.image_path.startswith("s3://"):
        raise HTTPException(status_code=501, detail="S3 download endpoint is not enabled yet")
    return FileResponse(run.image_path, media_type="image/png", filename=f"{run_id}.png")


@app.get("/v1/generations/{run_id}/variants/{variant}", tags=["experiments"])
def get_generation_variant(run_id: str, variant: str):
    run = repository.get(run_id)
    if not run or not run.image_path:
        raise HTTPException(status_code=404, detail="Generation not found")
    if not re.fullmatch(r"[a-z0-9_]+", variant):
        raise HTTPException(status_code=400, detail="Invalid variant name")
    if run.image_path.startswith("s3://"):
        raise HTTPException(status_code=501, detail="S3 download endpoint is not enabled yet")
    path = Path(run.image_path).parent / "variants" / f"{variant}.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Variant image not found")
    return FileResponse(path, media_type="image/png", filename=f"{run_id}-{variant}.png")


@app.get("/v1/generations/{run_id}/artifacts", tags=["experiments"])
def list_generation_artifacts(run_id: str) -> list[dict[str, str]]:
    run = repository.get(run_id)
    if not run or not run.image_path:
        raise HTTPException(status_code=404, detail="Generation not found")
    artifacts = [{"name": "final", "url": f"/v1/generations/{run_id}/image"}]
    if run.image_path.startswith("s3://"):
        return artifacts
    final_digest = hashlib.sha256(Path(run.image_path).read_bytes()).digest()
    variant_dir = Path(run.image_path).parent / "variants"
    if variant_dir.is_dir():
        artifacts.extend(
            {
                "name": path.stem,
                "url": f"/v1/generations/{run_id}/variants/{path.stem}",
            }
            for path in sorted(variant_dir.glob("*.png"))
            if re.fullmatch(r"[a-z0-9_]+", path.stem)
            and hashlib.sha256(path.read_bytes()).digest() != final_digest
        )
    return artifacts


@app.get("/v1/reports/summary", response_model=MetricsSummary, tags=["reports"])
def summary() -> MetricsSummary:
    return MetricsSummary(**repository.summary())


@app.post(
    "/v1/generations/{run_id}/physical-validations",
    response_model=PhysicalValidationResponse,
    tags=["field-validation"],
)
def create_physical_validation(
    run_id: str, request: PhysicalValidationRequest
) -> PhysicalValidationResponse:
    run = repository.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Generation not found")
    decoded_hash = (
        hashlib.sha256(request.decoded_payload.encode()).hexdigest()
        if request.decoded_payload
        else None
    )
    outcome = (
        "exact"
        if decoded_hash == run.payload_hash
        else ("wrong_payload" if decoded_hash else "not_detected")
    )
    values = request.model_dump(exclude={"decoded_payload"})
    values.update({"outcome": outcome, "decoded_hash": decoded_hash})
    saved = repository.add_physical_validation(run_id, values)
    metrics.PHYSICAL_VALIDATIONS.labels(outcome).inc()
    return PhysicalValidationResponse(**saved)


@app.get(
    "/v1/generations/{run_id}/physical-validations",
    response_model=list[PhysicalValidationResponse],
    tags=["field-validation"],
)
def list_physical_validations(run_id: str) -> list[PhysicalValidationResponse]:
    if not repository.get(run_id):
        raise HTTPException(status_code=404, detail="Generation not found")
    return [
        PhysicalValidationResponse(**item) for item in repository.list_physical_validations(run_id)
    ]


@app.get("/v1/lab/schema", tags=["laboratory"])
def get_lab_schema() -> dict:
    schema = method_schema(settings)
    schema["quality_scoring"] = {
        "clip_enabled": settings.lab_clip_scoring_enabled,
        "device": "cpu",
        "failure_policy": "non_blocking",
    }
    return schema


@app.post(
    "/v1/lab/campaigns",
    response_model=LabCampaignResponse,
    tags=["laboratory"],
)
def create_lab_campaign(request: LabCampaignCreate) -> LabCampaignResponse:
    try:
        campaign = lab_service.create_campaign(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return lab_campaign_response(campaign, include_trials=True)


@app.get(
    "/v1/lab/campaigns",
    response_model=list[LabCampaignResponse],
    tags=["laboratory"],
)
def list_lab_campaigns(
    limit: int = Query(default=50, ge=1, le=500),
) -> list[LabCampaignResponse]:
    return [
        lab_campaign_response(item, include_trials=False)
        for item in lab_repository.list_campaigns(limit)
    ]


@app.get(
    "/v1/lab/campaigns/{campaign_id}",
    response_model=LabCampaignResponse,
    tags=["laboratory"],
)
def get_lab_campaign(campaign_id: str) -> LabCampaignResponse:
    campaign = lab_repository.get_campaign(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return lab_campaign_response(campaign, include_trials=True)


@app.post(
    "/v1/lab/campaigns/{campaign_id}/cancel",
    response_model=LabCampaignResponse,
    tags=["laboratory"],
)
def cancel_lab_campaign(campaign_id: str) -> LabCampaignResponse:
    try:
        campaign = lab_service.cancel(campaign_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Campaign not found") from exc
    return lab_campaign_response(campaign, include_trials=True)


@app.get(
    "/v1/lab/trials/{trial_id}",
    response_model=LabTrialResponse,
    tags=["laboratory"],
)
def get_lab_trial(trial_id: str) -> LabTrialResponse:
    trial = lab_repository.get_trial(trial_id)
    if trial is None:
        raise HTTPException(status_code=404, detail="Trial not found")
    return lab_trial_response(trial)


@app.put(
    "/v1/lab/trials/{trial_id}/rating",
    response_model=LabRatingResponse,
    tags=["laboratory"],
)
def rate_lab_trial(trial_id: str, request: LabRatingRequest) -> LabRatingResponse:
    if lab_repository.get_trial(trial_id) is None:
        raise HTTPException(status_code=404, detail="Trial not found")
    rating = lab_repository.save_rating(trial_id, request.model_dump())
    metrics.LAB_RATINGS.inc()
    return LabRatingResponse(**rating)


@app.get("/v1/lab/campaigns/{campaign_id}/results.csv", tags=["laboratory"])
def export_lab_campaign(campaign_id: str) -> StreamingResponse:
    campaign = lab_repository.get_campaign(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    stream = io.StringIO()
    export_rows = []
    quality_names: set[str] = set()
    provenance_names: set[str] = set()
    for trial in lab_repository.list_trials(campaign_id):
        run = repository.get(trial["generation_run_id"]) if trial["generation_run_id"] else None
        rating = lab_repository.get_rating(trial["id"]) or {}
        quality_names.update(run.quality_metrics if run else {})
        provenance_names.update(run.provenance if run else {})
        export_rows.append((trial, run, rating))
    fields = [
        "trial_id",
        "prompt_id",
        "method_id",
        "seed",
        "status",
        "generation_run_id",
        "selected_variant",
        "selection_mode",
        "stage1_reused",
        "stage1_source_run_id",
        "scan_pass_rate",
        "exact_payload_match",
        "module_error_rate",
        "generation_ms",
        "validation_ms",
        "total_ms",
        "aesthetic_score",
        "aesthetic_ok",
        "human_scan_result",
        "human_scan_attempts",
        "human_scan_successes",
        "human_scan_device",
        "prompt_fidelity_score",
        "qr_discretion_score",
        "overall_score",
        "favorite",
        "notes",
        "error",
    ] + [f"quality_{name}" for name in sorted(quality_names)] + [
        f"provenance_{name}" for name in sorted(provenance_names)
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for trial, run, rating in export_rows:
        row = {
            "trial_id": trial["id"],
            "prompt_id": trial["prompt_id"],
            "method_id": trial["method_id"],
            "seed": trial["seed"],
            "status": trial["status"],
            "generation_run_id": trial["generation_run_id"],
            "selected_variant": run.selected_variant if run else None,
            "selection_mode": run.selection_mode if run else None,
            "stage1_reused": run.stage1_reused if run else None,
            "stage1_source_run_id": run.stage1_source_run_id if run else None,
            "scan_pass_rate": run.scan_pass_rate if run else None,
            "exact_payload_match": run.exact_payload_match if run else None,
            "module_error_rate": run.module_error_rate if run else None,
            "generation_ms": run.generation_ms if run else None,
            "validation_ms": run.validation_ms if run else None,
            "total_ms": run.total_ms if run else None,
            "aesthetic_score": rating.get("aesthetic_score"),
            "aesthetic_ok": rating.get("aesthetic_ok"),
            "human_scan_result": rating.get("human_scan_result"),
            "human_scan_attempts": rating.get("human_scan_attempts"),
            "human_scan_successes": rating.get("human_scan_successes"),
            "human_scan_device": rating.get("human_scan_device"),
            "prompt_fidelity_score": rating.get("prompt_fidelity_score"),
            "qr_discretion_score": rating.get("qr_discretion_score"),
            "overall_score": rating.get("overall_score"),
            "favorite": rating.get("favorite"),
            "notes": rating.get("notes"),
            "error": trial["error"],
        }
        row.update(
            {
                f"quality_{name}": run.quality_metrics.get(name) if run else None
                for name in quality_names
            }
        )
        row.update(
            {
                f"provenance_{name}": run.provenance.get(name) if run else None
                for name in provenance_names
            }
        )
        writer.writerow(row)
    filename = f"prooftag-lab-{campaign_id}.csv"
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/v1/reports/runs.csv", tags=["reports"])
def export_runs() -> StreamingResponse:
    stream = io.StringIO()
    fields = [
        "id",
        "created_at",
        "status",
        "backend",
        "seed",
        "attempts",
        "qr_version",
        "scan_pass_rate",
        "exact_payload_match",
        "module_error_rate",
        "generation_ms",
        "validation_ms",
        "total_ms",
        "payload_hash",
        "error",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for run in repository.list(100000):
        values = asdict(run)
        writer.writerow({field: values.get(field) for field in fields})
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prooftag-qr-runs.csv"},
    )
