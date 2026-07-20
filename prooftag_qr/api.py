from __future__ import annotations

import csv
import hashlib
import io
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__, metrics
from .artifacts import build_artifact_store
from .backends import build_backends
from .config import get_settings
from .domain import RunRecord
from .logging import configure_logging
from .repository import RunRepository
from .runtime import runtime_info
from .schemas import (
    AttemptResult,
    GenerationRequest,
    GenerationResponse,
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_directories()
    yield


app = FastAPI(
    title="Prooftag Aesthetic QR API",
    version=__version__,
    lifespan=lifespan,
)


def response_from_record(run: RunRecord) -> GenerationResponse:
    return GenerationResponse(
        id=run.id,
        created_at=run.created_at,
        completed_at=run.completed_at,
        status=run.status,
        backend=run.backend,
        seed=run.seed,
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
            )
            for item in run.validations
        ],
        attempt_details=[AttemptResult(**asdict(item)) for item in run.attempt_details],
        quality_metrics=run.quality_metrics,
    )


@app.get("/healthz", tags=["operations"])
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/readyz", tags=["operations"])
def ready() -> dict:
    repository.ping()
    return {"status": "ready", "backend": settings.default_backend}


@app.get("/v1/runtime", tags=["operations"])
def runtime() -> dict:
    return runtime_info()


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
