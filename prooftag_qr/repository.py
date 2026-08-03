from __future__ import annotations

import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine, delete, insert, select, update
from sqlalchemy.engine import Engine, RowMapping
from sqlalchemy.pool import NullPool

from .db import attempts, metadata, physical_validations, quality_metrics, runs, validations
from .domain import AttemptRecord, RunRecord, ValidationRecord


class RunRepository:
    def __init__(self, database: str | Path, create_schema: bool = True):
        if isinstance(database, Path):
            database.parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{database.resolve().as_posix()}"
        else:
            database_url = database
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        engine_options = {"pool_pre_ping": True, "connect_args": connect_args}
        if database_url.startswith("sqlite"):
            engine_options["poolclass"] = NullPool
        self.engine: Engine = create_engine(database_url, **engine_options)
        self._lock = threading.RLock()
        if create_schema:
            metadata.create_all(self.engine)

    def ping(self) -> None:
        with self.engine.connect() as connection:
            connection.execute(select(1))

    def close(self) -> None:
        self.engine.dispose()

    def save(self, run: RunRecord) -> None:
        values = asdict(run)
        validation_values = values.pop("validations")
        attempt_values = values.pop("attempt_details")
        quality_values = values.pop("quality_metrics")
        values["created_at"] = run.created_at.astimezone(UTC)
        values["completed_at"] = run.completed_at.astimezone(UTC) if run.completed_at else None

        with self._lock, self.engine.begin() as connection:
            exists = connection.execute(select(runs.c.id).where(runs.c.id == run.id)).first()
            if exists:
                connection.execute(update(runs).where(runs.c.id == run.id).values(**values))
            else:
                connection.execute(insert(runs).values(**values))

            connection.execute(delete(validations).where(validations.c.run_id == run.id))
            if validation_values:
                connection.execute(
                    insert(validations),
                    [
                        {
                            "run_id": run.id,
                            "decoder": item["decoder"],
                            "scenario": item["scenario"],
                            "success": item["success"],
                            "exact_payload_match": item["exact_payload_match"],
                            "latency_ms": item["latency_ms"],
                            "decoded_hash": item["decoded_hash"],
                            "parameters": item["parameters"],
                        }
                        for item in validation_values
                    ],
                )

            connection.execute(delete(attempts).where(attempts.c.run_id == run.id))
            if attempt_values:
                connection.execute(
                    insert(attempts),
                    [{"run_id": run.id, **item} for item in attempt_values],
                )

            connection.execute(delete(quality_metrics).where(quality_metrics.c.run_id == run.id))
            if quality_values:
                connection.execute(
                    insert(quality_metrics),
                    [
                        {"run_id": run.id, "name": name, "value": value}
                        for name, value in quality_values.items()
                    ],
                )

    def get(self, run_id: str) -> RunRecord | None:
        with self.engine.connect() as connection:
            row = connection.execute(select(runs).where(runs.c.id == run_id)).mappings().first()
            if not row:
                return None
            validation_rows = (
                connection.execute(
                    select(validations)
                    .where(validations.c.run_id == run_id)
                    .order_by(validations.c.id)
                )
                .mappings()
                .all()
            )
            attempt_rows = (
                connection.execute(
                    select(attempts).where(attempts.c.run_id == run_id).order_by(attempts.c.attempt)
                )
                .mappings()
                .all()
            )
            quality_rows = (
                connection.execute(
                    select(quality_metrics).where(quality_metrics.c.run_id == run_id)
                )
                .mappings()
                .all()
            )
        return self._to_record(row, validation_rows, attempt_rows, quality_rows)

    def list(self, limit: int = 100) -> list[RunRecord]:
        with self.engine.connect() as connection:
            rows_found = (
                connection.execute(select(runs).order_by(runs.c.created_at.desc()).limit(limit))
                .mappings()
                .all()
            )
        return [self._to_record(row, [], [], []) for row in rows_found]

    def summary(self) -> dict:
        with self.engine.connect() as connection:
            rows_found = (
                connection.execute(
                    select(
                        runs.c.status,
                        runs.c.scan_pass_rate,
                        runs.c.module_error_rate,
                        runs.c.total_ms,
                    )
                )
                .mappings()
                .all()
            )
        totals = [float(row["total_ms"]) for row in rows_found if row["total_ms"] is not None]
        scans = [
            float(row["scan_pass_rate"]) for row in rows_found if row["scan_pass_rate"] is not None
        ]
        modules = [
            float(row["module_error_rate"])
            for row in rows_found
            if row["module_error_rate"] is not None
        ]
        statuses = [row["status"] for row in rows_found]
        totals.sort()

        def percentile(values: list[float], fraction: float) -> float:
            if not values:
                return 0.0
            return values[min(round((len(values) - 1) * fraction), len(values) - 1)]

        accepted = statuses.count("accepted")
        return {
            "total_runs": len(rows_found),
            "accepted_runs": accepted,
            "rejected_runs": statuses.count("rejected"),
            "error_runs": statuses.count("error"),
            "acceptance_rate": accepted / len(rows_found) if rows_found else 0.0,
            "mean_scan_pass_rate": sum(scans) / len(scans) if scans else 0.0,
            "mean_module_error_rate": sum(modules) / len(modules) if modules else 0.0,
            "p50_total_ms": percentile(totals, 0.50),
            "p95_total_ms": percentile(totals, 0.95),
        }

    def add_physical_validation(self, run_id: str, values: dict) -> dict:
        payload = {**values, "run_id": run_id, "created_at": datetime.now(UTC)}
        with self._lock, self.engine.begin() as connection:
            result = connection.execute(insert(physical_validations).values(**payload))
            validation_id = result.inserted_primary_key[0]
        return self.get_physical_validation(validation_id)

    def get_physical_validation(self, validation_id: int) -> dict:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(physical_validations).where(physical_validations.c.id == validation_id)
                )
                .mappings()
                .first()
            )
        if not row:
            raise KeyError(validation_id)
        return dict(row)

    def list_physical_validations(self, run_id: str) -> list[dict]:
        with self.engine.connect() as connection:
            rows_found = (
                connection.execute(
                    select(physical_validations)
                    .where(physical_validations.c.run_id == run_id)
                    .order_by(physical_validations.c.created_at.desc())
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows_found]

    @staticmethod
    def _to_record(
        row: RowMapping,
        validation_rows: list[RowMapping],
        attempt_rows: list[RowMapping],
        quality_rows: list[RowMapping],
    ) -> RunRecord:
        return RunRecord(
            id=row["id"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            status=row["status"],
            backend=row["backend"],
            prompt=row["prompt"],
            payload_hash=row["payload_hash"],
            seed=row["seed"],
            selected_variant=row["selected_variant"],
            selection_mode=row["selection_mode"],
            stage1_reused=row["stage1_reused"],
            stage1_source_run_id=row["stage1_source_run_id"],
            attempts=row["attempts"],
            image_path=row["image_path"],
            qr_version=row["qr_version"],
            scan_pass_rate=row["scan_pass_rate"],
            exact_payload_match=row["exact_payload_match"],
            module_error_rate=row["module_error_rate"],
            generation_ms=row["generation_ms"],
            validation_ms=row["validation_ms"],
            total_ms=row["total_ms"],
            error=row["error"],
            validations=[
                ValidationRecord(
                    decoder=item["decoder"],
                    scenario=item["scenario"],
                    success=item["success"],
                    exact_payload_match=item["exact_payload_match"],
                    latency_ms=item["latency_ms"],
                    decoded_hash=item["decoded_hash"],
                    parameters=item["parameters"],
                )
                for item in validation_rows
            ],
            attempt_details=[
                AttemptRecord(
                    attempt=item["attempt"],
                    seed=item["seed"],
                    generation_ms=item["generation_ms"],
                    validation_ms=item["validation_ms"],
                    scan_pass_rate=item["scan_pass_rate"],
                    module_error_rate=item["module_error_rate"],
                    accepted=item["accepted"],
                )
                for item in attempt_rows
            ],
            quality_metrics={item["name"]: item["value"] for item in quality_rows},
            provenance=dict(row["provenance"] or {}),
        )
