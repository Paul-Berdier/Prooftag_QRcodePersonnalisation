from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .domain import AttemptRecord, RunRecord, ValidationRecord

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  completed_at TEXT,
  status TEXT NOT NULL,
  backend TEXT NOT NULL,
  prompt TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  seed INTEGER NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  image_path TEXT,
  qr_version INTEGER,
  scan_pass_rate REAL,
  exact_payload_match INTEGER,
  module_error_rate REAL,
  generation_ms REAL,
  validation_ms REAL,
  total_ms REAL,
  error TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE TABLE IF NOT EXISTS validations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  decoder TEXT NOT NULL,
  scenario TEXT NOT NULL,
  success INTEGER NOT NULL,
  exact_payload_match INTEGER NOT NULL,
  latency_ms REAL NOT NULL,
  decoded_hash TEXT,
  parameters_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_validations_run ON validations(run_id);
CREATE INDEX IF NOT EXISTS idx_validations_dimensions ON validations(decoder, scenario);
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  attempt INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  generation_ms REAL NOT NULL,
  validation_ms REAL NOT NULL,
  scan_pass_rate REAL NOT NULL,
  module_error_rate REAL NOT NULL,
  accepted INTEGER NOT NULL,
  UNIQUE(run_id, attempt)
);
CREATE TABLE IF NOT EXISTS quality_metrics (
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  value REAL NOT NULL,
  PRIMARY KEY(run_id, name)
);
CREATE TABLE IF NOT EXISTS physical_validations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  device TEXT NOT NULL,
  operating_system TEXT NOT NULL,
  scanner TEXT NOT NULL,
  print_profile TEXT NOT NULL,
  material TEXT NOT NULL,
  size_mm REAL,
  lighting TEXT NOT NULL,
  distance_cm REAL,
  angle_degrees REAL,
  scan_latency_ms REAL,
  outcome TEXT NOT NULL,
  decoded_hash TEXT,
  notes TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_physical_run ON physical_validations(run_id);
CREATE INDEX IF NOT EXISTS idx_physical_dimensions
  ON physical_validations(device, print_profile, material, outcome);
"""


class RunRepository:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with closing(self._connect()) as connection, connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, run: RunRecord) -> None:
        values = asdict(run)
        validations = values.pop("validations")
        attempts = values.pop("attempt_details")
        quality_metrics = values.pop("quality_metrics")
        values["created_at"] = run.created_at.astimezone(UTC).isoformat()
        values["completed_at"] = (
            run.completed_at.astimezone(UTC).isoformat() if run.completed_at else None
        )
        values["exact_payload_match"] = (
            int(run.exact_payload_match) if run.exact_payload_match is not None else None
        )
        columns = list(values)
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(f"{column}=excluded.{column}" for column in columns if column != "id")
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                f"INSERT INTO runs ({','.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}",
                [values[column] for column in columns],
            )
            connection.execute("DELETE FROM validations WHERE run_id = ?", (run.id,))
            connection.executemany(
                """
                INSERT INTO validations
                (run_id, decoder, scenario, success, exact_payload_match, latency_ms,
                 decoded_hash, parameters_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run.id,
                        item["decoder"],
                        item["scenario"],
                        int(item["success"]),
                        int(item["exact_payload_match"]),
                        item["latency_ms"],
                        item["decoded_hash"],
                        json.dumps(item["parameters"]),
                    )
                    for item in validations
                ],
            )
            connection.execute("DELETE FROM attempts WHERE run_id = ?", (run.id,))
            connection.executemany(
                """
                INSERT INTO attempts
                (run_id, attempt, seed, generation_ms, validation_ms, scan_pass_rate,
                 module_error_rate, accepted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run.id,
                        item["attempt"],
                        item["seed"],
                        item["generation_ms"],
                        item["validation_ms"],
                        item["scan_pass_rate"],
                        item["module_error_rate"],
                        int(item["accepted"]),
                    )
                    for item in attempts
                ],
            )
            connection.execute("DELETE FROM quality_metrics WHERE run_id = ?", (run.id,))
            connection.executemany(
                "INSERT INTO quality_metrics (run_id, name, value) VALUES (?, ?, ?)",
                [(run.id, name, value) for name, value in quality_metrics.items()],
            )

    def get(self, run_id: str) -> RunRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return None
            validations = connection.execute(
                "SELECT * FROM validations WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
            attempts = connection.execute(
                "SELECT * FROM attempts WHERE run_id = ? ORDER BY attempt", (run_id,)
            ).fetchall()
            quality_rows = connection.execute(
                "SELECT name, value FROM quality_metrics WHERE run_id = ?", (run_id,)
            ).fetchall()
        return self._to_record(row, validations, attempts, quality_rows)

    def list(self, limit: int = 100) -> list[RunRecord]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._to_record(row, [], [], []) for row in rows]

    def summary(self) -> dict:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT status, scan_pass_rate, module_error_rate, total_ms FROM runs"
            ).fetchall()
        totals = [float(row["total_ms"]) for row in rows if row["total_ms"] is not None]
        scans = [float(row["scan_pass_rate"]) for row in rows if row["scan_pass_rate"] is not None]
        modules = [
            float(row["module_error_rate"]) for row in rows if row["module_error_rate"] is not None
        ]
        statuses = [row["status"] for row in rows]
        totals.sort()

        def percentile(values: list[float], fraction: float) -> float:
            if not values:
                return 0.0
            return values[min(round((len(values) - 1) * fraction), len(values) - 1)]

        accepted = statuses.count("accepted")
        return {
            "total_runs": len(rows),
            "accepted_runs": accepted,
            "rejected_runs": statuses.count("rejected"),
            "error_runs": statuses.count("error"),
            "acceptance_rate": accepted / len(rows) if rows else 0.0,
            "mean_scan_pass_rate": sum(scans) / len(scans) if scans else 0.0,
            "mean_module_error_rate": sum(modules) / len(modules) if modules else 0.0,
            "p50_total_ms": percentile(totals, 0.50),
            "p95_total_ms": percentile(totals, 0.95),
        }

    def add_physical_validation(self, run_id: str, values: dict) -> dict:
        columns = [
            "run_id",
            "created_at",
            "device",
            "operating_system",
            "scanner",
            "print_profile",
            "material",
            "size_mm",
            "lighting",
            "distance_cm",
            "angle_degrees",
            "scan_latency_ms",
            "outcome",
            "decoded_hash",
            "notes",
        ]
        payload = {**values, "run_id": run_id, "created_at": datetime.now(UTC).isoformat()}
        with self._lock, closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                f"INSERT INTO physical_validations ({','.join(columns)}) "
                f"VALUES ({','.join('?' for _ in columns)})",
                [payload[column] for column in columns],
            )
            validation_id = cursor.lastrowid
        return self.get_physical_validation(validation_id)

    def get_physical_validation(self, validation_id: int) -> dict:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM physical_validations WHERE id = ?", (validation_id,)
            ).fetchone()
        if not row:
            raise KeyError(validation_id)
        return dict(row)

    def list_physical_validations(self, run_id: str) -> list[dict]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM physical_validations WHERE run_id = ? ORDER BY created_at DESC",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _to_record(
        row: sqlite3.Row,
        validation_rows: list[sqlite3.Row],
        attempt_rows: list[sqlite3.Row],
        quality_rows: list[sqlite3.Row],
    ) -> RunRecord:
        return RunRecord(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
            status=row["status"],
            backend=row["backend"],
            prompt=row["prompt"],
            payload_hash=row["payload_hash"],
            seed=row["seed"],
            attempts=row["attempts"],
            image_path=row["image_path"],
            qr_version=row["qr_version"],
            scan_pass_rate=row["scan_pass_rate"],
            exact_payload_match=(
                bool(row["exact_payload_match"]) if row["exact_payload_match"] is not None else None
            ),
            module_error_rate=row["module_error_rate"],
            generation_ms=row["generation_ms"],
            validation_ms=row["validation_ms"],
            total_ms=row["total_ms"],
            error=row["error"],
            validations=[
                ValidationRecord(
                    decoder=item["decoder"],
                    scenario=item["scenario"],
                    success=bool(item["success"]),
                    exact_payload_match=bool(item["exact_payload_match"]),
                    latency_ms=item["latency_ms"],
                    decoded_hash=item["decoded_hash"],
                    parameters=json.loads(item["parameters_json"]),
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
                    accepted=bool(item["accepted"]),
                )
                for item in attempt_rows
            ],
            quality_metrics={item["name"]: item["value"] for item in quality_rows},
        )
