"""Infrastructure de reprise sûre pour les futures générations et entraînements.

Le module est volontairement indépendant de PyTorch et de Kubernetes. Il fournit :

- un plan et des tâches identifiés par hash sémantique ;
- des transactions SQLite atomiques en mode WAL ;
- des leases/heartbeats pour détecter un worker disparu ;
- une classification explicite des échecs ;
- aucune relance aveugle d'un OOM ou d'une erreur déterministe ;
- des tentatives immuables et une promotion atomique des artefacts ;
- un journal append-only.

E045 l'utilise pour construire son inventaire. E046/E047 pourront réutiliser le
même contrat pour les générations GPU et les entraînements.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

TaskStatus = Literal[
    "pending",
    "running",
    "retry_wait",
    "succeeded",
    "failed",
    "blocked",
    "cancelled",
]
FailureKind = Literal[
    "transient",
    "resource",
    "deterministic",
    "unknown",
]


class ResilientExperimentError(RuntimeError):
    """Base des erreurs du contrat de reprise."""


class ContractMismatchError(ResilientExperimentError):
    """Un identifiant existant a été réutilisé avec une autre spécification."""


class ActiveLeaseError(ResilientExperimentError):
    """Une autre exécution possède encore un lease valide."""


class ArtifactPromotionError(ResilientExperimentError):
    """Une tentative ne peut pas être promue de façon sûre."""


@dataclass(frozen=True, slots=True)
class FailureDecision:
    kind: FailureKind
    retryable: bool
    operator_action_required: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ClaimedTask:
    run_id: str
    task_id: str
    kind: str
    spec: dict[str, Any]
    spec_hash: str
    attempt_no: int
    max_attempts: int
    lease_owner: str
    lease_expires_at: float
    artifact_dir: str | None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def atomic_write_text(path: Path, text: str, mode: int = 0o644) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def atomic_write_json(path: Path, value: Any, mode: int = 0o644) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        mode=mode,
    )


def classify_failure(error: BaseException | str) -> FailureDecision:
    """Classe un échec sans le masquer.

    Un OOM, un disque plein ou un quota n'est jamais relancé à l'identique :
    l'opérateur doit modifier les ressources, le batch, la précision ou le plan.
    """
    if isinstance(error, BaseException):
        name = type(error).__name__
        message = f"{name}: {error}"
    else:
        name = "Error"
        message = str(error)
    lowered = message.lower()

    resource_tokens = (
        "cuda out of memory",
        "out of memory",
        "oom",
        "no space left",
        "disk quota",
        "too many open files",
        "fsnotify watcher",
        "resource exhausted",
        "cannot allocate memory",
    )
    deterministic_tokens = (
        "assertionerror",
        "valueerror",
        "keyerror",
        "filenotfounderror",
        "contract",
        "schema",
        "invalid configuration",
        "dimension mismatch",
        "shape mismatch",
        "missing required",
        "payload mismatch",
        "checksum mismatch",
        "limite max_files dépassée",
        "configuration max_files insuffisante",
        "max files exceeded",
    )
    transient_tokens = (
        "timeout",
        "timed out",
        "connection reset",
        "connection refused",
        "temporary failure",
        "temporarily unavailable",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "pod evicted",
        "node lost",
        "preempted",
        "dns",
        "tls handshake timeout",
        "broken pipe",
    )

    if any(token in lowered for token in resource_tokens):
        return FailureDecision(
            kind="resource",
            retryable=False,
            operator_action_required=True,
            reason="Les mêmes ressources et la même spécification reproduiraient probablement l'échec.",
        )
    if any(token in lowered for token in deterministic_tokens):
        return FailureDecision(
            kind="deterministic",
            retryable=False,
            operator_action_required=True,
            reason="Le contrat, les entrées ou le code doivent être corrigés avant une nouvelle tentative.",
        )
    if any(token in lowered for token in transient_tokens):
        return FailureDecision(
            kind="transient",
            retryable=True,
            operator_action_required=False,
            reason="L'échec paraît transitoire ; une reprise bornée est autorisée.",
        )
    return FailureDecision(
        kind="unknown",
        retryable=True,
        operator_action_required=False,
        reason="Une seule reprise prudente est autorisée avant blocage opérateur.",
    )


class ResilientTaskStore:
    """File de tâches transactionnelle persistante."""

    def __init__(self, database: Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database,
            timeout=60,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    plan_hash TEXT NOT NULL,
                    source_commit TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    spec_hash TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 100,
                    max_attempts INTEGER NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    not_before REAL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    heartbeat_at REAL,
                    started_at TEXT,
                    finished_at TEXT,
                    artifact_dir TEXT,
                    result_json TEXT,
                    last_error_kind TEXT,
                    last_error_class TEXT,
                    last_error_message TEXT,
                    last_traceback TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_claim
                    ON tasks(run_id, status, not_before, priority, task_id);

                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    attempt_no INTEGER NOT NULL,
                    worker_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    checkpoint_path TEXT,
                    telemetry_json TEXT,
                    error_kind TEXT,
                    error_class TEXT,
                    error_message TEXT,
                    traceback_text TEXT,
                    UNIQUE(task_id, attempt_no)
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    task_id TEXT,
                    level TEXT NOT NULL,
                    event TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        *,
        run_id: str,
        task_id: str | None,
        level: str,
        event: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO events(timestamp, run_id, task_id, level, event, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                run_id,
                task_id,
                level,
                event,
                canonical_json(dict(payload or {})),
            ),
        )

    def register_run(
        self,
        *,
        run_id: str,
        plan: Mapping[str, Any],
        source_commit: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        plan_hash = stable_hash(plan)
        now = utc_now()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT plan_hash, source_commit FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if existing:
                if existing["plan_hash"] != plan_hash:
                    raise ContractMismatchError(
                        f"run_id {run_id} existe avec un autre plan"
                    )
                if existing["source_commit"] != source_commit:
                    raise ContractMismatchError(
                        f"run_id {run_id} existe avec un autre commit"
                    )
                connection.execute(
                    "UPDATE runs SET updated_at=? WHERE run_id=?",
                    (now, run_id),
                )
                return plan_hash

            connection.execute(
                """
                INSERT INTO runs(
                    run_id, plan_hash, source_commit, status,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    run_id,
                    plan_hash,
                    source_commit,
                    canonical_json(dict(metadata or {})),
                    now,
                    now,
                ),
            )
            self._event(
                connection,
                run_id=run_id,
                task_id=None,
                level="INFO",
                event="run_registered",
                payload={"plan_hash": plan_hash, "source_commit": source_commit},
            )
        return plan_hash

    def register_tasks(
        self,
        *,
        run_id: str,
        tasks: Sequence[Mapping[str, Any]],
    ) -> None:
        now = utc_now()
        with self.transaction() as connection:
            if not connection.execute(
                "SELECT 1 FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone():
                raise ResilientExperimentError(f"run inconnu : {run_id}")

            for item in tasks:
                task_id = str(item["task_id"])
                spec = dict(item.get("spec") or {})
                spec_hash = stable_hash(spec)
                existing = connection.execute(
                    "SELECT run_id, spec_hash FROM tasks WHERE task_id=?",
                    (task_id,),
                ).fetchone()
                if existing:
                    if existing["run_id"] != run_id or existing["spec_hash"] != spec_hash:
                        raise ContractMismatchError(
                            f"task_id {task_id} existe avec un autre contrat"
                        )
                    continue

                connection.execute(
                    """
                    INSERT INTO tasks(
                        task_id, run_id, kind, spec_hash, spec_json, status,
                        priority, max_attempts, artifact_dir, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        run_id,
                        str(item.get("kind") or "generic"),
                        spec_hash,
                        canonical_json(spec),
                        int(item.get("priority", 100)),
                        int(item.get("max_attempts", 2)),
                        (
                            str(item["artifact_dir"])
                            if item.get("artifact_dir") is not None
                            else None
                        ),
                        now,
                        now,
                    ),
                )
                self._event(
                    connection,
                    run_id=run_id,
                    task_id=task_id,
                    level="INFO",
                    event="task_registered",
                    payload={"spec_hash": spec_hash},
                )

    def recover_stale(
        self,
        *,
        run_id: str,
        now_epoch: float | None = None,
        force: bool = False,
    ) -> list[str]:
        now_epoch = time.time() if now_epoch is None else now_epoch
        recovered: list[str] = []
        with self.transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM tasks
                WHERE run_id=? AND status='running'
                """,
                (run_id,),
            ).fetchall()
            for row in rows:
                expires = row["lease_expires_at"]
                if not force and expires is not None and float(expires) > now_epoch:
                    continue
                next_status = (
                    "pending"
                    if int(row["attempt_count"]) < int(row["max_attempts"])
                    else "blocked"
                )
                connection.execute(
                    """
                    UPDATE tasks
                    SET status=?, lease_owner=NULL, lease_expires_at=NULL,
                        heartbeat_at=NULL, updated_at=?,
                        last_error_kind='transient',
                        last_error_class='StaleLease',
                        last_error_message='Worker disparu ou lease expiré'
                    WHERE task_id=?
                    """,
                    (next_status, utc_now(), row["task_id"]),
                )
                connection.execute(
                    """
                    UPDATE attempts
                    SET status='abandoned', finished_at=?,
                        error_kind='transient',
                        error_class='StaleLease',
                        error_message='Worker disparu ou lease expiré'
                    WHERE task_id=? AND attempt_no=? AND status='running'
                    """,
                    (utc_now(), row["task_id"], row["attempt_count"]),
                )
                self._event(
                    connection,
                    run_id=run_id,
                    task_id=row["task_id"],
                    level="WARNING",
                    event="stale_task_recovered",
                    payload={"next_status": next_status},
                )
                recovered.append(str(row["task_id"]))
        return recovered

    def claim_next(
        self,
        *,
        run_id: str,
        worker_id: str,
        lease_seconds: int = 600,
        kinds: Iterable[str] | None = None,
    ) -> ClaimedTask | None:
        now_epoch = time.time()
        now = utc_now()
        kind_list = list(kinds or [])
        with self.transaction() as connection:
            parameters: list[Any] = [run_id, now_epoch]
            kind_sql = ""
            if kind_list:
                placeholders = ",".join("?" for _ in kind_list)
                kind_sql = f" AND kind IN ({placeholders})"
                parameters.extend(kind_list)

            row = connection.execute(
                f"""
                SELECT * FROM tasks
                WHERE run_id=?
                  AND status IN ('pending', 'retry_wait')
                  AND (not_before IS NULL OR not_before <= ?)
                  {kind_sql}
                ORDER BY priority ASC, task_id ASC
                LIMIT 1
                """,
                tuple(parameters),
            ).fetchone()
            if row is None:
                return None

            attempt_no = int(row["attempt_count"]) + 1
            if attempt_no > int(row["max_attempts"]):
                connection.execute(
                    "UPDATE tasks SET status='blocked', updated_at=? WHERE task_id=?",
                    (now, row["task_id"]),
                )
                return None

            lease_expires = now_epoch + lease_seconds
            connection.execute(
                """
                UPDATE tasks
                SET status='running', attempt_count=?,
                    lease_owner=?, lease_expires_at=?, heartbeat_at=?,
                    started_at=COALESCE(started_at, ?), updated_at=?,
                    not_before=NULL
                WHERE task_id=?
                """,
                (
                    attempt_no,
                    worker_id,
                    lease_expires,
                    now_epoch,
                    now,
                    now,
                    row["task_id"],
                ),
            )
            connection.execute(
                """
                INSERT INTO attempts(
                    task_id, attempt_no, worker_id, status, started_at
                ) VALUES (?, ?, ?, 'running', ?)
                """,
                (row["task_id"], attempt_no, worker_id, now),
            )
            self._event(
                connection,
                run_id=run_id,
                task_id=row["task_id"],
                level="INFO",
                event="task_claimed",
                payload={
                    "worker_id": worker_id,
                    "attempt_no": attempt_no,
                    "lease_seconds": lease_seconds,
                },
            )
            return ClaimedTask(
                run_id=run_id,
                task_id=str(row["task_id"]),
                kind=str(row["kind"]),
                spec=json.loads(row["spec_json"]),
                spec_hash=str(row["spec_hash"]),
                attempt_no=attempt_no,
                max_attempts=int(row["max_attempts"]),
                lease_owner=worker_id,
                lease_expires_at=lease_expires,
                artifact_dir=row["artifact_dir"],
            )

    def heartbeat(
        self,
        task: ClaimedTask,
        *,
        lease_seconds: int = 600,
        telemetry: Mapping[str, Any] | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        now_epoch = time.time()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT status, lease_owner, attempt_count FROM tasks WHERE task_id=?",
                (task.task_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] != "running"
                or row["lease_owner"] != task.lease_owner
                or int(row["attempt_count"]) != task.attempt_no
            ):
                raise ActiveLeaseError(f"lease perdu pour {task.task_id}")

            connection.execute(
                """
                UPDATE tasks
                SET heartbeat_at=?, lease_expires_at=?, updated_at=?
                WHERE task_id=?
                """,
                (
                    now_epoch,
                    now_epoch + lease_seconds,
                    utc_now(),
                    task.task_id,
                ),
            )
            connection.execute(
                """
                UPDATE attempts
                SET telemetry_json=COALESCE(?, telemetry_json),
                    checkpoint_path=COALESCE(?, checkpoint_path)
                WHERE task_id=? AND attempt_no=? AND status='running'
                """,
                (
                    (
                        canonical_json(dict(telemetry))
                        if telemetry is not None
                        else None
                    ),
                    checkpoint_path,
                    task.task_id,
                    task.attempt_no,
                ),
            )

    def complete(
        self,
        task: ClaimedTask,
        *,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT status, lease_owner, attempt_count FROM tasks WHERE task_id=?",
                (task.task_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] != "running"
                or row["lease_owner"] != task.lease_owner
                or int(row["attempt_count"]) != task.attempt_no
            ):
                raise ActiveLeaseError(f"impossible de terminer {task.task_id}: lease invalide")

            connection.execute(
                """
                UPDATE tasks
                SET status='succeeded', finished_at=?, updated_at=?,
                    lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                    result_json=?, last_error_kind=NULL, last_error_class=NULL,
                    last_error_message=NULL, last_traceback=NULL
                WHERE task_id=?
                """,
                (
                    now,
                    now,
                    canonical_json(dict(result or {})),
                    task.task_id,
                ),
            )
            connection.execute(
                """
                UPDATE attempts
                SET status='succeeded', finished_at=?
                WHERE task_id=? AND attempt_no=? AND status='running'
                """,
                (now, task.task_id, task.attempt_no),
            )
            self._event(
                connection,
                run_id=task.run_id,
                task_id=task.task_id,
                level="INFO",
                event="task_succeeded",
                payload=dict(result or {}),
            )
            self._refresh_run_status(connection, task.run_id)

    def fail(
        self,
        task: ClaimedTask,
        error: BaseException,
        *,
        retry_delay_seconds: int = 30,
        force_retryable: bool | None = None,
    ) -> FailureDecision:
        decision = classify_failure(error)
        retryable = decision.retryable if force_retryable is None else force_retryable
        error_class = type(error).__name__
        error_message = str(error)[:4000]
        trace = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )[-16000:]
        now = utc_now()

        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id=?",
                (task.task_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] != "running"
                or row["lease_owner"] != task.lease_owner
                or int(row["attempt_count"]) != task.attempt_no
            ):
                raise ActiveLeaseError(f"impossible d'échouer {task.task_id}: lease invalide")

            attempts_left = int(row["attempt_count"]) < int(row["max_attempts"])
            # Un échec inconnu ne reçoit qu'une reprise. Un OOM/contrat reste bloqué.
            if retryable and attempts_left:
                status = "retry_wait"
                not_before = time.time() + retry_delay_seconds
            else:
                status = "blocked" if decision.operator_action_required else "failed"
                not_before = None

            connection.execute(
                """
                UPDATE tasks
                SET status=?, not_before=?, updated_at=?,
                    lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                    last_error_kind=?, last_error_class=?,
                    last_error_message=?, last_traceback=?
                WHERE task_id=?
                """,
                (
                    status,
                    not_before,
                    now,
                    decision.kind,
                    error_class,
                    error_message,
                    trace,
                    task.task_id,
                ),
            )
            connection.execute(
                """
                UPDATE attempts
                SET status=?, finished_at=?, error_kind=?, error_class=?,
                    error_message=?, traceback_text=?
                WHERE task_id=? AND attempt_no=? AND status='running'
                """,
                (
                    status,
                    now,
                    decision.kind,
                    error_class,
                    error_message,
                    trace,
                    task.task_id,
                    task.attempt_no,
                ),
            )
            self._event(
                connection,
                run_id=task.run_id,
                task_id=task.task_id,
                level="ERROR",
                event="task_failed",
                payload={
                    "kind": decision.kind,
                    "next_status": status,
                    "attempt_no": task.attempt_no,
                    "error_class": error_class,
                    "error_message": error_message,
                },
            )
            self._refresh_run_status(connection, task.run_id)
        return dataclasses.replace(decision, retryable=retryable and attempts_left)

    @staticmethod
    def _refresh_run_status(connection: sqlite3.Connection, run_id: str) -> None:
        counts = {
            row["status"]: int(row["count"])
            for row in connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM tasks WHERE run_id=? GROUP BY status
                """,
                (run_id,),
            )
        }
        if counts.get("blocked") or counts.get("failed"):
            status = "attention"
        elif counts.get("running") or counts.get("pending") or counts.get("retry_wait"):
            status = "running"
        else:
            status = "succeeded"
        connection.execute(
            "UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
            (status, utc_now(), run_id),
        )

    def task_rows(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE run_id=? ORDER BY priority, task_id",
                (run_id,),
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            for key in ("spec_json", "result_json"):
                if item.get(key):
                    with contextlib.suppress(json.JSONDecodeError):
                        item[key] = json.loads(item[key])
            output.append(item)
        return output

    def event_rows(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            with contextlib.suppress(json.JSONDecodeError):
                item["payload_json"] = json.loads(item["payload_json"])
            output.append(item)
        return output

    def summary(self, run_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            run = connection.execute(
                "SELECT * FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            counts = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM tasks WHERE run_id=? GROUP BY status
                """,
                (run_id,),
            ).fetchall()
        if run is None:
            raise ResilientExperimentError(f"run inconnu : {run_id}")
        return {
            "run": dict(run),
            "task_status_counts": {row["status"]: int(row["count"]) for row in counts},
            "tasks": self.task_rows(run_id),
        }


def build_artifact_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def promote_attempt(
    *,
    attempt_dir: Path,
    final_dir: Path,
    required_files: Sequence[str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Promeut une tentative après validation complète.

    Aucun dossier final incomplet n'est publié. Un dossier final existant n'est
    jamais écrasé : son manifeste doit correspondre ou l'opérateur intervient.
    """
    attempt_dir = attempt_dir.resolve()
    final_dir = final_dir.resolve()
    if not attempt_dir.is_dir():
        raise ArtifactPromotionError(f"tentative absente : {attempt_dir}")

    missing = [name for name in required_files if not (attempt_dir / name).is_file()]
    if missing:
        raise ArtifactPromotionError(f"artefacts requis absents : {missing}")

    manifest = {
        "metadata": dict(metadata),
        "files": build_artifact_manifest(attempt_dir),
    }
    manifest_hash = stable_hash(manifest)
    atomic_write_json(
        attempt_dir / "PROMOTION_MANIFEST.json",
        {**manifest, "manifest_hash": manifest_hash},
    )

    if final_dir.exists():
        existing = final_dir / "PROMOTION_MANIFEST.json"
        if not existing.is_file():
            raise ArtifactPromotionError(
                f"destination déjà présente sans manifeste : {final_dir}"
            )
        existing_manifest = json.loads(existing.read_text(encoding="utf-8"))
        if existing_manifest.get("manifest_hash") != manifest_hash:
            raise ArtifactPromotionError(
                f"destination déjà présente avec un autre manifeste : {final_dir}"
            )
        return existing_manifest

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(attempt_dir, final_dir)
    dir_fd = os.open(final_dir.parent, os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return json.loads(
        (final_dir / "PROMOTION_MANIFEST.json").read_text(encoding="utf-8")
    )
