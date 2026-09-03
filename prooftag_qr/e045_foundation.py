"""E045 — fondation de données, provenance et reprise du générateur adaptatif.

E045 ne lance ni diffusion, ni SR-MPGD, ni entraînement. Il transforme les
artefacts réellement présents sous /data en une base auditée et prépare le
contrat de reprise qui sera utilisé par E046/E047.

Principes :
- aucune suppression automatique ;
- chaque phase est idempotente et transactionnelle ;
- une reprise du même commit continue les phases terminées ;
- les erreurs fichier par fichier sont isolées ;
- les OOM/quotas/contrats ne sont jamais relancés à l'identique ;
- les holdouts et expériences invalidées restent indexés mais exclus du train ;
- les scans téléphone sont séparés des scores logiciels.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import statistics
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from PIL import Image

from .e045_parameter_space import (
    PARAMETERS,
    PARAMETER_SCHEMA,
    effective_configuration,
    effective_signature,
)
from .e045_phone_labels import COLUMNS as PHONE_COLUMNS
from .e045_phone_labels import import_captures, write_template
from .e045_registry import (
    EXPERIMENTS,
    REGISTRY_SCHEMA,
    SUBEXPERIMENTS,
    by_id,
    training_policy,
)
from .resilient_experiment import (
    ArtifactPromotionError,
    ResilientTaskStore,
    atomic_write_json,
    atomic_write_text,
    build_artifact_manifest,
    classify_failure,
    promote_attempt,
    sha256_file,
    stable_hash,
    utc_now,
)

EXPERIMENT = "e045-foundation-resilience-v1"
SCHEMA_VERSION = "e045-foundation-schema-v1"

ALLOWED_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".csv",
    ".md",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".safetensors",
    ".pt",
    ".pth",
    ".joblib",
    ".pkl",
    ".npy",
    ".npz",
    ".tar",
    ".gz",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".cache",
    "cache",
    "huggingface",
    "torch",
    "node_modules",
    "__pycache__",
    ".ipynb_checkpoints",
    "lost+found",
    "tmp",
    "temp",
}

OBSERVATION_NAME_TOKENS = (
    "result",
    "comparison",
    "export",
    "generation",
    "trial",
    "observation",
    "decision",
    "score",
    "complete",
    "verdict",
    "refinement",
)

TRACE_NAME_TOKENS = (
    "trace",
    "events",
    "progress",
    "manifest",
    "checksums",
    "plan",
    "runtime",
    "audit",
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
GENERIC_ARTIFACT_ROOT = "artifacts"
GENERIC_ARTIFACT_ALWAYS_KEEP_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".csv",
    ".md",
    ".txt",
    ".safetensors",
    ".pt",
    ".pth",
    ".joblib",
    ".pkl",
    ".npy",
    ".npz",
    ".tar",
    ".gz",
    ".tar.gz",
}
GENERIC_ARTIFACT_PRIORITY_IMAGE_TOKENS = (
    "final",
    "winner",
    "selected",
    "stage1",
    "stage-1",
    "stage2",
    "stage-2",
    "srpg",
    "srmpgd",
    "sr-mpgd",
    "contact-sheet",
    "contact_sheet",
    "pipeline",
)

EXPERIMENT_PATTERN = re.compile(r"(?i)(?:^|[^a-z0-9])(e\d{3})(?:[a-z0-9_-]*)")


@dataclass(frozen=True, slots=True)
class FoundationConfig:
    data_root: Path
    output_root: Path
    source_commit: str
    worker_id: str
    max_files: int = 200_000
    max_hash_bytes: int = 64 * 1024 * 1024
    max_parse_bytes: int = 64 * 1024 * 1024
    max_depth: int = 16

    def plan(self) -> dict[str, Any]:
        return {
            "experiment": EXPERIMENT,
            "schema_version": SCHEMA_VERSION,
            "registry_schema": REGISTRY_SCHEMA,
            "parameter_schema": PARAMETER_SCHEMA,
            "data_root": str(self.data_root.resolve()),
            "max_files": self.max_files,
            "max_hash_bytes": self.max_hash_bytes,
            "max_parse_bytes": self.max_parse_bytes,
            "max_depth": self.max_depth,
            "source_commit": self.source_commit,
        }

    @property
    def plan_id(self) -> str:
        return stable_hash(self.plan())[:16]

    @property
    def plan_dir(self) -> Path:
        return self.output_root / self.plan_id


class FoundationLock:
    """Lease fichier inter-pods.

    Un lock expiré est déplacé dans quarantine au lieu d'être supprimé.
    """

    def __init__(self, path: Path, owner: str, ttl_seconds: int = 6 * 3600):
        self.path = path
        self.owner = owner
        self.ttl_seconds = ttl_seconds
        self.token = uuid.uuid4().hex

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        payload = {
            "owner": self.owner,
            "token": self.token,
            "created_at_utc": utc_now(),
            "expires_at_epoch": now + self.ttl_seconds,
        }
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError:
            existing: dict[str, Any] = {}
            with contextlib.suppress(Exception):
                existing = json.loads(self.path.read_text(encoding="utf-8"))
            expires = float(existing.get("expires_at_epoch") or 0.0)
            if expires > now:
                raise RuntimeError(
                    f"E045 déjà actif: {existing.get('owner')} jusqu'à {expires}"
                )
            quarantine = self.path.parent / "quarantine"
            quarantine.mkdir(parents=True, exist_ok=True)
            stale = quarantine / (
                f"stale-lock-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
                f"{uuid.uuid4().hex[:8]}.json"
            )
            os.replace(self.path, stale)
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )

        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

    def heartbeat(self) -> None:
        if not self.path.is_file():
            raise RuntimeError("lock E045 perdu")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("token") != self.token:
            raise RuntimeError("lock E045 remplacé par un autre worker")
        payload["heartbeat_at_utc"] = utc_now()
        payload["expires_at_epoch"] = time.time() + self.ttl_seconds
        atomic_write_json(self.path, payload)

    def release(self) -> None:
        with contextlib.suppress(FileNotFoundError, json.JSONDecodeError):
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("token") == self.token:
                self.path.unlink()

    def __enter__(self) -> "FoundationLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def infer_experiment(path: str | Path, record: Mapping[str, Any] | None = None) -> str:
    record = record or {}
    for key in ("experiment", "experiment_id", "experiment_name", "method_id"):
        value = str(record.get(key) or "")
        match = EXPERIMENT_PATTERN.search(value)
        if match:
            return match.group(1).upper()
    match = EXPERIMENT_PATTERN.search(str(path))
    return match.group(1).upper() if match else "UNKNOWN"


def _inventory_connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            path TEXT PRIMARY KEY,
            relative_path TEXT NOT NULL,
            experiment_id TEXT NOT NULL,
            extension TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            sha256 TEXT,
            pixel_sha256 TEXT,
            width INTEGER,
            height INTEGER,
            mode TEXT,
            hash_status TEXT NOT NULL,
            error TEXT,
            indexed_at_utc TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_artifacts_experiment
            ON artifacts(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_artifacts_sha
            ON artifacts(sha256);
        CREATE INDEX IF NOT EXISTS idx_artifacts_pixel
            ON artifacts(pixel_sha256);

        CREATE TABLE IF NOT EXISTS observations (
            observation_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            source_record_index INTEGER NOT NULL,
            source_record_hash TEXT NOT NULL,
            experiment_id TEXT NOT NULL,
            stage TEXT,
            method_id TEXT,
            prompt_text TEXT,
            prompt_hash TEXT,
            prompt_family TEXT,
            payload_hash TEXT,
            payload_length INTEGER,
            seed TEXT,
            configuration_json TEXT,
            effective_config_hash TEXT,
            image_path TEXT,
            image_sha256 TEXT,
            technical_status TEXT,
            technical_error TEXT,
            qr_exact_presets INTEGER,
            qr_score REAL,
            original_exact INTEGER,
            phone_attempts INTEGER,
            phone_successes INTEGER,
            clip_aesthetic REAL,
            clip_score REAL,
            hpsv2 REAL,
            lpips REAL,
            module_error REAL,
            training_policy TEXT NOT NULL,
            eligible_parameter_advisor INTEGER NOT NULL,
            eligible_phone_model INTEGER NOT NULL,
            eligible_hard_negative INTEGER NOT NULL,
            evaluation_only INTEGER NOT NULL,
            quarantine_reasons_json TEXT NOT NULL,
            extracted_at_utc TEXT NOT NULL,
            UNIQUE(source_path, source_record_index, source_record_hash)
        );

        CREATE INDEX IF NOT EXISTS idx_observation_image
            ON observations(image_sha256);
        CREATE INDEX IF NOT EXISTS idx_observation_experiment
            ON observations(experiment_id);
        """
    )
    return connection


def _path_extension(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".tar.gz"):
        return ".tar.gz"
    return path.suffix.lower()


def _pixel_hash(path: Path) -> tuple[str, int, int, str]:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{image.width}x{image.height}:".encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest(), image.width, image.height, "RGB"


def _should_prune(path: Path, output_root: Path) -> bool:
    if path.name in EXCLUDED_DIR_NAMES:
        return True
    with contextlib.suppress(ValueError):
        path.resolve().relative_to(output_root.resolve())
        return True
    return False


def _is_generic_artifact_image(path: Path, data_root: Path) -> bool:
    """True pour les rasters du dépôt générique /data/artifacts.

    Ce dossier contient historiquement des centaines de milliers d'images
    intermédiaires. Elles ne sont pas toutes utiles à un conseiller de
    paramètres et sont souvent redondantes avec les sorties canoniques sous
    notebook-runs/e0xx-*. Une image réellement référencée par une observation
    structurée est réindexée à la demande plus tard.
    """
    try:
        relative = path.resolve().relative_to(data_root.resolve())
    except ValueError:
        return False
    return (
        bool(relative.parts)
        and relative.parts[0] == GENERIC_ARTIFACT_ROOT
        and _path_extension(path) in IMAGE_EXTENSIONS
    )


def _generic_artifact_image_is_priority(path: Path) -> bool:
    lower = path.as_posix().lower()
    return any(token in lower for token in GENERIC_ARTIFACT_PRIORITY_IMAGE_TOKENS)


def _walk_relevant_files(
    config: FoundationConfig,
    *,
    selection_stats: dict[str, Any] | None = None,
) -> Iterator[Path]:
    """Parcourt les fichiers scientifiques utiles sans hasher tout le raster cache.

    La règle importante est limitée au répertoire racine ``/data/artifacts`` :
    - documents structurés, modèles, manifests et tableaux : toujours conservés ;
    - images nommées final/winner/stage1/stage2/SRPG/SR-MPGD : conservées ;
    - autres images génériques : différées et hashées seulement si une ligne
      structurée les référence.

    Les vrais répertoires d'expérience (notebook-runs, e0xx-*, parameter-search,
    etc.) conservent le comportement historique intégral.
    """
    data_root = config.data_root.resolve()
    stats = selection_stats if selection_stats is not None else {}
    stats.setdefault("discovered_allowed_files", 0)
    stats.setdefault("selected_files", 0)
    stats.setdefault("generic_artifact_images_deferred", 0)
    stats.setdefault("deferred_by_extension", {})

    for current, directories, files in os.walk(data_root):
        current_path = Path(current)
        depth = len(current_path.relative_to(data_root).parts)
        directories[:] = [
            directory
            for directory in directories
            if depth < config.max_depth
            and not _should_prune(current_path / directory, config.output_root)
        ]
        for name in files:
            path = current_path / name
            extension = _path_extension(path)
            if extension not in ALLOWED_EXTENSIONS:
                continue

            stats["discovered_allowed_files"] += 1

            if _is_generic_artifact_image(path, data_root):
                if not _generic_artifact_image_is_priority(path):
                    stats["generic_artifact_images_deferred"] += 1
                    deferred = stats["deferred_by_extension"]
                    deferred[extension] = int(deferred.get(extension, 0)) + 1
                    continue

            stats["selected_files"] += 1
            yield path


def inventory_artifacts(config: FoundationConfig, plan_dir: Path) -> dict[str, Any]:
    database = plan_dir / "foundation.sqlite"
    progress_path = plan_dir / "inventory-progress.json"
    errors = 0
    visited = 0
    hashed = 0
    images = 0
    skipped_hash = 0
    selection_stats: dict[str, Any] = {}
    started = time.time()

    connection = _inventory_connection(database)
    try:
        for path in _walk_relevant_files(config, selection_stats=selection_stats):
            visited += 1
            if visited > config.max_files:
                atomic_write_json(
                    progress_path,
                    {
                        "visited": visited,
                        "status": "max_files_exceeded",
                        "max_files": config.max_files,
                    },
                )
                raise RuntimeError(
                    f"configuration max_files insuffisante: limite max_files dépassée "
                    f"({config.max_files}); augmenter explicitement ou corriger la sélection"
                )

            try:
                stat = path.stat()
            except OSError:
                errors += 1
                continue

            existing = connection.execute(
                """
                SELECT size_bytes, mtime_ns, hash_status
                FROM artifacts WHERE path=?
                """,
                (str(path),),
            ).fetchone()
            if (
                existing
                and int(existing["size_bytes"]) == stat.st_size
                and int(existing["mtime_ns"]) == stat.st_mtime_ns
                and existing["hash_status"] in {"hashed", "metadata_only"}
            ):
                continue

            extension = _path_extension(path)
            file_hash = None
            pixel_hash = None
            width = height = None
            mode = None
            status = "metadata_only"
            error = None
            try:
                if stat.st_size <= config.max_hash_bytes or extension in IMAGE_EXTENSIONS:
                    file_hash = sha256_file(path)
                    hashed += 1
                    status = "hashed"
                else:
                    skipped_hash += 1

                if extension in IMAGE_EXTENSIONS:
                    pixel_hash, width, height, mode = _pixel_hash(path)
                    images += 1
            except Exception as exc:
                errors += 1
                status = "error"
                error = f"{type(exc).__name__}: {exc}"[:2000]

            connection.execute(
                """
                INSERT INTO artifacts(
                    path, relative_path, experiment_id, extension,
                    size_bytes, mtime_ns, sha256, pixel_sha256,
                    width, height, mode, hash_status, error, indexed_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    relative_path=excluded.relative_path,
                    experiment_id=excluded.experiment_id,
                    extension=excluded.extension,
                    size_bytes=excluded.size_bytes,
                    mtime_ns=excluded.mtime_ns,
                    sha256=excluded.sha256,
                    pixel_sha256=excluded.pixel_sha256,
                    width=excluded.width,
                    height=excluded.height,
                    mode=excluded.mode,
                    hash_status=excluded.hash_status,
                    error=excluded.error,
                    indexed_at_utc=excluded.indexed_at_utc
                """,
                (
                    str(path),
                    str(path.relative_to(config.data_root.resolve())),
                    infer_experiment(path),
                    extension,
                    stat.st_size,
                    stat.st_mtime_ns,
                    file_hash,
                    pixel_hash,
                    width,
                    height,
                    mode,
                    status,
                    error,
                    utc_now(),
                ),
            )
            if visited % 100 == 0:
                connection.commit()
            if visited % 500 == 0:
                atomic_write_json(
                    progress_path,
                    {
                        "status": "running",
                        "visited": visited,
                        "hashed": hashed,
                        "images": images,
                        "errors": errors,
                        "elapsed_s": time.time() - started,
                        "last_path": str(path),
                    },
                )
        connection.commit()

        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM artifacts ORDER BY experiment_id, relative_path"
            )
        ]
    finally:
        connection.close()

    artifact_csv = plan_dir / "artifact-inventory.csv"
    fields = [
        "path",
        "relative_path",
        "experiment_id",
        "extension",
        "size_bytes",
        "mtime_ns",
        "sha256",
        "pixel_sha256",
        "width",
        "height",
        "mode",
        "hash_status",
        "error",
        "indexed_at_utc",
    ]
    with artifact_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    selection_summary = {
        **selection_stats,
        "policy": "defer_generic_artifact_rasters_unless_priority_or_referenced",
        "generic_artifact_root": str(config.data_root.resolve() / GENERIC_ARTIFACT_ROOT),
        "note": (
            "Les rasters génériques différés ne sont pas perdus : "
            "une observation structurée qui référence une image la réindexe à la demande."
        ),
    }
    atomic_write_json(plan_dir / "inventory-selection-summary.json", selection_summary)

    summary = {
        "artifact_count": len(rows),
        "visited_relevant_files": visited,
        "hashed_file_count": hashed,
        "image_count": images,
        "hash_skipped_large_file_count": skipped_hash,
        "error_count": errors,
        "generic_artifact_images_deferred": int(
            selection_stats.get("generic_artifact_images_deferred", 0)
        ),
        "elapsed_s": time.time() - started,
        "database": str(database),
    }
    atomic_write_json(plan_dir / "artifact-inventory-summary.json", summary)
    atomic_write_json(progress_path, {"status": "complete", **summary})
    return summary


def _first(record: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for alias in aliases:
        value = record.get(alias)
        if value is not None and str(value).strip() not in {"", "nan", "None"}:
            return value
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "oui", "pass", "passed", "success", "accepted"}:
        return True
    if text in {"0", "false", "no", "non", "fail", "failed", "rejected"}:
        return False
    return None


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    with contextlib.suppress(json.JSONDecodeError):
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _candidate_observation_file(path: Path, size: int, max_parse_bytes: int) -> bool:
    if size <= 0 or size > max_parse_bytes:
        return False
    lower = path.name.lower()
    if any(token in lower for token in TRACE_NAME_TOKENS):
        return False
    if path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
        return False
    if any(token in lower for token in OBSERVATION_NAME_TOKENS):
        return True
    return "exports" in {part.lower() for part in path.parts}


def _iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                yield index, dict(row)
        return
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                text = line.strip()
                if not text:
                    continue
                parsed = json.loads(text)
                if isinstance(parsed, Mapping):
                    yield index, dict(parsed)
        return

    parsed = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(parsed, list):
        for index, row in enumerate(parsed):
            if isinstance(row, Mapping):
                yield index, dict(row)
        return
    if not isinstance(parsed, Mapping):
        return

    for key in (
        "rows",
        "results",
        "records",
        "observations",
        "trials",
        "generations",
        "decisions",
        "items",
    ):
        value = parsed.get(key)
        if isinstance(value, list):
            for index, row in enumerate(value):
                if isinstance(row, Mapping):
                    yield index, dict(row)
            return

    # verdict/COMPLETE/scorecard single-record documents.
    yield 0, dict(parsed)


def _looks_like_observation(record: Mapping[str, Any]) -> bool:
    keys = {str(key).lower() for key in record}
    signals = {
        "prompt",
        "prompt_text",
        "seed",
        "method",
        "method_id",
        "recipe",
        "variant",
        "image_path",
        "image_sha256",
        "quality_qr_verify_any_exact",
        "qr_verify_exact_presets",
        "winner_ssr_exact_presets",
        "scan_pass_rate",
        "clip_aesthetic",
        "hpsv2_1",
        "status",
    }
    return bool(keys & signals)


def _infer_stage(method: str, record: Mapping[str, Any]) -> str | None:
    explicit = str(
        _first(record, ("stage", "pipeline_stage", "output_variant", "kind")) or ""
    ).lower()
    combined = f"{method} {explicit}".lower()
    if "srmpgd" in combined or "sr-mpgd" in combined:
        return "srmpgd"
    if "stage2" in combined or "srpg" in combined:
        return "stage2"
    if "stage1" in combined or "controlnet" in combined or "raw" in combined:
        return "stage1"
    if "qr_reference" in combined or "binary" in combined:
        return "qr"
    return explicit or None


def _configuration(record: Mapping[str, Any]) -> dict[str, Any]:
    configuration: dict[str, Any] = {}
    for key in (
        "method_configuration_json",
        "configuration_json",
        "config_json",
        "settings_json",
        "generation_configuration_json",
        "tools_json",
        "model_json",
    ):
        configuration.update(_parse_json_object(record.get(key)))

    for key in (
        "backend",
        "model",
        "model_id",
        "model_revision",
        "controlnet_model",
        "controlnet_scale",
        "steps",
        "guidance_scale",
        "strength",
        "seed",
        "error_correction",
        "qr_version",
        "qr_mask_pattern",
        "qr_module_size",
        "qr_padding_px",
        "srpg_steps",
        "srpg_strength",
        "srpg_controlnet_scale",
        "srpg_qr_weight",
        "srpg_perceptual_weight",
        "srmpgd_gamma",
        "srmpgd_iterations",
        "srmpgd_lpips_weight",
        "srmpgd_latent_radius_rms",
        "output_variant",
        "reuse_stage1",
    ):
        value = record.get(key)
        if value is not None and str(value).strip() not in {"", "nan", "None"}:
            configuration[key] = value
    return effective_configuration(configuration)


def _resolve_image_path(raw: Any, source_path: Path) -> str | None:
    if raw is None or not str(raw).strip():
        return None
    candidate = Path(str(raw))
    if candidate.is_file():
        return str(candidate)
    if not candidate.is_absolute():
        relative = (source_path.parent / candidate).resolve()
        if relative.is_file():
            return str(relative)

    # E044 promotes attempts atomically to prompts/<prompt-id>. Repair stale paths.
    parts = candidate.parts
    if "attempts" in parts:
        index = parts.index("attempts")
        suffix = parts[index + 2 :] if len(parts) > index + 2 else ()
        prompt_id = None
        if len(parts) > index + 1:
            attempt_name = parts[index + 1]
            prompt_id = attempt_name.split("-")[0]
        root = Path(*parts[:index])
        if prompt_id and suffix:
            repaired = root / "prompts" / prompt_id / Path(*suffix)
            if repaired.is_file():
                return str(repaired)
    return str(candidate)


def _ensure_referenced_artifact(
    connection: sqlite3.Connection,
    *,
    path: str | None,
    data_root: Path,
) -> str | None:
    """Retourne le hash d'un artefact et l'indexe à la demande si nécessaire.

    Cette fonction est la garantie qui permet de différer les centaines de
    milliers de PNG de ``/data/artifacts`` sans perdre une image réellement
    utilisée par un CSV/JSON historique.
    """
    if not path:
        return None
    candidate = Path(path)
    row = connection.execute(
        "SELECT pixel_sha256, sha256 FROM artifacts WHERE path=?",
        (str(candidate),),
    ).fetchone()
    if row is not None:
        return row["pixel_sha256"] or row["sha256"]
    if not candidate.is_file():
        return None

    try:
        stat = candidate.stat()
        extension = _path_extension(candidate)
        file_hash = sha256_file(candidate)
        pixel_hash = None
        width = height = None
        mode = None
        if extension in IMAGE_EXTENSIONS:
            pixel_hash, width, height, mode = _pixel_hash(candidate)

        try:
            relative_path = str(candidate.resolve().relative_to(data_root.resolve()))
        except ValueError:
            relative_path = str(candidate)

        connection.execute(
            """
            INSERT OR REPLACE INTO artifacts(
                path, relative_path, experiment_id, extension,
                size_bytes, mtime_ns, sha256, pixel_sha256,
                width, height, mode, hash_status, error, indexed_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'hashed', NULL, ?)
            """,
            (
                str(candidate),
                relative_path,
                infer_experiment(candidate),
                extension,
                stat.st_size,
                stat.st_mtime_ns,
                file_hash,
                pixel_hash,
                width,
                height,
                mode,
                utc_now(),
            ),
        )
        return pixel_hash or file_hash
    except Exception:
        # L'erreur technique de la ligne reste visible par image_path; elle ne
        # doit pas arrêter l'extraction de toutes les autres observations.
        return None


def _canonical_observation(
    *,
    source_path: Path,
    record_index: int,
    record: Mapping[str, Any],
    inventory: sqlite3.Connection,
) -> dict[str, Any]:
    experiment_id = infer_experiment(source_path, record)
    policy = training_policy(experiment_id)
    method = str(
        _first(
            record,
            (
                "method_id",
                "method",
                "recipe",
                "variant",
                "profile",
                "winner_variant",
                "name",
            ),
        )
        or ""
    )
    prompt = _first(record, ("prompt_text", "prompt", "positive_prompt"))
    prompt_text = str(prompt) if prompt is not None else None
    prompt_hash = (
        hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        if prompt_text
        else None
    )
    payload_hash = _first(
        record,
        (
            "payload_hash",
            "payload_sha256",
            "expected_payload_sha256",
        ),
    )
    payload_length = _to_int(_first(record, ("payload_length", "qr_payload_length")))
    seed = _first(record, ("seed", "effective_seed", "generation_seed"))
    configuration = _configuration(record)
    config_hash = effective_signature(configuration) if configuration else None

    image_path = _resolve_image_path(
        _first(
            record,
            (
                "image_path",
                "final_image_path",
                "selected_image_path",
                "artifact_path",
                "png_path",
                "output_path",
            ),
        ),
        source_path,
    )
    image_hash = _first(
        record,
        (
            "image_sha256",
            "pixel_sha256",
            "final_image_sha256",
            "raster_sha256",
            "selected_image_sha256",
        ),
    )
    image_hash = str(image_hash).lower() if image_hash else None
    if image_hash is None:
        image_hash = _ensure_referenced_artifact(
            inventory,
            path=image_path,
            data_root=Path("/data") if str(source_path).startswith("/data/") else source_path.parent,
        )

    exact_presets = _to_int(
        _first(
            record,
            (
                "qr_verify_exact_presets",
                "conservative_exact_presets",
                "winner_ssr_exact_presets",
                "quality_qr_verify_exact_presets",
                "qrverify_exact_presets",
            ),
        )
    )
    qr_score = _to_float(
        _first(
            record,
            (
                "qr_score",
                "ssr",
                "quality_qr_verify_tolerance_score",
                "conservative_tolerance",
                "scan_pass_rate",
                "winner_ssr",
            ),
        )
    )
    if qr_score is None and exact_presets is not None:
        qr_score = exact_presets / 37.0

    original_exact = _to_bool(
        _first(
            record,
            (
                "original_exact",
                "winner_original_exact",
                "quality_qr_verify_original_exact",
                "payload_exact",
            ),
        )
    )
    phone_attempts = _to_int(
        _first(
            record,
            (
                "phone_scan_attempts",
                "physical_scan_attempts",
                "scan_attempts",
            ),
        )
    )
    phone_successes = _to_int(
        _first(
            record,
            (
                "phone_scan_successes",
                "physical_scan_successes",
                "scan_successes",
            ),
        )
    )
    status = str(
        _first(record, ("status", "technical_status", "state", "result_status"))
        or "unknown"
    )
    technical_error = _first(
        record,
        ("error", "error_message", "technical_error", "failure_reason"),
    )
    failure_text = f"{status} {technical_error or ''}".lower()
    technical_failure = any(
        token in failure_text
        for token in ("error", "failed", "exception", "oom", "timeout", "cancel")
    )

    quarantine: list[str] = []
    if policy == "quarantine":
        quarantine.append("experiment_registry_quarantine")
    if experiment_id == "UNKNOWN":
        quarantine.append("unknown_experiment")
    if image_hash and (
        len(image_hash) != 64
        or any(character not in "0123456789abcdef" for character in image_hash)
    ):
        quarantine.append("invalid_image_hash")
        image_hash = None

    software_label = exact_presets is not None or qr_score is not None or original_exact is not None
    train_policies = {
        "training_candidate",
        "training_candidate_with_audit",
        "training_candidate_software_only",
        "training_candidate_physical",
    }
    eligible_advisor = (
        policy in train_policies
        and not technical_failure
        and software_label
        and prompt_text is not None
        and config_hash is not None
        and not quarantine
    )
    eligible_phone = (
        policy not in {"quarantine", "evaluation_only"}
        and image_hash is not None
        and phone_attempts is not None
        and phone_attempts > 0
        and phone_successes is not None
        and not quarantine
    )
    hard_negative = (
        policy == "hard_negative_only"
        and software_label
        and not technical_failure
    )
    evaluation_only = policy in {"evaluation_only", "methodology_only"}

    record_hash = stable_hash(record)
    observation_id = hashlib.sha256(
        f"{source_path}:{record_index}:{record_hash}".encode("utf-8")
    ).hexdigest()
    return {
        "observation_id": observation_id,
        "source_path": str(source_path),
        "source_record_index": record_index,
        "source_record_hash": record_hash,
        "experiment_id": experiment_id,
        "stage": _infer_stage(method, record),
        "method_id": method or None,
        "prompt_text": prompt_text,
        "prompt_hash": prompt_hash,
        "prompt_family": _first(record, ("prompt_family", "family")),
        "payload_hash": str(payload_hash).lower() if payload_hash else None,
        "payload_length": payload_length,
        "seed": str(seed) if seed is not None else None,
        "configuration_json": json.dumps(
            configuration,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if configuration
        else None,
        "effective_config_hash": config_hash,
        "image_path": image_path,
        "image_sha256": image_hash,
        "technical_status": status,
        "technical_error": str(technical_error)[:4000] if technical_error else None,
        "qr_exact_presets": exact_presets,
        "qr_score": qr_score,
        "original_exact": (
            int(original_exact) if original_exact is not None else None
        ),
        "phone_attempts": phone_attempts,
        "phone_successes": phone_successes,
        "clip_aesthetic": _to_float(
            _first(record, ("clip_aesthetic", "clip_aes", "quality_clip_aesthetic"))
        ),
        "clip_score": _to_float(
            _first(record, ("clip_score", "quality_clip_score"))
        ),
        "hpsv2": _to_float(
            _first(record, ("hpsv2_1", "hpsv2", "quality_hpsv2_1"))
        ),
        "lpips": _to_float(_first(record, ("lpips", "lpips_loss"))),
        "module_error": _to_float(
            _first(
                record,
                (
                    "module_error_rate",
                    "mer",
                    "full_module_error_rate",
                    "quality_module_error_rate",
                ),
            )
        ),
        "training_policy": policy,
        "eligible_parameter_advisor": int(eligible_advisor),
        "eligible_phone_model": int(eligible_phone),
        "eligible_hard_negative": int(hard_negative),
        "evaluation_only": int(evaluation_only),
        "quarantine_reasons_json": json.dumps(quarantine, ensure_ascii=False),
        "extracted_at_utc": utc_now(),
    }


def extract_observations(config: FoundationConfig, plan_dir: Path) -> dict[str, Any]:
    database = plan_dir / "foundation.sqlite"
    connection = _inventory_connection(database)
    files = connection.execute(
        """
        SELECT path, size_bytes FROM artifacts
        WHERE extension IN ('.csv', '.json', '.jsonl')
          AND hash_status != 'error'
        ORDER BY path
        """
    ).fetchall()

    source_files = 0
    parsed_records = 0
    accepted_records = 0
    parse_errors: list[dict[str, Any]] = []

    columns = [
        "observation_id",
        "source_path",
        "source_record_index",
        "source_record_hash",
        "experiment_id",
        "stage",
        "method_id",
        "prompt_text",
        "prompt_hash",
        "prompt_family",
        "payload_hash",
        "payload_length",
        "seed",
        "configuration_json",
        "effective_config_hash",
        "image_path",
        "image_sha256",
        "technical_status",
        "technical_error",
        "qr_exact_presets",
        "qr_score",
        "original_exact",
        "phone_attempts",
        "phone_successes",
        "clip_aesthetic",
        "clip_score",
        "hpsv2",
        "lpips",
        "module_error",
        "training_policy",
        "eligible_parameter_advisor",
        "eligible_phone_model",
        "eligible_hard_negative",
        "evaluation_only",
        "quarantine_reasons_json",
        "extracted_at_utc",
    ]

    try:
        for item in files:
            path = Path(item["path"])
            if not _candidate_observation_file(
                path,
                int(item["size_bytes"]),
                config.max_parse_bytes,
            ):
                continue
            source_files += 1
            try:
                for record_index, record in _iter_records(path):
                    parsed_records += 1
                    if not _looks_like_observation(record):
                        continue
                    observation = _canonical_observation(
                        source_path=path,
                        record_index=record_index,
                        record=record,
                        inventory=connection,
                    )
                    placeholders = ",".join("?" for _ in columns)
                    cursor = connection.execute(
                        f"""
                        INSERT OR IGNORE INTO observations({','.join(columns)})
                        VALUES ({placeholders})
                        """,
                        tuple(observation[column] for column in columns),
                    )
                    if cursor.rowcount == 1:
                        accepted_records += 1
                    if parsed_records % 500 == 0:
                        connection.commit()
            except Exception as exc:
                parse_errors.append(
                    {
                        "source_path": str(path),
                        "error": f"{type(exc).__name__}: {exc}"[:4000],
                    }
                )
        connection.commit()
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM observations ORDER BY experiment_id, source_path, source_record_index"
            )
        ]
    finally:
        connection.close()

    jsonl_path = plan_dir / "canonical-observations.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    csv_path = plan_dir / "canonical-observations.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    atomic_write_json(plan_dir / "observation-parse-errors.json", parse_errors)
    summary = {
        "candidate_source_file_count": source_files,
        "parsed_record_count": parsed_records,
        "canonical_observation_count": len(rows),
        "new_insert_count": accepted_records,
        "parse_error_count": len(parse_errors),
        "eligible_parameter_advisor_count": sum(
            int(row["eligible_parameter_advisor"]) for row in rows
        ),
        "eligible_phone_model_count": sum(
            int(row["eligible_phone_model"]) for row in rows
        ),
        "hard_negative_count": sum(
            int(row["eligible_hard_negative"]) for row in rows
        ),
        "evaluation_only_count": sum(int(row["evaluation_only"]) for row in rows),
    }
    atomic_write_json(plan_dir / "canonical-observations-summary.json", summary)
    return summary


def analyze_duplicates(plan_dir: Path) -> dict[str, Any]:
    connection = _inventory_connection(plan_dir / "foundation.sqlite")
    try:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM observations
                WHERE image_sha256 IS NOT NULL AND image_sha256 != ''
                ORDER BY image_sha256, observation_id
                """
            )
        ]
    finally:
        connection.close()

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_sha256"])].append(row)

    duplicate_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    noop_rows: list[dict[str, Any]] = []
    conflicted_hashes: set[str] = set()

    for image_hash, group in groups.items():
        if len(group) > 1:
            duplicate_rows.append(
                {
                    "image_sha256": image_hash,
                    "observation_count": len(group),
                    "experiment_ids": "|".join(sorted({str(row["experiment_id"]) for row in group})),
                    "methods": "|".join(sorted({str(row["method_id"]) for row in group if row["method_id"]})),
                    "source_paths": "|".join(sorted({str(row["source_path"]) for row in group})),
                }
            )

        label_sets = {
            "qr_exact_presets": {row["qr_exact_presets"] for row in group if row["qr_exact_presets"] is not None},
            "qr_score": {round(float(row["qr_score"]), 8) for row in group if row["qr_score"] is not None},
            "original_exact": {row["original_exact"] for row in group if row["original_exact"] is not None},
        }
        conflicting = {
            key: sorted(values)
            for key, values in label_sets.items()
            if len(values) > 1
        }
        if conflicting:
            conflicted_hashes.add(image_hash)
            conflict_rows.append(
                {
                    "image_sha256": image_hash,
                    "observation_count": len(group),
                    "conflicts_json": json.dumps(conflicting, ensure_ascii=False, sort_keys=True),
                    "experiments": "|".join(sorted({str(row["experiment_id"]) for row in group})),
                }
            )

        stages = {str(row["stage"] or "") for row in group}
        if "srmpgd" in stages and ("stage2" in stages or len(group) > 1):
            noop_rows.append(
                {
                    "image_sha256": image_hash,
                    "stages": "|".join(sorted(stages)),
                    "methods": "|".join(sorted({str(row["method_id"]) for row in group if row["method_id"]})),
                    "observation_count": len(group),
                    "interpretation": "SR-MPGD pixel-identique à un autre état; étiqueter no-op/Stage2.",
                }
            )

    def write_rows(name: str, values: list[dict[str, Any]]) -> None:
        path = plan_dir / name
        fields = sorted({key for row in values for key in row})
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or ["empty"])
            writer.writeheader()
            if values:
                writer.writerows(values)

    write_rows("duplicate-images.csv", duplicate_rows)
    write_rows("label-conflicts.csv", conflict_rows)
    write_rows("srmpgd-noop-images.csv", noop_rows)
    atomic_write_json(
        plan_dir / "conflicted-image-hashes.json",
        sorted(conflicted_hashes),
    )

    summary = {
        "image_hash_group_count": len(groups),
        "duplicate_group_count": len(duplicate_rows),
        "conflicting_label_group_count": len(conflict_rows),
        "srmpgd_noop_group_count": len(noop_rows),
    }
    atomic_write_json(plan_dir / "deduplication-summary.json", summary)
    return summary


def export_registry_and_parameter_space(plan_dir: Path) -> dict[str, Any]:
    atomic_write_json(
        plan_dir / "experiment-registry.json",
        {
            "schema": REGISTRY_SCHEMA,
            "experiments": list(EXPERIMENTS),
            "subexperiments": list(SUBEXPERIMENTS),
        },
    )
    atomic_write_json(
        plan_dir / "parameter-space.json",
        {
            "schema": PARAMETER_SCHEMA,
            "parameter_count": len(PARAMETERS),
            "parameters": list(PARAMETERS),
        },
    )
    return {
        "experiment_count": len(EXPERIMENTS),
        "subexperiment_count": len(SUBEXPERIMENTS),
        "parameter_count": len(PARAMETERS),
    }


def process_phone_labels(config: FoundationConfig, plan_dir: Path) -> dict[str, Any]:
    inputs = plan_dir / "inputs"
    template = inputs / "phone-captures.csv"
    write_template(template)

    candidates = [
        config.data_root / "e045-phone-captures.csv",
        config.data_root / "phone-captures.csv",
        template,
    ]
    selected = next(
        (
            path
            for path in candidates
            if path.is_file() and path.stat().st_size > 256
        ),
        None,
    )
    if selected is None:
        summary = {
            "status": "no_phone_capture_rows",
            "template": str(template),
            "valid_capture_count": 0,
            "rejected_capture_count": 0,
            "labeled_image_count": 0,
            "truth_source": "physical_phone_pending",
        }
        atomic_write_json(plan_dir / "phone-labels/phone-label-summary.json", summary)
        for name, fields in (
            ("phone-captures-valid.csv", PHONE_COLUMNS),
            ("phone-captures-rejected.csv", ("line_number", "rejection_reasons")),
            ("phone-labels-by-image.csv", ("image_sha256", "attempts", "successes")),
            ("phone-labels-by-device.csv", ("image_sha256", "device_id", "attempts", "successes")),
        ):
            path = plan_dir / "phone-labels" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(fields)
        return summary
    return import_captures(selected, plan_dir / "phone-labels")


def run_resilience_selftest(plan_dir: Path) -> dict[str, Any]:
    root = plan_dir / "resilience-selftest"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    store = ResilientTaskStore(root / "state.sqlite")
    run_id = "selftest"
    store.register_run(
        run_id=run_id,
        plan={"kind": "resilience-selftest-v1"},
        source_commit="0" * 40,
    )
    store.register_tasks(
        run_id=run_id,
        tasks=[
            {
                "task_id": "generation-transient",
                "kind": "generation",
                "spec": {"seed": 1, "prompt": "selftest"},
                "max_attempts": 2,
                "priority": 10,
            },
            {
                "task_id": "training-oom",
                "kind": "training",
                "spec": {"batch_size": 32},
                "max_attempts": 3,
                "priority": 20,
            },
            {
                "task_id": "generation-stale",
                "kind": "generation",
                "spec": {"seed": 2, "prompt": "stale"},
                "max_attempts": 2,
                "priority": 30,
            },
        ],
    )

    transient = store.claim_next(run_id=run_id, worker_id="worker-a", lease_seconds=60)
    assert transient and transient.task_id == "generation-transient"
    decision_transient = store.fail(
        transient,
        TimeoutError("temporary timeout"),
        retry_delay_seconds=0,
    )
    transient_retry = store.claim_next(
        run_id=run_id,
        worker_id="worker-b",
        lease_seconds=60,
    )
    assert transient_retry and transient_retry.task_id == "generation-transient"
    store.complete(transient_retry, result={"artifact": "ok.png"})

    oom = store.claim_next(run_id=run_id, worker_id="worker-a", lease_seconds=60)
    assert oom and oom.task_id == "training-oom"
    decision_oom = store.fail(
        oom,
        RuntimeError("CUDA out of memory while allocating tensor"),
    )

    stale = store.claim_next(run_id=run_id, worker_id="worker-stale", lease_seconds=0)
    assert stale and stale.task_id == "generation-stale"
    recovered = store.recover_stale(run_id=run_id, force=True)
    stale_retry = store.claim_next(
        run_id=run_id,
        worker_id="worker-recovery",
        lease_seconds=60,
    )
    assert stale_retry and stale_retry.task_id == "generation-stale"
    store.complete(stale_retry, result={"resumed": True})

    attempt = root / "attempts" / "valid"
    attempt.mkdir(parents=True)
    atomic_write_text(attempt / "checkpoint.json", '{"epoch": 3}\n')
    atomic_write_text(attempt / "result.json", '{"score": 1}\n')
    promoted = promote_attempt(
        attempt_dir=attempt,
        final_dir=root / "final" / "valid",
        required_files=("checkpoint.json", "result.json"),
        metadata={"task": "training"},
    )

    invalid_attempt = root / "attempts" / "invalid"
    invalid_attempt.mkdir(parents=True)
    atomic_write_text(invalid_attempt / "checkpoint.json", "{}\n")
    invalid_rejected = False
    try:
        promote_attempt(
            attempt_dir=invalid_attempt,
            final_dir=root / "final" / "invalid",
            required_files=("checkpoint.json", "result.json"),
            metadata={"task": "training"},
        )
    except ArtifactPromotionError:
        invalid_rejected = True

    summary = {
        "transient_classified_retryable": decision_transient.retryable,
        "transient_completed_on_attempt": transient_retry.attempt_no,
        "oom_kind": decision_oom.kind,
        "oom_retryable": decision_oom.retryable,
        "oom_operator_action_required": decision_oom.operator_action_required,
        "stale_recovered_tasks": recovered,
        "invalid_promotion_rejected": invalid_rejected,
        "valid_promotion_manifest_hash": promoted["manifest_hash"],
        "task_summary": store.summary(run_id),
        "passed": (
            decision_transient.retryable
            and transient_retry.attempt_no == 2
            and decision_oom.kind == "resource"
            and not decision_oom.retryable
            and invalid_rejected
            and "generation-stale" in recovered
        ),
    }
    atomic_write_json(root / "selftest-result.json", summary)
    if not summary["passed"]:
        raise AssertionError("selftest de reprise E045 échoué")
    return {
        "passed": True,
        "transient_retry_attempt": transient_retry.attempt_no,
        "oom_blocked": True,
        "stale_recovered": True,
        "invalid_promotion_rejected": True,
    }


def _csv_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def finalize_foundation(config: FoundationConfig, plan_dir: Path) -> dict[str, Any]:
    inventory = json.loads(
        (plan_dir / "artifact-inventory-summary.json").read_text(encoding="utf-8")
    )
    observations = json.loads(
        (plan_dir / "canonical-observations-summary.json").read_text(encoding="utf-8")
    )
    deduplication = json.loads(
        (plan_dir / "deduplication-summary.json").read_text(encoding="utf-8")
    )
    phone = json.loads(
        (plan_dir / "phone-labels/phone-label-summary.json").read_text(encoding="utf-8")
    )
    selftest = json.loads(
        (plan_dir / "resilience-selftest/selftest-result.json").read_text(encoding="utf-8")
    )

    summary = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "plan_id": config.plan_id,
        "source_commit": config.source_commit,
        "created_at_utc": utc_now(),
        "registry": {
            "experiment_count": len(EXPERIMENTS),
            "parameter_count": len(PARAMETERS),
        },
        "inventory": inventory,
        "observations": observations,
        "deduplication": deduplication,
        "phone_labels": phone,
        "resilience_selftest_passed": bool(selftest.get("passed")),
        "physical_truth_available": int(phone.get("valid_capture_count", 0)) > 0,
        "advisor_training_authorized": False,
        "phone_surrogate_training_authorized": False,
        "generation_campaign_authorized": False,
        "production_ready": False,
        "next_action": (
            "IMPORT_PHONE_LABELS_AND_REVIEW_QUARANTINE_BEFORE_E046"
            if int(phone.get("valid_capture_count", 0)) == 0
            else "REVIEW_DATA_CARD_AND_FREEZE_E046_DEVELOPMENT_SPLIT"
        ),
    }
    atomic_write_json(plan_dir / "summary.json", summary)

    data_card = {
        "schema": "e045-data-card-v1",
        "observation_count": observations["canonical_observation_count"],
        "advisor_eligible": observations["eligible_parameter_advisor_count"],
        "phone_model_eligible": observations["eligible_phone_model_count"],
        "hard_negatives": observations["hard_negative_count"],
        "evaluation_only": observations["evaluation_only_count"],
        "duplicate_groups": deduplication["duplicate_group_count"],
        "label_conflicts": deduplication["conflicting_label_group_count"],
        "known_limitations": [
            "Les schémas historiques sont hétérogènes; le record canonique conserve le chemin source.",
            "Les jeux E021/E022/E027-E031 restent evaluation_only.",
            "Le premier surrogate E016 est en quarantaine.",
            "Un score QR-Verify ne devient jamais un label téléphone.",
            "E044 reste logiciel tant que les captures physiques ne sont pas importées.",
            "Les fichiers volumineux peuvent être indexés sans SHA si max_hash_bytes est dépassé.",
        ],
        "split_policy": {
            "group_keys": [
                "prompt_hash",
                "payload_hash",
                "image_sha256",
                "effective_config_hash",
            ],
            "forbid_same_prompt_across_train_test": True,
            "forbid_same_pixel_hash_across_splits": True,
            "holdouts_never_train": True,
        },
    }
    atomic_write_json(plan_dir / "data-card.json", data_card)

    recovery_runbook = {
        "schema": "e045-recovery-runbook-v1",
        "same_commit_crash": [
            "Relancer exactement bash scripts/run-e045-foundation.sh.",
            "Les tâches succeeded restent ignorées.",
            "Un lease expiré redevient pending; les événements restent dans state.sqlite.",
        ],
        "generation_future": [
            "Une tentative écrit dans attempts/<task>/<attempt>.",
            "Le résultat n'est promu qu'après présence des fichiers requis et manifeste.",
            "Timeout/503 peuvent être repris au plus max_attempts.",
            "OOM, disque plein, checksum, schéma et payload mismatch restent blocked.",
            "Une configuration modifiée reçoit un nouveau spec_hash/task_id.",
        ],
        "training_future": [
            "Checkpoint temporaire puis os.replace atomique.",
            "État optimiseur, scheduler, scaler, RNG et epoch/step obligatoires.",
            "Reprise uniquement si dataset hash, split hash, code commit et architecture correspondent.",
            "OOM ne reprend pas avec le même batch; créer une nouvelle spécification.",
            "Le meilleur checkpoint et le dernier checkpoint sont tous deux conservés.",
        ],
        "forbidden": [
            "rm -rf automatique d'un résultat partiel",
            "relance infinie",
            "ignorer les erreurs techniques du dénominateur",
            "réutiliser un holdout pour entraîner puis le présenter comme test",
            "publier un no-op SR-MPGD comme image distincte",
        ],
    }
    atomic_write_json(plan_dir / "recovery-runbook.json", recovery_runbook)

    report = f"""# E045 — fondation de données et reprise

## Résumé

- plan : `{config.plan_id}`
- commit : `{config.source_commit}`
- expériences documentées : **{len(EXPERIMENTS)}**
- paramètres canoniques : **{len(PARAMETERS)}**
- artefacts indexés : **{inventory['artifact_count']}**
- observations canoniques : **{observations['canonical_observation_count']}**
- observations candidates conseiller : **{observations['eligible_parameter_advisor_count']}**
- observations avec labels téléphone : **{observations['eligible_phone_model_count']}**
- groupes de doublons : **{deduplication['duplicate_group_count']}**
- conflits de labels : **{deduplication['conflicting_label_group_count']}**
- selftest de reprise : **{'PASS' if selftest.get('passed') else 'FAIL'}**

## Décision

E045 ne donne aucune autorisation d'entraînement ou de génération massive.
La prochaine campagne ne doit commencer qu'après :

1. revue des conflits et de la quarantaine ;
2. import de captures téléphone physiques ;
3. gel des splits de développement et de holdout ;
4. définition d'un budget de reprise et de ressources par type de tâche.

## Résilience

Les générations et entraînements futurs utilisent `ResilientTaskStore` :
leases, heartbeats, reprises bornées, classification des erreurs et promotion
atomique des artefacts. Un OOM ou un contrat invalide est bloqué au lieu d'être
relancé à l'identique.
"""
    atomic_write_text(plan_dir / "report.md", report)

    # Les fichiers d'état SQLite, WAL, locks et COMPLETE restent mutables pendant
    # la transition finale. Le manifeste couvre uniquement les artefacts scientifiques
    # et rapports immuables; il n'est donc pas invalidé par store.complete().
    mutable_names = {
        "state.sqlite",
        "state.sqlite-wal",
        "state.sqlite-shm",
        # foundation.sqlite is a finalized scientific artifact and is included.
        "task-state.json",
        "COMPLETE.json",
        "artifact-manifest.json",
    }
    manifest = []
    for candidate in sorted(plan_dir.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.name in mutable_names or candidate.name.endswith(".tmp"):
            continue
        if candidate.name.startswith(".") and candidate.suffix == ".lock":
            continue
        manifest.append(
            {
                "path": candidate.relative_to(plan_dir).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    atomic_write_json(plan_dir / "artifact-manifest.json", manifest)
    complete = {
        **summary,
        "complete": True,
        "artifact_manifest_sha256": sha256_file(plan_dir / "artifact-manifest.json"),
    }
    atomic_write_json(plan_dir / "COMPLETE.json", complete)
    return complete


def _task_definitions(config: FoundationConfig) -> list[dict[str, Any]]:
    base = {"plan_id": config.plan_id, "schema": SCHEMA_VERSION}
    return [
        {
            "task_id": f"{config.plan_id}:010-registry",
            "kind": "foundation",
            "priority": 10,
            "max_attempts": 2,
            "spec": {**base, "phase": "registry"},
        },
        {
            "task_id": f"{config.plan_id}:020-inventory",
            "kind": "foundation",
            "priority": 20,
            "max_attempts": 2,
            "spec": {**base, "phase": "inventory"},
        },
        {
            "task_id": f"{config.plan_id}:030-observations",
            "kind": "foundation",
            "priority": 30,
            "max_attempts": 2,
            "spec": {**base, "phase": "observations"},
        },
        {
            "task_id": f"{config.plan_id}:040-deduplicate",
            "kind": "foundation",
            "priority": 40,
            "max_attempts": 2,
            "spec": {**base, "phase": "deduplicate"},
        },
        {
            "task_id": f"{config.plan_id}:050-phone-labels",
            "kind": "foundation",
            "priority": 50,
            "max_attempts": 2,
            "spec": {**base, "phase": "phone-labels"},
        },
        {
            "task_id": f"{config.plan_id}:060-resilience-selftest",
            "kind": "foundation",
            "priority": 60,
            "max_attempts": 1,
            "spec": {**base, "phase": "resilience-selftest"},
        },
        {
            "task_id": f"{config.plan_id}:070-finalize",
            "kind": "foundation",
            "priority": 70,
            "max_attempts": 2,
            "spec": {**base, "phase": "finalize"},
        },
    ]


def _run_phase(
    phase: str,
    *,
    config: FoundationConfig,
    plan_dir: Path,
) -> dict[str, Any]:
    if phase == "registry":
        return export_registry_and_parameter_space(plan_dir)
    if phase == "inventory":
        return inventory_artifacts(config, plan_dir)
    if phase == "observations":
        return extract_observations(config, plan_dir)
    if phase == "deduplicate":
        return analyze_duplicates(plan_dir)
    if phase == "phone-labels":
        return process_phone_labels(config, plan_dir)
    if phase == "resilience-selftest":
        return run_resilience_selftest(plan_dir)
    if phase == "finalize":
        return finalize_foundation(config, plan_dir)
    raise ValueError(f"phase E045 inconnue: {phase}")


def run_foundation(config: FoundationConfig, *, force_recover_stale: bool = False) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    plan_dir = config.plan_dir
    plan_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(plan_dir / "plan.json", config.plan())
    atomic_write_json(
        config.output_root / "LATEST.json",
        {
            "plan_id": config.plan_id,
            "plan_dir": str(plan_dir),
            "source_commit": config.source_commit,
            "status": "running",
            "updated_at_utc": utc_now(),
        },
    )

    lock = FoundationLock(
        config.output_root / f".{config.plan_id}.lock",
        owner=config.worker_id,
    )
    with lock:
        store = ResilientTaskStore(plan_dir / "state.sqlite")
        store.register_run(
            run_id=config.plan_id,
            plan=config.plan(),
            source_commit=config.source_commit,
            metadata={"worker_id": config.worker_id},
        )
        store.register_tasks(
            run_id=config.plan_id,
            tasks=_task_definitions(config),
        )
        store.recover_stale(
            run_id=config.plan_id,
            force=force_recover_stale,
        )

        while True:
            task = store.claim_next(
                run_id=config.plan_id,
                worker_id=config.worker_id,
                lease_seconds=6 * 3600,
            )
            if task is None:
                break
            phase = str(task.spec["phase"])
            lock.heartbeat()
            print(f"[E045] START {phase} attempt={task.attempt_no}", flush=True)
            try:
                result = _run_phase(
                    phase,
                    config=config,
                    plan_dir=plan_dir,
                )
                store.heartbeat(
                    task,
                    lease_seconds=6 * 3600,
                    telemetry=result,
                )
                store.complete(task, result=result)
                print(f"[E045] DONE {phase}: {result}", flush=True)
            except BaseException as exc:
                decision = store.fail(task, exc, retry_delay_seconds=0)
                atomic_write_json(
                    plan_dir / f"failure-{phase}-{task.attempt_no:02d}.json",
                    {
                        "phase": phase,
                        "attempt": task.attempt_no,
                        "error_class": type(exc).__name__,
                        "error": str(exc),
                        "decision": asdict(decision),
                        "timestamp_utc": utc_now(),
                    },
                )
                print(
                    f"[E045] FAIL {phase}: {type(exc).__name__}: {exc} "
                    f"kind={decision.kind} retryable={decision.retryable}",
                    flush=True,
                )
                raise

        state = store.summary(config.plan_id)
        atomic_write_json(plan_dir / "task-state.json", state)
        counts = state["task_status_counts"]
        if counts.get("blocked") or counts.get("failed"):
            raise RuntimeError(f"E045 bloqué: {counts}")
        if counts.get("pending") or counts.get("running") or counts.get("retry_wait"):
            raise RuntimeError(f"E045 incomplet: {counts}")

        complete_path = plan_dir / "COMPLETE.json"
        if not complete_path.is_file():
            raise FileNotFoundError(f"COMPLETE absent: {complete_path}")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        complete["task_status_counts"] = counts
        atomic_write_json(complete_path, complete)
        atomic_write_json(
            config.output_root / "LATEST.json",
            {
                "plan_id": config.plan_id,
                "plan_dir": str(plan_dir),
                "source_commit": config.source_commit,
                "status": "complete",
                "complete_path": str(complete_path),
                "updated_at_utc": utc_now(),
            },
        )
        return complete


def latest_plan(output_root: Path) -> Path:
    latest = output_root / "LATEST.json"
    if not latest.is_file():
        raise FileNotFoundError(f"aucun LATEST E045: {latest}")
    payload = json.loads(latest.read_text(encoding="utf-8"))
    return Path(payload["plan_dir"])


def status(output_root: Path) -> dict[str, Any]:
    plan_dir = latest_plan(output_root)
    response: dict[str, Any] = {
        "plan_dir": str(plan_dir),
        "complete": (plan_dir / "COMPLETE.json").is_file(),
    }
    if (plan_dir / "task-state.json").is_file():
        response["task_state"] = json.loads(
            (plan_dir / "task-state.json").read_text(encoding="utf-8")
        )
    elif (plan_dir / "state.sqlite").is_file() and (plan_dir / "plan.json").is_file():
        plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
        store = ResilientTaskStore(plan_dir / "state.sqlite")
        response["task_state"] = store.summary(stable_hash(plan)[:16])
    if (plan_dir / "summary.json").is_file():
        response["summary"] = json.loads(
            (plan_dir / "summary.json").read_text(encoding="utf-8")
        )
    failures = sorted(plan_dir.glob("failure-*.json"))
    response["failure_files"] = [str(path) for path in failures]
    return response


def verify_complete(output_root: Path) -> dict[str, Any]:
    plan_dir = latest_plan(output_root)
    complete = plan_dir / "COMPLETE.json"
    manifest_path = plan_dir / "artifact-manifest.json"
    if not complete.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("E045 incomplet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    mismatched: list[str] = []
    for item in manifest:
        path = plan_dir / item["path"]
        if not path.is_file():
            missing.append(item["path"])
        elif sha256_file(path) != item["sha256"]:
            mismatched.append(item["path"])
    result = {
        "plan_dir": str(plan_dir),
        "manifest_entry_count": len(manifest),
        "missing": missing,
        "mismatched": mismatched,
        "valid": not missing and not mismatched,
    }
    if not result["valid"]:
        raise RuntimeError(f"manifest E045 invalide: {result}")
    return result



def import_phone_revision(input_csv: Path, output_root: Path) -> dict[str, Any]:
    """Importe un lot physique immuable après la clôture d'un plan E045."""
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    input_sha = sha256_file(input_csv)
    import_dir = output_root / "phone-imports" / input_sha[:16]
    summary_path = import_dir / "phone-label-summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = import_captures(input_csv, import_dir)
    pointer = {
        "schema": "e045-phone-import-pointer-v1",
        "input_sha256": input_sha,
        "import_dir": str(import_dir),
        "summary_path": str(summary_path),
        "valid_capture_count": int(summary["valid_capture_count"]),
        "labeled_image_count": int(summary["labeled_image_count"]),
        "updated_at_utc": utc_now(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "PHONE_LATEST.json", pointer)
    return {**pointer, "summary": summary}


def _commit(value: str) -> str:
    value = value.strip().lower()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise argparse.ArgumentTypeError("commit Git complet de 40 caractères requis")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    run = sub.add_parser("run")
    run.add_argument("--data-root", type=Path, default=Path("/data"))
    run.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/e045-foundation-v1"),
    )
    run.add_argument("--source-commit", type=_commit, required=True)
    run.add_argument("--worker-id", default=f"e045-{os.getpid()}")
    run.add_argument("--max-files", type=int, default=200_000)
    run.add_argument("--max-hash-mb", type=int, default=64)
    run.add_argument("--max-parse-mb", type=int, default=64)
    run.add_argument("--max-depth", type=int, default=16)
    run.add_argument("--force-recover-stale", action="store_true")

    status_parser = sub.add_parser("status")
    status_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/e045-foundation-v1"),
    )

    verify = sub.add_parser("verify")
    verify.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/e045-foundation-v1"),
    )

    selftest = sub.add_parser("selftest")
    selftest.add_argument("--output-dir", type=Path, required=True)

    phone = sub.add_parser("import-phone")
    phone.add_argument("--input-csv", type=Path, required=True)
    phone.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/e045-foundation-v1"),
    )

    return parser


def _cli() -> int:
    args = _parser().parse_args()
    if args.action == "run":
        config = FoundationConfig(
            data_root=args.data_root,
            output_root=args.output_root,
            source_commit=args.source_commit,
            worker_id=args.worker_id,
            max_files=args.max_files,
            max_hash_bytes=args.max_hash_mb * 1024 * 1024,
            max_parse_bytes=args.max_parse_mb * 1024 * 1024,
            max_depth=args.max_depth,
        )
        result = run_foundation(
            config,
            force_recover_stale=args.force_recover_stale,
        )
    elif args.action == "status":
        result = status(args.output_root)
    elif args.action == "verify":
        result = verify_complete(args.output_root)
    elif args.action == "selftest":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = run_resilience_selftest(args.output_dir)
    elif args.action == "import-phone":
        result = import_phone_revision(args.input_csv, args.output_root)
    else:
        raise AssertionError(args.action)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
