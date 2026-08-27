from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

INCIDENT_MANIFEST_NAME = "incident-manifest.json"
INCIDENT_CHECKSUMS_NAME = "checksums.json"
INCIDENT_SCHEMA_VERSION = 1
_SNAPSHOT_MANIFEST_NAME = ".snapshot-manifest.json"
_VOLATILE_IDENTITY_KEYS = {
    "created_at",
    "created_at_utc",
    "timestamp",
    "updated_at",
}


@dataclass(frozen=True, slots=True)
class IncidentArchive:
    path: Path
    prefix: str
    manifest: dict[str, Any]
    reused: bool
    rejected_archives: tuple[dict[str, str], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def canonical_incident_material(value: Any) -> Any:
    """Remove wall-clock fields without discarding scientific failure evidence."""

    if isinstance(value, Mapping):
        return {
            str(key): canonical_incident_material(item)
            for key, item in value.items()
            if str(key) not in _VOLATILE_IDENTITY_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [canonical_incident_material(item) for item in value]
    return value


def atomic_copy_once(source: Path, target: Path) -> None:
    source = Path(source)
    target = Path(target)
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copy2(source, temporary)
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot_tree_once(
    source_root: Path,
    target_root: Path,
    *,
    excluded_top_level: Iterable[str] = (),
) -> list[str]:
    """Freeze all currently available regular files without following the snapshot itself."""

    source_root = Path(source_root)
    target_root = Path(target_root)
    snapshot_manifest_path = target_root / _SNAPSHOT_MANIFEST_NAME
    if snapshot_manifest_path.is_file():
        snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
        checksums = snapshot_manifest.get("artifact_checksums")
        if not isinstance(checksums, dict):
            raise RuntimeError("snapshot manifest checksum map is absent")
        expected = set(checksums)
        observed = {
            path.relative_to(target_root).as_posix()
            for path in target_root.rglob("*")
            if path.is_file() and path != snapshot_manifest_path
        }
        if observed != expected:
            raise RuntimeError("frozen incident snapshot inventory differs")
        for relative_name, expected_sha256 in checksums.items():
            source = target_root.joinpath(*_safe_relative_path(relative_name).parts)
            if sha256_file(source) != expected_sha256:
                raise RuntimeError(f"frozen incident snapshot differs: {relative_name}")
        return sorted(expected)

    excluded = {str(value) for value in excluded_top_level}
    sources = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and not path.is_relative_to(target_root)
    )
    copied: list[str] = []
    for source in sources:
        relative = source.relative_to(source_root)
        if relative.parts and relative.parts[0] in excluded:
            continue
        if source.name.endswith(".tmp"):
            continue
        target = target_root / relative
        atomic_copy_once(source, target)
        copied.append(relative.as_posix())
    snapshot_checksums = {
        relative_name: sha256_file(
            target_root.joinpath(*_safe_relative_path(relative_name).parts)
        )
        for relative_name in copied
    }
    _atomic_text(
        snapshot_manifest_path,
        json.dumps(
            {
                "schema_version": INCIDENT_SCHEMA_VERSION,
                "algorithm": "sha256",
                "artifact_checksums": snapshot_checksums,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
    )
    return copied


def _safe_relative_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(str(value))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe incident artifact path: {value!r}")
    return relative


def _artifact_files(root: Path) -> list[Path]:
    manifest_path = root / INCIDENT_MANIFEST_NAME
    checksums_path = root / INCIDENT_CHECKSUMS_NAME
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path not in {manifest_path, checksums_path}
        and not path.name.endswith(".tmp")
    )


def _identity_payload(
    *,
    kind: str,
    experiment: str,
    plan_id: str,
    identity_material: Any,
    artifact_checksums: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": INCIDENT_SCHEMA_VERSION,
        "kind": str(kind),
        "experiment": str(experiment),
        "plan_id": str(plan_id),
        "identity_material": canonical_incident_material(identity_material),
        "artifact_checksums": dict(sorted(artifact_checksums.items())),
    }


def prepare_incident_bundle(
    root: Path,
    *,
    kind: str,
    experiment: str,
    plan_id: str,
    identity_material: Any,
) -> dict[str, Any]:
    """Write a content-bound manifest for a frozen incident evidence directory."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    artifact_checksums = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in _artifact_files(root)
    }
    identity_payload = _identity_payload(
        kind=kind,
        experiment=experiment,
        plan_id=plan_id,
        identity_material=identity_material,
        artifact_checksums=artifact_checksums,
    )
    incident_identity = canonical_sha256(identity_payload)
    checksums = {
        "schema_version": INCIDENT_SCHEMA_VERSION,
        "algorithm": "sha256",
        "incident_identity_sha256": incident_identity,
        "artifact_checksums": artifact_checksums,
    }
    checksums_path = root / INCIDENT_CHECKSUMS_NAME
    _atomic_text(
        checksums_path,
        json.dumps(checksums, ensure_ascii=False, indent=2, allow_nan=False),
    )
    manifest = {
        **identity_payload,
        "created_at": datetime.now(UTC).isoformat(),
        "incident_identity_sha256": incident_identity,
        "checksums_sha256": sha256_file(checksums_path),
    }
    _atomic_text(
        root / INCIDENT_MANIFEST_NAME,
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
    )
    return manifest


def _read_member(bundle: tarfile.TarFile, name: str) -> bytes:
    member = bundle.extractfile(name)
    if member is None:
        raise RuntimeError(f"incident archive member is unreadable: {name}")
    return member.read()


def verify_incident_archive(
    path: Path,
    *,
    prefix: str,
    expected_kind: str,
    expected_experiment: str,
    expected_plan_id: str,
    expected_identity: str | None = None,
) -> dict[str, Any]:
    """Verify identity, inventory and every regular-file checksum in an incident archive."""

    path = Path(path)
    manifest_member = f"{prefix}/{INCIDENT_MANIFEST_NAME}"
    checksums_member = f"{prefix}/{INCIDENT_CHECKSUMS_NAME}"
    with tarfile.open(path, "r:gz") as bundle:
        members = bundle.getmembers()
        unsupported = [member.name for member in members if not member.isfile()]
        if unsupported:
            raise RuntimeError(
                f"incident archive contains non-regular members: {sorted(unsupported)}"
            )
        regular_names = [member.name for member in members]
        if len(regular_names) != len(set(regular_names)):
            raise RuntimeError("incident archive contains duplicate regular-file members")
        if manifest_member not in regular_names or checksums_member not in regular_names:
            raise RuntimeError("incident archive manifest or checksums are absent")
        manifest_bytes = _read_member(bundle, manifest_member)
        checksums_bytes = _read_member(bundle, checksums_member)
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        checksums = json.loads(checksums_bytes.decode("utf-8"))

        if manifest.get("schema_version") != INCIDENT_SCHEMA_VERSION:
            raise RuntimeError("unsupported incident manifest schema")
        if manifest.get("kind") != expected_kind:
            raise RuntimeError("incident archive kind differs")
        if manifest.get("experiment") != expected_experiment:
            raise RuntimeError("incident archive experiment differs")
        if manifest.get("plan_id") != expected_plan_id:
            raise RuntimeError("incident archive plan differs")

        artifact_checksums = manifest.get("artifact_checksums")
        if not isinstance(artifact_checksums, dict):
            raise RuntimeError("incident artifact checksum map is absent")
        for relative_name, expected_sha256 in artifact_checksums.items():
            _safe_relative_path(relative_name)
            if relative_name in {INCIDENT_MANIFEST_NAME, INCIDENT_CHECKSUMS_NAME}:
                raise RuntimeError(f"reserved incident artifact path: {relative_name}")
            if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                raise RuntimeError(f"invalid incident checksum: {relative_name}")

        identity_payload = _identity_payload(
            kind=manifest["kind"],
            experiment=manifest["experiment"],
            plan_id=manifest["plan_id"],
            identity_material=manifest.get("identity_material"),
            artifact_checksums=artifact_checksums,
        )
        actual_identity = canonical_sha256(identity_payload)
        if manifest.get("incident_identity_sha256") != actual_identity:
            raise RuntimeError("incident manifest identity is invalid")
        if expected_identity is not None and actual_identity != expected_identity:
            raise RuntimeError("incident archive identity differs from current evidence")
        if hashlib.sha256(checksums_bytes).hexdigest() != manifest.get("checksums_sha256"):
            raise RuntimeError("incident checksums document hash is invalid")
        if checksums != {
            "schema_version": INCIDENT_SCHEMA_VERSION,
            "algorithm": "sha256",
            "incident_identity_sha256": actual_identity,
            "artifact_checksums": artifact_checksums,
        }:
            raise RuntimeError("incident checksums document differs from manifest")

        expected_regular_names = {
            manifest_member,
            checksums_member,
            *(f"{prefix}/{name}" for name in artifact_checksums),
        }
        if set(regular_names) != expected_regular_names:
            missing = sorted(expected_regular_names - set(regular_names))
            unexpected = sorted(set(regular_names) - expected_regular_names)
            raise RuntimeError(
                f"incident archive inventory differs: missing={missing}, unexpected={unexpected}"
            )
        for relative_name, expected_sha256 in artifact_checksums.items():
            member_name = f"{prefix}/{relative_name}"
            actual_sha256 = hashlib.sha256(_read_member(bundle, member_name)).hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(f"incident checksum differs: {member_name}")
    return manifest


def _verify_incident_bundle_root(root: Path, manifest: Mapping[str, Any]) -> None:
    manifest_path = root / INCIDENT_MANIFEST_NAME
    checksums_path = root / INCIDENT_CHECKSUMS_NAME
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise RuntimeError("incident bundle manifest or checksums are absent")
    stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stored_manifest != dict(manifest):
        raise RuntimeError("incident bundle manifest changed before archive creation")
    checksums_bytes = checksums_path.read_bytes()
    if hashlib.sha256(checksums_bytes).hexdigest() != manifest.get("checksums_sha256"):
        raise RuntimeError("incident bundle checksums changed before archive creation")
    checksums = json.loads(checksums_bytes.decode("utf-8"))
    artifact_checksums = dict(manifest["artifact_checksums"])
    expected_checksums = {
        "schema_version": INCIDENT_SCHEMA_VERSION,
        "algorithm": "sha256",
        "incident_identity_sha256": manifest["incident_identity_sha256"],
        "artifact_checksums": artifact_checksums,
    }
    if checksums != expected_checksums:
        raise RuntimeError("incident bundle checksums differ from manifest")
    observed = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    expected = {
        *artifact_checksums,
        INCIDENT_MANIFEST_NAME,
        INCIDENT_CHECKSUMS_NAME,
    }
    if observed != expected:
        raise RuntimeError("incident bundle inventory changed before archive creation")
    for relative_name, expected_sha256 in artifact_checksums.items():
        if relative_name in {INCIDENT_MANIFEST_NAME, INCIDENT_CHECKSUMS_NAME}:
            raise RuntimeError(f"reserved incident artifact path: {relative_name}")
        source = root.joinpath(*_safe_relative_path(relative_name).parts)
        if sha256_file(source) != expected_sha256:
            raise RuntimeError(
                f"incident evidence changed before archive creation: {relative_name}"
            )


def _create_archive(path: Path, *, prefix: str, root: Path, manifest: Mapping[str, Any]) -> None:
    artifact_checksums = dict(manifest["artifact_checksums"])
    relative_names = sorted(
        [*artifact_checksums, INCIDENT_CHECKSUMS_NAME, INCIDENT_MANIFEST_NAME]
    )
    _verify_incident_bundle_root(root, manifest)

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        with tarfile.open(temporary, "w:gz") as bundle:
            for relative_name in relative_names:
                source = root.joinpath(*_safe_relative_path(relative_name).parts)
                bundle.add(source, arcname=f"{prefix}/{relative_name}", recursive=False)
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def resolve_incident_archive(
    primary_path: Path,
    *,
    primary_prefix: str,
    bundle_root: Path,
    manifest: Mapping[str, Any],
    maximum_recovery_attempts: int = 32,
) -> IncidentArchive:
    """Reuse a fully valid archive or create a content-addressed recovery without overwrite."""

    primary_path = Path(primary_path)
    bundle_root = Path(bundle_root)
    expected_identity = str(manifest["incident_identity_sha256"])
    expected_kind = str(manifest["kind"])
    expected_experiment = str(manifest["experiment"])
    expected_plan_id = str(manifest["plan_id"])
    candidate_path = primary_path
    candidate_prefix = primary_prefix
    rejected: list[dict[str, str]] = []

    for attempt in range(maximum_recovery_attempts):
        if candidate_path.is_file():
            try:
                archived_manifest = verify_incident_archive(
                    candidate_path,
                    prefix=candidate_prefix,
                    expected_kind=expected_kind,
                    expected_experiment=expected_experiment,
                    expected_plan_id=expected_plan_id,
                    expected_identity=expected_identity,
                )
                return IncidentArchive(
                    path=candidate_path,
                    prefix=candidate_prefix,
                    manifest=archived_manifest,
                    reused=True,
                    rejected_archives=tuple(rejected),
                )
            except Exception as exc:
                try:
                    rejected_sha256 = sha256_file(candidate_path)
                except OSError:
                    rejected_sha256 = "unreadable"
                rejection = {
                    "path": str(candidate_path),
                    "sha256": rejected_sha256,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                rejected.append(rejection)
                collision = canonical_sha256(
                    {
                        "desired_identity": expected_identity,
                        "rejected": rejection,
                        "attempt": attempt,
                    }
                )[:12]
                candidate_prefix = (
                    f"{primary_prefix}-recovery-{expected_identity[:12]}-{collision}"
                )
                candidate_path = primary_path.with_name(f"{candidate_prefix}.tar.gz")
                continue

        _create_archive(
            candidate_path,
            prefix=candidate_prefix,
            root=bundle_root,
            manifest=manifest,
        )
        archived_manifest = verify_incident_archive(
            candidate_path,
            prefix=candidate_prefix,
            expected_kind=expected_kind,
            expected_experiment=expected_experiment,
            expected_plan_id=expected_plan_id,
            expected_identity=expected_identity,
        )
        return IncidentArchive(
            path=candidate_path,
            prefix=candidate_prefix,
            manifest=archived_manifest,
            reused=False,
            rejected_archives=tuple(rejected),
        )

    raise RuntimeError("unable to allocate a valid content-addressed incident recovery archive")
