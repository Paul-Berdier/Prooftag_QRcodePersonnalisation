"""Safe, selective loading helpers for the CPU-only E030 re-score notebook.

E029 archives contain a large serialized advisor which E030 never needs.  This
module deliberately extracts only the scientific tables and gallery rasters.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

E029_EXPERIMENT = "e029-srmpgd-exact-raster-recovery-v4"
_ROOT_FILES = {
    "manifest.json",
    "training-report.json",
    "e029-state-results.csv",
    "e029-pairing-audit.csv",
    "e029-srmpgd-iteration-zero-raster-audit.csv",
    "e029-policy-decisions.csv",
    "e029-policy-report.json",
}
_GALLERY_FILES = {
    "e029-gallery/gallery-index.csv",
    "e029-gallery/gallery-audit.json",
    "e029-gallery/paired-advisor-sample.png",
    "e029-gallery/advisor-deliverable-winners.png",
}


def _normalized_sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} is not a valid SHA-256")
    return normalized


def validate_rescore_journal_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    expected_raster_sha256_by_source: Mapping[str, str],
    payload_sha256: str,
    scorer_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate resumable E030 records before they can skip decoder work.

    The journal is an optimization, not an authority.  Every complete record is
    therefore bound to the current source raster and to the complete scorer
    identity.  Identical duplicate records are tolerated for crash recovery;
    conflicting duplicates stop the run instead of selecting one implicitly.
    """

    expected_payload = _normalized_sha256(payload_sha256, field="payload_sha256")
    expected_identity = {
        "engine_version": str(scorer_identity.get("engine_version") or ""),
        "scoring_version": str(scorer_identity.get("scoring_version") or ""),
        "implementation_sha256": _normalized_sha256(
            scorer_identity.get("implementation_sha256"),
            field="scorer_identity.implementation_sha256",
        ),
        "repetitions": scorer_identity.get("repetitions"),
        "preset_count": scorer_identity.get("preset_count"),
    }
    if not expected_identity["engine_version"] or not expected_identity["scoring_version"]:
        raise ValueError("scorer identity versions must be non-empty")
    if type(expected_identity["repetitions"]) is not int or expected_identity["repetitions"] < 2:
        raise ValueError("scorer identity repetitions must be an integer >= 2")
    if type(expected_identity["preset_count"]) is not int or expected_identity["preset_count"] < 1:
        raise ValueError("scorer identity preset_count must be a positive integer")

    expected_rasters = {
        _normalized_sha256(source, field="source_png_sha256"): _normalized_sha256(
            raster, field="expected_raster_sha256"
        )
        for source, raster in expected_raster_sha256_by_source.items()
    }
    accepted: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        if row.get("run_id") != run_id:
            raise ValueError("rescore journal run_id does not match the active run")
        source_hash = _normalized_sha256(
            row.get("source_png_sha256"), field="journal.source_png_sha256"
        )
        expected_raster = expected_rasters.get(source_hash)
        if expected_raster is None:
            raise ValueError(f"rescore journal references an unknown source: {source_hash}")
        score = row.get("score")
        if not isinstance(score, Mapping):
            raise ValueError(f"rescore journal score is not an object: {source_hash}")
        if (
            _normalized_sha256(score.get("image_sha256"), field="journal.score.image_sha256")
            != expected_raster
        ):
            raise ValueError(f"rescore journal raster hash mismatch: {source_hash}")
        if (
            _normalized_sha256(score.get("payload_sha256"), field="journal.score.payload_sha256")
            != expected_payload
        ):
            raise ValueError(f"rescore journal payload hash mismatch: {source_hash}")
        for field in ("engine_version", "scoring_version", "implementation_sha256"):
            actual = score.get(field)
            if field == "implementation_sha256":
                actual = _normalized_sha256(actual, field=f"journal.score.{field}")
            else:
                actual = str(actual or "")
            if actual != expected_identity[field]:
                raise ValueError(f"rescore journal {field} mismatch: {source_hash}")
        for field in ("repetitions", "preset_count"):
            actual = score.get(field)
            if type(actual) is not int or actual != expected_identity[field]:
                raise ValueError(f"rescore journal {field} mismatch: {source_hash}")

        previous = accepted.get(source_hash)
        if previous is not None:
            if previous != row:
                raise ValueError(f"conflicting rescore journal duplicate: {source_hash}")
            continue
        accepted[source_hash] = row
    return list(accepted.values())


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 without loading large archives in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_e029_archive(
    download_root: Path,
    override: Path | str | None = None,
) -> Path:
    """Select an explicit archive or the newest E029 v4 archive in downloads."""

    if override is not None and str(override).strip():
        selected = Path(override).expanduser().resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"SOURCE_ARCHIVE does not exist: {selected}")
        if not selected.name.endswith(".tar.gz"):
            raise ValueError(f"SOURCE_ARCHIVE must be a .tar.gz file: {selected}")
        return selected

    root = Path(download_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"download directory does not exist: {root}")
    candidates = sorted(
        root.glob("*e029-srmpgd-exact-raster-recovery-v4.tar.gz"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no E029 v4 archive under {root}; set SOURCE_ARCHIVE explicitly")
    return candidates[0].resolve()


def discover_e029_export_directory(search_roots: list[Path] | tuple[Path, ...]) -> Path:
    """Return the newest complete, already-extracted E029 v4 export.

    Invalid/partial runner directories are skipped.  This lets E030 prefer the
    persistent PVC over the ephemeral ``/workspace/downloads`` directory.
    """

    candidates: list[Path] = []
    for configured_root in search_roots:
        root = Path(configured_root).expanduser().resolve()
        if not root.is_dir():
            continue
        for manifest_path in root.rglob("manifest.json"):
            candidate = manifest_path.parent.resolve()
            try:
                validate_e029_export(candidate)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            candidates.append(candidate)
    if not candidates:
        roots = ", ".join(str(Path(root)) for root in search_roots)
        raise FileNotFoundError(f"no complete E029 v4 export under: {roots}")
    return max(
        candidates,
        key=lambda path: ((path / "manifest.json").stat().st_mtime_ns, str(path)),
    )


def e029_export_sha256(root: Path) -> str:
    """Hash the complete selected E029 evidence set without the large advisor."""

    root = Path(root).resolve()
    validate_e029_export(root)
    paths = [root / name for name in sorted(_ROOT_FILES)]
    paths.extend(sorted(path for path in (root / "e029-gallery").rglob("*") if path.is_file()))
    digest = hashlib.sha256()
    digest.update(b"prooftag-e029-selective-export-v1\0")
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _safe_member_path(name: str) -> PurePosixPath:
    if "\\" in name:
        raise ValueError(f"unsafe tar member uses a backslash: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe tar member path: {name!r}")
    if not path.parts or any(part in {"", "."} for part in path.parts):
        raise ValueError(f"invalid tar member path: {name!r}")
    return path


def _relative_member(path: PurePosixPath, archive_root: str) -> PurePosixPath:
    if path.parts[0] != archive_root:
        raise ValueError(f"archive contains multiple roots: {archive_root!r} and {path.parts[0]!r}")
    return PurePosixPath(*path.parts[1:])


def _wanted(relative: PurePosixPath) -> bool:
    text = relative.as_posix()
    return (
        text in _ROOT_FILES
        or text in _GALLERY_FILES
        or (
            len(relative.parts) == 3
            and relative.parts[:2] == ("e029-gallery", "images")
            and relative.suffix.lower() == ".png"
        )
    )


def selective_extract_e029_archive(
    archive_path: Path,
    destination: Path,
    *,
    reserve_bytes: int = 256 * 1024 * 1024,
) -> Path:
    """Extract only E029 reports and PNGs, with traversal and disk guards.

    Extraction is file-atomic and therefore safely resumable.  The large
    ``*.joblib`` member is never written to disk.
    """

    archive_path = Path(archive_path).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as bundle:
        members = bundle.getmembers()
        if not members:
            raise ValueError("empty E029 archive")
        paths = [_safe_member_path(member.name) for member in members]
        roots = {path.parts[0] for path in paths}
        if len(roots) != 1:
            raise ValueError(f"archive must contain exactly one root: {sorted(roots)}")
        archive_root = next(iter(roots))
        selected: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        for member, path in zip(members, paths, strict=True):
            relative = _relative_member(path, archive_root)
            if not relative.parts or not _wanted(relative):
                continue
            if not member.isfile():
                raise ValueError(f"selected tar member is not a regular file: {member.name}")
            selected.append((member, relative))

        selected_names = {relative.as_posix() for _, relative in selected}
        if len(selected_names) != len(selected):
            raise ValueError("E029 archive contains duplicate selected member paths")
        missing = (_ROOT_FILES | {"e029-gallery/gallery-index.csv"}) - selected_names
        if missing:
            raise ValueError(f"E029 archive is missing required members: {sorted(missing)}")
        required_bytes = sum(member.size for member, _ in selected) + max(0, reserve_bytes)
        free_bytes = shutil.disk_usage(destination).free
        if free_bytes < required_bytes:
            raise OSError(
                f"insufficient disk space: need {required_bytes} bytes, have {free_bytes}"
            )

        for member, relative in selected:
            target = (destination / Path(*relative.parts)).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"tar member escapes destination: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read tar member: {member.name}")
            temporary = target.with_name(f".{target.name}.e030-part")
            with temporary.open("wb") as output:
                shutil.copyfileobj(extracted, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if temporary.stat().st_size != member.size:
                temporary.unlink(missing_ok=True)
                raise OSError(f"truncated tar member: {member.name}")
            temporary.replace(target)

    return destination


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_e029_export(root: Path) -> dict[str, Any]:
    """Refuse partial galleries before spending CPU time on QR-Verify."""

    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment") != E029_EXPERIMENT:
        raise ValueError(f"expected {E029_EXPERIMENT!r}, got {manifest.get('experiment')!r}")
    expected_rows = int(manifest.get("plan", {}).get("trial_count", 0))
    if expected_rows <= 0:
        raise ValueError("E029 manifest has no positive plan.trial_count")

    state_rows = _read_csv(root / "e029-state-results.csv")
    gallery_rows = _read_csv(root / "e029-gallery" / "gallery-index.csv")
    if len(state_rows) != expected_rows or len(gallery_rows) != expected_rows:
        raise ValueError(
            "incomplete E029 export: "
            f"state={len(state_rows)}, gallery={len(gallery_rows)}, expected={expected_rows}"
        )
    statuses = {str(row.get("status") or "") for row in state_rows}
    if not statuses <= {"accepted", "rejected"} or "" in statuses:
        raise ValueError(f"E029 state table contains failed generations: {sorted(statuses)}")

    gallery_root = (root / "e029-gallery").resolve()
    trial_ids: set[str] = set()
    hashes: set[str] = set()
    for row in gallery_rows:
        trial_id = row.get("trial_id", "")
        if not trial_id or trial_id in trial_ids:
            raise ValueError(f"missing or duplicate gallery trial_id: {trial_id!r}")
        trial_ids.add(trial_id)
        relative = Path(row.get("local_image", ""))
        image = (gallery_root / relative).resolve()
        if gallery_root not in image.parents or image.suffix.lower() != ".png":
            raise ValueError(f"unsafe gallery image path: {relative}")
        if not image.is_file():
            raise FileNotFoundError(f"gallery image is missing: {image}")
        expected_hash = row.get("image_sha256", "").lower()
        actual_hash = sha256_file(image)
        if len(expected_hash) != 64 or actual_hash != expected_hash:
            raise ValueError(f"gallery image checksum mismatch: {relative}")
        hashes.add(actual_hash)

    state_trials = {row.get("trial_id", "") for row in state_rows}
    if trial_ids != state_trials:
        raise ValueError("state and gallery trial IDs do not match exactly")
    return {
        "experiment": manifest["experiment"],
        "expected_rows": expected_rows,
        "state_rows": len(state_rows),
        "gallery_rows": len(gallery_rows),
        "unique_rasters": len(hashes),
        "payload_sha256": manifest["plan"]["payload_sha256"],
        "prompt_count": int(manifest["plan"]["prompt_count"]),
        "seed_count": int(manifest["plan"]["seed_count"]),
    }
