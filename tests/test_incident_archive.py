import tarfile
from pathlib import Path

import pytest

from prooftag_qr.incident_archive import (
    canonical_incident_material,
    prepare_incident_bundle,
    resolve_incident_archive,
    snapshot_tree_once,
    verify_incident_archive,
)


def _prepare(root: Path, *, error: str = "boom") -> dict:
    root.mkdir(parents=True, exist_ok=True)
    (root / "failure.json").write_text('{"error":"boom"}', encoding="utf-8")
    (root / "evidence").mkdir(exist_ok=True)
    (root / "evidence" / "result.csv").write_text("trial,status\n1,error\n", encoding="utf-8")
    return prepare_incident_bundle(
        root,
        kind="post_gpu_failure",
        experiment="e034-test",
        plan_id="plan-1",
        identity_material={
            "error": error,
            "created_at": "volatile",
            "state": {"updated_at": 123.0, "status": "failed"},
        },
    )


def _verify(path: Path, prefix: str, identity: str) -> dict:
    return verify_incident_archive(
        path,
        prefix=prefix,
        expected_kind="post_gpu_failure",
        expected_experiment="e034-test",
        expected_plan_id="plan-1",
        expected_identity=identity,
    )


def test_incident_archive_is_fully_verified_and_reused(tmp_path):
    root = tmp_path / "bundle"
    manifest = _prepare(root)
    primary = tmp_path / "incident.tar.gz"
    first = resolve_incident_archive(
        primary,
        primary_prefix="incident",
        bundle_root=root,
        manifest=manifest,
    )
    first_bytes = primary.read_bytes()
    second = resolve_incident_archive(
        primary,
        primary_prefix="incident",
        bundle_root=root,
        manifest=manifest,
    )

    assert first.path == second.path == primary
    assert first.reused is False
    assert second.reused is True
    assert primary.read_bytes() == first_bytes
    verified = _verify(primary, "incident", manifest["incident_identity_sha256"])
    assert verified["artifact_checksums"] == manifest["artifact_checksums"]


def test_changed_evidence_creates_content_addressed_recovery_without_overwrite(tmp_path):
    root = tmp_path / "bundle"
    first_manifest = _prepare(root)
    primary = tmp_path / "incident.tar.gz"
    resolve_incident_archive(
        primary,
        primary_prefix="incident",
        bundle_root=root,
        manifest=first_manifest,
    )
    primary_bytes = primary.read_bytes()

    (root / "evidence" / "result.csv").write_text(
        "trial,status\n1,completed_with_errors\n", encoding="utf-8"
    )
    second_manifest = prepare_incident_bundle(
        root,
        kind="post_gpu_failure",
        experiment="e034-test",
        plan_id="plan-1",
        identity_material={"error": "boom", "state": {"status": "failed"}},
    )
    recovery = resolve_incident_archive(
        primary,
        primary_prefix="incident",
        bundle_root=root,
        manifest=second_manifest,
    )

    assert primary.read_bytes() == primary_bytes
    assert recovery.path != primary
    assert f"recovery-{second_manifest['incident_identity_sha256'][:12]}" in recovery.path.name
    assert recovery.rejected_archives
    _verify(
        recovery.path,
        recovery.prefix,
        second_manifest["incident_identity_sha256"],
    )


def test_corrupt_existing_archive_is_never_overwritten(tmp_path):
    root = tmp_path / "bundle"
    manifest = _prepare(root)
    primary = tmp_path / "incident.tar.gz"
    resolve_incident_archive(
        primary,
        primary_prefix="incident",
        bundle_root=root,
        manifest=manifest,
    )
    primary.write_bytes(b"corrupt archive retained for audit")
    corrupt_bytes = primary.read_bytes()

    recovery = resolve_incident_archive(
        primary,
        primary_prefix="incident",
        bundle_root=root,
        manifest=manifest,
    )

    assert primary.read_bytes() == corrupt_bytes
    assert recovery.path != primary
    assert recovery.path.is_file()
    _verify(recovery.path, recovery.prefix, manifest["incident_identity_sha256"])


def test_archive_rejects_unlisted_regular_members(tmp_path):
    root = tmp_path / "bundle"
    manifest = _prepare(root)
    primary = tmp_path / "incident.tar.gz"
    result = resolve_incident_archive(
        primary,
        primary_prefix="incident",
        bundle_root=root,
        manifest=manifest,
    )
    unexpected = tmp_path / "unexpected.txt"
    unexpected.write_text("not in manifest", encoding="utf-8")
    altered = tmp_path / "altered.tar.gz"
    with tarfile.open(result.path, "r:gz") as source, tarfile.open(altered, "w:gz") as target:
        for member in source.getmembers():
            if member.isfile():
                stream = source.extractfile(member)
                assert stream is not None
                temporary = tmp_path / "member"
                temporary.write_bytes(stream.read())
                target.add(temporary, arcname=member.name)
        target.add(unexpected, arcname="incident/unexpected.txt")

    with pytest.raises(RuntimeError, match="inventory differs"):
        _verify(altered, "incident", manifest["incident_identity_sha256"])


def test_snapshot_and_canonical_identity_are_restart_stable(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "snapshot"
    source.mkdir()
    (source / "state.json").write_text("first", encoding="utf-8")
    snapshot_tree_once(source, target)
    (source / "state.json").write_text("second", encoding="utf-8")
    (source / "late.json").write_text("must not enter snapshot", encoding="utf-8")
    snapshot_tree_once(source, target)

    assert (target / "state.json").read_text(encoding="utf-8") == "first"
    assert not (target / "late.json").exists()
    assert canonical_incident_material(
        {"created_at": "a", "state": {"updated_at": 1, "status": "failed"}}
    ) == {"state": {"status": "failed"}}
