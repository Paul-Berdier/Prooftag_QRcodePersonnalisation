import json
from pathlib import Path

from prooftag_qr.e046_campaign import create_plan, load_plan


def _e045(root: Path):
    plan = root / "288024247c39d585"
    plan.mkdir(parents=True)
    complete = {
        "plan_id": "288024247c39d585",
        "source_commit": "f" * 40,
        "complete": True,
        "resilience_selftest_passed": True,
        "production_ready": False,
        "artifact_manifest_sha256": "c" * 64,
    }
    (plan / "COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    (root / "LATEST.json").write_text(
        json.dumps({
            "status": "complete",
            "plan_id": complete["plan_id"],
            "plan_dir": str(plan),
        }),
        encoding="utf-8",
    )


def test_plan_requires_and_records_e045_contract(tmp_path: Path):
    e045 = tmp_path / "e045"
    e045.mkdir()
    _e045(e045)
    output = tmp_path / "e046"

    plan = create_plan(
        output_root=output,
        profile="smoke",
        source_commit="a" * 40,
        e045_root=e045,
    )
    plan_dir, loaded = load_plan(output, plan["plan_id"])
    assert plan_dir.is_dir()
    assert loaded["e045_plan_id"] == "288024247c39d585"
    assert loaded["e045_manifest_sha256"] == "c" * 64
    assert len(loaded["candidates"]) == 6
    latest = json.loads((output / "LATEST.json").read_text(encoding="utf-8"))
    assert latest["status"] == "planned"


def test_different_profiles_create_different_plan_ids(tmp_path: Path):
    e045 = tmp_path / "e045"
    e045.mkdir()
    _e045(e045)
    output = tmp_path / "e046"

    smoke = create_plan(
        output_root=output,
        profile="smoke",
        source_commit="a" * 40,
        e045_root=e045,
    )
    pilot = create_plan(
        output_root=output,
        profile="pilot",
        source_commit="a" * 40,
        e045_root=e045,
    )
    assert smoke["plan_id"] != pilot["plan_id"]
