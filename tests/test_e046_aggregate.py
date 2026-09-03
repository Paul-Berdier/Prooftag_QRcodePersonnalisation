import hashlib
import json
from pathlib import Path

from PIL import Image

from prooftag_qr.e046_campaign import aggregate, create_plan, verify


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (value, value, value)).save(path)


def _e045(root: Path) -> None:
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
        json.dumps(
            {
                "status": "complete",
                "plan_id": complete["plan_id"],
                "plan_dir": str(plan),
            }
        ),
        encoding="utf-8",
    )


def _row(
    *,
    candidate: dict,
    image_path: Path,
    exact: int,
    original: bool,
    clip: float,
    hps: float,
    aesthetic: float,
) -> dict:
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    return {
        "candidate_id": candidate["id"],
        "prompt_id": candidate["prompt_id"],
        "prompt_variant_index": candidate["prompt_variant_index"],
        "prompt_family": candidate["prompt_family"],
        "prompt": candidate["prompt"],
        "payload": candidate["payload"],
        "payload_sha256": hashlib.sha256(
            candidate["payload"].encode("utf-8")
        ).hexdigest(),
        "parent_recipe_id": candidate["parent_recipe_id"],
        "seed": candidate["seed"],
        "source_kind": "parent",
        "stage": "stage2",
        "variant": "stage2_raw",
        "quiet_zone_variant": "raw",
        "srmpgd_recipe_id": None,
        "iteration": 0,
        "gamma": 0.0,
        "image_path": str(image_path),
        "latent_path": None,
        "image_sha256": image_hash,
        "eligible_final": True,
        "visual_guard_pass": True,
        "uniform_quiet_zone_replacement": False,
        "quiet_zone_delivery_guard_pass": None,
        "wechat_exact_presets": exact,
        "wechat_exact_rate": exact / 37,
        "wechat_original_exact": original,
        "clip_aesthetic": aesthetic,
        "hpsv2_1": hps,
        "clip_score": clip,
        "lpips": 0.0,
        "module_error_rate": 0.02,
        "visual_guard_checks": {"ok": True},
        "module_error_breakdown": {},
        "quiet_zone_metrics": {},
    }


def _prepare_plan(tmp_path: Path):
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
    plan_dir = output / plan["plan_id"]
    return output, plan, plan_dir


def test_aggregate_exports_one_automatic_valid_beautiful_qr_per_prompt(tmp_path: Path):
    output, plan, plan_dir = _prepare_plan(tmp_path)

    # Three candidates per prompt. Beautiful but invalid rows must never win.
    prompt_seen = {}
    for index, candidate in enumerate(plan["candidates"]):
        root = plan_dir / "parents" / candidate["id"]
        image_path = root / "images/stage2-raw.png"
        _write_image(image_path, 30 + index * 15)
        prompt_position = prompt_seen.get(candidate["prompt_id"], 0)
        prompt_seen[candidate["prompt_id"]] = prompt_position + 1

        if prompt_position == 0:
            exact, original = 36, True
            clip, hps, aesthetic = 0.72, 0.22, 5.2
        elif prompt_position == 1:
            exact, original = 34, True
            clip, hps, aesthetic = 0.90, 0.36, 6.4
        else:
            exact, original = 5, False
            clip, hps, aesthetic = 0.99, 0.50, 8.0

        scoring = root / "scoring"
        scoring.mkdir(parents=True)
        row = _row(
            candidate=candidate,
            image_path=image_path,
            exact=exact,
            original=original,
            clip=clip,
            hps=hps,
            aesthetic=aesthetic,
        )
        (scoring / "comparison.json").write_text(
            json.dumps([row]), encoding="utf-8"
        )

    (plan_dir / "PARENT_SCORING_COMPLETE.json").write_text(
        json.dumps({"scored_count": len(plan["candidates"])}),
        encoding="utf-8",
    )
    (plan_dir / "selected-parents.json").write_text(
        json.dumps({"selected": []}), encoding="utf-8"
    )
    (plan_dir / "REFINEMENT_SCORING_COMPLETE.json").write_text(
        json.dumps({"scored_count": 0}), encoding="utf-8"
    )

    verdict = aggregate(output_root=output, plan_id=plan["plan_id"])

    assert verdict["complete"] is True
    assert verdict["prompt_count"] == 2
    assert verdict["software_valid_prompt_count"] == 2
    assert verdict["unresolved_prompt_count"] == 0
    assert verdict["all_prompts_have_software_valid_final"] is True
    assert verdict["winner_wechat_exact_presets"] in {34, 36}
    assert verdict["winner_wechat_original_exact"] is True

    best = json.loads(
        (plan_dir / "dataset/best-by-prompt.json").read_text(encoding="utf-8")
    )
    assert len(best) == 2
    assert all(row["software_valid_final"] for row in best)
    assert all(row["wechat_exact_presets"] >= 34 for row in best)
    assert all(row["candidate_id"].split("_")[-1] != "s2" for row in best)

    for prompt_id in {candidate["prompt_id"] for candidate in plan["candidates"]}:
        assert (plan_dir / f"pipeline/by-prompt/{prompt_id}/FINAL-QR.png").is_file()
        metadata = json.loads(
            (plan_dir / f"pipeline/by-prompt/{prompt_id}/FINAL-metadata.json")
            .read_text(encoding="utf-8")
        )
        assert metadata["manual_verification_required_for_selection"] is False

    assert (plan_dir / "dataset/e047-training-contract.json").is_file()
    assert (plan_dir / "pipeline/99-FINAL-QR.png").is_file()
    assert verify(output_root=output, plan_id=plan["plan_id"])["valid"] is True


def test_aggregate_refuses_to_label_invalid_prompt_as_final(tmp_path: Path):
    output, plan, plan_dir = _prepare_plan(tmp_path)

    for index, candidate in enumerate(plan["candidates"]):
        root = plan_dir / "parents" / candidate["id"]
        image_path = root / "images/stage2-raw.png"
        _write_image(image_path, 25 + index * 12)
        first_prompt = candidate["prompt_id"] == plan["candidates"][0]["prompt_id"]
        row = _row(
            candidate=candidate,
            image_path=image_path,
            exact=36 if first_prompt else 10,
            original=first_prompt,
            clip=0.8,
            hps=0.25,
            aesthetic=5.5,
        )
        scoring = root / "scoring"
        scoring.mkdir(parents=True)
        (scoring / "comparison.json").write_text(
            json.dumps([row]), encoding="utf-8"
        )

    (plan_dir / "PARENT_SCORING_COMPLETE.json").write_text(
        json.dumps({"scored_count": len(plan["candidates"])}), encoding="utf-8"
    )
    (plan_dir / "selected-parents.json").write_text(
        json.dumps({"selected": []}), encoding="utf-8"
    )
    (plan_dir / "REFINEMENT_SCORING_COMPLETE.json").write_text(
        json.dumps({"scored_count": 0}), encoding="utf-8"
    )

    verdict = aggregate(output_root=output, plan_id=plan["plan_id"])
    assert verdict["unresolved_prompt_count"] == 1
    unresolved_key = verdict["unresolved_prompt_keys"][0]
    unresolved_prompt_id = unresolved_key.split("::", 1)[0]
    assert not (
        plan_dir / f"pipeline/by-prompt/{unresolved_prompt_id}/FINAL-QR.png"
    ).exists()
    assert (
        plan_dir / f"pipeline/by-prompt/{unresolved_prompt_id}/NO-VALID-FINAL.json"
    ).is_file()
