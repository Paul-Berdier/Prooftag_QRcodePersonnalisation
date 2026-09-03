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
    candidate_id: str,
    prompt_id: str,
    image_path: Path,
    source_kind: str,
    variant: str,
    exact: int,
    eligible: bool = True,
    recipe_id: str | None = None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "prompt_id": prompt_id,
        "prompt_family": "test",
        "source_kind": source_kind,
        "variant": variant,
        "srmpgd_recipe_id": recipe_id,
        "iteration": 1 if source_kind == "srmpgd" else 0,
        "gamma": 250 if source_kind == "srmpgd" else 0,
        "image_path": str(image_path),
        "latent_path": None,
        "image_sha256": f"{exact + 1:064x}",
        "eligible_final": eligible,
        "visual_guard_pass": eligible,
        "uniform_quiet_zone_replacement": False,
        "quiet_zone_delivery_guard_pass": variant.endswith("scene_qz"),
        "wechat_exact_presets": exact,
        "wechat_exact_rate": exact / 37,
        "wechat_original_exact": exact == 37,
        "clip_aesthetic": 5.0 + exact / 100,
        "hpsv2_1": 0.2,
        "clip_score": 0.6,
        "lpips": 0.01,
        "module_error_rate": 0.1,
        "visual_guard_checks": {"ok": eligible},
        "module_error_breakdown": {},
        "quiet_zone_metrics": {},
    }


def test_aggregate_builds_dataset_pareto_phone_queue_and_manifest(tmp_path: Path):
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

    parent_rows = []
    for index, candidate in enumerate(plan["candidates"]):
        root = plan_dir / "parents" / candidate["id"]
        images = root / "images"
        _write_image(images / "stage1-raw.png", 80 + index)
        _write_image(images / "stage1-scene-qz.png", 100 + index)
        _write_image(images / "stage2-raw.png", 120 + index)
        _write_image(images / "stage2-scene-qz.png", 140 + index)
        scoring = root / "scoring"
        scoring.mkdir(parents=True)
        rows = [
            _row(
                candidate_id=candidate["id"],
                prompt_id=candidate["prompt_id"],
                image_path=images / "stage1-raw.png",
                source_kind="parent",
                variant="stage1_raw",
                exact=index,
            ),
            _row(
                candidate_id=candidate["id"],
                prompt_id=candidate["prompt_id"],
                image_path=images / "stage2-raw.png",
                source_kind="parent",
                variant="stage2_raw",
                exact=5 + index,
            ),
        ]
        (scoring / "comparison.json").write_text(
            json.dumps(rows), encoding="utf-8"
        )
        parent_rows.extend(rows)

    (plan_dir / "PARENT_SCORING_COMPLETE.json").write_text(
        json.dumps({"scored_count": len(plan["candidates"])}),
        encoding="utf-8",
    )
    selected_candidate = plan["candidates"][0]
    selected_row = parent_rows[1]
    (plan_dir / "selected-parents.json").write_text(
        json.dumps({"selected": [selected_row]}),
        encoding="utf-8",
    )

    recipe_id = plan["srmpgd_recipes"][0]["id"]
    ref_root = (
        plan_dir / "refinements" / selected_candidate["id"] / recipe_id
    )
    image_path = ref_root / "scene-qz" / "iteration-001.png"
    _write_image(image_path, 180)
    scoring = ref_root / "scoring"
    scoring.mkdir(parents=True)
    refinement_row = _row(
        candidate_id=selected_candidate["id"],
        prompt_id=selected_candidate["prompt_id"],
        image_path=image_path,
        source_kind="srmpgd",
        variant="i001_scene_qz",
        exact=20,
        recipe_id=recipe_id,
    )
    (scoring / "comparison.json").write_text(
        json.dumps([refinement_row]),
        encoding="utf-8",
    )
    (plan_dir / "REFINEMENT_SCORING_COMPLETE.json").write_text(
        json.dumps({"scored_count": 1}),
        encoding="utf-8",
    )

    verdict = aggregate(output_root=output, plan_id=plan["plan_id"])
    assert verdict["complete"] is True
    assert verdict["winner_wechat_exact_presets"] == 20
    assert verdict["winner_uniform_quiet_zone_replacement"] is False
    assert verdict["phone_sample_pending_count"] >= 1
    assert (plan_dir / "dataset/e047-training-contract.json").is_file()
    assert (plan_dir / "dataset/phone-sample-pending.csv").is_file()
    assert (plan_dir / "pipeline/99-FINAL-QR.png").is_file()
    assert verify(output_root=output, plan_id=plan["plan_id"])["valid"] is True
