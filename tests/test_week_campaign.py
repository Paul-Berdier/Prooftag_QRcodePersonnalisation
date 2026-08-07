import json
from pathlib import Path

import pytest

from prooftag_qr.schemas import LabCampaignCreate
from prooftag_qr.week_campaign import (
    WeekCampaignRunner,
    build_week_batches,
    build_week_methods,
    build_week_prompts,
)


def test_week_plan_is_diverse_bounded_and_api_valid():
    prompts = build_week_prompts()
    methods = build_week_methods()
    batches = build_week_batches("https://ptag.io/t/w")

    assert len(prompts) == 300
    assert len({item["id"] for item in prompts}) == 300
    assert len({item["text"] for item in prompts}) == 300
    assert {item["id"].split("_")[1] for item in prompts} == {
        "simple",
        "scene",
        "detailed",
        "atypical",
    }
    assert len(methods) == 16
    assert len({item["id"] for item in methods}) == 16
    assert len(batches) == 30
    assert all(
        len(batch["prompts"]) * len(batch["methods"]) * len(batch["seeds"]) == 480
        for batch in batches
    )
    assert (
        sum(
            len(batch["prompts"]) * len(batch["methods"]) * len(batch["seeds"]) for batch in batches
        )
        == 14_400
    )
    for batch in batches:
        LabCampaignCreate.model_validate(batch)


def test_week_plan_covers_srpg_srmpgd_and_parameter_variations():
    methods = build_week_methods()
    ids = {item["id"] for item in methods}

    assert "diffqrcoder_stage1" in ids
    assert "diffqrcoder_paper_srpg" in ids
    assert "diffqrcoder_srmpgd_robust" in ids
    assert "e026w_srpg_q250_pg1" in ids
    assert "e026w_srpg_q750_pg1" in ids
    assert "e026w_srmpgd_g30_i4_l10" in ids
    assert "e026w_srmpgd_g300_i4_l10" in ids
    assert "e026w_srmpgd_g100_i8_l25" in ids


def test_week_batches_reject_payload_that_overflows_fixed_qr_geometry():
    with pytest.raises(ValueError, match="payload too long"):
        build_week_batches("https://example.invalid/" + "x" * 500)


def test_week_runner_persists_only_a_redacted_plan(tmp_path):
    payload = "https://ptag.io/t/private-week"
    runner = WeekCampaignRunner(
        api_url="http://127.0.0.1:9",
        payload=payload,
        output_root=tmp_path,
        duration_hours=1,
        minimum_free_gib=0,
        poll_seconds=0.01,
    )

    plan_text = runner.plan_path.read_text(encoding="utf-8")
    plan = json.loads(plan_text)
    state = json.loads(runner.state_path.read_text(encoding="utf-8"))
    assert payload not in plan_text
    assert plan["protocol"] == "e026w-v1"
    assert plan["payload_length"] == len(payload)
    assert len(plan["payload_sha256"]) == 64
    assert len(plan["prompt_bank_sha256"]) == 64
    assert state["plan_id"] == runner.plan_id
    assert state["active_campaign_id"] is None


def test_week_job_has_no_gpu_and_uses_the_persistent_data_volume():
    manifest = Path("deploy/k8s/e026-week-job.yaml").read_text(encoding="utf-8")
    launcher = Path("scripts/e026-week-campaign.sh").read_text(encoding="utf-8")

    assert "image: __IMAGE__" in manifest
    assert "nvidia.com/gpu" not in manifest
    assert "activeDeadlineSeconds: 604800" in manifest
    assert "claimName: prooftag-qr-data" in manifest
    assert 'E026_DURATION_HOURS\n              value: "162"' in manifest
    assert "PROOFTAG_QR_SAVE_DEBUG_ARTIFACTS" in launcher
    assert 'kubectl scale deployment "$vllm_deployment"' in launcher
    assert "deploy-start)" in launcher
    assert "bash scripts/deploy-app-image.sh" in launcher
    assert "validate_deployment_and_plan" in launcher
    assert "python -m prooftag_qr.week_campaign --plan-only" in launcher
    assert "restore_runtime" in launcher
