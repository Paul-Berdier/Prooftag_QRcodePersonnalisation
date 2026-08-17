import csv
import hashlib
import io
import json

from prooftag_qr import e026_recovery


def test_recovery_filters_payload_exports_terminal_campaigns_and_audits(tmp_path, monkeypatch):
    payload = "https://ptag.io/t/e026w"
    plan_dir = tmp_path / "02b9402fba845d79"
    plan_dir.mkdir()
    (plan_dir / "plan-redacted.json").write_text(
        json.dumps({"payload_sha256": hashlib.sha256(payload.encode()).hexdigest()}),
        encoding="utf-8",
    )
    fields = [
        "campaign_id",
        "payload_hash",
        "trial_id",
        "prompt_id",
        "prompt_text",
        "method_id",
        "method_configuration_json",
        "seed",
        "error_correction",
        "status",
        "generation_run_id",
    ]

    def export(campaign_id):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "campaign_id": campaign_id,
                "payload_hash": hashlib.sha256(payload.encode()).hexdigest(),
                "trial_id": f"trial-{campaign_id}",
                "prompt_id": "p1",
                "prompt_text": "A glass garden",
                "method_id": "srpg",
                "method_configuration_json": '{"id":"srpg"}',
                "seed": "11",
                "error_correction": "M",
                "status": "accepted",
                "generation_run_id": f"run-{campaign_id}",
            }
        )
        return stream.getvalue().encode()

    campaigns = [
        {
            "id": "one",
            "name": "E026W unattended batch 01",
            "payload_hash": hashlib.sha256(payload.encode()).hexdigest(),
            "status": "completed",
        },
        {
            "id": "two",
            "name": "E026W unattended batch 01 retry",
            "payload_hash": hashlib.sha256(payload.encode()).hexdigest(),
            "status": "interrupted",
        },
        {
            "id": "foreign",
            "name": "E026W unattended batch 02",
            "payload_hash": "different",
            "status": "completed",
        },
    ]

    def fake_get(_api_url, path, *, raw=False):
        if path.endswith("?limit=500"):
            return campaigns
        campaign_id = path.split("/")[4]
        return export(campaign_id)

    monkeypatch.setattr(e026_recovery, "_get", fake_get)
    summary = e026_recovery.recover_e026_exports(
        api_url="http://api",
        plan_dir=plan_dir,
    )

    assert summary["database_campaigns"] == 2
    assert summary["generated_rows"] == 2
    assert summary["unique_logical_rows"] == 1
    assert summary["duplicate_generated_rows"] == 1
    assert summary["campaign_statuses"] == {"completed": 1, "interrupted": 1}
    assert len(list((plan_dir / "exports-recovered").glob("*.csv"))) == 2

