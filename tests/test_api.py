import time


def test_api_generation_reports_physical_validation_and_lab(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOFTAG_QR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROOFTAG_QR_MODEL_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("PROOFTAG_QR_DEFAULT_BACKEND", "qr")
    monkeypatch.setenv("PROOFTAG_QR_VALIDATION_MIN_PASS_RATE", "1.0")

    from fastapi.testclient import TestClient

    from prooftag_qr.api import app

    client = TestClient(app)
    payload = "https://example.prooftag.test/t/api-test"

    response = client.post(
        "/v1/generations",
        json={"payload": payload, "backend": "qr", "seed": 11, "max_attempts": 1},
    )
    assert response.status_code == 200
    run = response.json()
    assert run["status"] == "accepted"
    assert run["scan_pass_rate"] == 1.0

    physical = client.post(
        f"/v1/generations/{run['id']}/physical-validations",
        json={
            "decoded_payload": payload,
            "device": "test-device",
            "material": "screen",
        },
    )
    assert physical.status_code == 200
    assert physical.json()["outcome"] == "exact"

    assert client.get("/v1/reports/summary").json()["accepted_runs"] == 1
    assert client.get("/metrics").status_code == 200

    runtime = client.get("/v1/runtime")
    assert runtime.status_code == 200
    assert "torch" in runtime.json()["packages"]
    assert "cuda_available" in runtime.json()
    assert runtime.json()["generation_config"]["latent_refinement_enabled"] is False
    assert runtime.json()["generation_config"]["guided_rediffusion_enabled"] is False
    assert runtime.json()["generation_config"]["srmpgd_enabled"] is False

    lab_page = client.get("/lab")
    assert lab_page.status_code == 200
    assert "srpg-effective-steps" in lab_page.text
    assert "20260729-srmpgd-paper-1" in lab_page.text
    lab_javascript = client.get("/lab-assets/app.js")
    assert lab_javascript.status_code == 200
    assert "effectiveSrpgSteps" in lab_javascript.text
    schema = client.get("/v1/lab/schema")
    assert schema.status_code == 200
    assert any(item["id"] == "srpg_late_2" for item in schema.json()["profiles"])
    assert any(item["id"] == "srpg_late_4" for item in schema.json()["profiles"])
    assert any(
        item["id"] == "srpg_late_4_srmpgd"
        for item in schema.json()["profiles"]
    )
    controlnet_profile = next(
        item for item in schema.json()["profiles"] if item["id"] == "controlnet_raw"
    )
    srpg_profile = next(
        item for item in schema.json()["profiles"] if item["id"] == "srpg_late_2"
    )
    full_restart_profile = next(
        item
        for item in schema.json()["profiles"]
        if item["id"] == "srpg_full_restart"
    )
    srmpgd_profile = next(
        item
        for item in schema.json()["profiles"]
        if item["id"] == "srpg_late_4_srmpgd"
    )
    assert controlnet_profile["model"]["base_model_id"]
    assert controlnet_profile["model"]["controlnet_model_id"]
    assert "Cetus-Mix_Whalefall" in controlnet_profile["model"]["base_model_id"]
    assert (
        controlnet_profile["model"]["controlnet_model_id"]
        == "monster-labs/control_v1p_sd15_qrcode_monster"
    )
    assert controlnet_profile["model"]["controlnet_model_subfolder"] == "v2"
    assert controlnet_profile["output_variant"] == "raw"
    assert srpg_profile["output_variant"] == "srpg"
    assert srpg_profile["reuse_stage1"] is True
    assert srpg_profile["enabled"] is True
    assert srpg_profile["tools"]["settings"]["srpg_strength"] == 0.05
    assert srmpgd_profile["output_variant"] == "srmpgd"
    assert srmpgd_profile["tools"]["srpg_enabled"] is True
    assert srmpgd_profile["tools"]["srmpgd_enabled"] is True
    assert srmpgd_profile["tools"]["settings"]["srmpgd_step_size"] == 1000.0
    assert srmpgd_profile["tools"]["settings"]["srmpgd_lpips_weight"] == 0.01
    assert full_restart_profile["enabled"] is False

    campaign_response = client.post(
        "/v1/lab/campaigns",
        json={
            "name": "API lab smoke",
            "payload": payload,
            "error_correction": "M",
            "prompts": [{"id": "smoke", "text": "reference"}],
            "seeds": [77],
            "methods": [
                {
                    "id": "reference",
                    "name": "QR reference",
                    "backend": "qr",
                    "generation": {
                        "steps": 1,
                        "guidance_scale": 0,
                        "controlnet_scale": 0,
                        "strength": 1,
                    },
                    "tools": {"settings": {}},
                }
            ],
            "max_attempts": 1,
        },
    )
    assert campaign_response.status_code == 200
    campaign_id = campaign_response.json()["id"]
    campaign = None
    for _ in range(100):
        campaign = client.get(f"/v1/lab/campaigns/{campaign_id}").json()
        if campaign["status"] not in {"queued", "running"}:
            break
        time.sleep(0.02)
    assert campaign is not None
    assert campaign["status"] == "completed"
    assert campaign["accepted_trials"] == 1
    trial = campaign["trials"][0]
    assert trial["generation"]["scan_pass_rate"] == 1.0

    rating = client.put(
        f"/v1/lab/trials/{trial['id']}/rating",
        json={
            "aesthetic_score": 4,
            "prompt_fidelity_score": 5,
            "qr_discretion_score": 1,
            "overall_score": 4,
            "favorite": True,
            "notes": "smoke",
        },
    )
    assert rating.status_code == 200
    assert rating.json()["favorite"] is True
    campaign_csv = client.get(f"/v1/lab/campaigns/{campaign_id}/results.csv")
    assert campaign_csv.status_code == 200
    assert "quality_brightness_mean" in campaign_csv.text.splitlines()[0]
    artifacts = client.get(
        f"/v1/generations/{trial['generation_run_id']}/artifacts"
    ).json()
    assert artifacts[0]["name"] == "final"
    assert len(artifacts) == 1
