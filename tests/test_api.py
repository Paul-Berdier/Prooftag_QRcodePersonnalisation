import time


def test_api_generation_reports_physical_validation_and_lab(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOFTAG_QR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROOFTAG_QR_MODEL_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("PROOFTAG_QR_DEFAULT_BACKEND", "qr")
    monkeypatch.setenv("PROOFTAG_QR_VALIDATION_MIN_PASS_RATE", "1.0")

    from fastapi.testclient import TestClient

    from prooftag_qr.api import app, artifact_store

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

    artifact_store.save_metadata(
        run["id"],
        "srmpgd_trace",
        {"selected_iteration": 0, "steps": [{"iteration": 0}]},
    )
    trace = client.get(f"/v1/generations/{run['id']}/metadata/srmpgd_trace")
    assert trace.status_code == 200
    assert trace.json()["steps"] == [{"iteration": 0}]
    assert client.get(f"/v1/generations/{run['id']}/metadata/MALFORMED").status_code == 400

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
    assert runtime.json()["generation_config"]["srmpgd_max_step_rms"] == 0.02
    assert runtime.json()["generation_config"]["srmpgd_max_total_delta_rms"] == 0.06
    assert runtime.json()["generation_config"]["srmpgd_max_lpips_loss"] == 0.15

    lab_page = client.get("/lab")
    assert lab_page.status_code == 200
    assert "PROOFTAG × DIFFQRCODER" in lab_page.text
    assert "20260805-e023-honest-metrics-1" in lab_page.text
    lab_javascript = client.get("/lab-assets/app.js")
    assert lab_javascript.status_code == 200
    assert "human_scan_result" in lab_javascript.text
    assert "Indice synthétique" in lab_javascript.text
    assert "HPS v2.1" in lab_javascript.text
    schema = client.get("/v1/lab/schema")
    assert schema.status_code == 200
    assert {item["id"] for item in schema.json()["profiles"]} == {
        "qr_reference",
        "diffqrcoder_stage1",
        "diffqrcoder_srpg",
        "diffqrcoder_paper_srpg",
        "diffqrcoder_srmpgd",
        "diffqrcoder_srmpgd_robust",
        "diffqrcoder_auto",
        "diffqrcoder_srpg_s035",
        "diffqrcoder_srpg_s050",
        "diffqrcoder_srpg_s080",
        "diffqrcoder_qart_srpg",
    }
    controlnet_profile = next(
        item for item in schema.json()["profiles"] if item["id"] == "diffqrcoder_stage1"
    )
    srpg_profile = next(
        item for item in schema.json()["profiles"] if item["id"] == "diffqrcoder_srpg"
    )
    srmpgd_profile = next(
        item for item in schema.json()["profiles"] if item["id"] == "diffqrcoder_srmpgd"
    )
    robust_srmpgd_profile = next(
        item for item in schema.json()["profiles"] if item["id"] == "diffqrcoder_srmpgd_robust"
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
    assert srpg_profile["tools"]["settings"]["srpg_steps"] == 40
    assert srpg_profile["tools"]["settings"]["srpg_qr_weight"] == 500.0
    assert srpg_profile["tools"]["settings"]["srpg_perceptual_weight"] == 2.0
    assert (
        srpg_profile["tools"]["settings"]["diffqrcoder_stage2_initialization"]
        == "paper_stage1_noise"
    )
    assert srpg_profile["tools"]["settings"]["diffqrcoder_stage2_target_mode"] == "binary_exact"
    assert srpg_profile["tools"]["settings"]["diffqrcoder_stage2_strength"] == 0.65
    assert srmpgd_profile["output_variant"] == "srmpgd"
    assert srmpgd_profile["tools"]["srpg_enabled"] is True
    assert srmpgd_profile["tools"]["srmpgd_enabled"] is True
    assert srmpgd_profile["tools"]["settings"]["srmpgd_max_iterations"] == 4
    assert srmpgd_profile["tools"]["settings"]["srmpgd_step_size"] == 100.0
    assert srmpgd_profile["tools"]["settings"]["srmpgd_lpips_weight"] == 0.10
    assert srmpgd_profile["tools"]["settings"]["srmpgd_max_step_rms"] == 0.02
    assert srmpgd_profile["tools"]["settings"]["srmpgd_max_total_delta_rms"] == 0.06
    assert srmpgd_profile["tools"]["settings"]["srmpgd_max_lpips_loss"] == 0.15
    assert srmpgd_profile["tools"]["settings"]["srmpgd_crop_padding_px"] == 78
    assert srmpgd_profile["model"]["controlnet_conditioning_profile"] == "binary"
    assert srmpgd_profile["model"]["diffqrcoder_upstream_enabled"] is True
    assert srmpgd_profile["enabled"] is True
    assert robust_srmpgd_profile["enabled"] is True
    assert robust_srmpgd_profile["tools"]["settings"]["srmpgd_robust_blur_weight"] == 1.0
    assert robust_srmpgd_profile["tools"]["settings"]["srmpgd_robust_downscale_weight"] == 1.0

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
            "aesthetic_ok": True,
            "human_scan_result": "scannable",
            "human_scan_attempts": 3,
            "human_scan_successes": 2,
            "human_scan_device": "Pixel test — native camera",
            "prompt_fidelity_score": 5,
            "qr_discretion_score": 1,
            "overall_score": 4,
            "favorite": True,
            "notes": "smoke",
        },
    )
    assert rating.status_code == 200
    assert rating.json()["favorite"] is True
    assert rating.json()["aesthetic_ok"] is True
    assert rating.json()["human_scan_result"] == "scannable"
    assert rating.json()["human_scan_attempts"] == 3
    assert rating.json()["human_scan_successes"] == 2
    campaign_csv = client.get(f"/v1/lab/campaigns/{campaign_id}/results.csv")
    assert campaign_csv.status_code == 200
    assert "quality_brightness_mean" in campaign_csv.text.splitlines()[0]
    assert "human_scan_result" in campaign_csv.text.splitlines()[0]
    assert "human_scan_attempts" in campaign_csv.text.splitlines()[0]
    assert "human_scan_device" in campaign_csv.text.splitlines()[0]
    assert "provenance_final_image_sha256" in campaign_csv.text.splitlines()[0]
    artifacts = client.get(f"/v1/generations/{trial['generation_run_id']}/artifacts").json()
    assert artifacts[0]["name"] == "final"
    assert len(artifacts) == 1
