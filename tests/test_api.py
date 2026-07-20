def test_api_generation_reports_and_physical_validation(tmp_path, monkeypatch):
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
