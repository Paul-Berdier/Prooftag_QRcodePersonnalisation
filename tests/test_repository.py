from datetime import UTC, datetime

from prooftag_qr.domain import AttemptRecord, RunRecord, ValidationRecord
from prooftag_qr.repository import RunRepository


def test_repository_round_trip(tmp_path):
    repository = RunRepository(tmp_path / "runs.sqlite3")
    run = RunRecord(
        id="run-1",
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        status="accepted",
        backend="qr",
        prompt="test",
        payload_hash="hash",
        seed=42,
        attempts=1,
        scan_pass_rate=1.0,
        exact_payload_match=True,
        module_error_rate=0.0,
        total_ms=12.5,
        validations=[ValidationRecord("opencv", "original", True, True, 3.2)],
        attempt_details=[AttemptRecord(1, 42, 2.0, 3.2, 1.0, 0.0, True)],
        quality_metrics={"brightness_mean": 0.5},
    )

    repository.save(run)
    restored = repository.get("run-1")

    assert restored is not None
    assert restored.status == "accepted"
    assert restored.validations[0].exact_payload_match
    assert restored.attempt_details[0].seed == 42
    assert restored.quality_metrics["brightness_mean"] == 0.5
    assert repository.summary()["acceptance_rate"] == 1.0

    physical = repository.add_physical_validation(
        run.id,
        {
            "device": "Pixel 7",
            "operating_system": "Android",
            "scanner": "native-camera",
            "print_profile": "laser",
            "material": "paper",
            "size_mm": 25.0,
            "lighting": "office",
            "distance_cm": 20.0,
            "angle_degrees": 5.0,
            "scan_latency_ms": 450.0,
            "outcome": "exact",
            "decoded_hash": "hash",
            "notes": "",
        },
    )
    assert physical["outcome"] == "exact"
    assert repository.list_physical_validations(run.id)[0]["device"] == "Pixel 7"
