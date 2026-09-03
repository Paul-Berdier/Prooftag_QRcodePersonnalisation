import csv
import hashlib
import uuid
from pathlib import Path

from prooftag_qr.e045_phone_labels import import_captures


def test_phone_import_separates_valid_and_invalid_rows(tmp_path: Path):
    expected = hashlib.sha256(b"https://example.test").hexdigest()
    image = "1" * 64
    path = tmp_path / "captures.csv"
    fields = [
        "capture_id","image_sha256","image_path","experiment_id","prompt_id",
        "device_id","device_model","os_name","os_version","scanner_app",
        "scanner_version","captured_at_utc","display_device","display_width_px",
        "display_height_px","display_brightness_pct","ambient_lux","distance_cm",
        "yaw_deg","pitch_deg","roll_deg","attempt_index","frames_observed",
        "time_to_decode_ms","decoded_exact","decoded_payload_sha256",
        "expected_payload_sha256","notes"
    ]
    rows = []
    for index in range(1, 4):
        rows.append({
            "capture_id": str(uuid.uuid4()),
            "image_sha256": image,
            "device_id": "phone-a",
            "device_model": "test",
            "scanner_app": "native",
            "captured_at_utc": "2026-09-02T12:00:00+00:00",
            "attempt_index": index,
            "decoded_exact": "1" if index < 3 else "0",
            "decoded_payload_sha256": expected if index < 3 else "",
            "expected_payload_sha256": expected,
        })
    rows.append({
        "capture_id": "not-a-uuid",
        "image_sha256": "bad",
        "device_id": "phone-b",
        "device_model": "test",
        "scanner_app": "native",
        "captured_at_utc": "bad-date",
        "attempt_index": 1,
        "decoded_exact": "1",
        "expected_payload_sha256": expected,
    })
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = import_captures(path, tmp_path / "output")
    assert summary["valid_capture_count"] == 3
    assert summary["rejected_capture_count"] == 1
    assert summary["labeled_image_count"] == 1

    by_image = list(csv.DictReader(
        (tmp_path / "output/phone-labels-by-image.csv").open(
            "r", encoding="utf-8", newline=""
        )
    ))
    assert by_image[0]["all_devices_pass_2_of_3"] == "True"


def test_phone_import_uses_immutable_hash_directory():
    root = Path(__file__).resolve().parents[1]
    text = (root / "prooftag_qr/e045_foundation.py").read_text(encoding="utf-8")
    assert '"phone-imports" / input_sha[:16]' in text
    assert "PHONE_LATEST.json" in text
    assert "atomic_write_json" in text
    assert 'sub.add_parser("import-phone")' in text
