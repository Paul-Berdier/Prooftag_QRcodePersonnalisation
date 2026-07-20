from prooftag_qr.qr import generate_qr, module_error_rate
from prooftag_qr.validation import OpenCVDecoder, QRValidator


def test_reference_qr_decodes_exact_payload():
    payload = "https://example.prooftag.test/t/abc123"
    blueprint = generate_qr(payload, "Q")

    assert blueprint.version >= 1
    assert OpenCVDecoder().decode(blueprint.image) == payload
    assert module_error_rate(blueprint.image, blueprint) == 0.0


def test_reference_qr_passes_original_scenario():
    payload = "https://example.prooftag.test/t/abc123"
    blueprint = generate_qr(payload, "H")
    records = QRValidator(decoders=[OpenCVDecoder()]).validate(blueprint.image, payload)
    original = [record for record in records if record.scenario == "original"]

    assert original
    assert all(record.exact_payload_match for record in original)
