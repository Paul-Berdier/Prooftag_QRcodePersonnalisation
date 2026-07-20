import numpy as np
from PIL import Image

from prooftag_qr.qr import (
    functional_pattern_mask,
    generate_qr,
    module_error_rate,
    repair_qr_modules,
)
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


def test_repair_locks_patterns_and_recovers_a_noisy_artwork():
    payload = "https://example.prooftag.test/t/repair-test"
    blueprint = generate_qr(payload, "H")
    noise = np.random.default_rng(42).integers(0, 256, (512, 512, 3), dtype=np.uint8)

    repaired = repair_qr_modules(
        Image.fromarray(noise), blueprint, center_scale=0.85, incorrect_only=False
    )
    records = QRValidator(decoders=[OpenCVDecoder()]).validate(repaired, payload)
    protected = functional_pattern_mask(blueprint)

    assert protected[0].all()
    assert protected[:, 0].all()
    assert protected[blueprint.border : blueprint.border + 9, blueprint.border].all()
    assert module_error_rate(repaired, blueprint) == 0.0
    assert all(record.exact_payload_match for record in records)


def test_repair_rejects_an_invalid_center_scale():
    blueprint = generate_qr("https://example.prooftag.test/t/scale", "H")

    try:
        repair_qr_modules(blueprint.image, blueprint, center_scale=1.01)
    except ValueError as exc:
        assert str(exc) == "center_scale must be between 0 and 1"
    else:
        raise AssertionError("An invalid repair scale must fail")


def test_tonal_repair_keeps_texture_while_remaining_robust():
    payload = "https://example.prooftag.test/t/tonal-repair"
    blueprint = generate_qr(payload, "H")
    noise = np.random.default_rng(2026).integers(0, 256, (512, 512, 3), dtype=np.uint8)

    repaired = repair_qr_modules(
        Image.fromarray(noise),
        blueprint,
        center_scale=0.95,
        preserve_tone=True,
    )
    records = QRValidator(decoders=[OpenCVDecoder()]).validate(repaired, payload)
    gray = np.asarray(repaired.convert("L"))
    clipped_ratio = float(((gray <= 3) | (gray >= 252)).mean())

    assert all(record.exact_payload_match for record in records)
    assert module_error_rate(repaired, blueprint) == 0.0
    assert clipped_ratio < 0.6
