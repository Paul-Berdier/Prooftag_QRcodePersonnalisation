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
