import numpy as np
from PIL import Image

from prooftag_qr.qr import (
    functional_pattern_mask,
    generate_qr,
    module_error_breakdown,
    module_error_rate,
    repair_qr_modules,
    restore_quiet_zone,
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


def test_restore_quiet_zone_preserves_the_artistic_core_and_clears_only_the_margin():
    blueprint = generate_qr("https://example.prooftag.test/t/quiet-zone", "M", size=128)
    painted = Image.new("RGB", blueprint.image.size, (12, 34, 56))
    restored = restore_quiet_zone(painted, blueprint)
    source = np.asarray(painted)
    result = np.asarray(restored)
    count = blueprint.matrix.shape[0]
    border = blueprint.border
    left = round(border * painted.width / count)
    right = round((count - border) * painted.width / count)
    top = round(border * painted.height / count)
    bottom = round((count - border) * painted.height / count)

    assert np.array_equal(result[top:bottom, left:right], source[top:bottom, left:right])
    assert np.all(result[:top] == 255)
    assert np.all(result[bottom:] == 255)
    assert np.all(result[:, :left] == 255)
    assert np.all(result[:, right:] == 255)
    assert module_error_breakdown(restored, blueprint)["quiet_zone"] == 0.0


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


def test_repair_rejects_an_invalid_confidence_margin():
    blueprint = generate_qr("https://example.prooftag.test/t/confidence", "H")

    try:
        repair_qr_modules(
            blueprint.image,
            blueprint,
            center_scale=0.85,
            confidence_margin=128,
        )
    except ValueError as exc:
        assert str(exc) == ("confidence_margin must be between 0 (inclusive) and 128 (exclusive)")
    else:
        raise AssertionError("An invalid confidence margin must fail")


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


def test_perceptual_repair_feathers_edges_and_preserves_functional_texture():
    payload = "https://example.prooftag.test/t/perceptual-repair"
    blueprint = generate_qr(payload, "H")
    noise = np.random.default_rng(2027).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    source = Image.fromarray(noise)

    repaired = repair_qr_modules(
        source,
        blueprint,
        center_scale=0.85,
        incorrect_only=True,
        preserve_tone=True,
        confidence_margin=64,
        tone_factor=0.35,
        edge_feather=0.22,
        preserve_functional_tone=True,
    )
    records = QRValidator(decoders=[OpenCVDecoder()]).validate(repaired, payload)
    repaired_array = np.asarray(repaired)
    changed = np.abs(repaired_array.astype(np.int16) - noise.astype(np.int16))

    assert all(record.exact_payload_match for record in records)
    assert module_error_rate(repaired, blueprint) == 0.0
    assert np.any(repaired_array[:40] != 255)
    assert changed.mean() < 75
