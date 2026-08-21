from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from prooftag_qr.qr import (
    adaptive_quiet_zone_color,
    diffqrcoder_module_error_rate,
    diffqrcoder_structure_metrics,
    functional_pattern_mask,
    generate_diffqrcoder_qr,
    generate_qr,
    module_error_breakdown,
    module_error_rate,
    prepare_scan_ready_image,
    repair_qr_modules,
    restore_quiet_zone,
)
from prooftag_qr.quality import image_sha256
from prooftag_qr.validation import (
    ConservativeQRVerifyScorer,
    Decoder,
    OpenCVDecoder,
    QRValidator,
    QRVerifyDecoder,
    WeChatQRCodeDecoder,
    compare_validation_to_reference,
    image_raster_sha256,
)


def test_reference_qr_decodes_exact_payload():
    payload = "https://example.prooftag.test/t/abc123"
    blueprint = generate_qr(payload, "Q")

    assert blueprint.version >= 1
    assert OpenCVDecoder().decode(blueprint.image) == payload
    assert module_error_rate(blueprint.image, blueprint) == 0.0


def test_wechat_decoder_requires_pinned_models_and_decodes(monkeypatch, tmp_path):
    payload = "https://ptag.io/t/wechat"
    for name in WeChatQRCodeDecoder.model_filenames:
        (tmp_path / name).write_bytes(b"model")

    class FakeDetector:
        def detectAndDecode(self, image):
            assert image.shape[2] == 3
            return (payload,), (np.zeros((4, 2)),)

    monkeypatch.setattr(
        "prooftag_qr.validation.cv2.wechat_qrcode_WeChatQRCode",
        lambda *paths: FakeDetector(),
        raising=False,
    )
    decoder = WeChatQRCodeDecoder(tmp_path)

    assert decoder.decode(Image.new("RGB", (32, 32), "white")) == payload


def test_reference_qr_passes_original_scenario():
    payload = "https://example.prooftag.test/t/abc123"
    blueprint = generate_qr(payload, "H")
    records = QRValidator(decoders=[OpenCVDecoder()]).validate(blueprint.image, payload)
    original = [record for record in records if record.scenario == "original"]

    assert original
    assert all(record.exact_payload_match for record in original)


def test_qr_verify_bridge_runs_all_presets_and_enforces_exact_payload():
    node_modules = Path("qr_verify_bridge/node_modules/qr-scanner-wechat")
    if not node_modules.is_dir():
        pytest.skip("run npm ci in qr_verify_bridge to enable the WASM integration test")
    payload = "https://ptag.io/t/qr-verify-test"
    image = generate_diffqrcoder_qr(payload, "M").image
    decoder = QRVerifyDecoder()
    try:
        records = QRValidator(decoders=[decoder]).validate(image, payload)
    finally:
        decoder.close()

    summary = compare_validation_to_reference(records, records)
    assert len(records) == 37
    assert records[0].scenario == "original"
    assert all(record.decoder == "qr_verify" for record in records)
    assert all(record.exact_payload_match for record in records)
    assert summary["qr_verify_any_exact"] is True
    assert summary["qr_verify_direct_exact"] is True
    assert summary["qr_verify_exact_presets"] == 37
    assert summary["qr_verify_tolerance_score"] == 1.0


class _WrongPayloadQRVerifyDecoder(QRVerifyDecoder):
    def __init__(self) -> None:
        pass

    def decode_presets(self, image: Image.Image):
        del image
        return [
            {
                "preset": "original" if index == 0 else f"preset_{index}",
                "text": "https://ptag.io/t/wrong",
                "latency_ms": 0.1,
            }
            for index in range(37)
        ]


def test_qr_verify_rejects_a_decoded_but_wrong_payload():
    decoder = _WrongPayloadQRVerifyDecoder()
    records = QRValidator(decoders=[decoder]).validate(
        Image.new("RGB", (64, 64), "white"),
        "https://ptag.io/t/expected",
    )

    assert len(records) == 37
    assert all(record.success for record in records)
    assert not any(record.exact_payload_match for record in records)


class _VaryingQRVerifyDecoder(QRVerifyDecoder):
    def __init__(self) -> None:
        self.calls = 0

    def decode_presets(self, image: Image.Image):
        del image
        exact_by_call = [
            {"original", "preset_1", "preset_2"},
            {"original", "preset_1"},
            {"preset_1", "preset_2"},
        ]
        exact = exact_by_call[self.calls % len(exact_by_call)]
        self.calls += 1
        return [
            {
                "preset": "original" if index == 0 else f"preset_{index}",
                "text": (
                    "https://ptag.io/t/repeated"
                    if ("original" if index == 0 else f"preset_{index}") in exact
                    else ""
                ),
                "latency_ms": 0.1,
            }
            for index in range(37)
        ]


def test_conservative_cache_uses_the_canonical_provenance_raster_hash():
    image = Image.new("RGB", (65, 63), (12, 34, 56))

    assert image_raster_sha256(image) == image_sha256(image)


def test_conservative_qr_verify_intersects_repeated_preset_results(tmp_path):
    decoder = _VaryingQRVerifyDecoder()
    scorer = ConservativeQRVerifyScorer(
        decoder,
        repetitions=3,
        cache_dir=tmp_path,
    )

    score = scorer.score(
        Image.new("RGB", (64, 64), "white"),
        "https://ptag.io/t/repeated",
    )

    assert decoder.calls == 3
    assert [run["exact_preset_count"] for run in score.runs] == [3, 2, 2]
    assert score.direct_exact_all_repetitions is False
    assert score.each_repetition_any_exact is True
    assert score.consistent_any_exact is True
    assert score.conservative_exact_presets == 1
    assert score.conservative_tolerance_score == pytest.approx(1 / 37)
    assert score.minimum_tolerance_score == pytest.approx(2 / 37)
    assert score.mean_tolerance_score == pytest.approx(7 / (3 * 37))
    assert score.maximum_tolerance_score == pytest.approx(3 / 37)
    assert score.unstable_preset_count == 2
    assert score.stable_preset_count == 35
    assert score.cache_hit is False
    assert score.cache_path is not None
    assert Path(score.cache_path).is_file()
    assert "https://ptag.io/t/repeated" not in Path(score.cache_path).read_text(
        encoding="utf-8"
    )
    assert all(
        "text" not in result
        for run in score.runs
        for result in run["preset_results"]
    )


class _CacheMustPreventDecode(QRVerifyDecoder):
    def __init__(self) -> None:
        self.calls = 0

    def decode_presets(self, image: Image.Image):
        del image
        self.calls += 1
        raise AssertionError("identical raster should have been served from cache")


def test_conservative_qr_verify_cache_is_reused_by_raster_payload_and_version(tmp_path):
    image = Image.new("RGB", (64, 64), "white")
    payload = "https://ptag.io/t/repeated"
    first = ConservativeQRVerifyScorer(
        _VaryingQRVerifyDecoder(),
        repetitions=3,
        cache_dir=tmp_path,
    ).score(image, payload)
    decoder = _CacheMustPreventDecode()
    second = ConservativeQRVerifyScorer(
        decoder,
        repetitions=3,
        cache_dir=tmp_path,
    ).score(image.copy(), payload)

    assert decoder.calls == 0
    assert second.cache_hit is True
    assert second.cache_key == first.cache_key
    assert second.image_sha256 == first.image_sha256
    assert second.payload_sha256 == first.payload_sha256
    assert second.conservative_tolerance_score == first.conservative_tolerance_score
    assert second.runs == first.runs


def test_conservative_qr_verify_cache_key_changes_with_raster_payload_and_version(tmp_path):
    decoder = _VaryingQRVerifyDecoder()
    image = Image.new("RGB", (64, 64), "white")
    payload = "https://ptag.io/t/repeated"
    first = ConservativeQRVerifyScorer(
        decoder,
        repetitions=3,
        cache_dir=tmp_path,
        engine_version="qr-verify@test-a",
    ).score(image, payload)
    second = ConservativeQRVerifyScorer(
        decoder,
        repetitions=3,
        cache_dir=tmp_path,
        engine_version="qr-verify@test-b",
    ).score(image, payload)
    changed_payload = ConservativeQRVerifyScorer(
        decoder,
        repetitions=3,
        cache_dir=tmp_path,
        engine_version="qr-verify@test-a",
    ).score(image, f"{payload}-other")
    changed_raster = ConservativeQRVerifyScorer(
        decoder,
        repetitions=3,
        cache_dir=tmp_path,
        engine_version="qr-verify@test-a",
    ).score(Image.new("RGB", (64, 64), "black"), payload)

    assert decoder.calls == 12
    assert len(
        {
            first.cache_key,
            second.cache_key,
            changed_payload.cache_key,
            changed_raster.cache_key,
        }
    ) == 4


def test_diffqrcoder_reference_uses_the_public_integer_geometry():
    payload = "https://ptag.io/t/lab01"
    blueprint = generate_diffqrcoder_qr(payload, "M")

    assert blueprint.image.size == (740, 740)
    assert blueprint.matrix.shape == (37, 37)
    assert blueprint.version == 3
    assert blueprint.border == 4
    assert OpenCVDecoder().decode(blueprint.image) == payload


def test_diffqrcoder_mer_uses_the_exact_580_pixel_core_geometry():
    blueprint = generate_diffqrcoder_qr("https://ptag.io/t/lab01", "M")
    core_matrix = blueprint.matrix[4:-4, 4:-4]
    core = Image.fromarray(np.where(core_matrix, 0, 255).astype(np.uint8)).resize(
        (580, 580),
        Image.Resampling.NEAREST,
    )
    candidate = Image.new("RGB", (736, 736), "white")
    candidate.paste(core.convert("RGB"), (78, 78))

    assert diffqrcoder_module_error_rate(
        candidate,
        blueprint,
        padding_px=78,
        module_size=20,
    ) == 0.0

    metrics = diffqrcoder_structure_metrics(
        candidate,
        blueprint,
        padding_px=78,
        module_size=20,
    )
    assert metrics["module_center_error_rate"] == 0.0
    assert metrics["functional_center_error_rate"] == 0.0
    assert metrics["center_confidence_p10"] == 1.0
    assert metrics["quiet_zone_dark_pixel_ratio"] == 0.0


class _ThresholdOnlyDecoder(Decoder):
    name = "threshold-only"

    def __init__(self, payload: str):
        self.payload = payload

    def decode(self, image: Image.Image) -> str:
        values = np.unique(np.asarray(image.convert("L")))
        return self.payload if set(values.tolist()) <= {0, 255} else ""


def test_phone_proxy_reports_preprocessing_without_changing_the_main_validator():
    payload = "https://ptag.io/t/phone-proxy"
    blueprint = generate_qr(payload, "M", size=256)
    low_contrast = np.where(
        np.asarray(blueprint.image.convert("L"))[..., None] < 128,
        80,
        180,
    ).astype(np.uint8)
    image = Image.fromarray(np.repeat(low_contrast, 3, axis=2))
    validator = QRValidator(decoders=[_ThresholdOnlyDecoder(payload)])

    regular = validator.validate(image, payload)
    proxy = validator.validate_phone_proxy(image, payload)

    assert not next(
        record for record in regular if record.scenario == "original"
    ).exact_payload_match
    assert len(proxy) == 1
    assert proxy[0].exact_payload_match is True
    assert proxy[0].scenario == "phone_proxy_original"
    assert proxy[0].parameters["selected_preprocessor"] in {
        "otsu_x2",
        "adaptive_x2",
    }


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


def test_adaptive_quiet_zone_keeps_a_light_version_of_the_artwork_palette():
    blueprint = generate_qr("https://example.prooftag.test/t/adaptive-zone", "M", size=128)
    painted = Image.new("RGB", blueprint.image.size, (32, 96, 176))

    color = adaptive_quiet_zone_color(
        painted,
        blueprint,
        minimum_luminance=0.82,
    )
    prepared = prepare_scan_ready_image(
        painted,
        blueprint,
        quiet_zone_mode="adaptive_light",
        quiet_zone_minimum_luminance=0.82,
    )
    luminance = np.dot(color, (0.299, 0.587, 0.114)) / 255
    count = blueprint.matrix.shape[0]
    border_px = round(blueprint.border * painted.width / count)

    assert color != (255, 255, 255)
    assert luminance >= 0.815
    assert prepared.getpixel((0, 0)) == color
    assert np.all(
        np.asarray(prepared)[border_px:-border_px, border_px:-border_px]
        == np.asarray(painted)[border_px:-border_px, border_px:-border_px]
    )


def test_functional_tonification_leaves_every_data_module_untouched():
    blueprint = generate_qr("https://example.prooftag.test/t/functional-only", "H", size=256)
    noise = np.random.default_rng(20260729).integers(
        0, 256, (256, 256, 3), dtype=np.uint8
    )
    prepared = prepare_scan_ready_image(
        Image.fromarray(noise),
        blueprint,
        quiet_zone_mode="none",
        functional_pattern_tone_factor=0.12,
    )
    result = np.asarray(prepared)
    protected = functional_pattern_mask(blueprint)
    count = protected.shape[0]
    changed_functional = 0

    for row in range(count):
        y0 = round(row * result.shape[0] / count)
        y1 = max(y0 + 1, round((row + 1) * result.shape[0] / count))
        for col in range(count):
            x0 = round(col * result.shape[1] / count)
            x1 = max(x0 + 1, round((col + 1) * result.shape[1] / count))
            changed = not np.array_equal(
                result[y0:y1, x0:x1],
                noise[y0:y1, x0:x1],
            )
            if protected[row, col]:
                changed_functional += int(changed)
            else:
                assert not changed

    breakdown = module_error_breakdown(prepared, blueprint)
    assert changed_functional == int(protected.sum())
    assert breakdown["functional"] == 0.0


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
