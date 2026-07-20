import json
import logging

from prooftag_qr.logging import JsonFormatter


def test_variant_log_preserves_attempt_quality_and_failures():
    record = logging.LogRecord(
        name="prooftag_qr.service",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="repair_variant_validated",
        args=(),
        exc_info=None,
    )
    record.attempt = 2
    record.seed = 43
    record.quality_metrics = {"changed_pixel_ratio": 0.42}
    record.validation_failures = [
        {"decoder": "zbar", "scenario": "downscale_75", "outcome": "not_detected"}
    ]

    payload = json.loads(JsonFormatter().format(record))

    assert payload["attempt"] == 2
    assert payload["seed"] == 43
    assert payload["quality_metrics"]["changed_pixel_ratio"] == 0.42
    assert payload["validation_failures"][0]["scenario"] == "downscale_75"
