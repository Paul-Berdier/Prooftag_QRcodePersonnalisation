from prooftag_qr.domain import ValidationRecord
from prooftag_qr.validation import summarize_validation_records


def _record(decoder: str, scenario: str, exact: bool) -> ValidationRecord:
    return ValidationRecord(
        decoder=decoder,
        scenario=scenario,
        success=exact,
        exact_payload_match=exact,
        latency_ms=1.0,
    )


def test_validation_summary_exposes_weak_decoder_and_scenario():
    records = [
        _record("opencv", "original", True),
        _record("zbar", "original", False),
        _record("opencv", "blur", True),
        _record("zbar", "blur", True),
    ]

    summary = summarize_validation_records(records)

    assert summary["decoder_pass_rates"] == {"opencv": 1.0, "zbar": 0.5}
    assert summary["scenario_pass_rates"] == {"blur": 1.0, "original": 0.5}
    assert summary["worst_decoder_pass_rate"] == 0.5
    assert summary["worst_scenario_pass_rate"] == 0.5
