import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "run_id",
            "backend",
            "status",
            "duration_ms",
            "attempt",
            "attempts",
            "seed",
            "repair_variant",
            "scan_pass_rate",
            "module_error_rate",
            "exact_payload_match",
            "quality_metrics",
            "validation_failures",
            "iterations",
            "steps",
            "scheduler_steps",
            "strength",
            "controlnet_scale",
            "initial_module_error_rate",
            "control_module_error_rate",
            "final_module_error_rate",
            "best_observed_module_error_rate",
            "changed_pixel_ratio",
            "mask_coverage",
            "mean_absolute_change",
            "unprojected_changed_pixel_ratio",
            "unprojected_mean_absolute_change",
            "best_observed_mean_absolute_change",
            "srl",
            "preservation_loss",
            "improved",
            "accepted",
            "converged",
            "rejection_reason",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
