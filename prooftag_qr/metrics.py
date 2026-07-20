from prometheus_client import Counter, Gauge, Histogram

RUNS = Counter(
    "prooftag_qr_runs_total",
    "Number of completed QR generation runs",
    ["backend", "status"],
)
ACTIVE_RUNS = Gauge("prooftag_qr_runs_active", "Number of QR generation runs in progress")
ATTEMPTS = Histogram(
    "prooftag_qr_attempts",
    "Number of attempts per completed run",
    buckets=(1, 2, 3, 4, 5, 8, 12, 20),
)
DURATION = Histogram(
    "prooftag_qr_duration_seconds",
    "Pipeline duration by stage",
    ["backend", "stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
VALIDATIONS = Counter(
    "prooftag_qr_validations_total",
    "QR validation outcomes",
    ["decoder", "scenario", "outcome"],
)
VALIDATION_DURATION = Histogram(
    "prooftag_qr_validation_duration_seconds",
    "Individual decoder latency",
    ["decoder"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)
SCAN_PASS_RATE = Histogram(
    "prooftag_qr_scan_pass_rate",
    "Share of validation cases decoding the exact payload",
    buckets=(0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
MODULE_ERROR_RATE = Histogram(
    "prooftag_qr_module_error_rate",
    "Estimated share of QR modules with wrong luminance",
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0),
)
IMAGE_QUALITY = Gauge(
    "prooftag_qr_image_quality_latest",
    "Latest image quality measurement; durable per-run values are stored in SQLite",
    ["metric"],
)
PHYSICAL_VALIDATIONS = Counter(
    "prooftag_qr_physical_validations_total",
    "Physical scan validation outcomes",
    ["outcome"],
)
MODEL_LOADS = Counter(
    "prooftag_qr_model_loads_total",
    "ControlNet model load outcomes",
    ["status"],
)
MODEL_LOAD_DURATION = Histogram(
    "prooftag_qr_model_load_duration_seconds",
    "Time spent loading the ControlNet pipeline by outcome",
    ["status"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1200),
)
MODEL_LOADED = Gauge(
    "prooftag_qr_model_loaded",
    "Whether the ControlNet pipeline is loaded in this process",
)
REPAIR_VARIANTS = Counter(
    "prooftag_qr_repair_variants_total",
    "QR repair variants validated by outcome",
    ["variant", "outcome"],
)
REPAIR_SELECTED = Counter(
    "prooftag_qr_repair_selected_total",
    "QR repair variants selected as the best candidate",
    ["variant"],
)
