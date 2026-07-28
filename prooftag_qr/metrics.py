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
REGENERATIONS = Counter(
    "prooftag_qr_regenerations_total",
    "New diffusion generations scheduled before global QR repair",
    ["reason"],
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
REPAIR_VARIANT_SCAN_PASS_RATE = Gauge(
    "prooftag_qr_repair_variant_scan_pass_rate",
    "Latest scan pass rate measured for each repair variant",
    ["variant"],
)
REPAIR_VARIANT_MODULE_ERROR_RATE = Gauge(
    "prooftag_qr_repair_variant_module_error_rate",
    "Latest module error rate measured for each repair variant",
    ["variant"],
)
REPAIR_VARIANT_IMAGE_QUALITY = Gauge(
    "prooftag_qr_repair_variant_image_quality",
    "Latest image quality measurement for each repair variant",
    ["variant", "metric"],
)
GUIDED_REDIFFUSIONS = Counter(
    "prooftag_qr_guided_rediffusions_total",
    "Second-pass guided diffusion outcomes",
    ["outcome"],
)
GUIDED_REDIFFUSION_DURATION = Histogram(
    "prooftag_qr_guided_rediffusion_duration_seconds",
    "Time spent in the guided img2img diffusion pass",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
)
GUIDED_REDIFFUSION_MODULE_ERROR_RATE = Gauge(
    "prooftag_qr_guided_rediffusion_module_error_rate",
    "Latest module error before and after guided rediffusion",
    ["stage"],
)
GUIDED_REDIFFUSION_IMAGE_CHANGE = Gauge(
    "prooftag_qr_guided_rediffusion_image_change",
    "Latest visual change caused by guided rediffusion",
    ["metric"],
)
SRPG_RUNS = Counter(
    "prooftag_qr_srpg_runs_total",
    "True in-denoising SRPG outcomes",
    ["outcome"],
)
SRPG_DURATION = Histogram(
    "prooftag_qr_srpg_duration_seconds",
    "Time spent in the SRPG DDIM loop",
    buckets=(1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120),
)
SRPG_MODULE_ERROR_RATE = Gauge(
    "prooftag_qr_srpg_module_error_rate",
    "Actual module error before and after SRPG",
    ["stage"],
)
SRPG_STEP_DIAGNOSTIC = Gauge(
    "prooftag_qr_srpg_step_diagnostic",
    "Diagnostics retained for every step of the latest SRPG run",
    ["step", "metric"],
)
SRPG_IMAGE_CHANGE = Gauge(
    "prooftag_qr_srpg_image_change",
    "Visual change caused by the latest SRPG run",
    ["metric"],
)
SRPG_PEAK_GPU_MEMORY_MIB = Gauge(
    "prooftag_qr_srpg_peak_gpu_memory_allocated_mib",
    "Peak CUDA memory allocated by the latest SRPG run in MiB",
)
SRPG_GRADIENT_CLIPS = Counter(
    "prooftag_qr_srpg_gradient_clips_total",
    "SRPG steps whose noise delta reached its safety cap",
)
LATENT_REFINEMENTS = Counter(
    "prooftag_qr_latent_refinements_total",
    "Latent SRL refinement outcomes",
    ["outcome"],
)
LATENT_REFINEMENT_DURATION = Histogram(
    "prooftag_qr_latent_refinement_duration_seconds",
    "Time spent optimizing a candidate in VAE latent space",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
)
LATENT_REFINEMENT_ITERATIONS = Histogram(
    "prooftag_qr_latent_refinement_iterations",
    "Number of SRL latent optimization iterations",
    buckets=(1, 2, 3, 4, 6, 8, 12, 16, 24, 40, 100),
)
LATENT_REFINEMENT_MODULE_ERROR_RATE = Gauge(
    "prooftag_qr_latent_refinement_module_error_rate",
    "Latest central-submodule error before and after latent optimization",
    ["stage"],
)
LATENT_REFINEMENT_LOSS = Gauge(
    "prooftag_qr_latent_refinement_loss",
    "Latest SRL and preservation objective components",
    ["component"],
)
LAB_CAMPAIGNS = Counter(
    "prooftag_qr_lab_campaigns_total",
    "Web laboratory campaigns by terminal outcome",
    ["status"],
)
LAB_TRIALS = Counter(
    "prooftag_qr_lab_trials_total",
    "Web laboratory trials by method and terminal outcome",
    ["method", "status"],
)
LAB_ACTIVE_CAMPAIGNS = Gauge(
    "prooftag_qr_lab_campaigns_active",
    "Web laboratory campaigns currently executing",
)
LAB_TRIAL_DURATION = Histogram(
    "prooftag_qr_lab_trial_duration_seconds",
    "End-to-end duration of a web laboratory trial",
    ["method"],
    buckets=(1, 5, 10, 30, 60, 120, 180, 300, 600),
)
LAB_RATINGS = Counter(
    "prooftag_qr_lab_ratings_total",
    "Human ratings recorded in the web laboratory",
)
LAB_QUALITY_SCORES = Counter(
    "prooftag_qr_lab_quality_scores_total",
    "CLIP and aesthetic scoring outcomes in the web laboratory",
    ["status"],
)
LAB_QUALITY_SCORE_DURATION = Histogram(
    "prooftag_qr_lab_quality_score_duration_seconds",
    "CPU duration of CLIP and aesthetic scoring in the web laboratory",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
