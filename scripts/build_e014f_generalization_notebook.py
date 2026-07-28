"""Build the E014F unseen-context generalization notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "19_e014e_mechanism_window_ablation.ipynb"
TARGET = ROOT / "notebooks" / "20_e014f_unseen_generalization_cascade.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


template = json.loads(TEMPLATE.read_text(encoding="utf-8"))


def template_source(index: int) -> str:
    return "".join(template["cells"][index]["source"])


imports = template_source(2)
imports = imports.replace("import random\n", "import random\nimport platform\n")
imports = imports.replace(
    "from prooftag_qr.geometry import AlignedQR, aligned_module_diagnostics  # noqa: E402\n",
    "from prooftag_qr.blueprints import (  # noqa: E402\n"
    "    build_adaptive_blueprint, exact_mask_candidates,\n"
    "    grid_visibility_score, reference_cost,\n"
    ")\n"
    "from prooftag_qr.geometry import (  # noqa: E402\n"
    "    AlignedQR, aligned_module_diagnostics, generate_aligned_qr,\n"
    ")\n",
)
imports = imports.replace(
    "19_e014e_mechanism_window_ablation.ipynb",
    "20_e014f_unseen_generalization_cascade.ipynb",
)
pipeline_helpers = template_source(8)

cells = [
    markdown(
        """# E014F — généralisation inconnue et cascade de livraison

E014E a isolé le mécanisme utile : le masque fonctionnel produit l'essentiel du gain, deux pas
préservent l'image et quatre pas maximisent la robustesse. E014F ne recherche pas une nouvelle
astuce et n'entraîne pas de sélecteur. Il vérifie si cette zone 2/3/4 pas se généralise à des
prompts, graines et payloads absents de toutes les campagnes précédentes.

Chaque contexte est reconstruit depuis zéro : QR exact, Stage 1, blueprint adaptatif exact,
Stage 2 FreeQR `fusion_all`, douze réparations tardives appariées et validation exacte par trois
décodeurs sur treize scénarios.
"""
    ),
    markdown(
        """## Protocole préenregistré

```text
12 prompts inconnus × 2 graines × 6 payloads = 24 contextes
                         |
           Stage 1 + blueprint adaptatif exact
                         |
       Stage 2 FreeQR fusion_all, 40 pas (source)
                         |
  4 recettes × fenêtres {2, 3, 4} = 12 réparations
                         |
      16 contextes calibration / 8 contextes holdout
                         |
 fixed recipe | oracle | cascade stricte 39/39 | physique
```

Le holdout contient quatre prompts entiers. La Stage 2 finale ne sert jamais à choisir le
blueprint. Chaque réparation utilise une pipeline fraîche et aucune projection de pixels.
La campagne complète produit 288 réparations. Elle est longue mais reprenable candidat par
candidat.
"""
    ),
    code(imports),
    markdown("## 1. Modèles épinglés, contextes inconnus et recettes"),
    code(
        """EXPERIMENT_NAME = 'e014f-unseen-generalization-cascade-v1'
RESUME_RUN_NAME = None
CONTEXT_LIMIT = None  # smoke test uniquement, dans un run au nom différent

BASE_MODEL_REPO = 'fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned'
BASE_MODEL_FILE = 'cetusMix_Whalefall2_fp16.safetensors'
CONFIG_MODEL_REPO = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
CONTROLNET_MODEL = 'monster-labs/control_v1p_sd15_qrcode_monster'
CONTROLNET_SUBFOLDER = 'v2'
DIFFQRCODER_COMMIT = 'e24ea73ee2e13c7e6e87cb422e8b11784e70ae00'

BASE_STEPS = 40
STAGE1_STEPS = 40
SOURCE_STAGE2_STEPS = 40
RESCUE_STEP_COUNTS = [2, 3, 4]
GUIDANCE_SCALE = 7.5
CONTROLNET_SCALE = 1.35
SCANNING_GUIDANCE = 500.0
PERCEPTUAL_GUIDANCE = 3.0
SOURCE_FUSION_CHANNEL = 1
SOURCE_FUSION_ALPHA = 0.15
GLOBAL_FUSION_CHANNEL = 1
RESCUE_SEED_OFFSET = 400_014
NEGATIVE_PROMPT = 'easynegative, unreadable text, letters, watermark'
QR_VERSION = 3
QR_ECC = 'M'
QR_MODULE_SIZE = 20
CANVAS_SIZE = 736
ADAPTIVE_FRACTIONS = [0.22, 0.30, 0.38, 0.46, 0.55, 0.70, 0.85]
SAVE_SOURCE_PREVIEW_EVERY = 10
MAX_AESTHETIC_DROP = 0.75
MAX_MEAN_ABSOLUTE_CHANGE = 0.18

PROMPT_SPECS = [
    {'id': 'u01_poppy', 'family': 'simple', 'split': 'calibration',
     'text': 'A single crimson poppy on a charcoal background, soft studio light, minimalist botanical photograph.'},
    {'id': 'u02_vase', 'family': 'simple', 'split': 'calibration',
     'text': 'One moon-shaped ceramic vase on an indigo shelf, restrained Japanese product photography.'},
    {'id': 'u03_cabin', 'family': 'medium', 'split': 'calibration',
     'text': 'An alpine cabin beside a winding stream, pine trees and early autumn fog, detailed landscape painting.'},
    {'id': 'u04_courtyard', 'family': 'medium', 'split': 'calibration',
     'text': 'A Mediterranean courtyard with blue tiles, lemon trees and a stone fountain, warm editorial photograph.'},
    {'id': 'u05_peacock', 'family': 'detailed', 'split': 'calibration',
     'text': 'An Art Nouveau peacock mosaic with turquoise feathers, gold vines and opal flowers, intricate craftsmanship.'},
    {'id': 'u06_manuscript', 'family': 'detailed', 'split': 'calibration',
     'text': 'An illuminated manuscript forest with foxes, owls, mushrooms and curling ivy, highly detailed medieval art.'},
    {'id': 'u07_festival', 'family': 'complex', 'split': 'calibration',
     'text': 'A neon night street-food festival with lanterns, cooks, signs and crowds in rain, cinematic illustration.'},
    {'id': 'u08_harbor', 'family': 'complex', 'split': 'calibration',
     'text': 'An aerial Mediterranean harbor market with boats, striped awnings, people and stacked old houses.'},
    {'id': 'u09_shell', 'family': 'simple', 'split': 'holdout',
     'text': 'A solitary pearl shell resting on black velvet, dramatic museum lighting, elegant macro photograph.'},
    {'id': 'u10_orchard', 'family': 'medium', 'split': 'holdout',
     'text': 'A spring apple orchard with a narrow path, wooden gate and distant hills in luminous morning mist.'},
    {'id': 'u11_tapestry', 'family': 'detailed', 'split': 'holdout',
     'text': 'A Persian night-garden tapestry with pomegranates, gazelles, irises and constellations, intricate woven texture.'},
    {'id': 'u12_station', 'family': 'complex', 'split': 'holdout',
     'text': 'A grand retro-futurist railway station filled with travelers, clocks, glass arches and glowing kiosks.'},
]
SEED_VARIANTS = [51_001, 61_001]
PAYLOADS = [
    'https://ptag.io/t/f01', 'https://ptag.io/t/f02',
    'https://ptag.io/t/f03', 'https://ptag.io/t/f04',
    'https://ptag.io/t/f05', 'https://ptag.io/t/f06',
]
RECIPES = [
    {'id': 'mask_s15', 'mechanism': 'mask_only', 'global_alpha': 0.00, 'structural_strength': 0.15},
    {'id': 'combined_a06_s10', 'mechanism': 'combined', 'global_alpha': 0.06, 'structural_strength': 0.10},
    {'id': 'combined_a10_s15', 'mechanism': 'combined', 'global_alpha': 0.10, 'structural_strength': 0.15},
    {'id': 'combined_a15_s15', 'mechanism': 'combined', 'global_alpha': 0.15, 'structural_strength': 0.15},
]
CASCADE_ORDER = [
    ('combined_a10_s15', 2), ('combined_a06_s10', 2),
    ('mask_s15', 2), ('combined_a15_s15', 2),
    ('combined_a06_s10', 3), ('mask_s15', 3),
    ('combined_a10_s15', 3), ('combined_a15_s15', 3),
    ('combined_a06_s10', 4), ('mask_s15', 4),
    ('combined_a10_s15', 4), ('combined_a15_s15', 4),
]

contexts_spec = []
for prompt_index, prompt in enumerate(PROMPT_SPECS):
    for seed_index, seed_base in enumerate(SEED_VARIANTS):
        contexts_spec.append({
            **prompt,
            'context_id': f"{prompt['id']}_s{seed_index + 1}",
            'seed': seed_base + prompt_index * 97,
            'seed_variant': seed_index + 1,
            'payload': PAYLOADS[(prompt_index * 2 + seed_index) % len(PAYLOADS)],
        })
if CONTEXT_LIMIT is not None:
    contexts_spec = contexts_spec[:int(CONTEXT_LIMIT)]

assert RESCUE_STEP_COUNTS == [2, 3, 4]
assert len({item['id'] for item in RECIPES}) == 4
assert set(CASCADE_ORDER) == {
    (recipe['id'], steps)
    for recipe in RECIPES for steps in RESCUE_STEP_COUNTS
}
if CONTEXT_LIMIT is None:
    assert len(contexts_spec) == 24
    assert sum(item['split'] == 'calibration' for item in contexts_spec) == 16
    assert sum(item['split'] == 'holdout' for item in contexts_spec) == 8

if RESUME_RUN_NAME:
    RUN_DIR = Path('/data/notebook-runs') / RESUME_RUN_NAME
    if not RUN_DIR.is_dir():
        raise FileNotFoundError(RUN_DIR)
else:
    RUN_DIR = Path('/data/notebook-runs') / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + EXPERIMENT_NAME
    )
    RUN_DIR.mkdir(parents=True, exist_ok=False)
SOURCE_RESULTS_PATH = RUN_DIR / 'source-results.jsonl'
RESULTS_PATH = RUN_DIR / 'results.jsonl'
ERRORS_PATH = RUN_DIR / 'errors.jsonl'


def cached_main_revision(repo_id):
    ref = (
        Path('/cache/huggingface/hub')
        / ('models--' + repo_id.replace('/', '--'))
        / 'refs' / 'main'
    )
    return ref.read_text(encoding='utf-8').strip() if ref.exists() else None


def resolve_revision(repo_id):
    cached = cached_main_revision(repo_id)
    return cached if cached else model_info(repo_id).sha


revision_path = RUN_DIR / 'resolved-model-revisions.json'
if revision_path.exists():
    resolved_revisions = json.loads(revision_path.read_text(encoding='utf-8'))
else:
    resolved_revisions = {
        'base_model': resolve_revision(BASE_MODEL_REPO),
        'config_model': resolve_revision(CONFIG_MODEL_REPO),
        'controlnet': resolve_revision(CONTROLNET_MODEL),
    }
    revision_path.write_text(
        json.dumps(resolved_revisions, indent=2), encoding='utf-8'
    )
BASE_MODEL_PATH = hf_hub_download(
    repo_id=BASE_MODEL_REPO, filename=BASE_MODEL_FILE,
    revision=resolved_revisions['base_model'], cache_dir='/cache/huggingface',
)
BASE_CONFIG_PATH = snapshot_download(
    repo_id=CONFIG_MODEL_REPO, revision=resolved_revisions['config_model'],
    cache_dir='/cache/huggingface',
    allow_patterns=['**/*.json', '*.json', '*.txt', '**/*.txt', '**/*.model'],
)
pipeline_source = UPSTREAM_ROOT / 'diffqrcoder' / 'pipeline_diffqrcoder.py'
pipeline_source_sha256 = hashlib.sha256(pipeline_source.read_bytes()).hexdigest()

print('Sortie :', RUN_DIR)
print('Contextes :', len(contexts_spec))
print('Réparations attendues :', len(contexts_spec) * len(RECIPES) * len(RESCUE_STEP_COUNTS))
display(pd.DataFrame(contexts_spec)[
    ['context_id', 'family', 'split', 'seed', 'payload']
])
"""
    ),
    markdown("## 2. Validation, géométrie et persistance"),
    code(
        """validator = QRValidator()
decoder_names = [decoder.name for decoder in validator.decoders]
if decoder_names != ['opencv', 'zbar', 'zxingcpp']:
    raise RuntimeError(f'Trois décodeurs obligatoires : {decoder_names}')
quality_scorer = CLIPQualityScorer(Path('/cache'), device='cpu')


def jsonl_rows(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def append_jsonl(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\\n')
        stream.flush()


def validation_summary(image, payload):
    records = validator.validate(image, payload)
    summary = summarize_validation_records(records)
    originals = [item for item in records if item.scenario == 'original']
    passed = sum(item.exact_payload_match for item in records)
    original_passed = sum(item.exact_payload_match for item in originals)
    return {
        'passed': passed, 'total': len(records),
        'pass_rate': passed / len(records),
        'strict_all': passed == len(records),
        'original_passed': original_passed,
        'original_total': len(originals),
        'original_all': original_passed == len(originals),
        'worst_decoder_pass_rate': summary['worst_decoder_pass_rate'],
        'worst_scenario_pass_rate': summary['worst_scenario_pass_rate'],
    }, [asdict(item) for item in records]


def original_probe(image, payload):
    rows = []
    for decoder in validator.decoders:
        started = time.perf_counter()
        decoded, decoder_error = decode_safely(decoder, image)
        rows.append({
            'decoder': decoder.name,
            'exact_payload_match': decoded == payload,
            'latency_ms': (time.perf_counter() - started) * 1000,
            'decoder_error': decoder_error,
        })
    passed = sum(row['exact_payload_match'] for row in rows)
    return {
        'original_probe_passed': passed,
        'original_probe_total': len(rows),
        'original_probe_all': passed == len(rows),
    }, rows


def score_image(image, prompt, payload):
    probe, probe_rows = original_probe(image, payload)
    validation, records = validation_summary(image, payload)
    try:
        quality = asdict(quality_scorer.score(image, prompt))
        quality_error = None
    except Exception as exc:
        quality = {
            'clip_similarity': None, 'clip_score': None,
            'clip_aesthetic': None,
        }
        quality_error = f'{type(exc).__name__}: {exc}'
    return probe, probe_rows, validation, records, quality, quality_error


def tensor_sha256(tensor):
    value = tensor.detach().to('cpu').contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def image_sha256(image):
    return hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()


def structural_mask_image(aligned):
    size, padding = aligned.canvas_size, aligned.padding_px
    module_size, core_size = aligned.module_size, aligned.core_size
    mask = np.ones((size, size), dtype=np.uint8) * 255
    mask[padding:padding + core_size, padding:padding + core_size] = 0
    functional = functional_pattern_mask(aligned.core_blueprint)
    for row, col in np.argwhere(functional):
        y0 = padding + int(row) * module_size
        x0 = padding + int(col) * module_size
        mask[y0:y0 + module_size, x0:x0 + module_size] = 255
    return Image.fromarray(mask, mode='L')


def make_gif(folder, output):
    paths = sorted(folder.glob('*.jpg'))
    frames = [Image.open(path).convert('RGB').resize((512, 512)) for path in paths]
    if frames:
        frames[0].save(
            output, save_all=True, append_images=frames[1:],
            duration=400, loop=0,
        )
    for frame in frames:
        frame.close()


def source_dir(context_id):
    return RUN_DIR / 'sources' / context_id


def candidate_dir(context_id, recipe_id, rescue_steps):
    return RUN_DIR / 'candidates' / context_id / f'{recipe_id}-steps{int(rescue_steps):02d}'


def source_rows():
    return jsonl_rows(SOURCE_RESULTS_PATH)


def result_rows():
    return jsonl_rows(RESULTS_PATH)


def completed_source_ids():
    complete = set()
    for row in source_rows():
        context_id = row['context_id']
        if context_id in complete:
            raise RuntimeError(f'Source dupliquée : {context_id}')
        folder = RUN_DIR / row['relative_output_dir']
        required = [
            folder / 'stage1.safetensors', folder / 'stage1.png',
            folder / 'blueprint.png', folder / 'matrix.npy',
            folder / 'functional-mask.png', folder / 'source.png',
            folder / 'source.safetensors', folder / 'source-validations.json',
            folder / 'metadata.json',
        ]
        if not all(path.exists() for path in required):
            raise RuntimeError(f'Source indexée mais incomplète : {context_id}')
        complete.add(context_id)
    return complete


def completed_candidate_keys():
    complete = set()
    for row in result_rows():
        key = row['context_id'], row['recipe'], int(row['rescue_steps'])
        if key in complete:
            raise RuntimeError(f'Candidat dupliqué : {key}')
        folder = RUN_DIR / row['relative_output_dir']
        required = [
            folder / 'final.png', folder / 'final.safetensors',
            folder / 'validations.json', folder / 'original-probe.json',
            folder / 'trace.json',
        ]
        if not all(path.exists() for path in required):
            raise RuntimeError(f'Candidat indexé mais incomplet : {key}')
        complete.add(key)
    return complete
"""
    ),
    markdown("## 3. Pipeline DiffQRCoder et fenêtre DDIM exacte"),
    code(pipeline_helpers),
]

# The remaining cells are appended below to keep this builder reviewable.
cells.extend([
    markdown("## 4. Construire une source inconnue complète"),
    code(
        """def source_preview_callback(label, folder, total_steps):
    folder.mkdir(parents=True, exist_ok=True)
    trace = []

    def callback(pipe_ref, step_index, timestep, callback_kwargs):
        trace.append({
            'step': int(step_index), 'timestep': int(timestep),
            'latent_sha256': tensor_sha256(callback_kwargs['latents']),
        })
        if step_index % SAVE_SOURCE_PREVIEW_EVERY == 0 or step_index == total_steps - 1:
            preview = decode_latent(pipe_ref, callback_kwargs['latents'])
            preview.save(folder / f'{step_index:03d}.jpg', quality=88)
            clear_output(wait=True)
            display(Markdown(f'**{label} — étape {step_index + 1}/{total_steps}**'))
            display(preview.resize((384, 384)))
        return callback_kwargs

    return callback, trace


def source_fusion_callback(label, folder, blueprint_latent, paired_noise):
    folder.mkdir(parents=True, exist_ok=True)
    trace = []

    def callback(pipe_ref, step_index, timestep, callback_kwargs):
        latent = callback_kwargs['latents'].detach()
        next_timestep = (
            pipe_ref.scheduler.timesteps[step_index + 1]
            if step_index + 1 < len(pipe_ref.scheduler.timesteps)
            else torch.tensor(0, device=latent.device, dtype=pipe_ref.scheduler.timesteps.dtype)
        )
        with torch.no_grad():
            noised_blueprint = pipe_ref.scheduler.add_noise(
                blueprint_latent, paired_noise, next_timestep.reshape(1)
            )
            channel = SOURCE_FUSION_CHANNEL
            latent[:, channel:channel + 1] = (
                (1 - SOURCE_FUSION_ALPHA) * latent[:, channel:channel + 1]
                + SOURCE_FUSION_ALPHA * noised_blueprint[:, channel:channel + 1]
            )
        trace.append({
            'step': int(step_index),
            'timestep_before_step': int(timestep),
            'target_timestep_after_step': int(next_timestep),
            'fusion_channel': SOURCE_FUSION_CHANNEL,
            'fusion_alpha': SOURCE_FUSION_ALPHA,
            'latent_sha256': tensor_sha256(latent),
        })
        if (
            step_index % SAVE_SOURCE_PREVIEW_EVERY == 0
            or step_index == SOURCE_STAGE2_STEPS - 1
        ):
            preview = decode_latent(pipe_ref, latent)
            preview.save(folder / f'{step_index:03d}.jpg', quality=88)
            clear_output(wait=True)
            display(Markdown(
                f'**{label} — étape {step_index + 1}/{SOURCE_STAGE2_STEPS}**'
            ))
            display(preview.resize((384, 384)))
        callback_kwargs['latents'] = latent
        return callback_kwargs

    return callback, trace


@torch.no_grad()
def stage2_paired_inputs(pipeline, stage1_tensor, seed):
    encoded = (
        pipeline.vae.encode(stage1_tensor * 2 - 1).latent_dist.mode()
        * pipeline.vae.config.scaling_factor
    )
    generator = torch.Generator(device='cuda').manual_seed(seed + 10_000)
    noise = torch.randn(
        encoded.shape, generator=generator,
        device='cuda', dtype=encoded.dtype,
    )
    pipeline.scheduler.set_timesteps(SOURCE_STAGE2_STEPS, device='cuda')
    initial = pipeline.scheduler.add_noise(
        encoded, noise, pipeline.scheduler.timesteps[:1]
    )
    return initial.detach(), noise.detach()


def choose_stage1_control(spec):
    candidates = []
    for mask_pattern in range(8):
        aligned = generate_aligned_qr(
            spec['payload'], version=QR_VERSION, error_correction=QR_ECC,
            mask_pattern=mask_pattern, module_size=QR_MODULE_SIZE,
            canvas_size=CANVAS_SIZE,
        )
        validation, records = validation_summary(aligned.image, spec['payload'])
        candidates.append((validation, records, aligned))
    candidates.sort(
        key=lambda item: (
            item[0]['strict_all'], item[0]['pass_rate'],
            item[0]['worst_decoder_pass_rate'],
            item[0]['worst_scenario_pass_rate'], -item[2].mask_pattern,
        ),
        reverse=True,
    )
    if not candidates[0][0]['strict_all']:
        raise RuntimeError(f"Aucun QR témoin strict pour {spec['context_id']}")
    return candidates[0]


def choose_blueprint(stage1_image, spec):
    exact_rows = []
    for candidate in exact_mask_candidates(
        spec['payload'], stage1_image, version=QR_VERSION,
        error_correction=QR_ECC, module_size=QR_MODULE_SIZE,
        canvas_size=CANVAS_SIZE,
    ):
        validation, _ = validation_summary(candidate.aligned.image, spec['payload'])
        exact_rows.append((validation, candidate))
    exact_rows.sort(
        key=lambda item: (
            item[0]['strict_all'], item[0]['pass_rate'],
            -item[1].reference_cost, -item[1].grid_visibility,
        ),
        reverse=True,
    )
    exact = exact_rows[0][1].aligned
    adaptive_rows = []
    for minimum_fraction in ADAPTIVE_FRACTIONS:
        adaptive = build_adaptive_blueprint(
            stage1_image, exact, minimum_data_fraction=minimum_fraction
        )
        validation, records = validation_summary(adaptive.image, spec['payload'])
        adaptive_rows.append((validation, records, adaptive, minimum_fraction))
    adaptive_rows.sort(
        key=lambda item: (
            item[0]['strict_all'], item[0]['pass_rate'],
            -item[2].reference_cost, -item[2].grid_visibility,
        ),
        reverse=True,
    )
    validation, records, adaptive, fraction = adaptive_rows[0]
    if validation['strict_all']:
        aligned = AlignedQR(
            image=adaptive.image, core_matrix=exact.core_matrix.copy(),
            version=exact.version, error_correction=exact.error_correction,
            mask_pattern=exact.mask_pattern, module_size=exact.module_size,
            padding_px=exact.padding_px, canvas_size=exact.canvas_size,
            payload=exact.payload,
        )
        method = 'adaptive_exact_payload'
        cost, visibility = adaptive.reference_cost, adaptive.grid_visibility
    else:
        aligned = exact
        validation, records = validation_summary(aligned.image, spec['payload'])
        fraction, method = None, 'exact_mask_fallback'
        cost = exact_rows[0][1].reference_cost
        visibility = exact_rows[0][1].grid_visibility
    if not validation['strict_all']:
        raise RuntimeError(
            f"Blueprint non strict pour {spec['context_id']}: {validation}"
        )
    return aligned, {
        'method': method, 'adaptive_fraction': fraction,
        'reference_cost': float(cost), 'grid_visibility': float(visibility),
        'validation': validation, 'records': records,
    }


def build_source(spec):
    if spec['context_id'] in completed_source_ids():
        print('SKIP source', spec['context_id'])
        return
    output_dir = source_dir(spec['context_id'])
    if output_dir.exists():
        print('Nettoyage source partielle :', output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    pipeline = load_pipeline()
    completed = False
    result1 = result2 = stage1_tensor = final_latent = None
    blueprint_latent = initial = noise = None
    try:
        control_validation, control_records, control = choose_stage1_control(spec)
        control.image.save(output_dir / 'stage1-control.png')
        (output_dir / 'stage1-control-validations.json').write_text(
            json.dumps(control_records, indent=2), encoding='utf-8'
        )
        seed_everything(spec['seed'])
        callback1, trace1 = source_preview_callback(
            f"{spec['context_id']} / Stage 1",
            output_dir / 'stage1-frames', STAGE1_STEPS,
        )
        started = time.perf_counter()
        result1 = pipeline._run_stage1(
            prompt=spec['text'], qrcode=control.image,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=STAGE1_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator(device='cuda').manual_seed(spec['seed']),
            controlnet_conditioning_scale=CONTROLNET_SCALE,
            callback_on_step_end=callback1,
            callback_on_step_end_tensor_inputs=['latents'],
            output_type='pt',
        )
        stage1_tensor = result1.images.detach()
        stage1_image = pipeline.image_processor.numpy_to_pil(
            pipeline.image_processor.pt_to_numpy(stage1_tensor.detach())
        )[0].convert('RGB')
        stage1_duration = time.perf_counter() - started
        stage1_image.save(output_dir / 'stage1.png')
        save_file(
            {'stage1': stage1_tensor.cpu().contiguous()},
            str(output_dir / 'stage1.safetensors'),
        )
        (output_dir / 'stage1-trace.json').write_text(
            json.dumps(trace1, indent=2), encoding='utf-8'
        )
        make_gif(output_dir / 'stage1-frames', output_dir / 'stage1.gif')

        aligned, blueprint_meta = choose_blueprint(stage1_image, spec)
        aligned.image.save(output_dir / 'blueprint.png')
        np.save(output_dir / 'matrix.npy', aligned.core_matrix)
        structural_mask_image(aligned).save(output_dir / 'functional-mask.png')
        (output_dir / 'blueprint-validations.json').write_text(
            json.dumps(blueprint_meta['records'], indent=2), encoding='utf-8'
        )

        release_guidance(pipeline)
        blueprint_latent = encode_image(pipeline, aligned.image)
        initial, noise = stage2_paired_inputs(pipeline, stage1_tensor, spec['seed'])
        callback2, trace2 = source_fusion_callback(
            f"{spec['context_id']} / Stage 2 fusion_all",
            output_dir / 'source-frames', blueprint_latent, noise,
        )
        started = time.perf_counter()
        result2 = pipeline._run_stage2(
            prompt=spec['text'], qrcode=aligned.image,
            qrcode_module_size=aligned.module_size,
            qrcode_padding=aligned.padding_px,
            ref_image=stage1_tensor, negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=SOURCE_STAGE2_STEPS,
            guidance_scale=GUIDANCE_SCALE, eta=0.0,
            generator=torch.Generator(device='cuda').manual_seed(spec['seed'] + 10_000),
            latents=initial.clone(),
            controlnet_conditioning_scale=CONTROLNET_SCALE,
            scanning_robust_guidance_scale=SCANNING_GUIDANCE,
            perceptual_guidance_scale=PERCEPTUAL_GUIDANCE,
            callback_on_step_end=callback2,
            callback_on_step_end_tensor_inputs=['latents'],
            output_type='latent',
        )
        final_latent = result2.images.detach()
        source_image = decode_latent(pipeline, final_latent)
        source_duration = time.perf_counter() - started
        source_image.save(output_dir / 'source.png')
        save_file(
            {'latents': final_latent.cpu().contiguous()},
            str(output_dir / 'source.safetensors'),
        )
        (output_dir / 'source-trace.json').write_text(
            json.dumps(trace2, indent=2), encoding='utf-8'
        )
        make_gif(output_dir / 'source-frames', output_dir / 'source.gif')
        probe, probe_rows, validation, records, quality, quality_error = score_image(
            source_image, spec['text'], spec['payload']
        )
        (output_dir / 'original-probe.json').write_text(
            json.dumps(probe_rows, indent=2), encoding='utf-8'
        )
        (output_dir / 'source-validations.json').write_text(
            json.dumps(records, indent=2), encoding='utf-8'
        )
        metadata = {
            **spec, 'version': aligned.version,
            'ecc': aligned.error_correction, 'mask_pattern': aligned.mask_pattern,
            'module_size': aligned.module_size, 'padding_px': aligned.padding_px,
            'canvas_size': aligned.canvas_size,
            'blueprint': {
                key: value for key, value in blueprint_meta.items()
                if key != 'records'
            },
            'stage1_duration_s': stage1_duration,
            'source_stage2_duration_s': source_duration,
        }
        (output_dir / 'metadata.json').write_text(
            json.dumps(metadata, indent=2), encoding='utf-8'
        )
        row = {
            'relative_output_dir': str(output_dir.relative_to(RUN_DIR)),
            **spec, 'blueprint_method': blueprint_meta['method'],
            'blueprint_adaptive_fraction': blueprint_meta['adaptive_fraction'],
            'blueprint_reference_cost': blueprint_meta['reference_cost'],
            'blueprint_grid_visibility': blueprint_meta['grid_visibility'],
            'stage1_duration_s': stage1_duration,
            'source_stage2_duration_s': source_duration,
            'stage1_tensor_sha256': tensor_sha256(stage1_tensor),
            'blueprint_sha256': image_sha256(aligned.image),
            'source_initial_latent_sha256': tensor_sha256(initial),
            'source_noise_sha256': tensor_sha256(noise),
            'source_final_latent_sha256': tensor_sha256(final_latent),
            'post_diffusion_pixel_projection': False,
            **probe, **validation, **quality,
            **aligned_module_diagnostics(source_image, aligned),
            'quality_error': quality_error,
        }
        append_jsonl(SOURCE_RESULTS_PATH, row)
        completed = True
        print(
            'SOURCE', spec['context_id'], validation['passed'], '/',
            validation['total'], 'original', probe['original_probe_passed'],
            '/', probe['original_probe_total'],
        )
    except Exception as exc:
        append_jsonl(ERRORS_PATH, {
            'phase': 'source', 'context_id': spec['context_id'],
            'error_type': type(exc).__name__, 'error': str(exc),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        raise
    finally:
        release_guidance(pipeline)
        del result1, result2, stage1_tensor, final_latent
        del blueprint_latent, initial, noise, pipeline
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if not completed and output_dir.exists():
            shutil.rmtree(output_dir)
"""
    ),
    markdown("## 5. Générer ou reprendre les 24 sources"),
    code(
        """for spec in contexts_spec:
    build_source(spec)

source_result_rows = source_rows()
if len(source_result_rows) != len(contexts_spec):
    raise RuntimeError(
        f'Sources incomplètes : {len(source_result_rows)}/{len(contexts_spec)}'
    )
source_frame = pd.DataFrame(source_result_rows)
source_frame.to_csv(RUN_DIR / 'source-results.csv', index=False)
display(source_frame[[
    'context_id', 'family', 'split', 'payload', 'blueprint_method',
    'passed', 'total', 'original_probe_passed',
    'clip_aesthetic', 'clip_score',
]])
"""
    ),
])

cells.extend([
    markdown("## 8. Généralisation fixe, oracle et cascade stricte"),
    code(
        """frame = pd.DataFrame(rows)
frame.to_csv(RUN_DIR / 'candidate-results.csv', index=False)

aggregate_rows = []
for split in ['calibration', 'holdout', 'all']:
    split_frame = frame if split == 'all' else frame[frame.split == split]
    for recipe in RECIPES:
        for rescue_steps in RESCUE_STEP_COUNTS:
            part = split_frame[
                (split_frame.recipe == recipe['id'])
                & (split_frame.rescue_steps == rescue_steps)
            ]
            if part.empty:
                continue
            aggregate_rows.append({
                'split': split,
                'candidate_id': f"{recipe['id']}-steps{rescue_steps:02d}",
                'recipe': recipe['id'],
                'global_alpha': recipe['global_alpha'],
                'structural_strength': recipe['structural_strength'],
                'rescue_steps': rescue_steps,
                'contexts': len(part),
                'original_3of3_contexts': int(part.original_probe_all.sum()),
                'strict_39of39_contexts': int(part.strict_all.sum()),
                'strict_rate': float(part.strict_all.mean()),
                'mean_ssr': float(part.pass_rate.mean()),
                'minimum_ssr': float(part.pass_rate.min()),
                'mean_clip_aesthetic': float(part.clip_aesthetic.mean()),
                'mean_clip_score': float(part.clip_score.mean()),
                'maximum_aesthetic_drop': float(part.aesthetic_drop.max()),
                'maximum_mean_absolute_change': float(part.mean_absolute_change.max()),
                'low_damage_gate': bool(
                    part.original_probe_all.all()
                    and (part.aesthetic_drop <= MAX_AESTHETIC_DROP).all()
                    and (part.mean_absolute_change <= MAX_MEAN_ABSOLUTE_CHANGE).all()
                ),
            })

aggregate_frame = pd.DataFrame(aggregate_rows)
aggregate_frame.to_csv(RUN_DIR / 'fixed-aggregates.csv', index=False)
display(aggregate_frame.sort_values(
    ['split', 'strict_rate', 'mean_ssr', 'mean_clip_aesthetic'],
    ascending=[True, False, False, False],
))


def fixed_rank(row):
    return (
        int(row['original_3of3_contexts']),
        int(row['strict_39of39_contexts']),
        float(row['minimum_ssr']), float(row['mean_ssr']),
        float(row['mean_clip_aesthetic']),
        -float(row['maximum_mean_absolute_change']),
    )


calibration_aggregates = aggregate_frame[
    aggregate_frame.split == 'calibration'
].to_dict('records')
calibration_winner = max(calibration_aggregates, key=fixed_rank)
holdout_winner = aggregate_frame[
    (aggregate_frame.split == 'holdout')
    & (aggregate_frame.candidate_id == calibration_winner['candidate_id'])
].iloc[0].to_dict()


def simulate_cascade(split):
    cascade_rows = []
    part = frame[frame.split == split]
    for context_id in sorted(part.context_id.unique()):
        candidates = {
            (row['recipe'], int(row['rescue_steps'])): row
            for row in part[part.context_id == context_id].to_dict('records')
        }
        selected = None
        for attempt, key in enumerate(CASCADE_ORDER, start=1):
            row = candidates[key]
            if bool(row['strict_all']):
                selected = {**row, 'attempts': attempt, 'accepted': True}
                break
        if selected is None:
            selected = {
                **candidates[CASCADE_ORDER[-1]],
                'attempts': len(CASCADE_ORDER), 'accepted': False,
            }
        selected['oracle_has_strict'] = any(
            bool(row['strict_all']) for row in candidates.values()
        )
        selected['cascade_missed_oracle'] = bool(
            selected['oracle_has_strict'] and not selected['accepted']
        )
        cascade_rows.append(selected)
    return cascade_rows


cascade_rows = simulate_cascade('calibration') + simulate_cascade('holdout')
cascade_frame = pd.DataFrame(cascade_rows)
cascade_frame.to_csv(RUN_DIR / 'cascade-results.csv', index=False)

cascade_summary = []
for split, part in cascade_frame.groupby('split'):
    cascade_summary.append({
        'split': split, 'contexts': len(part),
        'accepted_strict': int(part.accepted.sum()),
        'acceptance_rate': float(part.accepted.mean()),
        'mean_attempts': float(part.attempts.mean()),
        'oracle_strict_contexts': int(part.oracle_has_strict.sum()),
        'cascade_missed_oracle': int(part.cascade_missed_oracle.sum()),
        'accepted_mean_clip_aesthetic': (
            float(part.loc[part.accepted, 'clip_aesthetic'].mean())
            if part.accepted.any() else None
        ),
    })
cascade_summary_frame = pd.DataFrame(cascade_summary)
cascade_summary_frame.to_csv(RUN_DIR / 'cascade-summary.csv', index=False)
display(cascade_summary_frame)

failure_rows = []
for row in rows:
    records = json.loads(
        (RUN_DIR / row['relative_output_dir'] / 'validations.json')
        .read_text(encoding='utf-8')
    )
    for record in records:
        if not record['exact_payload_match']:
            failure_rows.append({
                'context_id': row['context_id'], 'prompt_id': row['prompt_id'],
                'family': row['family'], 'split': row['split'],
                'candidate_id': row['candidate_id'],
                'decoder': record['decoder'], 'scenario': record['scenario'],
                'decoder_error': record.get('decoder_error'),
            })
pd.DataFrame(failure_rows).to_csv(RUN_DIR / 'failure-catalog.csv', index=False)

figure, axes = plt.subplots(1, 2, figsize=(17, 7))
all_aggregates = aggregate_frame[aggregate_frame.split == 'all']
for recipe in [item['id'] for item in RECIPES]:
    part = all_aggregates[all_aggregates.recipe == recipe].sort_values('rescue_steps')
    axes[0].plot(part.rescue_steps, part.mean_ssr, marker='o', label=recipe)
    axes[1].plot(
        part.rescue_steps, part.mean_clip_aesthetic,
        marker='o', label=recipe,
    )
axes[0].set(
    title='E014F — SSR moyen sur contextes inconnus',
    xlabel='Pas tardifs', ylabel='SSR moyen', ylim=(0, 1),
)
axes[1].set(
    title='E014F — CLIP-aesthetic moyen',
    xlabel='Pas tardifs', ylabel='CLIP-aesthetic',
)
for axis in axes:
    axis.set_xticks(RESCUE_STEP_COUNTS)
    axis.grid(alpha=0.25)
    axis.legend()
figure.tight_layout()
figure.savefig(RUN_DIR / 'generalization-summary.png', dpi=160)
display(figure)
"""
    ),
    markdown("## 9. Décision, physique, manifeste et archive"),
    code(
        """def aggregate_dict(split, candidate_id):
    selected = aggregate_frame[
        (aggregate_frame.split == split)
        & (aggregate_frame.candidate_id == candidate_id)
    ]
    if len(selected) != 1:
        raise RuntimeError(f'Agrégat absent ou dupliqué : {split}/{candidate_id}')
    return {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in selected.iloc[0].to_dict().items()
    }


balanced_holdout = aggregate_dict('holdout', 'combined_a10_s15-steps02')
robust_holdout = aggregate_dict('holdout', 'combined_a15_s15-steps04')
cascade_holdout = next(row for row in cascade_summary if row['split'] == 'holdout')
if (
    cascade_holdout['acceptance_rate'] == 1.0
    and cascade_holdout['cascade_missed_oracle'] == 0
):
    status = 'software_cascade_candidate_pending_physical'
elif cascade_holdout['acceptance_rate'] >= 0.75:
    status = 'partial_generalization'
else:
    status = 'generalization_rejected'

decision = {
    'status': status, 'contexts': len(contexts_spec),
    'prompts': len({item['id'] for item in contexts_spec}),
    'payloads': len({item['payload'] for item in contexts_spec}),
    'calibration_contexts': sum(
        item['split'] == 'calibration' for item in contexts_spec
    ),
    'holdout_contexts': sum(item['split'] == 'holdout' for item in contexts_spec),
    'calibration_selected_fixed': calibration_winner,
    'calibration_selected_fixed_holdout': {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in holdout_winner.items()
    },
    'preregistered_balanced_holdout': balanced_holdout,
    'preregistered_robust_holdout': robust_holdout,
    'preregistered_cascade_order': CASCADE_ORDER,
    'cascade_holdout': cascade_holdout,
    'delivery_gate': 'exact payload, 39/39 software validations',
    'selector_training_allowed': False,
    'post_diffusion_pixel_projection': False,
    'physical_validation_required': True,
    'production_claim_allowed': False,
}
(RUN_DIR / 'DECISION.json').write_text(
    json.dumps(decision, indent=2), encoding='utf-8'
)
print('Décision :', decision)

physical_rows = []
for row in cascade_rows:
    if row['split'] != 'holdout':
        continue
    physical_rows.append({
        'context_id': row['context_id'], 'prompt_id': row['prompt_id'],
        'seed': row['seed'], 'payload': row['payload'],
        'accepted_software': row['accepted'],
        'candidate_id': row['candidate_id'],
        'image': f"{row['relative_output_dir']}/final.png",
        'software_original': (
            f"{row['original_probe_passed']}/{row['original_probe_total']}"
        ),
        'software_robust': f"{row['passed']}/{row['total']}",
        'phone_model': '', 'distance_cm': '', 'screen_or_print': '',
        'lighting': '', 'attempts': '', 'successes': '', 'notes': '',
    })
pd.DataFrame(physical_rows).to_csv(
    RUN_DIR / 'physical-validation-template.csv', index=False
)

runtime_versions = {
    'python': sys.version, 'platform': platform.platform(),
    'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
    'diffusers': importlib.metadata.version('diffusers'),
    'transformers': importlib.metadata.version('transformers'),
    'lpips': importlib.metadata.version('lpips'),
}
manifest = {
    'experiment': EXPERIMENT_NAME,
    'created_at': datetime.now(timezone.utc).isoformat(),
    'contexts': contexts_spec, 'context_limit': CONTEXT_LIMIT,
    'models': {
        'base_repo': BASE_MODEL_REPO, 'base_file': BASE_MODEL_FILE,
        'base_revision': resolved_revisions['base_model'],
        'config_repo': CONFIG_MODEL_REPO,
        'config_revision': resolved_revisions['config_model'],
        'controlnet_repo': CONTROLNET_MODEL,
        'controlnet_subfolder': CONTROLNET_SUBFOLDER,
        'controlnet_revision': resolved_revisions['controlnet'],
        'diffqrcoder_commit': DIFFQRCODER_COMMIT,
        'pipeline_source_sha256': pipeline_source_sha256,
    },
    'runtime_versions': runtime_versions,
    'qr': {
        'version': QR_VERSION, 'ecc': QR_ECC,
        'module_size': QR_MODULE_SIZE, 'canvas_size': CANVAS_SIZE,
        'payloads': PAYLOADS,
    },
    'source_generation': {
        'stage1_steps': STAGE1_STEPS,
        'stage2_steps': SOURCE_STAGE2_STEPS,
        'blueprint': 'adaptive exact-payload selected before Stage 2; exact legal-mask fallback',
        'freeqr_channel': SOURCE_FUSION_CHANNEL,
        'freeqr_alpha': SOURCE_FUSION_ALPHA,
    },
    'rescue': {
        'recipes': RECIPES, 'step_counts': RESCUE_STEP_COUNTS,
        'fresh_pipeline_per_candidate': True,
        'paired_input_audit': 'paired-input-audit.json',
    },
    'cascade_order': CASCADE_ORDER,
    'post_diffusion_pixel_projection': False,
    'decision': decision,
    'claim': (
        'Preregistered unseen prompt/seed/payload confirmation. '
        'Software validation only; not physical production proof.'
    ),
}
(RUN_DIR / 'manifest.json').write_text(
    json.dumps(manifest, indent=2), encoding='utf-8'
)

report_lines = [
    '# E014F — rapport automatique', '',
    f"- Statut : `{status}`",
    f"- Contextes : `{len(contexts_spec)}`",
    f"- Candidat fixe calibration : `{calibration_winner['candidate_id']}`",
    f"- Holdout fixe : `{holdout_winner['strict_39of39_contexts']}/"
    f"{holdout_winner['contexts']}` stricts, SSR `{holdout_winner['mean_ssr']:.4f}`",
    f"- Cascade holdout : `{cascade_holdout['accepted_strict']}/"
    f"{cascade_holdout['contexts']}` livrables à 39/39",
    f"- Tentatives moyennes : `{cascade_holdout['mean_attempts']:.2f}`", '',
    'Aucune affirmation physique ou production avant les tests physiques.',
]
(RUN_DIR / 'REPORT.md').write_text(
    '\\n'.join(report_lines) + '\\n', encoding='utf-8'
)

for context in contexts.values():
    context['source_latent_cpu'] = None
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print('VRAM après nettoyage GiB :', torch.cuda.memory_allocated() / 2**30)
archive = shutil.make_archive(str(RUN_DIR), 'gztar', RUN_DIR.parent, RUN_DIR.name)
print('Archive :', archive)
print('Serveur Linux : POD=$(kubectl get pod -n qr-core -l app=prooftag-qr-notebook -o jsonpath="{.items[0].metadata.name}")')
print(f'Serveur Linux : kubectl cp -n qr-core "$POD:{archive}" "$HOME/{Path(archive).name}"')
print(f'PowerShell PC : scp paul@pcIA:~/{Path(archive).name} "$HOME/Downloads/"')
"""
    ),
])

cells.extend([
    markdown("## 6. Charger les sources figées et préparer la réparation"),
    code(
        """def load_source_context(spec):
    folder = source_dir(spec['context_id'])
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    blueprint = Image.open(folder / 'blueprint.png').convert('RGB')
    matrix = np.load(folder / 'matrix.npy').astype(np.uint8)
    aligned = AlignedQR(
        image=blueprint, core_matrix=matrix, version=metadata['version'],
        error_correction=metadata['ecc'], mask_pattern=metadata['mask_pattern'],
        module_size=metadata['module_size'], padding_px=metadata['padding_px'],
        canvas_size=metadata['canvas_size'], payload=spec['payload'],
    )
    source_row = next(
        row for row in source_result_rows
        if row['context_id'] == spec['context_id']
    )
    return {
        **spec, 'aligned': aligned, 'blueprint_image': blueprint,
        'structural_mask': Image.open(folder / 'functional-mask.png').convert('L'),
        'source_image': Image.open(folder / 'source.png').convert('RGB'),
        'source_latent_cpu': load_file(
            str(folder / 'source.safetensors'), device='cpu'
        )['latents'],
        'source_row': source_row,
    }


contexts = {
    spec['context_id']: load_source_context(spec) for spec in contexts_spec
}


def rescue_callback(
    context, recipe, rescue_steps, output_dir,
    blueprint_latent, rescue_noise, structural_mask,
):
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    trace = []

    def callback(pipe_ref, step_index, timestep, callback_kwargs):
        latent = callback_kwargs['latents'].detach()
        before = latent.clone()
        next_timestep = (
            pipe_ref.scheduler.timesteps[step_index + 1]
            if step_index + 1 < len(pipe_ref.scheduler.timesteps)
            else torch.tensor(0, device=latent.device, dtype=pipe_ref.scheduler.timesteps.dtype)
        )
        with torch.no_grad():
            noised_blueprint = pipe_ref.scheduler.add_noise(
                blueprint_latent, rescue_noise, next_timestep.reshape(1)
            )
            global_alpha = float(recipe['global_alpha'])
            if global_alpha > 0:
                channel = GLOBAL_FUSION_CHANNEL
                latent[:, channel:channel + 1] = (
                    (1 - global_alpha) * latent[:, channel:channel + 1]
                    + global_alpha * noised_blueprint[:, channel:channel + 1]
                )
            structural_strength = float(recipe['structural_strength'])
            if structural_strength > 0:
                alpha = structural_mask * structural_strength
                latent = latent * (1 - alpha) + noised_blueprint * alpha
        delta = latent.float() - before.float()
        trace.append({
            'step': int(step_index),
            'timestep_before_step': int(timestep),
            'target_timestep_after_step': int(next_timestep),
            'global_fusion_channel': GLOBAL_FUSION_CHANNEL,
            'global_fusion_alpha': float(recipe['global_alpha']),
            'structural_strength': float(recipe['structural_strength']),
            'structural_mask_mean': float(structural_mask.float().mean()),
            'latent_delta_rms': float(delta.square().mean().sqrt()),
            'latent_sha256': tensor_sha256(latent),
            'post_diffusion_pixel_projection': False,
        })
        preview = decode_latent(pipe_ref, latent)
        preview.save(frames_dir / f'{step_index:03d}.jpg', quality=90)
        clear_output(wait=True)
        display(Markdown(
            f"**{context['context_id']} / {recipe['id']} — "
            f"pas {step_index + 1}/{rescue_steps}**"
        ))
        display(preview.resize((384, 384)))
        callback_kwargs['latents'] = latent
        return callback_kwargs

    return callback, trace


def run_candidate(context, recipe, rescue_steps):
    key = context['context_id'], recipe['id'], int(rescue_steps)
    if key in completed_candidate_keys():
        print('SKIP candidat', key)
        return
    output_dir = candidate_dir(context['context_id'], recipe['id'], rescue_steps)
    if output_dir.exists():
        print('Nettoyage candidat partiel :', output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    pipeline = load_pipeline()
    completed = False
    result = final_latent = source_latent = None
    blueprint_latent = source_tensor = initial = noise = mask = None
    try:
        release_guidance(pipeline)
        source_latent = context['source_latent_cpu'].to('cuda', dtype=torch.float16)
        source_tensor = pil_to_tensor(context['source_image'])
        blueprint_latent = encode_image(pipeline, context['blueprint_image'])
        mask = latent_structural_mask(context['structural_mask'], source_latent)
        timesteps = late_schedule(pipeline, rescue_steps)
        rescue_seed = context['seed'] + RESCUE_SEED_OFFSET
        seed_everything(rescue_seed)
        generator = torch.Generator(device='cuda').manual_seed(rescue_seed)
        noise = torch.randn(
            source_latent.shape, generator=generator,
            device='cuda', dtype=source_latent.dtype,
        )
        initial = pipeline.scheduler.add_noise(
            source_latent, noise, timesteps[:1]
        ).detach()
        callback, trace = rescue_callback(
            context, recipe, rescue_steps, output_dir,
            blueprint_latent, noise, mask,
        )
        started = time.perf_counter()
        result = pipeline._run_stage2(
            prompt=context['text'], qrcode=context['blueprint_image'],
            qrcode_module_size=context['aligned'].module_size,
            qrcode_padding=context['aligned'].padding_px,
            ref_image=source_tensor, negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=None,
            timesteps=[int(value) for value in timesteps.detach().cpu().tolist()],
            guidance_scale=GUIDANCE_SCALE, eta=0.0,
            generator=torch.Generator(device='cuda').manual_seed(rescue_seed),
            latents=initial.clone(),
            controlnet_conditioning_scale=CONTROLNET_SCALE,
            scanning_robust_guidance_scale=SCANNING_GUIDANCE,
            perceptual_guidance_scale=PERCEPTUAL_GUIDANCE,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=['latents'],
            output_type='latent',
        )
        final_latent = result.images.detach()
        image = decode_latent(pipeline, final_latent)
        duration = time.perf_counter() - started
        image.save(output_dir / 'final.png')
        save_file(
            {'latents': final_latent.cpu().contiguous()},
            str(output_dir / 'final.safetensors'),
        )
        (output_dir / 'trace.json').write_text(
            json.dumps(trace, indent=2), encoding='utf-8'
        )
        make_gif(output_dir / 'frames', output_dir / 'diffusion.gif')
        probe, probe_rows, validation, records, quality, quality_error = score_image(
            image, context['text'], context['payload']
        )
        (output_dir / 'original-probe.json').write_text(
            json.dumps(probe_rows, indent=2), encoding='utf-8'
        )
        (output_dir / 'validations.json').write_text(
            json.dumps(records, indent=2), encoding='utf-8'
        )
        source_quality = context['source_row']
        row = {
            'relative_output_dir': str(output_dir.relative_to(RUN_DIR)),
            'context_id': context['context_id'], 'prompt_id': context['id'],
            'family': context['family'], 'split': context['split'],
            'prompt': context['text'], 'seed': context['seed'],
            'seed_variant': context['seed_variant'], 'payload': context['payload'],
            'recipe': recipe['id'], 'mechanism': recipe['mechanism'],
            'global_alpha': float(recipe['global_alpha']),
            'structural_strength': float(recipe['structural_strength']),
            'rescue_steps': int(rescue_steps),
            'candidate_id': f"{recipe['id']}-steps{rescue_steps:02d}",
            'late_timesteps': [
                int(value) for value in timesteps.detach().cpu().tolist()
            ],
            'source_passed': int(source_quality['passed']),
            'source_total': int(source_quality['total']),
            'source_original_passed': int(source_quality['original_probe_passed']),
            'source_clip_aesthetic': source_quality['clip_aesthetic'],
            'source_clip_score': source_quality['clip_score'],
            'source_latent_sha256': tensor_sha256(source_latent),
            'blueprint_latent_sha256': tensor_sha256(blueprint_latent),
            'rescue_noise_sha256': tensor_sha256(noise),
            'structural_mask_sha256': tensor_sha256(mask),
            'initial_rescue_latent_sha256': tensor_sha256(initial),
            'final_latent_sha256': tensor_sha256(final_latent),
            'duration_s': duration, 'post_diffusion_pixel_projection': False,
            **probe, **validation, **quality,
            **aligned_module_diagnostics(image, context['aligned']),
            **image_change_metrics(image, context['source_image']),
            'aesthetic_drop': (
                float(source_quality['clip_aesthetic']) - float(quality['clip_aesthetic'])
                if source_quality.get('clip_aesthetic') is not None
                and quality.get('clip_aesthetic') is not None else None
            ),
            'quality_error': quality_error,
        }
        append_jsonl(RESULTS_PATH, row)
        completed = True
        print(
            context['context_id'], row['candidate_id'],
            validation['passed'], '/', validation['total'],
            'original', probe['original_probe_passed'], '/',
            probe['original_probe_total'],
        )
    except Exception as exc:
        append_jsonl(ERRORS_PATH, {
            'phase': 'rescue', 'context_id': context['context_id'],
            'recipe': recipe['id'], 'rescue_steps': int(rescue_steps),
            'error_type': type(exc).__name__, 'error': str(exc),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        raise
    finally:
        release_guidance(pipeline)
        del result, final_latent, source_latent
        del blueprint_latent, source_tensor, initial, noise, mask, pipeline
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if not completed and output_dir.exists():
            shutil.rmtree(output_dir)
"""
    ),
    markdown("## 7. Campagne appariée 4 recettes × 2/3/4 pas"),
    code(
        """for context_id in [item['context_id'] for item in contexts_spec]:
    context = contexts[context_id]
    for rescue_steps in RESCUE_STEP_COUNTS:
        for recipe in RECIPES:
            print('Candidat :', context_id, rescue_steps, recipe['id'])
            run_candidate(context, recipe, rescue_steps)

rows = result_rows()
expected = len(contexts_spec) * len(RECIPES) * len(RESCUE_STEP_COUNTS)
if len(rows) != expected:
    raise RuntimeError(f'Campagne incomplète : {len(rows)}/{expected}')

pairing_audit = {}
for context_id in [item['context_id'] for item in contexts_spec]:
    for rescue_steps in RESCUE_STEP_COUNTS:
        subset = [
            row for row in rows
            if row['context_id'] == context_id
            and int(row['rescue_steps']) == rescue_steps
        ]
        audit = {
            'source_latent_hashes': sorted({row['source_latent_sha256'] for row in subset}),
            'rescue_noise_hashes': sorted({row['rescue_noise_sha256'] for row in subset}),
            'initial_rescue_latent_hashes': sorted({
                row['initial_rescue_latent_sha256'] for row in subset
            }),
            'structural_mask_hashes': sorted({row['structural_mask_sha256'] for row in subset}),
            'late_timestep_schedules': sorted({tuple(row['late_timesteps']) for row in subset}),
        }
        if any(len(values) != 1 for values in audit.values()):
            raise RuntimeError(
                f'Entrées non appariées {context_id}/{rescue_steps}: {audit}'
            )
        pairing_audit[f'{context_id}/steps{rescue_steps:02d}'] = audit
(RUN_DIR / 'paired-input-audit.json').write_text(
    json.dumps(pairing_audit, indent=2), encoding='utf-8'
)
print('Campagne complète :', len(rows), 'réparations')
"""
    ),
])

# The builder appends the analysis block before the rescue block for easier
# maintenance above; restore the actual execution order in the notebook.
base_and_source_cells = cells[:13]
analysis_cells = cells[13:17]
rescue_cells = cells[17:21]
cells = base_and_source_cells + rescue_cells + analysis_cells
assert "## 6." in "".join(cells[13]["source"])
assert "## 8." in "".join(cells[17]["source"])

notebook = {
    "cells": cells,
    "metadata": template["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(TARGET)
