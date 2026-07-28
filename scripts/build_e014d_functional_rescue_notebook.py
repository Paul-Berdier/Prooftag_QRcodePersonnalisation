"""Build E014D late functional-pattern rediffusion notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "17_e014b_multicontext_generalization.ipynb"
TARGET = ROOT / "notebooks" / "18_e014d_functional_late_rediffusion.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
imports = "".join(template["cells"][2]["source"])
imports = imports.replace(
    "from prooftag_qr.geometry import AlignedQR, aligned_module_diagnostics  # noqa: E402\n",
    "from prooftag_qr.geometry import AlignedQR, aligned_module_diagnostics  # noqa: E402\n"
    "from prooftag_qr.qr import functional_pattern_mask  # noqa: E402\n"
    "from prooftag_qr.quality import image_change_metrics  # noqa: E402\n",
)
imports = imports.replace(
    "from prooftag_qr.validation import QRValidator, summarize_validation_records  # noqa: E402\n",
    "from prooftag_qr.validation import (  # noqa: E402\n"
    "    QRValidator, decode_safely, summarize_validation_records,\n"
    ")\n",
)
imports = imports.replace(
    "17_e014b_multicontext_generalization.ipynb",
    "18_e014d_functional_late_rediffusion.ipynb",
)

cells = [
    markdown(
        """# E014D — réparation fonctionnelle tardive après fusion latente

E014B v3 a fait progresser le SSR global de 15,8 % à 52,8 %, mais aucune image fusionnée
n'a passé les trois décodeurs sur l'original. Ce notebook conserve exactement la fusion gagnante
et teste une seule nouvelle hypothèse : **verrouiller la géométrie fonctionnelle pendant une
courte rediffusion à faible bruit**.

La sortie finale reste une sortie de diffusion. Aucun collage binaire ni projection de pixels
n'est effectué après le dernier pas.
"""
    ),
    markdown(
        """## Plan expérimental préenregistré

```text
meilleur fusion_all existant
        |
        +-- contrôle : aucune nouvelle diffusion
        |
        +-- force 0,15 --+
        +-- force 0,30 --+--> 8 pas DDIM tardifs, pipeline fraîche
        +-- force 0,45 --+
                              |
                              v
          fusion canal 1 + masque fonctionnel latent à chaque pas
                              |
                              v
                 original 3/3 puis validation 39/39
```

Le masque contient la quiet zone, les finders et séparateurs, les timings, les informations de
format/version et les alignments. Les modules de données ne sont pas verrouillés. Les trois forces
sont fixées avant le calcul GPU : ce notebook n'est pas une recherche Optuna.

Une recette fixe n'est promue que si elle passe l'original 3/3 dans les quatre contextes, ne
réduit pas `p3_detailed`, améliore le SSR moyen et respecte le budget esthétique. Une sélection
différente par prompt reste un diagnostic, pas une recette générale.
"""
    ),
    code(imports),
    markdown("## 1. Sources, modèles épinglés et répertoire de sortie"),
    code(
        """EXPERIMENT_NAME = 'e014d-functional-late-rediffusion-v1'
E014A_RUN_DIR = None
E014B_V2_RUN_DIR = None
E014B_V3_RUN_DIR = None
RESUME_RUN_NAME = None
CONTEXT_IDS = ['p1_simple', 'p2_medium', 'p3_detailed', 'p4_complex']

BASE_MODEL_REPO = 'fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned'
BASE_MODEL_FILE = 'cetusMix_Whalefall2_fp16.safetensors'
CONFIG_MODEL_REPO = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
CONTROLNET_MODEL = 'monster-labs/control_v1p_sd15_qrcode_monster'
CONTROLNET_SUBFOLDER = 'v2'
DIFFQRCODER_COMMIT = 'e24ea73ee2e13c7e6e87cb422e8b11784e70ae00'

BASE_STEPS = 40
RESCUE_STEPS = 8
GUIDANCE_SCALE = 7.5
CONTROLNET_SCALE = 1.35
SCANNING_GUIDANCE = 500.0
PERCEPTUAL_GUIDANCE = 3.0
GLOBAL_FUSION_CHANNEL = 1
GLOBAL_FUSION_ALPHA = 0.15
RESCUE_SEED_OFFSET = 200_014
MAX_AESTHETIC_DROP = 0.75
MAX_MEAN_ABSOLUTE_CHANGE = 0.18
NEGATIVE_PROMPT = 'easynegative, unreadable text, letters, watermark'


def latest_run(pattern):
    candidates = sorted(Path('/data/notebook-runs').glob(pattern))
    if not candidates:
        raise FileNotFoundError(f'Aucun run pour {pattern}')
    return candidates[-1]


E014A_RUN_DIR = Path(E014A_RUN_DIR) if E014A_RUN_DIR else latest_run(
    '*-e014a-deterministic-blueprint-pairing-v2'
)
E014B_V2_RUN_DIR = Path(E014B_V2_RUN_DIR) if E014B_V2_RUN_DIR else latest_run(
    '*-e014b-statistical-freeqr-confirmation-v2'
)
E014B_V3_RUN_DIR = Path(E014B_V3_RUN_DIR) if E014B_V3_RUN_DIR else latest_run(
    '*-e014b-multicontext-generalization-v3'
)

source_manifests = {
    'e014a': json.loads((E014A_RUN_DIR / 'manifest.json').read_text(encoding='utf-8')),
    'e014b_v2': json.loads((E014B_V2_RUN_DIR / 'manifest.json').read_text(encoding='utf-8')),
    'e014b_v3': json.loads((E014B_V3_RUN_DIR / 'manifest.json').read_text(encoding='utf-8')),
}
v3_models = source_manifests['e014b_v3']['models']
if v3_models['base_repo'] != BASE_MODEL_REPO:
    raise RuntimeError(f"Base E014B v3 inattendue : {v3_models['base_repo']}")
if v3_models['controlnet_repo'] != CONTROLNET_MODEL:
    raise RuntimeError(f"ControlNet E014B v3 inattendu : {v3_models['controlnet_repo']}")

if RESUME_RUN_NAME:
    RUN_DIR = Path('/data/notebook-runs') / RESUME_RUN_NAME
    if not RUN_DIR.is_dir():
        raise FileNotFoundError(RUN_DIR)
else:
    RUN_DIR = Path('/data/notebook-runs') / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + EXPERIMENT_NAME
    )
    RUN_DIR.mkdir(parents=True)
RESULTS_PATH = RUN_DIR / 'results.jsonl'


def cached_main_revision(repo_id):
    ref = (
        Path('/cache/huggingface/hub')
        / ('models--' + repo_id.replace('/', '--'))
        / 'refs' / 'main'
    )
    return ref.read_text(encoding='utf-8').strip() if ref.exists() else None


def resolve_revision(repo_id):
    cached = cached_main_revision(repo_id)
    if cached:
        return cached
    return model_info(repo_id).sha


revision_path = RUN_DIR / 'resolved-model-revisions.json'
if revision_path.exists():
    resolved_revisions = json.loads(revision_path.read_text(encoding='utf-8'))
else:
    resolved_revisions = {
        'base_model': v3_models['base_revision'],
        'config_model': resolve_revision(CONFIG_MODEL_REPO),
        'controlnet': v3_models['controlnet_revision'],
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

print('E014A :', E014A_RUN_DIR)
print('E014B v2 :', E014B_V2_RUN_DIR)
print('E014B v3 :', E014B_V3_RUN_DIR)
print('Révisions :', resolved_revisions)
print('Sortie :', RUN_DIR)
"""
    ),
    markdown("## 2. Contrôle et trois forces de réparation figées"),
    code(
        """CONTROL_ID = 'fusion_control'
RESCUE_RECIPES = [
    {'id': 'functional_s15', 'structural_strength': 0.15},
    {'id': 'functional_s30', 'structural_strength': 0.30},
    {'id': 'functional_s45', 'structural_strength': 0.45},
]
RECIPE_IDS = [CONTROL_ID] + [item['id'] for item in RESCUE_RECIPES]
assert [item['structural_strength'] for item in RESCUE_RECIPES] == [0.15, 0.30, 0.45]
assert RESCUE_STEPS == 8
assert GLOBAL_FUSION_CHANNEL == 1 and GLOBAL_FUSION_ALPHA == 0.15
display(pd.DataFrame([
    {
        'recette': CONTROL_ID,
        'rediffusion': False,
        'force_structurelle': 0.0,
        'pas_tardifs': 0,
    },
    *[
        {
            'recette': item['id'],
            'rediffusion': True,
            'force_structurelle': item['structural_strength'],
            'pas_tardifs': RESCUE_STEPS,
        }
        for item in RESCUE_RECIPES
    ],
]))
"""
    ),
    markdown("## 3. Charger les quatre sources fusionnées et construire le masque fonctionnel"),
    code(
        """validator = QRValidator()
decoder_names = [decoder.name for decoder in validator.decoders]
if decoder_names != ['opencv', 'zbar', 'zxingcpp']:
    raise RuntimeError(f'Trois décodeurs obligatoires, disponibles : {decoder_names}')
quality_scorer = CLIPQualityScorer(Path('/cache'), device='cpu')


def jsonl_rows(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def validation_summary(image, payload):
    records = validator.validate(image, payload)
    summary = summarize_validation_records(records)
    originals = [item for item in records if item.scenario == 'original']
    passed = sum(item.exact_payload_match for item in records)
    original_passed = sum(item.exact_payload_match for item in originals)
    return {
        'passed': passed,
        'total': len(records),
        'pass_rate': passed / len(records),
        'strict_all': passed == len(records),
        'original_passed': original_passed,
        'original_total': len(originals),
        'original_all': original_passed == len(originals),
        'worst_decoder_pass_rate': summary['worst_decoder_pass_rate'],
        'worst_scenario_pass_rate': summary['worst_scenario_pass_rate'],
    }, [asdict(item) for item in records]


def structural_mask_image(aligned):
    size = aligned.canvas_size
    padding = aligned.padding_px
    module_size = aligned.module_size
    core_size = aligned.core_size
    mask = np.ones((size, size), dtype=np.uint8) * 255
    mask[padding:padding + core_size, padding:padding + core_size] = 0
    functional = functional_pattern_mask(aligned.core_blueprint)
    for row, col in np.argwhere(functional):
        y0 = padding + int(row) * module_size
        x0 = padding + int(col) * module_size
        mask[y0:y0 + module_size, x0:x0 + module_size] = 255
    return Image.fromarray(mask, mode='L')


def selected_source_row(context_id):
    if context_id == 'p3_detailed':
        run_root = E014B_V2_RUN_DIR
        candidates = [
            row for row in jsonl_rows(run_root / 'results.jsonl')
            if row['recipe'] == 'fusion_all'
        ]
        if not candidates:
            raise RuntimeError('Aucune fusion_all dans E014B v2.')
        winner = max(
            candidates,
            key=lambda row: (
                row['original_all'], row['passed'],
                row.get('clip_aesthetic') or -999,
            ),
        )
        return winner, run_root / winner['run_name']
    run_root = E014B_V3_RUN_DIR
    candidates = [
        row for row in jsonl_rows(run_root / 'results.jsonl')
        if row['context_id'] == context_id and row['recipe'] == 'fusion_all'
    ]
    if not candidates:
        raise RuntimeError(f'Aucune fusion_all E014B v3 pour {context_id}.')
    winner = max(
        candidates,
        key=lambda row: (
            row['original_all'], row['passed'],
            row.get('clip_aesthetic') or -999,
        ),
    )
    return winner, run_root / context_id / winner['run_name']


def load_context(context_id):
    source_dir = E014A_RUN_DIR / context_id
    meta = json.loads((source_dir / 'selected-meta.json').read_text(encoding='utf-8'))
    prompt_case = next(
        item for item in source_manifests['e014a']['prompts']
        if item['id'] == context_id
    )
    blueprint = Image.open(source_dir / 'selected-blueprint.png').convert('RGB')
    matrix = np.load(source_dir / 'selected-matrix.npy').astype(np.uint8)
    aligned = AlignedQR(
        image=blueprint, core_matrix=matrix, version=meta['version'],
        error_correction='M', mask_pattern=-1, module_size=meta['module_size'],
        padding_px=meta['padding_px'], canvas_size=meta['canvas_size'],
        payload=meta['payload'],
    )
    source_row, source_output = selected_source_row(context_id)
    image_path = source_output / 'final.png'
    latent_path = source_output / 'final.safetensors'
    if not image_path.exists() or not latent_path.exists():
        raise FileNotFoundError(f'Source fusion incomplète : {source_output}')
    source_image = Image.open(image_path).convert('RGB')
    if source_image.size != (aligned.canvas_size, aligned.canvas_size):
        raise RuntimeError(
            f'Géométrie source {context_id} : {source_image.size} != '
            f'{(aligned.canvas_size, aligned.canvas_size)}'
        )
    return {
        'id': context_id,
        'prompt': prompt_case['text'],
        'seed': int(prompt_case['seed']),
        'payload': meta['payload'],
        'selected_blueprint': meta['selected_blueprint'],
        'aligned': aligned,
        'blueprint_image': blueprint,
        'structural_mask': structural_mask_image(aligned),
        'source_row': source_row,
        'source_output': source_output,
        'source_image': source_image,
        'source_latent_cpu': load_file(str(latent_path), device='cpu')['latents'],
    }


contexts = {context_id: load_context(context_id) for context_id in CONTEXT_IDS}
for context_id, context in contexts.items():
    output_dir = RUN_DIR / context_id
    output_dir.mkdir(exist_ok=True)
    context['blueprint_image'].save(output_dir / 'blueprint.png')
    context['source_image'].save(output_dir / 'source-fusion.png')
    context['structural_mask'].save(output_dir / 'functional-mask.png')
    blueprint_validation, blueprint_records = validation_summary(
        context['blueprint_image'], context['payload']
    )
    source_validation, source_records = validation_summary(
        context['source_image'], context['payload']
    )
    (output_dir / 'blueprint-validations.json').write_text(
        json.dumps(blueprint_records, indent=2), encoding='utf-8'
    )
    (output_dir / 'source-validations.json').write_text(
        json.dumps(source_records, indent=2), encoding='utf-8'
    )
    if not blueprint_validation['strict_all']:
        raise RuntimeError(f'Blueprint {context_id} non strict : {blueprint_validation}')
    context['source_validation'] = source_validation
    coverage = np.asarray(context['structural_mask'], dtype=np.float32).mean() / 255
    print(
        context_id,
        'source', source_validation['passed'], '/', source_validation['total'],
        'original', source_validation['original_passed'], '/',
        source_validation['original_total'],
        'couverture masque', round(float(coverage), 4),
    )
    display(context['source_image'].resize((256, 256)))
    display(context['structural_mask'].resize((256, 256)))
"""
    ),
    markdown("## 4. Pipeline fraîche, planning tardif et tenseurs appariés"),
    code(
        """def seed_everything(value):
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def tensor_sha256(tensor):
    value = tensor.detach().to('cpu').contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def load_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL, subfolder=CONTROLNET_SUBFOLDER,
        revision=resolved_revisions['controlnet'],
        torch_dtype=torch.float16, cache_dir='/cache/huggingface',
    )
    pipeline = DiffQRCoderPipeline.from_single_file(
        BASE_MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16,
        cache_dir='/cache/huggingface', safety_checker=None, use_safetensors=True,
        config=BASE_CONFIG_PATH,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to('cuda')
    for component in [
        pipeline.unet, pipeline.controlnet, pipeline.vae, pipeline.text_encoder
    ]:
        component.requires_grad_(False).eval()
    pipeline.enable_attention_slicing('max')
    pipeline.enable_vae_slicing()
    pipeline.unet.enable_gradient_checkpointing()
    pipeline.controlnet.enable_gradient_checkpointing()
    return pipeline


def release_guidance(pipeline):
    guidance = getattr(pipeline, 'srpg', None)
    if guidance is not None:
        try:
            guidance.to('cpu')
        except Exception:
            pass
        pipeline.srpg = None
        del guidance
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@torch.no_grad()
def pil_to_tensor(image):
    array = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    return (
        torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        .to('cuda', dtype=torch.float16)
    )


@torch.no_grad()
def encode_image(pipeline, image):
    tensor = pil_to_tensor(image)
    return (
        pipeline.vae.encode(tensor * 2 - 1).latent_dist.mode()
        * pipeline.vae.config.scaling_factor
    )


@torch.no_grad()
def decode_latent(pipeline, latent):
    dtype = next(pipeline.vae.parameters()).dtype
    decoded = pipeline.vae.decode(
        latent.detach().to(dtype=dtype) / pipeline.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    return pipeline.image_processor.postprocess(
        decoded.detach(), output_type='pil'
    )[0].convert('RGB')


def late_schedule(pipeline):
    pipeline.scheduler.set_timesteps(BASE_STEPS, device='cuda')
    full_schedule = pipeline.scheduler.timesteps.detach().clone()
    late = full_schedule[-RESCUE_STEPS:]
    if len(late) != RESCUE_STEPS:
        raise RuntimeError(f'Planning tardif incomplet : {late}')
    if not torch.all(late[:-1] > late[1:]):
        raise RuntimeError(f'Planning DDIM non décroissant : {late}')
    return late


def latent_structural_mask(mask_image, latent):
    mask = (
        torch.from_numpy(np.asarray(mask_image, dtype=np.float32) / 255.0)
        .unsqueeze(0).unsqueeze(0).to(latent.device, dtype=latent.dtype)
    )
    mask = F.interpolate(mask, size=latent.shape[-2:], mode='area')
    return mask.clamp(0, 1)
"""
    ),
    markdown("## 5. Callback : fusion globale et verrouillage fonctionnel dans le latent"),
    code(
        """def rescue_callback(
    pipeline, context, recipe, output_dir,
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
            else torch.tensor(
                0, device=latent.device, dtype=pipe_ref.scheduler.timesteps.dtype
            )
        )
        with torch.no_grad():
            noised_blueprint = pipe_ref.scheduler.add_noise(
                blueprint_latent, rescue_noise, next_timestep.reshape(1)
            )
            channel = GLOBAL_FUSION_CHANNEL
            latent[:, channel:channel + 1] = (
                (1 - GLOBAL_FUSION_ALPHA) * latent[:, channel:channel + 1]
                + GLOBAL_FUSION_ALPHA
                * noised_blueprint[:, channel:channel + 1]
            )
            alpha = structural_mask * recipe['structural_strength']
            latent = latent * (1 - alpha) + noised_blueprint * alpha
        delta = (latent.float() - before.float())
        trace.append({
            'step': int(step_index),
            'timestep_before_step': int(timestep),
            'target_timestep_after_step': int(next_timestep),
            'global_fusion_channel': GLOBAL_FUSION_CHANNEL,
            'global_fusion_alpha': GLOBAL_FUSION_ALPHA,
            'structural_strength': recipe['structural_strength'],
            'structural_mask_mean': float(structural_mask.float().mean()),
            'latent_delta_rms': float(delta.square().mean().sqrt()),
            'latent_sha256': tensor_sha256(latent),
            'post_diffusion_pixel_projection': False,
        })
        preview = decode_latent(pipeline, latent)
        preview.save(frames_dir / f'{step_index:03d}.jpg', quality=90)
        clear_output(wait=True)
        display(Markdown(
            f"**{context['id']} / {recipe['id']} — "
            f"pas tardif {step_index + 1}/{RESCUE_STEPS}**"
        ))
        display(preview.resize((384, 384)))
        callback_kwargs['latents'] = latent
        return callback_kwargs

    return callback, trace
"""
    ),
    markdown("## 6. Validation originale prioritaire, validation robuste et persistance"),
    code(
        """def original_probe(image, payload):
    records = []
    for decoder in validator.decoders:
        started = time.perf_counter()
        decoded, decoder_error = decode_safely(decoder, image)
        records.append({
            'decoder': decoder.name,
            'exact_payload_match': decoded == payload,
            'latency_ms': (time.perf_counter() - started) * 1000,
            'decoder_error': decoder_error,
        })
    passed = sum(item['exact_payload_match'] for item in records)
    return {
        'original_probe_passed': passed,
        'original_probe_total': len(records),
        'original_probe_all': passed == len(records),
    }, records


def append_row(row):
    with RESULTS_PATH.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\\n')
        stream.flush()


def result_rows():
    return jsonl_rows(RESULTS_PATH)


def completed_keys():
    keys = set()
    for row in result_rows():
        key = (row['context_id'], row['recipe'])
        output_dir = RUN_DIR / row['context_id'] / row['recipe']
        required = [
            output_dir / 'final.png',
            output_dir / 'validations.json',
            output_dir / 'original-probe.json',
        ]
        if not all(path.exists() for path in required):
            raise RuntimeError(f'Résultat indexé mais incomplet : {key}')
        keys.add(key)
    return keys


def make_gif(folder, output):
    paths = sorted(folder.glob('*.jpg'))
    frames = [Image.open(path).convert('RGB').resize((512, 512)) for path in paths]
    if frames:
        frames[0].save(
            output, save_all=True, append_images=frames[1:],
            duration=350, loop=0,
        )
    for frame in frames:
        frame.close()


def score_image(image, context):
    probe, probe_records = original_probe(image, context['payload'])
    validation, records = validation_summary(image, context['payload'])
    try:
        quality = asdict(quality_scorer.score(image, context['prompt']))
        quality_error = None
    except Exception as exc:
        quality = {
            'clip_similarity': None, 'clip_score': None,
            'clip_aesthetic': None,
        }
        quality_error = f'{type(exc).__name__}: {exc}'
    return probe, probe_records, validation, records, quality, quality_error


def persist_score(output_dir, probe_records, records):
    (output_dir / 'original-probe.json').write_text(
        json.dumps(probe_records, indent=2), encoding='utf-8'
    )
    (output_dir / 'validations.json').write_text(
        json.dumps(records, indent=2), encoding='utf-8'
    )
"""
    ),
    markdown("## 7. Enregistrer les quatre contrôles sans nouvelle diffusion"),
    code(
        """complete = completed_keys()
for context_id, context in contexts.items():
    key = (context_id, CONTROL_ID)
    if key in complete:
        print('SKIP', key)
        continue
    output_dir = RUN_DIR / context_id / CONTROL_ID
    if output_dir.exists():
        raise RuntimeError(f'Contrôle partiel : {output_dir}. Démarrer un nouveau run.')
    output_dir.mkdir(parents=True)
    image = context['source_image'].copy()
    image.save(output_dir / 'final.png')
    save_file(
        {'latents': context['source_latent_cpu'].contiguous()},
        str(output_dir / 'final.safetensors'),
    )
    probe, probe_records, validation, records, quality, quality_error = score_image(
        image, context
    )
    persist_score(output_dir, probe_records, records)
    row = {
        'context_id': context_id,
        'prompt': context['prompt'],
        'seed': context['seed'],
        'recipe': CONTROL_ID,
        'structural_strength': 0.0,
        'rescue_steps': 0,
        'late_timesteps': [],
        'source_run': str(context['source_output']),
        'source_passed': context['source_validation']['passed'],
        'source_original_passed': context['source_validation']['original_passed'],
        'source_latent_sha256': tensor_sha256(context['source_latent_cpu']),
        'initial_rescue_latent_sha256': None,
        'final_latent_sha256': tensor_sha256(context['source_latent_cpu']),
        'duration_s': 0.0,
        'post_diffusion_pixel_projection': False,
        **probe,
        **validation,
        **quality,
        **aligned_module_diagnostics(image, context['aligned']),
        **image_change_metrics(image, context['source_image']),
        'quality_error': quality_error,
    }
    append_row(row)
    print(context_id, CONTROL_ID, validation['passed'], '/', validation['total'])
"""
    ),
    markdown("## 8. Campagne GPU : douze pipelines fraîches et 96 pas tardifs"),
    code(
        """def run_rescue(context, recipe):
    output_dir = RUN_DIR / context['id'] / recipe['id']
    if output_dir.exists():
        raise RuntimeError(f'Candidat partiel : {output_dir}. Démarrer un nouveau run.')
    output_dir.mkdir(parents=True)
    pipeline = load_pipeline()
    result = None
    final_latent = None
    initial = None
    noise = None
    mask = None
    blueprint_latent = None
    source_tensor = None
    source_latent = None
    try:
        release_guidance(pipeline)
        source_latent = context['source_latent_cpu'].to('cuda', dtype=torch.float16)
        source_tensor = pil_to_tensor(context['source_image'])
        blueprint_latent = encode_image(pipeline, context['blueprint_image'])
        mask = latent_structural_mask(context['structural_mask'], source_latent)
        timesteps = late_schedule(pipeline)
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
            pipeline, context, recipe, output_dir,
            blueprint_latent, noise, mask,
        )
        started = time.perf_counter()
        result = pipeline._run_stage2(
            prompt=context['prompt'], qrcode=context['blueprint_image'],
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
        probe, probe_records, validation, records, quality, quality_error = score_image(
            image, context
        )
        persist_score(output_dir, probe_records, records)
        row = {
            'context_id': context['id'],
            'prompt': context['prompt'],
            'seed': context['seed'],
            'recipe': recipe['id'],
            'structural_strength': recipe['structural_strength'],
            'rescue_steps': RESCUE_STEPS,
            'late_timesteps': [
                int(value) for value in timesteps.detach().cpu().tolist()
            ],
            'source_run': str(context['source_output']),
            'source_passed': context['source_validation']['passed'],
            'source_original_passed': context['source_validation']['original_passed'],
            'source_latent_sha256': tensor_sha256(source_latent),
            'blueprint_latent_sha256': tensor_sha256(blueprint_latent),
            'rescue_noise_sha256': tensor_sha256(noise),
            'structural_mask_sha256': tensor_sha256(mask),
            'initial_rescue_latent_sha256': tensor_sha256(initial),
            'final_latent_sha256': tensor_sha256(final_latent),
            'duration_s': duration,
            'post_diffusion_pixel_projection': False,
            **probe,
            **validation,
            **quality,
            **aligned_module_diagnostics(image, context['aligned']),
            **image_change_metrics(image, context['source_image']),
            'quality_error': quality_error,
        }
        append_row(row)
        print(
            context['id'], recipe['id'],
            'original', probe['original_probe_passed'], '/',
            probe['original_probe_total'],
            'robuste', validation['passed'], '/', validation['total'],
        )
    finally:
        release_guidance(pipeline)
        del result, final_latent, initial, noise, mask
        del blueprint_latent, source_tensor, source_latent
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


complete = completed_keys()
for context_id in CONTEXT_IDS:
    for recipe in RESCUE_RECIPES:
        key = (context_id, recipe['id'])
        if key in complete:
            print('SKIP', key)
            continue
        print('Pipeline fraîche :', context_id, recipe)
        run_rescue(contexts[context_id], recipe)

rows = result_rows()
expected_rows = len(CONTEXT_IDS) * len(RECIPE_IDS)
if len(rows) != expected_rows:
    raise RuntimeError(f'Campagne incomplète : {len(rows)}/{expected_rows}')

pairing_audit = {}
for context_id in CONTEXT_IDS:
    subset = [
        row for row in rows
        if row['context_id'] == context_id and row['recipe'] != CONTROL_ID
    ]
    audit = {
        'source_latent_hashes': sorted({
            row['source_latent_sha256'] for row in subset
        }),
        'rescue_noise_hashes': sorted({
            row['rescue_noise_sha256'] for row in subset
        }),
        'initial_rescue_latent_hashes': sorted({
            row['initial_rescue_latent_sha256'] for row in subset
        }),
        'structural_mask_hashes': sorted({
            row['structural_mask_sha256'] for row in subset
        }),
        'late_timestep_schedules': sorted({
            tuple(row['late_timesteps']) for row in subset
        }),
    }
    if any(len(values) != 1 for values in audit.values()):
        raise RuntimeError(f'Entrées non appariées pour {context_id}: {audit}')
    pairing_audit[context_id] = audit
(RUN_DIR / 'paired-input-audit.json').write_text(
    json.dumps(pairing_audit, indent=2), encoding='utf-8'
)
print('Campagne complète :', len(rows), 'résultats')
"""
    ),
    markdown("## 9. Classement lexicographique, recette fixe et décision"),
    code(
        """frame = pd.json_normalize(rows, sep='__')
frame.to_csv(RUN_DIR / 'comparison.csv', index=False)


def rank_tuple(row):
    return (
        bool(row['original_probe_all']),
        int(row['passed']),
        float(row['worst_decoder_pass_rate']),
        float(row['worst_scenario_pass_rate']),
        float(row.get('clip_aesthetic') or -999),
        float(row.get('clip_score') or -999),
        -float(row['mean_absolute_change']),
    )


selected_rows = []
for context_id in CONTEXT_IDS:
    candidates = [
        row for row in rows
        if row['context_id'] == context_id and row['recipe'] != CONTROL_ID
    ]
    winner = max(candidates, key=rank_tuple)
    selected_rows.append(winner)
selected_frame = pd.DataFrame(selected_rows)
selected_frame.to_csv(RUN_DIR / 'context-adaptive-oracle.csv', index=False)

aggregate_rows = []
control_frame = frame[frame.recipe == CONTROL_ID].set_index('context_id')
for recipe in RESCUE_RECIPES:
    part = frame[frame.recipe == recipe['id']].set_index('context_id').loc[CONTEXT_IDS]
    aesthetic_drop = (
        control_frame.loc[CONTEXT_IDS].clip_aesthetic - part.clip_aesthetic
    )
    p3_non_regression = bool(
        part.loc['p3_detailed', 'passed']
        >= control_frame.loc['p3_detailed', 'passed']
    )
    original_all_contexts = int(part.original_probe_all.sum())
    fixed_gate = bool(
        original_all_contexts == len(CONTEXT_IDS)
        and float(part.pass_rate.mean())
        > float(control_frame.loc[CONTEXT_IDS].pass_rate.mean())
        and p3_non_regression
        and bool((aesthetic_drop <= MAX_AESTHETIC_DROP).all())
        and bool((part.mean_absolute_change <= MAX_MEAN_ABSOLUTE_CHANGE).all())
    )
    aggregate_rows.append({
        'recipe': recipe['id'],
        'structural_strength': recipe['structural_strength'],
        'original_3of3_contexts': original_all_contexts,
        'strict_39of39_contexts': int(part.strict_all.sum()),
        'mean_ssr': float(part.pass_rate.mean()),
        'minimum_ssr': float(part.pass_rate.min()),
        'mean_clip_aesthetic': float(part.clip_aesthetic.mean()),
        'maximum_aesthetic_drop': float(aesthetic_drop.max()),
        'maximum_mean_absolute_change': float(part.mean_absolute_change.max()),
        'p3_non_regression': p3_non_regression,
        'fixed_recipe_gate': fixed_gate,
    })
aggregate_frame = pd.DataFrame(aggregate_rows).sort_values(
    [
        'fixed_recipe_gate', 'original_3of3_contexts',
        'strict_39of39_contexts', 'mean_ssr', 'mean_clip_aesthetic',
    ],
    ascending=[False, False, False, False, False],
)
aggregate_frame.to_csv(RUN_DIR / 'fixed-recipe-aggregates.csv', index=False)
display(aggregate_frame)

winner = {
    key: (value.item() if isinstance(value, np.generic) else value)
    for key, value in aggregate_frame.iloc[0].to_dict().items()
}
adaptive_original_contexts = int(selected_frame.original_probe_all.sum())
if bool(winner['fixed_recipe_gate']) and int(winner['strict_39of39_contexts']) == 4:
    status = 'production_candidate_pending_physical'
    next_action = 'Valider téléphone, écran et impression avant livraison.'
elif bool(winner['fixed_recipe_gate']):
    status = 'fixed_functional_rescue_candidate'
    next_action = 'Confirmer la recette fixe sur au moins six nouveaux prompt/seed.'
elif adaptive_original_contexts == len(CONTEXT_IDS):
    status = 'context_adaptive_signal_only'
    next_action = 'Construire un sélecteur contextuel, sans revendiquer une recette générale.'
else:
    status = 'rejected'
    next_action = (
        'Ne pas augmenter encore la force : instrumenter séparément quiet zone, '
        'finders et timings.'
    )

decision = {
    'status': status,
    'fixed_recipe_winner': winner,
    'adaptive_original_3of3_contexts': adaptive_original_contexts,
    'contexts_total': len(CONTEXT_IDS),
    'ranking': [
        'original 3/3', 'SSR 39 tests', 'pire décodeur',
        'pire scénario', 'CLIP-aesthetic', 'CLIPScore', 'préservation',
    ],
    'no_post_diffusion_pixel_projection': True,
    'next': next_action,
}
(RUN_DIR / 'DECISION.json').write_text(
    json.dumps(decision, indent=2), encoding='utf-8'
)
print('Décision :', decision)

figure, axes = plt.subplots(1, 2, figsize=(16, 6))
for recipe_id in RECIPE_IDS:
    part = frame[frame.recipe == recipe_id].set_index('context_id').loc[CONTEXT_IDS]
    axes[0].plot(CONTEXT_IDS, part.pass_rate, marker='o', label=recipe_id)
    axes[1].plot(CONTEXT_IDS, part.clip_aesthetic, marker='o', label=recipe_id)
axes[0].set(title='SSR robuste par contexte', ylabel='SSR', ylim=(0, 1))
axes[1].set(title='CLIP-aesthetic par contexte', ylabel='CLIP-aesthetic')
for axis in axes:
    axis.grid(alpha=0.25)
    axis.tick_params(axis='x', rotation=25)
    axis.legend()
figure.tight_layout()
figure.savefig(RUN_DIR / 'functional-rescue-summary.png', dpi=160)
display(figure)

decoder_rows = []
scenario_rows = []
for row in rows:
    records = json.loads(
        (RUN_DIR / row['context_id'] / row['recipe'] / 'validations.json')
        .read_text(encoding='utf-8')
    )
    for record in records:
        decoder_rows.append({
            'context_id': row['context_id'], 'recipe': row['recipe'],
            'decoder': record['decoder'], 'scenario': record['scenario'],
            'passed': int(record['exact_payload_match']),
        })
        scenario_rows.append({
            'context_id': row['context_id'], 'recipe': row['recipe'],
            'scenario': record['scenario'],
            'decoder': record['decoder'],
            'passed': int(record['exact_payload_match']),
        })
pd.DataFrame(decoder_rows).to_csv(RUN_DIR / 'decoder-results.csv', index=False)
pd.DataFrame(scenario_rows).to_csv(RUN_DIR / 'scenario-results.csv', index=False)

physical_rows = []
for row in selected_rows:
    physical_rows.append({
        'context_id': row['context_id'],
        'recipe': row['recipe'],
        'image': f"{row['context_id']}/{row['recipe']}/final.png",
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
"""
    ),
    markdown("## 10. Manifeste, nettoyage GPU et archive"),
    code(
        """runtime_versions = {
    'python': sys.version,
    'torch': torch.__version__,
    'cuda_runtime': torch.version.cuda,
    'diffusers': importlib.metadata.version('diffusers'),
    'transformers': importlib.metadata.version('transformers'),
    'lpips': importlib.metadata.version('lpips'),
}
manifest = {
    'experiment': EXPERIMENT_NAME,
    'sources': {
        'e014a': str(E014A_RUN_DIR),
        'e014b_v2': str(E014B_V2_RUN_DIR),
        'e014b_v3': str(E014B_V3_RUN_DIR),
    },
    'contexts': [
        {
            'id': context['id'],
            'prompt': context['prompt'],
            'seed': context['seed'],
            'payload': context['payload'],
            'selected_blueprint': context['selected_blueprint'],
            'source_run': str(context['source_output']),
            'source_latent_sha256': tensor_sha256(context['source_latent_cpu']),
        }
        for context in contexts.values()
    ],
    'models': {
        'base_repo': BASE_MODEL_REPO,
        'base_file': BASE_MODEL_FILE,
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
    'base_steps': BASE_STEPS,
    'rescue_steps': RESCUE_STEPS,
    'global_fusion_channel': GLOBAL_FUSION_CHANNEL,
    'global_fusion_alpha': GLOBAL_FUSION_ALPHA,
    'recipes': RESCUE_RECIPES,
    'fresh_pipeline_per_rescue_candidate': True,
    'same_late_noise_within_context': True,
    'paired_input_audit': pairing_audit,
    'mask': (
        'quiet zone plus finder/separator, timing, format/version and '
        'alignment modules; data modules excluded'
    ),
    'post_diffusion_pixel_projection': False,
    'fixed_recipe_promotion_rule': {
        'original_3of3_in_all_contexts': True,
        'mean_ssr_above_control': True,
        'p3_non_regression': True,
        'maximum_clip_aesthetic_drop': MAX_AESTHETIC_DROP,
        'maximum_mean_absolute_change': MAX_MEAN_ABSOLUTE_CHANGE,
    },
    'production_rule': (
        'fixed recipe 39/39 in all four contexts, then physical validation'
    ),
    'decision': decision,
    'claim': (
        'Late low-noise functional latent rescue after E014B fusion; '
        'not pixel compositing, not a trained model and not production proof.'
    ),
}
(RUN_DIR / 'manifest.json').write_text(
    json.dumps(manifest, indent=2), encoding='utf-8'
)

for context in contexts.values():
    context['source_latent_cpu'] = None
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print('VRAM allouée après nettoyage GiB :', torch.cuda.memory_allocated() / 2**30)

archive = shutil.make_archive(str(RUN_DIR), 'gztar', RUN_DIR.parent, RUN_DIR.name)
print('Archive :', archive)
print('Serveur Linux : POD=$(kubectl get pod -n qr-core -l app=prooftag-qr-notebook -o jsonpath="{.items[0].metadata.name}")')
print(f'Serveur Linux : kubectl cp -n qr-core "$POD:{archive}" "$HOME/{Path(archive).name}"')
print(f'PowerShell PC : scp paul@pcIA:~/{Path(archive).name} "$HOME/Downloads/"')
"""
    ),
]

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
