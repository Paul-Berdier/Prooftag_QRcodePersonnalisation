"""Build E014B v3 multi-context generalization notebook from the audited v2 template."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "16_e014b_statistical_freeqr_confirmation.ipynb"
TARGET = ROOT / "notebooks" / "17_e014b_multicontext_generalization.ipynb"


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
cells = deepcopy(template["cells"])
assert len(cells) == 21
assert "E014B v2" in "".join(cells[0]["source"])

cells[0] = markdown(
    r"""# E014B v3 — généralisation multi-contexte de la fusion latente

E014B v2 a réparé `p3_detailed` de 2/39 à 31/39 dans quatre répétitions. Ce notebook ne
recherche aucun nouveau paramètre. Il teste si cette recette figée se généralise aux trois
autres contextes de la même campagne E014A v2 :

- `p1_simple` ;
- `p2_medium` ;
- `p4_complex`.

Pour chaque contexte, quatre blocs indépendants comparent la baseline DiffQRCoder à la fusion
canal 1, alpha 0,15, appliquée aux quarante pas. Une pipeline fraîche est chargée par bloc.
Les ordres `baseline → fusion` et `fusion → baseline` alternent deux fois chacun.
"""
)
cells[1] = markdown(
    r"""## Plan expérimental préenregistré

```text
pour chaque contexte p1, p2, p4 :
    bloc 1 : pipeline fraîche → baseline → fusion
    bloc 2 : pipeline fraîche → fusion → baseline
    bloc 3 : pipeline fraîche → baseline → fusion
    bloc 4 : pipeline fraîche → fusion → baseline

dans chaque bloc :
    même Stage 1 + même blueprint 39/39
    même latent initial + même bruit + même seed
```

La fusion est généralisée seulement si, dans **chacun** des trois contextes :

1. elle améliore la baseline dans au moins trois blocs sur quatre ;
2. son gain SSR moyen atteint au moins 3/39 ;
3. les trois décodeurs lisent l'image originale dans les quatre blocs ;
4. son pire SSR ne descend pas sous celui de la baseline ;
5. la perte CLIP-aesthetic moyenne ne dépasse pas 0,75 point.

`production_candidate` exige en plus 39/39 sur les douze sorties fusionnées. Le statut
`generalized_not_strict` signifie uniquement « mécanisme généralisé », jamais « livrable ».
"""
)

imports = "".join(cells[2]["source"])
imports = imports.replace("import gc\n", "import gc\nimport importlib.metadata\n")
imports = imports.replace(
    "from diffusers import ControlNetModel, DDIMScheduler\n",
    "from diffusers import ControlNetModel, DDIMScheduler\n"
    "from huggingface_hub import model_info\n",
)
imports = imports.replace(
    "16_e014b_statistical_freeqr_confirmation.ipynb",
    "17_e014b_multicontext_generalization.ipynb",
)
cells[2] = code(imports)

cells[3] = markdown("## 1. Sources E014A, modèles épinglés et contrat de campagne")
cells[4] = code(
    """EXPERIMENT_NAME = 'e014b-multicontext-generalization-v3'
E014A_RUN_DIR = None
CONTEXT_IDS = ['p1_simple', 'p2_medium', 'p4_complex']
RESUME_RUN_NAME = None

BASE_MODEL_REPO = 'fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned'
BASE_MODEL_FILE = 'cetusMix_Whalefall2_fp16.safetensors'
CONTROLNET_MODEL = 'monster-labs/control_v1p_sd15_qrcode_monster'
CONTROLNET_SUBFOLDER = 'v2'
DIFFQRCODER_COMMIT = 'e24ea73ee2e13c7e6e87cb422e8b11784e70ae00'
STEPS = 40
GUIDANCE_SCALE = 7.5
CONTROLNET_SCALE = 1.35
SCANNING_GUIDANCE = 500.0
PERCEPTUAL_GUIDANCE = 3.0
NEGATIVE_PROMPT = 'easynegative, unreadable text, letters, watermark'
REPEATS = 4
SAVE_PREVIEW_EVERY = 10
MIN_MEAN_GAIN = 3 / 39
MAX_AESTHETIC_DROP = 0.75
ORDERS = [
    ['baseline', 'fusion_all'],
    ['fusion_all', 'baseline'],
    ['baseline', 'fusion_all'],
    ['fusion_all', 'baseline'],
]

if E014A_RUN_DIR is None:
    candidates = sorted(
        Path('/data/notebook-runs').glob('*-e014a-deterministic-blueprint-pairing-v2')
    )
    if not candidates:
        raise FileNotFoundError('Aucun run E014A v2 sous /data/notebook-runs.')
    E014A_RUN_DIR = candidates[-1]
E014A_RUN_DIR = Path(E014A_RUN_DIR)
source_manifest_path = E014A_RUN_DIR / 'manifest.json'
if not source_manifest_path.exists():
    raise FileNotFoundError(source_manifest_path)
source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))

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
        print('Révision épinglée depuis le cache existant pour', repo_id, cached)
        return cached
    try:
        return model_info(repo_id).sha
    except Exception as exc:
        raise RuntimeError(
            f'Impossible de résoudre une révision immuable pour {repo_id}: {exc}'
        ) from exc


revision_path = RUN_DIR / 'resolved-model-revisions.json'
if revision_path.exists():
    resolved_revisions = json.loads(revision_path.read_text(encoding='utf-8'))
else:
    resolved_revisions = {
        'base_model': resolve_revision(BASE_MODEL_REPO),
        'controlnet': resolve_revision(CONTROLNET_MODEL),
    }
    revision_path.write_text(
        json.dumps(resolved_revisions, indent=2), encoding='utf-8'
    )
BASE_MODEL_URL = (
    f"https://huggingface.co/{BASE_MODEL_REPO}/blob/"
    f"{resolved_revisions['base_model']}/{BASE_MODEL_FILE}"
)

pipeline_source = UPSTREAM_ROOT / 'diffqrcoder' / 'pipeline_diffqrcoder.py'
pipeline_source_sha256 = hashlib.sha256(pipeline_source.read_bytes()).hexdigest()
print('Source E014A :', E014A_RUN_DIR)
print('Contextes :', CONTEXT_IDS)
print('Révisions :', resolved_revisions)
print('Sortie :', RUN_DIR)
"""
)

cells[5] = markdown("## 2. Deux recettes figées et blocs appariés")
cells[6] = code(
    """RECIPES = [
    {
        'id': 'baseline', 'channel': None, 'alpha': 0.0,
        'window': [0.0, 1.0],
    },
    {
        'id': 'fusion_all', 'channel': 1, 'alpha': 0.15,
        'window': [0.0, 1.0],
    },
]
recipe_by_id = {item['id']: item for item in RECIPES}
recipe_ids = [item['id'] for item in RECIPES]
assert recipe_ids == ['baseline', 'fusion_all']
assert len(ORDERS) == REPEATS == 4
assert sum(order[0] == 'baseline' for order in ORDERS) == 2
assert sum(order[0] == 'fusion_all' for order in ORDERS) == 2
assert all(sorted(order) == sorted(recipe_ids) for order in ORDERS)
display(pd.DataFrame(ORDERS, index=[f'bloc {index + 1}' for index in range(REPEATS)]))
"""
)

cells[7] = markdown("## 3. Charger et valider les trois contextes avant tout calcul GPU")
cells[8] = code(
    """validator = QRValidator()
quality_scorer = CLIPQualityScorer(Path('/cache'), device='cpu')


def seed_everything(value):
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def tensor_sha256(tensor):
    value = tensor.detach().to('cpu').contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


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


def load_context(context_id):
    source_dir = E014A_RUN_DIR / context_id
    required = {
        'meta': source_dir / 'selected-meta.json',
        'blueprint': source_dir / 'selected-blueprint.png',
        'matrix': source_dir / 'selected-matrix.npy',
        'stage1': source_dir / 'stage1.safetensors',
        'stage1_image': source_dir / 'stage1-reference.png',
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError('Artefacts E014A manquants : ' + ', '.join(missing))
    meta = json.loads(required['meta'].read_text(encoding='utf-8'))
    prompt_case = next(
        item for item in source_manifest['prompts'] if item['id'] == context_id
    )
    image = Image.open(required['blueprint']).convert('RGB')
    matrix = np.load(required['matrix']).astype(np.uint8)
    aligned = AlignedQR(
        image=image, core_matrix=matrix, version=meta['version'],
        error_correction='M', mask_pattern=-1, module_size=meta['module_size'],
        padding_px=meta['padding_px'], canvas_size=meta['canvas_size'],
        payload=meta['payload'],
    )
    return {
        'id': context_id,
        'prompt': prompt_case['text'],
        'seed': int(prompt_case['seed']),
        'payload': meta['payload'],
        'selected_blueprint': meta['selected_blueprint'],
        'blueprint_image': image,
        'aligned': aligned,
        'stage1_cpu': load_file(str(required['stage1']), device='cpu')['stage1'],
    }


contexts = {context_id: load_context(context_id) for context_id in CONTEXT_IDS}
payloads = {context['payload'] for context in contexts.values()}
if len(payloads) != 1:
    raise RuntimeError(f'Payloads incohérents entre contextes : {payloads}')

for context_id, context in contexts.items():
    validation, records = validation_summary(
        context['blueprint_image'], context['payload']
    )
    destination = RUN_DIR / context_id
    destination.mkdir(exist_ok=True)
    (destination / 'blueprint-validations.json').write_text(
        json.dumps(records, indent=2), encoding='utf-8'
    )
    if not validation['strict_all']:
        raise RuntimeError(
            f"Blueprint {context_id} seulement {validation['passed']}/"
            f"{validation['total']} : campagne interdite."
        )
    print(
        context_id, context['seed'], context['selected_blueprint'],
        validation['passed'], '/', validation['total'],
    )
    display(context['blueprint_image'].resize((256, 256)))
"""
)

cells[9] = markdown("## 4. Pipeline fraîche et entrées appariées par bloc")
cells[10] = code(
    """def load_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL, subfolder=CONTROLNET_SUBFOLDER,
        revision=resolved_revisions['controlnet'],
        torch_dtype=torch.float16, cache_dir='/cache/huggingface',
    )
    pipeline = DiffQRCoderPipeline.from_single_file(
        BASE_MODEL_URL, controlnet=controlnet, torch_dtype=torch.float16,
        cache_dir='/cache/huggingface', safety_checker=None, use_safetensors=True,
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


def close_pipeline(pipeline):
    release_guidance(pipeline)
    pipeline.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@torch.no_grad()
def encode_image(pipeline, image):
    array = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    tensor = (
        torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        .to('cuda', dtype=torch.float16)
    )
    return (
        pipeline.vae.encode(tensor * 2 - 1).latent_dist.mode()
        * pipeline.vae.config.scaling_factor
    )


@torch.no_grad()
def paired_inputs(pipeline, stage1_tensor, seed):
    encoded = (
        pipeline.vae.encode(stage1_tensor * 2 - 1).latent_dist.mode()
        * pipeline.vae.config.scaling_factor
    )
    generator = torch.Generator(device='cuda').manual_seed(seed + 10000)
    noise = torch.randn(
        encoded.shape, generator=generator, device='cuda', dtype=encoded.dtype
    )
    pipeline.scheduler.set_timesteps(STEPS, device='cuda')
    initial = pipeline.scheduler.add_noise(
        encoded, noise, pipeline.scheduler.timesteps[:1]
    )
    return initial.detach(), noise.detach()


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
"""
)

cells[11] = markdown("## 5. Fusion latente figée et visualisation de la diffusion")
cells[12] = code(
    """def callback_for(
    pipeline, context, recipe, output_dir, blueprint_latent, paired_noise
):
    frames = output_dir / 'frames'
    frames.mkdir(parents=True, exist_ok=True)
    trace = []

    def callback(pipe_ref, step_index, timestep, callback_kwargs):
        latent = callback_kwargs['latents'].detach()
        next_timestep = (
            pipe_ref.scheduler.timesteps[step_index + 1]
            if step_index + 1 < len(pipe_ref.scheduler.timesteps)
            else torch.tensor(
                0, device=latent.device, dtype=pipe_ref.scheduler.timesteps.dtype
            )
        )
        active = recipe['channel'] is not None
        if active:
            with torch.no_grad():
                noised_qr = pipe_ref.scheduler.add_noise(
                    blueprint_latent, paired_noise, next_timestep.reshape(1)
                )
                channel = recipe['channel']
                latent[:, channel:channel + 1] = (
                    (1 - recipe['alpha']) * latent[:, channel:channel + 1]
                    + recipe['alpha'] * noised_qr[:, channel:channel + 1]
                )
        trace.append({
            'step': int(step_index),
            'timestep_before_step': int(timestep),
            'target_timestep_after_step': int(next_timestep),
            'fusion_applied': active,
            'latent_sha256': tensor_sha256(latent),
        })
        if step_index % SAVE_PREVIEW_EVERY == 0 or step_index == STEPS - 1:
            preview = decode_latent(pipeline, latent)
            preview.save(frames / f'{step_index:03d}.jpg', quality=88)
            clear_output(wait=True)
            display(Markdown(
                f"**{context['id']} / {recipe['id']} — "
                f"étape {step_index + 1}/{STEPS}**"
            ))
            display(preview.resize((384, 384)))
        callback_kwargs['latents'] = latent
        return callback_kwargs

    return callback, trace
"""
)

cells[13] = markdown("## 6. Persistance atomique et refus des blocs partiels")
cells[14] = code(
    """def append_row(row):
    with RESULTS_PATH.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\\n')
        stream.flush()


def result_rows():
    if not RESULTS_PATH.exists():
        return []
    return [
        json.loads(line)
        for line in RESULTS_PATH.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def completed_blocks():
    rows = result_rows()
    complete = set()
    for context_id in CONTEXT_IDS:
        context_dir = RUN_DIR / context_id
        for repeat in range(1, REPEATS + 1):
            key = (context_id, repeat)
            subset = [
                row for row in rows
                if row['context_id'] == context_id and row['repeat'] == repeat
            ]
            if not subset:
                abandoned = sorted(context_dir.glob(f'r{repeat:02d}-p*-*'))
                if abandoned:
                    raise RuntimeError(
                        f'Bloc {context_id}/{repeat} interrompu avant sa première ligne. '
                        f'Artefacts : {[path.name for path in abandoned]}. '
                        'Démarrer un nouveau run.'
                    )
                continue
            if len(subset) != len(RECIPES):
                raise RuntimeError(
                    f'Bloc {context_id}/{repeat} partiel '
                    f'({len(subset)}/{len(RECIPES)}). Démarrer un nouveau run.'
                )
            if {row['recipe'] for row in subset} != set(recipe_ids):
                raise RuntimeError(f'Bloc {context_id}/{repeat} incohérent.')
            complete.add(key)
    return complete


def make_gif(folder, output):
    paths = sorted(folder.glob('*.jpg'))
    frames = [Image.open(path).convert('RGB').resize((512, 512)) for path in paths]
    if frames:
        frames[0].save(
            output, save_all=True, append_images=frames[1:],
            duration=300, loop=0,
        )
    for frame in frames:
        frame.close()


def run_recipe(
    pipeline, context, stage1_tensor, blueprint_latent, initial, noise,
    recipe, repeat, position,
):
    run_name = f"r{repeat:02d}-p{position:02d}-{recipe['id']}"
    output_dir = RUN_DIR / context['id'] / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    release_guidance(pipeline)
    seed_everything(context['seed'] + 10000)
    callback, trace = callback_for(
        pipeline, context, recipe, output_dir, blueprint_latent, noise
    )
    started = time.perf_counter()
    result = pipeline._run_stage2(
        prompt=context['prompt'], qrcode=context['blueprint_image'],
        qrcode_module_size=context['aligned'].module_size,
        qrcode_padding=context['aligned'].padding_px,
        ref_image=stage1_tensor, negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=STEPS, guidance_scale=GUIDANCE_SCALE, eta=0.0,
        generator=torch.Generator(device='cuda').manual_seed(
            context['seed'] + 10000
        ),
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
    validation, records = validation_summary(image, context['payload'])
    (output_dir / 'validations.json').write_text(
        json.dumps(records, indent=2), encoding='utf-8'
    )
    decoder_results = {}
    scenario_results = {}
    for record in records:
        for key, values in [
            (record['decoder'], decoder_results),
            (record['scenario'], scenario_results),
        ]:
            counters = values.setdefault(key, {'passed': 0, 'total': 0})
            counters['total'] += 1
            counters['passed'] += int(record['exact_payload_match'])
    try:
        quality = asdict(quality_scorer.score(image, context['prompt']))
        quality_error = None
    except Exception as exc:
        quality = {
            'clip_similarity': None, 'clip_score': None,
            'clip_aesthetic': None,
        }
        quality_error = f'{type(exc).__name__}: {exc}'
    row = {
        'run_name': run_name,
        'context_id': context['id'],
        'prompt': context['prompt'],
        'seed': context['seed'],
        'repeat': repeat,
        'position': position,
        'order': ORDERS[repeat - 1],
        'recipe': recipe['id'],
        'config': recipe,
        'initial_latent_sha256': tensor_sha256(initial),
        'final_latent_sha256': tensor_sha256(final_latent),
        'duration_s': duration,
        'decoder_results': decoder_results,
        'scenario_results': scenario_results,
        **validation,
        **quality,
        **aligned_module_diagnostics(image, context['aligned']),
        'quality_error': quality_error,
    }
    append_row(row)
    print(
        context['id'], run_name,
        validation['passed'], '/', validation['total'],
        'original', validation['original_passed'], '/', validation['original_total'],
    )
    del result, final_latent
    release_guidance(pipeline)
    gc.collect()
    torch.cuda.empty_cache()
"""
)

cells[15] = markdown("## 7. Campagne : douze pipelines fraîches, vingt-quatre diffusions")
cells[16] = code(
    """complete = completed_blocks()
for context_id in CONTEXT_IDS:
    context = contexts[context_id]
    for repeat, order in enumerate(ORDERS, start=1):
        key = (context_id, repeat)
        if key in complete:
            print('SKIP bloc complet', key)
            continue
        print('Pipeline fraîche pour', context_id, 'bloc', repeat, order)
        pipe = load_pipeline()
        stage1_tensor = context['stage1_cpu'].to('cuda', dtype=torch.float16)
        blueprint_latent = encode_image(pipe, context['blueprint_image'])
        paired_initial, paired_noise = paired_inputs(
            pipe, stage1_tensor, context['seed']
        )
        print('Latent initial :', tensor_sha256(paired_initial))
        for position, recipe_id in enumerate(order, start=1):
            run_recipe(
                pipe, context, stage1_tensor, blueprint_latent,
                paired_initial, paired_noise,
                recipe_by_id[recipe_id], repeat, position,
            )
        del stage1_tensor, blueprint_latent, paired_initial, paired_noise
        close_pipeline(pipe)
        del pipe

rows = result_rows()
expected_rows = len(CONTEXT_IDS) * REPEATS * len(RECIPES)
if len(rows) != expected_rows:
    raise RuntimeError(f'Campagne incomplète : {len(rows)}/{expected_rows}.')

initial_hashes_by_block = {}
for context_id in CONTEXT_IDS:
    for repeat in range(1, REPEATS + 1):
        hashes = sorted({
            row['initial_latent_sha256'] for row in rows
            if row['context_id'] == context_id and row['repeat'] == repeat
        })
        if len(hashes) != 1:
            raise RuntimeError(
                f'Latent initial non apparié dans {context_id}/{repeat}: {hashes}'
            )
        initial_hashes_by_block[f'{context_id}/r{repeat:02d}'] = hashes[0]
for context_id in CONTEXT_IDS:
    context_hashes = {
        initial_hashes_by_block[f'{context_id}/r{repeat:02d}']
        for repeat in range(1, REPEATS + 1)
    }
    if len(context_hashes) != 1:
        raise RuntimeError(
            f'Latent initial variable entre les blocs de {context_id}: {context_hashes}'
        )
print('Campagne complète :', len(rows), 'résultats')
"""
)

cells[17] = markdown("## 8. Généralisation, portes corrigées et matrices de robustesse")
cells[18] = code(
    """frame = pd.json_normalize(rows, sep='__')
frame.to_csv(RUN_DIR / 'comparison.csv', index=False)

decoder_rows = []
scenario_rows = []
for row in rows:
    common = {
        'context_id': row['context_id'],
        'repeat': row['repeat'],
        'recipe': row['recipe'],
    }
    for decoder, counters in row['decoder_results'].items():
        decoder_rows.append({
            **common, 'decoder': decoder,
            'passed': counters['passed'], 'total': counters['total'],
            'pass_rate': counters['passed'] / counters['total'],
        })
    for scenario, counters in row['scenario_results'].items():
        scenario_rows.append({
            **common, 'scenario': scenario,
            'passed': counters['passed'], 'total': counters['total'],
            'pass_rate': counters['passed'] / counters['total'],
        })
decoder_frame = pd.DataFrame(decoder_rows)
scenario_frame = pd.DataFrame(scenario_rows)
decoder_frame.to_csv(RUN_DIR / 'decoder-results.csv', index=False)
scenario_frame.to_csv(RUN_DIR / 'scenario-results.csv', index=False)

paired_rows = []
context_rows = []
for context_id in CONTEXT_IDS:
    context_part = frame[frame.context_id == context_id]
    baseline = context_part[context_part.recipe == 'baseline'].sort_values('repeat')
    fusion = context_part[context_part.recipe == 'fusion_all'].sort_values('repeat')
    baseline_by_repeat = {
        int(row.repeat): float(row.pass_rate) for row in baseline.itertuples()
    }
    differences = []
    for row in fusion.itertuples():
        difference = float(row.pass_rate) - baseline_by_repeat[int(row.repeat)]
        differences.append(difference)
        paired_rows.append({
            'context_id': context_id,
            'repeat': int(row.repeat),
            'baseline_pass_rate': baseline_by_repeat[int(row.repeat)],
            'fusion_pass_rate': float(row.pass_rate),
            'difference': difference,
        })
    baseline_span = float(
        baseline.pass_rate.max() - baseline.pass_rate.min()
    )
    aesthetic_drop = float(
        baseline.clip_aesthetic.mean() - fusion.clip_aesthetic.mean()
    )
    original_gate = bool(
        (fusion.original_passed == fusion.original_total).all()
        and (fusion.original_total > 0).all()
    )
    positive_repeats = int(sum(value > 0 for value in differences))
    mean_gain = float(np.mean(differences))
    context_gate = bool(
        positive_repeats >= 3
        and mean_gain >= MIN_MEAN_GAIN
        and mean_gain > baseline_span
        and float(fusion.pass_rate.min()) >= float(baseline.pass_rate.min())
        and original_gate
        and aesthetic_drop <= MAX_AESTHETIC_DROP
    )
    context_rows.append({
        'context_id': context_id,
        'baseline_ssr_mean': float(baseline.pass_rate.mean()),
        'fusion_ssr_mean': float(fusion.pass_rate.mean()),
        'fusion_ssr_min': float(fusion.pass_rate.min()),
        'mean_gain': mean_gain,
        'positive_repeats': positive_repeats,
        'baseline_span': baseline_span,
        'fusion_original_all_count': int(fusion.original_all.sum()),
        'fusion_strict_count': int(fusion.strict_all.sum()),
        'baseline_clip_aesthetic_mean': float(baseline.clip_aesthetic.mean()),
        'fusion_clip_aesthetic_mean': float(fusion.clip_aesthetic.mean()),
        'fusion_clip_score_mean': float(fusion.clip_score.mean()),
        'aesthetic_drop': aesthetic_drop,
        'context_gate': context_gate,
    })

paired_frame = pd.DataFrame(paired_rows)
context_frame = pd.DataFrame(context_rows)
paired_frame.to_csv(RUN_DIR / 'paired-effects.csv', index=False)
context_frame.to_csv(RUN_DIR / 'context-aggregates.csv', index=False)
display(context_frame)

all_contexts_generalize = bool(context_frame.context_gate.all())
all_fusion_strict = bool(
    frame[frame.recipe == 'fusion_all'].strict_all.all()
)
if all_contexts_generalize and all_fusion_strict:
    status = 'production_candidate'
    next_action = 'Lancer les captures physiques avant toute livraison.'
elif all_contexts_generalize:
    status = 'generalized_not_strict'
    next_action = (
        'Durcir la recette contre les scénarios faibles, en priorité print_dot_loss.'
    )
elif bool(context_frame.context_gate.any()):
    status = 'partial_generalization'
    next_action = (
        'Modéliser la dépendance au contexte avant toute optimisation générale.'
    )
else:
    status = 'rejected'
    next_action = 'Ne pas généraliser fusion_all ; revoir la méthode.'

decision = {
    'status': status,
    'all_contexts_generalize': all_contexts_generalize,
    'all_fusion_strict_39of39': all_fusion_strict,
    'contexts_passed': int(context_frame.context_gate.sum()),
    'contexts_total': len(CONTEXT_IDS),
    'required_original_gate': '3/3 in all four fusion repetitions per context',
    'minimum_mean_gain': MIN_MEAN_GAIN,
    'maximum_aesthetic_drop': MAX_AESTHETIC_DROP,
    'next': next_action,
}
(RUN_DIR / 'DECISION.json').write_text(
    json.dumps(decision, indent=2), encoding='utf-8'
)
print('Décision :', decision)

figure, axes = plt.subplots(1, 2, figsize=(15, 5))
for context_id in CONTEXT_IDS:
    part = paired_frame[paired_frame.context_id == context_id]
    axes[0].plot(
        part.repeat, part.difference, marker='o', label=context_id
    )
axes[0].axhline(0, color='black', linewidth=1)
axes[0].axhline(MIN_MEAN_GAIN, color='red', linestyle='--', label='gain minimal')
axes[0].set(
    title='Gain SSR apparié fusion - baseline',
    xlabel='bloc', ylabel='différence SSR', xticks=range(1, REPEATS + 1),
)
axes[0].legend()
axes[0].grid(alpha=0.25)

x = np.arange(len(CONTEXT_IDS))
width = 0.35
ordered = context_frame.set_index('context_id').loc[CONTEXT_IDS]
axes[1].bar(
    x - width / 2, ordered.baseline_ssr_mean, width, label='baseline'
)
axes[1].bar(
    x + width / 2, ordered.fusion_ssr_mean, width, label='fusion_all'
)
axes[1].set(
    title='SSR moyen par contexte',
    ylabel='SSR', xticks=x, xticklabels=CONTEXT_IDS, ylim=(0, 1),
)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.25)
figure.tight_layout()
figure.savefig(RUN_DIR / 'generalization-summary.png', dpi=160)
display(figure)

physical_rows = []
for context_id in CONTEXT_IDS:
    candidates = frame[
        (frame.context_id == context_id) & (frame.recipe == 'fusion_all')
    ].sort_values(
        ['strict_all', 'pass_rate', 'clip_aesthetic'],
        ascending=[False, False, False],
    )
    best = candidates.iloc[0]
    physical_rows.append({
        'context_id': context_id,
        'image': f"{context_id}/{best.run_name}/final.png",
        'software_passed': int(best.passed),
        'software_total': int(best.total),
        'phone_model': '',
        'distance_cm': '',
        'screen_or_print': '',
        'lighting': '',
        'attempts': '',
        'successes': '',
        'notes': '',
    })
pd.DataFrame(physical_rows).to_csv(
    RUN_DIR / 'physical-validation-template.csv', index=False
)
"""
)

cells[19] = markdown("## 9. Manifeste reproductible, nettoyage et archive")
cells[20] = code(
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
    'source_e014a': str(E014A_RUN_DIR),
    'contexts': [
        {
            'id': context['id'],
            'prompt': context['prompt'],
            'seed': context['seed'],
            'payload': context['payload'],
            'selected_blueprint': context['selected_blueprint'],
        }
        for context in contexts.values()
    ],
    'models': {
        'base_repo': BASE_MODEL_REPO,
        'base_file': BASE_MODEL_FILE,
        'base_revision': resolved_revisions['base_model'],
        'controlnet_repo': CONTROLNET_MODEL,
        'controlnet_subfolder': CONTROLNET_SUBFOLDER,
        'controlnet_revision': resolved_revisions['controlnet'],
        'diffqrcoder_commit': DIFFQRCODER_COMMIT,
        'pipeline_source_sha256': pipeline_source_sha256,
    },
    'runtime_versions': runtime_versions,
    'steps': STEPS,
    'guidance_scale': GUIDANCE_SCALE,
    'controlnet_scale': CONTROLNET_SCALE,
    'scanning_guidance': SCANNING_GUIDANCE,
    'perceptual_guidance': PERCEPTUAL_GUIDANCE,
    'repeats': REPEATS,
    'orders': ORDERS,
    'recipes': RECIPES,
    'fresh_pipeline_per_block': True,
    'strict_deterministic_algorithms': True,
    'initial_hashes_by_block': initial_hashes_by_block,
    'promotion_rule': {
        'positive_repeats_minimum': 3,
        'minimum_mean_gain': MIN_MEAN_GAIN,
        'mean_gain_exceeds_baseline_span': True,
        'fusion_worst_not_below_baseline_worst': True,
        'original_3of3_required_in_every_repeat': True,
        'maximum_clip_aesthetic_drop': MAX_AESTHETIC_DROP,
    },
    'production_rule': '39/39 for every fusion output plus all context gates',
    'decision': decision,
    'claim': (
        'Pre-registered multi-context generalization of the E014B v2 winner; '
        'not a parameter search and not an official FreeQR implementation.'
    ),
}
(RUN_DIR / 'manifest.json').write_text(
    json.dumps(manifest, indent=2), encoding='utf-8'
)

for context in contexts.values():
    context['stage1_cpu'] = None
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
)

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
