"""Build E014E mechanism and late-window ablation notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "18_e014d_functional_late_rediffusion.ipynb"
TARGET = ROOT / "notebooks" / "19_e014e_mechanism_window_ablation.ipynb"


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


def template_source(index: int) -> str:
    return "".join(template["cells"][index]["source"])


imports = template_source(2).replace(
    "18_e014d_functional_late_rediffusion.ipynb",
    "19_e014e_mechanism_window_ablation.ipynb",
)

context_loader = template_source(8)

pipeline_helpers = template_source(10)
pipeline_helpers = pipeline_helpers.replace(
    "def late_schedule(pipeline):",
    "def late_schedule(pipeline, rescue_steps):",
)
pipeline_helpers = pipeline_helpers.replace(
    "late = full_schedule[-RESCUE_STEPS:]",
    "late = full_schedule[-rescue_steps:]",
)
pipeline_helpers = pipeline_helpers.replace(
    "if len(late) != RESCUE_STEPS:",
    "if len(late) != rescue_steps:",
)

cells = [
    markdown(
        """# E014E — ablation mécanistique et fenêtre de rediffusion tardive

E014D a porté la recette fixe à 151/156 validations et rendu les douze images originales
lisibles par les trois décodeurs, mais avec une perte visuelle trop forte. E014E ne cherche pas
une force supplémentaire. Il sépare ce que la campagne précédente confondait :

1. la rediffusion DiffQRCoder tardive elle-même ;
2. la fusion globale du canal latent 1 ;
3. le masque latent limité aux motifs fonctionnels ;
4. le nombre de pas tardifs.

La priorité reste la lecture exacte. La deuxième cible est de réduire la perte
CLIP-aesthetic et la modification absolue par rapport au contrôle E014B.
"""
    ),
    markdown(
        """## Protocole préenregistré

```text
                         sources E014B appariées
                                  |
               +------------------+------------------+
               |                                     |
       PHASE A — mécanismes                   contrôle sans rediffusion
       p2_medium + p3_detailed
       11 recettes × 4 pas
               |
       référence 0,15/0,15 forcée
       + 2 meilleures recettes faibles
               |
       PHASE B — fenêtre temporelle
       3 recettes × {2,4,6,8} pas × 4 contextes
               |
       p1/p4 = validation après sélection initiale
               |
       original 3/3 → SSR → esthétique → préservation
```

Chaque candidat utilise une pipeline fraîche. Dans un même contexte, toutes les recettes
réutilisent exactement le même latent source et le même bruit. Pour une même longueur de fenêtre,
le latent initial doit aussi être identique. Aucune projection de pixels n'est autorisée après la
diffusion.

`rediffusion_only` signifie ici la rediffusion guidée native de DiffQRCoder, avec ControlNet,
Scanning-Robust Guidance et Perceptual Guidance, mais sans nos deux injections latentes
supplémentaires.
"""
    ),
    code(imports),
    markdown("## 1. Sources, modèles épinglés et espace expérimental"),
    code(
        """EXPERIMENT_NAME = 'e014e-mechanism-window-ablation-v1'
E014A_RUN_DIR = None
E014B_V2_RUN_DIR = None
E014B_V3_RUN_DIR = None
RESUME_RUN_NAME = None

CONTEXT_IDS = ['p1_simple', 'p2_medium', 'p3_detailed', 'p4_complex']
SCREENING_CONTEXT_IDS = ['p2_medium', 'p3_detailed']
HOLDOUT_CONTEXT_IDS = ['p1_simple', 'p4_complex']

BASE_MODEL_REPO = 'fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned'
BASE_MODEL_FILE = 'cetusMix_Whalefall2_fp16.safetensors'
CONFIG_MODEL_REPO = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
CONTROLNET_MODEL = 'monster-labs/control_v1p_sd15_qrcode_monster'
CONTROLNET_SUBFOLDER = 'v2'
DIFFQRCODER_COMMIT = 'e24ea73ee2e13c7e6e87cb422e8b11784e70ae00'

BASE_STEPS = 40
PHASE_A_STEPS = 4
PHASE_B_STEP_COUNTS = [2, 4, 6, 8]
GUIDANCE_SCALE = 7.5
CONTROLNET_SCALE = 1.35
SCANNING_GUIDANCE = 500.0
PERCEPTUAL_GUIDANCE = 3.0
GLOBAL_FUSION_CHANNEL = 1
RESCUE_SEED_OFFSET = 200_014
MAX_AESTHETIC_DROP = 0.75
MAX_MEAN_ABSOLUTE_CHANGE = 0.18
NEGATIVE_PROMPT = 'easynegative, unreadable text, letters, watermark'

REFERENCE_RECIPE_ID = 'combined_a15_s15'
PROMOTED_COUNT = 3
CONTROL_ID = 'fusion_control'

PHASE_A_RECIPES = [
    {'id': 'rediffusion_only', 'mechanism': 'rediffusion', 'global_alpha': 0.00, 'structural_strength': 0.00},
    {'id': 'global_a03', 'mechanism': 'global_only', 'global_alpha': 0.03, 'structural_strength': 0.00},
    {'id': 'global_a06', 'mechanism': 'global_only', 'global_alpha': 0.06, 'structural_strength': 0.00},
    {'id': 'global_a10', 'mechanism': 'global_only', 'global_alpha': 0.10, 'structural_strength': 0.00},
    {'id': 'mask_s05', 'mechanism': 'mask_only', 'global_alpha': 0.00, 'structural_strength': 0.05},
    {'id': 'mask_s10', 'mechanism': 'mask_only', 'global_alpha': 0.00, 'structural_strength': 0.10},
    {'id': 'mask_s15', 'mechanism': 'mask_only', 'global_alpha': 0.00, 'structural_strength': 0.15},
    {'id': 'combined_a03_s05', 'mechanism': 'combined', 'global_alpha': 0.03, 'structural_strength': 0.05},
    {'id': 'combined_a06_s10', 'mechanism': 'combined', 'global_alpha': 0.06, 'structural_strength': 0.10},
    {'id': 'combined_a10_s15', 'mechanism': 'combined', 'global_alpha': 0.10, 'structural_strength': 0.15},
    {'id': 'combined_a15_s15', 'mechanism': 'e014d_reference', 'global_alpha': 0.15, 'structural_strength': 0.15},
]

assert CONTEXT_IDS == ['p1_simple', 'p2_medium', 'p3_detailed', 'p4_complex']
assert set(SCREENING_CONTEXT_IDS).isdisjoint(HOLDOUT_CONTEXT_IDS)
assert sorted(SCREENING_CONTEXT_IDS + HOLDOUT_CONTEXT_IDS) == sorted(CONTEXT_IDS)
assert PHASE_B_STEP_COUNTS == [2, 4, 6, 8]
assert REFERENCE_RECIPE_ID in {item['id'] for item in PHASE_A_RECIPES}
assert len({item['id'] for item in PHASE_A_RECIPES}) == len(PHASE_A_RECIPES)


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
display(pd.DataFrame(PHASE_A_RECIPES))
"""
    ),
    markdown("## 2. Charger les quatre sources E014B et le masque fonctionnel"),
    code(context_loader),
    markdown("## 3. Pipeline DDIM à fenêtre tardive dynamique"),
    code(pipeline_helpers),
    markdown("## 4. Injection instrumentée : globale, masque ou combinaison"),
    code(
        """def rescue_callback(
    pipeline, context, recipe, rescue_steps, output_dir,
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
        preview = decode_latent(pipeline, latent)
        preview.save(frames_dir / f'{step_index:03d}.jpg', quality=90)
        clear_output(wait=True)
        display(Markdown(
            f"**{context['id']} / {recipe['id']} — "
            f"pas tardif {step_index + 1}/{rescue_steps}**"
        ))
        display(preview.resize((384, 384)))
        callback_kwargs['latents'] = latent
        return callback_kwargs

    return callback, trace
"""
    ),
    markdown("## 5. Validation, reprise sûre et persistance atomique par candidat"),
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


def append_jsonl(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\\n')
        stream.flush()


def result_rows():
    return jsonl_rows(RESULTS_PATH)


def candidate_key(phase, context_id, recipe_id, rescue_steps):
    return phase, context_id, recipe_id, int(rescue_steps)


def candidate_dir(phase, context_id, recipe_id, rescue_steps):
    return (
        RUN_DIR / phase / context_id
        / f'{recipe_id}-steps{int(rescue_steps):02d}'
    )


def completed_keys():
    keys = set()
    for row in result_rows():
        key = candidate_key(
            row['phase'], row['context_id'], row['recipe'], row['rescue_steps']
        )
        if key in keys:
            raise RuntimeError(f'Résultat dupliqué : {key}')
        output_dir = RUN_DIR / row['relative_output_dir']
        required = [
            output_dir / 'final.png',
            output_dir / 'final.safetensors',
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
            duration=400, loop=0,
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


def record_control(context):
    phase = 'control'
    key = candidate_key(phase, context['id'], CONTROL_ID, 0)
    if key in completed_keys():
        print('SKIP', key)
        return
    output_dir = candidate_dir(phase, context['id'], CONTROL_ID, 0)
    if output_dir.exists():
        shutil.rmtree(output_dir)
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
        'phase': phase,
        'relative_output_dir': str(output_dir.relative_to(RUN_DIR)),
        'context_id': context['id'],
        'prompt': context['prompt'],
        'seed': context['seed'],
        'recipe': CONTROL_ID,
        'mechanism': 'control',
        'global_alpha': 0.0,
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
        **probe, **validation, **quality,
        **aligned_module_diagnostics(image, context['aligned']),
        **image_change_metrics(image, context['source_image']),
        'quality_error': quality_error,
    }
    append_jsonl(RESULTS_PATH, row)
    print(context['id'], CONTROL_ID, validation['passed'], '/', validation['total'])


def run_candidate(phase, context, recipe, rescue_steps):
    key = candidate_key(phase, context['id'], recipe['id'], rescue_steps)
    if key in completed_keys():
        print('SKIP', key)
        return
    output_dir = candidate_dir(
        phase, context['id'], recipe['id'], rescue_steps
    )
    if output_dir.exists():
        print('Nettoyage du candidat non indexé :', output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    pipeline = load_pipeline()
    completed = False
    result = final_latent = initial = noise = mask = None
    blueprint_latent = source_tensor = source_latent = None
    try:
        release_guidance(pipeline)
        source_latent = context['source_latent_cpu'].to(
            'cuda', dtype=torch.float16
        )
        source_tensor = pil_to_tensor(context['source_image'])
        blueprint_latent = encode_image(pipeline, context['blueprint_image'])
        mask = latent_structural_mask(
            context['structural_mask'], source_latent
        )
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
            pipeline, context, recipe, rescue_steps, output_dir,
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
            'phase': phase,
            'relative_output_dir': str(output_dir.relative_to(RUN_DIR)),
            'context_id': context['id'],
            'prompt': context['prompt'],
            'seed': context['seed'],
            'recipe': recipe['id'],
            'mechanism': recipe['mechanism'],
            'global_alpha': float(recipe['global_alpha']),
            'structural_strength': float(recipe['structural_strength']),
            'rescue_steps': int(rescue_steps),
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
            **probe, **validation, **quality,
            **aligned_module_diagnostics(image, context['aligned']),
            **image_change_metrics(image, context['source_image']),
            'quality_error': quality_error,
        }
        append_jsonl(RESULTS_PATH, row)
        completed = True
        print(
            phase, context['id'], recipe['id'], rescue_steps,
            'original', probe['original_probe_passed'], '/',
            probe['original_probe_total'],
            'robuste', validation['passed'], '/', validation['total'],
        )
    except Exception as exc:
        append_jsonl(ERRORS_PATH, {
            'phase': phase, 'context_id': context['id'],
            'recipe': recipe['id'], 'rescue_steps': int(rescue_steps),
            'error_type': type(exc).__name__, 'error': str(exc),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        raise
    finally:
        release_guidance(pipeline)
        del result, final_latent, initial, noise, mask
        del blueprint_latent, source_tensor, source_latent, pipeline
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if not completed and output_dir.exists():
            shutil.rmtree(output_dir)
"""
    ),
    markdown("## 6. Enregistrer les contrôles E014B sans diffusion"),
    code(
        """for context in contexts.values():
    record_control(context)

control_rows = [
    row for row in result_rows() if row['phase'] == 'control'
]
if len(control_rows) != len(CONTEXT_IDS):
    raise RuntimeError(
        f'Contrôles incomplets : {len(control_rows)}/{len(CONTEXT_IDS)}'
    )
"""
    ),
    markdown("## 7. Phase A — isoler les mécanismes sur p2 et p3"),
    code(
        """for context_id in SCREENING_CONTEXT_IDS:
    for recipe in PHASE_A_RECIPES:
        print('Phase A :', context_id, recipe)
        run_candidate(
            'phase_a', contexts[context_id], recipe, PHASE_A_STEPS
        )

phase_a_rows = [
    row for row in result_rows() if row['phase'] == 'phase_a'
]
expected_phase_a = len(SCREENING_CONTEXT_IDS) * len(PHASE_A_RECIPES)
if len(phase_a_rows) != expected_phase_a:
    raise RuntimeError(
        f'Phase A incomplète : {len(phase_a_rows)}/{expected_phase_a}'
    )

phase_a_frame = pd.DataFrame(phase_a_rows)
phase_a_frame.to_csv(RUN_DIR / 'phase-a-results.csv', index=False)

phase_a_aggregates = []
for recipe in PHASE_A_RECIPES:
    part = phase_a_frame[
        phase_a_frame.recipe == recipe['id']
    ].set_index('context_id').loc[SCREENING_CONTEXT_IDS]
    phase_a_aggregates.append({
        'recipe': recipe['id'],
        'mechanism': recipe['mechanism'],
        'global_alpha': recipe['global_alpha'],
        'structural_strength': recipe['structural_strength'],
        'original_3of3_contexts': int(part.original_probe_all.sum()),
        'strict_39of39_contexts': int(part.strict_all.sum()),
        'minimum_ssr': float(part.pass_rate.min()),
        'mean_ssr': float(part.pass_rate.mean()),
        'mean_clip_aesthetic': float(part.clip_aesthetic.mean()),
        'mean_clip_score': float(part.clip_score.mean()),
        'maximum_mean_absolute_change': float(part.mean_absolute_change.max()),
    })

phase_a_aggregate_frame = pd.DataFrame(phase_a_aggregates).sort_values(
    [
        'original_3of3_contexts', 'strict_39of39_contexts',
        'minimum_ssr', 'mean_ssr', 'mean_clip_aesthetic',
        'mean_clip_score', 'maximum_mean_absolute_change',
    ],
    ascending=[False, False, False, False, False, False, True],
)
phase_a_aggregate_frame.to_csv(
    RUN_DIR / 'phase-a-aggregates.csv', index=False
)
display(phase_a_aggregate_frame)

ranked_non_reference = [
    row for row in phase_a_aggregate_frame.to_dict('records')
    if row['recipe'] != REFERENCE_RECIPE_ID
]
promoted_ids = [REFERENCE_RECIPE_ID] + [
    row['recipe'] for row in ranked_non_reference[:PROMOTED_COUNT - 1]
]
if len(set(promoted_ids)) != PROMOTED_COUNT:
    raise RuntimeError(f'Promotion invalide : {promoted_ids}')
recipe_by_id = {item['id']: item for item in PHASE_A_RECIPES}
promoted_recipes = [recipe_by_id[item] for item in promoted_ids]
promotion = {
    'screening_contexts': SCREENING_CONTEXT_IDS,
    'holdout_contexts': HOLDOUT_CONTEXT_IDS,
    'reference_forced': REFERENCE_RECIPE_ID,
    'promoted_ids': promoted_ids,
    'ranking_is_hypothesis_generation_only': True,
}
(RUN_DIR / 'phase-a-promotion.json').write_text(
    json.dumps(promotion, indent=2), encoding='utf-8'
)
print('Recettes promues :', promoted_ids)

figure, axis = plt.subplots(figsize=(10, 7))
for mechanism, part in phase_a_aggregate_frame.groupby('mechanism'):
    axis.scatter(
        part.maximum_mean_absolute_change, part.mean_ssr,
        s=90, label=mechanism,
    )
    for _, row in part.iterrows():
        axis.annotate(
            row.recipe,
            (row.maximum_mean_absolute_change, row.mean_ssr),
            fontsize=8,
        )
axis.set(
    xlabel='Modification absolue maximale',
    ylabel='SSR moyen p2/p3',
    title='Phase A — frontière fonction / préservation',
)
axis.grid(alpha=0.25)
axis.legend()
figure.tight_layout()
figure.savefig(RUN_DIR / 'phase-a-pareto.png', dpi=160)
display(figure)
"""
    ),
    markdown("## 8. Phase B — comparer 2, 4, 6 et 8 pas sur quatre contextes"),
    code(
        """for context_id in CONTEXT_IDS:
    for rescue_steps in PHASE_B_STEP_COUNTS:
        for recipe in promoted_recipes:
            print('Phase B :', context_id, rescue_steps, recipe)
            run_candidate(
                'phase_b', contexts[context_id], recipe, rescue_steps
            )

phase_b_rows = [
    row for row in result_rows() if row['phase'] == 'phase_b'
]
expected_phase_b = (
    len(CONTEXT_IDS) * len(PHASE_B_STEP_COUNTS) * len(promoted_recipes)
)
if len(phase_b_rows) != expected_phase_b:
    raise RuntimeError(
        f'Phase B incomplète : {len(phase_b_rows)}/{expected_phase_b}'
    )

all_rows = result_rows()
expected_total = len(CONTEXT_IDS) + expected_phase_a + expected_phase_b
if len(all_rows) != expected_total:
    raise RuntimeError(
        f'Campagne incomplète : {len(all_rows)}/{expected_total}'
    )

pairing_audit = {'phase_a': {}, 'phase_b': {}}
for context_id in SCREENING_CONTEXT_IDS:
    subset = [
        row for row in phase_a_rows if row['context_id'] == context_id
    ]
    audit = {
        'source_latent_hashes': sorted({row['source_latent_sha256'] for row in subset}),
        'rescue_noise_hashes': sorted({row['rescue_noise_sha256'] for row in subset}),
        'initial_rescue_latent_hashes': sorted({row['initial_rescue_latent_sha256'] for row in subset}),
        'structural_mask_hashes': sorted({row['structural_mask_sha256'] for row in subset}),
        'late_timestep_schedules': sorted({tuple(row['late_timesteps']) for row in subset}),
    }
    if any(len(values) != 1 for values in audit.values()):
        raise RuntimeError(f'Phase A non appariée pour {context_id}: {audit}')
    pairing_audit['phase_a'][context_id] = audit

for context_id in CONTEXT_IDS:
    for rescue_steps in PHASE_B_STEP_COUNTS:
        subset = [
            row for row in phase_b_rows
            if row['context_id'] == context_id
            and int(row['rescue_steps']) == rescue_steps
        ]
        audit = {
            'source_latent_hashes': sorted({row['source_latent_sha256'] for row in subset}),
            'rescue_noise_hashes': sorted({row['rescue_noise_sha256'] for row in subset}),
            'initial_rescue_latent_hashes': sorted({row['initial_rescue_latent_sha256'] for row in subset}),
            'structural_mask_hashes': sorted({row['structural_mask_sha256'] for row in subset}),
            'late_timestep_schedules': sorted({tuple(row['late_timesteps']) for row in subset}),
        }
        if any(len(values) != 1 for values in audit.values()):
            raise RuntimeError(
                f'Phase B non appariée pour {context_id}/{rescue_steps}: {audit}'
            )
        pairing_audit['phase_b'][f'{context_id}/steps{rescue_steps:02d}'] = audit

(RUN_DIR / 'paired-input-audit.json').write_text(
    json.dumps(pairing_audit, indent=2), encoding='utf-8'
)
print('Campagne complète :', len(all_rows), 'résultats')
"""
    ),
    markdown("## 9. Classement fixe, holdout et diagnostic contextuel"),
    code(
        """phase_b_frame = pd.DataFrame(phase_b_rows)
phase_b_frame.to_csv(RUN_DIR / 'phase-b-results.csv', index=False)
control_frame = pd.DataFrame(control_rows).set_index('context_id')


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


aggregate_rows = []
for recipe in promoted_recipes:
    for rescue_steps in PHASE_B_STEP_COUNTS:
        part = phase_b_frame[
            (phase_b_frame.recipe == recipe['id'])
            & (phase_b_frame.rescue_steps == rescue_steps)
        ].set_index('context_id').loc[CONTEXT_IDS]
        control = control_frame.loc[CONTEXT_IDS]
        aesthetic_drop = control.clip_aesthetic - part.clip_aesthetic
        holdout = part.loc[HOLDOUT_CONTEXT_IDS]
        fixed_gate = bool(
            int(part.original_probe_all.sum()) == len(CONTEXT_IDS)
            and float(part.pass_rate.mean()) > float(control.pass_rate.mean())
            and bool((aesthetic_drop <= MAX_AESTHETIC_DROP).all())
            and bool((part.mean_absolute_change <= MAX_MEAN_ABSOLUTE_CHANGE).all())
        )
        aggregate_rows.append({
            'recipe': recipe['id'],
            'mechanism': recipe['mechanism'],
            'global_alpha': recipe['global_alpha'],
            'structural_strength': recipe['structural_strength'],
            'rescue_steps': rescue_steps,
            'original_3of3_contexts': int(part.original_probe_all.sum()),
            'strict_39of39_contexts': int(part.strict_all.sum()),
            'mean_ssr': float(part.pass_rate.mean()),
            'minimum_ssr': float(part.pass_rate.min()),
            'mean_clip_aesthetic': float(part.clip_aesthetic.mean()),
            'mean_clip_score': float(part.clip_score.mean()),
            'maximum_aesthetic_drop': float(aesthetic_drop.max()),
            'maximum_mean_absolute_change': float(part.mean_absolute_change.max()),
            'holdout_original_3of3_contexts': int(holdout.original_probe_all.sum()),
            'holdout_mean_ssr': float(holdout.pass_rate.mean()),
            'holdout_mean_clip_aesthetic': float(holdout.clip_aesthetic.mean()),
            'fixed_recipe_gate': fixed_gate,
        })

aggregate_frame = pd.DataFrame(aggregate_rows).sort_values(
    [
        'fixed_recipe_gate', 'original_3of3_contexts',
        'holdout_original_3of3_contexts', 'strict_39of39_contexts',
        'minimum_ssr', 'mean_ssr', 'mean_clip_aesthetic',
        'maximum_mean_absolute_change',
    ],
    ascending=[False, False, False, False, False, False, False, True],
)
aggregate_frame.to_csv(
    RUN_DIR / 'phase-b-fixed-aggregates.csv', index=False
)
display(aggregate_frame)

selected_rows = []
for context_id in CONTEXT_IDS:
    candidates = [
        row for row in phase_b_rows if row['context_id'] == context_id
    ]
    selected_rows.append(max(candidates, key=rank_tuple))
selected_frame = pd.DataFrame(selected_rows)
selected_frame.to_csv(
    RUN_DIR / 'context-adaptive-oracle.csv', index=False
)

winner = {
    key: (value.item() if isinstance(value, np.generic) else value)
    for key, value in aggregate_frame.iloc[0].to_dict().items()
}
if bool(winner['fixed_recipe_gate']) and int(winner['strict_39of39_contexts']) == 4:
    status = 'production_candidate_pending_physical'
    next_action = 'Confirmer sur nouveaux prompts, téléphones et impressions.'
elif bool(winner['fixed_recipe_gate']):
    status = 'fixed_low_damage_candidate'
    next_action = 'Confirmer la recette fixe sur prompts et graines inconnus.'
elif int(winner['original_3of3_contexts']) == len(CONTEXT_IDS):
    status = 'functional_tradeoff_only'
    next_action = (
        'Conserver la recette comme secours fonctionnel, puis réduire encore '
        'la rediffusion ou localiser les contours fragiles.'
    )
else:
    status = 'rejected'
    next_action = 'Ne pas entraîner de sélecteur ; revoir le mécanisme.'

decision = {
    'status': status,
    'fixed_winner': winner,
    'screening_contexts': SCREENING_CONTEXT_IDS,
    'holdout_contexts': HOLDOUT_CONTEXT_IDS,
    'promoted_recipes': promoted_ids,
    'context_adaptive_original_3of3_contexts': int(
        selected_frame.original_probe_all.sum()
    ),
    'ranking': [
        'original 3/3', 'SSR 39 tests', 'pire décodeur',
        'pire scénario', 'CLIP-aesthetic', 'CLIPScore', 'préservation',
    ],
    'selector_training_allowed': False,
    'no_post_diffusion_pixel_projection': True,
    'next': next_action,
}
(RUN_DIR / 'DECISION.json').write_text(
    json.dumps(decision, indent=2), encoding='utf-8'
)
print('Décision :', decision)

figure, axes = plt.subplots(1, 2, figsize=(16, 6))
for recipe_id in promoted_ids:
    part = aggregate_frame[aggregate_frame.recipe == recipe_id].sort_values(
        'rescue_steps'
    )
    axes[0].plot(
        part.rescue_steps, part.mean_ssr, marker='o', label=recipe_id
    )
    axes[1].plot(
        part.rescue_steps, part.mean_clip_aesthetic,
        marker='o', label=recipe_id,
    )
axes[0].set(
    title='Phase B — SSR moyen par fenêtre',
    xlabel='Pas tardifs', ylabel='SSR moyen', ylim=(0, 1),
)
axes[1].set(
    title='Phase B — esthétique moyenne par fenêtre',
    xlabel='Pas tardifs', ylabel='CLIP-aesthetic',
)
for axis in axes:
    axis.set_xticks(PHASE_B_STEP_COUNTS)
    axis.grid(alpha=0.25)
    axis.legend()
figure.tight_layout()
figure.savefig(RUN_DIR / 'phase-b-window-summary.png', dpi=160)
display(figure)

decoder_rows = []
for row in all_rows:
    records = json.loads(
        (RUN_DIR / row['relative_output_dir'] / 'validations.json')
        .read_text(encoding='utf-8')
    )
    for record in records:
        decoder_rows.append({
            'phase': row['phase'], 'context_id': row['context_id'],
            'recipe': row['recipe'], 'rescue_steps': row['rescue_steps'],
            'decoder': record['decoder'], 'scenario': record['scenario'],
            'passed': int(record['exact_payload_match']),
        })
pd.DataFrame(decoder_rows).to_csv(
    RUN_DIR / 'decoder-scenario-results.csv', index=False
)

physical_rows = []
for row in selected_rows:
    physical_rows.append({
        'context_id': row['context_id'],
        'recipe': row['recipe'],
        'rescue_steps': row['rescue_steps'],
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
"""
    ),
    markdown("## 10. Manifeste, rapport automatique et archive"),
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
    'phase_a': {
        'contexts': SCREENING_CONTEXT_IDS,
        'steps': PHASE_A_STEPS,
        'recipes': PHASE_A_RECIPES,
        'promotion': promotion,
    },
    'phase_b': {
        'contexts': CONTEXT_IDS,
        'holdout_contexts': HOLDOUT_CONTEXT_IDS,
        'step_counts': PHASE_B_STEP_COUNTS,
        'promoted_recipes': promoted_recipes,
    },
    'fresh_pipeline_per_candidate': True,
    'same_noise_within_context': True,
    'paired_input_audit': pairing_audit,
    'mask': (
        'quiet zone plus finder/separator, timing, format/version and '
        'alignment modules; data modules excluded'
    ),
    'post_diffusion_pixel_projection': False,
    'promotion_limits': {
        'maximum_clip_aesthetic_drop': MAX_AESTHETIC_DROP,
        'maximum_mean_absolute_change': MAX_MEAN_ABSOLUTE_CHANGE,
    },
    'decision': decision,
    'claim': (
        'Mechanism and late-window ablation after E014D; '
        'hypothesis generation on four contexts, not selector training '
        'and not production proof.'
    ),
}
(RUN_DIR / 'manifest.json').write_text(
    json.dumps(manifest, indent=2), encoding='utf-8'
)

report_lines = [
    '# E014E — rapport automatique',
    '',
    f"- Statut : `{decision['status']}`",
    f"- Recettes promues : `{', '.join(promoted_ids)}`",
    f"- Gagnant fixe : `{winner['recipe']}` / `{winner['rescue_steps']}` pas",
    f"- Original 3/3 : `{winner['original_3of3_contexts']}/{len(CONTEXT_IDS)}` contextes",
    f"- Strict 39/39 : `{winner['strict_39of39_contexts']}/{len(CONTEXT_IDS)}` contextes",
    f"- SSR moyen : `{winner['mean_ssr']:.4f}`",
    f"- CLIP-aesthetic moyen : `{winner['mean_clip_aesthetic']:.4f}`",
    f"- Perte esthétique maximale : `{winner['maximum_aesthetic_drop']:.4f}`",
    f"- Modification absolue maximale : `{winner['maximum_mean_absolute_change']:.4f}`",
    '',
    'Le classement est exploratoire. Aucun sélecteur ne peut être entraîné '
    'à partir de quatre contextes.',
]
(RUN_DIR / 'REPORT.md').write_text(
    '\\n'.join(report_lines) + '\\n', encoding='utf-8'
)

for context in contexts.values():
    context['source_latent_cpu'] = None
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print('VRAM allouée après nettoyage GiB :', torch.cuda.memory_allocated() / 2**30)

archive = shutil.make_archive(
    str(RUN_DIR), 'gztar', RUN_DIR.parent, RUN_DIR.name
)
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
