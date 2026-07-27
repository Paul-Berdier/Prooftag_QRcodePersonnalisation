"""Build E014B v2 statistical FreeQR confirmation without editing notebook JSON."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "16_e014b_statistical_freeqr_confirmation.ipynb"


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


CELLS = [
    markdown(
        r"""# E014B v2 — confirmation statistique FreeQR sur un cas difficile

E014C a montré qu'un run Stage 2 complet peut laisser un état GPU/pipeline qui influence légèrement
le run suivant. Une comparaison unique A/B n'est donc pas suffisante.

Ce notebook ne recherche plus de paramètres. Il confirme quatre recettes **figées** sur
`p3_detailed`, cas difficile d'E014A :

1. baseline DiffQRCoder sans fusion ;
2. fusion canal 1, alpha 0,15, toute la trajectoire ;
3. fusion canal 1, alpha 0,15, fenêtre early ;
4. fusion complète avec petit gradient central Prooftag.

Chaque recette est exécutée quatre fois. L'ordre suit un carré latin équilibré de Williams :
chaque recette apparaît une fois à chaque position et chaque transition entre deux recettes
apparaît une fois. Une pipeline fraîche est chargée au début de chaque répétition. Le résultat
est interprété comme une distribution, jamais comme une paire bit-à-bit.
"""
    ),
    markdown(
        r"""## Plan expérimental

```text
répétition 1 : baseline → fusion all → fusion gradient → fusion early
répétition 2 : fusion all → fusion early → baseline → fusion gradient
répétition 3 : fusion early → fusion gradient → fusion all → baseline
répétition 4 : fusion gradient → baseline → fusion early → fusion all

Chaque répétition :
    pipeline fraîche
        │
        ├── même Stage 1
        ├── même latent initial
        ├── même seed
        └── ordre différent et équilibré
```

Une recette n'est promue que si son gain de scannabilité dépasse la variabilité observée de la
baseline et ne détruit pas l'esthétique. Aucune image n'est livrable sans porte 39/39.
"""
    ),
    code(
        """from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import ControlNetModel, DDIMScheduler
from IPython.display import Markdown, clear_output, display
from PIL import Image
from safetensors.torch import load_file, save_file

UPSTREAM_ROOT = Path('/opt/DiffQRCoder')
if not (UPSTREAM_ROOT / 'diffqrcoder' / 'pipeline_diffqrcoder.py').exists():
    raise RuntimeError('DiffQRCoder absent : reconstruire Dockerfile.notebook.')
sys.path.insert(0, str(UPSTREAM_ROOT))

from diffqrcoder import DiffQRCoderPipeline  # noqa: E402
import diffqrcoder.srpg as upstream_srpg  # noqa: E402
from prooftag_qr.geometry import AlignedQR, aligned_module_diagnostics  # noqa: E402
from prooftag_qr.quality_scoring import CLIPQualityScorer  # noqa: E402
from prooftag_qr.validation import QRValidator, summarize_validation_records  # noqa: E402


class PaperLPIPSLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = lpips.LPIPS(net='vgg', verbose=False)
        self.model.requires_grad_(False).eval()

    def forward(self, x, y):
        return self.model(x * 2 - 1, y * 2 - 1).mean()


upstream_srpg.PerceptualLoss = PaperLPIPSLoss
assert torch.cuda.is_available(), 'Lancer dans le pod GPU.'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=False)
free_gib, total_gib = torch.cuda.mem_get_info()
print('GPU :', torch.cuda.get_device_name(0))
print(f'VRAM libre : {free_gib / 2**30:.2f} / {total_gib / 2**30:.2f} Gio')
if free_gib / 2**30 < 18.0:
    raise RuntimeError(
        'Moins de 18 Gio libres. Depuis PowerShell, relancer avec '
        r'.\\scripts\\notebook-remote.ps1 -Reset '
        r'-Notebook 16_e014b_statistical_freeqr_confirmation.ipynb'
    )
"""
    ),
    markdown("## 1. Source E014A, cas difficile et contrat QR"),
    code(
        """EXPERIMENT_NAME = 'e014b-statistical-freeqr-confirmation-v2'
E014A_RUN_DIR = None
PROMPT_ID = 'p3_detailed'
RESUME_RUN_NAME = None

BASE_MODEL_URL = 'https://huggingface.co/fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned/blob/main/cetusMix_Whalefall2_fp16.safetensors'
CONTROLNET_MODEL = 'monster-labs/control_v1p_sd15_qrcode_monster'
CONTROLNET_SUBFOLDER = 'v2'
STEPS = 40
GUIDANCE_SCALE = 7.5
CONTROLNET_SCALE = 1.35
SCANNING_GUIDANCE = 500.0
PERCEPTUAL_GUIDANCE = 3.0
NEGATIVE_PROMPT = 'easynegative, unreadable text, letters, watermark'
REPEATS = 4
SAVE_PREVIEW_EVERY = 5

if E014A_RUN_DIR is None:
    candidates = sorted(
        Path('/data/notebook-runs').glob('*-e014a-deterministic-blueprint-pairing-v2')
    )
    if not candidates:
        raise FileNotFoundError('Aucun run E014A v2 sous /data/notebook-runs.')
    E014A_RUN_DIR = candidates[-1]
E014A_RUN_DIR = Path(E014A_RUN_DIR)
source_dir = E014A_RUN_DIR / PROMPT_ID
required = [
    E014A_RUN_DIR / 'manifest.json',
    source_dir / 'selected-meta.json',
    source_dir / 'selected-blueprint.png',
    source_dir / 'selected-matrix.npy',
    source_dir / 'stage1.safetensors',
    source_dir / 'stage1-reference.png',
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError('Artefacts E014A manquants : ' + ', '.join(missing))

source_manifest = json.loads(required[0].read_text(encoding='utf-8'))
meta = json.loads(required[1].read_text(encoding='utf-8'))
prompt_case = next(item for item in source_manifest['prompts'] if item['id'] == PROMPT_ID)
prompt = prompt_case['text']
seed = int(prompt_case['seed'])
payload = meta['payload']
blueprint_image = Image.open(required[2]).convert('RGB')
matrix = np.load(required[3]).astype(np.uint8)
aligned = AlignedQR(
    image=blueprint_image, core_matrix=matrix, version=meta['version'],
    error_correction='M', mask_pattern=-1, module_size=meta['module_size'],
    padding_px=meta['padding_px'], canvas_size=meta['canvas_size'], payload=payload,
)
stage1_tensor_cpu = load_file(str(required[4]), device='cpu')['stage1']

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
print('Source :', E014A_RUN_DIR)
print('Prompt :', prompt)
print('Blueprint :', meta['selected_blueprint'])
print('Seed :', seed)
print('Sortie :', RUN_DIR)
display(blueprint_image.resize((384, 384)))
"""
    ),
    markdown("## 2. Recettes figées et ordre latin"),
    code(
        """RECIPES = [
    {
        'id': 'baseline', 'channel': None, 'alpha': 0.0,
        'window': [0.0, 1.0], 'scan_gradient': False,
        'scan_lr': 0.0, 'scan_every': 4,
    },
    {
        'id': 'fusion_all', 'channel': 1, 'alpha': 0.15,
        'window': [0.0, 1.0], 'scan_gradient': False,
        'scan_lr': 0.0, 'scan_every': 4,
    },
    {
        'id': 'fusion_early', 'channel': 1, 'alpha': 0.15,
        'window': [0.0, 0.35], 'scan_gradient': False,
        'scan_lr': 0.0, 'scan_every': 4,
    },
    {
        'id': 'fusion_gradient', 'channel': 1, 'alpha': 0.15,
        'window': [0.0, 1.0], 'scan_gradient': True,
        'scan_lr': 0.06, 'scan_every': 4,
    },
]
recipe_by_id = {item['id']: item for item in RECIPES}
recipe_ids = [item['id'] for item in RECIPES]
WILLIAMS_FIRST_ROW = [0, 1, 3, 2]
LATIN_ORDERS = [
    [
        recipe_ids[(recipe_index + offset) % len(recipe_ids)]
        for recipe_index in WILLIAMS_FIRST_ROW
    ]
    for offset in range(REPEATS)
]
assert REPEATS == len(RECIPES) == 4
assert all(sorted(order) == sorted(recipe_ids) for order in LATIN_ORDERS)
transitions = [
    (order[index], order[index + 1])
    for order in LATIN_ORDERS
    for index in range(len(order) - 1)
]
assert len(transitions) == len(set(transitions)) == 12
display(pd.DataFrame(LATIN_ORDERS, index=[f'r{index + 1}' for index in range(REPEATS)]))
"""
    ),
    markdown("## 3. Validation du blueprint et utilitaires déterministes"),
    code(
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


def validation_summary(image):
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
        'worst_decoder_pass_rate': summary['worst_decoder_pass_rate'],
        'worst_scenario_pass_rate': summary['worst_scenario_pass_rate'],
    }, [asdict(item) for item in records]


blueprint_validation, blueprint_records = validation_summary(blueprint_image)
(RUN_DIR / 'blueprint-validations.json').write_text(
    json.dumps(blueprint_records, indent=2), encoding='utf-8'
)
if not blueprint_validation['strict_all']:
    raise RuntimeError(
        f"Le blueprint source ne passe que {blueprint_validation['passed']}/"
        f"{blueprint_validation['total']} : campagne interdite."
    )
print('Blueprint source :', blueprint_validation['passed'], '/', blueprint_validation['total'])
"""
    ),
    markdown("## 4. Pipeline fraîche par répétition"),
    code(
        """def load_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL, subfolder=CONTROLNET_SUBFOLDER,
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
def paired_inputs(pipeline, stage1_tensor):
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
    ),
    markdown("## 5. Fusion latente et gradient central optionnel"),
    code(
        """target_dark = torch.as_tensor(
    aligned.core_matrix.astype(bool), device='cuda'
).unsqueeze(0).unsqueeze(0)


def differentiable_module_loss(pipeline, latent):
    dtype = next(pipeline.vae.parameters()).dtype
    decoded = pipeline.vae.decode(
        latent.to(dtype=dtype) / pipeline.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    unit = (decoded.float() / 2 + 0.5).clamp(0, 1)
    gray = 0.299 * unit[:, 0:1] + 0.587 * unit[:, 1:2] + 0.114 * unit[:, 2:3]
    p = aligned.padding_px
    core = gray[:, :, p:p + aligned.core_size, p:p + aligned.core_size]
    modules = core.reshape(
        1, 1, aligned.core_modules, aligned.module_size,
        aligned.core_modules, aligned.module_size,
    ).permute(0, 1, 2, 4, 3, 5)
    inset = aligned.module_size // 3
    centers = modules[..., inset:-inset, inset:-inset].mean(dim=(-1, -2))
    dark_loss = F.relu(centers - 0.45)
    light_loss = F.relu(0.65 - centers)
    return torch.where(target_dark, dark_loss, light_loss).mean()


def fusion_active(step_index, window):
    progress = step_index / max(STEPS - 1, 1)
    return window[0] <= progress <= window[1]


def callback_for(
    pipeline, recipe, output_dir, blueprint_latent, paired_noise
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
        active = (
            recipe['channel'] is not None
            and fusion_active(step_index, recipe['window'])
        )
        scan_loss_value = None
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
        if (
            recipe['scan_gradient'] and active
            and step_index % recipe['scan_every'] == 0
        ):
            with torch.enable_grad():
                working = latent.detach().float().requires_grad_(True)
                loss = differentiable_module_loss(pipeline, working)
                gradient = torch.autograd.grad(loss, working)[0]
                channel = recipe['channel']
                channel_gradient = gradient[:, channel:channel + 1]
                rms = channel_gradient.square().mean().sqrt().clamp_min(1e-8)
                updated = working.detach()
                updated[:, channel:channel + 1] -= (
                    recipe['scan_lr'] * channel_gradient / rms
                )
                latent = updated.to(dtype=callback_kwargs['latents'].dtype).detach()
                scan_loss_value = float(loss.detach().cpu())
        trace.append({
            'step': int(step_index),
            'timestep_before_step': int(timestep),
            'target_timestep_after_step': int(next_timestep),
            'fusion_applied': active,
            'scan_loss': scan_loss_value,
            'latent_sha256': tensor_sha256(latent),
        })
        if step_index % SAVE_PREVIEW_EVERY == 0 or step_index == STEPS - 1:
            preview = decode_latent(pipeline, latent)
            preview.save(frames / f'{step_index:03d}.jpg', quality=88)
            if step_index % 10 == 0 or step_index == STEPS - 1:
                clear_output(wait=True)
                display(Markdown(
                    f"**{recipe['id']} — étape {step_index + 1}/{STEPS}**"
                ))
                display(preview.resize((384, 384)))
        callback_kwargs['latents'] = latent
        return callback_kwargs

    return callback, trace
"""
    ),
    markdown("## 6. Exécution persistée sans reprise partielle d'un bloc"),
    code(
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


def completed_repeats():
    rows = result_rows()
    complete = set()
    for repeat in range(1, REPEATS + 1):
        subset = [row for row in rows if row['repeat'] == repeat]
        if not subset:
            abandoned = sorted(RUN_DIR.glob(f'r{repeat:02d}-p*-*'))
            if abandoned:
                raise RuntimeError(
                    f'Répétition {repeat} interrompue avant la première ligne de résultat. '
                    f'Artefacts partiels : {[path.name for path in abandoned]}. '
                    'La reprendre changerait l historique de pipeline. Démarrer un nouveau run.'
                )
            continue
        if len(subset) != len(RECIPES):
            raise RuntimeError(
                f'Répétition {repeat} partielle ({len(subset)}/{len(RECIPES)}). '
                'La reprendre changerait l historique de pipeline. Démarrer un nouveau run.'
            )
        if {row['recipe'] for row in subset} != set(recipe_ids):
            raise RuntimeError(f'Répétition {repeat} incohérente.')
        complete.add(repeat)
    return complete


def make_gif(folder, output):
    paths = sorted(folder.glob('*.jpg'))
    frames = [Image.open(path).convert('RGB').resize((512, 512)) for path in paths]
    if frames:
        frames[0].save(
            output, save_all=True, append_images=frames[1:],
            duration=220, loop=0,
        )
    for frame in frames:
        frame.close()


def run_recipe(
    pipeline, stage1_tensor, blueprint_latent, initial, noise,
    recipe, repeat, position,
):
    run_name = f"r{repeat:02d}-p{position:02d}-{recipe['id']}"
    output_dir = RUN_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    release_guidance(pipeline)
    seed_everything(seed + 10000)
    callback, trace = callback_for(
        pipeline, recipe, output_dir, blueprint_latent, noise
    )
    started = time.perf_counter()
    result = pipeline._run_stage2(
        prompt=prompt, qrcode=blueprint_image,
        qrcode_module_size=aligned.module_size,
        qrcode_padding=aligned.padding_px,
        ref_image=stage1_tensor, negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=STEPS, guidance_scale=GUIDANCE_SCALE, eta=0.0,
        generator=torch.Generator(device='cuda').manual_seed(seed + 10000),
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
    validation, records = validation_summary(image)
    (output_dir / 'validations.json').write_text(
        json.dumps(records, indent=2), encoding='utf-8'
    )
    try:
        quality = asdict(quality_scorer.score(image, prompt))
        quality_error = None
    except Exception as exc:
        quality = {
            'clip_similarity': None, 'clip_score': None,
            'clip_aesthetic': None,
        }
        quality_error = f'{type(exc).__name__}: {exc}'
    row = {
        'run_name': run_name,
        'repeat': repeat,
        'position': position,
        'order': LATIN_ORDERS[repeat - 1],
        'recipe': recipe['id'],
        'config': recipe,
        'prompt_id': PROMPT_ID,
        'prompt': prompt,
        'seed': seed,
        'initial_latent_sha256': tensor_sha256(initial),
        'final_latent_sha256': tensor_sha256(final_latent),
        'duration_s': duration,
        **validation,
        **quality,
        **aligned_module_diagnostics(image, aligned),
        'quality_error': quality_error,
    }
    append_row(row)
    print(
        run_name, validation['passed'], '/', validation['total'],
        'original', validation['original_passed'], '/', validation['original_total'],
    )
    del result, final_latent
    release_guidance(pipeline)
    gc.collect()
    torch.cuda.empty_cache()
"""
    ),
    markdown("## 7. Campagne équilibrée : quatre pipelines fraîches"),
    code(
        """complete = completed_repeats()
for repeat, order in enumerate(LATIN_ORDERS, start=1):
    if repeat in complete:
        print('SKIP répétition complète', repeat)
        continue
    print('Chargement pipeline fraîche pour répétition', repeat, order)
    pipe = load_pipeline()
    stage1_tensor = stage1_tensor_cpu.to('cuda', dtype=torch.float16)
    blueprint_latent = encode_image(pipe, blueprint_image)
    paired_initial, paired_noise = paired_inputs(pipe, stage1_tensor)
    initial_hash = tensor_sha256(paired_initial)
    print('Latent initial :', initial_hash)
    for position, recipe_id in enumerate(order, start=1):
        run_recipe(
            pipe, stage1_tensor, blueprint_latent,
            paired_initial, paired_noise,
            recipe_by_id[recipe_id], repeat, position,
        )
    del stage1_tensor, blueprint_latent, paired_initial, paired_noise
    close_pipeline(pipe)
    del pipe

rows = result_rows()
if len(rows) != REPEATS * len(RECIPES):
    raise RuntimeError(
        f'Campagne incomplète : {len(rows)}/{REPEATS * len(RECIPES)} résultats.'
    )
initial_hashes_by_repeat = {
    repeat: sorted({
        row['initial_latent_sha256'] for row in rows if row['repeat'] == repeat
    })
    for repeat in range(1, REPEATS + 1)
}
if any(len(hashes) != 1 for hashes in initial_hashes_by_repeat.values()):
    raise RuntimeError(
        'Une répétition ne partage pas un latent initial unique entre ses quatre recettes.'
    )
print('Campagne complète :', len(rows), 'résultats')
print('Hashes initiaux par répétition :', initial_hashes_by_repeat)
"""
    ),
    markdown("## 8. Effets appariés, variabilité du contrôle et décision"),
    code(
        """frame = pd.json_normalize(rows, sep='__')
frame.to_csv(RUN_DIR / 'comparison.csv', index=False)


def bootstrap_mean_ci(values, iterations=20000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260727)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True)
    means = samples.mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


baseline = frame[frame.recipe == 'baseline'].sort_values('repeat')
baseline_by_repeat = {
    int(row.repeat): float(row.pass_rate)
    for row in baseline.itertuples()
}
baseline_span = float(baseline.pass_rate.max() - baseline.pass_rate.min())
baseline_aesthetic = float(baseline.clip_aesthetic.mean())

aggregates = []
paired_effects = {}
for recipe_id in recipe_ids:
    part = frame[frame.recipe == recipe_id].sort_values('repeat')
    pass_rates = part.pass_rate.astype(float).tolist()
    aggregate = {
        'recipe': recipe_id,
        'runs': len(part),
        'strict_count': int(part.strict_all.sum()),
        'original_3of3_count': int((part.original_passed == 3).sum()),
        'ssr_mean': float(part.pass_rate.mean()),
        'ssr_median': float(part.pass_rate.median()),
        'ssr_min': float(part.pass_rate.min()),
        'ssr_max': float(part.pass_rate.max()),
        'ssr_mean_ci95': bootstrap_mean_ci(pass_rates),
        'clip_aesthetic_mean': float(part.clip_aesthetic.mean()),
        'clip_score_mean': float(part.clip_score.mean()),
        'duration_mean_s': float(part.duration_s.mean()),
    }
    if recipe_id != 'baseline':
        diffs = [
            float(row.pass_rate) - baseline_by_repeat[int(row.repeat)]
            for row in part.itertuples()
        ]
        effect = {
            'paired_differences': diffs,
            'mean_difference': float(np.mean(diffs)),
            'median_difference': float(np.median(diffs)),
            'min_difference': float(np.min(diffs)),
            'max_difference': float(np.max(diffs)),
            'positive_repeats': int(sum(value > 0 for value in diffs)),
            'difference_ci95': bootstrap_mean_ci(diffs),
            'exceeds_baseline_span': float(np.mean(diffs)) > baseline_span,
            'aesthetic_drop': baseline_aesthetic - aggregate['clip_aesthetic_mean'],
        }
        effect['promoted'] = bool(
            effect['positive_repeats'] >= 3
            and effect['mean_difference'] > baseline_span
            and aggregate['ssr_min'] >= float(baseline.pass_rate.min())
            and effect['aesthetic_drop'] <= 0.5
        )
        paired_effects[recipe_id] = effect
        aggregate.update({
            'paired_mean_difference': effect['mean_difference'],
            'positive_repeats': effect['positive_repeats'],
            'promoted': effect['promoted'],
        })
    else:
        aggregate.update({
            'paired_mean_difference': 0.0,
            'positive_repeats': 0,
            'promoted': False,
        })
    aggregates.append(aggregate)

aggregate_frame = pd.DataFrame(aggregates).sort_values(
    ['promoted', 'ssr_mean', 'clip_aesthetic_mean'],
    ascending=[False, False, False],
)
aggregate_frame.to_csv(RUN_DIR / 'aggregate.csv', index=False)
(RUN_DIR / 'paired-effects.json').write_text(
    json.dumps(paired_effects, indent=2), encoding='utf-8'
)
display(aggregate_frame)

promoted = aggregate_frame[aggregate_frame.promoted]
if promoted.empty:
    decision = {
        'status': 'no_promotion',
        'reason': (
            'Aucune fusion ne dépasse la variabilité de la baseline sur au moins '
            'trois répétitions sans perte esthétique excessive.'
        ),
        'next': 'Ne pas lancer la confirmation multi-contexte ; revoir la fusion.',
    }
else:
    winner = promoted.iloc[0]
    decision = {
        'status': 'promoted',
        'recipe': winner.recipe,
        'ssr_mean': float(winner.ssr_mean),
        'paired_mean_difference': float(winner.paired_mean_difference),
        'next': 'Confirmer la recette figée sur p1, p2 et p4.',
    }
(RUN_DIR / 'DECISION.json').write_text(
    json.dumps(decision, indent=2), encoding='utf-8'
)
print('Décision :', decision)

figure, axes = plt.subplots(1, 2, figsize=(14, 5))
for recipe_id in recipe_ids:
    part = frame[frame.recipe == recipe_id].sort_values('repeat')
    axes[0].plot(part.repeat, part.pass_rate, marker='o', label=recipe_id)
axes[0].set(
    title='SSR par répétition et recette',
    xlabel='répétition', ylabel='SSR', xticks=range(1, REPEATS + 1),
)
axes[0].legend()
axes[0].grid(alpha=0.25)
axes[1].scatter(
    aggregate_frame.ssr_mean, aggregate_frame.clip_aesthetic_mean,
    s=100,
)
for row in aggregate_frame.itertuples():
    axes[1].annotate(row.recipe, (row.ssr_mean, row.clip_aesthetic_mean))
axes[1].set(
    title='Compromis moyen',
    xlabel='SSR moyen', ylabel='CLIP-aesthetic moyen',
)
axes[1].grid(alpha=0.25)
figure.tight_layout()
figure.savefig(RUN_DIR / 'statistical-comparison.png', dpi=160)
display(figure)
"""
    ),
    markdown("## 9. Manifeste, nettoyage GPU et archive"),
    code(
        """manifest = {
    'experiment': EXPERIMENT_NAME,
    'source_e014a': str(E014A_RUN_DIR),
    'prompt_id': PROMPT_ID,
    'prompt': prompt,
    'seed': seed,
    'payload': payload,
    'selected_e014a_blueprint': meta['selected_blueprint'],
    'steps': STEPS,
    'repeats': REPEATS,
    'latin_orders': LATIN_ORDERS,
    'recipes': RECIPES,
    'fresh_pipeline_per_repeat': True,
    'strict_deterministic_algorithms': True,
    'initial_hashes_by_repeat': initial_hashes_by_repeat,
    'same_initial_latent_across_repeats': (
        len({hashes[0] for hashes in initial_hashes_by_repeat.values()}) == 1
    ),
    'promotion_rule': {
        'positive_repeats_minimum': 3,
        'mean_gain_must_exceed_baseline_span': True,
        'worst_ssr_not_below_baseline_worst': True,
        'maximum_clip_aesthetic_drop': 0.5,
    },
    'decision': decision,
    'claim': (
        'FreeQR-inspired confirmation with balanced repeated measures; '
        'not an official FreeQR implementation.'
    ),
}
(RUN_DIR / 'manifest.json').write_text(
    json.dumps(manifest, indent=2), encoding='utf-8'
)

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
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(TARGET)
