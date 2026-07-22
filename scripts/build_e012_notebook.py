# ruff: noqa: E501
"""Build the auditable E012 notebook without hand-editing notebook JSON."""

from __future__ import annotations

import json
from pathlib import Path


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


cells = [
    markdown(
        r"""# E012 — DiffQRCoder et SR-MPGD réellement aligné sur le papier

Ce notebook remplace les variantes SR-MPGD d'E011. Il conserve le **latent propre exact de la Stage 2**, utilise le **QR binaire original** pour SRL, et optimise l'objectif séparé publié :

$$z_i = z_{i-1} - 1000 \, \nabla_z [L_{SR}(D(z), y) + 0{,}01 L_{LPIPS}(D(z), x_0)]$$

Le nombre d'itérations n'étant pas donné dans l'article, chaque état de 0 à 20 est sauvegardé, validé par tous les décodeurs/scénarios, puis la boucle s'arrête au premier 26/26. À défaut, elle conserve le meilleur état selon l'ordre : lecture, pire décodeur, pire scénario, LPIPS, MER.

La Stage 2 reçoit également un **vrai LPIPS VGG différentiable**. La cible QArt Reed–Solomon exacte n'étant pas publiée dans DiffQRCoder, la condition Stage 2 reste le proxy matriciel exporté et validé ; cette limite est inscrite dans le manifeste et interdit l'étiquette « reproduction bit-à-bit du papier ».
"""
    ),
    markdown(
        """## Chaîne observée et responsabilités

```text
QR v3/M/mask4 ──► Stage 1 ControlNet (40 pas)
                         │ image artistique x-hat
                         ├──► proxy QArt matriciel validé ──► condition Stage 2
                         └──► encodage VAE + bruit apparié ──► Stage 2 SRPG (40 ou 100 pas)
                                                                  │
                                                                  ├── image x0 + métriques
                                                                  └── latent propre z0 exact
                                                                        │
                                                                        ▼
                                                           SR-MPGD Eq. 13–14
                                                           SRL(QR original) + 0,01 LPIPS(x0)
                                                                        │
                                                         validation 26 tests à chaque itération
                                                                        │
                                             26/26 : arrêt et livraison / sinon meilleur observé
```

Les deux nombres de pas Stage 2 isolent l'effet observé localement : **40**, valeur de l'article, et **100**, valeur qui avait rendu un exemple lisible au téléphone. Aucun autre paramètre ne change entre ces deux profils.
"""
    ),
    code(
        """from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import lpips
import matplotlib.pyplot as plt
import numpy as np
import qrcode
import torch
from diffusers import ControlNetModel, DDIMScheduler
from IPython.display import Markdown, clear_output, display
from PIL import Image
from qrcode.exceptions import DataOverflowError
from safetensors.torch import load_file, save_file

UPSTREAM_ROOT = Path('/opt/DiffQRCoder')
EXPECTED_COMMIT = 'e24ea73ee2e13c7e6e87cb422e8b11784e70ae00'
if not (UPSTREAM_ROOT / 'diffqrcoder' / 'pipeline_diffqrcoder.py').exists():
    raise RuntimeError('DiffQRCoder absent : reconstruire Dockerfile.notebook')
sys.path.insert(0, str(UPSTREAM_ROOT))

from diffqrcoder import DiffQRCoderPipeline  # noqa: E402
import diffqrcoder.srpg as upstream_srpg  # noqa: E402
from diffqrcoder.losses import ScanningRobustLoss  # noqa: E402
from prooftag_qr.experiments import image_context_features  # noqa: E402
from prooftag_qr.qr import QRBlueprint, functional_pattern_mask, module_error_rate  # noqa: E402
from prooftag_qr.quality import image_change_metrics, image_quality_metrics  # noqa: E402
from prooftag_qr.quality_scoring import CLIPQualityScorer  # noqa: E402
from prooftag_qr.srmpgd import SRMPGDConfig, run_srmpgd  # noqa: E402
from prooftag_qr.validation import QRValidator, summarize_validation_records  # noqa: E402


class PaperLPIPSLoss(torch.nn.Module):
    # Le papier définit un LPIPS appris (Appendix B.1). Le dépôt public emploie une
    # moyenne de features VGG dont torch.tensor détache le graphe. Cette classe restaure
    # le LPIPS appris et transforme correctement [0,1] vers l'entrée LPIPS [-1,1].
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = lpips.LPIPS(net='vgg', verbose=False)
        self.model.requires_grad_(False).eval()

    def forward(self, x, y):
        return self.model(x * 2 - 1, y * 2 - 1).mean()


upstream_srpg.PerceptualLoss = PaperLPIPSLoss
UPSTREAM_PATCHES = [{
    'scope': 'Stage 2 SRPG',
    'reason': 'le papier emploie LPIPS et exige son gradient vers le latent',
    'upstream_problem': 'VGG feature MSE agrégé par torch.tensor, donc graphe détaché',
    'replacement': 'lpips.LPIPS(net=vgg), entrée [-1,1], paramètres gelés mais gradient d entrée conservé',
}]
UPSTREAM_HASHES = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [
        UPSTREAM_ROOT / 'diffqrcoder' / 'pipeline_diffqrcoder.py',
        UPSTREAM_ROOT / 'diffqrcoder' / 'srpg.py',
        UPSTREAM_ROOT / 'diffqrcoder' / 'losses' / 'scanning_robust_loss.py',
        UPSTREAM_ROOT / 'diffqrcoder' / 'losses' / 'perceptual_loss.py',
    ]
}

print('torch       :', torch.__version__)
print('diffusers   :', importlib.metadata.version('diffusers'))
print('lpips       :', importlib.metadata.version('lpips'))
print('CUDA        :', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'absent')
print('upstream    :', EXPECTED_COMMIT)
print('patch SRPG  :', UPSTREAM_PATCHES[0]['replacement'])
assert torch.cuda.is_available(), 'Exécuter ce notebook dans le pod GPU, pas avec Python Windows.'
assert importlib.metadata.version('diffusers') == '0.32.2'
"""
    ),
    markdown("## 1. Paramètres figés, profils appariés et reprise"),
    code(
        """EXPERIMENT_NAME = 'e012-diffqrcoder-faithful-srmpgd-v1'
RESUME_RUN_NAME = None  # mettre le nom d'un dossier existant après une interruption
PAYLOAD = 'https://ptag.io/t/e012'  # URL courte Prooftag ; doit tenir strictement en v3/M
NEGATIVE_PROMPT = 'easynegative'

PROMPTS = [
    {'id': 'p1_simple', 'seed': 1101, 'text': 'A single white lotus flower floating on a dark calm pond, elegant editorial photograph.'},
    {'id': 'p2_medium', 'seed': 2202, 'text': 'A Japanese garden with a red bridge, mossy stones and soft morning mist, detailed photography.'},
    {'id': 'p3_detailed', 'seed': 3303, 'text': 'An ornate botanical tapestry of white lilies, pale blue leaves and dark vines, intricate textile illustration.'},
    {'id': 'p4_complex', 'seed': 4404, 'text': 'A lively old European market square, café terraces, flowers, bicycles and a gothic cathedral, cinematic morning light.'},
]

QR_VERSION = 3
QR_ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_M
QR_MASK_PATTERN = 4
QR_MODULE_SIZE = 20
QR_BORDER_MODULES = 4
QR_INTERNAL_PADDING = 78  # sortie VAE 736 : 736 - 2*78 = 580 = 29 modules * 20

BASE_MODEL_URL = 'https://huggingface.co/fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned/blob/main/cetusMix_Whalefall2_fp16.safetensors'
CONTROLNET_MODEL = 'monster-labs/control_v1p_sd15_qrcode_monster'
CONTROLNET_SUBFOLDER = 'v2'
STAGE1_STEPS = 40
STAGE2_PROFILES = [
    {'name': 'paper40', 'steps': 40, 'srg': 500.0, 'pg': 3.0},
    {'name': 'observed100', 'steps': 100, 'srg': 500.0, 'pg': 3.0},
]
GUIDANCE_SCALE = 7.5
CONTROLNET_SCALE = 1.35
ETA = 0.0
DISPLAY_EVERY = 5
SAVE_EVERY_DIFFUSION_STEP = 1
MEMORY_PROFILE = 'rtx_20gb'

SRMPGD_CONFIG = SRMPGDConfig(
    max_iterations=20,       # non publié : tous les états sont évalués, arrêt au premier strict
    step_size=1000.0,        # gamma, Eq. 14 et ablation du papier
    lpips_weight=0.01,       # lambda, Eq. 13 et ablation du papier
    lpips_net='vgg',
    crop_padding_px=QR_INTERNAL_PADDING,
    dark_threshold=0.45,
    light_threshold=0.65,
    center_fraction=1 / 3,
)
QART_IMPLEMENTATION = 'matrix-preserving visual proxy; exact Reed-Solomon QArt unavailable upstream'
SRL_IMPLEMENTATION = 'public DiffQRCoder ScanningRobustLoss, unchanged'
EXPECTED_RESULTS = len(PROMPTS) * len(STAGE2_PROFILES) * 2

if RESUME_RUN_NAME:
    RUN_DIR = Path('/data/notebook-runs') / RESUME_RUN_NAME
    if not RUN_DIR.is_dir():
        raise FileNotFoundError(RUN_DIR)
else:
    run_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{EXPERIMENT_NAME}"
    RUN_DIR = Path('/data/notebook-runs') / run_name
    RUN_DIR.mkdir(parents=True, exist_ok=False)
print('Résultats :', RUN_DIR)
print('Résultats attendus :', EXPECTED_RESULTS)
"""
    ),
    markdown("## 2. QR de contrôle et portes d'intégrité"),
    code(
        """qr = qrcode.QRCode(
    version=QR_VERSION,
    error_correction=QR_ERROR_CORRECTION,
    box_size=QR_MODULE_SIZE,
    border=QR_BORDER_MODULES,
    mask_pattern=QR_MASK_PATTERN,
)
qr.add_data(PAYLOAD)
try:
    qr.make(fit=False)
except DataOverflowError as exc:
    raise ValueError('Payload trop long pour QR v3/M : raccourcir l URL, ne pas changer silencieusement de version.') from exc

qr_image = qr.make_image(fill_color='black', back_color='white').convert('RGB')
matrix = np.asarray(qr.get_matrix(), dtype=np.uint8)
blueprint = QRBlueprint(image=qr_image, matrix=matrix, version=QR_VERSION, border=QR_BORDER_MODULES)
payload_hash = hashlib.sha256(PAYLOAD.encode('utf-8')).hexdigest()
qr_image.save(RUN_DIR / '00_qr_control.png')
assert qr_image.size == (740, 740) and matrix.shape == (37, 37)

validator = QRValidator()
control_records = validator.validate(qr_image, PAYLOAD)
assert control_records and all(item.exact_payload_match for item in control_records), 'Le QR témoin doit réussir tous les tests.'
print('Décodeurs :', [decoder.name for decoder in validator.decoders])
print('QR témoin :', len(control_records), '/', len(control_records), 'payload SHA-256', payload_hash)
display(qr_image.resize((370, 370)))
"""
    ),
    markdown("## 3. Instrumentation, persistance immédiate et validation"),
    code(
        """def cuda_memory_gib():
    free, total = torch.cuda.mem_get_info()
    return {
        'allocated': torch.cuda.memory_allocated() / 2**30,
        'reserved': torch.cuda.memory_reserved() / 2**30,
        'free_driver': free / 2**30,
        'total_driver': total / 2**30,
        'peak_allocated': torch.cuda.max_memory_allocated() / 2**30,
    }


def release_previous_gpu_objects():
    names = ['quality_scorer', 'pipe', 'controlnet', 'stage1_tensor', 'base_latent', 'srmpgd_result']
    for name in names:
        value = globals().pop(name, None)
        if value is not None and hasattr(value, 'to'):
            try:
                value.to('cpu')
            except Exception:
                pass
    shell = get_ipython()
    if shell is not None:
        history = shell.user_ns.get('Out')
        if isinstance(history, dict):
            history.clear()
        for history_name in ('_', '__', '___'):
            shell.user_ns.pop(history_name, None)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


RESULTS_PATH = RUN_DIR / 'results.jsonl'


def result_rows():
    if not RESULTS_PATH.exists():
        return []
    return [json.loads(line) for line in RESULTS_PATH.read_text(encoding='utf-8').splitlines() if line.strip()]


def result_index():
    return {(row['prompt_id'], row['profile'], row['variant']): row for row in result_rows()}


def row_key(row):
    return row['prompt_id'], row['profile'], row['variant']


def drop_result_keys(keys):
    keys = set(keys)
    retained = [row for row in result_rows() if row_key(row) not in keys]
    temporary = RESULTS_PATH.with_suffix('.rewrite')
    temporary.write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\\n' for row in retained),
        encoding='utf-8',
    )
    temporary.replace(RESULTS_PATH)


def trace_artifacts_complete(trace_path, frames_folder, gif_path, expected=None):
    if not trace_path.exists() or not gif_path.exists():
        return False
    try:
        trace = json.loads(trace_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    required = len(trace) if expected is None else expected
    return len(trace) == required and len(list(frames_folder.glob('*.png'))) == required


def append_result(row):
    with RESULTS_PATH.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\\n')
        stream.flush()


def save_tensor(path, key, tensor):
    # Safetensors conserve le dtype : le latent Stage 2 FP16 est repris bit pour bit.
    save_file({key: tensor.detach().cpu().contiguous()}, str(path))


def load_tensor(path, key, device='cuda', dtype=torch.float16):
    return load_file(str(path), device='cpu')[key].to(device=device, dtype=dtype)


def decode_latents(pipeline, latents):
    with torch.no_grad():
        decoded = pipeline.vae.decode(
            latents.to(dtype=next(pipeline.vae.parameters()).dtype) / pipeline.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        return pipeline.image_processor.postprocess(decoded.detach(), output_type='pil')[0].convert('RGB')


def make_gif(folder, output):
    paths = sorted(folder.glob('*.png'))
    if not paths:
        return
    frames = [Image.open(path).convert('RGB').resize((512, 512)) for path in paths]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=180, loop=0)
    for frame in frames:
        frame.close()


def validation_payload(image):
    records = validator.validate(image, PAYLOAD)
    summary = summarize_validation_records(records)
    passed = sum(item.exact_payload_match for item in records)
    originals = [item for item in records if item.scenario == 'original']
    original_passed = sum(item.exact_payload_match for item in originals)
    values = {
        'passed': passed,
        'total': len(records),
        'pass_rate': passed / len(records),
        'software_ssr': passed / len(records),
        'strict_all': passed == len(records),
        'original_passed': original_passed,
        'original_total': len(originals),
        'original_pass_rate': original_passed / len(originals) if originals else 0.0,
        'original_ssr': original_passed / len(originals) if originals else 0.0,
        'worst_decoder_pass_rate': summary['worst_decoder_pass_rate'],
        'worst_scenario_pass_rate': summary['worst_scenario_pass_rate'],
    }
    return values, [asdict(item) for item in records]


def rank_key(row):
    return (
        bool(row['strict_all']), row['pass_rate'], row['worst_decoder_pass_rate'],
        row['worst_scenario_pass_rate'], row.get('clip_aesthetic') or float('-inf'),
        row.get('clip_score') or float('-inf'), -row['module_error_rate'],
    )


print('VRAM avant nettoyage :', cuda_memory_gib())
release_previous_gpu_objects()
memory_after_cleanup = cuda_memory_gib()
print('VRAM après nettoyage :', memory_after_cleanup)
if memory_after_cleanup['allocated'] > 1.0 or memory_after_cleanup['free_driver'] < 15.0:
    raise RuntimeError('GPU non propre. Utiliser Kernel > Restart Kernel, puis Run All Cells et vérifier nvidia-smi.')
"""
    ),
    markdown("## 4. Chargement de Cetus-Mix et QR Monster v2"),
    code(
        """load_started = time.perf_counter()
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    subfolder=CONTROLNET_SUBFOLDER,
    torch_dtype=torch.float16,
    cache_dir='/cache/huggingface',
)
pipe = DiffQRCoderPipeline.from_single_file(
    BASE_MODEL_URL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    cache_dir='/cache/huggingface',
    safety_checker=None,
    use_safetensors=True,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to('cuda')
for component in [pipe.unet, pipe.controlnet, pipe.vae, pipe.text_encoder]:
    component.requires_grad_(False).eval()
pipe.enable_attention_slicing('max')
pipe.enable_vae_slicing()
pipe.unet.enable_gradient_checkpointing()
pipe.controlnet.enable_gradient_checkpointing()
official_srmpgd_srl = ScanningRobustLoss(module_size=QR_MODULE_SIZE).to(
    device='cuda', dtype=torch.float32
).requires_grad_(False).eval()
print(f'Pipeline chargée en {time.perf_counter() - load_started:.1f}s')
print('VRAM :', cuda_memory_gib())
"""
    ),
    markdown("## 5. Stage 1, condition QArt proxy et Stage 2 avec latent final exact"),
    code(
        """def qart_proxy(stage1_image):
    # Limite explicite : ce proxy ne réencode pas les blocs Reed-Solomon. Il conserve la matrice
    # et adapte les centres à l'image Stage 1. Le fichier et sa validation sont toujours exportés.
    source = np.asarray(stage1_image.resize(qr_image.size)).astype(np.float32)
    count = matrix.shape[0]
    protected = functional_pattern_mask(blueprint)
    for row in range(count):
        for col in range(count):
            y0, y1 = row * QR_MODULE_SIZE, (row + 1) * QR_MODULE_SIZE
            x0, x1 = col * QR_MODULE_SIZE, (col + 1) * QR_MODULE_SIZE
            fraction = 0.72 if protected[row, col] else 0.38
            margin = max(1, round(QR_MODULE_SIZE * (1 - fraction) / 2))
            target = 0 if matrix[row, col] else 255
            region = source[y0 + margin:y1 - margin, x0 + margin:x1 - margin]
            source[y0 + margin:y1 - margin, x0 + margin:x1 - margin] = 0.15 * region + 0.85 * target
    source[:QR_BORDER_MODULES * QR_MODULE_SIZE, :] = 255
    source[-QR_BORDER_MODULES * QR_MODULE_SIZE:, :] = 255
    source[:, :QR_BORDER_MODULES * QR_MODULE_SIZE] = 255
    source[:, -QR_BORDER_MODULES * QR_MODULE_SIZE:] = 255
    return Image.fromarray(np.clip(source, 0, 255).astype(np.uint8))


@torch.no_grad()
def paper_stage2_latents(pipeline, stage1_tensor, seed, steps):
    normalized = stage1_tensor.to('cuda', dtype=torch.float16) * 2 - 1
    encoded = pipeline.vae.encode(normalized).latent_dist.mode() * pipeline.vae.config.scaling_factor
    generator = torch.Generator(device='cuda').manual_seed(seed)
    noise = torch.randn(encoded.shape, generator=generator, device='cuda', dtype=encoded.dtype)
    pipeline.scheduler.set_timesteps(steps, device='cuda')
    return pipeline.scheduler.add_noise(encoded, noise, pipeline.scheduler.timesteps[:1])


def diffusion_callback(prompt_id, phase, steps):
    folder = RUN_DIR / 'frames' / prompt_id / phase
    folder.mkdir(parents=True, exist_ok=True)
    trace = []
    started = time.perf_counter()

    def callback(pipeline, step_index, timestep, callback_kwargs):
        if step_index % SAVE_EVERY_DIFFUSION_STEP == 0 or step_index == steps - 1:
            # Il s'agit de l'état latent DDIM après le pas, pas de x0|t. Le libellé évite
            # la confusion présente dans les notebooks précédents.
            preview = decode_latents(pipeline, callback_kwargs['latents'])
            row = {
                'step': int(step_index), 'timestep': int(timestep),
                'elapsed_s': time.perf_counter() - started,
                'decoded_latent_state_mer': module_error_rate(preview, blueprint),
            }
            trace.append(row)
            preview.save(folder / f'{step_index:03d}.png')
            if step_index % DISPLAY_EVERY == 0 or step_index == steps - 1:
                clear_output(wait=True)
                display(Markdown(f"**{prompt_id} / {phase} — {step_index + 1}/{steps} — état latent décodé, MER {row['decoded_latent_state_mer']:.2%}**"))
                display(preview.resize((430, 430)))
        return callback_kwargs

    return callback, trace, folder


@torch.no_grad()
def run_stage1(prompt_case, prompt_dir):
    callback, trace, folder = diffusion_callback(prompt_case['id'], 'stage1', STAGE1_STEPS)
    started = time.perf_counter()
    result = pipe._run_stage1(
        # Ne pas imposer 740 : le préprocesseur Diffusers ramène le QR 740 à 736,
        # taille divisible par 8 utilisée par la VAE et attendue par le padding 78.
        prompt=prompt_case['text'], qrcode=qr_image,
        negative_prompt=NEGATIVE_PROMPT, num_inference_steps=STAGE1_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=torch.Generator(device='cuda').manual_seed(prompt_case['seed']),
        controlnet_conditioning_scale=CONTROLNET_SCALE,
        callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents'],
        output_type='pt',
    )
    tensor = result.images.detach()
    image = pipe.image_processor.numpy_to_pil(pipe.image_processor.pt_to_numpy(tensor))[0].convert('RGB')
    image.save(prompt_dir / 'stage1.png')
    save_tensor(prompt_dir / 'stage1.safetensors', 'stage1', tensor)
    duration = time.perf_counter() - started
    (prompt_dir / 'stage1-time.json').write_text(
        json.dumps({'duration_s': duration}, indent=2), encoding='utf-8'
    )
    (prompt_dir / 'stage1-trace.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
    make_gif(folder, prompt_dir / 'stage1.gif')
    return tensor, image, duration


@torch.no_grad()
def run_stage2(prompt_case, profile, stage1_tensor, target, profile_dir):
    steps = profile['steps']
    phase = f"stage2_{profile['name']}"
    callback, trace, folder = diffusion_callback(prompt_case['id'], phase, steps)
    initial = paper_stage2_latents(pipe, stage1_tensor, prompt_case['seed'] + 10000, steps)
    started = time.perf_counter()
    result = pipe._run_stage2(
        prompt=prompt_case['text'], qrcode=target, qrcode_module_size=QR_MODULE_SIZE,
        qrcode_padding=QR_INTERNAL_PADDING, ref_image=stage1_tensor,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=steps, guidance_scale=GUIDANCE_SCALE, eta=ETA,
        generator=torch.Generator(device='cuda').manual_seed(prompt_case['seed'] + 10000),
        latents=initial, controlnet_conditioning_scale=CONTROLNET_SCALE,
        scanning_robust_guidance_scale=profile['srg'],
        perceptual_guidance_scale=profile['pg'],
        callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents'],
        output_type='latent',
    )
    final_latent = result.images.detach()
    base_image = decode_latents(pipe, final_latent)
    base_image.save(profile_dir / 'base.png')
    save_tensor(profile_dir / 'stage2-final-latent.safetensors', 'latents', final_latent)
    duration = time.perf_counter() - started
    (profile_dir / 'stage2-time.json').write_text(
        json.dumps({'duration_s': duration}, indent=2), encoding='utf-8'
    )
    (profile_dir / 'stage2-trace.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
    make_gif(folder, profile_dir / 'stage2.gif')
    return final_latent, base_image, duration
"""
    ),
    markdown("## 6. SR-MPGD fidèle : validation et image à chaque itération"),
    code(
        """quality_scorer = CLIPQualityScorer(Path('/cache'), device='cpu')


def evaluate_final(prompt_case, profile, variant, image, duration_s, reference_image, extra):
    validation, records = validation_payload(image)
    try:
        quality = asdict(quality_scorer.score(image, prompt_case['text']))
        quality_error = None
    except Exception as exc:
        quality = {'clip_similarity': None, 'clip_score': None, 'clip_aesthetic': None}
        quality_error = f'{type(exc).__name__}: {exc}'
    context = image_context_features(image, blueprint)
    change = image_change_metrics(image, reference_image)
    row = {
        'prompt_id': prompt_case['id'], 'prompt': prompt_case['text'], 'seed': prompt_case['seed'],
        'profile': profile['name'], 'stage2_steps': profile['steps'], 'variant': variant,
        'duration_s': duration_s, 'module_error_rate': module_error_rate(image, blueprint),
        **validation, **quality, **image_quality_metrics(image), **change, **context, **extra,
        'quality_error': quality_error,
    }
    validation_path = RUN_DIR / prompt_case['id'] / profile['name'] / f'validations-{variant}.json'
    validation_path.write_text(json.dumps(records, indent=2), encoding='utf-8')
    return row


def run_paper_srmpgd(prompt_case, profile, base_latent, profile_dir):
    frames = profile_dir / 'srmpgd-frames'
    frames.mkdir(exist_ok=True)
    validation_by_iteration = {}

    def validate_iteration(image, iteration):
        values, records = validation_payload(image)
        validation_by_iteration[str(iteration)] = records
        return values

    def preview_iteration(image, step):
        image.save(frames / f'{step.iteration:03d}.png')
        clear_output(wait=True)
        display(Markdown(
            f"**{prompt_case['id']} / {profile['name']} / SR-MPGD — état {step.iteration}/{SRMPGD_CONFIG.max_iterations} — "
            f"scan {step.passed}/{step.total}, SRL {step.scanning_robust_loss:.5f}, LPIPS {step.lpips_loss:.5f}, MER {step.actual_module_error_rate:.2%}**"
        ))
        display(image.resize((430, 430)))

    result = run_srmpgd(
        pipe, base_latent, blueprint, SRMPGD_CONFIG,
        scanning_loss=official_srmpgd_srl,
        validation_callback=validate_iteration,
        preview_callback=preview_iteration,
    )
    result.image.save(profile_dir / 'srmpgd-selected.png')
    save_tensor(profile_dir / 'srmpgd-selected-latent.safetensors', 'latents', result.latent)
    (profile_dir / 'srmpgd-trace.json').write_text(
        json.dumps([asdict(step) for step in result.steps], indent=2), encoding='utf-8'
    )
    (profile_dir / 'srmpgd-validations.json').write_text(
        json.dumps(validation_by_iteration, indent=2), encoding='utf-8'
    )
    make_gif(frames, profile_dir / 'srmpgd.gif')
    return result
"""
    ),
    markdown("## 7. Campagne quatre prompts × 40/100 pas × sans/avec SR-MPGD"),
    code(
        """torch.cuda.reset_peak_memory_stats()
for prompt_case in PROMPTS:
    prompt_dir = RUN_DIR / prompt_case['id']
    prompt_dir.mkdir(exist_ok=True)
    stage1_tensor_path = prompt_dir / 'stage1.safetensors'
    stage1_ready = (
        stage1_tensor_path.exists()
        and (prompt_dir / 'stage1.png').exists()
        and (prompt_dir / 'stage1-time.json').exists()
        and trace_artifacts_complete(
            prompt_dir / 'stage1-trace.json',
            RUN_DIR / 'frames' / prompt_case['id'] / 'stage1',
            prompt_dir / 'stage1.gif',
            STAGE1_STEPS,
        )
    )
    force_profile_regeneration = not stage1_ready
    if stage1_ready:
        stage1_tensor = load_tensor(stage1_tensor_path, 'stage1')
        stage1_image = Image.open(prompt_dir / 'stage1.png').convert('RGB')
        stage1_seconds = json.loads(
            (prompt_dir / 'stage1-time.json').read_text(encoding='utf-8')
        )['duration_s']
        print('Reprise Stage 1 :', prompt_case['id'])
    else:
        stale_prompt_keys = {
            key for key in result_index() if key[0] == prompt_case['id']
        }
        if stale_prompt_keys:
            drop_result_keys(stale_prompt_keys)
        stage1_tensor, stage1_image, stage1_seconds = run_stage1(prompt_case, prompt_dir)

    target = qart_proxy(stage1_image)
    target.save(prompt_dir / 'qart-proxy-condition.png')
    target_validation, target_records = validation_payload(target)
    (prompt_dir / 'qart-proxy-validations.json').write_text(json.dumps(target_records, indent=2), encoding='utf-8')
    if target_validation['original_passed'] != target_validation['original_total']:
        raise RuntimeError(f"Le proxy QArt de {prompt_case['id']} ne préserve pas le payload sur l image originale.")

    for profile in STAGE2_PROFILES:
        profile_dir = prompt_dir / profile['name']
        profile_dir.mkdir(exist_ok=True)
        keys = result_index()
        base_key = (prompt_case['id'], profile['name'], 'base_srpg')
        refined_key = (prompt_case['id'], profile['name'], 'faithful_srmpgd')
        stage2_trace_ready = trace_artifacts_complete(
            profile_dir / 'stage2-trace.json',
            RUN_DIR / 'frames' / prompt_case['id'] / f"stage2_{profile['name']}",
            profile_dir / 'stage2.gif',
            profile['steps'],
        )
        base_artifacts_ready = (
            (profile_dir / 'base.png').exists()
            and (profile_dir / 'stage2-final-latent.safetensors').exists()
            and (profile_dir / 'stage2-time.json').exists()
            and (profile_dir / 'validations-base_srpg.json').exists()
            and stage2_trace_ready
        )
        refined_trace_ready = trace_artifacts_complete(
            profile_dir / 'srmpgd-trace.json',
            profile_dir / 'srmpgd-frames',
            profile_dir / 'srmpgd.gif',
        )
        refined_artifacts_ready = (
            (profile_dir / 'srmpgd-selected.png').exists()
            and (profile_dir / 'srmpgd-selected-latent.safetensors').exists()
            and (profile_dir / 'srmpgd-validations.json').exists()
            and (profile_dir / 'validations-faithful_srmpgd.json').exists()
            and refined_trace_ready
        )
        if (
            not force_profile_regeneration
            and base_key in keys
            and refined_key in keys
            and base_artifacts_ready
            and refined_artifacts_ready
        ):
            print('Déjà terminé :', prompt_case['id'], profile['name'])
            continue
        stale_keys = set()
        if force_profile_regeneration or (base_key in keys and not base_artifacts_ready):
            stale_keys.update({base_key, refined_key})
        elif refined_key in keys and not refined_artifacts_ready:
            stale_keys.add(refined_key)
        if refined_key in keys and base_key not in keys:
            stale_keys.add(refined_key)
        if stale_keys:
            drop_result_keys(stale_keys)

        latent_path = profile_dir / 'stage2-final-latent.safetensors'
        if not force_profile_regeneration and stage2_trace_ready and all([
            latent_path.exists(),
            (profile_dir / 'base.png').exists(),
            (profile_dir / 'stage2-time.json').exists(),
        ]):
            base_latent = load_tensor(latent_path, 'latents')
            base_image = Image.open(profile_dir / 'base.png').convert('RGB')
            stage2_seconds = json.loads(
                (profile_dir / 'stage2-time.json').read_text(encoding='utf-8')
            )['duration_s']
            print('Reprise latent Stage 2 :', prompt_case['id'], profile['name'])
        else:
            drop_result_keys({base_key, refined_key})
            base_latent, base_image, stage2_seconds = run_stage2(
                prompt_case, profile, stage1_tensor, target, profile_dir
            )

        if base_key not in result_index():
            base_row = evaluate_final(
                prompt_case, profile, 'base_srpg', base_image,
                stage1_seconds + stage2_seconds,
                stage1_image,
                {
                    'stage1_s': stage1_seconds, 'stage2_s': stage2_seconds,
                    'srmpgd_s': 0.0, 'srmpgd_selected_iteration': None,
                    'srmpgd_stop_reason': None,
                    'stage2_latent_sha256': hashlib.sha256(latent_path.read_bytes()).hexdigest(),
                },
            )
            append_result(base_row)

        # Le module LPIPS de Stage 2 n'est plus utile. Le déplacer avant le LPIPS FP32 de SR-MPGD
        # évite de garder deux VGG sur la RTX 20 Gio.
        if hasattr(pipe, 'srpg'):
            pipe.srpg.to('cpu')
            delattr(pipe, 'srpg')
        gc.collect()
        torch.cuda.empty_cache()

        if refined_key not in result_index():
            srmpgd_result = run_paper_srmpgd(
                prompt_case, profile, base_latent.float(), profile_dir
            )
            refined_row = evaluate_final(
                prompt_case, profile, 'faithful_srmpgd', srmpgd_result.image,
                stage1_seconds + stage2_seconds + srmpgd_result.duration_s,
                base_image,
                {
                    'stage1_s': stage1_seconds, 'stage2_s': stage2_seconds,
                    'srmpgd_s': srmpgd_result.duration_s,
                    'srmpgd_selected_iteration': srmpgd_result.selected_iteration,
                    'srmpgd_stop_reason': srmpgd_result.stop_reason,
                    'srmpgd_step_size': SRMPGD_CONFIG.step_size,
                    'srmpgd_lpips_weight': SRMPGD_CONFIG.lpips_weight,
                    'stage2_latent_sha256': hashlib.sha256(latent_path.read_bytes()).hexdigest(),
                },
            )
            append_result(refined_row)

        cached_lpips = getattr(pipe, '_prooftag_srmpgd_lpips_vgg', None)
        if cached_lpips is not None:
            cached_lpips.to('cpu')
        del base_latent
        gc.collect()
        torch.cuda.empty_cache()

    del stage1_tensor
    gc.collect()
    torch.cuda.empty_cache()

print('Campagne GPU terminée. Lignes persistées :', len(result_rows()))
"""
    ),
    markdown("## 8. Comparaison finale, courbes et porte de livraison"),
    code(
        """rows = list(result_index().values())
assert len(rows) == EXPECTED_RESULTS, f'Campagne incomplète : {len(rows)}/{EXPECTED_RESULTS}'

with (RUN_DIR / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
    writer.writeheader()
    writer.writerows(rows)

columns = [
    ('paper40', 'base_srpg'), ('paper40', 'faithful_srmpgd'),
    ('observed100', 'base_srpg'), ('observed100', 'faithful_srmpgd'),
]
fig, axes = plt.subplots(len(PROMPTS), len(columns), figsize=(16, 16))
for row_index, prompt_case in enumerate(PROMPTS):
    for col_index, (profile_name, variant) in enumerate(columns):
        row = next(item for item in rows if item['prompt_id'] == prompt_case['id'] and item['profile'] == profile_name and item['variant'] == variant)
        filename = 'base.png' if variant == 'base_srpg' else 'srmpgd-selected.png'
        image = Image.open(RUN_DIR / prompt_case['id'] / profile_name / filename).convert('RGB')
        axes[row_index, col_index].imshow(image)
        axes[row_index, col_index].set_title(f"{prompt_case['id']} / {profile_name} / {variant}\\nscan {row['passed']}/{row['total']} | MER {row['module_error_rate']:.2%}\\naes {row['clip_aesthetic']} | clip {row['clip_score']}")
        axes[row_index, col_index].axis('off')
fig.suptitle('E012 — effet causal de 40/100 pas et du SR-MPGD fidèle')
fig.tight_layout()
fig.savefig(RUN_DIR / 'comparison-4x4.png', dpi=160, bbox_inches='tight')
display(fig)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
labels = [f"{row['prompt_id']}\\n{row['profile']}\\n{row['variant']}" for row in rows]
positions = np.arange(len(rows))
for axis, key, title in [
    (axes[0, 0], 'pass_rate', 'Taux de validation QR'),
    (axes[0, 1], 'clip_aesthetic', 'CLIP-aesthetic'),
    (axes[1, 0], 'clip_score', 'CLIPScore'),
    (axes[1, 1], 'duration_s', 'Durée totale (s)'),
]:
    axis.bar(positions, [row[key] if row[key] is not None else 0 for row in rows])
    axis.set_title(title)
    axis.set_xticks(positions, labels, rotation=75, fontsize=7)
    axis.grid(axis='y', alpha=0.25)
axes[0, 0].axhline(1.0, color='red', linestyle='--')
fig.tight_layout()
fig.savefig(RUN_DIR / 'metrics-overview.png', dpi=160, bbox_inches='tight')
display(fig)

selected_by_prompt = {}
for prompt_case in PROMPTS:
    candidates = [row for row in rows if row['prompt_id'] == prompt_case['id']]
    selected = max(candidates, key=rank_key)
    selected_by_prompt[prompt_case['id']] = selected
    source = RUN_DIR / prompt_case['id'] / selected['profile'] / ('base.png' if selected['variant'] == 'base_srpg' else 'srmpgd-selected.png')
    status = 'DELIVERABLE' if selected['strict_all'] else 'BEST_OBSERVED_NOT_DELIVERABLE'
    shutil.copy2(source, RUN_DIR / f"{prompt_case['id']}-{status}.png")
    print(prompt_case['id'], status, selected['profile'], selected['variant'], f"{selected['passed']}/{selected['total']}")
"""
    ),
    markdown("## 9. Manifeste, tests physiques, rapport et archive"),
    code(
        """manifest = {
    'experiment': EXPERIMENT_NAME,
    'created_at': datetime.now(timezone.utc).isoformat(),
    'diffqrcoder_commit': EXPECTED_COMMIT,
    'upstream_hashes': UPSTREAM_HASHES,
    'upstream_patches': UPSTREAM_PATCHES,
    'faithfulness': {
        'stage2_initialization': 'VAE encoding of Stage 1 plus paired Gaussian noise',
        'stage2_perceptual': 'learned LPIPS VGG with differentiable input',
        'stage2_condition': QART_IMPLEMENTATION,
        'srmpgd_initialization': 'exact clean Stage 2 latent, never PNG re-encoding',
        'srmpgd_target': 'original binary QR y, not QArt proxy',
        'srmpgd_srl': SRL_IMPLEMENTATION,
        'srmpgd_objective': 'SRL + 0.01 * LPIPS(decoded latent, detached Stage 2 x0)',
        'srmpgd_step_size': 1000.0,
        'srmpgd_iteration_count': 'not published; states 0..20 evaluated with strict early stop',
    },
    'payload_sha256': payload_hash,
    'prompts': PROMPTS,
    'qr': {
        'version': QR_VERSION, 'error_correction': 'M', 'mask_pattern': QR_MASK_PATTERN,
        'module_size': QR_MODULE_SIZE, 'border': QR_BORDER_MODULES,
        'internal_padding': QR_INTERNAL_PADDING,
    },
    'models': {
        'base': BASE_MODEL_URL, 'controlnet': CONTROLNET_MODEL,
        'controlnet_subfolder': CONTROLNET_SUBFOLDER,
    },
    'diffusion': {
        'stage1_steps': STAGE1_STEPS, 'stage2_profiles': STAGE2_PROFILES,
        'guidance_scale': GUIDANCE_SCALE, 'controlnet_scale': CONTROLNET_SCALE,
        'eta': ETA, 'scheduler': type(pipe.scheduler).__name__,
    },
    'srmpgd': asdict(SRMPGD_CONFIG),
    'software': {
        'torch': torch.__version__, 'diffusers': importlib.metadata.version('diffusers'),
        'transformers': importlib.metadata.version('transformers'),
        'lpips': importlib.metadata.version('lpips'),
    },
    'peak_gpu_memory_gib': torch.cuda.max_memory_allocated() / 2**30,
    'validation_count_per_candidate': rows[0]['total'],
    'results': rows,
    'selected_by_prompt': selected_by_prompt,
}
(RUN_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
(RUN_DIR / 'upstream-patches.json').write_text(json.dumps(UPSTREAM_PATCHES, indent=2, ensure_ascii=False), encoding='utf-8')
shutil.copy2('/workspace/notebooks/09_diffqrcoder_faithful_srmpgd.ipynb', RUN_DIR / '09_diffqrcoder_faithful_srmpgd.ipynb')

with (RUN_DIR / 'physical-validation.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(['prompt_id', 'profile', 'variant', 'device', 'medium', 'distance_cm', 'angle_deg', 'lighting', 'attempt', 'success', 'notes'])
    for prompt_id, selected in selected_by_prompt.items():
        for device in ['Pixel 7', 'iPhone 13', 'autre téléphone']:
            for medium in ['écran', 'impression']:
                for attempt in range(1, 11):
                    writer.writerow([prompt_id, selected['profile'], selected['variant'], device, medium, '', '', '', attempt, '', ''])

strict_count = sum(row['strict_all'] for row in rows)
report_lines = [
    '# Rapport automatique E012', '',
    f'- Résultats complets : {len(rows)}/{EXPECTED_RESULTS}',
    f'- Candidats stricts : {strict_count}/{len(rows)}',
    f'- Pic VRAM : {manifest["peak_gpu_memory_gib"]:.2f} Gio',
    '- SR-MPGD : gamma=1000, LPIPS=0,01, latent Stage 2 exact, QR original.',
    f'- Limite QArt : {QART_IMPLEMENTATION}.', '',
    '## Sélection par prompt', '',
]
for prompt_id, selected in selected_by_prompt.items():
    report_lines.append(
        f"- {prompt_id}: {selected['profile']} / {selected['variant']} — {selected['passed']}/{selected['total']} — "
        f"CLIP-aes={selected['clip_aesthetic']} — CLIPScore={selected['clip_score']}"
    )
(RUN_DIR / 'run-report.md').write_text('\\n'.join(report_lines) + '\\n', encoding='utf-8')

archive_path = Path(shutil.make_archive(str(RUN_DIR), 'gztar', root_dir=RUN_DIR.parent, base_dir=RUN_DIR.name))
archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
print('Archive :', archive_path)
print('SHA-256 :', archive_hash)
print('Serveur Linux :')
print("POD=$(kubectl get pod -n qr-core -l app=prooftag-qr-notebook -o jsonpath='{.items[0].metadata.name}')")
print(f'kubectl cp -n qr-core "${{POD}}:{archive_path}" "$HOME/{archive_path.name}"')
print('PowerShell Windows :')
print(f'scp paul@pcIA:~/{archive_path.name} "$HOME/Downloads/"')
"""
    ),
    markdown(
        """## Interprétation autorisée

- Une amélioration de MER sans amélioration des décodeurs n'est pas un succès.
- Une sortie SR-MPGD non sélectionnée reste dans les frames pour expliquer l'évolution, mais ne remplace jamais automatiquement la base.
- `DELIVERABLE` signifie seulement 26/26 logiciel dans cette campagne. La validation physique doit encore être remplie.
- Les résultats 40 et 100 pas sont appariés par prompt/seed. Le changement de fondation vers SDXL ne sera testé qu'après cette baseline, dans une expérience séparée.
- La présence du proxy QArt interdit de reprendre directement les 99–100 % de l'article comme taux attendu pour Prooftag.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path(__file__).resolve().parents[1] / "notebooks" / "09_diffqrcoder_faithful_srmpgd.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(target)
