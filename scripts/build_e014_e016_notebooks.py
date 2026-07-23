# ruff: noqa: E501
"""Build E014A, E014B, E015 and E016 without hand-editing notebook JSON."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


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


def write_notebook(name: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
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
    target = NOTEBOOKS / name
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(target)


COMMON_PROMPTS = """PROMPTS = [
    {'id': 'p1_simple', 'seed': 1101, 'text': 'A single white lotus flower floating on a dark calm pond, elegant editorial photograph.'},
    {'id': 'p2_medium', 'seed': 2202, 'text': 'A Japanese garden with a red bridge, mossy stones and soft morning mist, detailed photography.'},
    {'id': 'p3_detailed', 'seed': 3303, 'text': 'An ornate botanical tapestry of white lilies, pale blue leaves and dark vines, intricate textile illustration.'},
    {'id': 'p4_complex', 'seed': 4404, 'text': 'A lively old European market square, café terraces, flowers, bicycles and a gothic cathedral, cinematic morning light.'},
]"""


E014A_CELLS = [
    markdown(
        r"""# E014A — vrai QArt, payload strict et blueprint adaptatif

Objectif : isoler **la construction de la condition QR** avant de modifier la diffusion. Pour chacun
des quatre prompts, la même image Stage 1 et le même bruit Stage 2 servent à comparer :

1. `binary_mask4_m` : QR v3/M/masque 4, payload exact ;
2. `qart_fragment_l` : QArt public réel, correction L, fragment `#…`, URL canonique identique ;
3. `exact_payload_mask_search_m` : meilleur des huit masques QR légaux, payload strict ;
4. `adaptive_exact_payload_m` : centres adaptés à la luminance, motifs fonctionnels binaires.

**Précision scientifique importante.** Le QArt public de `andrewyur/qart` ne peut pas être appelé
« exact-payload » : il obtient ses degrés de liberté en ajoutant un fragment. La troisième variante
est donc une recherche de masque standard strict, pas un faux QArt. Chaque sortie est testée par
OpenCV, ZBar et ZXing-cpp si disponibles, sous treize dégradations. Aucune image n'est déclarée
livrable sans porte stricte.
"""
    ),
    markdown(
        r"""## Déroulement

```text
QR binaire fixe ──► DiffQRCoder Stage 1 ──► référence artistique commune
                                               │
             ┌──────────────┬──────────────────┼──────────────────┐
             ▼              ▼                  ▼                  ▼
         binaire M      QArt réel L     8 masques exacts M   adaptatif exact M
             │              │                  │                  │
             └──────── validation des blueprints + géométrie ─────┘
                                               │
                  mêmes latent initial / seed / prompt / paramètres Stage 2
                                               │
                    SSR robuste → CLIP-aes → CLIPScore → grille visible
```

Le premier lancement complet prend du temps : 4 prompts × 4 Stage 2 × 40 pas. Mettre
`PROMPT_LIMIT = 1` pour un test de plomberie, puis revenir à `None` pour l'expérience officielle.
"""
    ),
    code(
        """from __future__ import annotations

import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from diffusers import ControlNetModel, DDIMScheduler
from IPython.display import Markdown, clear_output, display
from PIL import Image
from safetensors.torch import load_file, save_file

UPSTREAM_ROOT = Path('/opt/DiffQRCoder')
DIFFQRCODER_COMMIT = 'e24ea73ee2e13c7e6e87cb422e8b11784e70ae00'
QART_COMMIT = '6e0e00804a1994db7098432c19fadfc552071e30'
if not (UPSTREAM_ROOT / 'diffqrcoder' / 'pipeline_diffqrcoder.py').exists():
    raise RuntimeError('DiffQRCoder absent : reconstruire Dockerfile.notebook.')
if not Path('/usr/local/bin/qart').exists():
    raise RuntimeError('QArt absent : reconstruire Dockerfile.notebook avec le stage qart-builder.')
sys.path.insert(0, str(UPSTREAM_ROOT))

from diffqrcoder import DiffQRCoderPipeline  # noqa: E402
import diffqrcoder.srpg as upstream_srpg  # noqa: E402
from prooftag_qr.blueprints import (  # noqa: E402
    align_qart_output, build_adaptive_blueprint, canonical_url_match,
    exact_mask_candidates, grid_visibility_score, reference_cost,
)
from prooftag_qr.geometry import AlignedQR, aligned_module_diagnostics, generate_aligned_qr  # noqa: E402
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
assert torch.cuda.is_available(), 'Lancer dans le pod GPU, pas avec Python Windows.'
print('GPU :', torch.cuda.get_device_name(0))
print('DiffQRCoder :', DIFFQRCODER_COMMIT)
print('QArt :', QART_COMMIT)
"""
    ),
    markdown("## 1. Contrat expérimental et dossier reprenable"),
    code(
        f"""EXPERIMENT_NAME = 'e014a-real-qart-exact-adaptive-v1'
RESUME_RUN_NAME = None
PAYLOAD = 'https://ptag.io/t/e014'
PROMPT_LIMIT = None  # 1 = smoke test ; None = campagne officielle sur les quatre prompts
RUN_STAGE2 = True
{COMMON_PROMPTS}
ACTIVE_PROMPTS = PROMPTS[:PROMPT_LIMIT] if PROMPT_LIMIT else PROMPTS

QR_VERSION = 3
QR_MODULE_SIZE = 20
CANVAS_SIZE = 768
BASELINE_ECC = 'M'
BASELINE_MASK = 4
NEGATIVE_PROMPT = 'easynegative, unreadable text, letters, watermark'
BASE_MODEL_URL = 'https://huggingface.co/fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned/blob/main/cetusMix_Whalefall2_fp16.safetensors'
CONTROLNET_MODEL = 'monster-labs/control_v1p_sd15_qrcode_monster'
CONTROLNET_SUBFOLDER = 'v2'
STAGE1_STEPS = 40
STAGE2_STEPS = 40
GUIDANCE_SCALE = 7.5
CONTROLNET_SCALE = 1.35
SCANNING_GUIDANCE = 500.0
PERCEPTUAL_GUIDANCE = 3.0
QART_THRESHOLDS = [96, 112, 128, 144, 160]
QART_REPEATS = 3  # le CLI public n'expose pas de seed : quantifier sa variabilité
DISPLAY_EVERY = 5
SAVE_EVERY_STEP = True
BLUEPRINT_NAMES = [
    'binary_mask4_m', 'qart_fragment_l',
    'exact_payload_mask_search_m', 'adaptive_exact_payload_m',
]

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
print('Sortie :', RUN_DIR)
print('Exécutions Stage 2 prévues :', len(ACTIVE_PROMPTS) * 4 if RUN_STAGE2 else 0)
"""
    ),
    markdown("## 2. Fonctions d'audit : validation, reprise, frames et géométrie"),
    code(
        """validator = QRValidator()
print('Décodeurs actifs :', [decoder.name for decoder in validator.decoders])
if len(validator.decoders) < 3:
    print('AVERTISSEMENT : moins de trois décodeurs. Vérifier zbar et zxing-cpp.')


def append_jsonl(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\\n')
        stream.flush()


def existing_keys():
    if not RESULTS_PATH.exists():
        return set()
    return {
        (row['prompt_id'], row['blueprint'])
        for line in RESULTS_PATH.read_text(encoding='utf-8').splitlines() if line.strip()
        for row in [json.loads(line)]
    }


def validation_summary(image, mode='exact'):
    if mode == 'canonical_url':
        records = validator.validate(
            image, PAYLOAD, matcher=canonical_url_match, match_mode='canonical_url_without_fragment'
        )
    else:
        records = validator.validate(image, PAYLOAD)
    summary = summarize_validation_records(records)
    originals = [record for record in records if record.scenario == 'original']
    passed = sum(record.exact_payload_match for record in records)
    original_passed = sum(record.exact_payload_match for record in originals)
    return {
        'passed': passed, 'total': len(records), 'pass_rate': passed / len(records),
        'strict_all': passed == len(records),
        'original_passed': original_passed, 'original_total': len(originals),
        'worst_decoder_pass_rate': summary['worst_decoder_pass_rate'],
        'worst_scenario_pass_rate': summary['worst_scenario_pass_rate'],
    }, [asdict(record) for record in records]


def decode_latents(pipeline, latents):
    with torch.no_grad():
        dtype = next(pipeline.vae.parameters()).dtype
        decoded = pipeline.vae.decode(
            latents.detach().to(dtype=dtype) / pipeline.vae.config.scaling_factor,
            return_dict=False,
        )[0]
    return pipeline.image_processor.postprocess(decoded.detach(), output_type='pil')[0].convert('RGB')


def make_gif(folder, output):
    paths = sorted(folder.glob('*.jpg'))
    if not paths:
        return
    frames = [Image.open(path).convert('RGB').resize((512, 512)) for path in paths]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=150, loop=0)
    for frame in frames:
        frame.close()


def diffusion_callback(pipeline, aligned, label, steps, folder):
    folder.mkdir(parents=True, exist_ok=True)
    trace = []
    started = time.perf_counter()

    def callback(pipe_ref, step_index, timestep, callback_kwargs):
        if SAVE_EVERY_STEP or step_index % DISPLAY_EVERY == 0 or step_index == steps - 1:
            preview = decode_latents(pipeline, callback_kwargs['latents'])
            diagnostics = aligned_module_diagnostics(preview, aligned)
            row = {
                'step': int(step_index), 'timestep': int(timestep),
                'elapsed_s': time.perf_counter() - started, **diagnostics,
            }
            trace.append(row)
            preview.save(folder / f'{{step_index:03d}}.jpg', quality=88)
            if step_index % DISPLAY_EVERY == 0 or step_index == steps - 1:
                clear_output(wait=True)
                display(Markdown(
                    f"**{{label}} — {{step_index + 1}}/{{steps}} — "
                    f"MER {{diagnostics['module_error_rate']:.2%}} — "
                    f"marge {{diagnostics['minimum_threshold_margin']:.3f}}**"
                ))
                display(preview.resize((430, 430)))
        return callback_kwargs

    return callback, trace


@torch.no_grad()
def paired_stage2_latents(pipeline, stage1_tensor, seed, steps):
    normalized = stage1_tensor.to('cuda', dtype=torch.float16) * 2 - 1
    encoded = pipeline.vae.encode(normalized).latent_dist.mode() * pipeline.vae.config.scaling_factor
    generator = torch.Generator(device='cuda').manual_seed(seed)
    noise = torch.randn(encoded.shape, generator=generator, device='cuda', dtype=encoded.dtype)
    pipeline.scheduler.set_timesteps(steps, device='cuda')
    return pipeline.scheduler.add_noise(encoded, noise, pipeline.scheduler.timesteps[:1])
"""
    ),
    markdown("## 3. Charger une seule fois DiffQRCoder et figer tous ses poids"),
    code(
        """controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL, subfolder=CONTROLNET_SUBFOLDER, torch_dtype=torch.float16,
    cache_dir='/cache/huggingface',
)
pipe = DiffQRCoderPipeline.from_single_file(
    BASE_MODEL_URL, controlnet=controlnet, torch_dtype=torch.float16,
    cache_dir='/cache/huggingface', safety_checker=None, use_safetensors=True,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to('cuda')
for component in [pipe.unet, pipe.controlnet, pipe.vae, pipe.text_encoder]:
    component.requires_grad_(False).eval()
pipe.enable_attention_slicing('max')
pipe.enable_vae_slicing()
pipe.unet.enable_gradient_checkpointing()
pipe.controlnet.enable_gradient_checkpointing()
print('Pipeline prête ; VRAM allouée GiB :', torch.cuda.memory_allocated() / 2**30)
"""
    ),
    markdown(
        """## 4. Stage 1 puis construction des quatre blueprints

La condition Stage 1 reste le même QR binaire v3/M/masque 4. QArt reçoit ensuite **l'image Stage 1**
comme image cible. Sa sortie 980×980 contient une bordure de dix modules : elle est recadrée sur le
cœur exact 29×29, jamais redimensionnée, puis centrée dans 768×768. Les cinq seuils QArt sont tous
validés ; le meilleur est retenu par scannabilité canonique puis proximité à la référence.
"""
    ),
    code(
        """def run_qart(reference_path, output_path, threshold):
    command = [
        '/usr/local/bin/qart', 'build', str(QR_VERSION), PAYLOAD,
        str(reference_path), str(output_path), '--module-size', str(QR_MODULE_SIZE),
        '--threshold', str(threshold), '--benchmark',
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return {'command': command, 'stdout': completed.stdout, 'stderr': completed.stderr}


def save_target(prompt_dir, name, aligned, mode, extra):
    target_dir = prompt_dir / 'blueprints' / name
    target_dir.mkdir(parents=True, exist_ok=True)
    aligned.image.save(target_dir / 'condition.png')
    np.save(target_dir / 'matrix.npy', aligned.core_matrix)
    validation, records = validation_summary(aligned.image, mode)
    metadata = {
        'name': name, 'match_mode': mode, 'version': aligned.version,
        'ecc': aligned.error_correction, 'mask_pattern': aligned.mask_pattern,
        'module_size': aligned.module_size, 'padding_px': aligned.padding_px,
        'canvas_size': aligned.canvas_size, 'payload': PAYLOAD,
        'grid_visibility': grid_visibility_score(aligned.image, aligned),
        **validation, **extra,
    }
    (target_dir / 'validations.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    (target_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return {'aligned': aligned, 'metadata': metadata, 'dir': target_dir}


def load_saved_target(prompt_dir, name):
    target_dir = prompt_dir / 'blueprints' / name
    metadata = json.loads((target_dir / 'metadata.json').read_text(encoding='utf-8'))
    matrix = np.load(target_dir / 'matrix.npy').astype(np.uint8)
    image = Image.open(target_dir / 'condition.png').convert('RGB')
    aligned = AlignedQR(
        image=image, core_matrix=matrix, version=metadata['version'],
        error_correction=metadata['ecc'], mask_pattern=metadata['mask_pattern'],
        module_size=metadata['module_size'], padding_px=metadata['padding_px'],
        canvas_size=metadata['canvas_size'], payload=metadata['payload'],
    )
    return {'aligned': aligned, 'metadata': metadata, 'dir': target_dir}


stage1_states = {}
target_states = {}
for prompt_case in ACTIVE_PROMPTS:
    prompt_dir = RUN_DIR / prompt_case['id']
    prompt_dir.mkdir(parents=True, exist_ok=True)
    complete_stage1 = all(
        (prompt_dir / name).exists()
        for name in ['stage1.safetensors', 'stage1-reference.png', 'stage1-trace.json']
    )
    complete_targets = all(
        all((prompt_dir / 'blueprints' / name / artifact).exists() for artifact in [
            'condition.png', 'matrix.npy', 'metadata.json', 'validations.json',
        ])
        for name in BLUEPRINT_NAMES
    )
    if complete_stage1 and complete_targets:
        print('REPRISE artefacts figés :', prompt_case['id'])
        stage1_tensor = load_file(
            str(prompt_dir / 'stage1.safetensors'), device='cpu'
        )['stage1'].to('cuda', dtype=torch.float16)
        stage1_image = Image.open(prompt_dir / 'stage1-reference.png').convert('RGB')
        timing_path = prompt_dir / 'stage1-time.json'
        stage1_duration = (
            json.loads(timing_path.read_text(encoding='utf-8'))['duration_s']
            if timing_path.exists() else 0.0
        )
        stage1_states[prompt_case['id']] = {
            'tensor': stage1_tensor, 'image': stage1_image,
            'duration_s': stage1_duration,
        }
        target_states[prompt_case['id']] = {
            name: load_saved_target(prompt_dir, name) for name in BLUEPRINT_NAMES
        }
        continue
    if any(key[0] == prompt_case['id'] for key in existing_keys()):
        raise RuntimeError(
            f"Résultats Stage 2 présents mais artefacts blueprint incomplets pour "
            f"{prompt_case['id']}; restaurer le dossier au lieu de régénérer QArt."
        )
    baseline = generate_aligned_qr(
        PAYLOAD, version=QR_VERSION, error_correction=BASELINE_ECC,
        mask_pattern=BASELINE_MASK, module_size=QR_MODULE_SIZE, canvas_size=CANVAS_SIZE,
    )
    preflight = []
    for mask_pattern in range(8):
        candidate = generate_aligned_qr(
            PAYLOAD, version=QR_VERSION, error_correction=BASELINE_ECC,
            mask_pattern=mask_pattern, module_size=QR_MODULE_SIZE, canvas_size=CANVAS_SIZE,
        )
        validation, _ = validation_summary(candidate.image, 'exact')
        preflight.append((validation, candidate))
    preflight.sort(
        key=lambda item: (
            item[0]['strict_all'], item[0]['pass_rate'],
            item[0]['worst_decoder_pass_rate'], item[0]['worst_scenario_pass_rate'],
            item[1].mask_pattern == BASELINE_MASK, -item[1].mask_pattern,
        ),
        reverse=True,
    )
    if not preflight[0][0]['strict_all']:
        raise RuntimeError(
            f"Aucun QR témoin v{QR_VERSION}/{BASELINE_ECC} ne passe la porte stricte ; "
            "changer explicitement payload/version/ECC avant toute diffusion."
        )
    stage1_control = preflight[0][1]
    (prompt_dir / 'stage1-control-preflight.json').write_text(
        json.dumps([
            {'mask': item[1].mask_pattern, **item[0]} for item in preflight
        ], indent=2), encoding='utf-8'
    )
    stage1_control.image.save(prompt_dir / 'stage1-control.png')
    stage1_folder = prompt_dir / 'frames-stage1'
    callback, trace = diffusion_callback(
        pipe, stage1_control, f"{{prompt_case['id']}} / Stage 1", STAGE1_STEPS, stage1_folder
    )
    started = time.perf_counter()
    result = pipe._run_stage1(
        prompt=prompt_case['text'], qrcode=stage1_control.image, negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=STAGE1_STEPS, guidance_scale=GUIDANCE_SCALE,
        generator=torch.Generator(device='cuda').manual_seed(prompt_case['seed']),
        controlnet_conditioning_scale=CONTROLNET_SCALE,
        callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents'],
        output_type='pt',
    )
    stage1_tensor = result.images.detach()
    stage1_image = pipe.image_processor.numpy_to_pil(
        pipe.image_processor.pt_to_numpy(stage1_tensor.detach())
    )[0].convert('RGB')
    stage1_image.save(prompt_dir / 'stage1-reference.png')
    save_file({'stage1': stage1_tensor.cpu().contiguous()}, str(prompt_dir / 'stage1.safetensors'))
    (prompt_dir / 'stage1-trace.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
    make_gif(stage1_folder, prompt_dir / 'stage1.gif')
    stage1_duration = time.perf_counter() - started
    (prompt_dir / 'stage1-time.json').write_text(
        json.dumps({'duration_s': stage1_duration}, indent=2), encoding='utf-8'
    )
    stage1_states[prompt_case['id']] = {
        'tensor': stage1_tensor, 'image': stage1_image,
        'duration_s': stage1_duration,
    }

    targets = {}
    targets['binary_mask4_m'] = save_target(
        prompt_dir, 'binary_mask4_m', baseline, 'exact',
        {'algorithm': 'standard QR, fixed mask 4', 'reference_cost': reference_cost(baseline.image, stage1_image)},
    )

    exact_candidates = exact_mask_candidates(
        PAYLOAD, stage1_image, version=QR_VERSION, error_correction=BASELINE_ECC,
        module_size=QR_MODULE_SIZE, canvas_size=CANVAS_SIZE,
    )
    exact_rows = []
    for candidate in exact_candidates:
        validation, _ = validation_summary(candidate.aligned.image, 'exact')
        exact_rows.append((validation, candidate))
    exact_rows.sort(
        key=lambda item: (
            item[0]['strict_all'], item[0]['pass_rate'],
            item[0]['worst_decoder_pass_rate'], item[0]['worst_scenario_pass_rate'],
            -item[1].reference_cost, -item[1].grid_visibility,
        ),
        reverse=True,
    )
    exact_winner = exact_rows[0][1].aligned
    targets['exact_payload_mask_search_m'] = save_target(
        prompt_dir, 'exact_payload_mask_search_m', exact_winner, 'exact',
        {'algorithm': 'best of eight legal QR masks; not QArt',
         'reference_cost': exact_rows[0][1].reference_cost,
         'all_mask_costs': [
             {
                 'mask': item[1].aligned.mask_pattern,
                 'reference_cost': item[1].reference_cost,
                 **item[0],
             }
             for item in exact_rows
         ]},
    )

    adaptive_rows = []
    for minimum_fraction in [0.22, 0.30, 0.38, 0.46, 0.55, 0.70, 0.85]:
        adaptive = build_adaptive_blueprint(
            stage1_image, exact_winner, minimum_data_fraction=minimum_fraction
        )
        validation, _ = validation_summary(adaptive.image, 'exact')
        adaptive_rows.append((validation, adaptive, minimum_fraction))
    adaptive_rows.sort(
        key=lambda item: (
            item[0]['strict_all'], item[0]['pass_rate'],
            -item[1].reference_cost, -item[1].grid_visibility,
        ),
        reverse=True,
    )
    adaptive_validation, adaptive, adaptive_fraction = adaptive_rows[0]
    adaptive_aligned = AlignedQR(
        image=adaptive.image, core_matrix=exact_winner.core_matrix.copy(),
        version=QR_VERSION, error_correction=BASELINE_ECC,
        mask_pattern=exact_winner.mask_pattern, module_size=QR_MODULE_SIZE,
        padding_px=exact_winner.padding_px, canvas_size=CANVAS_SIZE, payload=PAYLOAD,
    )
    np.save(prompt_dir / 'adaptive-center-fractions.npy', adaptive.center_fractions)
    targets['adaptive_exact_payload_m'] = save_target(
        prompt_dir, 'adaptive_exact_payload_m', adaptive_aligned, 'exact',
        {'algorithm': 'Prooftag luminance-adaptive centers; functional patterns binary',
         'minimum_data_fraction': adaptive_fraction,
         'reference_cost': adaptive.reference_cost,
         'screened_minimum_fractions': [row[2] for row in adaptive_rows]},
    )

    qart_rows = []
    qart_errors = []
    for threshold in QART_THRESHOLDS:
        for repeat in range(QART_REPEATS):
            raw_path = prompt_dir / f'qart-raw-threshold-{{threshold}}-repeat-{{repeat}}.png'
            try:
                provenance = run_qart(prompt_dir / 'stage1-reference.png', raw_path, threshold)
                aligned_qart = align_qart_output(
                    Image.open(raw_path), payload=PAYLOAD, version=QR_VERSION,
                    module_size=QR_MODULE_SIZE, canvas_size=CANVAS_SIZE,
                )
                validation, _ = validation_summary(aligned_qart.image, 'canonical_url')
                qart_rows.append({
                    'validation': validation, 'aligned': aligned_qart,
                    'threshold': threshold, 'repeat': repeat,
                    'reference_cost': reference_cost(aligned_qart.image, stage1_image),
                    'provenance': provenance,
                    'sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                })
            except Exception as exc:
                qart_errors.append({
                    'threshold': threshold, 'repeat': repeat,
                    'error': f'{{type(exc).__name__}}: {{exc}}',
                })
    (prompt_dir / 'qart-errors.json').write_text(
        json.dumps(qart_errors, indent=2), encoding='utf-8'
    )
    if not qart_rows:
        raise RuntimeError('Tous les essais QArt ont échoué ; voir qart-errors.json.')
    qart_rows.sort(
        key=lambda row: (
            row['validation']['strict_all'], row['validation']['pass_rate'],
            -row['reference_cost'],
        ),
        reverse=True,
    )
    qart_winner = qart_rows[0]
    targets['qart_fragment_l'] = save_target(
        prompt_dir, 'qart_fragment_l', qart_winner['aligned'], 'canonical_url',
        {'algorithm': 'andrewyur/qart real Reed-Solomon degrees of freedom',
         'exact_payload': False, 'canonical_url_only': True,
         'threshold': qart_winner['threshold'], 'repeat': qart_winner['repeat'],
         'reference_cost': qart_winner['reference_cost'],
         'raw_sha256': qart_winner['sha256'], 'provenance': qart_winner['provenance']},
    )
    target_states[prompt_case['id']] = targets

    blueprint_rows = [state['metadata'] for state in targets.values()]
    pd.DataFrame(blueprint_rows).to_csv(prompt_dir / 'blueprint-comparison.csv', index=False)
    display(pd.DataFrame(blueprint_rows)[
        ['name', 'ecc', 'match_mode', 'passed', 'total', 'reference_cost', 'grid_visibility']
    ])
"""
    ),
    markdown(
        """## 5. Comparaison Stage 2 appariée

Les quatre branches réutilisent le même tensor Stage 1, le même latent initial DDIM, la même seed,
le même prompt et les mêmes poids. Seule l'image de condition QR change. Une frame décodée et ses
diagnostics sont écrits à chaque pas. Le QArt est évalué en URL canonique ; les trois autres en
égalité stricte du payload.
"""
    ),
    code(
        """quality_scorer = CLIPQualityScorer(Path('/cache'), device='cpu')


def rank_row(row):
    return (
        row['strict_all'], row['pass_rate'], row['worst_decoder_pass_rate'],
        row['worst_scenario_pass_rate'], row.get('clip_aesthetic') or -999,
        row.get('clip_score') or -999, -row['grid_visibility'],
    )


if RUN_STAGE2:
    for prompt_case in ACTIVE_PROMPTS:
        prompt_id = prompt_case['id']
        stage1 = stage1_states[prompt_id]
        for blueprint_name, state in target_states[prompt_id].items():
            if (prompt_id, blueprint_name) in existing_keys():
                print('SKIP', prompt_id, blueprint_name)
                continue
            aligned = state['aligned']
            output_dir = RUN_DIR / prompt_id / 'stage2' / blueprint_name
            output_dir.mkdir(parents=True, exist_ok=True)
            callback, trace = diffusion_callback(
                pipe, aligned, f"{{prompt_id}} / {{blueprint_name}}",
                STAGE2_STEPS, output_dir / 'frames',
            )
            initial = paired_stage2_latents(
                pipe, stage1['tensor'], prompt_case['seed'] + 10000, STAGE2_STEPS
            )
            started = time.perf_counter()
            result = pipe._run_stage2(
                prompt=prompt_case['text'], qrcode=aligned.image,
                qrcode_module_size=aligned.module_size, qrcode_padding=aligned.padding_px,
                ref_image=stage1['tensor'], negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=STAGE2_STEPS, guidance_scale=GUIDANCE_SCALE, eta=0.0,
                generator=torch.Generator(device='cuda').manual_seed(prompt_case['seed'] + 10000),
                latents=initial.clone(), controlnet_conditioning_scale=CONTROLNET_SCALE,
                scanning_robust_guidance_scale=SCANNING_GUIDANCE,
                perceptual_guidance_scale=PERCEPTUAL_GUIDANCE,
                callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents'],
                output_type='latent',
            )
            final_latent = result.images.detach()
            image = decode_latents(pipe, final_latent)
            duration = time.perf_counter() - started
            image.save(output_dir / 'final.png')
            save_file(
                {'latents': final_latent.cpu().contiguous()},
                str(output_dir / 'final-latent.safetensors'),
            )
            (output_dir / 'trace.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
            make_gif(output_dir / 'frames', output_dir / 'diffusion.gif')
            mode = state['metadata']['match_mode']
            validation, records = validation_summary(image, mode)
            (output_dir / 'validations.json').write_text(
                json.dumps(records, indent=2), encoding='utf-8'
            )
            try:
                quality = asdict(quality_scorer.score(image, prompt_case['text']))
                quality_error = None
            except Exception as exc:
                quality = {'clip_similarity': None, 'clip_score': None, 'clip_aesthetic': None}
                quality_error = f'{{type(exc).__name__}}: {{exc}}'
            row = {
                'prompt_id': prompt_id, 'prompt': prompt_case['text'],
                'seed': prompt_case['seed'], 'blueprint': blueprint_name,
                'payload_contract': mode, 'stage1_duration_s': stage1['duration_s'],
                'stage2_duration_s': duration,
                **validation, **quality, **aligned_module_diagnostics(image, aligned),
                'grid_visibility': grid_visibility_score(image, aligned),
                'blueprint_reference_cost': state['metadata']['reference_cost'],
                'quality_error': quality_error,
            }
            append_jsonl(RESULTS_PATH, row)
            print(prompt_id, blueprint_name, validation['passed'], '/', validation['total'])
            del result, final_latent, initial
            gc.collect()
            torch.cuda.empty_cache()
"""
    ),
    markdown("## 6. Décision, manifeste et archive"),
    code(
        """rows = [
    json.loads(line) for line in RESULTS_PATH.read_text(encoding='utf-8').splitlines()
    if line.strip()
] if RESULTS_PATH.exists() else []
frame = pd.DataFrame(rows)
if not frame.empty:
    frame.to_csv(RUN_DIR / 'comparison.csv', index=False)
    display(frame.sort_values(
        ['strict_all', 'pass_rate', 'clip_aesthetic', 'clip_score'],
        ascending=[False, False, False, False],
    )[
        ['prompt_id', 'blueprint', 'passed', 'total', 'clip_aesthetic',
         'clip_score', 'grid_visibility', 'stage2_duration_s']
    ])

    aggregate = frame.groupby('blueprint').agg(
        contexts=('prompt_id', 'nunique'),
        strict_contexts=('strict_all', 'sum'),
        mean_ssr=('pass_rate', 'mean'),
        worst_ssr=('pass_rate', 'min'),
        mean_aesthetic=('clip_aesthetic', 'mean'),
        mean_clip=('clip_score', 'mean'),
        mean_grid_visibility=('grid_visibility', 'mean'),
        mean_seconds=('stage2_duration_s', 'mean'),
    ).reset_index()
    aggregate.to_csv(RUN_DIR / 'aggregate.csv', index=False)
    display(aggregate.sort_values(
        ['strict_contexts', 'worst_ssr', 'mean_ssr', 'mean_aesthetic'],
        ascending=False,
    ))

    for prompt_case in ACTIVE_PROMPTS:
        prompt_rows = [row for row in rows if row['prompt_id'] == prompt_case['id']]
        exact_rows = [row for row in prompt_rows if row['payload_contract'] == 'exact']
        winner = max(exact_rows, key=rank_row)
        source = RUN_DIR / prompt_case['id'] / 'blueprints' / winner['blueprint']
        destination = RUN_DIR / prompt_case['id']
        shutil.copy2(source / 'condition.png', destination / 'selected-blueprint.png')
        shutil.copy2(source / 'matrix.npy', destination / 'selected-matrix.npy')
        selected = {
            'prompt_id': prompt_case['id'], 'prompt': prompt_case['text'],
            'payload': PAYLOAD, 'selected_blueprint': winner['blueprint'],
            'selection_rule': 'exact payload only; strict SSR then weak links then aesthetics',
            'module_size': QR_MODULE_SIZE, 'padding_px': target_states[prompt_case['id']][winner['blueprint']]['aligned'].padding_px,
            'canvas_size': CANVAS_SIZE, 'version': QR_VERSION,
            'stage1_tensor': str(destination / 'stage1.safetensors'),
            'blueprint_path': str(destination / 'selected-blueprint.png'),
            'matrix_path': str(destination / 'selected-matrix.npy'),
        }
        (destination / 'selected-meta.json').write_text(
            json.dumps(selected, indent=2), encoding='utf-8'
        )

manifest = {
    'experiment': EXPERIMENT_NAME, 'created_at': datetime.now(timezone.utc).isoformat(),
    'diffqrcoder_commit': DIFFQRCODER_COMMIT, 'qart_commit': QART_COMMIT,
    'payload': PAYLOAD, 'prompts': ACTIVE_PROMPTS, 'canvas_size': CANVAS_SIZE,
    'module_size': QR_MODULE_SIZE, 'stage1_steps': STAGE1_STEPS,
    'stage2_steps': STAGE2_STEPS, 'decoders': [decoder.name for decoder in validator.decoders],
    'scientific_limits': [
        'QArt public appends a fragment and uses ECC L; it is canonical-URL, not exact-payload.',
        'QArt public exposes no deterministic seed; repeated outputs and SHA-256 are persisted.',
        'Exact-payload mask search uses standard QR masks and is not labelled QArt.',
        'Adaptive blueprint is a Prooftag engineering method, not a paper reproduction.',
        'Software validation is not physical SSR.',
    ],
}
(RUN_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
archive = shutil.make_archive(
    str(RUN_DIR), 'gztar', root_dir=RUN_DIR.parent, base_dir=RUN_DIR.name
)
print('Archive :', archive)
"""
    ),
]


E014B_CELLS = [
    markdown(
        r"""# E014B — fusion latente FreeQR, canal et timestep contrôlés

Ce notebook part du **blueprint exact-payload sélectionné par E014A**. Il ne prétend pas exécuter
un dépôt FreeQR officiel : il reconstruit et journalise le mécanisme publié de fusion d'une
représentation QR dans un canal latent, puis ajoute séparément une petite loss différentiable de
lecture. L'ablation est factorisée afin de savoir *ce qui* aide :

1. baseline sans fusion ;
2. canal latent 0, 1, 2 ou 3 ;
3. fenêtre temporelle early/middle/late/all ;
4. coefficient de fusion ;
5. meilleur réglage avec ou sans gradient de lecture.

Chaque exécution réutilise le même Stage 1, le même bruit initial et les mêmes paramètres
DiffQRCoder. Les frames montrent l'état latent décodé après chaque pas.
"""
    ),
    markdown(
        r"""## Chaîne expérimentale

```text
latent Stage 2 z_t ─────────────────────────────────────────────────► z_0
       │              à chaque timestep sélectionné
       ├── canal c ← (1-α) canal c + α·QR_latent_bruité(t suivant)
       │
       └── option : gradient d'une loss de marge par module, canal c seulement

phase 1 : canal  →  phase 2 : fenêtre  →  phase 3 : α  →  phase 4 : gradient
             sélection scannabilité stricte avant toute esthétique
```

La fusion au callback a lieu **après** le pas DDIM ; la cible est donc bruitée au timestep suivant.
Ce choix est écrit dans chaque trace pour éviter l'erreur d'alignement d'un timestep.
"""
    ),
    code(
        """from __future__ import annotations

import gc
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
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import ControlNetModel, DDIMScheduler
from IPython.display import Markdown, clear_output, display
from PIL import Image
from safetensors.torch import load_file, save_file

UPSTREAM_ROOT = Path('/opt/DiffQRCoder')
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
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
"""
    ),
    markdown("## 1. Reprendre un résultat E014A sans deviner sa géométrie"),
    code(
        """EXPERIMENT_NAME = 'e014b-freeqr-latent-channel-timestep-v1'
E014A_RUN_DIR = None  # Path('/data/notebook-runs/...-e014a-real-qart-exact-adaptive-v1')
PROMPT_ID = 'p1_simple'  # répéter ensuite p2/p3/p4 pour la confirmation
RESUME_RUN_NAME = None

if E014A_RUN_DIR is None:
    candidates = sorted(Path('/data/notebook-runs').glob('*-e014a-real-qart-exact-adaptive-v1'))
    if not candidates:
        raise FileNotFoundError('Aucun E014A : exécuter le notebook 11 avant E014B.')
    E014A_RUN_DIR = candidates[-1]
else:
    E014A_RUN_DIR = Path(E014A_RUN_DIR)
source_dir = E014A_RUN_DIR / PROMPT_ID
meta = json.loads((source_dir / 'selected-meta.json').read_text(encoding='utf-8'))
prompt = meta['prompt']
payload = meta['payload']
blueprint_image = Image.open(source_dir / 'selected-blueprint.png').convert('RGB')
matrix = np.load(source_dir / 'selected-matrix.npy').astype(np.uint8)
aligned = AlignedQR(
    image=blueprint_image, core_matrix=matrix, version=meta['version'],
    error_correction='M', mask_pattern=-1, module_size=meta['module_size'],
    padding_px=meta['padding_px'], canvas_size=meta['canvas_size'], payload=payload,
)
stage1_tensor = load_file(str(source_dir / 'stage1.safetensors'), device='cpu')['stage1']
stage1_image = Image.open(source_dir / 'stage1-reference.png').convert('RGB')

if RESUME_RUN_NAME:
    RUN_DIR = Path('/data/notebook-runs') / RESUME_RUN_NAME
else:
    RUN_DIR = Path('/data/notebook-runs') / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + EXPERIMENT_NAME
    )
    RUN_DIR.mkdir(parents=True)
RESULTS_PATH = RUN_DIR / 'results.jsonl'
print('Source :', E014A_RUN_DIR)
print('Blueprint exact :', meta['selected_blueprint'])
print('Sortie :', RUN_DIR)
display(blueprint_image.resize((430, 430)))
"""
    ),
    markdown("## 2. Paramètres fixes et plan d'ablation séquentiel"),
    code(
        """BASE_MODEL_URL = 'https://huggingface.co/fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned/blob/main/cetusMix_Whalefall2_fp16.safetensors'
CONTROLNET_MODEL = 'monster-labs/control_v1p_sd15_qrcode_monster'
CONTROLNET_SUBFOLDER = 'v2'
STEPS = 40
GUIDANCE_SCALE = 7.5
CONTROLNET_SCALE = 1.35
SCANNING_GUIDANCE = 500.0
PERCEPTUAL_GUIDANCE = 3.0
NEGATIVE_PROMPT = 'easynegative, unreadable text, letters, watermark'
SEED = next(item['seed'] for item in json.loads(
    (E014A_RUN_DIR / 'manifest.json').read_text(encoding='utf-8')
)['prompts'] if item['id'] == PROMPT_ID)
DISPLAY_EVERY = 5
SAVE_EVERY_STEP = True

BASE_CONFIG = {
    'channel': None, 'alpha': 0.0, 'window': [0.0, 1.0],
    'scan_gradient': False, 'scan_lr': 0.0, 'scan_every': 4,
}
CHANNEL_ALPHA = 0.15
WINDOWS = {
    'early': [0.00, 0.35], 'middle': [0.30, 0.70],
    'late': [0.65, 1.00], 'all': [0.00, 1.00],
}
ALPHAS = [0.05, 0.10, 0.15, 0.22]
"""
    ),
    markdown("## 3. Pipeline, latent QR propre et même bruit pour toutes les branches"),
    code(
        """controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL, subfolder=CONTROLNET_SUBFOLDER, torch_dtype=torch.float16,
    cache_dir='/cache/huggingface',
)
pipe = DiffQRCoderPipeline.from_single_file(
    BASE_MODEL_URL, controlnet=controlnet, torch_dtype=torch.float16,
    cache_dir='/cache/huggingface', safety_checker=None, use_safetensors=True,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to('cuda')
for component in [pipe.unet, pipe.controlnet, pipe.vae, pipe.text_encoder]:
    component.requires_grad_(False).eval()
pipe.enable_attention_slicing('max')
pipe.enable_vae_slicing()
stage1_tensor = stage1_tensor.to('cuda', dtype=torch.float16)


@torch.no_grad()
def encode_image(image):
    array = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to('cuda', torch.float16)
    return pipe.vae.encode(tensor * 2 - 1).latent_dist.mode() * pipe.vae.config.scaling_factor


@torch.no_grad()
def decode_latent(latent):
    decoded = pipe.vae.decode(
        latent.detach().to(dtype=next(pipe.vae.parameters()).dtype) / pipe.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    return pipe.image_processor.postprocess(decoded.detach(), output_type='pil')[0].convert('RGB')


@torch.no_grad()
def initial_latent():
    encoded_reference = pipe.vae.encode(
        stage1_tensor * 2 - 1
    ).latent_dist.mode() * pipe.vae.config.scaling_factor
    generator = torch.Generator(device='cuda').manual_seed(SEED + 10000)
    noise = torch.randn(
        encoded_reference.shape, generator=generator, device='cuda',
        dtype=encoded_reference.dtype,
    )
    pipe.scheduler.set_timesteps(STEPS, device='cuda')
    return pipe.scheduler.add_noise(
        encoded_reference, noise, pipe.scheduler.timesteps[:1]
    ), noise


blueprint_latent = encode_image(blueprint_image)
paired_initial, paired_noise = initial_latent()
print('Latent image :', tuple(paired_initial.shape), 'latent blueprint :', tuple(blueprint_latent.shape))
assert paired_initial.shape[1] == 4
"""
    ),
    markdown(
        """## 4. Callback de fusion et loss différentiable

La loss n'est **pas** le SRL officiel de DiffQRCoder : c'est une marge centrale simple, utilisée
uniquement pour tester l'interaction avec la fusion FreeQR. Son effet est séparé dans la phase 4.
Le latent rendu au pipeline est toujours détaché, ce qui évite l'erreur NumPy sur tensor avec grad.
"""
    ),
    code(
        """target_dark = torch.as_tensor(
    aligned.core_matrix.astype(bool), device='cuda'
).unsqueeze(0).unsqueeze(0)


def differentiable_module_loss(latent):
    dtype = next(pipe.vae.parameters()).dtype
    decoded = pipe.vae.decode(
        latent.to(dtype=dtype) / pipe.vae.config.scaling_factor, return_dict=False
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


def callback_for(config, output_dir):
    frames = output_dir / 'frames'
    frames.mkdir(parents=True, exist_ok=True)
    trace = []
    started = time.perf_counter()

    def callback(pipeline, step_index, timestep, callback_kwargs):
        latent = callback_kwargs['latents'].detach()
        next_timestep = (
            pipeline.scheduler.timesteps[step_index + 1]
            if step_index + 1 < len(pipeline.scheduler.timesteps)
            else torch.tensor(
                0, device=latent.device, dtype=pipeline.scheduler.timesteps.dtype
            )
        )
        fusion_applied = config['channel'] is not None and fusion_active(step_index, config['window'])
        scan_loss_value = None
        if fusion_applied:
            with torch.no_grad():
                noised_qr = pipeline.scheduler.add_noise(
                    blueprint_latent, paired_noise, next_timestep.reshape(1)
                )
                channel = config['channel']
                latent[:, channel:channel + 1] = (
                    (1 - config['alpha']) * latent[:, channel:channel + 1]
                    + config['alpha'] * noised_qr[:, channel:channel + 1]
                )
        if (
            config['scan_gradient'] and fusion_applied
            and step_index % config['scan_every'] == 0
        ):
            with torch.enable_grad():
                working = latent.detach().float().requires_grad_(True)
                loss = differentiable_module_loss(working)
                gradient = torch.autograd.grad(loss, working)[0]
                channel = config['channel']
                channel_gradient = gradient[:, channel:channel + 1]
                rms = channel_gradient.square().mean().sqrt().clamp_min(1e-8)
                updated = working.detach()
                updated[:, channel:channel + 1] -= (
                    config['scan_lr'] * channel_gradient / rms
                )
                latent = updated.to(dtype=callback_kwargs['latents'].dtype).detach()
                scan_loss_value = float(loss.detach().cpu())
        preview = decode_latent(latent)
        diagnostics = aligned_module_diagnostics(preview, aligned)
        row = {
            'step': int(step_index), 'timestep_before_step': int(timestep),
            'target_timestep_after_step': int(next_timestep),
            'fusion_applied': fusion_applied, 'scan_loss': scan_loss_value,
            'elapsed_s': time.perf_counter() - started, **diagnostics,
        }
        trace.append(row)
        if SAVE_EVERY_STEP or step_index % DISPLAY_EVERY == 0 or step_index == STEPS - 1:
            preview.save(frames / f'{step_index:03d}.jpg', quality=88)
        if step_index % DISPLAY_EVERY == 0 or step_index == STEPS - 1:
            clear_output(wait=True)
            display(Markdown(
                f"**{config['name']} — {step_index + 1}/{STEPS} — "
                f"fusion={fusion_applied} — MER={diagnostics['module_error_rate']:.2%}**"
            ))
            display(preview.resize((430, 430)))
        callback_kwargs['latents'] = latent.detach()
        return callback_kwargs

    return callback, trace
"""
    ),
    markdown("## 5. Exécuteur apparié, validation et persistance immédiate"),
    code(
        """validator = QRValidator()
quality_scorer = CLIPQualityScorer(Path('/cache'), device='cpu')


def validation_summary(image):
    records = validator.validate(image, payload)
    summary = summarize_validation_records(records)
    passed = sum(item.exact_payload_match for item in records)
    return {
        'passed': passed, 'total': len(records), 'pass_rate': passed / len(records),
        'strict_all': passed == len(records),
        'worst_decoder_pass_rate': summary['worst_decoder_pass_rate'],
        'worst_scenario_pass_rate': summary['worst_scenario_pass_rate'],
    }, [asdict(item) for item in records]


def append_row(row):
    with RESULTS_PATH.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\\n')
        stream.flush()


def completed_names():
    if not RESULTS_PATH.exists():
        return set()
    names = {
        json.loads(line)['name']
        for line in RESULTS_PATH.read_text(encoding='utf-8').splitlines() if line.strip()
    }
    for name in names:
        required = [
            RUN_DIR / name / 'final.png', RUN_DIR / name / 'final.safetensors',
            RUN_DIR / name / 'trace.json', RUN_DIR / name / 'validations.json',
            RUN_DIR / name / 'diffusion.gif',
        ]
        if not all(path.exists() for path in required):
            raise RuntimeError(
                f'Résultat {name} indexé mais artefacts incomplets ; '
                'restaurer le dossier avant la reprise.'
            )
    return names


def make_gif(folder, output):
    paths = sorted(folder.glob('*.jpg'))
    frames = [Image.open(path).convert('RGB').resize((512, 512)) for path in paths]
    if frames:
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=150, loop=0)
    for frame in frames:
        frame.close()


def run_config(config):
    if config['name'] in completed_names():
        print('SKIP', config['name'])
        return
    output_dir = RUN_DIR / config['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    callback, trace = callback_for(config, output_dir)
    started = time.perf_counter()
    result = pipe._run_stage2(
        prompt=prompt, qrcode=blueprint_image,
        qrcode_module_size=aligned.module_size, qrcode_padding=aligned.padding_px,
        ref_image=stage1_tensor, negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=STEPS, guidance_scale=GUIDANCE_SCALE, eta=0.0,
        generator=torch.Generator(device='cuda').manual_seed(SEED + 10000),
        latents=paired_initial.clone(), controlnet_conditioning_scale=CONTROLNET_SCALE,
        scanning_robust_guidance_scale=SCANNING_GUIDANCE,
        perceptual_guidance_scale=PERCEPTUAL_GUIDANCE,
        callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents'],
        output_type='latent',
    )
    final_latent = result.images.detach()
    image = decode_latent(final_latent)
    duration = time.perf_counter() - started
    image.save(output_dir / 'final.png')
    save_file({'latents': final_latent.cpu().contiguous()}, str(output_dir / 'final.safetensors'))
    (output_dir / 'trace.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
    make_gif(output_dir / 'frames', output_dir / 'diffusion.gif')
    validation, records = validation_summary(image)
    (output_dir / 'validations.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    try:
        quality = asdict(quality_scorer.score(image, prompt))
        quality_error = None
    except Exception as exc:
        quality = {'clip_similarity': None, 'clip_score': None, 'clip_aesthetic': None}
        quality_error = f'{type(exc).__name__}: {exc}'
    row = {
        'name': config['name'], 'prompt_id': PROMPT_ID, 'prompt': prompt,
        'duration_s': duration, 'config': config, **validation, **quality,
        **aligned_module_diagnostics(image, aligned), 'quality_error': quality_error,
    }
    append_row(row)
    print(config['name'], validation['passed'], '/', validation['total'])
    del result, final_latent
    gc.collect()
    torch.cuda.empty_cache()


def rows():
    return [
        json.loads(line) for line in RESULTS_PATH.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ] if RESULTS_PATH.exists() else []


def rank(row):
    return (
        row['strict_all'], row['pass_rate'], row['worst_decoder_pass_rate'],
        row['worst_scenario_pass_rate'], row.get('clip_aesthetic') or -999,
        row.get('clip_score') or -999, -row['module_error_rate'],
    )
"""
    ),
    markdown("## 6. Phase 1 — trouver le canal"),
    code(
        """run_config({'name': 'baseline_no_fusion', **BASE_CONFIG})
for channel in range(4):
    run_config({
        'name': f'channel_{channel}', **BASE_CONFIG,
        'channel': channel, 'alpha': CHANNEL_ALPHA, 'window': WINDOWS['all'],
    })
phase1 = [row for row in rows() if row['name'].startswith('channel_')]
best_channel_row = max(phase1, key=rank)
BEST_CHANNEL = best_channel_row['config']['channel']
print('Canal promu :', BEST_CHANNEL, best_channel_row['passed'], '/', best_channel_row['total'])
display(pd.DataFrame(phase1)[
    ['name', 'passed', 'total', 'module_error_rate', 'clip_aesthetic', 'clip_score']
])
"""
    ),
    markdown("## 7. Phase 2 — trouver la fenêtre temporelle"),
    code(
        """for window_name, window in WINDOWS.items():
    run_config({
        'name': f'window_{window_name}', **BASE_CONFIG,
        'channel': BEST_CHANNEL, 'alpha': CHANNEL_ALPHA, 'window': window,
    })
phase2 = [row for row in rows() if row['name'].startswith('window_')]
best_window_row = max(phase2, key=rank)
BEST_WINDOW = best_window_row['config']['window']
print('Fenêtre promue :', BEST_WINDOW)
display(pd.DataFrame(phase2)[
    ['name', 'passed', 'total', 'module_error_rate', 'clip_aesthetic', 'clip_score']
])
"""
    ),
    markdown("## 8. Phase 3 — trouver la force de fusion"),
    code(
        """for alpha in ALPHAS:
    run_config({
        'name': f'alpha_{alpha:.2f}', **BASE_CONFIG,
        'channel': BEST_CHANNEL, 'alpha': alpha, 'window': BEST_WINDOW,
    })
phase3 = [row for row in rows() if row['name'].startswith('alpha_')]
best_alpha_row = max(phase3, key=rank)
BEST_ALPHA = best_alpha_row['config']['alpha']
print('Alpha promu :', BEST_ALPHA)
display(pd.DataFrame(phase3)[
    ['name', 'passed', 'total', 'module_error_rate', 'clip_aesthetic', 'clip_score']
])
"""
    ),
    markdown("## 9. Phase 4 — isoler l'apport du gradient de lecture"),
    code(
        """for scan_lr in [0.01, 0.03, 0.06]:
    run_config({
        'name': f'best_gradient_{scan_lr:.2f}', **BASE_CONFIG,
        'channel': BEST_CHANNEL, 'alpha': BEST_ALPHA, 'window': BEST_WINDOW,
        'scan_gradient': True, 'scan_lr': scan_lr, 'scan_every': 4,
    })
all_rows = rows()
winner = max(all_rows, key=rank)
pd.DataFrame(all_rows).to_csv(RUN_DIR / 'comparison.csv', index=False)
display(pd.DataFrame(all_rows).sort_values(
    ['strict_all', 'pass_rate', 'clip_aesthetic'], ascending=False
)[['name', 'passed', 'total', 'clip_aesthetic', 'clip_score', 'module_error_rate', 'duration_s']])
print('Gagnant observé :', winner['name'])
if not winner['strict_all']:
    print('NON LIVRABLE : aucune configuration n a franchi tous les tests logiciels.')
"""
    ),
    markdown("## 10. Courbes, manifeste et archive"),
    code(
        """winner_trace = json.loads(
    (RUN_DIR / winner['name'] / 'trace.json').read_text(encoding='utf-8')
)
figure, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot([row['step'] for row in winner_trace], [row['module_error_rate'] for row in winner_trace])
axes[0].set(title='MER par pas', xlabel='pas', ylabel='MER')
axes[1].plot(
    [row['step'] for row in winner_trace],
    [np.nan if row['scan_loss'] is None else row['scan_loss'] for row in winner_trace],
)
axes[1].set(title='Loss différentiable (seulement aux pas actifs)', xlabel='pas', ylabel='loss')
for axis in axes:
    axis.grid(alpha=0.25)
figure.tight_layout()
figure.savefig(RUN_DIR / 'winner-trace.png', dpi=160)
display(figure)

manifest = {
    'experiment': EXPERIMENT_NAME, 'source_e014a': str(E014A_RUN_DIR),
    'prompt_id': PROMPT_ID, 'seed': SEED, 'payload': payload,
    'selected_e014a_blueprint': meta['selected_blueprint'],
    'steps': STEPS, 'winner': winner['name'], 'winner_strict': winner['strict_all'],
    'best_channel': BEST_CHANNEL, 'best_window': BEST_WINDOW, 'best_alpha': BEST_ALPHA,
    'timestep_alignment': 'callback after scheduler step uses next scheduler timestep',
    'claim': 'FreeQR-inspired channel/timestep reconstruction, not an official FreeQR code release',
    'gradient_claim': 'Prooftag central-module margin loss, not DiffQRCoder SR-MPGD',
}
(RUN_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
archive = shutil.make_archive(str(RUN_DIR), 'gztar', RUN_DIR.parent, RUN_DIR.name)
print('Archive :', archive)
"""
    ),
]


E015_CELLS = [
    markdown(
        r"""# E015 — SD 1.5, SDXL et FLUX comme références esthétiques seulement

Cette expérience répond à une question limitée : **quel backbone produit la meilleure image de
référence pour nos quatre prompts ?** Elle ne compare ni ControlNet QR, ni SRPG, ni SR-MPGD, car
ceux-ci ne sont pas compatibles de façon identique avec les trois familles.

- SD 1.5 : Cetus-Mix Whalefall, utilisé par DiffQRCoder ;
- SDXL : `stabilityai/stable-diffusion-xl-base-1.0` ;
- FLUX : `black-forest-labs/FLUX.1-schnell`.

Après chaque référence, le même constructeur adaptatif exact-payload d'E014A est appliqué pour
mesurer son **potentiel d'intégration**, sans relancer une diffusion QR. Les modèles sont chargés
l'un après l'autre sur la RTX 4000 Ada 20 Go et entièrement libérés entre deux familles.
"""
    ),
    markdown(
        r"""## Ce que le tableau final permettra — et ne permettra pas — de conclure

```text
même prompt + seed logique
       ├── SD1.5 / 30 pas / 768
       ├── SDXL  / 30 pas / 768
       └── FLUX  /  4 pas / 768
                │
                ├── temps, pic VRAM, CLIP-aes, CLIPScore
                └── exact mask + adaptatif → SSR du blueprint (pas d'image QR finale)
```

Le nombre de pas recommandé diffère par architecture : les temps représentent une recette
opérationnelle, pas un benchmark FLOP-à-FLOP. Une seed identique n'engendre pas le même bruit entre
architectures ; la comparaison est appariée par prompt, pas pixel-à-pixel.
"""
    ),
    code(
        """from __future__ import annotations

import gc
import json
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from diffusers import FluxPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download, model_info
from IPython.display import display
from PIL import Image

from prooftag_qr.blueprints import build_adaptive_blueprint, exact_mask_candidates
from prooftag_qr.geometry import AlignedQR
from prooftag_qr.quality_scoring import CLIPQualityScorer
from prooftag_qr.validation import QRValidator, summarize_validation_records

assert torch.cuda.is_available(), 'Lancer dans le pod GPU.'
print(torch.cuda.get_device_name(0))
"""
    ),
    markdown("## 1. Contrat, modèles et reprise"),
    code(
        f"""EXPERIMENT_NAME = 'e015-aesthetic-backbone-reference-v1'
RESUME_RUN_NAME = None
PAYLOAD = 'https://ptag.io/t/e015'
PROMPT_LIMIT = None
{COMMON_PROMPTS}
ACTIVE_PROMPTS = PROMPTS[:PROMPT_LIMIT] if PROMPT_LIMIT else PROMPTS
NEGATIVE_PROMPT = 'text, watermark, letters, low quality, malformed'
CANVAS_SIZE = 768
QR_VERSION = 3
QR_MODULE_SIZE = 20
QR_ECC = 'M'

MODEL_SPECS = [
    {{
        'name': 'sd15_cetus', 'kind': 'sd15',
        'repo': 'fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned',
        'filename': 'cetusMix_Whalefall2_fp16.safetensors',
        'steps': 30, 'guidance': 7.5,
    }},
    {{
        'name': 'sdxl_base_1_0', 'kind': 'sdxl',
        'repo': 'stabilityai/stable-diffusion-xl-base-1.0',
        'steps': 30, 'guidance': 7.0,
    }},
    {{
        'name': 'flux_1_schnell', 'kind': 'flux',
        'repo': 'black-forest-labs/FLUX.1-schnell',
        'steps': 4, 'guidance': 0.0,
    }},
]

if RESUME_RUN_NAME:
    RUN_DIR = Path('/data/notebook-runs') / RESUME_RUN_NAME
else:
    RUN_DIR = Path('/data/notebook-runs') / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + EXPERIMENT_NAME
    )
    RUN_DIR.mkdir(parents=True)
RESULTS_PATH = RUN_DIR / 'results.jsonl'
print('Sortie :', RUN_DIR)
"""
    ),
    markdown("## 2. Utilitaires mémoire, chargement séquentiel et métriques"),
    code(
        """def release_pipeline(pipeline):
    if pipeline is not None:
        try:
            pipeline.to('cpu')
        except Exception:
            pass
        del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def resolve_spec(spec, revision=None):
    resolved = dict(spec)
    resolved['revision'] = revision or model_info(spec['repo']).sha
    if spec['kind'] == 'sd15':
        resolved['local_model'] = hf_hub_download(
            repo_id=spec['repo'], filename=spec['filename'],
            revision=resolved['revision'], cache_dir='/cache/huggingface',
        )
    return resolved


def load_pipeline(spec):
    started = time.perf_counter()
    if spec['kind'] == 'sd15':
        pipeline = StableDiffusionPipeline.from_single_file(
            spec['local_model'], torch_dtype=torch.float16, cache_dir='/cache/huggingface',
            safety_checker=None, use_safetensors=True,
        )
        pipeline = pipeline.to('cuda')
    elif spec['kind'] == 'sdxl':
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            spec['repo'], revision=spec['revision'],
            torch_dtype=torch.float16, variant='fp16',
            cache_dir='/cache/huggingface', use_safetensors=True,
        )
        pipeline.enable_model_cpu_offload()
    elif spec['kind'] == 'flux':
        pipeline = FluxPipeline.from_pretrained(
            spec['repo'], revision=spec['revision'],
            torch_dtype=torch.bfloat16, cache_dir='/cache/huggingface',
        )
        pipeline.enable_model_cpu_offload()
    else:
        raise ValueError(spec['kind'])
    if hasattr(pipeline, 'enable_attention_slicing'):
        pipeline.enable_attention_slicing()
    return pipeline, time.perf_counter() - started


def generate_reference(pipeline, spec, case):
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    kwargs = {
        'prompt': case['text'], 'height': CANVAS_SIZE, 'width': CANVAS_SIZE,
        'num_inference_steps': spec['steps'],
        'generator': torch.Generator(device='cuda').manual_seed(case['seed']),
    }
    if spec['kind'] == 'flux':
        kwargs.update({'guidance_scale': spec['guidance'], 'max_sequence_length': 256})
    else:
        kwargs.update({
            'negative_prompt': NEGATIVE_PROMPT, 'guidance_scale': spec['guidance'],
        })
    image = pipeline(**kwargs).images[0].convert('RGB')
    torch.cuda.synchronize()
    return image, time.perf_counter() - started, torch.cuda.max_memory_allocated() / 2**30


validator = QRValidator()
quality_scorer = CLIPQualityScorer(Path('/cache'), device='cpu')


def validate_exact(image):
    records = validator.validate(image, PAYLOAD)
    summary = summarize_validation_records(records)
    passed = sum(item.exact_payload_match for item in records)
    return {
        'passed': passed, 'total': len(records), 'pass_rate': passed / len(records),
        'strict_all': passed == len(records),
        'worst_decoder_pass_rate': summary['worst_decoder_pass_rate'],
        'worst_scenario_pass_rate': summary['worst_scenario_pass_rate'],
    }


def append_row(row):
    with RESULTS_PATH.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\\n')
        stream.flush()


def completed():
    if not RESULTS_PATH.exists():
        return set()
    keys = {
        (row['model'], row['prompt_id'])
        for line in RESULTS_PATH.read_text(encoding='utf-8').splitlines() if line.strip()
        for row in [json.loads(line)]
    }
    for model_name, prompt_id in keys:
        output_dir = RUN_DIR / model_name / prompt_id
        required = [
            output_dir / 'reference.png', output_dir / 'adaptive-blueprint.png',
            output_dir / 'exact-mask-blueprint.png',
        ]
        if not all(path.exists() for path in required):
            raise RuntimeError(
                f'Résultat {model_name}/{prompt_id} indexé mais artefacts incomplets.'
            )
    return keys
"""
    ),
    markdown("## 3. Générer les douze références et leurs blueprints adaptatifs"),
    code(
        """RESOLVED_PATH = RUN_DIR / 'resolved-model-revisions.json'
if RESOLVED_PATH.exists():
    saved_revisions = json.loads(RESOLVED_PATH.read_text(encoding='utf-8'))
else:
    saved_revisions = {
        spec['name']: model_info(spec['repo']).sha for spec in MODEL_SPECS
    }
    RESOLVED_PATH.write_text(
        json.dumps(saved_revisions, indent=2), encoding='utf-8'
    )
resolved_specs = [
    resolve_spec(spec, saved_revisions[spec['name']]) for spec in MODEL_SPECS
]

for spec in resolved_specs:
    pending = [case for case in ACTIVE_PROMPTS if (spec['name'], case['id']) not in completed()]
    if not pending:
        print('SKIP modèle complet :', spec['name'])
        continue
    pipeline, load_seconds = load_pipeline(spec)
    print(spec['name'], 'chargé en', round(load_seconds, 1), 's')
    for case in pending:
        output_dir = RUN_DIR / spec['name'] / case['id']
        output_dir.mkdir(parents=True, exist_ok=True)
        image, generation_seconds, peak_vram = generate_reference(
            pipeline, spec, case
        )
        image.save(output_dir / 'reference.png')

        exact = exact_mask_candidates(
            PAYLOAD, image, version=QR_VERSION, error_correction=QR_ECC,
            module_size=QR_MODULE_SIZE, canvas_size=CANVAS_SIZE,
        )[0].aligned
        adaptive_candidates = []
        for minimum_fraction in [0.22, 0.30, 0.38, 0.46, 0.55, 0.70, 0.85]:
            adaptive = build_adaptive_blueprint(
                image, exact, minimum_data_fraction=minimum_fraction
            )
            aligned_adaptive = AlignedQR(
                image=adaptive.image, core_matrix=exact.core_matrix.copy(),
                version=exact.version, error_correction=exact.error_correction,
                mask_pattern=exact.mask_pattern, module_size=exact.module_size,
                padding_px=exact.padding_px, canvas_size=exact.canvas_size, payload=exact.payload,
            )
            validation = validate_exact(adaptive.image)
            adaptive_candidates.append((validation, adaptive, aligned_adaptive, minimum_fraction))
        adaptive_candidates.sort(
            key=lambda item: (
                item[0]['strict_all'], item[0]['pass_rate'],
                -item[1].reference_cost, -item[1].grid_visibility,
            ),
            reverse=True,
        )
        validation, adaptive, aligned_adaptive, minimum_fraction = adaptive_candidates[0]
        adaptive.image.save(output_dir / 'adaptive-blueprint.png')
        exact.image.save(output_dir / 'exact-mask-blueprint.png')
        quality = asdict(quality_scorer.score(image, case['text']))
        row = {
            'model': spec['name'], 'kind': spec['kind'], 'model_id': spec['repo'],
            'resolved_revision': spec['revision'],
            'prompt_id': case['id'], 'prompt': case['text'], 'seed': case['seed'],
            'steps': spec['steps'], 'guidance': spec['guidance'],
            'load_seconds': load_seconds, 'generation_seconds': generation_seconds,
            'peak_vram_gib': peak_vram, **quality,
            'adaptive_minimum_fraction': minimum_fraction,
            'adaptive_reference_cost': adaptive.reference_cost,
            'adaptive_grid_visibility': adaptive.grid_visibility,
            **{f'adaptive_{key}': value for key, value in validation.items()},
        }
        append_row(row)
        display(image.resize((384, 384)))
        display(adaptive.image.resize((384, 384)))
    release_pipeline(pipeline)
"""
    ),
    markdown("## 4. Comparaison appariée et décision pour la suite"),
    code(
        """rows = [
    json.loads(line) for line in RESULTS_PATH.read_text(encoding='utf-8').splitlines()
    if line.strip()
]
frame = pd.DataFrame(rows)
frame.to_csv(RUN_DIR / 'comparison.csv', index=False)
display(frame[[
    'model', 'prompt_id', 'generation_seconds', 'peak_vram_gib',
    'clip_aesthetic', 'clip_score', 'adaptive_pass_rate',
    'adaptive_reference_cost', 'adaptive_grid_visibility',
]])

aggregate = frame.groupby('model').agg(
    prompts=('prompt_id', 'nunique'),
    mean_seconds=('generation_seconds', 'mean'),
    max_vram_gib=('peak_vram_gib', 'max'),
    mean_aesthetic=('clip_aesthetic', 'mean'),
    worst_aesthetic=('clip_aesthetic', 'min'),
    mean_clip=('clip_score', 'mean'),
    adaptive_strict=('adaptive_strict_all', 'sum'),
    adaptive_worst_ssr=('adaptive_pass_rate', 'min'),
    adaptive_mean_cost=('adaptive_reference_cost', 'mean'),
).reset_index()
aggregate.to_csv(RUN_DIR / 'aggregate.csv', index=False)
display(aggregate.sort_values(
    ['adaptive_strict', 'adaptive_worst_ssr', 'mean_aesthetic', 'mean_clip'],
    ascending=False,
))

figure, axes = plt.subplots(1, 3, figsize=(18, 5))
for model_name, part in frame.groupby('model'):
    axes[0].scatter(part.clip_score, part.clip_aesthetic, label=model_name, s=70)
    axes[1].scatter(part.generation_seconds, part.clip_aesthetic, label=model_name, s=70)
    axes[2].scatter(part.adaptive_pass_rate, part.adaptive_reference_cost, label=model_name, s=70)
axes[0].set(xlabel='CLIPScore', ylabel='CLIP-aesthetic')
axes[1].set(xlabel='secondes', ylabel='CLIP-aesthetic')
axes[2].set(xlabel='SSR blueprint adaptatif', ylabel='coût visuel vs référence')
for axis in axes:
    axis.grid(alpha=0.25)
axes[0].legend()
figure.tight_layout()
figure.savefig(RUN_DIR / 'objectives.png', dpi=160)
display(figure)
"""
    ),
    markdown("## 5. Manifeste, limites et archive"),
    code(
        """manifest = {
    'experiment': EXPERIMENT_NAME, 'models': MODEL_SPECS,
    'resolved_revisions': saved_revisions, 'prompts': ACTIVE_PROMPTS,
    'payload': PAYLOAD, 'canvas_size': CANVAS_SIZE,
    'selection_scope': 'aesthetic reference only; no claim about QR diffusion compatibility',
    'limits': [
        'Different recommended step budgets are an operational comparison, not equal compute.',
        'Same integer seed does not create identical latent noise across architectures.',
        'Adaptive blueprint validation does not predict a final ControlNet/SRPG image.',
        'FLUX and SDXL are not inserted into the SD1.5 DiffQRCoder pipeline here.',
    ],
}
(RUN_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
archive = shutil.make_archive(str(RUN_DIR), 'gztar', RUN_DIR.parent, RUN_DIR.name)
print('Archive :', archive)
"""
    ),
]


E016_CELLS = [
    markdown(
        r"""# E016 — simulateur différentiable de scannabilité, calibré sur les vrais décodeurs

But : remplacer une loss QR dessinée à la main par un petit réseau qui approxime la probabilité de
lecture d'OpenCV, ZBar et ZXing-cpp après dégradations. Le réseau n'est jamais promu parce que sa
loss baisse : il doit améliorer les **vrais décodeurs** sur un jeu holdout et, plus tard, sur des
captures téléphone/impression.

Le notebook :

1. indexe des images générées et des captures physiques optionnelles ;
2. applique les treize scénarios du validateur et demande les labels aux vrais décodeurs ;
3. divise par groupe source/prompt pour éviter une fuite d'images quasi identiques ;
4. entraîne un CNN multi-sorties entièrement différentiable ;
5. mesure AUCPR, ROC-AUC, Brier et calibration par décodeur ;
6. tente une optimisation de pixels bornée et vérifie le résultat avec les vrais décodeurs ;
7. exporte TorchScript seulement si les seuils minimaux sont atteints.
"""
    ),
    markdown(
        r"""## Garde-fous

```text
image source ─► dégradations réelles ─► labels OpenCV/ZBar/ZXing ─► CNN différentiable
      │                                      │                         │
      └──────── groupe anti-fuite ───────────┘                         ▼
                                                            gradient sur holdout
                                                                     │
                                  vrais décodeurs avant/après ◄───────┘
```

- Pas assez de positifs/négatifs : arrêt, dataset conservé.
- Pas de captures physiques : modèle marqué `digital_only`, jamais production.
- Le CNN peut être trompé adversarialement : seul le gain des vrais décodeurs compte.
"""
    ),
    code(
        """from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from prooftag_qr.validation import DEFAULT_SCENARIOS, QRValidator

assert torch.cuda.is_available(), 'Le CNN peut tourner sur CPU, mais le pod GPU est recommandé.'
DEVICE = torch.device('cuda')
print(torch.cuda.get_device_name(0))
"""
    ),
    markdown("## 1. Sources, seuils d'arrêt et captures physiques"),
    code(
        """EXPERIMENT_NAME = 'e016-differentiable-real-decoder-surrogate-v1'
DEFAULT_PAYLOAD = 'https://ptag.io/t/e014'
SOURCE_RUNS = []  # ex. [Path('/data/notebook-runs/...e014a...'), Path('/data/notebook-runs/...e014b...')]
PHYSICAL_CSV = Path('/data/physical-captures/labels.csv')
IMAGE_SIZE = 256
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
MIN_SOURCE_GROUPS = 12
MIN_SAMPLES = 120
MIN_CLASS_COUNT_PER_DECODER = 10
REQUIRE_PHYSICAL_FOR_PRODUCTION = True

RUN_DIR = Path('/data/notebook-runs') / (
    datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + EXPERIMENT_NAME
)
RUN_DIR.mkdir(parents=True)
DATASET_IMAGE_DIR = RUN_DIR / 'labelled-images'
DATASET_IMAGE_DIR.mkdir()

if not SOURCE_RUNS:
    patterns = ['*-e014a-real-qart-exact-adaptive-v1', '*-e014b-freeqr-latent-channel-timestep-v1']
    for pattern in patterns:
        matches = sorted(Path('/data/notebook-runs').glob(pattern))
        if matches:
            SOURCE_RUNS.append(matches[-1])
SOURCE_RUNS = [Path(path) for path in SOURCE_RUNS]
if not SOURCE_RUNS:
    raise FileNotFoundError('Aucune source E014A/E014B. Renseigner SOURCE_RUNS.')

template = pd.DataFrame(columns=[
    'image_path', 'expected_payload', 'group_id', 'device',
    'screen_or_print', 'distance_cm', 'lighting', 'notes',
])
template.to_csv(RUN_DIR / 'physical-captures-template.csv', index=False)
print('Sources :', SOURCE_RUNS)
print('Gabarit physique :', RUN_DIR / 'physical-captures-template.csv')
"""
    ),
    markdown("## 2. Construire le dataset avec les vrais décodeurs"),
    code(
        """validator = QRValidator()
decoder_names = [decoder.name for decoder in validator.decoders]
print('Décodeurs :', decoder_names)
if len(decoder_names) < 3:
    raise RuntimeError('E016 exige OpenCV + ZBar + ZXing-cpp pour ne pas suradapter un seul moteur.')


def source_pngs(run_dir):
    for path in run_dir.rglob('*.png'):
        if path.name == 'final.png' and 'frames' not in path.parts:
            yield path


def payload_for_run(run_dir):
    manifest = run_dir / 'manifest.json'
    if manifest.exists():
        return json.loads(manifest.read_text(encoding='utf-8')).get('payload', DEFAULT_PAYLOAD)
    return DEFAULT_PAYLOAD


def context_group_for(run_dir, source_path, manifest):
    prompt_id = manifest.get('prompt_id')
    seed = manifest.get('seed')
    prompt_specs = manifest.get('prompts', [])
    if not prompt_id:
        relative_parts = source_path.relative_to(run_dir).parts
        known_ids = {str(item.get('id')) for item in prompt_specs}
        prompt_id = next((part for part in relative_parts if part in known_ids), None)
    if prompt_id and prompt_specs:
        matching = next(
            (item for item in prompt_specs if str(item.get('id')) == str(prompt_id)),
            {},
        )
        seed = matching.get('seed', seed)
    if not prompt_id:
        raise ValueError(
            f'Contexte prompt/seed introuvable pour {source_path}; '
            'ajouter prompt_id/seed au manifest au lieu de créer une fuite.'
        )
    return hashlib.sha256(
        f'{payload_for_run(run_dir)}:{prompt_id}:{seed}'.encode('utf-8')
    ).hexdigest()[:16]


records = []
for run_dir in SOURCE_RUNS:
    expected_payload = payload_for_run(run_dir)
    manifest_path = run_dir / 'manifest.json'
    manifest = (
        json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest_path.exists() else {}
    )
    for source_path in source_pngs(run_dir):
        source = Image.open(source_path).convert('RGB')
        source_group = context_group_for(run_dir, source_path, manifest)
        for scenario in DEFAULT_SCENARIOS:
            transformed = scenario.apply(source)
            key = hashlib.sha256(
                f'{source_group}:{scenario.name}'.encode('utf-8')
            ).hexdigest()[:20]
            saved_path = DATASET_IMAGE_DIR / f'{key}.jpg'
            transformed.save(saved_path, quality=95)
            row = {
                'image_path': str(saved_path), 'source_path': str(source_path),
                'source_run': run_dir.name, 'source_group': source_group,
                'scenario': scenario.name, 'physical': False,
                'expected_payload': expected_payload,
                'expected_payload_hash': hashlib.sha256(expected_payload.encode()).hexdigest(),
            }
            for decoder in validator.decoders:
                decoded = decoder.decode(transformed)
                row[f'label_{decoder.name}'] = int(decoded == expected_payload)
                row[f'detected_{decoder.name}'] = int(bool(decoded))
            records.append(row)

physical_count = 0
if PHYSICAL_CSV.exists():
    physical_frame = pd.read_csv(PHYSICAL_CSV)
    for index, source_row in physical_frame.iterrows():
        path = Path(source_row.image_path)
        if not path.exists():
            print('Capture absente, ignorée :', path)
            continue
        image = Image.open(path).convert('RGB')
        expected_payload = str(source_row.expected_payload)
        group = str(source_row.group_id)
        saved_path = DATASET_IMAGE_DIR / f'physical-{index:05d}.jpg'
        image.save(saved_path, quality=98)
        row = {
            'image_path': str(saved_path), 'source_path': str(path),
            'source_run': 'physical', 'source_group': f'physical:{group}',
            'scenario': 'physical_original', 'physical': True,
            'expected_payload': expected_payload,
            'expected_payload_hash': hashlib.sha256(expected_payload.encode()).hexdigest(),
        }
        for decoder in validator.decoders:
            decoded = decoder.decode(image)
            row[f'label_{decoder.name}'] = int(decoded == expected_payload)
            row[f'detected_{decoder.name}'] = int(bool(decoded))
        records.append(row)
        physical_count += 1

dataset_frame = pd.DataFrame(records)
dataset_frame.to_csv(RUN_DIR / 'decoder-dataset.csv', index=False)
print('Lignes :', len(dataset_frame), 'groupes :', dataset_frame.source_group.nunique())
print('Captures physiques :', physical_count)
display(dataset_frame[[f'label_{name}' for name in decoder_names]].sum().to_frame('positifs'))
"""
    ),
    markdown("## 3. Vérifier l'identifiabilité avant d'entraîner"),
    code(
        """problems = []
if len(dataset_frame) < MIN_SAMPLES:
    problems.append(f'{len(dataset_frame)} lignes < {MIN_SAMPLES}')
if dataset_frame.source_group.nunique() < MIN_SOURCE_GROUPS:
    problems.append(
        f'{dataset_frame.source_group.nunique()} groupes < {MIN_SOURCE_GROUPS}'
    )
for name in decoder_names:
    positives = int(dataset_frame[f'label_{name}'].sum())
    negatives = len(dataset_frame) - positives
    if min(positives, negatives) < MIN_CLASS_COUNT_PER_DECODER:
        problems.append(
            f'{name}: classe minoritaire {min(positives, negatives)} < '
            f'{MIN_CLASS_COUNT_PER_DECODER}'
        )
if problems:
    (RUN_DIR / 'STOP-INSUFFICIENT-DATA.json').write_text(
        json.dumps({'problems': problems}, indent=2), encoding='utf-8'
    )
    raise RuntimeError('Dataset non identifiable : ' + '; '.join(problems))
"""
    ),
    markdown("## 4. Split par source, Dataset PyTorch et CNN multi-décodeur"),
    code(
        """groups = dataset_frame.source_group.to_numpy()
outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=20260723)
train_val_index, test_index = next(outer.split(dataset_frame, groups=groups))
train_val = dataset_frame.iloc[train_val_index].reset_index(drop=True)
test_frame = dataset_frame.iloc[test_index].reset_index(drop=True)
inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=20260724)
train_index, val_index = next(
    inner.split(train_val, groups=train_val.source_group.to_numpy())
)
train_frame = train_val.iloc[train_index].reset_index(drop=True)
val_frame = train_val.iloc[val_index].reset_index(drop=True)
assert set(train_frame.source_group).isdisjoint(val_frame.source_group)
assert set(train_frame.source_group).isdisjoint(test_frame.source_group)
assert set(val_frame.source_group).isdisjoint(test_frame.source_group)
print('train/val/test :', len(train_frame), len(val_frame), len(test_frame))


class DecoderDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(row.image_path).convert('RGB').resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
        )
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1)
        labels = torch.tensor(
            [row[f'label_{name}'] for name in decoder_names], dtype=torch.float32
        )
        return tensor, labels


class ScanSurrogate(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        channels = [3, 32, 64, 128, 192]
        blocks = []
        for source, target in zip(channels[:-1], channels[1:]):
            blocks.extend([
                nn.Conv2d(source, target, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(target), nn.SiLU(),
                nn.Conv2d(target, target, 3, padding=1, groups=target, bias=False),
                nn.BatchNorm2d(target), nn.SiLU(),
            ])
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.15),
            nn.Linear(channels[-1], outputs),
        )

    def forward(self, image):
        return self.head(self.features(image))


loaders = {
    'train': DataLoader(DecoderDataset(train_frame), batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'val': DataLoader(DecoderDataset(val_frame), batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    'test': DataLoader(DecoderDataset(test_frame), batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
}
model = ScanSurrogate(len(decoder_names)).to(DEVICE)
positives = torch.tensor(
    [train_frame[f'label_{name}'].sum() for name in decoder_names], dtype=torch.float32
)
negatives = len(train_frame) - positives
criterion = nn.BCEWithLogitsLoss(pos_weight=(negatives / positives.clamp_min(1)).to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
"""
    ),
    markdown("## 5. Entraînement avec sélection sur loss validation"),
    code(
        """history = []
best_state = None
best_val = float('inf')
for epoch in range(1, EPOCHS + 1):
    epoch_values = {}
    for phase in ['train', 'val']:
        model.train(phase == 'train')
        total_loss = 0.0
        total_items = 0
        for images, labels in loaders[phase]:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.set_grad_enabled(phase == 'train'):
                logits = model(images)
                loss = criterion(logits, labels)
                if phase == 'train':
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            total_loss += float(loss.detach()) * len(images)
            total_items += len(images)
        epoch_values[phase] = total_loss / total_items
    history.append({'epoch': epoch, **epoch_values})
    if epoch_values['val'] < best_val:
        best_val = epoch_values['val']
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    print(f"epoch {epoch:02d} train={epoch_values['train']:.4f} val={epoch_values['val']:.4f}")

model.load_state_dict(best_state)
pd.DataFrame(history).to_csv(RUN_DIR / 'training-history.csv', index=False)
"""
    ),
    markdown("## 6. Calibration et holdout sans fuite"),
    code(
        """def predict(loader):
    model.eval()
    truths, probabilities = [], []
    with torch.no_grad():
        for images, labels in loader:
            probabilities.append(torch.sigmoid(model(images.to(DEVICE))).cpu().numpy())
            truths.append(labels.numpy())
    return np.concatenate(truths), np.concatenate(probabilities)


y_test, p_test = predict(loaders['test'])
metrics = {}
figure, axes = plt.subplots(1, len(decoder_names), figsize=(6 * len(decoder_names), 5))
axes = np.atleast_1d(axes)
for index, name in enumerate(decoder_names):
    truth, probability = y_test[:, index], p_test[:, index]
    both_classes = len(np.unique(truth)) == 2
    metrics[name] = {
        'average_precision': float(average_precision_score(truth, probability)),
        'brier': float(brier_score_loss(truth, probability)),
        'roc_auc': float(roc_auc_score(truth, probability)) if both_classes else None,
        'positives': int(truth.sum()), 'negatives': int(len(truth) - truth.sum()),
        'holdout_has_both_classes': both_classes,
    }
    observed, predicted = calibration_curve(truth, probability, n_bins=8, strategy='quantile')
    axes[index].plot(predicted, observed, marker='o', label=name)
    axes[index].plot([0, 1], [0, 1], '--', color='gray')
    axes[index].set(xlabel='probabilité prédite', ylabel='fréquence réelle', title=name)
    axes[index].grid(alpha=0.25)
figure.tight_layout()
figure.savefig(RUN_DIR / 'calibration.png', dpi=160)
display(figure)
(RUN_DIR / 'test-metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
display(pd.DataFrame(metrics).T)
"""
    ),
    markdown(
        """## 7. Audit adversarial : le gradient améliore-t-il les vrais décodeurs ?

On choisit un négatif holdout, optimise au plus ±8/255 par pixel avec pénalité TV, puis redemande
les labels aux vrais décodeurs. Une probabilité CNN plus haute sans amélioration réelle est un
échec du surrogate, pas un succès.
"""
    ),
    code(
        """strict_probability = p_test.min(axis=1)
candidate_index = int(np.argmin(strict_probability))
source_row = test_frame.iloc[candidate_index]
before = Image.open(source_row.image_path).convert('RGB')
before_tensor = torch.from_numpy(
    np.asarray(before.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0
).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
working = before_tensor.clone().detach().requires_grad_(True)
pixel_optimizer = torch.optim.Adam([working], lr=0.01)
audit_trace = []
for iteration in range(31):
    logits = model(working)
    target = torch.ones_like(logits)
    success_loss = F.binary_cross_entropy_with_logits(logits, target)
    delta = working - before_tensor
    tv = (
        delta[:, :, 1:, :].sub(delta[:, :, :-1, :]).abs().mean()
        + delta[:, :, :, 1:].sub(delta[:, :, :, :-1]).abs().mean()
    )
    loss = success_loss + 0.10 * tv + 0.25 * delta.square().mean()
    audit_trace.append({
        'iteration': iteration, 'objective': float(loss.detach()),
        'predicted_min_probability': float(torch.sigmoid(logits).min().detach()),
        'delta_rms': float(delta.square().mean().sqrt().detach()),
    })
    if iteration == 30:
        break
    pixel_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    pixel_optimizer.step()
    with torch.no_grad():
        working.copy_(torch.max(torch.min(working, before_tensor + 8 / 255), before_tensor - 8 / 255))
        working.clamp_(0, 1)

after_array = (
    working.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 255
).round().astype(np.uint8)
after = Image.fromarray(after_array).resize(before.size, Image.Resampling.LANCZOS)
before.save(RUN_DIR / 'audit-before.png')
after.save(RUN_DIR / 'audit-after.png')
(RUN_DIR / 'audit-trace.json').write_text(json.dumps(audit_trace, indent=2), encoding='utf-8')

expected_payload = str(source_row.expected_payload)
before_records = validator.validate(before, expected_payload)
after_records = validator.validate(after, expected_payload)
real_before = sum(item.exact_payload_match for item in before_records)
real_after = sum(item.exact_payload_match for item in after_records)
audit = {
    'real_before': real_before, 'real_after': real_after,
    'real_total': len(before_records),
    'surrogate_before': audit_trace[0]['predicted_min_probability'],
    'surrogate_after': audit_trace[-1]['predicted_min_probability'],
    'real_decoder_improved': real_after > real_before,
}
(RUN_DIR / 'gradient-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(audit)
display(before.resize((420, 420)))
display(after.resize((420, 420)))
"""
    ),
    markdown("## 8. Porte de promotion, TorchScript et archive"),
    code(
        """metric_gate = all(
    values['holdout_has_both_classes']
    and values['average_precision'] >= 0.75
    and values['brier'] <= 0.20
    for values in metrics.values()
)
physical_gate = physical_count > 0 or not REQUIRE_PHYSICAL_FOR_PRODUCTION
promotion = {
    'enough_data': True,
    'metric_gate': metric_gate,
    'real_decoder_gradient_gate': audit['real_decoder_improved'],
    'physical_gate': physical_gate,
    'physical_samples': physical_count,
}
promotion['research_usable'] = (
    promotion['metric_gate'] and promotion['real_decoder_gradient_gate']
)
promotion['production_usable'] = promotion['research_usable'] and promotion['physical_gate']

model = model.eval().cpu()
scripted = torch.jit.trace(model, torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
scripted.save(str(RUN_DIR / 'scan-surrogate.torchscript.pt'))
model_card = {
    'experiment': EXPERIMENT_NAME, 'decoder_outputs': decoder_names,
    'input': {'shape': [3, IMAGE_SIZE, IMAGE_SIZE], 'range': [0, 1]},
    'source_runs': [str(path) for path in SOURCE_RUNS],
    'split': 'GroupShuffleSplit by source_group; train/val/test disjoint',
    'metrics': metrics, 'promotion': promotion,
    'warning': 'Never replace external decoders or physical tests with this surrogate.',
}
(RUN_DIR / 'surrogate-card.json').write_text(json.dumps(model_card, indent=2), encoding='utf-8')
print('Promotion :', promotion)
if not promotion['production_usable']:
    print('NON PRODUCTION : conserver comme outil de recherche uniquement.')
archive = shutil.make_archive(str(RUN_DIR), 'gztar', RUN_DIR.parent, RUN_DIR.name)
print('Archive :', archive)
"""
    ),
]


write_notebook("11_e014a_qart_blueprint_bakeoff.ipynb", E014A_CELLS)
write_notebook("12_e014b_freeqr_latent_fusion.ipynb", E014B_CELLS)
write_notebook("13_e015_aesthetic_backbone_reference.ipynb", E015_CELLS)
write_notebook("14_e016_differentiable_scan_surrogate.ipynb", E016_CELLS)
