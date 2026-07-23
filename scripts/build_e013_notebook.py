# ruff: noqa: E501
"""Build E013: exact geometry, SD1.5/SD2.1 bake-off, optimization and policy data."""

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
        r"""# E013 — géométrie exacte, DiffQRCoder SD 1.5, ControlNet QR SD 2.1 et recette adaptative

Ce notebook répond à quatre questions sans les mélanger :

1. la grille QR est-elle exacte avant toute diffusion ?
2. DiffQRCoder SD 1.5 ou le ControlNet QR natif SD 2.1 donne-t-il le meilleur compromis ?
3. le SR-MPGD du papier et celui du dépôt public améliorent-ils vraiment la lecture sans détruire l'image ?
4. peut-on apprendre une politique qui choisit les paramètres et le nombre de tentatives selon le prompt ?

La priorité est lexicographique : **payload exact et robustesse**, puis **CLIP-aesthetic**, **CLIPScore**, enfin le temps. Une sortie ne porte jamais le nom `DELIVERABLE` sans avoir franchi tous les tests logiciels.

```text
URL courte Prooftag
        │
        ▼
QR v3 aligné au pixel ── contrôle OpenCV + ZBar ── échec → STOP
        │
        ├──────────────► DiffQRCoder / SD 1.5 / QR Monster v2
        │                        │ Stage 1 puis Stage 2 SRPG
        │                        ├ base
        │                        ├ SR-MPGD papier
        │                        └ SR-MPGD dépôt public
        │
        └──────────────► SD 2.1 / DionTimmer QR ControlNet
                                 │ text-to-image puis rescue img2img optionnel
                                 ├ base
                                 └ SR-MPGD papier
                                          │
                                          ▼
              26 tests + MER exacte + CLIP-aes + CLIPScore + durée + VRAM
                                          │
                     Optuna contraint par la lecture, qualité en objectifs secondaires
                                          │
                         données tabulaires → CatBoost quand l'échantillon suffit
                                          │
                         top-K candidats → validation → livraison ou rejet
```
"""
    ),
    markdown("## 0. Environnement, versions et provenance"),
    code(
        """from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
)
from IPython.display import Markdown, clear_output, display
from PIL import Image
from safetensors.torch import load_file, save_file

UPSTREAM_ROOT = Path('/opt/DiffQRCoder')
DIFFQRCODER_COMMIT = 'e24ea73ee2e13c7e6e87cb422e8b11784e70ae00'
if not (UPSTREAM_ROOT / 'diffqrcoder' / 'pipeline_diffqrcoder.py').exists():
    raise RuntimeError('DiffQRCoder absent : reconstruire Dockerfile.notebook.')
sys.path.insert(0, str(UPSTREAM_ROOT))

from diffqrcoder import DiffQRCoderPipeline  # noqa: E402
from diffqrcoder.losses import ScanningRobustLoss  # noqa: E402
from prooftag_qr.geometry import (  # noqa: E402
    aligned_module_diagnostics,
    aligned_module_error_rate,
    generate_aligned_qr,
)
from prooftag_qr.policy import (  # noqa: E402
    attempts_for_target,
    candidate_rank,
    deliverable_candidate,
    delivery_probability,
)
from prooftag_qr.quality import image_change_metrics, image_quality_metrics  # noqa: E402
from prooftag_qr.quality_scoring import CLIPQualityScorer, project_embedding  # noqa: E402
from prooftag_qr.srmpgd import SRMPGDConfig, run_srmpgd  # noqa: E402
from prooftag_qr.validation import QRValidator, summarize_validation_records  # noqa: E402

print('torch      :', torch.__version__)
print('diffusers  :', importlib.metadata.version('diffusers'))
print('optuna     :', importlib.metadata.version('optuna'))
print('CUDA       :', torch.cuda.is_available())
print('GPU        :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'absent')
print('DiffQRCoder:', DIFFQRCODER_COMMIT)
assert torch.cuda.is_available(), 'Exécuter dans le notebook distant sur la RTX, pas avec Python Windows.'
"""
    ),
    markdown("## 1. Protocole modifiable et budgets explicites"),
    code(
        """EXPERIMENT_NAME = 'e013-exact-geometry-sd15-sd21-policy-v1'
RESUME_RUN_NAME = None  # nom du dossier /data/notebook-runs/... après interruption

PAYLOAD = 'https://ptag.io/t/e013'
QR_VERSION = 3
QR_MODULE_SIZE = 20
QR_MASK_BASELINE = 4
QR_ECC_BASELINE = 'H'

PROMPTS = [
    {'id': 'p1_simple', 'seed': 1101, 'text': 'A single white lotus flower floating on a dark calm pond, elegant editorial photograph.'},
    {'id': 'p2_medium', 'seed': 2202, 'text': 'A Japanese garden with a red bridge, mossy stones and soft morning mist, detailed photography.'},
    {'id': 'p3_detailed', 'seed': 3303, 'text': 'An ornate botanical tapestry of white lilies, pale blue leaves and dark vines, intricate textile illustration.'},
    {'id': 'p4_complex', 'seed': 4404, 'text': 'A lively old European market square, cafe terraces, flowers, bicycles and a gothic cathedral, cinematic morning light.'},
]
NEGATIVE_PROMPTS = {
    'minimal': 'text, watermark, barcode',
    'standard': 'easynegative, text, watermark, blurry, low quality, barcode',
    'structure_safe': (
        'easynegative, text, watermark, blurry, low quality, barcode, '
        'regular checkerboard, repeating squares, tiny high-frequency details'
    ),
}

SD15_BASE = 'https://huggingface.co/fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned/blob/main/cetusMix_Whalefall2_fp16.safetensors'
SD15_CONTROLNET = 'monster-labs/control_v1p_sd15_qrcode_monster'
SD15_CONTROLNET_SUBFOLDER = 'v2'
SD21_BASE_CANDIDATES = [
    'stabilityai/stable-diffusion-2-1',
    'sd2-community/stable-diffusion-2-1',
]
SD21_CONTROLNET = 'DionTimmer/controlnet_qrcode-control_v11p_sd21'

BASELINE_CONFIGS = [
    {'name': 'sd15_744_40', 'model': 'sd15_diffqrcoder', 'canvas': 744, 'module_size': 20, 'steps': 40, 'stage1_steps': 40, 'cfg': 7.5, 'control': 1.35, 'srg': 500.0, 'pg': 3.0, 'rescue_strength': 0.0},
    {'name': 'sd15_744_100', 'model': 'sd15_diffqrcoder', 'canvas': 744, 'module_size': 20, 'steps': 100, 'stage1_steps': 40, 'cfg': 7.5, 'control': 1.35, 'srg': 500.0, 'pg': 3.0, 'rescue_strength': 0.0},
    {'name': 'sd15_768_40', 'model': 'sd15_diffqrcoder', 'canvas': 768, 'module_size': 20, 'steps': 40, 'stage1_steps': 40, 'cfg': 7.5, 'control': 1.35, 'srg': 500.0, 'pg': 3.0, 'rescue_strength': 0.0},
    {'name': 'sd15_768_m16_40', 'model': 'sd15_diffqrcoder', 'canvas': 768, 'module_size': 16, 'steps': 40, 'stage1_steps': 40, 'cfg': 7.5, 'control': 1.35, 'srg': 500.0, 'pg': 3.0, 'rescue_strength': 0.0},
    {'name': 'sd21_768_50', 'model': 'sd21_dion', 'canvas': 768, 'module_size': 20, 'steps': 50, 'stage1_steps': 0, 'cfg': 12.0, 'control': 1.50, 'srg': 0.0, 'pg': 0.0, 'rescue_strength': 0.0},
    {'name': 'sd21_768_100', 'model': 'sd21_dion', 'canvas': 768, 'module_size': 20, 'steps': 100, 'stage1_steps': 0, 'cfg': 12.0, 'control': 1.50, 'srg': 0.0, 'pg': 0.0, 'rescue_strength': 0.0},
    {'name': 'sd21_768_rescue', 'model': 'sd21_dion', 'canvas': 768, 'module_size': 20, 'steps': 50, 'stage1_steps': 0, 'cfg': 12.0, 'control': 1.50, 'srg': 0.0, 'pg': 0.0, 'rescue_strength': 0.35},
    {'name': 'sd21_768_m16_50', 'model': 'sd21_dion', 'canvas': 768, 'module_size': 16, 'steps': 50, 'stage1_steps': 0, 'cfg': 12.0, 'control': 1.50, 'srg': 0.0, 'pg': 0.0, 'rescue_strength': 0.0},
]
for config in BASELINE_CONFIGS:
    config.update({
        'ecc': QR_ECC_BASELINE, 'mask': QR_MASK_BASELINE, 'eta': 0.0,
        'negative_profile': 'standard', 'control_start': 0.0, 'control_end': 1.0,
        'refinement': 'none', 'paper_iterations': 20, 'paper_step_size': 1000.0,
        'paper_lpips_weight': 0.01, 'dark_threshold': 0.45,
        'light_threshold': 0.65, 'center_fraction': 1 / 3,
    })

PAPER_SRMPGD = SRMPGDConfig(
    max_iterations=20,
    step_size=1000.0,
    lpips_weight=0.01,
    lpips_net='vgg',
    crop_padding_px=0,  # remplacé par le padding exact de chaque géométrie
    dark_threshold=0.45,
    light_threshold=0.65,
    center_fraction=1 / 3,
)
UPSTREAM_SRMPGD_ITERATIONS = 20
UPSTREAM_SRMPGD_LR = 0.1

RUN_BASELINE = True
RUN_OPTUNA = True
OPTUNA_TRIALS_PER_MODEL = 32
RUN_CONFIRMATION = True
CONFIRMATION_CONTEXTS = 4
DELIVERY_TARGET = 0.999
MAX_DELIVERY_ATTEMPTS = 12

SAVE_BASELINE_EVERY_STEP = True
SAVE_SEARCH_EVERY_STEP = False
DISPLAY_EVERY = 10

if RESUME_RUN_NAME:
    RUN_DIR = Path('/data/notebook-runs') / RESUME_RUN_NAME
    if not RUN_DIR.is_dir():
        raise FileNotFoundError(RUN_DIR)
else:
    run_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{EXPERIMENT_NAME}"
    RUN_DIR = Path('/data/notebook-runs') / run_name
    RUN_DIR.mkdir(parents=True, exist_ok=False)

RESULTS_PATH = RUN_DIR / 'results.jsonl'
SEARCH_RESULTS_PATH = RUN_DIR / 'search-results.jsonl'
print('Dossier :', RUN_DIR)
EXPECTED_BASELINE = len(PROMPTS) * sum(
    3 if config['model'] == 'sd15_diffqrcoder' else 2
    for config in BASELINE_CONFIGS
)
print('Baseline attendue :', EXPECTED_BASELINE, 'lignes')
"""
    ),
    markdown(
        """## 2. QR exact : aucune interpolation de la matrice

Le cœur v3 contient 29×29 modules. Avec des modules de 20 px il mesure exactement 580 px :

- SD 1.5 à 744 px : `(744 - 580) / 2 = 82 px`, soit 4,10 modules blancs ;
- SD 2.1 à 768 px : `(768 - 580) / 2 = 94 px`, soit 4,70 modules blancs.

Contrairement à E012, aucun QR 740 px n'est redimensionné en 736 px. Chaque contrôle doit réussir les deux décodeurs sur l'image originale avant le chargement d'un modèle.
"""
    ),
    code(
        """validator = QRValidator()
decoder_names = [decoder.name for decoder in validator.decoders]
missing = {'opencv', 'zbar'} - set(decoder_names)
if missing:
    raise RuntimeError(f'Décodeurs absents {sorted(missing)} : reconstruire Dockerfile.notebook.')

GEOMETRIES = {}
geometry_audit = []
for canvas in [744, 768]:
    for module_size in [16, 20]:
      for ecc in ['M', 'Q', 'H']:
        aligned = generate_aligned_qr(
            PAYLOAD,
            version=QR_VERSION,
            error_correction=ecc,
            mask_pattern=QR_MASK_BASELINE,
            module_size=module_size,
            canvas_size=canvas,
        )
        key = (canvas, module_size, ecc, QR_MASK_BASELINE)
        GEOMETRIES[key] = aligned
        path = RUN_DIR / f'00_qr_control_{canvas}_m{module_size}_{ecc}.png'
        aligned.image.save(path)
        records = validator.validate(aligned.image, PAYLOAD)
        originals = [item for item in records if item.scenario == 'original']
        passed = sum(item.exact_payload_match for item in records)
        original_passed = sum(item.exact_payload_match for item in originals)
        row = {
            'canvas': canvas, 'ecc': ecc, 'mask': QR_MASK_BASELINE,
            'core_modules': aligned.core_modules, 'module_size': aligned.module_size,
            'core_size': aligned.core_size, 'padding_px': aligned.padding_px,
            'quiet_zone_modules': aligned.quiet_zone_modules,
            'passed': passed, 'total': len(records),
            'original_passed': original_passed, 'original_total': len(originals),
            **aligned_module_diagnostics(aligned.image, aligned),
        }
        geometry_audit.append(row)
        assert original_passed == len(originals), f'QR témoin original invalide : {row}'
        assert row['module_error_rate'] == 0

(RUN_DIR / 'geometry-audit.json').write_text(json.dumps(geometry_audit, indent=2), encoding='utf-8')
display(pd.DataFrame(geometry_audit))
display(GEOMETRIES[(768, 20, 'H', 4)].image.resize((384, 384)))
"""
    ),
    markdown("## 3. Persistance, validation, scores et images intermédiaires"),
    code(
        """def append_jsonl(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\\n')


def jsonl_rows(path):
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'{path}:{line_number} JSON invalide') from exc
    return rows


def result_key(row):
    return (
        row['phase'], row['prompt_id'], row['model'], row['profile'],
        row['variant'], int(row['seed']),
    )


def result_index(path=RESULTS_PATH):
    return {result_key(row): row for row in jsonl_rows(path)}


def save_tensor(path, key, tensor):
    save_file({key: tensor.detach().cpu().contiguous()}, str(path))


def load_tensor(path, key, dtype=torch.float16):
    return load_file(str(path), device='cpu')[key].to('cuda', dtype=dtype)


def cuda_memory_gib():
    free, total = torch.cuda.mem_get_info()
    return {
        'allocated': torch.cuda.memory_allocated() / 2**30,
        'reserved': torch.cuda.memory_reserved() / 2**30,
        'free_driver': free / 2**30,
        'total_driver': total / 2**30,
        'peak_allocated': torch.cuda.max_memory_allocated() / 2**30,
    }


def release_gpu(*objects):
    for value in objects:
        if value is None:
            continue
        if hasattr(value, 'to'):
            try:
                value.to('cpu')
            except Exception:
                pass
    del objects
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def decode_latent(pipe, latent):
    with torch.no_grad():
        dtype = next(pipe.vae.parameters()).dtype
        decoded = pipe.vae.decode(
            latent.detach().to(dtype=dtype) / pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        return pipe.image_processor.postprocess(decoded.detach(), output_type='pil')[0].convert('RGB')


def make_gif(frame_dir, output):
    paths = sorted(frame_dir.glob('*.jpg'))
    if not paths:
        return
    frames = [Image.open(path).convert('RGB').resize((512, 512)) for path in paths]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=160, loop=0)
    for frame in frames:
        frame.close()


def diffusion_callback(pipe_ref, aligned, label, steps, frame_dir, save_every_step):
    frame_dir.mkdir(parents=True, exist_ok=True)
    trace = []
    started = time.perf_counter()

    def callback(pipeline, step_index, timestep, callback_kwargs):
        should_save = save_every_step or step_index % DISPLAY_EVERY == 0 or step_index == steps - 1
        if should_save:
            preview = decode_latent(pipe_ref, callback_kwargs['latents'])
            diagnostics = aligned_module_diagnostics(preview, aligned)
            row = {
                'step': int(step_index), 'timestep': int(timestep),
                'elapsed_s': time.perf_counter() - started, **diagnostics,
            }
            trace.append(row)
            preview.save(frame_dir / f'{step_index:03d}.jpg', quality=88)
            if step_index % DISPLAY_EVERY == 0 or step_index == steps - 1:
                clear_output(wait=True)
                display(Markdown(
                    f"**{label} — {step_index + 1}/{steps} — "
                    f"MER {diagnostics['module_error_rate']:.2%}, "
                    f"marge {diagnostics['minimum_threshold_margin']:.3f}**"
                ))
                display(preview.resize((430, 430)))
        return callback_kwargs

    return callback, trace


def validation_payload(image):
    records = validator.validate(image, PAYLOAD)
    summary = summarize_validation_records(records)
    passed = sum(item.exact_payload_match for item in records)
    originals = [item for item in records if item.scenario == 'original']
    original_passed = sum(item.exact_payload_match for item in originals)
    return {
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
    }, [asdict(item) for item in records]


quality_scorer = CLIPQualityScorer(Path('/cache'), device='cpu')
prompt_embedding_cache = {}


def evaluate_candidate(phase, case, config, variant, image, aligned, duration_s, reference, artifact_dir, extra=None):
    validation, records = validation_payload(image)
    try:
        quality = asdict(quality_scorer.score(image, case['text']))
        quality_error = None
    except Exception as exc:
        quality = {'clip_similarity': None, 'clip_score': None, 'clip_aesthetic': None}
        quality_error = f'{type(exc).__name__}: {exc}'
    if case['text'] not in prompt_embedding_cache:
        try:
            prompt_embedding_cache[case['text']] = project_embedding(
                quality_scorer.text_embedding(case['text']), dimensions=16
            )
        except Exception as exc:
            prompt_embedding_cache[case['text']] = [None] * 16
            embedding_error = f'{type(exc).__name__}: {exc}'
            quality_error = (
                f'{quality_error}; prompt embedding: {embedding_error}'
                if quality_error else f'prompt embedding: {embedding_error}'
            )
    prompt_features = {
        f'prompt_embedding_{index:02d}': value
        for index, value in enumerate(prompt_embedding_cache[case['text']])
    }
    row = {
        'phase': phase, 'prompt_id': case['id'], 'prompt': case['text'],
        'seed': int(case['seed']), 'model': config['model'], 'profile': config['name'],
        'variant': variant, 'duration_s': duration_s,
        'canvas': aligned.canvas_size, 'ecc': aligned.error_correction,
        'mask_pattern': aligned.mask_pattern, 'module_size': aligned.module_size,
        'padding_px': aligned.padding_px, 'quiet_zone_modules': aligned.quiet_zone_modules,
        **validation, **quality, **prompt_features, **image_quality_metrics(image),
        **image_change_metrics(image, reference),
        **aligned_module_diagnostics(image, aligned),
        'parameters': config, 'quality_error': quality_error,
        **(extra or {}),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / f'validations-{variant}.json').write_text(
        json.dumps(records, indent=2), encoding='utf-8'
    )
    return row


print('VRAM au départ :', cuda_memory_gib())
if cuda_memory_gib()['free_driver'] < 15:
    raise RuntimeError('Moins de 15 Gio libres : arrêter vLLM puis redémarrer le kernel.')
"""
    ),
    markdown("## 4. Moteur A — DiffQRCoder SD 1.5, Stage 1 + Stage 2 SRPG"),
    code(
        """class SD15Runner:
    model_name = 'sd15_diffqrcoder'

    def __init__(self):
        started = time.perf_counter()
        controlnet = ControlNetModel.from_pretrained(
            SD15_CONTROLNET, subfolder=SD15_CONTROLNET_SUBFOLDER,
            torch_dtype=torch.float16, cache_dir='/cache/huggingface',
        )
        self.pipe = DiffQRCoderPipeline.from_single_file(
            SD15_BASE, controlnet=controlnet, torch_dtype=torch.float16,
            cache_dir='/cache/huggingface', safety_checker=None, use_safetensors=True,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to('cuda')
        for component in [self.pipe.unet, self.pipe.controlnet, self.pipe.vae, self.pipe.text_encoder]:
            component.requires_grad_(False).eval()
        self.pipe.enable_attention_slicing('max')
        self.pipe.enable_vae_slicing()
        self.pipe.unet.enable_gradient_checkpointing()
        self.pipe.controlnet.enable_gradient_checkpointing()
        self.load_s = time.perf_counter() - started
        print(f'SD1.5 chargé en {self.load_s:.1f}s', cuda_memory_gib())

    @torch.no_grad()
    def _stage2_initial(self, stage1_tensor, seed, steps):
        normalized = stage1_tensor.to('cuda', dtype=torch.float16) * 2 - 1
        encoded = self.pipe.vae.encode(normalized).latent_dist.mode()
        encoded = encoded * self.pipe.vae.config.scaling_factor
        generator = torch.Generator(device='cuda').manual_seed(seed)
        noise = torch.randn(encoded.shape, generator=generator, device='cuda', dtype=encoded.dtype)
        self.pipe.scheduler.set_timesteps(steps, device='cuda')
        return self.pipe.scheduler.add_noise(encoded, noise, self.pipe.scheduler.timesteps[:1])

    @torch.no_grad()
    def generate(self, case, config, aligned, output_dir, *, save_frames):
        output_dir.mkdir(parents=True, exist_ok=True)
        stage1_dir = output_dir / 'stage1-frames'
        cb1, trace1 = diffusion_callback(
            self.pipe, aligned, f"{case['id']} / {config['name']} / stage1",
            config['stage1_steps'], stage1_dir, save_frames,
        )
        started = time.perf_counter()
        stage1 = self.pipe._run_stage1(
            prompt=case['text'], qrcode=aligned.image,
            height=aligned.canvas_size, width=aligned.canvas_size,
            negative_prompt=NEGATIVE_PROMPTS[config['negative_profile']],
            num_inference_steps=config['stage1_steps'], guidance_scale=config['cfg'],
            generator=torch.Generator(device='cuda').manual_seed(case['seed']),
            controlnet_conditioning_scale=config['control'],
            control_guidance_start=config.get('control_start', 0.0),
            control_guidance_end=config.get('control_end', 1.0),
            callback_on_step_end=cb1, callback_on_step_end_tensor_inputs=['latents'],
            output_type='pt',
        )
        stage1_s = time.perf_counter() - started
        stage1_tensor = stage1.images.detach()
        stage1_image = self.pipe.image_processor.numpy_to_pil(
            self.pipe.image_processor.pt_to_numpy(stage1_tensor.detach())
        )[0].convert('RGB')
        stage1_image.save(output_dir / 'stage1.png')
        save_tensor(output_dir / 'stage1.safetensors', 'stage1', stage1_tensor)
        make_gif(stage1_dir, output_dir / 'stage1.gif')

        initial = self._stage2_initial(stage1_tensor, case['seed'] + 10000, config['steps'])
        save_tensor(output_dir / 'stage2-initial-latent.safetensors', 'latents', initial)
        stage2_dir = output_dir / 'stage2-frames'
        cb2, trace2 = diffusion_callback(
            self.pipe, aligned, f"{case['id']} / {config['name']} / stage2",
            config['steps'], stage2_dir, save_frames,
        )
        started = time.perf_counter()
        result = self.pipe._run_stage2(
            prompt=case['text'], qrcode=aligned.image,
            qrcode_module_size=aligned.module_size, qrcode_padding=aligned.padding_px,
            ref_image=stage1_tensor, height=aligned.canvas_size, width=aligned.canvas_size,
            negative_prompt=NEGATIVE_PROMPTS[config['negative_profile']],
            num_inference_steps=config['steps'], guidance_scale=config['cfg'],
            eta=config.get('eta', 0.0),
            generator=torch.Generator(device='cuda').manual_seed(case['seed'] + 10000),
            latents=initial.clone(), controlnet_conditioning_scale=config['control'],
            control_guidance_start=config.get('control_start', 0.0),
            control_guidance_end=config.get('control_end', 1.0),
            scanning_robust_guidance_scale=config['srg'],
            perceptual_guidance_scale=config['pg'],
            callback_on_step_end=cb2, callback_on_step_end_tensor_inputs=['latents'],
            output_type='latent',
        )
        stage2_s = time.perf_counter() - started
        latent = result.images.detach()
        image = decode_latent(self.pipe, latent)
        image.save(output_dir / 'base.png')
        save_tensor(output_dir / 'stage2-final-latent.safetensors', 'latents', latent)
        make_gif(stage2_dir, output_dir / 'stage2.gif')
        (output_dir / 'diffusion-traces.json').write_text(
            json.dumps({'stage1': trace1, 'stage2': trace2}, indent=2), encoding='utf-8'
        )
        return {
            'image': image, 'latent': latent, 'stage1_tensor': stage1_tensor,
            'stage1_image': stage1_image, 'initial': initial,
            'stage1_s': stage1_s, 'stage2_s': stage2_s,
            'duration_s': stage1_s + stage2_s,
        }

    @torch.no_grad()
    def upstream_refine(self, case, config, aligned, state, output_dir):
        # Rejoue exactement la Stage 2 avec le même latent initial et active les paramètres
        # par défaut du dépôt public : 20 itérations SGD, lr=0,1, référence Stage 1.
        frame_dir = output_dir / 'upstream-stage2-frames'
        cb, trace = diffusion_callback(
            self.pipe, aligned, f"{case['id']} / {config['name']} / upstream",
            config['steps'], frame_dir, SAVE_BASELINE_EVERY_STEP,
        )
        started = time.perf_counter()
        result = self.pipe._run_stage2(
            prompt=case['text'], qrcode=aligned.image,
            qrcode_module_size=aligned.module_size, qrcode_padding=aligned.padding_px,
            ref_image=state['stage1_tensor'],
            height=aligned.canvas_size, width=aligned.canvas_size,
            negative_prompt=NEGATIVE_PROMPTS[config['negative_profile']],
            num_inference_steps=config['steps'], guidance_scale=config['cfg'],
            eta=config.get('eta', 0.0),
            generator=torch.Generator(device='cuda').manual_seed(case['seed'] + 10000),
            latents=state['initial'].clone(),
            controlnet_conditioning_scale=config['control'],
            control_guidance_start=config.get('control_start', 0.0),
            control_guidance_end=config.get('control_end', 1.0),
            scanning_robust_guidance_scale=config['srg'],
            perceptual_guidance_scale=config['pg'],
            srmpgd_num_iteration=UPSTREAM_SRMPGD_ITERATIONS,
            srmpgd_lr=UPSTREAM_SRMPGD_LR,
            callback_on_step_end=cb, callback_on_step_end_tensor_inputs=['latents'],
            output_type='latent',
        )
        duration = time.perf_counter() - started
        latent = result.images.detach()
        image = decode_latent(self.pipe, latent)
        image.save(output_dir / 'upstream-srmpgd.png')
        save_tensor(output_dir / 'upstream-final-latent.safetensors', 'latents', latent)
        (output_dir / 'upstream-trace.json').write_text(
            json.dumps(trace, indent=2), encoding='utf-8'
        )
        make_gif(frame_dir, output_dir / 'upstream.gif')
        return image, latent, duration

    def close(self):
        release_gpu(self.pipe)
        self.pipe = None
"""
    ),
    markdown("## 5. Moteur B — SD 2.1 et DionTimmer QR ControlNet"),
    code(
        """class SD21Runner:
    model_name = 'sd21_dion'

    def __init__(self):
        started = time.perf_counter()
        controlnet = ControlNetModel.from_pretrained(
            SD21_CONTROLNET, torch_dtype=torch.float16, cache_dir='/cache/huggingface',
        )
        errors = []
        self.base_id = None
        for base_id in SD21_BASE_CANDIDATES:
            try:
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    base_id, controlnet=controlnet, torch_dtype=torch.float16,
                    cache_dir='/cache/huggingface', safety_checker=None,
                )
                self.base_id = base_id
                break
            except Exception as exc:
                errors.append(f'{base_id}: {type(exc).__name__}: {exc}')
        if self.base_id is None:
            raise RuntimeError('Aucune fondation SD2.1 chargeable:\\n' + '\\n'.join(errors))
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to('cuda')
        for component in [self.pipe.unet, self.pipe.controlnet, self.pipe.vae, self.pipe.text_encoder]:
            component.requires_grad_(False).eval()
        self.pipe.enable_attention_slicing('max')
        self.pipe.enable_vae_slicing()
        self.pipe.unet.enable_gradient_checkpointing()
        self.pipe.controlnet.enable_gradient_checkpointing()
        self.load_s = time.perf_counter() - started
        print(f'SD2.1 chargé en {self.load_s:.1f}s depuis {self.base_id}', cuda_memory_gib())

    @torch.no_grad()
    def generate(self, case, config, aligned, output_dir, *, save_frames):
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_dir = output_dir / 'text2img-frames'
        cb, trace = diffusion_callback(
            self.pipe, aligned, f"{case['id']} / {config['name']} / text2img",
            config['steps'], frame_dir, save_frames,
        )
        started = time.perf_counter()
        result = self.pipe(
            prompt=case['text'],
            negative_prompt=NEGATIVE_PROMPTS[config['negative_profile']],
            image=aligned.image, height=aligned.canvas_size, width=aligned.canvas_size,
            num_inference_steps=config['steps'], guidance_scale=config['cfg'],
            generator=torch.Generator(device='cuda').manual_seed(case['seed']),
            controlnet_conditioning_scale=config['control'],
            control_guidance_start=config.get('control_start', 0.0),
            control_guidance_end=config.get('control_end', 1.0),
            callback_on_step_end=cb, callback_on_step_end_tensor_inputs=['latents'],
            output_type='latent',
        )
        text2img_s = time.perf_counter() - started
        latent = result.images.detach()
        image = decode_latent(self.pipe, latent)
        image.save(output_dir / 'text2img.png')
        save_tensor(output_dir / 'text2img-latent.safetensors', 'latents', latent)
        make_gif(frame_dir, output_dir / 'text2img.gif')

        rescue_s = 0.0
        if config.get('rescue_strength', 0) > 0:
            img2img = StableDiffusionControlNetImg2ImgPipeline(**self.pipe.components)
            rescue_dir = output_dir / 'rescue-frames'
            cb2, trace2 = diffusion_callback(
                img2img, aligned, f"{case['id']} / {config['name']} / rescue",
                config['steps'], rescue_dir, save_frames,
            )
            started = time.perf_counter()
            rescued = img2img(
                prompt=case['text'],
                negative_prompt=NEGATIVE_PROMPTS[config['negative_profile']],
                image=image, control_image=aligned.image,
                width=aligned.canvas_size, height=aligned.canvas_size,
                strength=config['rescue_strength'],
                num_inference_steps=config['steps'], guidance_scale=config['cfg'],
                controlnet_conditioning_scale=max(config['control'], 1.5),
                control_guidance_start=config.get('control_start', 0.0),
                control_guidance_end=config.get('control_end', 1.0),
                generator=torch.Generator(device='cuda').manual_seed(case['seed'] + 10000),
                callback_on_step_end=cb2, callback_on_step_end_tensor_inputs=['latents'],
                output_type='latent',
            )
            rescue_s = time.perf_counter() - started
            latent = rescued.images.detach()
            image = decode_latent(img2img, latent)
            image.save(output_dir / 'rescue.png')
            save_tensor(output_dir / 'rescue-latent.safetensors', 'latents', latent)
            make_gif(rescue_dir, output_dir / 'rescue.gif')
            trace = {'text2img': trace, 'rescue': trace2}
            del img2img
        else:
            trace = {'text2img': trace}

        image.save(output_dir / 'base.png')
        save_tensor(output_dir / 'final-latent.safetensors', 'latents', latent)
        (output_dir / 'diffusion-traces.json').write_text(
            json.dumps(trace, indent=2), encoding='utf-8'
        )
        return {
            'image': image, 'latent': latent, 'stage1_image': image,
            'text2img_s': text2img_s, 'rescue_s': rescue_s,
            'duration_s': text2img_s + rescue_s,
        }

    def close(self):
        release_gpu(self.pipe)
        self.pipe = None
"""
    ),
    markdown("## 6. SR-MPGD papier, état par état, avec protection esthétique"),
    code(
        """def run_paper_refinement(runner, case, config, aligned, state, output_dir, *, save_frames=True):
    frame_dir = output_dir / 'paper-srmpgd-frames'
    frame_dir.mkdir(parents=True, exist_ok=True)
    validations = {}
    paper_config = replace(
        PAPER_SRMPGD,
        max_iterations=config.get('paper_iterations', PAPER_SRMPGD.max_iterations),
        step_size=config.get('paper_step_size', PAPER_SRMPGD.step_size),
        lpips_weight=config.get('paper_lpips_weight', PAPER_SRMPGD.lpips_weight),
        crop_padding_px=aligned.padding_px,
        dark_threshold=config.get('dark_threshold', PAPER_SRMPGD.dark_threshold),
        light_threshold=config.get('light_threshold', PAPER_SRMPGD.light_threshold),
        center_fraction=config.get('center_fraction', PAPER_SRMPGD.center_fraction),
    )
    official_srl = ScanningRobustLoss(module_size=aligned.module_size).to(
        device='cuda', dtype=torch.float32
    ).requires_grad_(False).eval()

    def scanning_loss(image_core, target_core):
        return official_srl(image_core, target_core)

    def validate_iteration(image, iteration):
        values, records = validation_payload(image)
        validations[str(iteration)] = records
        return values

    def preview_iteration(image, step):
        if save_frames:
            image.save(frame_dir / f'{step.iteration:03d}.jpg', quality=90)
        clear_output(wait=True)
        display(Markdown(
            f"**{case['id']} / {config['name']} / paper SR-MPGD "
            f"{step.iteration}/{paper_config.max_iterations} — "
            f"{step.passed}/{step.total}, LPIPS {step.lpips_loss:.4f}, "
            f"MER {step.actual_module_error_rate:.2%}**"
        ))
        display(image.resize((430, 430)))

    result = run_srmpgd(
        runner.pipe, state['latent'].float(), aligned.core_blueprint, paper_config,
        scanning_loss=scanning_loss,
        validation_callback=validate_iteration,
        preview_callback=preview_iteration,
    )
    result.image.save(output_dir / 'paper-srmpgd.png')
    save_tensor(output_dir / 'paper-srmpgd-latent.safetensors', 'latents', result.latent)
    (output_dir / 'paper-srmpgd-trace.json').write_text(
        json.dumps([asdict(step) for step in result.steps], indent=2), encoding='utf-8'
    )
    (output_dir / 'paper-srmpgd-validations.json').write_text(
        json.dumps(validations, indent=2), encoding='utf-8'
    )
    (output_dir / 'paper-srmpgd-summary.json').write_text(
        json.dumps({
            'selected_iteration': result.selected_iteration,
            'stop_reason': result.stop_reason,
            'duration_s': result.duration_s,
            'initial_module_error_rate': result.initial_module_error_rate,
            'final_module_error_rate': result.final_module_error_rate,
        }, indent=2),
        encoding='utf-8',
    )
    make_gif(frame_dir, output_dir / 'paper-srmpgd.gif')
    release_gpu(official_srl)
    return result
"""
    ),
    markdown("## 7. Baseline appariée : 4 prompts, 6 recettes, toutes les variantes"),
    code(
        """def geometry_for(config, *, ecc=QR_ECC_BASELINE, mask=QR_MASK_BASELINE):
    module_size = config.get('module_size', QR_MODULE_SIZE)
    key = (config['canvas'], module_size, ecc, mask)
    if key not in GEOMETRIES:
        GEOMETRIES[key] = generate_aligned_qr(
            PAYLOAD, version=QR_VERSION, error_correction=ecc, mask_pattern=mask,
            module_size=module_size, canvas_size=config['canvas'],
        )
    return GEOMETRIES[key]


def baseline_key(case, config, variant):
    return ('baseline', case['id'], config['model'], config['name'], variant, case['seed'])


torch.cuda.reset_peak_memory_stats()
if RUN_BASELINE:
    for model_name, runner_type in [
        ('sd15_diffqrcoder', SD15Runner),
        ('sd21_dion', SD21Runner),
    ]:
        configs = [item for item in BASELINE_CONFIGS if item['model'] == model_name]
        runner = runner_type()
        for case in PROMPTS:
            for config in configs:
                aligned = geometry_for(config)
                output_dir = RUN_DIR / 'baseline' / case['id'] / config['name']
                index = result_index()
                required_variants = (
                    ['base', 'paper_srmpgd', 'upstream_srmpgd']
                    if model_name == 'sd15_diffqrcoder'
                    else ['base', 'paper_srmpgd']
                )
                if all(baseline_key(case, config, variant) in index for variant in required_variants):
                    print('Déjà terminé :', case['id'], config['name'])
                    continue

                state = runner.generate(
                    case, config, aligned, output_dir,
                    save_frames=SAVE_BASELINE_EVERY_STEP,
                )
                if baseline_key(case, config, 'base') not in result_index():
                    base_row = evaluate_candidate(
                        'baseline', case, config, 'base', state['image'], aligned,
                        state['duration_s'], state['stage1_image'], output_dir,
                        {'model_load_s': runner.load_s, 'peak_vram_gib': cuda_memory_gib()['peak_allocated']},
                    )
                    append_jsonl(RESULTS_PATH, base_row)

                if baseline_key(case, config, 'paper_srmpgd') not in result_index():
                    paper = run_paper_refinement(
                        runner, case, config, aligned, state, output_dir, save_frames=True
                    )
                    paper_row = evaluate_candidate(
                        'baseline', case, config, 'paper_srmpgd', paper.image, aligned,
                        state['duration_s'] + paper.duration_s, state['image'], output_dir,
                        {
                            'refinement_s': paper.duration_s,
                            'selected_iteration': paper.selected_iteration,
                            'stop_reason': paper.stop_reason,
                            'paper_gamma': PAPER_SRMPGD.step_size,
                            'paper_lpips_weight': PAPER_SRMPGD.lpips_weight,
                        },
                    )
                    append_jsonl(RESULTS_PATH, paper_row)
                    del paper

                if (
                    model_name == 'sd15_diffqrcoder'
                    and baseline_key(case, config, 'upstream_srmpgd') not in result_index()
                ):
                    upstream_image, upstream_latent, upstream_s = runner.upstream_refine(
                        case, config, aligned, state, output_dir
                    )
                    upstream_row = evaluate_candidate(
                        'baseline', case, config, 'upstream_srmpgd',
                        upstream_image, aligned, state['stage1_s'] + upstream_s,
                        state['image'], output_dir,
                        {
                            'refinement_s': upstream_s,
                            'upstream_iterations': UPSTREAM_SRMPGD_ITERATIONS,
                            'upstream_lr': UPSTREAM_SRMPGD_LR,
                            'upstream_reference': 'Stage 1 public implementation',
                        },
                    )
                    append_jsonl(RESULTS_PATH, upstream_row)
                    del upstream_latent

                del state
                gc.collect()
                torch.cuda.empty_cache()
        runner.close()
        del runner
        gc.collect()
        torch.cuda.empty_cache()

baseline_rows = [row for row in jsonl_rows(RESULTS_PATH) if row['phase'] == 'baseline']
print('Baseline persistée :', len(baseline_rows))
display(pd.DataFrame(baseline_rows)[[
    'prompt_id', 'profile', 'variant', 'passed', 'total', 'strict_all',
    'module_error_rate', 'clip_aesthetic', 'clip_score', 'duration_s',
]])
"""
    ),
    markdown("## 8. Comparaison visuelle et agrégats avant optimisation"),
    code(
        """baseline_frame = pd.DataFrame(baseline_rows)
if baseline_frame.empty:
    raise RuntimeError('La baseline est vide.')

aggregate = (
    baseline_frame.groupby(['model', 'profile', 'variant'], dropna=False)
    .agg(
        mean_pass_rate=('pass_rate', 'mean'),
        worst_pass_rate=('pass_rate', 'min'),
        strict_prompts=('strict_all', 'sum'),
        mean_original=('original_pass_rate', 'mean'),
        mean_aesthetic=('clip_aesthetic', 'mean'),
        mean_clip=('clip_score', 'mean'),
        mean_duration_s=('duration_s', 'mean'),
        mean_mer=('module_error_rate', 'mean'),
    )
    .reset_index()
    .sort_values(
        ['strict_prompts', 'worst_pass_rate', 'mean_pass_rate', 'mean_aesthetic', 'mean_clip'],
        ascending=False,
    )
)
aggregate.to_csv(RUN_DIR / 'baseline-aggregates.csv', index=False)
display(aggregate)

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
labels = [f"{r.profile}\\n{r.variant}" for r in aggregate.itertuples()]
positions = np.arange(len(labels))
for axis, column, title in [
    (axes[0, 0], 'mean_pass_rate', 'SSR logiciel moyen'),
    (axes[0, 1], 'worst_pass_rate', 'Pire SSR parmi les prompts'),
    (axes[1, 0], 'mean_aesthetic', 'CLIP-aesthetic moyen'),
    (axes[1, 1], 'mean_duration_s', 'Temps moyen (s)'),
]:
    axis.bar(positions, aggregate[column].fillna(0))
    axis.set_title(title)
    axis.set_xticks(positions, labels, rotation=70, fontsize=8)
    axis.grid(axis='y', alpha=0.25)
axes[0, 0].axhline(1, color='red', linestyle='--')
axes[0, 1].axhline(1, color='red', linestyle='--')
fig.tight_layout()
fig.savefig(RUN_DIR / 'baseline-metrics.png', dpi=160, bbox_inches='tight')
display(fig)

variants = [
    ('sd15_744_40', 'base'), ('sd15_744_40', 'paper_srmpgd'),
    ('sd15_744_40', 'upstream_srmpgd'), ('sd21_768_50', 'base'),
    ('sd21_768_50', 'paper_srmpgd'),
]
fig, axes = plt.subplots(len(PROMPTS), len(variants), figsize=(20, 16))
for row_index, case in enumerate(PROMPTS):
    for column_index, (profile, variant) in enumerate(variants):
        row = next(
            item for item in baseline_rows
            if item['prompt_id'] == case['id']
            and item['profile'] == profile and item['variant'] == variant
        )
        filename = {
            'base': 'base.png',
            'paper_srmpgd': 'paper-srmpgd.png',
            'upstream_srmpgd': 'upstream-srmpgd.png',
        }[variant]
        image = Image.open(RUN_DIR / 'baseline' / case['id'] / profile / filename).convert('RGB')
        axes[row_index, column_index].imshow(image)
        axes[row_index, column_index].axis('off')
        axes[row_index, column_index].set_title(
            f"{case['id']}\\n{profile}/{variant}\\n"
            f"{row['passed']}/{row['total']} | aes={row['clip_aesthetic']}",
            fontsize=8,
        )
fig.tight_layout()
fig.savefig(RUN_DIR / 'baseline-comparison.png', dpi=150, bbox_inches='tight')
display(fig)
"""
    ),
    markdown(
        """## 9. Recherche Optuna contrainte et contextuelle

Chaque famille est optimisée séparément pour ne charger qu'un modèle en VRAM. Les objectifs sont :

1. maximiser le taux de validation ;
2. maximiser CLIP-aesthetic ;
3. maximiser CLIPScore ;
4. minimiser le temps.

La contrainte `1 - pass_rate <= 0` signifie qu'une candidate non 26/26 ne peut pas devenir la « meilleure recette » uniquement grâce à son aspect. Les prompts tournent entre les essais ; le contexte est enregistré pour le futur sélecteur.
"""
    ),
    code(
        """def suggest_config(trial, model_name):
    if model_name == 'sd15_diffqrcoder':
        canvas = trial.suggest_categorical('canvas', [744, 768])
        return {
            'name': f"optuna-{model_name}-{trial.number:04d}",
            'model': model_name,
            'canvas': canvas,
            'module_size': trial.suggest_categorical('module_size', [16, 20]),
            'steps': trial.suggest_int('steps', 30, 110, step=10),
            'stage1_steps': trial.suggest_int('stage1_steps', 30, 60, step=10),
            'cfg': trial.suggest_float('cfg', 5.0, 12.0, step=0.5),
            'control': trial.suggest_float('control', 0.9, 1.8, step=0.05),
            'srg': trial.suggest_float('srg', 300.0, 1000.0, step=50.0),
            'pg': trial.suggest_float('pg', 0.5, 5.0, step=0.25),
            'rescue_strength': 0.0,
            'eta': trial.suggest_categorical('eta', [0.0, 0.1, 0.3]),
            'negative_profile': trial.suggest_categorical(
                'negative_profile', list(NEGATIVE_PROMPTS)
            ),
            'control_start': trial.suggest_categorical(
                'control_start', [0.0, 0.05, 0.1]
            ),
            'control_end': trial.suggest_categorical('control_end', [0.8, 0.9, 1.0]),
            'ecc': trial.suggest_categorical('ecc', ['M', 'Q', 'H']),
            'mask': trial.suggest_int('mask', 0, 7),
            'refinement': trial.suggest_categorical('refinement', ['none', 'paper']),
            'paper_iterations': trial.suggest_categorical(
                'paper_iterations', [5, 10, 20]
            ),
            'paper_step_size': trial.suggest_categorical(
                'paper_step_size', [100.0, 300.0, 1000.0]
            ),
            'paper_lpips_weight': trial.suggest_categorical(
                'paper_lpips_weight', [0.01, 0.03, 0.1]
            ),
            'dark_threshold': trial.suggest_float(
                'dark_threshold', 0.35, 0.50, step=0.05
            ),
            'light_threshold': trial.suggest_float(
                'light_threshold', 0.55, 0.70, step=0.05
            ),
            'center_fraction': trial.suggest_categorical(
                'center_fraction', [0.25, 1 / 3, 0.5]
            ),
        }
    return {
        'name': f"optuna-{model_name}-{trial.number:04d}",
        'model': model_name,
        'canvas': 768,
        'module_size': trial.suggest_categorical('module_size', [16, 20]),
        'steps': trial.suggest_int('steps', 40, 150, step=10),
        'stage1_steps': 0,
        'cfg': trial.suggest_float('cfg', 7.0, 20.0, step=0.5),
        'control': trial.suggest_float('control', 1.0, 2.2, step=0.05),
        'srg': 0.0,
        'pg': 0.0,
        'rescue_strength': trial.suggest_categorical('rescue_strength', [0.0, 0.25, 0.35, 0.45]),
        'eta': 0.0,
        'negative_profile': trial.suggest_categorical(
            'negative_profile', list(NEGATIVE_PROMPTS)
        ),
        'control_start': trial.suggest_categorical('control_start', [0.0, 0.05, 0.1]),
        'control_end': trial.suggest_categorical('control_end', [0.8, 0.9, 1.0]),
        'ecc': trial.suggest_categorical('ecc', ['M', 'Q', 'H']),
        'mask': trial.suggest_int('mask', 0, 7),
        'refinement': trial.suggest_categorical('refinement', ['none', 'paper']),
        'paper_iterations': trial.suggest_categorical('paper_iterations', [5, 10, 20]),
        'paper_step_size': trial.suggest_categorical(
            'paper_step_size', [100.0, 300.0, 1000.0]
        ),
        'paper_lpips_weight': trial.suggest_categorical(
            'paper_lpips_weight', [0.01, 0.03, 0.1]
        ),
        'dark_threshold': trial.suggest_float('dark_threshold', 0.35, 0.50, step=0.05),
        'light_threshold': trial.suggest_float('light_threshold', 0.55, 0.70, step=0.05),
        'center_fraction': trial.suggest_categorical(
            'center_fraction', [0.25, 1 / 3, 0.5]
        ),
    }


def search_one_model(model_name, runner_type):
    storage = f"sqlite:///{RUN_DIR / f'optuna-{model_name}.sqlite3'}"

    def constraints_func(trial):
        return trial.user_attrs.get('constraints', [1.0])

    sampler = optuna.samplers.TPESampler(
        seed=20260723, multivariate=True, group=True,
        constraints_func=constraints_func,
    )
    study = optuna.create_study(
        study_name=f'{EXPERIMENT_NAME}-{model_name}',
        storage=storage, load_if_exists=True,
        directions=['maximize', 'maximize', 'maximize', 'minimize'],
        sampler=sampler,
    )
    runner = runner_type()

    def objective(trial):
        config = suggest_config(trial, model_name)
        context_index = trial.number % len(PROMPTS)
        case = dict(PROMPTS[context_index])
        seed_index = trial.suggest_int('seed_index', 0, 2)
        case['seed'] += seed_index * 100_000
        ecc, mask = config['ecc'], config['mask']
        aligned = geometry_for(config, ecc=ecc, mask=mask)
        output_dir = RUN_DIR / 'search' / model_name / f'trial-{trial.number:04d}'
        state = runner.generate(
            case, config, aligned, output_dir,
            save_frames=SAVE_SEARCH_EVERY_STEP,
        )
        image = state['image']
        duration = state['duration_s']
        variant = 'base'
        refinement_s = 0.0
        refinement_metadata = {}
        if config['refinement'] == 'paper':
            try:
                paper = run_paper_refinement(
                    runner, case, config, aligned, state, output_dir, save_frames=False
                )
                image = paper.image
                refinement_s = paper.duration_s
                duration += refinement_s
                variant = 'paper_srmpgd'
                refinement_metadata = {
                    'selected_iteration': paper.selected_iteration,
                    'stop_reason': paper.stop_reason,
                    'numerical_refinement_fallback': paper.stop_reason.startswith(
                        'non_finite'
                    ),
                }
            except FloatingPointError as exc:
                # Défense supplémentaire pour une ancienne version de run_srmpgd ou une
                # autre branche numérique : la diffusion finie reste une observation valide.
                variant = 'base_after_failed_paper'
                refinement_metadata = {
                    'selected_iteration': 0,
                    'stop_reason': f'floating_point_error: {exc}',
                    'numerical_refinement_fallback': True,
                }
        row = evaluate_candidate(
            'search', case, config, variant, image, aligned, duration,
            state['image'], output_dir,
            {
                'trial_number': trial.number, 'seed_index': seed_index,
                'refinement_s': refinement_s, 'peak_vram_gib': cuda_memory_gib()['peak_allocated'],
                **refinement_metadata,
            },
        )
        append_jsonl(SEARCH_RESULTS_PATH, row)
        trial.set_user_attr('constraints', [1.0 - row['pass_rate']])
        trial.set_user_attr('row_key', list(result_key(row)))
        trial.set_user_attr('original_pass_rate', row['original_pass_rate'])
        trial.set_user_attr('worst_decoder_pass_rate', row['worst_decoder_pass_rate'])
        del state
        gc.collect()
        torch.cuda.empty_cache()
        return (
            row['pass_rate'],
            row['clip_aesthetic'] if row['clip_aesthetic'] is not None else -100.0,
            row['clip_score'] if row['clip_score'] is not None else -100.0,
            row['duration_s'],
        )

    completed_count = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, OPTUNA_TRIALS_PER_MODEL - completed_count)
    try:
        if remaining:
            study.optimize(
                objective,
                n_trials=remaining,
                gc_after_trial=True,
                catch=(FloatingPointError,),
            )
    finally:
        runner.close()
        gc.collect()
        torch.cuda.empty_cache()
    return study


studies = {}
if RUN_OPTUNA:
    studies['sd15_diffqrcoder'] = search_one_model('sd15_diffqrcoder', SD15Runner)
    studies['sd21_dion'] = search_one_model('sd21_dion', SD21Runner)

search_rows = jsonl_rows(SEARCH_RESULTS_PATH)
print('Essais de recherche persistés :', len(search_rows))
"""
    ),
    markdown("## 10. Pareto, importance des paramètres et généralisation sur les 4 prompts"),
    code(
        """search_frame = pd.DataFrame(search_rows)
if not search_frame.empty:
    search_frame.to_csv(RUN_DIR / 'search-results.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for axis, model_name in zip(axes, ['sd15_diffqrcoder', 'sd21_dion']):
        part = search_frame[search_frame.model == model_name]
        scatter = axis.scatter(
            part.pass_rate, part.clip_aesthetic,
            c=part.clip_score, s=np.where(part.strict_all, 150, 35),
            cmap='viridis', alpha=0.8,
        )
        axis.axvline(1.0, color='red', linestyle='--')
        axis.set_title(model_name)
        axis.set_xlabel('SSR robuste')
        axis.set_ylabel('CLIP-aesthetic')
        axis.grid(alpha=0.25)
        fig.colorbar(scatter, ax=axis, label='CLIPScore')
    fig.tight_layout()
    fig.savefig(RUN_DIR / 'search-objectives.png', dpi=160)
    display(fig)

    importance_all = {}
    from optuna.importance import get_param_importances
    for model_name, study in studies.items():
        completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 8:
            importance_all[model_name] = {'warning': 'moins de 8 essais complets'}
            continue
        try:
            importance_all[model_name] = get_param_importances(
                study, target=lambda trial: trial.values[0]
            )
        except Exception as exc:
            importance_all[model_name] = {'error': f'{type(exc).__name__}: {exc}'}
    (RUN_DIR / 'parameter-importance.json').write_text(
        json.dumps(importance_all, indent=2), encoding='utf-8'
    )
    display(importance_all)


def search_rank(row):
    return candidate_rank(row)


confirmation_rows = []
if RUN_CONFIRMATION and search_rows:
    for model_name, runner_type in [
        ('sd15_diffqrcoder', SD15Runner),
        ('sd21_dion', SD21Runner),
    ]:
        candidates = sorted(
            [row for row in search_rows if row['model'] == model_name],
            key=search_rank, reverse=True,
        )
        unique = []
        seen = set()
        for row in candidates:
            signature = json.dumps(row['parameters'], sort_keys=True)
            if signature not in seen:
                unique.append(row)
                seen.add(signature)
            if len(unique) == 3:
                break
        runner = runner_type()
        for source in unique:
            config = dict(source['parameters'])
            config['name'] = f"confirm-{source['profile']}"
            for case in PROMPTS[:CONFIRMATION_CONTEXTS]:
                aligned = geometry_for(
                    config, ecc=config.get('ecc', QR_ECC_BASELINE),
                    mask=config.get('mask', QR_MASK_BASELINE),
                )
                output_dir = RUN_DIR / 'confirmation' / model_name / source['profile'] / case['id']
                state = runner.generate(case, config, aligned, output_dir, save_frames=False)
                image = state['image']
                duration = state['duration_s']
                variant = 'base'
                refinement_metadata = {}
                if config.get('refinement') == 'paper':
                    paper = run_paper_refinement(
                        runner, case, config, aligned, state, output_dir, save_frames=False
                    )
                    image, duration, variant = (
                        paper.image, duration + paper.duration_s, 'paper_srmpgd'
                    )
                    refinement_metadata = {
                        'selected_iteration': paper.selected_iteration,
                        'stop_reason': paper.stop_reason,
                        'numerical_refinement_fallback': paper.stop_reason.startswith(
                            'non_finite'
                        ),
                    }
                row = evaluate_candidate(
                    'confirmation', case, config, variant, image, aligned, duration,
                    state['image'], output_dir,
                    {
                        'source_trial': source.get('trial_number'),
                        **refinement_metadata,
                    },
                )
                append_jsonl(RESULTS_PATH, row)
                confirmation_rows.append(row)
                if config.get('refinement') == 'paper':
                    del paper
                del state
                gc.collect()
                torch.cuda.empty_cache()
        runner.close()
        del runner

if not confirmation_rows:
    confirmation_rows = [
        row for row in jsonl_rows(RESULTS_PATH) if row['phase'] == 'confirmation'
    ]
print('Confirmations :', len(confirmation_rows))
"""
    ),
    markdown("## 11. Dataset de politique et mini-modèle CatBoost avec garde anti-surapprentissage"),
    code(
        """policy_rows = baseline_rows + search_rows + confirmation_rows
policy_frame = pd.json_normalize(policy_rows, sep='__')
policy_frame.to_csv(RUN_DIR / 'policy-dataset.csv', index=False)
with (RUN_DIR / 'policy-dataset.jsonl').open('w', encoding='utf-8') as stream:
    for row in policy_rows:
        stream.write(json.dumps(row, ensure_ascii=False) + '\\n')

POLICY_MIN_ROWS = 100
POLICY_MIN_POSITIVES = 12
policy_status = {
    'rows': len(policy_frame),
    'strict_positives': int(policy_frame.strict_all.sum()) if not policy_frame.empty else 0,
    'trained': False,
    'reason': None,
}

if len(policy_frame) < POLICY_MIN_ROWS:
    policy_status['reason'] = (
        f'{len(policy_frame)} lignes seulement ; minimum {POLICY_MIN_ROWS}. '
        'Les données sont exportées mais aucun modèle trompeur n est entraîné.'
    )
elif policy_frame.strict_all.sum() < POLICY_MIN_POSITIVES:
    policy_status['reason'] = (
        f'{int(policy_frame.strict_all.sum())} succès stricts seulement ; '
        f'minimum {POLICY_MIN_POSITIVES}.'
    )
else:
    from catboost import CatBoostClassifier, Pool
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    feature_columns = [
        'prompt', 'model', 'variant', 'canvas', 'ecc', 'mask_pattern', 'module_size',
        'parameters__steps', 'parameters__stage1_steps', 'parameters__cfg',
        'parameters__control', 'parameters__srg', 'parameters__pg',
        'parameters__rescue_strength', 'parameters__refinement',
        'parameters__eta', 'parameters__negative_profile',
        'parameters__control_start', 'parameters__control_end',
        'parameters__paper_iterations', 'parameters__paper_step_size',
        'parameters__paper_lpips_weight', 'parameters__dark_threshold',
        'parameters__light_threshold', 'parameters__center_fraction',
    ] + [f'prompt_embedding_{index:02d}' for index in range(16)]
    available = [column for column in feature_columns if column in policy_frame.columns]
    X = policy_frame[available].copy()
    categorical = [
        column for column in ['prompt', 'model', 'variant', 'ecc', 'parameters__refinement']
        if column in available
    ]
    for column in categorical:
        X[column] = X[column].fillna('missing').astype(str)
    for column in set(available) - set(categorical):
        X[column] = X[column].fillna(-1)
    y = policy_frame.strict_all.astype(int)
    groups = policy_frame.prompt_id
    model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.04, loss_function='Logloss',
        eval_metric='PRAUC', random_seed=20260723, verbose=False,
        auto_class_weights='Balanced',
    )
    splitter = GroupKFold(n_splits=min(4, groups.nunique()))
    probabilities = cross_val_predict(
        model, X, y, groups=groups, cv=splitter, method='predict_proba',
        params={'cat_features': categorical},
    )[:, 1]
    metrics = {
        'average_precision': average_precision_score(y, probabilities),
        'brier': brier_score_loss(y, probabilities),
        'roc_auc': roc_auc_score(y, probabilities),
        'validation': 'GroupKFold by prompt_id; no prompt leakage',
    }
    model.fit(Pool(X, y, cat_features=categorical))
    model.save_model(str(RUN_DIR / 'prooftag-parameter-selector.cbm'))
    policy_status.update({'trained': True, 'features': available, 'metrics': metrics})

(RUN_DIR / 'policy-status.json').write_text(
    json.dumps(policy_status, indent=2), encoding='utf-8'
)
print(policy_status)
"""
    ),
    markdown(
        """## 12. Porte de livraison et budget de tentatives

Il n'existe pas une recette fixe garantissant tous les prompts. La stratégie production est :

1. prédire/ranker quelques configurations ;
2. générer des seeds différents ;
3. tester chaque image ;
4. livrer le premier 26/26 ayant la meilleure qualité ;
5. rejeter le lot si le budget est épuisé.

Sous une approximation d'indépendance, `1-(1-p)^N` estime la probabilité d'au moins un succès. Le notebook donne le budget mesuré, mais le plafonne pour éviter une boucle infinie.
"""
    ),
    code(
        """all_observations = baseline_rows + search_rows + confirmation_rows
strict_rate = (
    sum(row['strict_all'] for row in all_observations) / len(all_observations)
    if all_observations else 0.0
)
required = attempts_for_target(strict_rate, DELIVERY_TARGET)
budget = MAX_DELIVERY_ATTEMPTS if required is None else min(required, MAX_DELIVERY_ATTEMPTS)
delivery_math = {
    'observed_single_attempt_probability': strict_rate,
    'target_probability': DELIVERY_TARGET,
    'independence_required_attempts': required,
    'operational_cap': MAX_DELIVERY_ATTEMPTS,
    'selected_budget': budget,
    'estimated_probability_at_budget': delivery_probability(strict_rate, budget),
    'warning': 'independence is an approximation; confirmation across prompts and physical tests remains mandatory',
}
(RUN_DIR / 'delivery-budget.json').write_text(
    json.dumps(delivery_math, indent=2), encoding='utf-8'
)
print(delivery_math)

selected = {}
for case in PROMPTS:
    candidates = [
        row for row in all_observations
        if row['prompt_id'] == case['id'] and row['strict_all']
    ]
    selected[case['id']] = deliverable_candidate(candidates)
    print(case['id'], 'DELIVERABLE' if selected[case['id']] else 'REJECTED', selected[case['id']])
"""
    ),
    markdown("## 13. Manifeste, journal d'erreurs, tests physiques et archive"),
    code(
        """all_results = jsonl_rows(RESULTS_PATH) + search_rows
manifest = {
    'experiment': EXPERIMENT_NAME,
    'created_at': datetime.now(timezone.utc).isoformat(),
    'payload_sha256': hashlib.sha256(PAYLOAD.encode()).hexdigest(),
    'diffqrcoder_commit': DIFFQRCODER_COMMIT,
    'models': {
        'sd15_base': SD15_BASE,
        'sd15_controlnet': f'{SD15_CONTROLNET}/{SD15_CONTROLNET_SUBFOLDER}',
        'sd21_base_candidates': SD21_BASE_CANDIDATES,
        'sd21_controlnet': SD21_CONTROLNET,
    },
    'geometry': {
        'version': QR_VERSION, 'module_size': QR_MODULE_SIZE,
        'formula': 'padding=(canvas - core_modules*module_size)/2; no QR resize',
        'audit': geometry_audit,
    },
    'baseline_configs': BASELINE_CONFIGS,
    'paper_srmpgd': asdict(PAPER_SRMPGD),
    'upstream_srmpgd': {
        'iterations': UPSTREAM_SRMPGD_ITERATIONS, 'lr': UPSTREAM_SRMPGD_LR,
        'reference': 'Stage 1; public repository behavior',
    },
    'optimization': {
        'trials_per_model': OPTUNA_TRIALS_PER_MODEL,
        'sampler': 'constrained multivariate group TPESampler',
        'directions': ['scan max', 'CLIP-aesthetic max', 'CLIPScore max', 'time min'],
        'constraint': '1 - strict robust pass rate <= 0',
    },
    'policy': policy_status,
    'delivery_budget': delivery_math,
    'peak_gpu_memory_gib': torch.cuda.max_memory_allocated() / 2**30,
    'software': {
        name: importlib.metadata.version(name)
        for name in ['diffusers', 'transformers', 'optuna', 'catboost', 'torchvision']
    },
    'result_count': len(all_results),
}
(RUN_DIR / 'manifest.json').write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
)

with (RUN_DIR / 'physical-validation.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow([
        'prompt_id', 'profile', 'variant', 'device', 'medium', 'distance_cm',
        'angle_deg', 'lighting_lux', 'print_size_mm', 'attempt', 'success',
        'latency_s', 'decoded_payload_sha256', 'notes',
    ])
    for case in PROMPTS:
        chosen = selected[case['id']]
        profile = chosen['profile'] if chosen else 'none'
        variant = chosen['variant'] if chosen else 'none'
        for device in ['Pixel 7', 'iPhone 13', 'industrial_scanner']:
            for medium in ['screen', 'matte_print', 'glossy_print']:
                for attempt in range(1, 11):
                    writer.writerow([
                        case['id'], profile, variant, device, medium,
                        '', '', '', '', attempt, '', '', '', '',
                    ])

report = [
    '# Rapport automatique E013', '',
    f'- Résultats enregistrés : {len(all_results)}',
    f'- Succès stricts : {sum(row["strict_all"] for row in all_results)}/{len(all_results)}',
    f'- Pic VRAM : {manifest["peak_gpu_memory_gib"]:.2f} Gio',
    f'- Taux strict observé : {strict_rate:.2%}',
    f'- Budget théorique pour {DELIVERY_TARGET:.1%} : {required}',
    f'- Budget opérationnel plafonné : {budget}', '',
    '## Décisions', '',
    '- La géométrie exacte est une condition préalable, pas un hyperparamètre esthétique.',
    '- SD 1.5 et SD 2.1 ne partagent pas le même ControlNet ; les checkpoints ne sont pas interchangeables.',
    '- Le SR-MPGD papier et celui du dépôt public sont rapportés séparément.',
    '- Un résultat non strict reste une donnée de recherche, jamais une image livrable.',
    '- Le sélecteur CatBoost n est entraîné que si le volume et le nombre de succès sont suffisants.', '',
]
for case in PROMPTS:
    chosen = selected[case['id']]
    report.append(
        f"- {case['id']}: "
        + (
            f"DELIVERABLE {chosen['profile']}/{chosen['variant']} "
            f"{chosen['passed']}/{chosen['total']}"
            if chosen else 'REJECTED — aucune candidate 26/26'
        )
    )
(RUN_DIR / 'run-report.md').write_text('\\n'.join(report) + '\\n', encoding='utf-8')

shutil.copy2(
    '/workspace/notebooks/10_exact_geometry_sd15_sd21_policy.ipynb',
    RUN_DIR / '10_exact_geometry_sd15_sd21_policy.ipynb',
)
archive = Path(shutil.make_archive(
    str(RUN_DIR), 'gztar', root_dir=RUN_DIR.parent, base_dir=RUN_DIR.name
))
archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
print('Archive :', archive)
print('SHA-256:', archive_hash)
print("Serveur: POD=$(kubectl get pod -n qr-core -l app=prooftag-qr-notebook -o jsonpath='{.items[0].metadata.name}')")
print(f'Serveur: kubectl cp -n qr-core "${{POD}}:{archive}" "$HOME/{archive.name}"')
print(f'Windows: scp paul@pcIA:~/{archive.name} "$HOME/Downloads/"')
"""
    ),
    markdown(
        """## Interprétation autorisée

- Les deux fondations sont réellement différentes : DiffQRCoder utilise SD 1.5 + QR Monster v2 ; la seconde branche utilise SD 2.1 + le ControlNet SD 2.1 de DionTimmer.
- L'usage SD 2.1 suit la résolution recommandée de 768 px et teste aussi le sauvetage img2img, mais ne prétend pas être DiffQRCoder porté sur SD 2.1.
- Les images de chaque pas de baseline sont conservées. La recherche n'enregistre que les points d'observation pour limiter le disque et le surcoût de décodage VAE.
- Un 26/26 logiciel n'est pas une garantie universelle. Les impressions et téléphones doivent remplir `physical-validation.csv`.
- Le mini-modèle ne crée pas de garantie : il réduit le nombre de mauvais essais. La garantie opérationnelle vient de la validation et du rejet des échecs.
- Les probabilités multi-tentatives supposent des essais approximativement indépendants. Des seeds et configurations diversifiés sont donc nécessaires.
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

target = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "10_exact_geometry_sd15_sd21_policy.ipynb"
)
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(target)
