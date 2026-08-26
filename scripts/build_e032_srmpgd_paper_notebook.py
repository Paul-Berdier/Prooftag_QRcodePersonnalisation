"""Build the resumable, visual E032 SR-MPGD paper-reconstruction notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "27_e032_srmpgd_paper_reconstruction.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    markdown(
        """# E032 — reconstruction visuelle et appariée du SR-MPGD du papier

E032 vérifie enfin le mécanisme décrit par les équations 12 à 14, sans le confondre avec le
SR-MPGD sécurisé de production. La campagne gèle **5 prompts simples + 5 atypiques**, trois seeds
et quatre sorties appariées : Stage 1, Stage 2 SRPG « papier », SR-MPGD `paper_equations`, puis
le même latent Stage 2 traité par `guarded_production`.

Le profil papier est désactivé dans le schéma Web Lab pour éviter une utilisation accidentelle ;
ce notebook l'active explicitement dans sa requête immuable. Un contexte prompt/seed forme une
petite campagne autonome. Après une coupure, une campagne déjà terminée et son export valide ne
sont **jamais relancés** ; au pire, seul le contexte interrompu est refait.

Les rasters finaux et les checkpoints SR-MPGD 0, 1, 2, 4, 8, 12 et 20 sont téléchargés, affichés
et archivés. QR-Verify, MER, saturation, LPIPS, CLIP-Aesthetic, CLIPScore et HPS v2.1 sont publiés.
Les métriques CLIP/HPS des quatre sorties finales viennent de l'API épinglée. Leur calcul sur les
210 checkpoints intermédiaires est optionnel et reprenable, car il peut durer plusieurs heures
sur CPU.

**Limite :** le QArt exact privé des auteurs n'est pas disponible. E032 reconstruit les équations
avec la cible QArt publique disponible dans Prooftag. QR-Verify reste un test logiciel et non une
probabilité de scan téléphone. Aucune sortie E032 n'est automatiquement livrable.
"""
    ),
    markdown("## 0. Imports et utilitaires persistants"),
    code(
        """# ruff: noqa: E402
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import time
from collections import deque
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image as NotebookImage
from IPython.display import Markdown, clear_output, display
from PIL import Image

from prooftag_qr.advisor_gallery import (
    download_advisor_gallery,
    render_advisor_contact_sheet,
    write_gallery_index,
)
from prooftag_qr.advisor_inference import (
    AdvisorInferencePlan,
    AdvisorInferenceRunner,
    load_advisor_inference_results,
)
from prooftag_qr.e030_offline import sha256_file
from prooftag_qr.quality import image_quality_metrics
from prooftag_qr.quality_scoring import CLIPQualityScorer
from prooftag_qr.schemas import LabCampaignCreate
from prooftag_qr.validation import ConservativeQRVerifyScorer, image_raster_sha256


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str))


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + '\\n')
        stream.flush()
        os.fsync(stream.fileno())


def canonical_sha256(value):
    body = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str
    ).encode('utf-8')
    return hashlib.sha256(body).hexdigest()


def api_json(path):
    request = Request(
        f"{COLLECTION_API_URL.rstrip('/')}{path}",
        headers={'Accept': 'application/json'},
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode('utf-8'))


def api_bytes(path):
    request = Request(f"{COLLECTION_API_URL.rstrip('/')}{path}")
    with urlopen(request, timeout=120) as response:
        return response.read()


def finite(value):
    if value is None or str(value).strip() == '':
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def valid_png(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as source:
            source.verify()
        return True
    except Exception:
        return False


def download_png(path, endpoint):
    path = Path(path)
    if valid_png(path):
        return path
    body = api_bytes(endpoint)
    image = Image.open(BytesIO(body)).convert('RGB')
    temporary = path.with_suffix('.png.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(temporary, format='PNG', optimize=True)
    os.replace(temporary, path)
    if not valid_png(path):
        raise RuntimeError(f'PNG téléchargé invalide : {path}')
    return path


print('Notebook CPU : orchestration, validation et galeries. API Kubernetes : génération GPU.')
"""
    ),
    markdown("## 1. Protocole gelé : prompts, seeds, métriques et checkpoints"),
    code(
        """EXPERIMENT = 'e032-srmpgd-paper-reconstruction-v1'
COLLECTION_API_URL = 'http://prooftag-qr-svc.qr-core.svc.cluster.local:8080'
COLLECTION_PAYLOAD = os.environ.get('PROOFTAG_E032_PAYLOAD', 'https://ptag.io/t/e032')
ERROR_CORRECTION = 'M'
SEEDS = [51_001, 62_017, 73_133]
CHECKPOINT_ITERATIONS = [0, 1, 2, 4, 8, 12, 20]
RUN_E032 = True
QR_VERIFY_REPETITIONS = 3
RUN_INTERMEDIATE_CLIP_HPS = False
POLL_SECONDS = 15.0

OUTPUT_ROOT = Path('/data/e032-srmpgd-paper')
QR_VERIFY_CACHE = Path('/data/qr-verify-conservative-cache')
DOWNLOAD_ROOT = Path('/workspace/downloads')
ARCHIVE_ROOT = Path('/data/e032-srmpgd-paper-archives')
for directory in [OUTPUT_ROOT, QR_VERIFY_CACHE, DOWNLOAD_ROOT, ARCHIVE_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    {
        'id': 'e032_simple_tea_room',
        'family': 'simple',
        'text': (
            'a serene ceramic tea room in soft morning light, centered composition, '
            'highly detailed'
        ),
    },
    {
        'id': 'e032_simple_snow_fox',
        'family': 'simple',
        'text': (
            'a red fox standing in a snowy pine forest, cinematic natural light, '
            'detailed illustration'
        ),
    },
    {
        'id': 'e032_simple_lighthouse',
        'family': 'simple',
        'text': 'a lighthouse on a rocky coast at sunset, dramatic clouds, painterly realism',
    },
    {
        'id': 'e032_simple_library',
        'family': 'simple',
        'text': (
            'a cozy library reading nook with warm lamps and wooden shelves, '
            'symmetrical interior'
        ),
    },
    {
        'id': 'e032_simple_greenhouse',
        'family': 'simple',
        'text': (
            'a sunlit greenhouse filled with tomato plants and terracotta pots, '
            'botanical photograph'
        ),
    },
    {
        'id': 'e032_atypical_mobius_railway',
        'family': 'atypical',
        'text': (
            'a transparent glass Mobius railway floating above a coral desert, '
            'impossible architecture'
        ),
    },
    {
        'id': 'e032_atypical_jellyfish_cathedral',
        'family': 'atypical',
        'text': (
            'a clockwork jellyfish cathedral beneath a solar eclipse, '
            'baroque surrealism, intricate brass'
        ),
    },
    {
        'id': 'e032_atypical_cup_storm',
        'family': 'atypical',
        'text': (
            'an origami thunderstorm unfolding inside a porcelain teacup, '
            'macro studio photography'
        ),
    },
    {
        'id': 'e032_atypical_moon_observatory',
        'family': 'atypical',
        'text': (
            'a bioluminescent fungal observatory growing on a tiny moon, deep space, '
            'luminous details'
        ),
    },
    {
        'id': 'e032_atypical_copper_opera',
        'family': 'atypical',
        'text': (
            'a brutalist opera house woven from copper roots and black silk, '
            'architectural concept art'
        ),
    },
]
NEGATIVE_PROMPT = (
    'easynegative, low quality, worst quality, blurry, deformed, watermark, text, logo, '
    'oversaturated, clipped highlights, posterized colors'
)

assert len(PROMPTS) == 10
assert sum(item['family'] == 'simple' for item in PROMPTS) == 5
assert sum(item['family'] == 'atypical' for item in PROMPTS) == 5
assert len(SEEDS) == 3 and len(set(SEEDS)) == 3
assert CHECKPOINT_ITERATIONS == [0, 1, 2, 4, 8, 12, 20]
print(
    'Plan brut :', len(PROMPTS), 'prompts ×', len(SEEDS),
    'seeds × 4 méthodes = 120 rasters finaux',
)
"""
    ),
    markdown("## 2. Préflight API et construction des quatre méthodes exactement appariées"),
    code(
        """schema = api_json('/v1/lab/schema')
api_runtime = api_json('/v1/runtime')
notebook_commit = os.environ.get('PROOFTAG_GIT_COMMIT', '').strip().lower()
notebook_image = os.environ.get('PROOFTAG_RUNTIME_IMAGE', '').strip()
notebook_image_digest = os.environ.get(
    'PROOFTAG_RUNTIME_IMAGE_DIGEST', ''
).strip().lower()
if not re.fullmatch(r'[0-9a-f]{40}', notebook_commit):
    raise RuntimeError('Commit notebook absent : déployer E032 avec une image versionnée.')
if not notebook_image.endswith(f':{notebook_commit[:12]}'):
    raise RuntimeError(f'Image notebook non liée au commit : {notebook_image!r}')
if not re.fullmatch(r'sha256:[0-9a-f]{64}', notebook_image_digest):
    raise RuntimeError('Digest OCI notebook absent ou invalide.')
api_identity = api_runtime.get('deployment_identity') or {}
if api_identity.get('configured') is not True:
    raise RuntimeError('Identité de déploiement API absente.')
if api_identity.get('git_commit') != notebook_commit:
    raise RuntimeError(
        f"Commit API différent du notebook : {api_identity.get('git_commit')!r}"
    )
api_image = str(api_identity.get('image') or '')
api_image_digest = str(api_identity.get('image_digest') or '').lower()
if not api_image.endswith(f':{notebook_commit[:12]}'):
    raise RuntimeError(f'Image API non liée au commit : {api_image!r}')
if not re.fullmatch(r'sha256:[0-9a-f]{64}', api_image_digest):
    raise RuntimeError('Digest OCI API absent ou invalide.')
runtime_binding = {
    'git_commit': notebook_commit,
    'notebook_image': notebook_image,
    'notebook_image_digest': notebook_image_digest,
    'api_image': api_image,
    'api_image_digest': api_image_digest,
    'diffqrcoder_revision': schema['notes']['upstream_revision'],
}
profiles = {profile['id']: profile for profile in schema['profiles']}
required_profiles = {
    'diffqrcoder_stage1', 'diffqrcoder_paper_srpg',
    'diffqrcoder_paper_srmpgd_guarded', 'diffqrcoder_paper_srmpgd',
}
missing_profiles = required_profiles - profiles.keys()
if missing_profiles:
    raise RuntimeError(f'API trop ancienne, profils E032 absents : {sorted(missing_profiles)}')
for profile_id in [
    'diffqrcoder_paper_srmpgd_guarded', 'diffqrcoder_paper_srmpgd',
]:
    if profiles[profile_id]['enabled'] is not False:
        raise RuntimeError(
            f'Le profil {profile_id} doit rester désactivé par défaut dans le schéma.'
        )
if schema['validation'].get('engine') != 'antfu/qr-verify@0.2.0':
    raise RuntimeError(f"Moteur QR-Verify inattendu : {schema['validation']}")
quality_contract = schema.get('quality_scoring', {})
if not quality_contract.get('clip_enabled') or not quality_contract.get('hpsv2_1_enabled'):
    raise RuntimeError(
        'E032 exige CLIP/CLIP-Aesthetic et HPS v2.1 actifs dans l API. '
        f'Contrat observé : {quality_contract}'
    )
if quality_contract.get('failure_policy') != 'fail_closed':
    raise RuntimeError('E032 exige les métriques de qualité API en mode fail-closed.')
quality_plan_binding = {
    'clip_enabled': quality_contract['clip_enabled'],
    'hpsv2_1_enabled': quality_contract['hpsv2_1_enabled'],
    'failure_policy': quality_contract['failure_policy'],
    # metrics contient les identifiants/révisions attendus, mais aucun état
    # dynamique dépendant du chargement à chaud des modèles dans l'API.
    'metrics': quality_contract['metrics'],
}

stage1 = deepcopy(profiles['diffqrcoder_stage1'])
paper_srpg = deepcopy(profiles['diffqrcoder_paper_srpg'])
guarded_srmpgd = deepcopy(profiles['diffqrcoder_paper_srmpgd_guarded'])
paper_srmpgd = deepcopy(profiles['diffqrcoder_paper_srmpgd'])
for method in [stage1, paper_srpg, guarded_srmpgd, paper_srmpgd]:
    method.pop('description', None)
    method['enabled'] = True
paper_srpg['require_exact_stage1_reuse'] = True
guarded_srmpgd['require_exact_stage1_reuse'] = True
paper_srmpgd['require_exact_stage1_reuse'] = True

# Ces deux profils proviennent du schéma versionné. Ils ont le même Stage 2,
# gamma et LPIPS ; seul le protocole de descente et ses gardes diffèrent.
METHODS = [stage1, paper_srpg, guarded_srmpgd, paper_srmpgd]
METHOD_IDS = [method['id'] for method in METHODS]
assert METHOD_IDS == [
    'diffqrcoder_stage1', 'diffqrcoder_paper_srpg',
    'diffqrcoder_paper_srmpgd_guarded', 'diffqrcoder_paper_srmpgd',
]
assert paper_srmpgd['tools']['settings']['srmpgd_protocol'] == 'paper_equations'
assert paper_srmpgd['tools']['settings']['srmpgd_step_size'] == 1000.0
assert paper_srmpgd['tools']['settings']['srmpgd_lpips_weight'] == 0.01
assert guarded_srmpgd['tools']['settings']['srmpgd_protocol'] == 'guarded_production'
assert guarded_srmpgd['tools']['settings']['srmpgd_max_initial_module_error_rate'] == 1.0
assert paper_srmpgd['tools']['settings']['srmpgd_crop_padding_px'] == 78
assert guarded_srmpgd['tools']['settings']['srmpgd_crop_padding_px'] == 78
# Le VAE produit 736 px. QR v3 sans quiet zone contient 29 modules de 20 px.
assert 736 - 2 * 78 == 29 * 20

stage2_cache_fields = [
    'srpg_steps', 'diffqrcoder_stage2_strength',
    'diffqrcoder_stage2_initialization', 'srpg_controlnet_scale',
    'srpg_qr_weight', 'srpg_perceptual_weight', 'srpg_eta',
    'srpg_seed_offset', 'diffqrcoder_control_guidance_start',
    'diffqrcoder_control_guidance_end', 'diffqrcoder_stage2_target_mode',
    'diffqrcoder_qart_thresholds',
]
paper_stage2_signature = {
    key: paper_srpg['tools']['settings'].get(key) for key in stage2_cache_fields
}
for method in [guarded_srmpgd, paper_srmpgd]:
    candidate_signature = {
        key: method['tools']['settings'].get(key) for key in stage2_cache_fields
    }
    if candidate_signature != paper_stage2_signature:
        raise RuntimeError(
            f"{method['id']} ne partage pas exactement la configuration Stage 2 parent."
        )

display(pd.DataFrame([{
    'id': method['id'],
    'sortie': method['output_variant'],
    'Stage 2': method['tools'].get('srpg_enabled', False),
    'SR-MPGD': method['tools'].get('srmpgd_enabled', False),
    'protocole': method['tools']['settings'].get('srmpgd_protocol', '-'),
    'gamma': method['tools']['settings'].get('srmpgd_step_size', '-'),
    'LPIPS λ': method['tools']['settings'].get('srmpgd_lpips_weight', '-'),
    'réutilisation exacte': method.get('require_exact_stage1_reuse', False),
} for method in METHODS]))
display(pd.DataFrame([runtime_binding]).T.rename(columns={0: 'identité liée au plan'}))
"""
    ),
    markdown("## 3. Figer le plan : 30 petites campagnes reprenables"),
    code(
        """prompt_requests = [{
    'id': item['id'],
    'text': item['text'],
    'negative_prompt': NEGATIVE_PROMPT,
} for item in PROMPTS]
prompt_by_id = {item['id']: item for item in PROMPTS}

plan_material = {
    'experiment': EXPERIMENT,
    'payload_sha256': hashlib.sha256(COLLECTION_PAYLOAD.encode('utf-8')).hexdigest(),
    'payload_length': len(COLLECTION_PAYLOAD),
    'error_correction': ERROR_CORRECTION,
    'prompts': PROMPTS,
    'negative_prompt': NEGATIVE_PROMPT,
    'seeds': SEEDS,
    'methods': METHODS,
    'checkpoints': CHECKPOINT_ITERATIONS,
    'validation': schema['validation'],
    'quality_scoring': quality_plan_binding,
    'upstream_revision': schema['notes']['upstream_revision'],
    'runtime_binding': runtime_binding,
}
plan_id = canonical_sha256(plan_material)[:16]
campaigns = []
for prompt in prompt_requests:
    for seed in SEEDS:
        request = LabCampaignCreate.model_validate({
            'name': f"E032 {plan_id} {prompt['id']} seed-{seed}",
            'payload': COLLECTION_PAYLOAD,
            'error_correction': ERROR_CORRECTION,
            'prompts': [prompt],
            'seeds': [seed],
            'methods': METHODS,
            'max_attempts': 1,
        })
        campaigns.append(request.model_dump(mode='json'))

pipeline_states = {
    'diffqrcoder_stage1': 'stage1',
    'diffqrcoder_paper_srpg': 'stage2_parent',
    'diffqrcoder_paper_srmpgd_guarded': 'guarded_srmpgd',
    'diffqrcoder_paper_srmpgd': 'paper_srmpgd',
}
predictions = tuple({
    'prompt_id': prompt['id'],
    'prompt_text': prompt['text'],
    'prompt_family': prompt_by_id[prompt['id']]['family'],
    'plan_method_id': method['id'],
    'source_method_id': method['id'],
    'requested_source_output_variant': method['output_variant'],
    'pipeline_state': pipeline_states[method['id']],
    'role': 'e032_paired_ablation',
    'advisor_rank': index + 1,
} for prompt in prompt_requests for index, method in enumerate(METHODS))
public_plan = {
    **plan_material,
    'plan_id': plan_id,
    'campaign_count': len(campaigns),
    'trial_count': len(campaigns) * len(METHODS),
    'campaigns': [{
        'name': item['name'],
        'prompt_id': item['prompts'][0]['id'],
        'seed': item['seeds'][0],
        'method_ids': [method['id'] for method in item['methods']],
    } for item in campaigns],
}
# Le payload clair est uniquement dans AdvisorInferencePlan.payload et les requêtes
# en mémoire. plan-redacted.json ne contient que son hash et sa longueur.
public_plan.pop('payload', None)
plan = AdvisorInferencePlan(
    plan_id=plan_id,
    payload=COLLECTION_PAYLOAD,
    campaigns=tuple(campaigns),
    predictions=predictions,
    public=public_plan,
)
assert len(plan.campaigns) == 30
assert plan.public['trial_count'] == 120
assert all(len(item['prompts']) == len(item['seeds']) == 1 for item in plan.campaigns)
display(pd.DataFrame([{
    'plan': plan.plan_id,
    'campagnes reprenables': len(plan.campaigns),
    'prompts': len(PROMPTS),
    'seeds': len(SEEDS),
    'méthodes': len(METHODS),
    'rasters finaux': plan.public['trial_count'],
}]))
"""
    ),
    markdown(
        """## 4. Exécuter ou reprendre sans régénérer les campagnes terminées

Le runner lie chaque campagne à ce plan, vérifie son payload haché et sa spécification complète,
puis vérifie le CSV exporté. Une entrée `completed` avec export valide est sautée. Le nombre
d'essais distant est fixé à un : un échec reste visible au lieu d'être masqué par une nouvelle
campagne automatique.
"""
    ),
    code(
        """events = deque(maxlen=30)
started = time.monotonic()


def progress(event):
    events.append(event)
    clear_output(wait=True)
    latest = events[-1]
    display(Markdown('### Progression E032'))
    display(pd.DataFrame([{
        'événement': latest.get('event'),
        'campagne': latest.get('index'),
        'état': latest.get('status', 'running'),
        'essais': f"{latest.get('completed_trials', 0)}/{latest.get('total_trials', 0)}",
        'acceptés API': latest.get('accepted_trials', 0),
        'méthode': latest.get('current_method_id'),
        'seed': latest.get('current_seed'),
        'temps (h)': round((time.monotonic() - started) / 3600, 2),
    }]))
    display(pd.DataFrame(list(events)[-10:]))


runner = AdvisorInferenceRunner(
    plan=plan,
    api_url=COLLECTION_API_URL,
    output_root=OUTPUT_ROOT,
    poll_seconds=POLL_SECONDS,
    maximum_campaign_attempts=1,
    reject_campaigns_with_errors=True,
    stop_on_first_failed_campaign=True,
    progress_callback=progress,
)
print('Plan persistant :', runner.output_dir)
print('État de reprise :', runner.state_path)
runner_summary = runner.run() if RUN_E032 else runner.summary()
display(pd.DataFrame([runner_summary]).T.rename(columns={0: 'valeur'}))
"""
    ),
    markdown(
        """## 4 bis. Diagnostic obligatoire avant toute exception

Le runner exporte une campagne même lorsqu'elle se termine avec une vraie erreur. Cette cellule lit
donc **tous** les CSV, y compris `completed_with_errors`, avant de décider si l'analyse scientifique
peut continuer. Elle affiche les messages complets, télécharge chaque raster déjà produit et crée
une archive de diagnostic. Elle ne soumet aucune campagne et ne relance aucune génération GPU.
"""
    ),
    code(
        """RUN_DIR = runner.output_dir / 'analysis'
RUN_DIR.mkdir(parents=True, exist_ok=True)
diagnostic_dir = RUN_DIR / 'campaign-diagnostic'
diagnostic_dir.mkdir(parents=True, exist_ok=True)

diagnostic_rows = []
for export_path in sorted(runner.exports_dir.glob('*.csv')):
    with export_path.open('r', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            diagnostic_rows.append({**row, 'export_path': str(export_path)})
if not diagnostic_rows:
    raise RuntimeError('Aucun export E032 disponible : impossible de diagnostiquer sans CSV.')

diagnostic_frame = pd.DataFrame(diagnostic_rows)
diagnostic_frame.to_csv(
    diagnostic_dir / 'e032-all-trials-diagnostic.csv', index=False
)
status_table = pd.crosstab(
    diagnostic_frame['method_id'], diagnostic_frame['status'], margins=True
)
display(Markdown('### Statuts réels par méthode'))
display(status_table)

error_mask = (
    diagnostic_frame['status'].eq('error')
    | diagnostic_frame['error'].fillna('').astype(str).str.strip().ne('')
)
error_frame = diagnostic_frame.loc[error_mask].copy()
error_columns = [
    'campaign_id', 'prompt_id', 'seed', 'method_id', 'status', 'error',
    'generation_run_id', 'export_path',
]
display(Markdown('### Erreurs complètes — aucune troncature'))
with pd.option_context(
    'display.max_colwidth', None,
    'display.max_rows', max(200, len(error_frame) + 5),
):
    display(error_frame[error_columns].sort_values(['method_id', 'prompt_id', 'seed']))

if error_frame.empty:
    error_groups = pd.DataFrame(columns=['method_id', 'status', 'error', 'count'])
else:
    error_groups = (
        error_frame.groupby(['method_id', 'status', 'error'], dropna=False)
        .size()
        .rename('count')
        .reset_index()
        .sort_values(['count', 'method_id'], ascending=[False, True])
    )
display(Markdown('### Signatures d erreur regroupées'))
with pd.option_context('display.max_colwidth', None, 'display.max_rows', 200):
    display(error_groups)

history_statuses = pd.DataFrame(runner.state.get('history', []))
campaign_status_counts = (
    history_statuses['status'].value_counts(dropna=False).to_dict()
    if not history_statuses.empty else {}
)
diagnostic_report = {
    'plan_id': plan.plan_id,
    'runner_summary': runner_summary,
    'campaign_status_counts': campaign_status_counts,
    'trial_status_by_method': {
        str(method_id): {
            str(status): int(count)
            for status, count in values.items()
        }
        for method_id, values in diagnostic_frame.groupby('method_id')['status']
        .value_counts().unstack(fill_value=0).to_dict('index').items()
    },
    'error_groups': error_groups.fillna('').to_dict('records'),
}
diagnostic_json = diagnostic_dir / 'e032-error-diagnostic.json'
atomic_json(diagnostic_json, diagnostic_report)

successful = diagnostic_frame[
    diagnostic_frame['status'].isin(['accepted', 'rejected'])
    & diagnostic_frame['generation_run_id'].fillna('').astype(str).str.strip().ne('')
].copy()
success_entries = []
for row in successful.to_dict('records'):
    qr_success = finite(row.get('quality_qr_verify_any_exact'))
    if qr_success is None:
        exact_text = str(row.get('exact_payload_match') or '').strip().lower()
        qr_success = 1.0 if exact_text in {'1', '1.0', 'true'} else 0.0
    qr_tolerance = finite(row.get('quality_qr_verify_tolerance_score'))
    if qr_tolerance is None:
        qr_tolerance = finite(row.get('scan_pass_rate'))
    saturation_values = [
        value for value in (
            finite(row.get('quality_high_saturation_pixel_ratio')),
            finite(row.get('quality_rgb_clipped_channel_ratio')),
        ) if value is not None
    ]
    success_entries.append({
        'section': 'completed_output',
        'campaign_id': row['campaign_id'],
        'trial_id': row['trial_id'],
        'prompt_id': row['prompt_id'],
        'prompt_text': row['prompt_text'],
        'method_id': row['method_id'],
        'output_variant': row.get('selected_variant'),
        'seed': int(row['seed']),
        'generation_run_id': row['generation_run_id'],
        'status': row['status'],
        'qr_success': qr_success,
        'qr_tolerance': qr_tolerance,
        'clip_aesthetic': finite(row.get('quality_clip_aesthetic')),
        'clip_score': finite(row.get('quality_clip_score')),
        'hpsv2_1': finite(row.get('quality_hpsv2_1')),
        'saturation_risk': max(saturation_values) if saturation_values else None,
        'error': row.get('error'),
    })

diagnostic_gallery = download_advisor_gallery(
    success_entries,
    api_url=COLLECTION_API_URL,
    output_dir=diagnostic_dir / 'successful-images',
)
write_gallery_index(diagnostic_gallery, diagnostic_dir)
download_failures = [row for row in diagnostic_gallery if row.get('download_error')]
if download_failures:
    display(Markdown('### Images réussies devenues indisponibles côté API'))
    with pd.option_context('display.max_colwidth', None):
        display(pd.DataFrame(download_failures))

sheet_dir = diagnostic_dir / 'successful-contact-sheets'
sheet_dir.mkdir(exist_ok=True)
successful_sheets = []
grouped_success = {}
for entry in diagnostic_gallery:
    grouped_success.setdefault(
        (entry['prompt_id'], int(entry['seed'])), []
    ).append(entry)
for (prompt_id, seed), entries in sorted(grouped_success.items()):
    entries.sort(key=lambda item: METHOD_IDS.index(item['method_id']))
    sheet_path = sheet_dir / f'{prompt_id}-seed-{seed}.png'
    render_advisor_contact_sheet(
        entries,
        title=f'E032 sorties produites — {prompt_id} — seed {seed}',
        output_path=sheet_path,
        columns=4,
    )
    successful_sheets.append(sheet_path)

display(Markdown(f'### Rasters déjà produits : {len(diagnostic_gallery)}'))
for sheet_path in successful_sheets:
    display(NotebookImage(filename=str(sheet_path), width=1150))

diagnostic_archive = DOWNLOAD_ROOT / f'{plan.plan_id}-e032-diagnostic.tar.gz'
temporary_diagnostic_archive = diagnostic_archive.with_suffix('.tar.gz.tmp')
with tarfile.open(temporary_diagnostic_archive, 'w:gz') as bundle:
    bundle.add(diagnostic_dir, arcname=f'{plan.plan_id}-e032-diagnostic')
os.replace(temporary_diagnostic_archive, diagnostic_archive)
print('Diagnostic CSV :', diagnostic_dir / 'e032-all-trials-diagnostic.csv')
print('Diagnostic JSON :', diagnostic_json)
print('Archive téléchargeable :', diagnostic_archive)

if runner_summary['status'] != 'completed':
    raise RuntimeError(
        'E032 contient de vraies erreurs de trial. Les campagnes ne sont pas relancées. '
        'Lire le tableau ci-dessus et transmettre l archive de diagnostic.'
    )
"""
    ),
    markdown("## 5. Matrice finale, qualité et preuve de l'appariement exact"),
    code(
        """rows = load_advisor_inference_results(runner.output_dir)
frame = pd.DataFrame(rows)
if len(frame) != 120:
    raise RuntimeError(f'Matrice E032 incomplète : {len(frame)}/120.')

metric_columns = ['clip_aesthetic', 'clip_score', 'hpsv2_1']
for metric in metric_columns:
    missing = frame[frame[metric].map(finite).isna()]
    if not missing.empty:
        display(missing[['trial_id', 'prompt_id', 'method_id', metric]])
        raise RuntimeError(f'{metric} absent/non fini sur {len(missing)} sorties finales.')

pairing_rows = []
for (prompt_id, seed), group in frame.groupby(['prompt_id', 'seed'], sort=True):
    by_method = {row.method_id: row for row in group.itertuples(index=False)}
    if set(by_method) != set(METHOD_IDS):
        raise RuntimeError(f'Contexte incomplet {prompt_id}/{seed}: {sorted(by_method)}')
    parent = by_method['diffqrcoder_paper_srpg']
    for method_id in METHOD_IDS[1:]:
        row = by_method[method_id]
        stage1_ok = finite(row.stage1_reused) == 1.0 and bool(row.stage1_source_run_id)
        if method_id == 'diffqrcoder_paper_srpg':
            stage2_ok = row.stage2_pairing_status == 'generated_source'
        else:
            stage2_ok = (
                finite(row.stage2_pairing_exact) == 1.0
                and row.stage2_source_run_id == parent.generation_run_id
                and row.stage2_source_latent_sha256 == row.stage2_latent_sha256
                and row.srmpgd_stage2_image_sha256 == parent.final_image_sha256
                and finite(row.srmpgd_iteration_zero_exact) == 1.0
            )
        pairing_rows.append({
            'prompt_id': prompt_id, 'seed': int(seed), 'method_id': method_id,
            'stage1_exact': stage1_ok, 'stage2_exact': stage2_ok,
            'stage2_source_run_id': row.stage2_source_run_id,
            'parent_run_id': parent.generation_run_id,
            'stage2_latent_sha256': row.stage2_latent_sha256,
            'stage2_parent_image_sha256': parent.final_image_sha256,
            'srmpgd_stage2_image_sha256': row.srmpgd_stage2_image_sha256,
            'srmpgd_iteration_zero_exact': row.srmpgd_iteration_zero_exact,
        })
pairing = pd.DataFrame(pairing_rows)
pairing.to_csv(RUN_DIR / 'e032-pairing-audit.csv', index=False)
if not pairing[['stage1_exact', 'stage2_exact']].all(axis=None):
    display(pairing[~pairing[['stage1_exact', 'stage2_exact']].all(axis=1)])
    raise RuntimeError(
        'Appariement exact E032 non prouvé : aucune comparaison n est interprétable.'
    )

frame.to_csv(RUN_DIR / 'e032-final-results.csv', index=False)
display(frame[[
    'prompt_id', 'seed', 'method_id', 'status', 'qr_success', 'qr_tolerance',
    'module_error_rate', 'saturation_risk', 'clip_aesthetic', 'clip_score',
    'hpsv2_1', 'srmpgd_selected_iteration', 'duration_ms',
]].sort_values(['prompt_id', 'seed', 'method_id']).head(24))
print('Appariement exact prouvé :', len(pairing), '/', len(pairing))
"""
    ),
    markdown("## 6. Télécharger les 120 images finales et créer une planche par prompt/seed"),
    code(
        """final_gallery_dir = RUN_DIR / 'final-gallery'
gallery_source = [{**row, 'section': row['pipeline_state']} for row in rows]
gallery_entries = download_advisor_gallery(
    gallery_source,
    api_url=COLLECTION_API_URL,
    output_dir=final_gallery_dir / 'images',
)
missing_images = [entry for entry in gallery_entries if entry.get('download_error')]
if missing_images:
    display(pd.DataFrame(missing_images))
    raise RuntimeError(f'{len(missing_images)} images finales indisponibles.')
write_gallery_index(gallery_entries, final_gallery_dir)
gallery_by_key = {
    (entry['prompt_id'], int(entry['seed']), entry['method_id']): entry
    for entry in gallery_entries
}

# L'API expose son score de campagne, mais E032 recalcule aussi chaque raster
# final trois fois avec l'intersection conservatrice des 37 presets. Le journal
# est adressé par le contenu : une reprise ne rescane jamais un raster déjà fait.
final_qrverify_path = RUN_DIR / 'e032-final-qrverify-scores.jsonl'
final_qrverify_rows = {}
if final_qrverify_path.is_file():
    for line in final_qrverify_path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            item = json.loads(line)
            # E032 v1 final n'a jamais été exécuté avant l'introduction de
            # record_key. Refuser plutôt que fusionner un ancien journal ambigu.
            if 'record_key' not in item or 'measurement_key' not in item:
                raise RuntimeError(
                    'Journal final E032 ancien/ambigu : archiver le dossier de plan '
                    'avant de relancer avec le code actuel.'
                )
            final_qrverify_rows[item['record_key']] = item
final_qr_scorer = ConservativeQRVerifyScorer(
    repetitions=QR_VERIFY_REPETITIONS,
    cache_dir=QR_VERIFY_CACHE,
)
expected_final_record_keys = set()
for position, entry in enumerate(gallery_entries, start=1):
    with Image.open(entry['local_image']) as source:
        image = source.convert('RGB')
    raster_sha256 = image_raster_sha256(image)
    measurement_key = canonical_sha256({
        'image_raster_sha256': raster_sha256,
        'payload_sha256': hashlib.sha256(COLLECTION_PAYLOAD.encode()).hexdigest(),
        'qr_verify_repetitions': QR_VERIFY_REPETITIONS,
        'engine_version': final_qr_scorer.engine_version,
        'implementation_sha256': final_qr_scorer.implementation_sha256,
        'scoring_version': final_qr_scorer.scoring_version,
    })
    record_key = canonical_sha256({
        'measurement_key': measurement_key,
        'generation_run_id': entry['generation_run_id'],
        'method_id': entry['method_id'],
    })
    expected_final_record_keys.add(record_key)
    if record_key not in final_qrverify_rows:
        score = final_qr_scorer.score(image, COLLECTION_PAYLOAD).to_dict()
        score_row = {
            'measurement_key': measurement_key,
            'record_key': record_key,
            'generation_run_id': entry['generation_run_id'],
            'prompt_id': entry['prompt_id'],
            'seed': int(entry['seed']),
            'method_id': entry['method_id'],
            'image_raster_sha256': raster_sha256,
            'qr_verify_conservative_exact': score['consistent_any_exact'],
            'qr_verify_conservative_tolerance': score[
                'conservative_tolerance_score'
            ],
            'qr_verify_conservative_exact_presets': score[
                'conservative_exact_presets'
            ],
            'qr_verify_preset_count': score['preset_count'],
            'qr_verify_unstable_presets': score['unstable_preset_count'],
            'qr_verify_cache_key': score['cache_key'],
        }
        append_jsonl(final_qrverify_path, score_row)
        final_qrverify_rows[record_key] = score_row
    if position % 5 == 0 or position == len(gallery_entries):
        clear_output(wait=True)
        print(f'QR-Verify final conservateur : {position}/{len(gallery_entries)}')
final_qr_scorer.close()

final_score_frame = pd.DataFrame([
    row for key, row in final_qrverify_rows.items()
    if key in expected_final_record_keys
])
if len(final_score_frame) != len(gallery_entries):
    raise RuntimeError(
        f'Re-scoring final incomplet : {len(final_score_frame)}/{len(gallery_entries)}.'
    )
if final_score_frame.record_key.nunique() != len(gallery_entries):
    raise RuntimeError('Une sortie finale E032 ne possède pas son record_key unique.')
alias_audit = (
    final_score_frame.groupby('image_raster_sha256', as_index=False)
    .agg(outputs=('record_key', 'nunique'), measurements=('measurement_key', 'nunique'))
)
if not alias_audit[alias_audit.outputs > 1].empty:
    # Des sorties identiques sont légitimes (par exemple guarded no-op = parent),
    # mais chaque sortie doit rester une ligne distincte et partager une mesure.
    invalid_aliases = alias_audit[
        (alias_audit.outputs > 1) & (alias_audit.measurements != 1)
    ]
    if not invalid_aliases.empty:
        raise RuntimeError('Des rasters aliasés ne partagent pas la même mesure QR-Verify.')
alias_audit.to_csv(RUN_DIR / 'e032-final-raster-alias-audit.csv', index=False)
final_score_frame.to_csv(RUN_DIR / 'e032-final-qrverify-scores.csv', index=False)
frame = frame.merge(
    final_score_frame[[
        'generation_run_id', 'qr_verify_conservative_exact',
        'qr_verify_conservative_tolerance',
        'qr_verify_conservative_exact_presets', 'qr_verify_unstable_presets',
    ]],
    on='generation_run_id', how='left', validate='one_to_one',
)
frame.to_csv(RUN_DIR / 'e032-final-results-conservative.csv', index=False)
final_score_by_run = {
    row['generation_run_id']: row for row in final_score_frame.to_dict('records')
}
for entry in gallery_entries:
    entry.update(final_score_by_run[entry['generation_run_id']])


def title_for(entry):
    return (
        f"{entry['method_id']}\\n"
        f"QRV-3x={finite(entry.get('qr_verify_conservative_tolerance')) or 0:.1%}  "
        f"MER={finite(entry.get('module_error_rate')) or 0:.1%}\\n"
        f"AES={finite(entry.get('clip_aesthetic')) or 0:.2f}  "
        f"HPS={finite(entry.get('hpsv2_1')) or 0:.3f}"
    )


context_sheet_dir = RUN_DIR / 'context-contact-sheets'
context_sheet_dir.mkdir(exist_ok=True)
context_sheets = []
for prompt in PROMPTS:
    for seed in SEEDS:
        entries = [gallery_by_key[(prompt['id'], seed, method_id)] for method_id in METHOD_IDS]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.7))
        for axis, entry in zip(axes, entries):
            with Image.open(entry['local_image']) as source:
                axis.imshow(source.convert('RGB'))
            axis.set_title(title_for(entry), fontsize=8)
            axis.axis('off')
        fig.suptitle(f"{prompt['id']} — seed {seed} — {prompt['text']}", fontsize=11)
        fig.tight_layout()
        path = context_sheet_dir / f"{prompt['id']}-seed-{seed}.png"
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        context_sheets.append(path)

print('Planches finales :', len(context_sheets))
for path in context_sheets:
    display(NotebookImage(filename=str(path), width=1100))
"""
    ),
    markdown("## 7. Télécharger les checkpoints SR-MPGD et les traces, avec reprise par fichier"),
    code(
        """checkpoint_root = RUN_DIR / 'paper-srmpgd-checkpoints'
checkpoint_root.mkdir(exist_ok=True)
paper_rows = [row for row in rows if row['method_id'] == 'diffqrcoder_paper_srmpgd']
if len(paper_rows) != 30:
    raise RuntimeError(f'Sorties paper SR-MPGD incomplètes : {len(paper_rows)}/30.')

checkpoint_records = []
for position, row in enumerate(paper_rows, start=1):
    run_id = row['generation_run_id']
    context_dir = checkpoint_root / row['prompt_id'] / f"seed-{int(row['seed'])}"
    context_dir.mkdir(parents=True, exist_ok=True)
    artifact_catalog = api_json(f'/v1/generations/{run_id}/artifacts')
    artifact_urls = {item['name']: item['url'] for item in artifact_catalog}
    trace_path = context_dir / 'srmpgd-trace.json'
    if not trace_path.is_file():
        atomic_json(trace_path, api_json(f'/v1/generations/{run_id}/metadata/srmpgd_trace'))
    trace = json.loads(trace_path.read_text(encoding='utf-8'))
    trace_by_iteration = {int(item['iteration']): item for item in trace['steps']}
    for iteration in CHECKPOINT_ITERATIONS:
        name = f'srmpgd_iteration_{iteration:03d}'
        endpoint = artifact_urls.get(name) or artifact_urls.get(f'attempt_1_{name}')
        source_kind = 'variant'
        if endpoint is None and iteration == int(row['srmpgd_selected_iteration'] or -1):
            endpoint = f'/v1/generations/{run_id}/image'
            source_kind = 'final'
        if endpoint is None:
            checkpoint_records.append({
                'prompt_id': row['prompt_id'], 'prompt': row['prompt_text'],
                'seed': int(row['seed']), 'generation_run_id': run_id,
                'iteration': iteration, 'local_image': None,
                'download_error': 'checkpoint absent; arrêt anticipé ou artefact non publié',
                **trace_by_iteration.get(iteration, {}),
            })
            continue
        image_path = download_png(context_dir / f'iteration-{iteration:03d}.png', endpoint)
        checkpoint_records.append({
            'prompt_id': row['prompt_id'], 'prompt': row['prompt_text'],
            'prompt_family': row['prompt_family'], 'seed': int(row['seed']),
            'generation_run_id': run_id, 'iteration': iteration,
            'local_image': str(image_path), 'download_error': None,
            'artifact_source': source_kind,
            **trace_by_iteration.get(iteration, {}),
        })
    clear_output(wait=True)
    print(f'Checkpoints : {position}/{len(paper_rows)} contextes')

checkpoint_frame = pd.DataFrame(checkpoint_records)
checkpoint_frame.to_csv(RUN_DIR / 'e032-checkpoint-downloads.csv', index=False)
missing_checkpoints = checkpoint_frame[checkpoint_frame.local_image.isna()]
if not missing_checkpoints.empty:
    display(Markdown(
        f'**Attention : {len(missing_checkpoints)} checkpoints manquent.** '
        'Ils ne sont pas remplacés ni interpolés.'
    ))
    display(missing_checkpoints[['prompt_id', 'seed', 'iteration', 'download_error']])
else:
    print('Tous les checkpoints demandés sont présents :', len(checkpoint_frame))
"""
    ),
    markdown(
        """## 8. Re-scoring QR-Verify et qualité des checkpoints

Chaque raster est adressé par son hash. Relancer la cellule après une coupure reprend au premier
raster absent du journal. QR-Verify est exécuté trois fois et conserve l'intersection des presets
exacts. LPIPS, SRL et MER viennent de la trace différentiable de l'itération correspondante.
CLIP/HPS intermédiaires sont optionnels ; les quatre sorties finales ont toujours leurs scores API.
"""
    ),
    code(
        """checkpoint_scores_path = RUN_DIR / 'e032-checkpoint-scores.jsonl'
existing_scores = {}
if checkpoint_scores_path.is_file():
    for line in checkpoint_scores_path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            item = json.loads(line)
            if 'record_key' not in item or 'measurement_key' not in item:
                raise RuntimeError(
                    'Journal checkpoints E032 ancien/ambigu : archiver le dossier '
                    'de plan avant de relancer avec le code actuel.'
                )
            existing_scores[item['record_key']] = item

qr_scorer = ConservativeQRVerifyScorer(
    repetitions=QR_VERIFY_REPETITIONS,
    cache_dir=QR_VERIFY_CACHE,
)
intermediate_quality_scorer = (
    CLIPQualityScorer(
        Path('/cache/huggingface'), device='cpu',
        hps_enabled=True, hps_fail_closed=True,
    )
    if RUN_INTERMEDIATE_CLIP_HPS else None
)
available_checkpoints = checkpoint_frame[checkpoint_frame.local_image.notna()].to_dict('records')
expected_record_keys = set()
for position, item in enumerate(available_checkpoints, start=1):
    with Image.open(item['local_image']) as source:
        image = source.convert('RGB')
    raster_sha256 = image_raster_sha256(image)
    measurement_key = canonical_sha256({
        'raster_sha256': raster_sha256,
        'prompt': item['prompt'],
        'payload_sha256': hashlib.sha256(COLLECTION_PAYLOAD.encode()).hexdigest(),
        'qr_repetitions': QR_VERIFY_REPETITIONS,
        'engine_version': qr_scorer.engine_version,
        'implementation_sha256': qr_scorer.implementation_sha256,
        'scoring_version': qr_scorer.scoring_version,
        'perceptual': RUN_INTERMEDIATE_CLIP_HPS,
    })
    record_key = canonical_sha256({
        'measurement_key': measurement_key,
        'generation_run_id': item['generation_run_id'],
        'iteration': int(item['iteration']),
    })
    expected_record_keys.add(record_key)
    if record_key in existing_scores:
        continue
    qr_score = qr_scorer.score(image, COLLECTION_PAYLOAD).to_dict()
    visual = image_quality_metrics(image)
    perceptual = {
        'clip_similarity': None, 'clip_score': None,
        'clip_aesthetic': None, 'hpsv2_1': None,
    }
    if intermediate_quality_scorer is not None:
        perceptual = asdict(intermediate_quality_scorer.score(image, item['prompt']))
    row = {
        **item,
        'measurement_key': measurement_key,
        'record_key': record_key,
        'image_raster_sha256': raster_sha256,
        'qr_verify_exact': qr_score['consistent_any_exact'],
        'qr_verify_tolerance': qr_score['conservative_tolerance_score'],
        'qr_verify_exact_presets': qr_score['conservative_exact_presets'],
        'qr_verify_preset_count': qr_score['preset_count'],
        'qr_verify_unstable_presets': qr_score['unstable_preset_count'],
        **visual,
        **perceptual,
    }
    append_jsonl(checkpoint_scores_path, row)
    existing_scores[record_key] = row
    if position % 5 == 0 or position == len(available_checkpoints):
        clear_output(wait=True)
        display(pd.DataFrame([{
            'checkpoints disponibles': len(available_checkpoints),
            'scorés/persistés': len(existing_scores),
            'dernier contexte': f"{item['prompt_id']} / {item['seed']}",
            'dernière itération': item['iteration'],
            'CLIP/HPS intermédiaires': RUN_INTERMEDIATE_CLIP_HPS,
        }]))
qr_scorer.close()

score_frame = pd.DataFrame([
    row for key, row in existing_scores.items() if key in expected_record_keys
])
if len(score_frame) != len(available_checkpoints):
    raise RuntimeError(f'Re-scoring incomplet : {len(score_frame)}/{len(available_checkpoints)}.')
if score_frame.record_key.nunique() != len(available_checkpoints):
    raise RuntimeError('Un checkpoint E032 ne possède pas son record_key unique.')
checkpoint_alias_audit = (
    score_frame.groupby(['image_raster_sha256', 'prompt'], as_index=False)
    .agg(outputs=('record_key', 'nunique'), measurements=('measurement_key', 'nunique'))
)
invalid_checkpoint_aliases = checkpoint_alias_audit[
    (checkpoint_alias_audit.outputs > 1)
    & (checkpoint_alias_audit.measurements != 1)
]
if not invalid_checkpoint_aliases.empty:
    raise RuntimeError('Audit incohérent des checkpoints aliasés.')
checkpoint_alias_audit.to_csv(
    RUN_DIR / 'e032-checkpoint-raster-alias-audit.csv', index=False
)
score_frame.to_csv(RUN_DIR / 'e032-checkpoint-scores.csv', index=False)
display(score_frame[[
    'prompt_id', 'seed', 'iteration', 'qr_verify_exact', 'qr_verify_tolerance',
    'actual_module_error_rate', 'scanning_robust_loss', 'lpips_loss',
    'saturation_mean', 'high_saturation_pixel_ratio', 'clip_aesthetic',
    'clip_score', 'hpsv2_1',
]].head(21))
"""
    ),
    markdown("## 9. Galerie des itérations 0, 1, 2, 4, 8, 12 et 20"),
    code(
        """iteration_sheet_dir = RUN_DIR / 'iteration-contact-sheets'
iteration_sheet_dir.mkdir(exist_ok=True)
iteration_sheets = []
score_lookup = {
    (item.prompt_id, int(item.seed), int(item.iteration)): item
    for item in score_frame.itertuples(index=False)
}
for prompt in PROMPTS:
    for seed in SEEDS:
        records = [
            score_lookup.get((prompt['id'], seed, iteration))
            for iteration in CHECKPOINT_ITERATIONS
        ]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8.2))
        for axis, iteration, record in zip(axes.flat, CHECKPOINT_ITERATIONS, records):
            if record is None:
                axis.text(0.5, 0.5, f'itération {iteration} absente', ha='center', va='center')
            else:
                with Image.open(record.local_image) as source:
                    axis.imshow(source.convert('RGB'))
                axis.set_title(
                    f"i={iteration}  QRV={record.qr_verify_tolerance:.1%}\\n"
                    f"MER={finite(record.actual_module_error_rate) or 0:.1%}  "
                    f"LPIPS={finite(record.lpips_loss) or 0:.4f}\\n"
                    f"sat={record.saturation_mean:.3f}  "
                    f"SRL={finite(record.scanning_robust_loss) or 0:.4f}",
                    fontsize=8,
                )
            axis.axis('off')
        axes.flat[-1].axis('off')
        fig.suptitle(
            f"{prompt['id']} — seed {seed} — évolution SR-MPGD paper_equations",
            fontsize=11,
        )
        fig.tight_layout()
        path = iteration_sheet_dir / f"{prompt['id']}-seed-{seed}.png"
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        iteration_sheets.append(path)

print('Planches d évolution :', len(iteration_sheets))
for path in iteration_sheets:
    display(NotebookImage(filename=str(path), width=1100))
"""
    ),
    markdown("## 10. Résultats agrégés : le mécanisme progresse-t-il réellement ?"),
    code(
        """final_summary = (
    frame.groupby(['method_id', 'pipeline_state'], as_index=False)
    .agg(
        contexts=('trial_id', 'count'),
        qr_verify_success_rate=('qr_verify_conservative_exact', 'mean'),
        qr_verify_tolerance_mean=('qr_verify_conservative_tolerance', 'mean'),
        mer_mean=('module_error_rate', 'mean'),
        saturation_mean=('saturation_risk', 'mean'),
        clip_aesthetic_mean=('clip_aesthetic', 'mean'),
        clip_score_mean=('clip_score', 'mean'),
        hpsv2_1_mean=('hpsv2_1', 'mean'),
        duration_ms_mean=('duration_ms', 'mean'),
    )
)
final_summary.to_csv(RUN_DIR / 'e032-final-summary.csv', index=False)
display(final_summary)

iteration_summary = (
    score_frame.groupby('iteration', as_index=False)
    .agg(
        contexts=('record_key', 'count'),
        qr_verify_exact_rate=('qr_verify_exact', 'mean'),
        qr_verify_tolerance_mean=('qr_verify_tolerance', 'mean'),
        mer_mean=('actual_module_error_rate', 'mean'),
        srl_mean=('scanning_robust_loss', 'mean'),
        lpips_mean=('lpips_loss', 'mean'),
        saturation_mean=('saturation_mean', 'mean'),
        high_saturation_ratio=('high_saturation_pixel_ratio', 'mean'),
    )
)
iteration_summary.to_csv(RUN_DIR / 'e032-iteration-summary.csv', index=False)
display(iteration_summary)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes[0, 0].plot(iteration_summary.iteration, iteration_summary.qr_verify_tolerance_mean, marker='o')
axes[0, 0].set(title='QR-Verify exact conservateur', xlabel='itération', ylabel='tolérance')
axes[0, 1].plot(
    iteration_summary.iteration, iteration_summary.mer_mean,
    marker='o', color='#dc2626',
)
axes[0, 1].set(title='MER réelle du raster', xlabel='itération', ylabel='MER')
axes[1, 0].plot(
    iteration_summary.iteration, iteration_summary.srl_mean,
    marker='o', color='#7c3aed',
)
axes[1, 0].set(title='Scanning Robust Loss', xlabel='itération', ylabel='SRL')
axes[1, 1].plot(
    iteration_summary.iteration, iteration_summary.saturation_mean,
    marker='o', color='#ea580c', label='saturation',
)
axes[1, 1].plot(
    iteration_summary.iteration, iteration_summary.lpips_mean,
    marker='s', label='LPIPS',
)
axes[1, 1].set(title='Dégradation visuelle', xlabel='itération')
axes[1, 1].legend()
for axis in axes.flat:
    axis.grid(alpha=0.25)
fig.tight_layout()
evolution_path = RUN_DIR / 'e032-mechanism-evolution.png'
fig.savefig(evolution_path, dpi=160)
display(fig)

paper_final = final_summary[final_summary.method_id == 'diffqrcoder_paper_srmpgd'].iloc[0]
stage2_parent = final_summary[final_summary.method_id == 'diffqrcoder_paper_srpg'].iloc[0]
mechanism_verdict = {
    'srl_decreased_0_to_20': bool(
        iteration_summary.sort_values('iteration').iloc[-1].srl_mean
        < iteration_summary.sort_values('iteration').iloc[0].srl_mean
    ),
    'qr_verify_tolerance_delta_paper_vs_stage2': float(
        paper_final.qr_verify_tolerance_mean - stage2_parent.qr_verify_tolerance_mean
    ),
    'mer_delta_paper_vs_stage2': float(paper_final.mer_mean - stage2_parent.mer_mean),
    'clip_aesthetic_delta_paper_vs_stage2': float(
        paper_final.clip_aesthetic_mean - stage2_parent.clip_aesthetic_mean
    ),
    'hps_delta_paper_vs_stage2': float(paper_final.hpsv2_1_mean - stage2_parent.hpsv2_1_mean),
    'claim_phone_probability': False,
    'automatic_delivery_authorized': False,
}
atomic_json(RUN_DIR / 'e032-mechanism-verdict.json', mechanism_verdict)
display(pd.DataFrame([mechanism_verdict]).T.rename(columns={0: 'valeur'}))
"""
    ),
    markdown("## 11. Rapport, manifeste et archive finale"),
    code(
        """report_path = RUN_DIR / 'e032-report.md'
report = f'''# Rapport E032 — reconstruction SR-MPGD

- Plan : `{plan.plan_id}`
- Contextes : 10 prompts × 3 seeds
- Parent : Stage 2 SRPG complet, même latent réutilisé par les deux SR-MPGD
- Papier : 20 itérations, gamma 1000, LPIPS 0,01, QR original, sans oracle
- Garde : mêmes parent/gamma/lambda, avec portes et caps E019
- QR-Verify : {QR_VERIFY_REPETITIONS} répétitions, intersection des presets exacts
- CLIP/HPS intermédiaires : {RUN_INTERMEDIATE_CLIP_HPS}

## Verdict mécanistique

```json
{json.dumps(mechanism_verdict, ensure_ascii=False, indent=2)}
```

## Interprétation autorisée

E032 mesure si les équations font décroître la SRL et améliorent QR-Verify sur des rasters
strictement appariés. Les planches permettent d'identifier saturation, taches et perte du prompt.

## Interprétation interdite

E032 ne mesure pas une probabilité de scan téléphone, ne reproduit pas le QArt privé des auteurs
et n'autorise aucune livraison automatique. Une hausse moyenne ne remplace pas l'analyse par
prompt/seed ni un futur holdout physique.
'''
atomic_text(report_path, report)

control_dir = RUN_DIR / 'control-plane'
control_dir.mkdir(exist_ok=True)
for source in [runner.plan_path, runner.predictions_path, runner.state_path]:
    shutil.copy2(source, control_dir / source.name)
exports_copy = control_dir / 'exports'
exports_copy.mkdir(exist_ok=True)
for source in sorted(runner.exports_dir.glob('*.csv')):
    shutil.copy2(source, exports_copy / source.name)

artifact_files = sorted(
    path for path in RUN_DIR.rglob('*')
    if path.is_file() and path.name != 'e032-artifact-manifest.json'
    and not path.name.endswith('.tmp')
)
manifest = {
    'experiment': EXPERIMENT,
    'created_at': datetime.now(UTC).isoformat(),
    'plan': plan.public,
    'runner': runner_summary,
    'api_validation': schema['validation'],
    'api_quality_scoring': quality_contract,
    'final_rasters': len(frame),
    'checkpoint_rasters_available': len(score_frame),
    'required_checkpoint_iterations': CHECKPOINT_ITERATIONS,
    'pairing_rows_verified': len(pairing),
    'mechanism_verdict': mechanism_verdict,
    'claims': {
        'phone_probability': False,
        'private_qart_reproduced': False,
        'automatic_delivery': False,
    },
    'artifact_checksums': {
        str(path.relative_to(RUN_DIR)).replace('\\\\', '/'): sha256_file(path)
        for path in artifact_files
    },
}
manifest_path = RUN_DIR / 'e032-artifact-manifest.json'
atomic_json(manifest_path, manifest)

archive_path = ARCHIVE_ROOT / f'{plan.plan_id}-{EXPERIMENT}.tar.gz'
temporary_archive = archive_path.with_suffix('.tar.gz.tmp')
with tarfile.open(temporary_archive, 'w:gz') as bundle:
    bundle.add(RUN_DIR, arcname=f'{plan.plan_id}-{EXPERIMENT}')
os.replace(temporary_archive, archive_path)
archive_sha256 = sha256_file(archive_path)
atomic_text(archive_path.with_suffix('.tar.gz.sha256'), f'{archive_sha256}  {archive_path.name}\\n')
for source in [archive_path, archive_path.with_suffix('.tar.gz.sha256')]:
    shutil.copy2(source, DOWNLOAD_ROOT / source.name)

print('Rapport :', report_path)
print('Archive :', archive_path)
print('SHA-256 :', archive_sha256)
print('Copie téléchargeable :', DOWNLOAD_ROOT / archive_path.name)
"""
    ),
    markdown(
        """## Lecture finale

1. Les **30 planches finales** comparent directement Stage 1, Stage 2, SR-MPGD papier et gardé.
2. Les **30 planches d'évolution** montrent les itérations 0, 1, 2, 4, 8, 12 et 20.
3. `e032-mechanism-evolution.png` indique si SRL/MER/QR-Verify évoluent dans le bon sens en moyenne.
4. Téléchargez l'archive dans `/workspace/downloads` pour conserver tous les PNG et tableaux.
5. Ne choisissez aucune recette de production avant d'avoir examiné les résultats par contexte.

Pour calculer aussi CLIP-Aesthetic, CLIPScore et HPS sur **chaque checkpoint**, mettez
`RUN_INTERMEDIATE_CLIP_HPS = True` dans la cellule 1 puis relancez à partir de la cellule 8. Aucune
génération GPU ne sera recommencée : le journal et les PNG déjà téléchargés sont réutilisés.
"""
    ),
]

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

TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(TARGET)
