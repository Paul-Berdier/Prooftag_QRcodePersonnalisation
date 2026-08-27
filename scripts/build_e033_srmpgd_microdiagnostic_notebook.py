"""Build the resumable E033 SR-MPGD numerical micro-diagnostic notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "28_e033_srmpgd_microdiagnostic.ipynb"


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
        """# E033 — microdiagnostic SR-MPGD : recette publique, FP16 et FP32

E032 a montré deux problèmes distincts : son Stage 2 détruisait déjà fortement l'image, puis sa
descente SR-MPGD ne faisait pratiquement rien. E033 ne relance donc **pas** une campagne de trente
contextes. Il isole le mécanisme sur un seul cas facile et entièrement visible : une serre,
le seed `51001` et une URL courte.

Les cinq sorties sont produites dans **une seule campagne reprenable** :

1. `diffqrcoder_stage1`, référence esthétique ;
2. `diffqrcoder_paper_srpg`, témoin du Stage 2 E032 qui avait saturé ;
3. `e033_public_demo_srpg`, paramètres de la démo publique (ControlNet 1,05, SRG 50, PG 20) ;
4. `e033_equation_srmpgd_fp16`, mêmes équations avec le VAE dans sa précision modèle ;
5. `e033_equation_srmpgd_fp32`, branche primaire avec décodage VAE FP32.

Les branches FP16 et FP32 doivent réutiliser exactement le même Stage 2 public. Pour chacune,
E033 affiche aussi le même latent simplement redécodé par le VAE, sans mise à jour, afin de ne pas
confondre une différence de précision avec l'effet d'Eq. 14. Ce plan correctif s'arrête après une
seule mise à jour : les rasters des itérations 0 et 1 sont téléchargés par leur **URL directe**,
même lorsqu'un raster est identique au résultat final et n'apparaît donc pas dans la liste
d'artefacts de l'API. Quatre mises à jour ne seront autorisées que dans un nouveau plan après
réussite de cette porte mémoire.

Le verdict primaire porte uniquement sur FP32 : gradients image et latent initiaux finis et
strictement positifs, pas appliqué et déplacement latent strictement positifs, puis SRL de
l'itération 1 strictement inférieure à la SRL initiale. Une porte échouée est un résultat `STOP`,
pas une erreur d'exécution : les planches, traces, CSV et l'archive sont toujours écrits.
"""
    ),
    markdown("## 0. Imports et écritures atomiques"),
    code(
        """# ruff: noqa: E402
from __future__ import annotations

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
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image as NotebookImage
from IPython.display import Markdown, clear_output, display
from PIL import Image

from prooftag_qr.advisor_gallery import download_advisor_gallery, write_gallery_index
from prooftag_qr.advisor_inference import (
    AdvisorInferencePlan,
    AdvisorInferenceRunner,
    load_advisor_inference_results,
)
from prooftag_qr.e030_offline import sha256_file
from prooftag_qr.schemas import LabCampaignCreate
from prooftag_qr.validation import image_raster_sha256


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


def canonical_sha256(value):
    body = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str
    ).encode('utf-8')
    return hashlib.sha256(body).hexdigest()


def finite(value):
    if value is None or isinstance(value, bool) or str(value).strip() == '':
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


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


def download_direct_png(path, endpoint):
    path = Path(path)
    if valid_png(path):
        return path
    image = Image.open(BytesIO(api_bytes(endpoint))).convert('RGB')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.png.tmp')
    image.save(temporary, format='PNG', optimize=True)
    os.replace(temporary, path)
    if not valid_png(path):
        raise RuntimeError(f'PNG direct invalide : {path}')
    return path


print('Notebook CPU : orchestration et planches. API Kubernetes : génération GPU.')
"""
    ),
    markdown("## 1. Cas unique gelé — aucun élargissement automatique"),
    code(
        """EXPERIMENT = 'e033-srmpgd-microdiagnostic-v1'
COLLECTION_API_URL = 'http://prooftag-qr-svc.qr-core.svc.cluster.local:8080'
COLLECTION_PAYLOAD = os.environ.get('PROOFTAG_E033_PAYLOAD', 'https://ptag.io/t/e033')
ERROR_CORRECTION = 'M'
PROMPT = {
    'id': 'e033_simple_greenhouse',
    'family': 'simple',
    'text': (
        'a sunlit greenhouse filled with tomato plants and terracotta pots, '
        'botanical photograph'
    ),
}
NEGATIVE_PROMPT = (
    'easynegative, low quality, worst quality, blurry, deformed, watermark, text, logo, '
    'oversaturated, clipped highlights, posterized colors'
)
SEED = 51_001
MILESTONE_ITERATIONS = [0, 1]
RUN_E033 = True
POLL_SECONDS = 15.0
AUTOMATIC_EXPANSION_AUTHORIZED = False

OUTPUT_ROOT = Path('/data/e033-srmpgd-microdiagnostic')
DOWNLOAD_ROOT = Path('/workspace/downloads')
ARCHIVE_ROOT = Path('/data/e033-srmpgd-microdiagnostic-archives')
for directory in [OUTPUT_ROOT, DOWNLOAD_ROOT, ARCHIVE_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

assert COLLECTION_PAYLOAD.startswith('https://') and len(COLLECTION_PAYLOAD) <= 64
assert SEED == 51_001
assert MILESTONE_ITERATIONS == [0, 1]
assert AUTOMATIC_EXPANSION_AUTHORIZED is False
print('Un prompt × un seed × cinq méthodes = cinq générations finales.')
"""
    ),
    markdown("## 2. Préflight versionné et méthodes appariées"),
    code(
        """schema = api_json('/v1/lab/schema')
api_runtime = api_json('/v1/runtime')
notebook_commit = os.environ.get('PROOFTAG_GIT_COMMIT', '').strip().lower()
notebook_image = os.environ.get('PROOFTAG_RUNTIME_IMAGE', '').strip()
notebook_image_digest = os.environ.get('PROOFTAG_RUNTIME_IMAGE_DIGEST', '').strip().lower()
if not re.fullmatch(r'[0-9a-f]{40}', notebook_commit):
    raise RuntimeError('Commit notebook absent : déployer E033 avec une image versionnée.')
if not notebook_image.endswith(f':{notebook_commit[:12]}'):
    raise RuntimeError(f'Image notebook non liée au commit : {notebook_image!r}')
if not re.fullmatch(r'sha256:[0-9a-f]{64}', notebook_image_digest):
    raise RuntimeError('Digest OCI notebook absent ou invalide.')
api_identity = api_runtime.get('deployment_identity') or {}
if api_identity.get('configured') is not True:
    raise RuntimeError('Identité de déploiement API absente.')
if api_identity.get('git_commit') != notebook_commit:
    raise RuntimeError('Les commits API et notebook sont différents.')
api_image = str(api_identity.get('image') or '')
api_image_digest = str(api_identity.get('image_digest') or '').lower()
if not api_image.endswith(f':{notebook_commit[:12]}'):
    raise RuntimeError(f'Image API non liée au commit : {api_image!r}')
if not re.fullmatch(r'sha256:[0-9a-f]{64}', api_image_digest):
    raise RuntimeError('Digest OCI API absent ou invalide.')

profiles = {profile['id']: profile for profile in schema['profiles']}
METHOD_IDS = [
    'diffqrcoder_stage1',
    'diffqrcoder_paper_srpg',
    'e033_public_demo_srpg',
    'e033_equation_srmpgd_fp16',
    'e033_equation_srmpgd_fp32',
]
missing = set(METHOD_IDS) - profiles.keys()
if missing:
    raise RuntimeError(f'API trop ancienne, profils E033 absents : {sorted(missing)}')
for profile_id in METHOD_IDS[1:]:
    if profiles[profile_id]['enabled'] is not False:
        raise RuntimeError(f'{profile_id} doit rester désactivé hors du microdiagnostic.')
quality_contract = schema.get('quality_scoring', {})
if not quality_contract.get('clip_enabled') or not quality_contract.get('hpsv2_1_enabled'):
    raise RuntimeError('E033 exige CLIP, CLIP-Aesthetic et HPS v2.1 actifs.')
if quality_contract.get('failure_policy') != 'fail_closed':
    raise RuntimeError('E033 exige les métriques perceptuelles en mode fail-closed.')
quality_plan_binding = {
    'clip_enabled': quality_contract['clip_enabled'],
    'hpsv2_1_enabled': quality_contract['hpsv2_1_enabled'],
    'failure_policy': quality_contract['failure_policy'],
    # La provenance effective peut changer après un chargement à chaud. Les pins
    # immuables des métriques suffisent à lier le plan sans casser sa reprise.
    'metrics': quality_contract['metrics'],
}

METHODS = [deepcopy(profiles[method_id]) for method_id in METHOD_IDS]
for method in METHODS:
    method.pop('description', None)
    method['enabled'] = True
for method in METHODS[1:]:
    method['require_exact_stage1_reuse'] = True
method_by_id = {method['id']: method for method in METHODS}
public_stage2 = method_by_id['e033_public_demo_srpg']
fp16 = method_by_id['e033_equation_srmpgd_fp16']
fp32 = method_by_id['e033_equation_srmpgd_fp32']

public_settings = public_stage2['tools']['settings']
assert public_settings['diffqrcoder_stage2_initialization'] == 'public_random'
assert public_settings['diffqrcoder_stage2_target_mode'] == 'binary_exact'
assert public_settings['srpg_controlnet_scale'] == 1.05
assert public_settings['srpg_qr_weight'] == 50.0
assert public_settings['srpg_perceptual_weight'] == 20.0
assert fp16['tools']['settings']['srmpgd_decode_precision'] == 'model'
assert fp32['tools']['settings']['srmpgd_decode_precision'] == 'float32'
for method in [fp16, fp32]:
    settings = method['tools']['settings']
    assert settings['srmpgd_protocol'] == 'paper_equations'
    assert settings['srmpgd_max_iterations'] == 1
    assert settings['srmpgd_step_size'] == 1000.0
    assert settings['srmpgd_gradient_scale'] == 32768.0
    assert settings['srmpgd_lpips_weight'] == 0.01
    assert settings['srmpgd_lpips_device'] == 'cpu'

stage2_cache_fields = [
    'srpg_steps', 'diffqrcoder_stage2_strength',
    'diffqrcoder_stage2_initialization', 'srpg_controlnet_scale',
    'srpg_qr_weight', 'srpg_perceptual_weight', 'srpg_eta',
    'srpg_seed_offset', 'diffqrcoder_control_guidance_start',
    'diffqrcoder_control_guidance_end', 'diffqrcoder_stage2_target_mode',
    'diffqrcoder_qart_thresholds',
]
public_stage2_signature = {
    key: public_settings.get(key) for key in stage2_cache_fields
}
for method in [fp16, fp32]:
    candidate_signature = {
        key: method['tools']['settings'].get(key) for key in stage2_cache_fields
    }
    if candidate_signature != public_stage2_signature:
        raise RuntimeError(f"{method['id']} ne partage pas le Stage 2 public parent.")

runtime_binding = {
    'git_commit': notebook_commit,
    'notebook_image': notebook_image,
    'notebook_image_digest': notebook_image_digest,
    'api_image': api_image,
    'api_image_digest': api_image_digest,
    'diffqrcoder_revision': schema['notes']['upstream_revision'],
}
display(pd.DataFrame([{
    'méthode': method['id'],
    'sortie': method['output_variant'],
    'Stage 2': method['tools'].get('srpg_enabled', False),
    'SR-MPGD': method['tools'].get('srmpgd_enabled', False),
    'précision VAE': method['tools']['settings'].get('srmpgd_decode_precision', '-'),
} for method in METHODS]))
"""
    ),
    markdown("## 3. Plan immuable : une seule campagne reprenable"),
    code(
        """prompt_request = {
    'id': PROMPT['id'],
    'text': PROMPT['text'],
    'negative_prompt': NEGATIVE_PROMPT,
}
plan_material = {
    'experiment': EXPERIMENT,
    'payload_sha256': hashlib.sha256(COLLECTION_PAYLOAD.encode('utf-8')).hexdigest(),
    'payload_length': len(COLLECTION_PAYLOAD),
    'error_correction': ERROR_CORRECTION,
    'prompt': PROMPT,
    'negative_prompt': NEGATIVE_PROMPT,
    'seed': SEED,
    'methods': METHODS,
    'milestones': MILESTONE_ITERATIONS,
    'runtime_binding': runtime_binding,
    'validation': schema['validation'],
    'quality_scoring': quality_plan_binding,
    'automatic_expansion_authorized': AUTOMATIC_EXPANSION_AUTHORIZED,
}
plan_id = canonical_sha256(plan_material)[:16]
campaign_request = LabCampaignCreate.model_validate({
    'name': f'E033 {plan_id} serre seed-{SEED}',
    'payload': COLLECTION_PAYLOAD,
    'error_correction': ERROR_CORRECTION,
    'prompts': [prompt_request],
    'seeds': [SEED],
    'methods': METHODS,
    'max_attempts': 1,
}).model_dump(mode='json')

pipeline_states = {
    'diffqrcoder_stage1': 'stage1',
    'diffqrcoder_paper_srpg': 'e032_stage2_control',
    'e033_public_demo_srpg': 'public_stage2_parent',
    'e033_equation_srmpgd_fp16': 'equations_fp16',
    'e033_equation_srmpgd_fp32': 'equations_fp32_primary',
}
predictions = tuple({
    'prompt_id': PROMPT['id'],
    'prompt_text': PROMPT['text'],
    'prompt_family': PROMPT['family'],
    'plan_method_id': method_id,
    'source_method_id': method_id,
    'requested_source_output_variant': method_by_id[method_id]['output_variant'],
    'pipeline_state': pipeline_states[method_id],
    'role': 'e033_paired_microdiagnostic',
    'advisor_rank': position,
} for position, method_id in enumerate(METHOD_IDS, start=1))
public_plan = {
    **plan_material,
    'plan_id': plan_id,
    'campaign_count': 1,
    'trial_count': len(METHOD_IDS),
    'campaigns': [{
        'name': campaign_request['name'],
        'prompt_id': PROMPT['id'],
        'seed': SEED,
        'method_ids': METHOD_IDS,
    }],
}
plan = AdvisorInferencePlan(
    plan_id=plan_id,
    payload=COLLECTION_PAYLOAD,
    campaigns=(campaign_request,),
    predictions=predictions,
    public=public_plan,
)
assert len(plan.campaigns) == 1
assert plan.public['campaign_count'] == 1
assert plan.public['trial_count'] == 5
display(pd.DataFrame([{
    'plan': plan.plan_id,
    'campagnes reprenables': 1,
    'prompt': PROMPT['id'],
    'seed': SEED,
    'générations finales': 5,
    'élargissement automatique': AUTOMATIC_EXPANSION_AUTHORIZED,
}]))
"""
    ),
    markdown(
        """## 4. Exécuter ou reprendre

Le runner conserve le plan, l'état et le CSV exporté sous `/data`. Une campagne active est reprise
après une coupure, mais **aucune campagne terminale n'est automatiquement régénérée** : E033 utilise
une seule tentative. Une erreur technique produit un diagnostic et demande une décision humaine ;
un arrêt scientifique sur gradient nul reste une sortie normale. Il n'existe aucune boucle vers
trente contextes.
"""
    ),
    code(
        """events = deque(maxlen=20)
started = time.monotonic()


def progress(event):
    events.append(event)
    clear_output(wait=True)
    latest = events[-1]
    display(Markdown('### Progression E033'))
    display(pd.DataFrame([{
        'événement': latest.get('event'),
        'état': latest.get('status', 'running'),
        'essais': f"{latest.get('completed_trials', 0)}/{latest.get('total_trials', 0)}",
        'acceptés API': latest.get('accepted_trials', 0),
        'méthode': latest.get('current_method_id'),
        'seed': latest.get('current_seed'),
        'temps (min)': round((time.monotonic() - started) / 60, 1),
    }]))
    display(pd.DataFrame(list(events)[-8:]))


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
runner_summary = runner.run() if RUN_E033 else runner.summary()
display(pd.DataFrame([runner_summary]).T.rename(columns={0: 'valeur'}))
E033_TECHNICAL_STOP = runner_summary['status'] != 'completed'
TECHNICAL_ARCHIVE_DOWNLOAD = None

if E033_TECHNICAL_STOP:
    # Le runner strict a consommé son unique essai et son état fail-fast
    # interdit toute nouvelle soumission au prochain Run All. On transforme donc l'arrêt
    # en diagnostic téléchargeable au lieu de masquer la vraie erreur par une exception.
    diagnostic_root = runner.output_dir / 'technical-failure'
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    state = json.loads(runner.state_path.read_text(encoding='utf-8'))
    history = pd.DataFrame(state.get('history', []))
    history.to_csv(diagnostic_root / 'attempt-history.csv', index=False)

    export_rows = []
    failure_rows = []
    remote_rows = []
    export_copy_root = diagnostic_root / 'exports'
    export_copy_root.mkdir(exist_ok=True)
    remote_root = diagnostic_root / 'remote-campaigns'
    remote_root.mkdir(exist_ok=True)
    for item in state.get('history', []):
        campaign_id = str(item.get('campaign_id') or '')
        export_path = Path(str(item.get('export_path') or ''))
        export_diagnostic = {
            'attempt': item.get('attempt'),
            'campaign_id': campaign_id,
            'campaign_status': item.get('status'),
            'export_path': str(export_path),
            'export_exists': export_path.is_file(),
            'rows': 0,
            'error_rows': 0,
            'status_counts': '{}',
            'read_error': None,
        }
        if export_path.is_file():
            shutil.copy2(export_path, export_copy_root / export_path.name)
            try:
                exported = pd.read_csv(export_path, dtype=str, keep_default_na=False)
                export_diagnostic['rows'] = len(exported)
                if 'status' in exported:
                    export_diagnostic['status_counts'] = json.dumps(
                        exported['status'].value_counts().to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                error_mask = pd.Series(False, index=exported.index)
                if 'status' in exported:
                    error_mask |= exported['status'].str.lower().eq('error')
                if 'error' in exported:
                    error_mask |= exported['error'].str.strip().ne('')
                failed = exported[error_mask].copy()
                export_diagnostic['error_rows'] = len(failed)
                for record in failed.to_dict(orient='records'):
                    failure_rows.append({
                        'attempt': item.get('attempt'),
                        'campaign_id': campaign_id,
                        'trial_id': record.get('trial_id'),
                        'method_id': record.get('method_id'),
                        'status': record.get('status'),
                        'generation_run_id': record.get('generation_run_id'),
                        'error': record.get('error'),
                    })
            except Exception as exc:
                export_diagnostic['read_error'] = f'{type(exc).__name__}: {exc}'
        export_rows.append(export_diagnostic)

        if campaign_id:
            remote_error = None
            remote = None
            try:
                remote = api_json(f'/v1/lab/campaigns/{campaign_id}')
                atomic_json(remote_root / f'{campaign_id}.json', remote)
            except Exception as exc:
                remote_error = f'{type(exc).__name__}: {exc}'
            remote_rows.append({
                'attempt': item.get('attempt'),
                'campaign_id': campaign_id,
                'status': (remote or {}).get('status'),
                'campaign_error': (remote or {}).get('error'),
                'remote_read_error': remote_error,
            })

    export_diagnostics = pd.DataFrame(export_rows)
    trial_failures = pd.DataFrame(failure_rows)
    remote_diagnostics = pd.DataFrame(remote_rows)
    export_diagnostics.to_csv(
        diagnostic_root / 'export-diagnostics.csv', index=False
    )
    trial_failures.to_csv(diagnostic_root / 'failed-trials.csv', index=False)
    remote_diagnostics.to_csv(
        diagnostic_root / 'remote-campaign-diagnostics.csv', index=False
    )
    for source in [runner.plan_path, runner.predictions_path, runner.state_path]:
        shutil.copy2(source, diagnostic_root / source.name)

    failure_manifest = {
        'experiment': EXPERIMENT,
        'plan_id': plan.plan_id,
        'created_at': datetime.now(UTC).isoformat(),
        'runner_summary': runner_summary,
        'attempts': state.get('attempts', {}),
        'active_campaign': state.get('active_campaign'),
        'failed_campaigns': state.get('failed_campaigns', []),
        'history': state.get('history', []),
        'export_diagnostics': export_rows,
        'failed_trials': failure_rows,
        'remote_campaigns': remote_rows,
        'next_action': 'inspect_archive_without_regenerating',
    }
    atomic_json(diagnostic_root / 'technical-failure.json', failure_manifest)
    artifact_files = sorted(
        path for path in diagnostic_root.rglob('*')
        if path.is_file() and path.name != 'checksums.json'
        and not path.name.endswith('.tmp')
    )
    atomic_json(
        diagnostic_root / 'checksums.json',
        {
            str(path.relative_to(diagnostic_root)).replace('\\\\', '/'):
                sha256_file(path)
            for path in artifact_files
        },
    )
    technical_archive = ARCHIVE_ROOT / (
        f'{plan.plan_id}-{EXPERIMENT}-technical-failure.tar.gz'
    )
    temporary_archive = Path(f'{technical_archive}.tmp')
    with tarfile.open(temporary_archive, 'w:gz') as bundle:
        bundle.add(
            diagnostic_root,
            arcname=f'{plan.plan_id}-{EXPERIMENT}-technical-failure',
        )
    os.replace(temporary_archive, technical_archive)
    technical_sha256 = sha256_file(technical_archive)
    technical_checksum = Path(f'{technical_archive}.sha256')
    atomic_text(
        technical_checksum,
        f'{technical_sha256}  {technical_archive.name}\\n',
    )
    for source in [technical_archive, technical_checksum]:
        shutil.copy2(source, DOWNLOAD_ROOT / source.name)
    TECHNICAL_ARCHIVE_DOWNLOAD = DOWNLOAD_ROOT / technical_archive.name

    display(Markdown('### STOP technique E033 — aucune nouvelle génération'))
    display(history if not history.empty else pd.DataFrame([{'historique': 'vide'}]))
    display(export_diagnostics)
    display(
        trial_failures
        if not trial_failures.empty
        else pd.DataFrame([{'erreurs de trials': 'aucune erreur détaillée dans les CSV'}])
    )
    display(remote_diagnostics)
    display(Markdown(
        '**Ne relancez pas la campagne.** Les essais existants, leurs erreurs et les exports '
        f'disponibles ont été archivés dans `{TECHNICAL_ARCHIVE_DOWNLOAD}`. Téléchargez cette '
        'archive pour corriger la cause avant de créer un nouveau plan.'
    ))
else:
    display(Markdown('**Campagne complète : audit scientifique E033 autorisé.**'))
"""
    ),
    markdown("## 5. Audit d'appariement exact avant toute comparaison"),
    code(
        """RUN_DIR = runner.output_dir / 'analysis'
RUN_DIR.mkdir(parents=True, exist_ok=True)
rows = load_advisor_inference_results(runner.output_dir)
frame = pd.DataFrame(rows)
if len(frame) != 5 or set(frame.method_id) != set(METHOD_IDS):
    raise RuntimeError(f'Matrice E033 incomplète : {len(frame)}/5, {sorted(frame.method_id)}')
for metric in ['clip_aesthetic', 'clip_score', 'hpsv2_1']:
    missing = frame[frame[metric].map(finite).isna()]
    if not missing.empty:
        raise RuntimeError(f'{metric} absent ou non fini sur {len(missing)} sorties.')

by_method = {row.method_id: row for row in frame.itertuples(index=False)}
stage1 = by_method['diffqrcoder_stage1']
parent = by_method['e033_public_demo_srpg']
pairing_rows = []
for method_id in METHOD_IDS[1:]:
    row = by_method[method_id]
    stage1_exact = (
        finite(row.stage1_reused) == 1.0
        and row.stage1_source_run_id == stage1.generation_run_id
        and row.stage1_image_sha256 == stage1.final_image_sha256
    )
    if method_id in {'diffqrcoder_paper_srpg', 'e033_public_demo_srpg'}:
        stage2_exact = row.stage2_pairing_status == 'generated_source'
        parent_run_id = row.generation_run_id
    else:
        stage2_exact = (
            finite(row.stage2_pairing_exact) == 1.0
            and row.stage2_source_run_id == parent.generation_run_id
            and row.stage2_source_method_id == parent.method_id
            and row.stage2_source_latent_sha256 == row.stage2_latent_sha256
            and row.srmpgd_stage2_image_sha256 == parent.final_image_sha256
            and finite(row.srmpgd_iteration_zero_exact) == 1.0
        )
        parent_run_id = parent.generation_run_id
    pairing_rows.append({
        'method_id': method_id,
        'stage1_exact': stage1_exact,
        'stage2_exact': stage2_exact,
        'stage1_source_run_id': row.stage1_source_run_id,
        'expected_stage1_run_id': stage1.generation_run_id,
        'stage1_image_sha256': row.stage1_image_sha256,
        'expected_stage1_image_sha256': stage1.final_image_sha256,
        'stage2_source_run_id': row.stage2_source_run_id,
        'expected_stage2_run_id': parent_run_id,
        'stage2_latent_sha256': row.stage2_latent_sha256,
        'stage2_source_latent_sha256': row.stage2_source_latent_sha256,
        'iteration_zero_exact': row.srmpgd_iteration_zero_exact,
    })
pairing = pd.DataFrame(pairing_rows)
pairing.to_csv(RUN_DIR / 'e033-pairing-audit.csv', index=False)
frame.to_csv(RUN_DIR / 'e033-final-results.csv', index=False)
if not pairing[['stage1_exact', 'stage2_exact']].all(axis=None):
    display(pairing[~pairing[['stage1_exact', 'stage2_exact']].all(axis=1)])
    raise RuntimeError('Appariement exact E033 non prouvé : comparaison interdite.')
display(pairing)
print('Stage 1 partagé et Stage 2 public parent prouvés par IDs, latent et raster.')
"""
    ),
    markdown("## 6. Télécharger et afficher les cinq sorties finales"),
    code(
        """gallery_dir = RUN_DIR / 'final-gallery'
gallery_entries = download_advisor_gallery(
    [{**row, 'section': row['pipeline_state']} for row in rows],
    api_url=COLLECTION_API_URL,
    output_dir=gallery_dir / 'images',
)
download_errors = [entry for entry in gallery_entries if entry.get('download_error')]
if download_errors:
    display(pd.DataFrame(download_errors))
    raise RuntimeError(f'{len(download_errors)} images finales indisponibles.')
write_gallery_index(gallery_entries, gallery_dir)
gallery_by_method = {entry['method_id']: entry for entry in gallery_entries}


def final_title(entry):
    return (
        f"{entry['method_id']}\\n"
        f"QRV={finite(entry.get('qr_tolerance')) or 0:.1%}  "
        f"MER={finite(entry.get('module_error_rate')) or 0:.1%}\\n"
        f"AES={finite(entry.get('clip_aesthetic')) or 0:.2f}  "
        f"HPS={finite(entry.get('hpsv2_1')) or 0:.3f}"
    )


fig, axes = plt.subplots(1, 5, figsize=(20, 4.8))
for axis, method_id in zip(axes, METHOD_IDS, strict=True):
    entry = gallery_by_method[method_id]
    with Image.open(entry['local_image']) as source:
        axis.imshow(source.convert('RGB'))
    axis.set_title(final_title(entry), fontsize=8)
    axis.axis('off')
fig.suptitle(f"E033 — {PROMPT['id']} — seed {SEED}", fontsize=12)
fig.tight_layout()
final_sheet = RUN_DIR / 'e033-final-contact-sheet.png'
fig.savefig(final_sheet, dpi=150, bbox_inches='tight')
plt.close(fig)
display(NotebookImage(filename=str(final_sheet), width=1400))
"""
    ),
    markdown(
        """## 7. Télécharger directement les états 000 et 001

Cette cellule n'interroge volontairement **pas** `/artifacts`. Elle appelle directement
`/variants/srmpgd_iteration_000`, puis 001. Ainsi, l'itération 0 reste téléchargeable
même lorsque son contenu est identique au Stage 2 parent et que le catalogue la déduplique.
"""
    ),
    code(
        """MILESTONE_METHOD_IDS = [
    'e033_equation_srmpgd_fp16',
    'e033_equation_srmpgd_fp32',
]
milestone_root = RUN_DIR / 'srmpgd-milestones'
milestone_rows = []
redecode_rows = []
traces = {}
for method_id in MILESTONE_METHOD_IDS:
    row = by_method[method_id]
    run_id = row.generation_run_id
    method_dir = milestone_root / method_id
    method_dir.mkdir(parents=True, exist_ok=True)
    trace_path = method_dir / 'srmpgd-trace.json'
    if not trace_path.is_file():
        atomic_json(
            trace_path,
            api_json(f'/v1/generations/{run_id}/metadata/srmpgd_trace'),
        )
    trace = json.loads(trace_path.read_text(encoding='utf-8'))
    traces[method_id] = trace
    trace_by_iteration = {int(item['iteration']): item for item in trace['steps']}
    redecode_variant = 'srmpgd_redecoded_iteration_000'
    redecode_endpoint = f'/v1/generations/{run_id}/variants/{redecode_variant}'
    redecode_path = None
    redecode_error = None
    try:
        redecode_path = download_direct_png(
            method_dir / 'iteration-000-vae-redecode.png', redecode_endpoint
        )
    except HTTPError as exc:
        if exc.code != 404:
            raise
        redecode_error = 'HTTP 404: témoin de redécodage VAE absent'
    redecode_raster_sha256 = None
    if redecode_path is not None:
        with Image.open(redecode_path) as source:
            redecode_raster_sha256 = image_raster_sha256(source.convert('RGB'))
    redecode_rows.append({
        'method_id': method_id,
        'generation_run_id': run_id,
        'available': redecode_path is not None,
        'direct_endpoint': redecode_endpoint,
        'local_image': str(redecode_path) if redecode_path is not None else None,
        'download_error': redecode_error,
        'image_raster_sha256': redecode_raster_sha256,
        'trace_image_sha256': trace.get('initial_redecoded_image_sha256'),
        'mean_absolute_change_from_stage2': (
            trace.get('initial_redecode_change', {}).get('mean_absolute_change')
        ),
    })
    for iteration in MILESTONE_ITERATIONS:
        variant = f'srmpgd_iteration_{iteration:03d}'
        # URL directe obligatoire : ne pas la remplacer par le catalogue /artifacts.
        endpoint = f'/v1/generations/{run_id}/variants/{variant}'
        image_path = None
        raster_sha256 = None
        download_error = None
        try:
            image_path = download_direct_png(
                method_dir / f'iteration-{iteration:03d}.png', endpoint
            )
            with Image.open(image_path) as source:
                raster_sha256 = image_raster_sha256(source.convert('RGB'))
        except HTTPError as exc:
            if exc.code != 404:
                raise
            # Un arrêt numérique à i0 ne publie légitimement pas i1.
            # C'est un résultat scientifique à archiver, pas une panne du runner.
            download_error = 'HTTP 404: jalon absent (arrêt anticipé probable)'
        trace_step = trace_by_iteration.get(iteration, {})
        milestone_rows.append({
            'method_id': method_id,
            'generation_run_id': run_id,
            'iteration': iteration,
            'direct_endpoint': endpoint,
            'available': image_path is not None,
            'local_image': str(image_path) if image_path is not None else None,
            'download_error': download_error,
            'stop_reason': trace.get('stop_reason'),
            'image_raster_sha256': raster_sha256,
            'scanning_robust_loss': trace_step.get('scanning_robust_loss'),
            'gradient_rms': trace_step.get('gradient_rms'),
            'gradient_scale': trace_step.get('gradient_scale'),
            'image_gradient_rms': trace_step.get('image_gradient_rms'),
            'latent_delta_rms': trace_step.get('latent_delta_rms'),
            'applied_step_rms': trace_step.get('applied_step_rms'),
            'objective': trace_step.get('objective'),
            'actual_module_error_rate': trace_step.get('actual_module_error_rate'),
        })

milestones = pd.DataFrame(milestone_rows)
redecodes = pd.DataFrame(redecode_rows)
if len(milestones) != len(MILESTONE_METHOD_IDS) * len(MILESTONE_ITERATIONS):
    raise RuntimeError('Le journal des quatre requêtes directes E033 est incomplet.')
milestones.to_csv(RUN_DIR / 'e033-milestones.csv', index=False)
redecodes.to_csv(RUN_DIR / 'e033-vae-redecode-controls.csv', index=False)
REDECODE_CONTROLS_AVAILABLE = bool(
    len(redecodes) == len(MILESTONE_METHOD_IDS) and redecodes.available.all()
)
REDECODE_CONTROLS_VERIFIED = bool(
    REDECODE_CONTROLS_AVAILABLE
    and (
        redecodes.image_raster_sha256 == redecodes.trace_image_sha256
    ).all()
)

with Image.open(gallery_by_method['e033_public_demo_srpg']['local_image']) as source:
    parent_raster_sha256 = image_raster_sha256(source.convert('RGB'))
iteration_zero_rows = milestones[
    (milestones.iteration == 0) & milestones.available
]
iteration_zero_hashes = set(iteration_zero_rows.image_raster_sha256.dropna())
DIRECT_ITERATION_ZERO_AVAILABLE = len(iteration_zero_rows) == 2
DIRECT_ITERATION_ZERO_EXACT = (
    DIRECT_ITERATION_ZERO_AVAILABLE
    and iteration_zero_hashes == {parent_raster_sha256}
)
if DIRECT_ITERATION_ZERO_AVAILABLE and not DIRECT_ITERATION_ZERO_EXACT:
    raise RuntimeError(
        'Les rasters directs de l itération 0 ne sont pas le Stage 2 public exact.'
    )
print(
    'Requêtes directes terminées :', int(milestones.available.sum()), '/ 4 jalons présents ;',
    'itérations 0 exactes =', DIRECT_ITERATION_ZERO_EXACT,
)
print(
    'Témoins VAE sans update disponibles :', int(redecodes.available.sum()), '/ 2 ;',
    'hashes vérifiés =', REDECODE_CONTROLS_VERIFIED,
)
missing_milestones = milestones[~milestones.available]
if not missing_milestones.empty:
    display(Markdown(
        '**Jalons absents :** ils restent dans le CSV avec `available=False`. '
        'C est le résultat attendu lorsqu une branche s arrête numériquement avant i1.'
    ))
    display(missing_milestones.reindex(columns=[
        'method_id', 'iteration', 'direct_endpoint', 'stop_reason', 'download_error',
    ]))
"""
    ),
    markdown("## 8. Planches des jalons — à regarder avant le verdict"),
    code(
        """trace_lookup = {
    (row.method_id, int(row.iteration)): row
    for row in milestones.itertuples(index=False)
}
redecode_lookup = {
    row.method_id: row for row in redecodes.itertuples(index=False)
}
milestone_column_count = 2 + len(MILESTONE_ITERATIONS)
fig, axes = plt.subplots(
    len(MILESTONE_METHOD_IDS), milestone_column_count,
    figsize=(3.5 * milestone_column_count, 7.8),
)
for row_index, method_id in enumerate(MILESTONE_METHOD_IDS):
    parent_entry = gallery_by_method['e033_public_demo_srpg']
    with Image.open(parent_entry['local_image']) as source:
        axes[row_index, 0].imshow(source.convert('RGB'))
    axes[row_index, 0].set_title('Stage 2 public parent', fontsize=8)
    axes[row_index, 0].axis('off')
    redecode = redecode_lookup[method_id]
    if bool(redecode.available) and redecode.local_image:
        with Image.open(redecode.local_image) as source:
            axes[row_index, 1].imshow(source.convert('RGB'))
        axes[row_index, 1].set_title(
            'VAE redécodé, aucun pas\\n'
            f"MAE={finite(redecode.mean_absolute_change_from_stage2) or 0:.3e}",
            fontsize=8,
        )
    else:
        axes[row_index, 1].text(
            0.5, 0.5, 'témoin VAE absent', ha='center', va='center',
            color='#991b1b', transform=axes[row_index, 1].transAxes,
        )
    axes[row_index, 1].axis('off')
    for column, iteration in enumerate(MILESTONE_ITERATIONS, start=2):
        record = trace_lookup[(method_id, iteration)]
        if bool(record.available) and record.local_image:
            with Image.open(record.local_image) as source:
                axes[row_index, column].imshow(source.convert('RGB'))
            axes[row_index, column].set_title(
                f"i={iteration}  SRL={finite(record.scanning_robust_loss) or 0:.6f}\\n"
                f"grad={finite(record.gradient_rms) or 0:.3e}  "
                f"Δz={finite(record.latent_delta_rms) or 0:.3e}",
                fontsize=8,
            )
        else:
            axes[row_index, column].set_facecolor('#f3f4f6')
            axes[row_index, column].text(
                0.5, 0.55, f'i={iteration} absent', ha='center', va='center',
                fontsize=11, color='#991b1b', transform=axes[row_index, column].transAxes,
            )
            axes[row_index, column].text(
                0.5, 0.38, 'arrêt anticipé / 404', ha='center', va='center',
                fontsize=8, transform=axes[row_index, column].transAxes,
            )
        axes[row_index, column].axis('off')
    axes[row_index, 0].set_ylabel(method_id, fontsize=8)
fig.suptitle('E033 — évolution directe et strictement appariée du SR-MPGD', fontsize=12)
fig.tight_layout()
milestone_sheet = RUN_DIR / 'e033-milestone-contact-sheet.png'
fig.savefig(milestone_sheet, dpi=160, bbox_inches='tight')
plt.close(fig)
display(NotebookImage(filename=str(milestone_sheet), width=1400))
display(milestones.reindex(columns=[
    'method_id', 'iteration', 'scanning_robust_loss', 'gradient_rms',
    'gradient_scale', 'image_gradient_rms', 'latent_delta_rms',
    'applied_step_rms', 'available',
    'stop_reason', 'download_error',
]))
display(redecodes.reindex(columns=[
    'method_id', 'available', 'image_raster_sha256', 'trace_image_sha256',
    'mean_absolute_change_from_stage2', 'download_error',
]))
"""
    ),
    markdown("## 9. Portes primaires FP32 — verdict local uniquement"),
    code(
        """primary_steps = {
    int(item['iteration']): item
    for item in traces['e033_equation_srmpgd_fp32']['steps']
}
primary_milestones = milestones[
    milestones.method_id == 'e033_equation_srmpgd_fp32'
]
PRIMARY_MILESTONES_AVAILABLE = bool(
    len(primary_milestones) == len(MILESTONE_ITERATIONS)
    and primary_milestones.available.all()
)
gradient_0 = finite(primary_steps.get(0, {}).get('gradient_rms'))
image_gradient_0 = finite(primary_steps.get(0, {}).get('image_gradient_rms'))
applied_step_0 = finite(primary_steps.get(0, {}).get('applied_step_rms'))
latent_delta_1 = finite(primary_steps.get(1, {}).get('latent_delta_rms'))
srl_0 = finite(primary_steps.get(0, {}).get('scanning_robust_loss'))
srl_1 = finite(primary_steps.get(1, {}).get('scanning_robust_loss'))

gate_rows = [
    {
        'porte': 'temoins_vae_sans_update_disponibles',
        'valeur': int(redecodes.available.sum()),
        'référence': '2/2 et hashes trace = raster',
        'réussie': REDECODE_CONTROLS_VERIFIED,
    },
    {
        'porte': 'jalons_fp32_000_001_disponibles',
        'valeur': int(primary_milestones.available.sum()),
        'référence': '2/2',
        'réussie': PRIMARY_MILESTONES_AVAILABLE,
    },
    {
        'porte': 'iteration_zero_directe_identique_au_stage2_parent',
        'valeur': DIRECT_ITERATION_ZERO_EXACT,
        'référence': 'True pour FP16 et FP32',
        'réussie': DIRECT_ITERATION_ZERO_EXACT,
    },
    {
        'porte': 'gradient_image_fp32_iteration_0_fini_et_positif',
        'valeur': image_gradient_0,
        'référence': '> 0 et fini',
        'réussie': image_gradient_0 is not None and image_gradient_0 > 0,
    },
    {
        'porte': 'gradient_fp32_iteration_0_fini_et_positif',
        'valeur': gradient_0,
        'référence': '> 0 et fini',
        'réussie': gradient_0 is not None and gradient_0 > 0,
    },
    {
        'porte': 'pas_fp32_iteration_0_fini_et_positif',
        'valeur': applied_step_0,
        'référence': '> 0 et fini',
        'réussie': applied_step_0 is not None and applied_step_0 > 0,
    },
    {
        'porte': 'deplacement_latent_fp32_iteration_1_positif',
        'valeur': latent_delta_1,
        'référence': '> 0 et fini',
        'réussie': latent_delta_1 is not None and latent_delta_1 > 0,
    },
    {
        'porte': 'srl_fp32_diminue_a_iteration_1',
        'valeur': srl_1,
        'référence': f'< SRL0={srl_0}',
        'réussie': (
            srl_0 is not None
            and srl_1 is not None
            and srl_1 < srl_0
        ),
    },
]
gate_frame = pd.DataFrame(gate_rows)
PRIMARY_FP32_GATES_PASSED = bool(gate_frame['réussie'].all())
mechanism_verdict = {
    'experiment': EXPERIMENT,
    'plan_id': plan.plan_id,
    'primary_method': 'e033_equation_srmpgd_fp32',
    'primary_fp32_gates_passed': PRIMARY_FP32_GATES_PASSED,
    'primary_milestones_available': PRIMARY_MILESTONES_AVAILABLE,
    'vae_redecode_controls_available': REDECODE_CONTROLS_AVAILABLE,
    'vae_redecode_controls_verified': REDECODE_CONTROLS_VERIFIED,
    'direct_iteration_zero_exact': DIRECT_ITERATION_ZERO_EXACT,
    'image_gradient_iteration_0': image_gradient_0,
    'gradient_iteration_0': gradient_0,
    'applied_step_iteration_0': applied_step_0,
    'latent_delta_iteration_1': latent_delta_1,
    'srl_iteration_0': srl_0,
    'srl_iteration_1': srl_1,
    'automatic_expansion_authorized': False,
    'next_action': (
        'manual_review_then_design_four_iteration_gate'
        if PRIMARY_FP32_GATES_PASSED
        else 'stop_and_fix_numerics_without_expanding'
    ),
}
gate_frame.to_csv(RUN_DIR / 'e033-primary-fp32-gates.csv', index=False)
atomic_json(RUN_DIR / 'e033-mechanism-verdict.json', mechanism_verdict)
display(gate_frame)
display(pd.DataFrame([mechanism_verdict]).T.rename(columns={0: 'valeur'}))
if not PRIMARY_FP32_GATES_PASSED:
    display(Markdown(
        '**STOP scientifique : au moins une porte FP32 échoue.** '
        'Aucun élargissement automatique n est lancé ; l archive sera tout de même créée.'
    ))
"""
    ),
    markdown("## 10. Rapport, manifeste et archive — même si une porte échoue"),
    code(
        """report_path = RUN_DIR / 'e033-report.md'
report = f'''# Rapport E033 — microdiagnostic SR-MPGD

- Plan : `{plan.plan_id}`
- Cas : `{PROMPT['id']}`, seed `{SEED}`, payload court
- Comparaison : Stage 1, ancien Stage 2 E032, Stage 2 démo publique, SR-MPGD FP16, SR-MPGD FP32
- Témoins VAE sans mise à jour : FP16 et FP32
- Jalons directs : {MILESTONE_ITERATIONS}
- Portes FP32 réussies : `{PRIMARY_FP32_GATES_PASSED}`
- Élargissement automatique : `False`

## Verdict mécanistique

```json
{json.dumps(mechanism_verdict, ensure_ascii=False, indent=2)}
```

## Interprétation autorisée

E033 indique si la chaîne de gradient FP32 produit réellement un déplacement latent et si la
Scanning Robust Loss décroît sur ce cas unique. Les planches montrent séparément la dégradation
du Stage 2 et l'effet propre d'une mise à jour. Ce résultat ne valide pas encore quatre
itérations : elles nécessiteront un nouveau plan après réussite de cette porte.

## Interprétation interdite

Un PASS ne prouve ni la généralisation, ni une probabilité de scan téléphone, ni la supériorité
esthétique. Un STOP interdit seulement d'étendre ce mécanisme défectueux à trente contextes.
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
    if path.is_file() and path.name != 'e033-artifact-manifest.json'
    and not path.name.endswith('.tmp')
)
manifest = {
    'experiment': EXPERIMENT,
    'created_at': datetime.now(UTC).isoformat(),
    'plan': plan.public,
    'runner': runner_summary,
    'runtime_binding': runtime_binding,
    'final_rasters': len(frame),
    'direct_milestone_rasters': len(milestones),
    'direct_vae_redecode_rasters': int(redecodes.available.sum()),
    'required_milestone_iterations': MILESTONE_ITERATIONS,
    'pairing_rows_verified': len(pairing),
    'mechanism_verdict': mechanism_verdict,
    'claims': {
        'phone_probability': False,
        'generalization': False,
        'automatic_expansion': False,
        'automatic_delivery': False,
    },
    'artifact_checksums': {
        str(path.relative_to(RUN_DIR)).replace('\\\\', '/'): sha256_file(path)
        for path in artifact_files
    },
}
manifest_path = RUN_DIR / 'e033-artifact-manifest.json'
atomic_json(manifest_path, manifest)

archive_path = ARCHIVE_ROOT / f'{plan.plan_id}-{EXPERIMENT}.tar.gz'
temporary_archive = Path(f'{archive_path}.tmp')
with tarfile.open(temporary_archive, 'w:gz') as bundle:
    bundle.add(RUN_DIR, arcname=f'{plan.plan_id}-{EXPERIMENT}')
os.replace(temporary_archive, archive_path)
archive_sha256 = sha256_file(archive_path)
checksum_path = Path(f'{archive_path}.sha256')
atomic_text(checksum_path, f'{archive_sha256}  {archive_path.name}\\n')
for source in [archive_path, checksum_path]:
    shutil.copy2(source, DOWNLOAD_ROOT / source.name)

print('Rapport :', report_path)
print('Archive créée même en cas de STOP :', archive_path)
print('SHA-256 :', archive_sha256)
print('Copie téléchargeable :', DOWNLOAD_ROOT / archive_path.name)
"""
    ),
    markdown("## 11. Lecture finale"),
    code(
        """if PRIMARY_FP32_GATES_PASSED:
    display(Markdown(
        '**PASS mécanistique local.** Inspecter les deux planches et le rapport avant de '
        'concevoir un petit holdout séparé. E033 ne lance aucune expansion automatiquement.'
    ))
else:
    display(Markdown(
        '**STOP mécanistique archivé.** Le gradient, le déplacement latent ou la baisse de SRL '
        'n est pas démontré. Corriger le mécanisme puis créer un nouveau plan ; '
        'ne pas relancer '
        'trente contextes avec ces résultats.'
    ))
print('Archive :', DOWNLOAD_ROOT / archive_path.name)
"""
    ),
]

# Lors d'un STOP technique, les cellules scientifiques suivantes doivent s'ignorer proprement :
# la cellule d'exécution a déjà produit l'archive de diagnostic et le runner fail-fast n'enverra
# aucune nouvelle campagne lors d'un nouveau Run All.
scientific_section = False
for cell in cells:
    source = "".join(cell.get("source", []))
    if cell["cell_type"] == "markdown" and source.startswith("## 5."):
        scientific_section = True
    if scientific_section and cell["cell_type"] == "code":
        indented = "".join(
            f"    {line}" if line.strip() else line
            for line in source.splitlines(keepends=True)
        )
        guarded = (
            "if E033_TECHNICAL_STOP:\n"
            "    display(Markdown(\n"
            "        f'**Cellule ignorée après le STOP technique.** Archive : ' \n"
            "        f'`{TECHNICAL_ARCHIVE_DOWNLOAD}`'\n"
            "    ))\n"
            "else:\n"
            f"{indented}"
        )
        cell["source"] = guarded.splitlines(True)

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
