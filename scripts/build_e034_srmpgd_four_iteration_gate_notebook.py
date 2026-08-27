"""Build the resumable E034 four-iteration SR-MPGD gate notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "29_e034_srmpgd_four_iteration_gate.ipynb"


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
        """# E034 — SR-MPGD : porte appariée à quatre itérations

E033 a prouvé sur un cas unique que l'implémentation des Eq. 13-14 produit un gradient fini,
déplace réellement le latent et diminue la Scanning Robust Loss après un pas. E034 poursuit
**uniquement ce même cas** et ce même Stage 2 public pendant quatre mises à jour. Il ne relance ni
E032, ni E033, ni une campagne multi-prompt.

Les quatre sorties sont produites dans **une seule campagne reprenable** :

1. `diffqrcoder_stage1`, référence esthétique et source commune ;
2. `e033_public_demo_srpg`, parent Stage 2 public inchangé ;
3. `e034_equation_srmpgd_fp16`, quatre mises à jour en précision VAE du modèle ;
4. `e034_equation_srmpgd_fp32`, quatre mises à jour avec le VAE temporairement en FP32.

Les deux branches SR-MPGD réutilisent exactement le même latent Stage 2. Le notebook télécharge
par URL directe le parent redécodé par le VAE puis les rasters `i0`, `i1`, `i2` et `i4`. Les
hashes E033 connus du Stage 1, du parent et du latent sont gelés : une divergence arrête l'audit
avant toute interprétation.

Le verdict distingue quatre dimensions : fonctionnement mathématique, signal QR-Verify,
préservation visuelle et aptitude à la production. Une porte échouée donne un `STOP` scientifique,
mais les traces, scores, images, CSV et l'archive restent disponibles. E034 n'autorise aucun
élargissement automatique.
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
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import matplotlib.pyplot as plt
import numpy as np
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
from prooftag_qr.incident_archive import (
    prepare_incident_bundle,
    resolve_incident_archive,
    snapshot_tree_once,
)
from prooftag_qr.schemas import LabCampaignCreate
from prooftag_qr.qr import diffqrcoder_module_error_rate, generate_diffqrcoder_qr
from prooftag_qr.quality import image_change_metrics, image_quality_metrics
from prooftag_qr.validation import (
    ConservativeQRVerifyScorer,
    canonical_conservative_qr_verify_evidence,
    image_raster_sha256,
)


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
    atomic_text(
        path,
        json.dumps(
            value, ensure_ascii=False, indent=2, default=str, allow_nan=False
        ),
    )


def atomic_json_once(path, value):
    path = Path(path)
    if not path.is_file():
        atomic_json(path, value)


def atomic_copy_once(source, target):
    source = Path(source)
    target = Path(target)
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def canonical_sha256(value):
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
        allow_nan=False,
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
        """EXPERIMENT = 'e034-srmpgd-four-iteration-gate-v1'
COLLECTION_API_URL = 'http://prooftag-qr-svc.qr-core.svc.cluster.local:8080'
# Le payload E033 est littéral : un override rendrait les hashes parents impossibles à reproduire.
COLLECTION_PAYLOAD = 'https://ptag.io/t/e033'
COLLECTION_PAYLOAD_SHA256 = (
    '12834cad09eb0680af5a71c0f8c20627fba9746c117e0da3e1c6a14f18952475'
)
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
MILESTONE_ITERATIONS = [0, 1, 2, 4]
RUN_E034 = True
POLL_SECONDS = 15.0
AUTOMATIC_EXPANSION_AUTHORIZED = False

E033_EXPECTED = {
    'stage1_raster_sha256': '02e0bea8e5c539cda599f6158a3a07bf1a9eed3db2f0e468d58b56f430458d53',
    'stage2_parent_raster_sha256': (
        '8cb36a623aa999567f51402615cceb8917505f3ba7e9c06c34e6bbef045e9721'
    ),
    'stage2_parent_latent_sha256': (
        '6bd10526053cb9af9a80b123b29c66919e60523f6703a9f7d4cf10a5506e2146'
    ),
}
VISUAL_GUARDS = {
    'maximum_mean_absolute_change': 0.04,
    'maximum_clipped_pixel_ratio_increase': 0.01,
    'maximum_saturation_mean_increase': 0.04,
    'maximum_high_saturation_ratio_increase': 0.05,
}

OUTPUT_ROOT = Path('/data/e034-srmpgd-four-iteration-gate')
DOWNLOAD_ROOT = Path('/workspace/downloads')
ARCHIVE_ROOT = Path('/data/e034-srmpgd-four-iteration-gate-archives')
for directory in [OUTPUT_ROOT, DOWNLOAD_ROOT, ARCHIVE_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

assert COLLECTION_PAYLOAD.startswith('https://') and len(COLLECTION_PAYLOAD) <= 64
assert hashlib.sha256(COLLECTION_PAYLOAD.encode('utf-8')).hexdigest() == (
    COLLECTION_PAYLOAD_SHA256
)
assert SEED == 51_001
assert MILESTONE_ITERATIONS == [0, 1, 2, 4]
assert AUTOMATIC_EXPANSION_AUTHORIZED is False
print('Un prompt × un seed × quatre méthodes = quatre générations finales.')
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
    raise RuntimeError('Commit notebook absent : déployer E034 avec une image versionnée.')
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
if schema.get('validation', {}).get('engine') != 'antfu/qr-verify@0.2.0':
    raise RuntimeError('E034 exige antfu/qr-verify@0.2.0 épinglé.')
METHOD_IDS = [
    'diffqrcoder_stage1',
    'e033_public_demo_srpg',
    'e034_equation_srmpgd_fp16',
    'e034_equation_srmpgd_fp32',
]
missing = set(METHOD_IDS) - profiles.keys()
if missing:
    raise RuntimeError(f'API trop ancienne, profils E034 absents : {sorted(missing)}')
for profile_id in METHOD_IDS[1:]:
    if profiles[profile_id]['enabled'] is not False:
        raise RuntimeError(f'{profile_id} doit rester désactivé hors du microdiagnostic.')
quality_contract = schema.get('quality_scoring', {})
if not quality_contract.get('clip_enabled') or not quality_contract.get('hpsv2_1_enabled'):
    raise RuntimeError('E034 exige CLIP, CLIP-Aesthetic et HPS v2.1 actifs.')
if quality_contract.get('failure_policy') != 'fail_closed':
    raise RuntimeError('E034 exige les métriques perceptuelles en mode fail-closed.')
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
fp16 = method_by_id['e034_equation_srmpgd_fp16']
fp32 = method_by_id['e034_equation_srmpgd_fp32']

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
    assert settings['srmpgd_max_iterations'] == 4
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
preflight_blueprint = generate_diffqrcoder_qr(
    COLLECTION_PAYLOAD,
    ERROR_CORRECTION,
    version=3,
    mask_pattern=4,
    module_size=20,
    border=4,
)
preflight_scorer = ConservativeQRVerifyScorer(
    repetitions=2,
    cache_dir=OUTPUT_ROOT / 'preflight-qr-verify-cache',
)
try:
    preflight_score = preflight_scorer.score(
        preflight_blueprint.image,
        COLLECTION_PAYLOAD,
    )
finally:
    preflight_scorer.close()
if not (
    preflight_score.preset_count == 37
    and preflight_score.conservative_exact_presets == 37
    and preflight_score.direct_exact_all_repetitions
):
    raise RuntimeError(
        'Préflight QR-Verify local invalide sur le QR binaire exact : '
        f'{preflight_score.conservative_exact_presets}/{preflight_score.preset_count}.'
    )
local_qr_verify_binding = {
    'engine_version': preflight_score.engine_version,
    'implementation_sha256': preflight_score.implementation_sha256,
    'scoring_version': preflight_score.scoring_version,
    'repetitions': preflight_score.repetitions,
    'preset_count': preflight_score.preset_count,
    'blueprint_image_sha256': preflight_score.image_sha256,
    'conservative_exact_presets': preflight_score.conservative_exact_presets,
    'direct_exact_all_repetitions': preflight_score.direct_exact_all_repetitions,
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
pipeline_states = {
    'diffqrcoder_stage1': 'stage1',
    'e033_public_demo_srpg': 'public_stage2_parent',
    'e034_equation_srmpgd_fp16': 'four_updates_fp16',
    'e034_equation_srmpgd_fp32': 'four_updates_fp32_primary',
}
prediction_contract = [{
    'plan_method_id': method_id,
    'source_method_id': method_id,
    'requested_source_output_variant': method_by_id[method_id]['output_variant'],
    'pipeline_state': pipeline_states[method_id],
    'role': 'e034_four_iteration_gate',
    'advisor_rank': position,
} for position, method_id in enumerate(METHOD_IDS, start=1)]
plan_material = {
    'experiment': EXPERIMENT,
    'payload_sha256': hashlib.sha256(COLLECTION_PAYLOAD.encode('utf-8')).hexdigest(),
    'payload_length': len(COLLECTION_PAYLOAD),
    'error_correction': ERROR_CORRECTION,
    'prompt': PROMPT,
    'negative_prompt': NEGATIVE_PROMPT,
    'seed': SEED,
    'methods': METHODS,
    'prediction_contract': prediction_contract,
    'milestones': MILESTONE_ITERATIONS,
    'runtime_binding': runtime_binding,
    'validation': schema['validation'],
    'local_qr_verify_preflight': local_qr_verify_binding,
    'quality_scoring': quality_plan_binding,
    'automatic_expansion_authorized': AUTOMATIC_EXPANSION_AUTHORIZED,
}
plan_id = canonical_sha256(plan_material)[:16]
campaign_request = LabCampaignCreate.model_validate({
    'name': f'E034 {plan_id} serre seed-{SEED}',
    'payload': COLLECTION_PAYLOAD,
    'error_correction': ERROR_CORRECTION,
    'prompts': [prompt_request],
    'seeds': [SEED],
    'methods': METHODS,
    'max_attempts': 1,
}).model_dump(mode='json')

predictions = tuple({
    'prompt_id': PROMPT['id'],
    'prompt_text': PROMPT['text'],
    'prompt_family': PROMPT['family'],
    **contract,
} for contract in prediction_contract)
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
assert plan.public['trial_count'] == 4
display(pd.DataFrame([{
    'plan': plan.plan_id,
    'campagnes reprenables': 1,
    'prompt': PROMPT['id'],
    'seed': SEED,
    'générations finales': 4,
    'élargissement automatique': AUTOMATIC_EXPANSION_AUTHORIZED,
}]))
"""
    ),
    markdown(
        """## 4. Exécuter ou reprendre

Le runner conserve le plan, l'état et le CSV exporté sous `/data`. Une campagne active est reprise
après une coupure, mais **aucune campagne terminale n'est automatiquement régénérée** : E034 utilise
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
    display(Markdown('### Progression E034'))
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


RUNNER_EXCEPTION = None
TECHNICAL_ARCHIVE_DOWNLOAD = None
try:
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
    runner_summary = runner.run() if RUN_E034 else runner.summary()
except Exception as exc:
    RUNNER_EXCEPTION = f'{type(exc).__name__}: {exc}'
    fallback_output = OUTPUT_ROOT / plan.plan_id
    fallback_output.mkdir(parents=True, exist_ok=True)
    fallback_plan = fallback_output / 'plan-redacted.json'
    fallback_predictions = fallback_output / 'advisor-predictions.jsonl'
    fallback_state = fallback_output / 'state.json'
    if not fallback_plan.is_file():
        atomic_json(fallback_plan, plan.public)
    if not fallback_predictions.is_file():
        atomic_text(
            fallback_predictions,
            ''.join(
                json.dumps(item, ensure_ascii=False, allow_nan=False) + '\\n'
                for item in predictions
            ),
        )
    if not fallback_state.is_file():
        atomic_json(fallback_state, {
            'plan_id': plan.plan_id,
            'status': 'runner_exception',
            'history': [],
            'attempts': {},
            'failed_campaigns': [],
            'active_campaign': None,
            'error': RUNNER_EXCEPTION,
        })
    runner = SimpleNamespace(
        output_dir=fallback_output,
        plan_path=fallback_plan,
        predictions_path=fallback_predictions,
        state_path=fallback_state,
        exports_dir=fallback_output / 'exports',
    )
    runner.exports_dir.mkdir(exist_ok=True)
    runner_summary = {
        'plan_id': plan.plan_id,
        'status': 'runner_exception',
        'error': RUNNER_EXCEPTION,
        'campaigns': 1,
        'completed_campaigns': 0,
        'failed_campaigns': [0],
        'exports': len(list(runner.exports_dir.glob('*.csv'))),
        'state_path': str(fallback_state),
    }
display(pd.DataFrame([runner_summary]).T.rename(columns={0: 'valeur'}))
E034_TECHNICAL_STOP = runner_summary['status'] != 'completed'

if E034_TECHNICAL_STOP:
    # Le runner strict a consommé son unique essai et son état fail-fast
    # interdit toute nouvelle soumission au prochain Run All. On transforme donc l'arrêt
    # en diagnostic téléchargeable au lieu de masquer la vraie erreur par une exception.
    diagnostic_root = runner.output_dir / 'technical-failure'
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    state_read_error = None
    try:
        state = json.loads(runner.state_path.read_text(encoding='utf-8'))
    except Exception as exc:
        state_read_error = f'{type(exc).__name__}: {exc}'
        state = {
            'plan_id': plan.plan_id,
            'status': 'unreadable_state',
            'history': [],
            'attempts': {},
            'failed_campaigns': [],
            'active_campaign': None,
            'error': state_read_error,
        }
        atomic_text(diagnostic_root / 'state-read-error.txt', state_read_error + '\\n')
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
    control_copy_rows = []
    for source in [runner.plan_path, runner.predictions_path, runner.state_path]:
        copied = False
        copy_error = None
        try:
            if source.is_file():
                atomic_copy_once(source, diagnostic_root / source.name)
                copied = True
            else:
                copy_error = 'source absente'
        except Exception as exc:
            copy_error = f'{type(exc).__name__}: {exc}'
        control_copy_rows.append({
            'source': str(source),
            'copied': copied,
            'error': copy_error,
        })
    pd.DataFrame(control_copy_rows).to_csv(
        diagnostic_root / 'control-copy-diagnostics.csv', index=False
    )
    atomic_json_once(diagnostic_root / 'api-runtime.json', api_runtime)
    atomic_json_once(diagnostic_root / 'lab-schema.json', schema)

    failure_manifest = {
        'experiment': EXPERIMENT,
        'plan_id': plan.plan_id,
        'created_at': datetime.now(UTC).isoformat(),
        'runner_summary': runner_summary,
        'runtime_binding': runtime_binding,
        'attempts': state.get('attempts', {}),
        'active_campaign': state.get('active_campaign'),
        'failed_campaigns': state.get('failed_campaigns', []),
        'history': state.get('history', []),
        'state_read_error': state_read_error,
        'control_copies': control_copy_rows,
        'export_diagnostics': export_rows,
        'failed_trials': failure_rows,
        'remote_campaigns': remote_rows,
        'next_action': 'inspect_archive_without_regenerating',
    }
    atomic_json_once(diagnostic_root / 'technical-failure.json', failure_manifest)
    incident_manifest = prepare_incident_bundle(
        diagnostic_root,
        kind='technical_failure',
        experiment=EXPERIMENT,
        plan_id=plan.plan_id,
        identity_material=failure_manifest,
    )
    primary_technical_archive = ARCHIVE_ROOT / (
        f'{plan.plan_id}-{EXPERIMENT}-technical-failure.tar.gz'
    )
    technical_result = resolve_incident_archive(
        primary_technical_archive,
        primary_prefix=f'{plan.plan_id}-{EXPERIMENT}-technical-failure',
        bundle_root=diagnostic_root,
        manifest=incident_manifest,
    )
    technical_archive = technical_result.path
    technical_sha256 = sha256_file(technical_archive)
    technical_checksum = Path(f'{technical_archive}.sha256')
    atomic_text(
        technical_checksum,
        f'{technical_sha256}  {technical_archive.name}\\n',
    )
    for source in [technical_archive, technical_checksum]:
        shutil.copy2(source, DOWNLOAD_ROOT / source.name)
    TECHNICAL_ARCHIVE_DOWNLOAD = DOWNLOAD_ROOT / technical_archive.name

    display(Markdown('### STOP technique E034 — aucune nouvelle génération'))
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
        'archive pour corriger la cause avant de créer un nouveau plan. '
        f'Identité incident : `{incident_manifest["incident_identity_sha256"]}`.'
    ))
else:
    display(Markdown('**Campagne complète : audit scientifique E034 autorisé.**'))

E034_POST_GPU_STOP = False
POST_GPU_ARCHIVE_DOWNLOAD = None


def archive_post_gpu_failure(section, exc):
    error = f'{type(exc).__name__}: {exc}'
    fingerprint = canonical_sha256({
        'plan_id': plan.plan_id,
        'section': section,
        'error': error,
    })[:12]
    failure_root = runner.output_dir / 'post-gpu-failure' / fingerprint
    failure_root.mkdir(parents=True, exist_ok=True)
    failure_payload = {
        'experiment': EXPERIMENT,
        'plan_id': plan.plan_id,
        'section': section,
        'error': error,
        'created_at': datetime.now(UTC).isoformat(),
        'runner_summary': runner_summary,
        'runtime_binding': runtime_binding,
        'next_action': 'inspect_archive_without_regenerating_gpu_trials',
    }
    atomic_json_once(failure_root / 'failure.json', failure_payload)
    snapshot_tree_once(
        runner.output_dir,
        failure_root / 'evidence',
        excluded_top_level={'post-gpu-failure'},
    )
    incident_manifest = prepare_incident_bundle(
        failure_root,
        kind='post_gpu_failure',
        experiment=EXPERIMENT,
        plan_id=plan.plan_id,
        identity_material=failure_payload,
    )
    prefix = f'{plan.plan_id}-{EXPERIMENT}-post-gpu-{fingerprint}'
    archive_result = resolve_incident_archive(
        ARCHIVE_ROOT / f'{prefix}.tar.gz',
        primary_prefix=prefix,
        bundle_root=failure_root,
        manifest=incident_manifest,
    )
    archive = archive_result.path
    archive_sha256 = sha256_file(archive)
    checksum = Path(f'{archive}.sha256')
    atomic_text(checksum, f'{archive_sha256}  {archive.name}\\n')
    for source in [archive, checksum]:
        shutil.copy2(source, DOWNLOAD_ROOT / source.name)
    return DOWNLOAD_ROOT / archive.name
"""
    ),
    markdown("## 5. Audit d'appariement exact avant toute comparaison"),
    code(
        """RUN_DIR = runner.output_dir / 'analysis'
RUN_DIR.mkdir(parents=True, exist_ok=True)
rows = load_advisor_inference_results(runner.output_dir)
frame = pd.DataFrame(rows)
observed_methods = set(frame['method_id']) if 'method_id' in frame else set()
if len(frame) != 4 or observed_methods != set(METHOD_IDS):
    raise RuntimeError(f'Matrice E034 incomplète : {len(frame)}/4, {sorted(observed_methods)}')
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
    if method_id == 'e033_public_demo_srpg':
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
pairing.to_csv(RUN_DIR / 'e034-pairing-audit.csv', index=False)
frame.to_csv(RUN_DIR / 'e034-final-results.csv', index=False)
PAIRING_EXACT = bool(pairing[['stage1_exact', 'stage2_exact']].all(axis=None))
if not PAIRING_EXACT:
    display(pairing[~pairing[['stage1_exact', 'stage2_exact']].all(axis=1)])
    display(Markdown(
        '**STOP appariement :** une sortie ne prouve pas son parent exact. '
        'Les artefacts seront archivés, sans conclusion causale.'
    ))

reproduction_checks = {
    'stage1_raster_sha256': stage1.final_image_sha256,
    'stage2_parent_raster_sha256': parent.final_image_sha256,
    'stage2_parent_latent_sha256': parent.stage2_latent_sha256,
}
PARENT_REPRODUCED = reproduction_checks == E033_EXPECTED
if not PARENT_REPRODUCED:
    atomic_json(RUN_DIR / 'e034-parent-reproduction-stop.json', {
        'expected': E033_EXPECTED,
        'observed': reproduction_checks,
        'verdict': 'STOP_parent_not_reproduced',
    })
    display(Markdown(
        '**STOP reproduction :** le parent E033 gelé n est pas reproduit bit à bit. '
        'L audit longitudinal est interdit, mais le diagnostic sera archivé.'
    ))
display(pairing)
print('Appariement exact =', PAIRING_EXACT, '; parent E033 reproduit =', PARENT_REPRODUCED)
"""
    ),
    markdown("## 6. Télécharger et afficher les quatre sorties finales"),
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


fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
for axis, method_id in zip(axes, METHOD_IDS, strict=True):
    entry = gallery_by_method[method_id]
    with Image.open(entry['local_image']) as source:
        axis.imshow(source.convert('RGB'))
    axis.set_title(final_title(entry), fontsize=8)
    axis.axis('off')
fig.suptitle(f"E034 — {PROMPT['id']} — seed {SEED}", fontsize=12)
fig.tight_layout()
final_sheet = RUN_DIR / 'e034-final-contact-sheet.png'
fig.savefig(final_sheet, dpi=150, bbox_inches='tight')
plt.close(fig)
display(NotebookImage(filename=str(final_sheet), width=1400))
"""
    ),
    markdown(
        """## 7. Télécharger directement les états 000, 001, 002 et 004

Cette cellule n'interroge volontairement **pas** `/artifacts`. Elle appelle directement
`/variants/srmpgd_iteration_000`, puis 001, 002 et 004. Ainsi, l'itération 0 reste téléchargeable
même lorsque son contenu est identique au Stage 2 parent et que le catalogue la déduplique.
"""
    ),
    code(
        """MILESTONE_METHOD_IDS = [
    'e034_equation_srmpgd_fp16',
    'e034_equation_srmpgd_fp32',
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
        'scanning_robust_loss': trace_by_iteration.get(0, {}).get(
            'scanning_robust_loss'
        ),
        'lpips_loss': trace_by_iteration.get(0, {}).get('lpips_loss'),
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
            'trace_image_sha256': trace_step.get('image_sha256'),
            'scanning_robust_loss': trace_step.get('scanning_robust_loss'),
            'gradient_rms': trace_step.get('gradient_rms'),
            'gradient_scale': trace_step.get('gradient_scale'),
            'image_gradient_rms': trace_step.get('image_gradient_rms'),
            'lpips_image_gradient_rms': trace_step.get('lpips_image_gradient_rms'),
            'weighted_lpips_image_gradient_rms': trace_step.get(
                'weighted_lpips_image_gradient_rms'
            ),
            'objective_image_gradient_rms': trace_step.get(
                'objective_image_gradient_rms'
            ),
            'lpips_loss': trace_step.get('lpips_loss'),
            'latent_delta_rms': trace_step.get('latent_delta_rms'),
            'next_step_rms': trace_step.get('next_step_rms'),
            'applied_step_rms': trace_step.get('applied_step_rms'),
            'step_scale': trace_step.get('step_scale'),
            'objective': trace_step.get('objective'),
            'actual_module_error_rate': trace_step.get('actual_module_error_rate'),
        })

milestones = pd.DataFrame(milestone_rows)
redecodes = pd.DataFrame(redecode_rows)
if len(milestones) != len(MILESTONE_METHOD_IDS) * len(MILESTONE_ITERATIONS):
    raise RuntimeError('Le journal des huit requêtes directes E034 est incomplet.')
milestones.to_csv(RUN_DIR / 'e034-milestones.csv', index=False)
redecodes.to_csv(RUN_DIR / 'e034-vae-redecode-controls.csv', index=False)
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
DIRECT_MILESTONE_HASHES_VERIFIED = bool(
    milestones.available.all()
    and milestones.trace_image_sha256.notna().all()
    and (milestones.image_raster_sha256 == milestones.trace_image_sha256).all()
)
if DIRECT_ITERATION_ZERO_AVAILABLE and not DIRECT_ITERATION_ZERO_EXACT:
    display(Markdown(
        '**STOP raster i0 :** le jalon direct n est pas le Stage 2 parent exact.'
    ))
print(
    'Requêtes directes terminées :', int(milestones.available.sum()), '/ 8 jalons présents ;',
    'itérations 0 exactes =', DIRECT_ITERATION_ZERO_EXACT,
    '; hashes jalons =', DIRECT_MILESTONE_HASHES_VERIFIED,
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
    markdown("## 8. Rescorer le témoin VAE et chaque jalon avec QR-Verify"),
    code(
        """blueprint = generate_diffqrcoder_qr(
    COLLECTION_PAYLOAD,
    ERROR_CORRECTION,
    version=3,
    mask_pattern=4,
    module_size=20,
    border=4,
)
scorer = ConservativeQRVerifyScorer(
    repetitions=3,
    cache_dir=RUN_DIR / 'qr-verify-cache',
)
local_score_rows = []
local_score_details = []

with Image.open(gallery_by_method['e033_public_demo_srpg']['local_image']) as source:
    parent_image = source.convert('RGB')


def score_local_image(kind, method_id, iteration, path):
    record = {
        'kind': kind,
        'method_id': method_id,
        'iteration': iteration,
        'local_image': str(path) if path else None,
        'score_error': None,
        'raster_sha256': None,
        'qr_verify_exact_presets': None,
        'qr_verify_preset_count': None,
        'qr_verify_tolerance': None,
        'qr_verify_consistent_any_exact': None,
        'module_error_rate': None,
        'mean_absolute_change': None,
        'clipped_pixel_ratio_increase': None,
        'saturation_mean_increase': None,
        'high_saturation_ratio_increase': None,
    }
    if not path:
        record['score_error'] = 'raster absent'
        local_score_rows.append(record)
        return
    try:
        with Image.open(path) as source:
            image = source.convert('RGB')
        qr_score = scorer.score(image, COLLECTION_PAYLOAD)
        quality = image_quality_metrics(image)
        change = image_change_metrics(image, parent_image)
        record.update({
            'raster_sha256': image_raster_sha256(image),
            'qr_verify_exact_presets': qr_score.conservative_exact_presets,
            'qr_verify_preset_count': qr_score.preset_count,
            'qr_verify_tolerance': qr_score.conservative_tolerance_score,
            'qr_verify_consistent_any_exact': qr_score.consistent_any_exact,
            'module_error_rate': diffqrcoder_module_error_rate(
                image, blueprint, padding_px=78, module_size=20
            ),
            **quality,
            **change,
        })
        local_score_details.append({
            'kind': kind,
            'method_id': method_id,
            'iteration': iteration,
            'qr_verify': canonical_conservative_qr_verify_evidence(qr_score),
        })
    except Exception as exc:
        record['score_error'] = f'{type(exc).__name__}: {exc}'
    local_score_rows.append(record)


score_local_image(
    'stage2_parent', 'e033_public_demo_srpg', None,
    gallery_by_method['e033_public_demo_srpg']['local_image'],
)
for row in redecodes.itertuples(index=False):
    score_local_image(
        'vae_redecode', row.method_id, 0,
        row.local_image if bool(row.available) else None,
    )
for row in milestones.itertuples(index=False):
    score_local_image(
        'srmpgd_milestone', row.method_id, int(row.iteration),
        row.local_image if bool(row.available) else None,
    )
scorer.close()

local_scores = pd.DataFrame(local_score_rows)
local_scores.to_csv(RUN_DIR / 'e034-local-raster-scores.csv', index=False)
atomic_text(
    RUN_DIR / 'e034-local-qr-verify-details.jsonl',
    ''.join(
        json.dumps(item, ensure_ascii=False, default=str) + '\\n'
        for item in local_score_details
    ),
)
LOCAL_SCORING_COMPLETE = bool(
    len(local_scores) == 11
    and local_scores.score_error.fillna('').str.strip().eq('').all()
)
display(local_scores.reindex(columns=[
    'kind', 'method_id', 'iteration', 'qr_verify_exact_presets',
    'qr_verify_preset_count', 'qr_verify_tolerance', 'module_error_rate',
    'mean_absolute_change', 'clipped_pixel_ratio_increase',
    'saturation_mean_increase', 'high_saturation_ratio_increase', 'score_error',
]))
if not LOCAL_SCORING_COMPLETE:
    display(Markdown(
        '**STOP mesure locale :** au moins un raster n a pas pu être rescanné. '
        'Les images et les erreurs restent archivées.'
    ))
"""
    ),
    markdown("## 9. Planches des jalons — à regarder avant le verdict"),
    code(
        """trace_lookup = {
    (row.method_id, int(row.iteration)): row
    for row in milestones.itertuples(index=False)
}
redecode_lookup = {
    row.method_id: row for row in redecodes.itertuples(index=False)
}
local_score_lookup = {
    (row.kind, row.method_id, None if pd.isna(row.iteration) else int(row.iteration)): row
    for row in local_scores.itertuples(index=False)
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
        redecode_local = local_score_lookup[('vae_redecode', method_id, 0)]
        axes[row_index, 1].set_title(
            'VAE redécodé, aucun pas\\n'
            f"QRV={finite(redecode_local.qr_verify_tolerance) or 0:.1%}  "
            f"MER={finite(redecode_local.module_error_rate) or 0:.1%}\\n"
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
            local = local_score_lookup[('srmpgd_milestone', method_id, iteration)]
            axes[row_index, column].set_title(
                f"i={iteration}  SRL={finite(record.scanning_robust_loss) or 0:.6f}\\n"
                f"QRV={finite(local.qr_verify_tolerance) or 0:.1%}  "
                f"MER={finite(local.module_error_rate) or 0:.1%}\\n"
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
fig.suptitle('E034 — quatre mises à jour SR-MPGD strictement appariées', fontsize=12)
fig.tight_layout()
milestone_sheet = RUN_DIR / 'e034-milestone-contact-sheet.png'
fig.savefig(milestone_sheet, dpi=160, bbox_inches='tight')
plt.close(fig)
display(NotebookImage(filename=str(milestone_sheet), width=1400))
display(milestones.reindex(columns=[
    'method_id', 'iteration', 'scanning_robust_loss', 'gradient_rms',
    'gradient_scale', 'image_gradient_rms', 'lpips_loss',
    'lpips_image_gradient_rms', 'weighted_lpips_image_gradient_rms',
    'objective_image_gradient_rms', 'latent_delta_rms',
    'next_step_rms', 'applied_step_rms', 'step_scale', 'available',
    'stop_reason', 'download_error',
]))
display(redecodes.reindex(columns=[
    'method_id', 'available', 'image_raster_sha256', 'trace_image_sha256',
    'scanning_robust_loss', 'lpips_loss',
    'mean_absolute_change_from_stage2', 'download_error',
]))
"""
    ),
    markdown("## 10. Portes FP32 — mécanisme, scan et préservation visuelle"),
    code(
        """PRIMARY_METHOD = 'e034_equation_srmpgd_fp32'
FP16_METHOD = 'e034_equation_srmpgd_fp16'
primary_trace = traces[PRIMARY_METHOD]
primary_steps = {
    int(item['iteration']): item for item in primary_trace['steps']
}
primary_milestones = milestones[milestones.method_id == PRIMARY_METHOD]
TRACE_COMPLETE = sorted(primary_steps) == [0, 1, 2, 3, 4]
PRIMARY_MILESTONES_AVAILABLE = bool(
    len(primary_milestones) == len(MILESTONE_ITERATIONS)
    and primary_milestones.available.all()
)
TRACE_PROTOCOL_EXACT = bool(
    primary_trace.get('protocol') == 'paper_equations'
    and primary_trace.get('selected_iteration') == 4
    and primary_trace.get('stop_reason') == 'max_iterations'
)


def positive_trace_value(iteration, field):
    value = finite(primary_steps.get(iteration, {}).get(field))
    return value is not None and value > 0


UPDATE_GRADIENTS_VALID = bool(
    TRACE_COMPLETE
    and all(
        positive_trace_value(iteration, field)
        for iteration in range(4)
        for field in [
            'gradient_rms', 'image_gradient_rms',
            'objective_image_gradient_rms', 'applied_step_rms',
        ]
    )
)
LPIPS_GRADIENTS_VALID = bool(
    TRACE_COMPLETE
    and all(
        positive_trace_value(iteration, field)
        for iteration in [1, 2, 3]
        for field in [
            'lpips_loss', 'lpips_image_gradient_rms',
            'weighted_lpips_image_gradient_rms',
        ]
    )
)
LPIPS_INITIAL_ZERO = bool(
    TRACE_COMPLETE
    and all(
        finite(primary_steps[0].get(field)) is not None
        and abs(finite(primary_steps[0].get(field))) <= tolerance
        for field, tolerance in [
            ('lpips_loss', 1e-8),
            ('lpips_image_gradient_rms', 1e-10),
            ('weighted_lpips_image_gradient_rms', 1e-12),
        ]
    )
)
LPIPS_REFERENCE_PAPER = bool(
    primary_trace.get('lpips_reference_mode') == 'paper_stage2_float'
    and primary_trace.get('lpips_reference_image_sha256')
        == primary_trace.get('initial_redecoded_image_sha256')
)


def lpips_weighting_consistent(iteration):
    lpips_gradient = finite(
        primary_steps.get(iteration, {}).get('lpips_image_gradient_rms')
    )
    weighted_gradient = finite(
        primary_steps.get(iteration, {}).get('weighted_lpips_image_gradient_rms')
    )
    return (
        lpips_gradient is not None
        and weighted_gradient is not None
        and math.isclose(
            weighted_gradient,
            0.01 * lpips_gradient,
            rel_tol=1e-5,
            abs_tol=1e-12,
        )
    )


LPIPS_WEIGHTING_CONSISTENT = bool(
    TRACE_COMPLETE
    and all(lpips_weighting_consistent(iteration) for iteration in range(4))
)
LATENT_DELTAS_VALID = bool(
    TRACE_COMPLETE
    and all(positive_trace_value(iteration, 'latent_delta_rms') for iteration in range(1, 5))
)


def paper_step_consistent(iteration):
    requested = finite(primary_steps.get(iteration, {}).get('next_step_rms'))
    applied = finite(primary_steps.get(iteration, {}).get('applied_step_rms'))
    scale = finite(primary_steps.get(iteration, {}).get('step_scale'))
    return (
        requested is not None
        and requested > 0
        and applied is not None
        and scale is not None
        and math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(applied, requested, rel_tol=1e-5, abs_tol=1e-10)
    )


PAPER_STEPS_UNCLIPPED_AND_CONSISTENT = bool(
    TRACE_COMPLETE
    and all(paper_step_consistent(iteration) for iteration in range(4))
)


def objective_identity(index):
    objective = finite(primary_steps.get(index, {}).get('objective'))
    scanning_loss = finite(
        primary_steps.get(index, {}).get('scanning_robust_loss')
    )
    lpips_loss = finite(primary_steps.get(index, {}).get('lpips_loss'))
    return (
        objective is not None
        and scanning_loss is not None
        and lpips_loss is not None
        and math.isclose(
            objective,
            scanning_loss + 0.01 * lpips_loss,
            rel_tol=1e-5,
            abs_tol=1e-8,
        )
    )


OBJECTIVES_CONSISTENT = bool(
    TRACE_COMPLETE
    and all(objective_identity(index) for index in range(5))
)
objectives = [finite(primary_steps.get(index, {}).get('objective')) for index in range(5)]
srl_values = [
    finite(primary_steps.get(index, {}).get('scanning_robust_loss')) for index in range(5)
]
OBJECTIVE_FINAL_LOWER = bool(
    None not in objectives and objectives[-1] < objectives[0]
)
SRL_FINAL_LOWER = bool(
    None not in srl_values and srl_values[-1] < srl_values[0]
)
OBJECTIVE_MONOTONIC_STEPS = (
    sum(right <= left + 1e-8 for left, right in zip(objectives, objectives[1:]))
    if None not in objectives else 0
)

primary_i4 = primary_milestones[primary_milestones.iteration == 4]
FINAL_RASTER_IS_I4 = bool(
    len(primary_i4) == 1
    and bool(primary_i4.iloc[0].available)
    and primary_i4.iloc[0].image_raster_sha256 == by_method[PRIMARY_METHOD].final_image_sha256
)

fp16_i4 = milestones[(milestones.method_id == FP16_METHOD) & (milestones.iteration == 4)]
fp32_i4 = primary_i4
FP16_FP32_PSNR = None
FP16_FP32_MAX_CHANNEL_DELTA = None
FP16_FP32_IDENTICAL = False
if (
    len(fp16_i4) == 1
    and len(fp32_i4) == 1
    and fp16_i4.iloc[0].available
    and fp32_i4.iloc[0].available
):
    with Image.open(fp16_i4.iloc[0].local_image) as source:
        fp16_array = np.asarray(source.convert('RGB'), dtype=np.float32) / 255.0
    with Image.open(fp32_i4.iloc[0].local_image) as source:
        fp32_array = np.asarray(source.convert('RGB'), dtype=np.float32) / 255.0
    mse = float(np.mean((fp16_array - fp32_array) ** 2))
    FP16_FP32_IDENTICAL = mse == 0.0
    if not FP16_FP32_IDENTICAL:
        FP16_FP32_PSNR = -10.0 * math.log10(mse)
    FP16_FP32_MAX_CHANNEL_DELTA = float(np.max(np.abs(fp16_array - fp32_array)))
FP16_FP32_CONSISTENT = bool(
    (FP16_FP32_IDENTICAL or (
        FP16_FP32_PSNR is not None and FP16_FP32_PSNR >= 35.0
    ))
    and FP16_FP32_MAX_CHANNEL_DELTA is not None
    and FP16_FP32_MAX_CHANNEL_DELTA <= 8 / 255
)

parent_local = local_scores[
    (local_scores.kind == 'stage2_parent')
    & (local_scores.method_id == 'e033_public_demo_srpg')
].iloc[0]
final_local = local_scores[
    (local_scores.kind == 'srmpgd_milestone')
    & (local_scores.method_id == PRIMARY_METHOD)
    & (local_scores.iteration == 4)
].iloc[0]
visual_values = {
    key: finite(getattr(final_local, key))
    for key in [
        'mean_absolute_change', 'clipped_pixel_ratio_increase',
        'saturation_mean_increase', 'high_saturation_ratio_increase',
    ]
}
LOCAL_VISUAL_GUARDS_PASS = bool(
    LOCAL_SCORING_COMPLETE
    and all(value is not None for value in visual_values.values())
    and visual_values['mean_absolute_change']
        <= VISUAL_GUARDS['maximum_mean_absolute_change']
    and visual_values['clipped_pixel_ratio_increase']
        <= VISUAL_GUARDS['maximum_clipped_pixel_ratio_increase']
    and visual_values['saturation_mean_increase']
        <= VISUAL_GUARDS['maximum_saturation_mean_increase']
    and visual_values['high_saturation_ratio_increase']
        <= VISUAL_GUARDS['maximum_high_saturation_ratio_increase']
)
parent_api = by_method['e033_public_demo_srpg']
final_api = by_method[PRIMARY_METHOD]
parent_aesthetic = {
    key: finite(getattr(parent_api, key))
    for key in ['clip_aesthetic', 'clip_score', 'hpsv2_1']
}
final_aesthetic = {
    key: finite(getattr(final_api, key))
    for key in ['clip_aesthetic', 'clip_score', 'hpsv2_1']
}
AESTHETIC_PROXY_GUARDS_PASS = bool(
    all(value is not None for value in [*parent_aesthetic.values(), *final_aesthetic.values()])
    and final_aesthetic['clip_aesthetic'] >= parent_aesthetic['clip_aesthetic'] - 0.50
    and final_aesthetic['clip_score'] >= parent_aesthetic['clip_score'] - 0.05
    and final_aesthetic['hpsv2_1'] >= parent_aesthetic['hpsv2_1'] - 0.03
)
VISUAL_PROXY_PASS = LOCAL_VISUAL_GUARDS_PASS and AESTHETIC_PROXY_GUARDS_PASS
parent_qrv = finite(parent_local.qr_verify_tolerance)
final_qrv = finite(final_local.qr_verify_tolerance)
parent_mer = finite(parent_local.module_error_rate)
final_mer = finite(final_local.module_error_rate)
SCAN_PROGRESS_PASS = bool(
    LOCAL_SCORING_COMPLETE
    and all(value is not None for value in [parent_qrv, final_qrv, parent_mer, final_mer])
    and (
        final_qrv > parent_qrv
        or final_mer < parent_mer
    )
)
SINGLE_CASE_QR_VERIFY_PASS = bool(
    LOCAL_SCORING_COMPLETE
    and finite(final_local.qr_verify_exact_presets) is not None
    and finite(final_local.qr_verify_exact_presets) >= 1
)
gate_rows = [
    ('provenance', 'appariement_exact', PAIRING_EXACT, True),
    ('provenance', 'parent_e033_reproduit', PARENT_REPRODUCED, True),
    ('provenance', 'temoins_vae_verifies', REDECODE_CONTROLS_VERIFIED, True),
    ('provenance', 'jalons_directs_hashes_verifies', DIRECT_MILESTONE_HASHES_VERIFIED, True),
    ('provenance', 'iteration_zero_identique_parent', DIRECT_ITERATION_ZERO_EXACT, True),
    ('mecanisme', 'trace_exacte_i0_a_i4', TRACE_COMPLETE, True),
    ('mecanisme', 'reference_lpips_x0_float_du_papier', LPIPS_REFERENCE_PAPER, True),
    ('mecanisme', 'lpips_nulle_a_i0', LPIPS_INITIAL_ZERO, True),
    ('mecanisme', 'protocole_final_i4_sans_oracle', TRACE_PROTOCOL_EXACT, True),
    ('mecanisme', 'gradients_et_pas_i0_a_i3', UPDATE_GRADIENTS_VALID, True),
    ('mecanisme', 'gradient_lpips_actif_i1_a_i3', LPIPS_GRADIENTS_VALID, True),
    ('mecanisme', 'gradient_lpips_pondere_eq13', LPIPS_WEIGHTING_CONSISTENT, True),
    ('mecanisme', 'deplacements_latents_i1_a_i4', LATENT_DELTAS_VALID, True),
    (
        'mecanisme', 'pas_papier_non_ecretes_et_coherents',
        PAPER_STEPS_UNCLIPPED_AND_CONSISTENT, True,
    ),
    ('mecanisme', 'objectif_eq13_recompose', OBJECTIVES_CONSISTENT, True),
    ('mecanisme', 'objectif_final_inferieur_initial', OBJECTIVE_FINAL_LOWER, True),
    ('mecanisme', 'srl_finale_inferieure_initiale', SRL_FINAL_LOWER, True),
    ('mecanisme', 'sortie_finale_est_i4', FINAL_RASTER_IS_I4, True),
    ('numerique', 'fp16_fp32_coherents', FP16_FP32_CONSISTENT, 'PSNR>=35 dB et maxΔ<=8/255'),
    ('visuel_proxy', 'saturation_et_clipping_sous_gardes', LOCAL_VISUAL_GUARDS_PASS, VISUAL_GUARDS),
    (
        'visuel_proxy', 'clip_hps_sous_gardes', AESTHETIC_PROXY_GUARDS_PASS,
        'ΔAES>=-0.50; ΔCLIP>=-0.05; ΔHPS>=-0.03',
    ),
    ('scan', 'progres_qr_verify_ou_mer', SCAN_PROGRESS_PASS, True),
    ('scan', 'au_moins_un_preset_qr_verify_exact', SINGLE_CASE_QR_VERIFY_PASS, True),
]
gate_frame = pd.DataFrame(gate_rows, columns=['dimension', 'porte', 'réussie', 'référence'])
MECHANISM_PASS = bool(
    gate_frame[gate_frame.dimension.isin(['provenance', 'mecanisme', 'numerique'])]
    ['réussie'].all()
)
READY_FOR_MANUAL_REVIEW = bool(
    MECHANISM_PASS
    and SCAN_PROGRESS_PASS
    and SINGLE_CASE_QR_VERIFY_PASS
    and VISUAL_PROXY_PASS
)
mechanism_verdict = {
    'experiment': EXPERIMENT,
    'plan_id': plan.plan_id,
    'primary_method': PRIMARY_METHOD,
    'mechanism_pass': MECHANISM_PASS,
    'scan_progress_pass': SCAN_PROGRESS_PASS,
    'single_case_qr_verify_pass': SINGLE_CASE_QR_VERIFY_PASS,
    'visual_proxy_pass': VISUAL_PROXY_PASS,
    'ready_for_manual_review': READY_FOR_MANUAL_REVIEW,
    'manual_visual_review_required': True,
    'production_ready': False,
    'automatic_expansion_authorized': False,
    'trace_iterations': sorted(primary_steps),
    'objective_values': objectives,
    'scanning_robust_loss_values': srl_values,
    'objective_monotonic_steps_out_of_four': OBJECTIVE_MONOTONIC_STEPS,
    'fp16_fp32_psnr_db': FP16_FP32_PSNR,
    'fp16_fp32_identical': FP16_FP32_IDENTICAL,
    'fp16_fp32_max_channel_delta': FP16_FP32_MAX_CHANNEL_DELTA,
    'parent_qr_verify_tolerance': parent_qrv,
    'final_qr_verify_tolerance': final_qrv,
    'parent_module_error_rate': parent_mer,
    'final_module_error_rate': final_mer,
    'next_action': (
        'manual_visual_review_then_small_holdout'
        if READY_FOR_MANUAL_REVIEW
        else 'stop_without_expansion_and_inspect_the_failed_dimension'
    ),
}
gate_frame.to_csv(RUN_DIR / 'e034-primary-fp32-gates.csv', index=False)
atomic_json(RUN_DIR / 'e034-mechanism-verdict.json', mechanism_verdict)
display(gate_frame)
display(pd.DataFrame([mechanism_verdict]).T.rename(columns={0: 'valeur'}))
if not READY_FOR_MANUAL_REVIEW:
    display(Markdown(
        '**STOP scientifique local :** mécanisme, QR-Verify ou garde visuelle insuffisant. '
        'Aucun élargissement automatique ; l archive complète sera tout de même créée.'
    ))
"""
    ),
    markdown("## 11. Rapport, manifeste et archive — même si une porte échoue"),
    code(
        """report_path = RUN_DIR / 'e034-report.md'
report = f'''# Rapport E034 — porte SR-MPGD à quatre itérations

- Plan : `{plan.plan_id}`
- Cas E033 reproduit : `{PROMPT['id']}`, seed `{SEED}`, même payload court
- Comparaison : Stage 1, Stage 2 public parent, SR-MPGD quatre pas FP16 et FP32
- Témoins VAE sans mise à jour : FP16 et FP32
- Jalons directs : {MILESTONE_ITERATIONS}
- Mécanisme FP32 : `{MECHANISM_PASS}`
- Progrès QR : `{SCAN_PROGRESS_PASS}`
- Au moins un preset QR-Verify exact : `{SINGLE_CASE_QR_VERIFY_PASS}`
- Gardes visuelles automatiques : `{VISUAL_PROXY_PASS}`
- Revue visuelle humaine : requise
- Production : `False`
- Élargissement automatique : `False`

## Verdict séparé

```json
{json.dumps(mechanism_verdict, ensure_ascii=False, indent=2)}
```

## Interprétation autorisée

E034 mesure sur le parent E033 exact si quatre applications fixes des Eq. 13-14 restent
numériquement actives, si LPIPS est nulle sur sa référence flottante `x0=D(z0)` puis contribue
après la première mise à jour, si la SRL et le MER progressent, et si les
changements restent sous des gardes de saturation, clipping et proxies esthétiques. Les planches
montrent séparément le parent, son redécodage VAE et i0/i1/i2/i4.

## Interprétation interdite

Même un PASS complet ne prouve ni la généralisation, ni une probabilité de scan téléphone, ni la
supériorité esthétique. QR-Verify est un banc logiciel ; la décision esthétique reste humaine.
Un STOP indique précisément la dimension à corriger et interdit tout élargissement automatique.
'''
atomic_text(report_path, report)

control_dir = RUN_DIR / 'control-plane'
control_dir.mkdir(exist_ok=True)
for source in [runner.plan_path, runner.predictions_path, runner.state_path]:
    atomic_copy_once(source, control_dir / source.name)
atomic_json_once(control_dir / 'api-runtime.json', api_runtime)
atomic_json_once(control_dir / 'lab-schema.json', schema)
exports_copy = control_dir / 'exports'
exports_copy.mkdir(exist_ok=True)
for source in sorted(runner.exports_dir.glob('*.csv')):
    atomic_copy_once(source, exports_copy / source.name)

artifact_files = sorted(
    path for path in RUN_DIR.rglob('*')
    if path.is_file() and path.name != 'e034-artifact-manifest.json'
    and not path.name.endswith('.tmp')
)
manifest_material = {
    'experiment': EXPERIMENT,
    'plan': plan.public,
    'runner': runner_summary,
    'runtime_binding': runtime_binding,
    'final_rasters': len(frame),
    'direct_milestone_rasters': int(milestones.available.sum()),
    'direct_vae_redecode_rasters': int(redecodes.available.sum()),
    'required_milestone_iterations': MILESTONE_ITERATIONS,
    'pairing_rows': len(pairing),
    'pairing_exact': PAIRING_EXACT,
    'parent_e033_reproduced': PARENT_REPRODUCED,
    'local_scoring_complete': LOCAL_SCORING_COMPLETE,
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
manifest = {
    **manifest_material,
    'created_at': datetime.now(UTC).isoformat(),
    'analysis_identity_sha256': canonical_sha256(manifest_material),
}
manifest_path = RUN_DIR / 'e034-artifact-manifest.json'
atomic_json(manifest_path, manifest)

archive_path = ARCHIVE_ROOT / f'{plan.plan_id}-{EXPERIMENT}.tar.gz'
archive_prefix = f'{plan.plan_id}-{EXPERIMENT}'


def verify_archive(path):
    with tarfile.open(path, 'r:gz') as bundle:
        archived_names = set(bundle.getnames())
        expected_manifest_name = f'{archive_prefix}/e034-artifact-manifest.json'
        manifest_stream = bundle.extractfile(expected_manifest_name)
        if manifest_stream is None:
            raise RuntimeError('Archive E034 invalide : manifeste absent.')
        archived_manifest = json.loads(manifest_stream.read().decode('utf-8'))
        if archived_manifest.get('plan', {}).get('plan_id') != plan.plan_id:
            raise RuntimeError('Archive E034 invalide : plan différent.')
        for relative_path, expected_sha256 in archived_manifest[
            'artifact_checksums'
        ].items():
            member_name = f'{archive_prefix}/{relative_path}'
            if member_name not in archived_names:
                raise RuntimeError(f'Archive E034 incomplète : {member_name}')
            member = bundle.extractfile(member_name)
            if member is None:
                raise RuntimeError(f'Archive E034 illisible : {member_name}')
            actual_sha256 = hashlib.sha256(member.read()).hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(f'Checksum E034 invalide : {member_name}')
        return archived_manifest


if not archive_path.is_file():
    temporary_archive = Path(f'{archive_path}.tmp')
    with tarfile.open(temporary_archive, 'w:gz') as bundle:
        bundle.add(RUN_DIR, arcname=archive_prefix)
    os.replace(temporary_archive, archive_path)
else:
    print('Archive E034 existante : vérification sans réécriture ni génération GPU.')
archived_manifest = verify_archive(archive_path)
if archived_manifest.get('analysis_identity_sha256') != manifest[
    'analysis_identity_sha256'
]:
    raise RuntimeError(
        'Archive E034 existante mais obsolète pour cette analyse ; '
        'elle n est ni écrasée ni présentée comme résultat courant.'
    )
with tarfile.open(archive_path, 'r:gz') as bundle:
    archived_names = set(bundle.getnames())
    expected_manifest_name = f'{archive_prefix}/e034-artifact-manifest.json'
    if expected_manifest_name not in archived_names:
        raise RuntimeError('Archive E034 invalide : manifeste absent.')
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
    markdown("## 12. Lecture finale"),
    code(
        """if READY_FOR_MANUAL_REVIEW:
    display(Markdown(
        '**PASS local sous réserve de la revue humaine.** Inspectez les deux planches. '
        'Si l esthétique est acceptable, le prochain travail est un petit holdout séparé ; '
        'E034 ne lance aucune expansion automatiquement.'
    ))
else:
    display(Markdown(
        '**STOP scientifique archivé.** Consultez la colonne `dimension` des portes pour savoir '
        'si le problème vient de la provenance, du mécanisme, du scan ou du visuel. '
        'Ne lancez pas de campagne multi-prompt avec ce résultat.'
    ))
print('Archive :', DOWNLOAD_ROOT / archive_path.name)
"""
    ),
]

# Lors d'un STOP technique, les cellules scientifiques suivantes doivent s'ignorer proprement :
# la cellule d'exécution a déjà produit l'archive de diagnostic et le runner fail-fast n'enverra
# aucune nouvelle campagne lors d'un nouveau Run All.
scientific_section = False
scientific_code_index = 0
for cell in cells:
    source = "".join(cell.get("source", []))
    if cell["cell_type"] == "markdown" and source.startswith("## 5."):
        scientific_section = True
    if scientific_section and cell["cell_type"] == "code":
        scientific_code_index += 1
        try_indented = "".join(
            f"        {line}" if line.strip() else line
            for line in source.splitlines(keepends=True)
        )
        guarded = (
            "if E034_TECHNICAL_STOP:\n"
            "    display(Markdown(\n"
            "        f'**Cellule ignorée après le STOP technique.** Archive : ' \n"
            "        f'`{TECHNICAL_ARCHIVE_DOWNLOAD}`'\n"
            "    ))\n"
            "elif E034_POST_GPU_STOP:\n"
            "    display(Markdown(\n"
            "        f'**Cellule ignorée après le STOP post-GPU.** Archive : ' \n"
            "        f'`{POST_GPU_ARCHIVE_DOWNLOAD}`'\n"
            "    ))\n"
            "else:\n"
            "    try:\n"
            f"{try_indented}"
            "    except Exception as exc:\n"
            "        E034_POST_GPU_STOP = True\n"
            "        POST_GPU_ARCHIVE_DOWNLOAD = archive_post_gpu_failure(\n"
            f"            'scientific_cell_{scientific_code_index:02d}', exc\n"
            "        )\n"
            "        display(Markdown(\n"
            "            '**STOP post-GPU archivé — aucune régénération.** ' \n"
            "            f'Erreur : `{type(exc).__name__}: {exc}`. ' \n"
            "            f'Archive : `{POST_GPU_ARCHIVE_DOWNLOAD}`'\n"
            "        ))\n"
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
