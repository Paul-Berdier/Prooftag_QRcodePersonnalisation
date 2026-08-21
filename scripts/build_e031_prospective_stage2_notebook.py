"""Build the E031 prospective, paired Stage-2 holdout notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "26_e031_prospective_stage2_holdout.ipynb"


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
        """# E031 — holdout prospectif Stage 2, QR-Verify fiable et revue humaine

E031 répond à une question précise : **la recette fixe, le conseiller par prompt ou une seconde
seed généralise-t-il le mieux à des prompts jamais vus ?**

Le protocole est gelé avant la première génération : 20 prompts simples et 20 atypiques, puis
trois branches appariées pour chacun :

1. Stage 2 fixe, seed A ;
2. Stage 2 conseillée, même seed A ;
3. Stage 2 fixe, seed B.

Chaque Stage 2 possède son Stage 1 exact comme prérequis, soit **240 essais API** dont 120 rasters
Stage 2 évalués. Stage 1 n'est jamais livré. SR-MPGD est interdit dans cette expérience. Les
120 rasters sont tous générés même si le premier passe : c'est indispensable pour mesurer sans
biais le conseiller et la nouvelle seed. Le coût conditionnel est ensuite rejoué hors ligne.

La porte finale utilise cinq répétitions de 37 presets QR-Verify, l'intersection des succès, le
payload exact et une garde de saturation. Deux seuils sont publiés : le seuil historique 30/37
et la cible stricte 36/37. HPS v2.1, CLIP-Aesthetic et CLIPScore ne départagent qu'après la porte
QR. Une galerie aveugle est créée pour la fidélité réelle au prompt.

**Ce notebook ne prouve pas une probabilité de scan téléphone et ne permet pas d'annoncer 99 %.**
La reprise est liée au plan, aux campagnes et aux rasters : après une coupure, relancer Run All.
"""
    ),
    markdown("## 0. Imports et utilitaires déterministes"),
    code(
        """# ruff: noqa: E402
from __future__ import annotations

import csv
import glob
import hashlib
import html
import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import tarfile
import time
import unicodedata
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

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
from prooftag_qr.e030_offline import sha256_file, validate_rescore_journal_rows
from prooftag_qr.e031_prospective import (
    E031_BRANCHES,
    E031_EXPERIMENT,
    E031_POLICIES,
    E031_PRIMARY_SEED,
    E031_QR_VERIFY_PRESET_COUNT,
    E031_QR_VERIFY_REPETITIONS,
    E031_RETRY_SEED,
    E031_SATURATION_THRESHOLD,
    E031_STRICT_QR_TOLERANCE_THRESHOLD,
    audit_e031_pairing,
    build_e031_holdout_prompts,
    build_e031_prospective_plan,
    candidate_pool_sha256,
    enrich_e031_stage2_results,
    evaluate_e031_policies,
    e031_candidate_rank,
    recommend_e031_advisor_chains,
    wilson_interval,
)
from prooftag_qr.parameter_advisor import E026ParameterAdvisor, load_lab_exports
from prooftag_qr.quality import image_quality_metrics
from prooftag_qr.quality_scoring import (
    DEFAULT_AESTHETIC_WEIGHTS_SHA256,
    DEFAULT_AESTHETIC_WEIGHTS_URL,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_MODEL_REVISION,
    DEFAULT_HPS_CHECKPOINT_FILENAME,
    DEFAULT_HPS_CHECKPOINT_REPO,
    DEFAULT_HPS_CHECKPOINT_REVISION,
    DEFAULT_HPS_CHECKPOINT_SHA256,
    DEFAULT_HPS_MODEL_VERSION,
    DEFAULT_HPS_PACKAGE_NAME,
    DEFAULT_HPS_PACKAGE_VERSION,
    DEFAULT_HPS_SOURCE_REVISION,
    CLIPQualityScorer,
    project_embedding,
)
from prooftag_qr.policy import ConservativeDeliveryGate, assess_stage2_candidate
from prooftag_qr.validation import ConservativeQRVerifyScorer, image_raster_sha256


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != 'nt':
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def canonical_sha256(value):
    body = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str
    ).encode('utf-8')
    return hashlib.sha256(body).hexdigest()


def notebook_semantic_sha256(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    material = [
        {
            'cell_type': cell.get('cell_type'),
            'source': ''.join(cell.get('source', [])),
        }
        for cell in raw.get('cells', [])
    ]
    return canonical_sha256(material)


def static_quality_contract(provenance):
    return {
        'policy': provenance['policy'],
        'clip': {
            'model_id': provenance['clip']['model_id'],
            'requested_revision': provenance['clip']['requested_revision'],
        },
        'clip_aesthetic': {
            'weights_url': provenance['clip_aesthetic']['weights_url'],
            'expected_sha256': provenance['clip_aesthetic']['expected_sha256'],
        },
        'hpsv2_1': {
            key: provenance['hpsv2_1'][key]
            for key in [
                'enabled', 'model_version', 'package_name',
                'expected_package_version', 'expected_source_revision',
                'checkpoint_repo', 'checkpoint_filename',
                'requested_checkpoint_revision', 'expected_checkpoint_sha256',
            ]
        },
    }


def normalized_prompt(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value)).split()).casefold()


def git_commit():
    configured = os.environ.get('PROOFTAG_GIT_COMMIT', '').strip().lower()
    if re.fullmatch(r'[0-9a-f]{40}', configured):
        return configured
    try:
        discovered = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, cwd='/app',
            stderr=subprocess.DEVNULL,
        ).strip().lower()
    except Exception:
        discovered = ''
    if not re.fullmatch(r'[0-9a-f]{40}', discovered):
        raise RuntimeError('Commit runtime inconnu : redéployer E031 avec le script versionné.')
    return discovered


print('Python :', subprocess.check_output(['python', '--version'], text=True).strip())
print('Mode notebook : CPU orchestrateur ; génération : API GPU Kubernetes')
"""
    ),
    markdown("## 1. Configuration, versions et préflight fail-closed"),
    code(
        """EXPERIMENT_NAME = E031_EXPERIMENT
COLLECTION_API_URL = 'http://prooftag-qr-svc.qr-core.svc.cluster.local:8080'
COLLECTION_PAYLOAD = os.environ.get('PROOFTAG_E031_PAYLOAD', 'https://ptag.io/t/e031')
ERROR_CORRECTION = 'M'
QR_CONTEXT = {
    'qr_version': 3,
    'qr_mask_pattern': 4,
    'qr_module_size': 20,
    'qr_padding_px': 78,
}
HOLDOUT_PROMPT_COUNT = 40
PROMPT_EMBEDDING_DIMENSIONS = 32
POLL_SECONDS = 15.0
RUN_E031 = True
MINIMUM_ROWS = 100
MINIMUM_PROMPT_GROUPS = 12
MINIMUM_CLASS_COUNT = 12

INPUT_GLOBS = [
    '/workspace/imports/prooftag-lab-*.csv',
    '/data/e026-input/prooftag-lab-*.csv',
    '/data/e026-week/*/exports/*.csv',
    '/data/e026-week/*/exports-recovered/*.csv',
    '/data/e026j-inference/*/exports/*.csv',
    '/data/e027-holdout/*/exports/*.csv',
    '/data/e028-hierarchical/*/exports/*.csv',
    '/data/e029-srmpgd-raster/*/exports/*.csv',
]
OUTPUT_ROOT = Path('/data/e031-prospective-stage2-holdout')
MODEL_ROOT = Path('/data/e031-prospective-stage2-models')
ARCHIVE_ROOT = Path('/data/e031-prospective-stage2-archives')
QR_VERIFY_CACHE = Path('/data/qr-verify-conservative-cache')
DOWNLOAD_ROOT = Path('/workspace/downloads')
NOTEBOOK_SOURCE = Path('/workspace/notebooks/26_e031_prospective_stage2_holdout.ipynb')
PROTOCOL_SOURCE = Path('/app/docs/e031-prospective-stage2-holdout.md')
for directory in [OUTPUT_ROOT, MODEL_ROOT, ARCHIVE_ROOT, QR_VERIFY_CACHE, DOWNLOAD_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)
for required_source in [NOTEBOOK_SOURCE, PROTOCOL_SOURCE]:
    if not required_source.is_file():
        raise FileNotFoundError(f'Source E031 absente de l image : {required_source}')
notebook_semantic_sha256_initial = notebook_semantic_sha256(NOTEBOOK_SOURCE)
protocol_document_sha256 = sha256_file(PROTOCOL_SOURCE)

runtime_commit = git_commit()
runtime_image = os.environ.get('PROOFTAG_RUNTIME_IMAGE', '').strip()
runtime_image_digest = os.environ.get('PROOFTAG_RUNTIME_IMAGE_DIGEST', '').strip().lower()
if not runtime_image.endswith(f':{runtime_commit[:12]}'):
    raise RuntimeError(f'Image notebook non liée au commit : {runtime_image!r}')
if not re.fullmatch(r'sha256:[0-9a-f]{64}', runtime_image_digest):
    raise RuntimeError('Digest OCI notebook absent : utiliser deploy-e031-notebook.sh.')

health = json.loads(urlopen(f'{COLLECTION_API_URL}/healthz', timeout=15).read())
ready = json.loads(urlopen(f'{COLLECTION_API_URL}/readyz', timeout=15).read())
schema = json.loads(urlopen(f'{COLLECTION_API_URL}/v1/lab/schema', timeout=30).read())
api_runtime = json.loads(urlopen(f'{COLLECTION_API_URL}/v1/runtime', timeout=30).read())
if health.get('status') != 'ok' or ready.get('status') != 'ready':
    raise RuntimeError(f'API indisponible : health={health}, ready={ready}')
if schema['validation']['engine'] != 'antfu/qr-verify@0.2.0':
    raise RuntimeError('Moteur QR-Verify API inattendu.')
if schema['validation']['tolerance_presets'] != E031_QR_VERIFY_PRESET_COUNT:
    raise RuntimeError('Nombre de presets QR-Verify inattendu.')
api_identity = api_runtime.get('deployment_identity') or {}
if api_identity.get('configured') is not True:
    raise RuntimeError('Identité de déploiement API absente.')
if api_identity.get('git_commit') != runtime_commit:
    raise RuntimeError(
        f"Commit API différent du notebook : {api_identity.get('git_commit')!r}"
    )
api_runtime_image = str(api_identity.get('image') or '')
api_runtime_digest = str(api_identity.get('image_digest') or '').lower()
if not api_runtime_image.endswith(f':{runtime_commit[:12]}'):
    raise RuntimeError(f'Image API non liée au commit : {api_runtime_image!r}')
if not re.fullmatch(r'sha256:[0-9a-f]{64}', api_runtime_digest):
    raise RuntimeError('Digest OCI API absent ou invalide.')
quality = schema['quality_scoring']
if not quality['clip_enabled'] or not quality['hpsv2_1_enabled']:
    raise RuntimeError('E031 exige CLIP-Aesthetic, CLIPScore et HPS v2.1 actifs dans l API.')
if quality.get('failure_policy') != 'fail_closed':
    raise RuntimeError('E031 exige une API qualité en mode fail-closed.')
expected_quality_contract = {
    'policy': 'fail_closed',
    'clip': {
        'model_id': DEFAULT_CLIP_MODEL,
        'requested_revision': DEFAULT_CLIP_MODEL_REVISION,
    },
    'clip_aesthetic': {
        'weights_url': DEFAULT_AESTHETIC_WEIGHTS_URL,
        'expected_sha256': DEFAULT_AESTHETIC_WEIGHTS_SHA256,
    },
    'hpsv2_1': {
        'enabled': True,
        'model_version': DEFAULT_HPS_MODEL_VERSION,
        'package_name': DEFAULT_HPS_PACKAGE_NAME,
        'expected_package_version': DEFAULT_HPS_PACKAGE_VERSION,
        'expected_source_revision': DEFAULT_HPS_SOURCE_REVISION,
        'checkpoint_repo': DEFAULT_HPS_CHECKPOINT_REPO,
        'checkpoint_filename': DEFAULT_HPS_CHECKPOINT_FILENAME,
        'requested_checkpoint_revision': DEFAULT_HPS_CHECKPOINT_REVISION,
        'expected_checkpoint_sha256': DEFAULT_HPS_CHECKPOINT_SHA256,
    },
}
api_quality_contract = static_quality_contract(quality['provenance'])
runtime_quality_contract = static_quality_contract(api_runtime['quality_scoring'])
if api_quality_contract != expected_quality_contract:
    raise RuntimeError(f'Pins qualité API inattendus : {api_quality_contract}')
if runtime_quality_contract != expected_quality_contract:
    raise RuntimeError('Contrat qualité incohérent entre /v1/runtime et /v1/lab/schema.')
if schema['notes']['upstream_revision'] != 'e24ea73ee2e13c7e6e87cb422e8b11784e70ae00':
    raise RuntimeError('Révision DiffQRCoder API inattendue.')

generation_config = api_runtime['generation_config']
expected_revisions = {
    'base_model_revision': 'f914b3679760c1c3baea6bb1815867bf1c9c92a4',
    'base_model_config_revision': '451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
    'controlnet_model_revision': '560fb7b15d0badb409f8cd578a2bfe63bd4b8046',
}
for field, expected in expected_revisions.items():
    if generation_config.get(field) != expected:
        raise RuntimeError(
            f'Révision modèle API incorrecte pour {field}: '
            f"{generation_config.get(field)!r} != {expected!r}"
        )

display(pd.DataFrame([{
    'commit notebook': runtime_commit,
    'image notebook': runtime_image,
    'digest notebook': runtime_image_digest,
    'image API': api_runtime_image,
    'digest API': api_runtime_digest,
    'API': health.get('version'),
    'QR-Verify': schema['validation']['engine'],
    'presets': schema['validation']['tolerance_presets'],
    'CLIP': quality['clip_enabled'],
    'HPS v2.1': quality['hpsv2_1_enabled'],
    'payload SHA-256': hashlib.sha256(COLLECTION_PAYLOAD.encode()).hexdigest(),
    'payload longueur': len(COLLECTION_PAYLOAD),
}]))
"""
    ),
    markdown(
        """## 2. Geler le conseiller avant de révéler le holdout

Les exports historiques forment uniquement l'ensemble d'apprentissage. Les textes E031 sont
ensuite comparés à tous les textes appris ; toute égalité exacte après normalisation NFKC arrête
l'expérience. Le fingerprint porte sur les lignes, les candidats, les features et le rapport
d'entraînement. La similarité sémantique reste une limite documentée, pas un contrôle fictif.
"""
    ),
    code(
        """csv_paths = sorted({path for pattern in INPUT_GLOBS for path in glob.glob(pattern)})
if not csv_paths:
    raise RuntimeError('Aucun export historique trouvé pour entraîner le conseiller E031.')
if any('/e031-' in path.replace('\\\\', '/') for path in csv_paths):
    raise RuntimeError('Fuite de holdout : un export E031 apparaît dans INPUT_GLOBS.')
print('Exports historiques :', len(csv_paths))

quality_scorer = None
embedding_cache = {}


def prompt_embedding(prompt):
    global quality_scorer
    key = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    if key not in embedding_cache:
        if quality_scorer is None:
            quality_scorer = CLIPQualityScorer(
                Path('/cache/huggingface'), device='cpu', hps_enabled=False
            )
        embedding_cache[key] = project_embedding(
            quality_scorer.text_embedding(prompt),
            dimensions=PROMPT_EMBEDDING_DIMENSIONS,
            seed=20260721,
        )
    return embedding_cache[key]


dataset = load_lab_exports(csv_paths, prompt_catalog={}, embedding_provider=prompt_embedding)
if not dataset.records:
    raise RuntimeError('Dataset E031 vide après chargement des exports.')
prompt_embedding(dataset.records[0].prompt_text)
advisor_quality_provenance = quality_scorer.provenance()
if advisor_quality_provenance['clip']['revision_verified'] is not True:
    raise RuntimeError('Révision réelle du CLIP de l advisor non vérifiée.')
if advisor_quality_provenance['clip_aesthetic']['sha256_verified'] is not True:
    raise RuntimeError('Poids CLIP-Aesthetic de l advisor non vérifiés.')
clip_model_revision = advisor_quality_provenance['clip']['effective_revision']
aesthetic_weights_path = quality_scorer.aesthetic_weights_path
if not aesthetic_weights_path.is_file():
    raise RuntimeError('Poids CLIP-Aesthetic de l advisor absents après chargement.')
aesthetic_weights_sha256 = sha256_file(aesthetic_weights_path)
problems = []
if dataset.audit['usable_rows'] < MINIMUM_ROWS:
    problems.append(f"{dataset.audit['usable_rows']} lignes < {MINIMUM_ROWS}")
if dataset.audit['prompt_groups'] < MINIMUM_PROMPT_GROUPS:
    problems.append(f"{dataset.audit['prompt_groups']} prompts < {MINIMUM_PROMPT_GROUPS}")
if min(dataset.audit['qr_successes'], dataset.audit['qr_failures']) < MINIMUM_CLASS_COUNT:
    problems.append('classe QR-Verify minoritaire insuffisante')
if problems:
    raise RuntimeError('Dataset E031 non identifiable : ' + '; '.join(problems))

advisor = E026ParameterAdvisor(trees=512, uncertainty_penalty=0.75, random_state=20260821)
training_report = advisor.fit(
    dataset.records,
    minimum_rows=MINIMUM_ROWS,
    minimum_groups=MINIMUM_PROMPT_GROUPS,
    minimum_class_count=MINIMUM_CLASS_COUNT,
)
training_prompt_texts = sorted({record.prompt_text for record in dataset.records})
holdout_prompts = build_e031_holdout_prompts(HOLDOUT_PROMPT_COUNT)
historical_normalized = {normalized_prompt(text) for text in training_prompt_texts}
overlap = [
    item['id'] for item in holdout_prompts
    if normalized_prompt(item['text']) in historical_normalized
]
if overlap:
    raise RuntimeError(f'Fuite exacte NFKC du holdout E031 : {overlap}')
holdout_normalized = [normalized_prompt(item['text']) for item in holdout_prompts]
if len(set(holdout_normalized)) != HOLDOUT_PROMPT_COUNT:
    raise RuntimeError('Identifiants ou textes E031 non uniques après normalisation NFKC.')
prompt_registry_rows = [
    {
        'origin': 'historical_training',
        'prompt_id': '',
        'normalized_text': value,
        'prompt_sha256': hashlib.sha256(value.encode('utf-8')).hexdigest(),
    }
    for value in sorted(historical_normalized)
] + [
    {
        'origin': 'e031_holdout',
        'prompt_id': item['id'],
        'normalized_text': normalized_prompt(item['text']),
        'prompt_sha256': hashlib.sha256(
            normalized_prompt(item['text']).encode('utf-8')
        ).hexdigest(),
    }
    for item in holdout_prompts
]
prompt_registry_sha256 = canonical_sha256(prompt_registry_rows)
dataset_material = [
    {
        'trial_id': record.trial_id,
        'group_id': record.group_id,
        'prompt_text_sha256': hashlib.sha256(record.prompt_text.encode()).hexdigest(),
        'parameters': record.parameters,
        'targets': record.targets,
    }
    for record in sorted(dataset.records, key=lambda item: item.trial_id)
]
dataset_sha256 = canonical_sha256(dataset_material)
pool_sha256 = candidate_pool_sha256(dataset.candidates)
advisor_fingerprint = {
    'class': 'E026ParameterAdvisor',
    'dataset_sha256': dataset_sha256,
    'candidate_pool_sha256': pool_sha256,
    'training_report': training_report,
    'feature_names': advisor.feature_names,
    'clip_model_id': quality_scorer.model_id,
    'clip_model_revision': clip_model_revision,
    'aesthetic_weights_sha256': aesthetic_weights_sha256,
    'quality_provenance': advisor_quality_provenance,
}
advisor_sha256 = canonical_sha256(advisor_fingerprint)
model_path = MODEL_ROOT / f'{advisor_sha256}.joblib'
temporary_model_path = model_path.with_suffix('.joblib.tmp')
advisor.save(temporary_model_path)
with temporary_model_path.open('rb') as model_stream:
    os.fsync(model_stream.fileno())
os.replace(temporary_model_path, model_path)
if os.name != 'nt':
    model_directory = os.open(model_path.parent, os.O_RDONLY)
    try:
        os.fsync(model_directory)
    finally:
        os.close(model_directory)
model_file_sha256 = sha256_file(model_path)
display(pd.DataFrame([{
    **dataset.audit,
    'dataset_sha256': dataset_sha256,
    'candidate_pool_sha256': pool_sha256,
    'advisor_semantic_sha256': advisor_sha256,
    'advisor_file_sha256': model_file_sha256,
    'prompt_registry_sha256': prompt_registry_sha256,
    'clip_model_revision': clip_model_revision,
    'aesthetic_weights_sha256': aesthetic_weights_sha256,
}]).T.rename(columns={0: 'valeur'}))
"""
    ),
    markdown("## 3. Construire et figer les 40 prompts et les trois branches"),
    code(
        """# Le registre et l'absence de chevauchement exact NFKC ont été figés avant
# toute recommandation et avant toute génération, dans la cellule précédente.
advisor_chains = recommend_e031_advisor_chains(
    advisor=advisor,
    candidates=dataset.candidates,
    prompts=holdout_prompts,
    payload=COLLECTION_PAYLOAD,
    prompt_embedding_provider=prompt_embedding,
    error_correction=ERROR_CORRECTION,
    qr_context=QR_CONTEXT,
    scan_probability_threshold=0.80,
)
base_plan = build_e031_prospective_plan(
    prompts=holdout_prompts,
    payload=COLLECTION_PAYLOAD,
    advisor_chains=advisor_chains,
    advisor_sha256=advisor_sha256,
    candidate_pool_sha256=pool_sha256,
    primary_seed=E031_PRIMARY_SEED,
    retry_seed=E031_RETRY_SEED,
    error_correction=ERROR_CORRECTION,
)
runtime_binding = {
    'base_plan_id': base_plan.plan_id,
    'runtime_commit': runtime_commit,
    'runtime_image': runtime_image,
    'runtime_image_digest': runtime_image_digest,
    'api_runtime_commit': api_identity['git_commit'],
    'api_runtime_image': api_runtime_image,
    'api_runtime_image_digest': api_runtime_digest,
    'model_revisions': expected_revisions,
    'diffqrcoder_upstream_revision': schema['notes']['upstream_revision'],
    'api_validation_contract_sha256': canonical_sha256(schema['validation']),
    # Les champs effective_* sont volontairement exclus : ils dépendent de
    # l état chaud du processus. Les pins statiques sont liés au plan et les
    # effectifs sont prouvés séparément pour chaque raster Stage 2.
    'api_quality_contract_sha256': canonical_sha256(api_quality_contract),
    'prompt_registry_sha256': prompt_registry_sha256,
    'advisor_clip_model_revision': clip_model_revision,
    'advisor_aesthetic_weights_sha256': aesthetic_weights_sha256,
    'notebook_semantic_sha256': notebook_semantic_sha256_initial,
    'protocol_document_sha256': protocol_document_sha256,
}
bound_plan_id = canonical_sha256(runtime_binding)[:16]
plan_public = {
    **base_plan.public,
    **runtime_binding,
    'plan_id': bound_plan_id,
}
plan = AdvisorInferencePlan(
    plan_id=bound_plan_id,
    payload=base_plan.payload,
    campaigns=base_plan.campaigns,
    predictions=base_plan.predictions,
    public=plan_public,
)
if plan.public['trial_count'] != HOLDOUT_PROMPT_COUNT * 6:
    raise RuntimeError('Plan E031 incomplet.')
if plan.public['srmpgd_trial_count'] != 0:
    raise RuntimeError('SR-MPGD est interdit dans E031.')
for campaign in plan.campaigns:
    if [method['output_variant'] for method in campaign['methods']] != ['raw', 'srpg']:
        raise RuntimeError('Ordre Stage 1 -> Stage 2 invalide.')
    if not campaign['methods'][1]['require_exact_stage1_reuse']:
        raise RuntimeError('Le Stage 2 ne force pas la réutilisation exacte du Stage 1.')

display(pd.DataFrame([{
    'plan': plan.plan_id,
    'prompts inédits': plan.public['prompt_count'],
    'simples': sum('_simple_' in item['id'] for item in holdout_prompts),
    'atypiques': sum('_atypical_' in item['id'] for item in holdout_prompts),
    'branches Stage 2': len(E031_BRANCHES),
    'Stage 1 internes': plan.public['stage1_trial_count'],
    'Stage 2 évalués': plan.public['stage2_trial_count'],
    'essais API': plan.public['trial_count'],
    'groupes advisor': plan.public['advisor_recipe_groups'],
    'SR-MPGD': plan.public['srmpgd_trial_count'],
}]))
display(pd.DataFrame(plan.predictions).groupby(
    ['branch_id', 'pipeline_state']
).size().rename('essais').reset_index())
"""
    ),
    markdown(
        """## 4. Génération API persistante et reprenable

Le dossier est le hash du plan. Un export `completed_with_errors` est refusé. Une campagne
distante ne peut être reprise que si son nom lié au plan, son payload haché et sa spécification
complète correspondent. Un export absent ou corrompu est retéléchargé après une coupure.
"""
    ),
    code(
        """events = deque(maxlen=20)
started = time.monotonic()


def progress(event):
    events.append(event)
    clear_output(wait=True)
    latest = events[-1]
    display(Markdown('### Progression E031'))
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
    display(pd.DataFrame(list(events)[-8:]))


runner = AdvisorInferenceRunner(
    plan=plan,
    api_url=COLLECTION_API_URL,
    output_root=OUTPUT_ROOT,
    poll_seconds=POLL_SECONDS,
    maximum_campaign_attempts=2,
    reject_campaigns_with_errors=True,
    progress_callback=progress,
)
print('Plan persistant :', runner.output_dir)
print('État de reprise :', runner.state_path)
runner_summary = runner.run() if RUN_E031 else runner.summary()
if runner_summary['status'] != 'completed':
    raise RuntimeError(f"Campagne E031 incomplète : {runner_summary}")
display(pd.DataFrame([runner_summary]).T.rename(columns={0: 'valeur'}))
"""
    ),
    markdown("## 5. Charger les résultats, prouver l'appariement et télécharger les 120 Stage 2"),
    code(
        """RUN_DIR = runner.output_dir / 'analysis'
RUN_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(prompt_registry_rows).to_csv(
    RUN_DIR / 'e031-prompt-registry.csv', index=False
)
atomic_json(RUN_DIR / 'e031-protocol.json', plan.public)
rows = load_advisor_inference_results(runner.output_dir)
state_frame = pd.DataFrame(rows)
expected_states = HOLDOUT_PROMPT_COUNT * 6
if len(state_frame) != expected_states:
    raise RuntimeError(f'Matrice E031 incomplète : {len(state_frame)}/{expected_states}.')
state_frame.to_csv(RUN_DIR / 'e031-state-results.csv', index=False)

pairing = pd.DataFrame(audit_e031_pairing(rows))
pairing.to_csv(RUN_DIR / 'e031-pairing-audit.csv', index=False)
if len(pairing) != HOLDOUT_PROMPT_COUNT * 3 or not pairing.complete.all():
    display(pairing[~pairing.complete])
    raise RuntimeError('Au moins un Stage 2 ne prouve pas son Stage 1 exact.')

stage2_rows = [row for row in rows if row['pipeline_state'] == 'stage2']
raw_export_rows = {}
for source_path in sorted({Path(row['_source_file']) for row in stage2_rows}):
    with source_path.open('r', encoding='utf-8', newline='') as stream:
        for raw in csv.DictReader(stream):
            key = (str(source_path), str(raw.get('trial_id') or ''))
            if key in raw_export_rows:
                raise RuntimeError(f'Ligne export dupliquée : {key}')
            raw_export_rows[key] = raw
expected_run_quality_provenance = {
    'provenance_quality_clip_model_id': DEFAULT_CLIP_MODEL,
    'provenance_quality_clip_model_revision': DEFAULT_CLIP_MODEL_REVISION,
    'provenance_quality_aesthetic_weights_sha256': (
        DEFAULT_AESTHETIC_WEIGHTS_SHA256
    ),
    'provenance_quality_hps_package_version': DEFAULT_HPS_PACKAGE_VERSION,
    'provenance_quality_hps_source_revision': DEFAULT_HPS_SOURCE_REVISION,
    'provenance_quality_hps_checkpoint_revision': DEFAULT_HPS_CHECKPOINT_REVISION,
    'provenance_quality_hps_checkpoint_sha256': DEFAULT_HPS_CHECKPOINT_SHA256,
}
quality_provenance_audit_rows = []
for row in stage2_rows:
    key = (str(Path(row['_source_file'])), str(row['trial_id']))
    raw = raw_export_rows.get(key)
    if raw is None:
        raise RuntimeError(f'Export brut absent pour le Stage 2 {row["trial_id"]}.')
    mismatches = {
        field: {'expected': expected, 'actual': raw.get(field)}
        for field, expected in expected_run_quality_provenance.items()
        if raw.get(field) != expected
    }
    if mismatches:
        raise RuntimeError(
            f'Provenance qualité invalide pour {row["trial_id"]}: {mismatches}'
        )
    quality_provenance_audit_rows.append({
        'trial_id': row['trial_id'],
        'generation_run_id': row['generation_run_id'],
        'prompt_id': row['prompt_id'],
        'branch_id': row['branch_id'],
        'verified': True,
        **{field: raw[field] for field in expected_run_quality_provenance},
    })
quality_provenance_audit = pd.DataFrame(quality_provenance_audit_rows)
if len(quality_provenance_audit) != HOLDOUT_PROMPT_COUNT * 3:
    raise RuntimeError('Audit de provenance qualité Stage 2 incomplet.')
quality_provenance_audit.to_csv(
    RUN_DIR / 'e031-quality-provenance-audit.csv', index=False
)
for metric in ['clip_aesthetic', 'clip_score', 'hpsv2_1']:
    invalid = [
        row['trial_id'] for row in stage2_rows
        if _finite(row.get(metric)) is None
    ]
    if invalid:
        raise RuntimeError(
            f'Métrique {metric} absente ou non finie sur {len(invalid)} Stage 2.'
        )

gallery_dir = RUN_DIR / 'e031-gallery'
gallery_entries = download_advisor_gallery(
    stage2_rows,
    api_url=COLLECTION_API_URL,
    output_dir=gallery_dir / 'images',
    timeout=60,
)
if len(gallery_entries) != HOLDOUT_PROMPT_COUNT * 3:
    raise RuntimeError('Téléchargement Stage 2 incomplet.')
for entry in gallery_entries:
    if entry.get('download_error') or not entry.get('local_image'):
        raise RuntimeError(
            f"Image indisponible : {entry.get('trial_id')}: "
            f"{entry.get('download_error')}"
        )
    with Image.open(entry['local_image']) as source:
        rgb = source.convert('RGB')
        raster_hash = image_raster_sha256(rgb)
        local_quality = image_quality_metrics(rgb)
    expected = str(entry.get('final_image_sha256') or '').lower()
    if raster_hash != expected:
        raise RuntimeError(f"Hash raster API incorrect pour {entry.get('trial_id')}.")
    entry['downloaded_raster_sha256'] = raster_hash
    entry['downloaded_png_sha256'] = sha256_file(Path(entry['local_image']))
    entry['local_high_saturation_pixel_ratio'] = local_quality[
        'high_saturation_pixel_ratio'
    ]
    entry['local_rgb_clipped_channel_ratio'] = local_quality[
        'rgb_clipped_channel_ratio'
    ]
    entry['local_saturation_risk'] = max(
        entry['local_high_saturation_pixel_ratio'],
        entry['local_rgb_clipped_channel_ratio'],
    )
    if not math.isfinite(entry['local_saturation_risk']):
        raise RuntimeError(f"Saturation locale invalide : {entry.get('trial_id')}")
write_gallery_index(gallery_entries, gallery_dir)
print('Stage 2 téléchargés et prouvés :', len(gallery_entries))
print('Rasters uniques :', len({item['downloaded_raster_sha256'] for item in gallery_entries}))
"""
    ),
    markdown("## 6. QR-Verify 5 × 37 par raster, cache par contenu et journal fsync"),
    code(
        """scorer_probe = ConservativeQRVerifyScorer(
    repetitions=E031_QR_VERIFY_REPETITIONS,
    cache_dir=QR_VERIFY_CACHE,
)
scorer_identity = {
    'engine_version': scorer_probe.engine_version,
    'scoring_version': scorer_probe.scoring_version,
    'implementation_sha256': scorer_probe.implementation_sha256,
    'repetitions': E031_QR_VERIFY_REPETITIONS,
    'preset_count': scorer_probe.decoder.preset_count,
}
scorer_probe.close()
if scorer_identity['preset_count'] != E031_QR_VERIFY_PRESET_COUNT:
    raise RuntimeError('Le scorer local ne possède pas 37 presets.')

payload_sha256 = hashlib.sha256(COLLECTION_PAYLOAD.encode('utf-8')).hexdigest()
rescore_run_id = canonical_sha256({
    'plan_id': plan.plan_id,
    'payload_sha256': payload_sha256,
    **scorer_identity,
})[:16]
JOURNAL_PATH = RUN_DIR / 'e031-rescore-results.jsonl'
PROGRESS_PATH = RUN_DIR / 'e031-rescore-progress.json'
unique_by_raster = {}
for entry in gallery_entries:
    unique_by_raster.setdefault(entry['downloaded_raster_sha256'], entry)
work_entries = sorted(unique_by_raster.values(), key=lambda item: item['downloaded_raster_sha256'])
expected_raster_by_png = {
    entry['downloaded_png_sha256']: entry['downloaded_raster_sha256']
    for entry in work_entries
}


def raw_journal_rows():
    if not JOURNAL_PATH.exists():
        return []
    parsed = []
    for number, line in enumerate(JOURNAL_PATH.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            print(f'Ligne JSONL partielle ignorée : {number}')
    return parsed


def journal_rows():
    return validate_rescore_journal_rows(
        raw_journal_rows(),
        run_id=rescore_run_id,
        expected_raster_sha256_by_source=expected_raster_by_png,
        payload_sha256=payload_sha256,
        scorer_identity=scorer_identity,
    )


completed = {row['score']['image_sha256']: row for row in journal_rows()}
scorer = ConservativeQRVerifyScorer(
    repetitions=E031_QR_VERIFY_REPETITIONS,
    cache_dir=QR_VERIFY_CACHE,
)
rescore_started = time.monotonic()
try:
    for position, entry in enumerate(work_entries, 1):
        raster_hash = entry['downloaded_raster_sha256']
        if raster_hash in completed:
            continue
        with Image.open(entry['local_image']) as source:
            score = scorer.score(source.convert('RGB'), COLLECTION_PAYLOAD).to_dict()
        record = {
            'run_id': rescore_run_id,
            'source_png_sha256': entry['downloaded_png_sha256'],
            'generation_run_id': entry['generation_run_id'],
            'score': score,
        }
        with JOURNAL_PATH.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\\n')
            stream.flush()
            os.fsync(stream.fileno())
        completed[raster_hash] = record
        progress_row = {
            'status': 'running',
            'completed': len(completed),
            'total': len(work_entries),
            'current': entry['prompt_id'],
            'cache_hit': score['cache_hit'],
            'elapsed_minutes': round((time.monotonic() - rescore_started) / 60, 2),
        }
        atomic_json(PROGRESS_PATH, progress_row)
        clear_output(wait=True)
        display(Markdown('### Progression QR-Verify répétable'))
        display(pd.DataFrame([progress_row]))
finally:
    scorer.close()
if len(completed) != len(work_entries):
    raise RuntimeError(f'Rescoring incomplet : {len(completed)}/{len(work_entries)}')
atomic_json(PROGRESS_PATH, {
    'status': 'completed', 'completed': len(completed), 'total': len(work_entries),
})
print('Rescoring terminé :', len(completed), 'rasters uniques × 5 × 37 presets')
"""
    ),
    markdown("## 7. Enrichissement, portes QR et politiques prospectives"),
    code(
        """validated_journal = journal_rows()
scores_by_raster = {row['score']['image_sha256']: row['score'] for row in validated_journal}
path_by_run = {entry['generation_run_id']: entry for entry in gallery_entries}
rows_with_downloads = []
for row in rows:
    enriched_row = dict(row)
    if row['pipeline_state'] == 'stage2':
        artifact = path_by_run[row['generation_run_id']]
        enriched_row.update({
            'local_image': artifact['local_image'],
            'downloaded_raster_sha256': artifact['downloaded_raster_sha256'],
            'downloaded_png_sha256': artifact['downloaded_png_sha256'],
            # Recalcul local sur le raster téléchargé : une métrique API
            # manquante ne peut jamais devenir implicitement zéro.
            'saturation_risk': artifact['local_saturation_risk'],
            'local_high_saturation_pixel_ratio': artifact[
                'local_high_saturation_pixel_ratio'
            ],
            'local_rgb_clipped_channel_ratio': artifact[
                'local_rgb_clipped_channel_ratio'
            ],
        })
    rows_with_downloads.append(enriched_row)

enriched_rows = enrich_e031_stage2_results(rows_with_downloads, scores_by_raster)
enriched_frame = pd.DataFrame([
    {
        **row,
        'qr_verify_observations_json': json.dumps(row['qr_verify_observations']),
        'qr_verify_observations': None,
    }
    for row in enriched_rows
])
enriched_frame.to_csv(RUN_DIR / 'e031-stage2-rescore.csv', index=False)

policy_report = evaluate_e031_policies(enriched_rows)
decisions = pd.DataFrame(policy_report['decisions'])
policy_summary = pd.DataFrame(policy_report['summary'])
decisions.to_csv(RUN_DIR / 'e031-policy-decisions.csv', index=False)
policy_summary.to_csv(RUN_DIR / 'e031-policy-summary.csv', index=False)
atomic_json(RUN_DIR / 'e031-policy-report.json', policy_report)
if decisions.stage1_was_delivered.any() or decisions.srmpgd_was_requested.any():
    raise RuntimeError('Violation du protocole : Stage 1 livré ou SR-MPGD demandé.')

software_family_rows = []
for (gate_name, policy, family), subset in decisions.groupby(
    ['gate', 'policy', 'prompt_family'], sort=True
):
    delivered = int(subset.deliverable.sum())
    low, high = wilson_interval(delivered, len(subset))
    software_family_rows.append({
        'gate': gate_name,
        'policy': policy,
        'prompt_family': family,
        'prompts': len(subset),
        'delivered': delivered,
        'delivery_rate': delivered / len(subset),
        'wilson_95_low': low,
        'wilson_95_high': high,
    })
software_family_summary = pd.DataFrame(software_family_rows)
software_family_summary.to_csv(
    RUN_DIR / 'e031-policy-summary-by-family.csv', index=False
)

display(Markdown('### Résultats logiciels — ils ne sont pas des résultats téléphone'))
display(policy_summary[[
    'gate', 'policy', 'delivered', 'prompts', 'delivery_rate',
    'wilson_95_low', 'wilson_95_high', 'mean_stage2_attempts',
    'mean_qr_tolerance_delivered', 'mean_hpsv2_1_delivered',
    'mean_clip_aesthetic_delivered', 'mean_clip_score_delivered',
    'selected_branch_counts',
]])
display(software_family_summary)

score_rows = []
for score in scores_by_raster.values():
    score_rows.append({
        key: value for key, value in score.items()
        if key not in {'runs', 'stable_preset_ids', 'unstable_preset_ids'}
    })
score_frame = pd.DataFrame(score_rows)
score_frame.to_csv(RUN_DIR / 'e031-unique-raster-rescore.csv', index=False)
print(
    'Rasters instables :', int((score_frame.unstable_preset_count > 0).sum()),
    '/', len(score_frame),
)
"""
    ),
    markdown("## 8. Graphes appariés et garde contre les conclusions prématurées"),
    code(
        """fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
branch_order = list(E031_BRANCHES)
branch_labels = ['Fixe seed A', 'Advisor seed A', 'Fixe seed B']
branch_colors = ['#2563eb', '#f97316', '#7c3aed']
branch_summary = enriched_frame.groupby('branch_id').agg(
    qr=('conservative_qr_tolerance', 'mean'),
    aes=('clip_aesthetic', 'mean'),
    hps=('hpsv2_1', 'mean'),
).reindex(branch_order)
axes[0].bar(branch_labels, branch_summary.qr, color=branch_colors)
axes[0].axhline(36 / 37, color='black', linestyle='--', label='36/37 strict')
axes[0].set(title='QR-Verify conservateur', ylim=(0, 1.03))
axes[0].legend()
axes[1].bar(branch_labels, branch_summary.aes, color=branch_colors)
axes[1].set(title='CLIP-Aesthetic — proxy')
axes[2].bar(branch_labels, branch_summary.hps, color=branch_colors)
axes[2].set(title='HPS v2.1 — proxy')
for axis in axes:
    axis.tick_params(axis='x', rotation=18)
    axis.grid(axis='y', alpha=0.25)
fig.tight_layout()
fig.savefig(RUN_DIR / 'e031-branch-scorecard.png', dpi=170)
display(fig)

strict_summary = policy_summary[policy_summary.gate == 'strict'].copy()
fig, axis = plt.subplots(figsize=(14, 6))
rates = strict_summary.delivery_rate.to_numpy(dtype=float)
lows = strict_summary.wilson_95_low.to_numpy(dtype=float)
highs = strict_summary.wilson_95_high.to_numpy(dtype=float)
error_low = np.maximum(0.0, rates - lows)
error_high = np.maximum(0.0, highs - rates)
labels = strict_summary.policy.str.replace('_', ' ')
axis.bar(labels, rates, color='#16a34a')
axis.errorbar(
    range(len(strict_summary)), rates,
    yerr=np.vstack([error_low, error_high]),
    fmt='none', ecolor='black', capsize=5,
)
axis.set_ylim(0, 1.05)
axis.set_title('Porte stricte 36/37 — IC Wilson 95 %')
axis.tick_params(axis='x', rotation=25)
axis.grid(axis='y', alpha=0.25)
fig.tight_layout()
fig.savefig(RUN_DIR / 'e031-strict-policy-delivery.png', dpi=170)
display(fig)

print('best_of_three est un oracle rétrospectif, jamais une politique de production.')
print('Aucune conclusion esthétique ne sera tirée avant la revue humaine aveugle.')
"""
    ),
    markdown(
        """## 9. Galerie humaine aveugle

Les cartes masquent méthode, seed et scores. Notez chaque image dans
`e031-human-review.csv`, puis relancez cette cellule et la suivante. Les quatre contrôles
dupliqués mesurent la cohérence du jugement. Les champs téléphone restent optionnels et ne sont
jamais remplacés par QR-Verify.
"""
    ),
    code(
        """blind_dir = RUN_DIR / 'e031-blind-review'
blind_dir.mkdir(exist_ok=True)
blind_image_dir = blind_dir / 'images'
blind_image_dir.mkdir(exist_ok=True)
rng = random.Random(int(plan.plan_id, 16))
source_items = [dict(item) for item in gallery_entries]
rng.shuffle(source_items)
duplicate_sources = rng.sample(source_items, 4)
review_items = [(item, None) for item in source_items] + [
    (item, item['generation_run_id']) for item in duplicate_sources
]
rng.shuffle(review_items)

review_rows = []
reveal_rows = []
for index, (item, duplicate_of) in enumerate(review_items, 1):
    blind_id = f'E031-B{index:03d}'
    blind_image = blind_image_dir / f'{blind_id}.png'
    with Image.open(item['local_image']) as source:
        source.convert('RGB').save(blind_image, format='PNG', optimize=True)
    review_rows.append({
        'blind_id': blind_id,
        'prompt_id': item['prompt_id'],
        'prompt_text': item['prompt_text'],
        'image_path': f'images/{blind_id}.png',
        'aesthetic_1_5': '',
        'prompt_fidelity_1_5': '',
        'qr_discretion_1_5': '',
        'subject_present_yes_no': '',
        'grid_too_visible_yes_no': '',
        'fatal_artifact_yes_no': '',
        'overall_1_5': '',
        'phone_scan_optional_yes_no': '',
        'notes': '',
    })
    reveal_rows.append({
        'blind_id': blind_id,
        'generation_run_id': item['generation_run_id'],
        'branch_id': item['branch_id'],
        'seed': int(item['seed']),
        'duplicate_of_generation_run_id': duplicate_of or '',
    })

review_path = blind_dir / 'e031-human-review.csv'
reveal_path = RUN_DIR / 'e031-human-review-reveal.csv'
if not review_path.exists():
    pd.DataFrame(review_rows).to_csv(review_path, index=False)
else:
    existing_review = pd.read_csv(review_path, keep_default_na=False)
    if list(existing_review.blind_id) != [row['blind_id'] for row in review_rows]:
        raise RuntimeError('La feuille humaine existante appartient à un autre ordre aveugle.')
pd.DataFrame(reveal_rows).to_csv(reveal_path, index=False)

html_cards = []
for row in review_rows:
    image_path = html.escape(row['image_path'])
    html_cards.append(
        f"<article><img src='{image_path}'><h3>{row['blind_id']}</h3>"
        f"<p><b>{html.escape(row['prompt_id'])}</b><br>{html.escape(row['prompt_text'])}</p>"
        '<p>Reporter les notes dans e031-human-review.csv</p></article>'
    )
gallery_html = '''<!doctype html><meta charset="utf-8"><title>E031 revue aveugle</title>
<style>body{font-family:sans-serif;background:#0b1119;color:#eef2ff}main{display:grid;
grid-template-columns:repeat(3,minmax(260px,1fr));gap:16px}article{background:#111827;
padding:12px;border-radius:12px}img{width:100%;aspect-ratio:1;object-fit:contain;background:white}
p{line-height:1.35}</style><h1>E031 — revue humaine aveugle</h1><main>''' + \
    ''.join(html_cards) + '</main>'
atomic_text(blind_dir / 'e031-human-review.html', gallery_html)

ratings = pd.read_csv(review_path, keep_default_na=False)
rating_columns = ['aesthetic_1_5', 'prompt_fidelity_1_5', 'qr_discretion_1_5', 'overall_1_5']
human_review_complete = all(
    pd.to_numeric(ratings[column], errors='coerce').between(1, 5).all()
    for column in rating_columns
) and all(
    ratings[column].str.strip().str.casefold().isin({'yes', 'no', 'oui', 'non'}).all()
    for column in ['subject_present_yes_no', 'grid_too_visible_yes_no', 'fatal_artifact_yes_no']
)
if human_review_complete:
    for column in rating_columns:
        ratings[column] = pd.to_numeric(ratings[column], errors='raise')
    reveal = pd.read_csv(reveal_path, keep_default_na=False)
    human_results = ratings.merge(reveal, on='blind_id', validate='one_to_one')
    yes_values = {'yes', 'oui'}
    human_results['human_approved'] = (
        (human_results.aesthetic_1_5 >= 3)
        & (human_results.prompt_fidelity_1_5 >= 3)
        & (human_results.qr_discretion_1_5 >= 3)
        & human_results.subject_present_yes_no.str.strip().str.casefold().isin(yes_values)
        & ~human_results.grid_too_visible_yes_no.str.strip().str.casefold().isin(yes_values)
        & ~human_results.fatal_artifact_yes_no.str.strip().str.casefold().isin(yes_values)
    )
    human_results.to_csv(RUN_DIR / 'e031-human-review-complete.csv', index=False)
    human_originals = human_results[
        human_results.duplicate_of_generation_run_id == ''
    ].copy()
    if len(human_originals) != HOLDOUT_PROMPT_COUNT * 3:
        raise RuntimeError('La revue humaine ne couvre pas les 120 sorties originales.')
    if human_originals.generation_run_id.duplicated().any():
        raise RuntimeError('Run dupliqué parmi les 120 sorties humaines originales.')
    human_summary = human_originals.groupby('branch_id').agg(
        images=('blind_id', 'count'),
        aesthetic=('aesthetic_1_5', 'mean'),
        prompt_fidelity=('prompt_fidelity_1_5', 'mean'),
        qr_discretion=('qr_discretion_1_5', 'mean'),
        overall=('overall_1_5', 'mean'),
        human_approved=('human_approved', 'sum'),
    ).reset_index()
    human_summary.to_csv(RUN_DIR / 'e031-human-review-summary.csv', index=False)

    original_by_run = human_originals.set_index('generation_run_id')
    duplicate_agreement = []
    for _, duplicate in human_results[
        human_results.duplicate_of_generation_run_id != ''
    ].iterrows():
        target = original_by_run.loc[duplicate.duplicate_of_generation_run_id]
        duplicate_agreement.append({
            'blind_id': duplicate.blind_id,
            'original_blind_id': target.blind_id,
            **{
                f'absolute_difference_{column}': abs(
                    float(duplicate[column]) - float(target[column])
                )
                for column in rating_columns
            },
            **{
                f'agreement_{column}': (
                    str(duplicate[column]).strip().casefold()
                    == str(target[column]).strip().casefold()
                )
                for column in [
                    'subject_present_yes_no', 'grid_too_visible_yes_no',
                    'fatal_artifact_yes_no',
                ]
            },
        })
    agreement_frame = pd.DataFrame(duplicate_agreement)
    if len(agreement_frame) != 4:
        raise RuntimeError('Les quatre contrôles humains dupliqués sont incomplets.')
    agreement_frame.to_csv(RUN_DIR / 'e031-human-duplicate-agreement.csv', index=False)

    strict_gate = ConservativeDeliveryGate(
        qr_tolerance_threshold=E031_STRICT_QR_TOLERANCE_THRESHOLD,
        saturation_threshold=E031_SATURATION_THRESHOLD,
        minimum_qr_observations=E031_QR_VERIFY_REPETITIONS,
    )
    human_by_run = human_originals.set_index('generation_run_id').to_dict('index')
    candidate_by_prompt_branch = {}
    final_status_rows = []
    for candidate in enriched_rows:
        run_id = candidate['generation_run_id']
        human = human_by_run.get(run_id)
        if human is None:
            raise RuntimeError(f'Note humaine absente pour le run {run_id}.')
        assessment = assess_stage2_candidate(candidate, strict_gate)
        status = {
            'prompt_id': candidate['prompt_id'],
            'branch_id': candidate['branch_id'],
            'generation_run_id': run_id,
            'software_deliverable': assessment.deliverable,
            'human_approved': bool(human['human_approved']),
            'final_deliverable': (
                assessment.deliverable and bool(human['human_approved'])
            ),
            'software_rejection_reasons': ';'.join(assessment.rejection_reasons),
            'human_overall_1_5': float(human['overall_1_5']),
            'human_prompt_fidelity_1_5': float(human['prompt_fidelity_1_5']),
        }
        final_status_rows.append(status)
        candidate_by_prompt_branch[(candidate['prompt_id'], candidate['branch_id'])] = (
            candidate, status
        )
    final_status_frame = pd.DataFrame(final_status_rows)
    final_status_frame.to_csv(RUN_DIR / 'e031-final-candidate-status.csv', index=False)

    final_decisions = []
    for policy, branches in E031_POLICIES.items():
        for prompt_id in sorted({item['prompt_id'] for item in enriched_rows}):
            attempted = [candidate_by_prompt_branch[(prompt_id, branch)] for branch in branches]
            passing = [item for item in attempted if item[1]['final_deliverable']]
            if policy == 'best_of_three' and passing:
                selected_candidate, selected_status = max(
                    passing,
                    key=lambda item: (
                        e031_candidate_rank(item[0]),
                        item[1]['human_overall_1_5'],
                        item[1]['human_prompt_fidelity_1_5'],
                    ),
                )
                attempts_used = len(attempted)
                human_reviews_used = sum(
                    bool(status['software_deliverable']) for _, status in attempted
                )
            else:
                selected_candidate = selected_status = None
                attempts_used = 0
                human_reviews_used = 0
                for candidate, status in attempted:
                    attempts_used += 1
                    if status['software_deliverable']:
                        human_reviews_used += 1
                    if status['final_deliverable']:
                        selected_candidate, selected_status = candidate, status
                        break
            final_decisions.append({
                'prompt_id': prompt_id,
                'prompt_family': 'simple' if '_simple_' in prompt_id else 'atypical',
                'policy': policy,
                'final_deliverable': selected_candidate is not None,
                'selected_branch': (
                    selected_candidate['branch_id'] if selected_candidate else None
                ),
                'selected_generation_run_id': (
                    selected_candidate['generation_run_id'] if selected_candidate else None
                ),
                'stage2_attempts_used': attempts_used,
                'human_reviews_used': human_reviews_used,
            })
    final_decisions_frame = pd.DataFrame(final_decisions)
    final_decisions_frame.to_csv(RUN_DIR / 'e031-final-policy-decisions.csv', index=False)
    final_summary_rows = []
    for policy in E031_POLICIES:
        subset = final_decisions_frame[final_decisions_frame.policy == policy]
        delivered = int(subset.final_deliverable.sum())
        low, high = wilson_interval(delivered, len(subset))
        final_summary_rows.append({
            'policy': policy,
            'prompts': len(subset),
            'final_delivered': delivered,
            'final_delivery_rate': delivered / len(subset),
            'wilson_95_low': low,
            'wilson_95_high': high,
            'mean_stage2_attempts': float(subset.stage2_attempts_used.mean()),
            'mean_human_reviews': float(subset.human_reviews_used.mean()),
        })
    final_policy_summary = pd.DataFrame(final_summary_rows)
    final_policy_summary.to_csv(RUN_DIR / 'e031-final-policy-summary.csv', index=False)
    final_family_rows = []
    for (policy, family), subset in final_decisions_frame.groupby(
        ['policy', 'prompt_family'], sort=True
    ):
        delivered = int(subset.final_deliverable.sum())
        low, high = wilson_interval(delivered, len(subset))
        final_family_rows.append({
            'policy': policy,
            'prompt_family': family,
            'prompts': len(subset),
            'final_delivered': delivered,
            'final_delivery_rate': delivered / len(subset),
            'wilson_95_low': low,
            'wilson_95_high': high,
        })
    final_family_summary = pd.DataFrame(final_family_rows)
    final_family_summary.to_csv(
        RUN_DIR / 'e031-final-policy-summary-by-family.csv', index=False
    )
    display(human_summary)
    display(agreement_frame)
    display(final_policy_summary)
    display(final_family_summary)
else:
    human_summary = pd.DataFrame()
    agreement_frame = pd.DataFrame()
    final_status_frame = pd.DataFrame()
    final_decisions_frame = pd.DataFrame()
    final_policy_summary = pd.DataFrame()
    final_family_summary = pd.DataFrame()
    display(Markdown(
        f'**Revue humaine en attente.** Ouvrir `{blind_dir / "e031-human-review.html"}`, '
        f'remplir `{review_path}`, puis relancer les cellules 9 et 10.'
    ))
print('Cartes aveugles :', len(review_rows), 'dont 4 contrôles dupliqués.')
"""
    ),
    markdown("## 10. Rapport, manifeste, checksums et archive"),
    code(
        """standard = policy_summary[policy_summary.gate == 'standard'].to_dict('records')
strict = policy_summary[policy_summary.gate == 'strict'].to_dict('records')
report = {
    'experiment': EXPERIMENT_NAME,
    'created_at': datetime.now(UTC).isoformat(),
    'plan_id': plan.plan_id,
    'prompt_count': HOLDOUT_PROMPT_COUNT,
    'stage2_rasters': len(gallery_entries),
    'unique_stage2_rasters': len(work_entries),
    'qr_verify_repetitions': E031_QR_VERIFY_REPETITIONS,
    'qr_verify_presets': E031_QR_VERIFY_PRESET_COUNT,
    'standard_gate': standard,
    'strict_36_of_37_gate': strict,
    'software_gate_by_family': software_family_summary.to_dict('records'),
    'human_review_complete': human_review_complete,
    'human_branch_summary': human_summary.to_dict('records'),
    'human_duplicate_agreement': agreement_frame.to_dict('records'),
    'final_gate_summary': final_policy_summary.to_dict('records'),
    'final_gate_by_family': final_family_summary.to_dict('records'),
    'stage1_delivery_allowed': False,
    'srmpgd_requested': False,
    'physical_phone_claim': False,
    'advisor_trained_before_holdout': True,
    'all_three_branches_generated_for_every_prompt': True,
}
atomic_json(RUN_DIR / 'e031-report.json', report)

strict_production = next(
    row for row in strict if row['policy'] == 'fixed_advisor_then_seed_retry'
)
if human_review_complete:
    final_production = next(
        row for row in final_policy_summary.to_dict('records')
        if row['policy'] == 'fixed_advisor_then_seed_retry'
    )
    final_line = (
        f"- Cascade finale QR stricte + humain : "
        f"**{int(final_production['final_delivered'])}/"
        f"{int(final_production['prompts'])}** livrables, "
        f"IC Wilson 95 % [{final_production['wilson_95_low']:.3f}, "
        f"{final_production['wilson_95_high']:.3f}]."
    )
else:
    final_line = '- Cascade finale QR stricte + humain : **non calculée, revue en attente**.'
report_markdown = f'''# Rapport E031 — holdout prospectif Stage 2

- Plan immuable : `{plan.plan_id}` ; conseiller : `{advisor_sha256}`.
- Holdout : **{HOLDOUT_PROMPT_COUNT} prompts nouveaux**, 20 simples et 20 atypiques.
- Génération appariée : **{plan.public['trial_count']} essais API**, dont
  **{len(gallery_entries)} Stage 2** évalués ; Stage 1 livré : **jamais** ; SR-MPGD : **jamais**.
- QR-Verify : **{E031_QR_VERIFY_REPETITIONS} × {E031_QR_VERIFY_PRESET_COUNT} presets**,
  agrégation par intersection, {int((score_frame.unstable_preset_count > 0).sum())}
  rasters instables sur {len(score_frame)} rasters uniques.
- Cascade préenregistrée, porte stricte 36/37 :
  **{int(strict_production['delivered'])}/{int(strict_production['prompts'])}** livrables,
  IC Wilson 95 % [{strict_production['wilson_95_low']:.3f},
  {strict_production['wilson_95_high']:.3f}],
  {strict_production['mean_stage2_attempts']:.2f} Stage 2 nécessaires en moyenne.
- Revue humaine aveugle : **{'terminée' if human_review_complete else 'en attente'}**.
{final_line}

## Interprétation autorisée

E031 mesure la généralisation logicielle des trois branches et le coût de leur cascade sur un
holdout gelé. HPS, CLIP-Aesthetic et CLIPScore restent des proxys. La qualité esthétique finale
ne peut être affirmée que lorsque la feuille aveugle est complète.

## Interprétation interdite

Ces résultats ne sont pas une probabilité de scan téléphone ou impression et ne démontrent pas
99 % de scannabilité physique. `best_of_three` est un oracle rétrospectif, pas une politique de
production. Aucun paramètre ne doit être réentraîné sur E031 avant de publier ce rapport.
'''
atomic_text(RUN_DIR / 'e031-report.md', report_markdown)
display(Markdown(report_markdown))

notebook_source = NOTEBOOK_SOURCE
protocol_source = PROTOCOL_SOURCE
notebook_semantic_sha256_final = notebook_semantic_sha256(notebook_source)
if notebook_semantic_sha256_final != notebook_semantic_sha256_initial:
    raise RuntimeError('Les sources des cellules E031 ont changé depuis le gel du plan.')
if sha256_file(protocol_source) != protocol_document_sha256:
    raise RuntimeError('Le protocole E031 a changé depuis le gel du plan.')
validation_source = Path(inspect.getfile(ConservativeQRVerifyScorer)).resolve()
bridge_source = Path(os.environ['PROOFTAG_QR_QR_VERIFY_BRIDGE']).resolve()
package_lock = bridge_source.with_name('package-lock.json')
for required in [
    notebook_source, protocol_source, validation_source, bridge_source,
    package_lock, model_path, aesthetic_weights_path,
]:
    if not required.is_file():
        raise FileNotFoundError(f'Preuve de provenance absente : {required}')

control_dir = RUN_DIR / 'e031-control-plane'
export_copy_dir = control_dir / 'exports'
export_copy_dir.mkdir(parents=True, exist_ok=True)
for source in [runner.plan_path, runner.predictions_path, runner.state_path]:
    shutil.copy2(source, control_dir / source.name)
for source in sorted(runner.exports_dir.glob('*.csv')):
    shutil.copy2(source, export_copy_dir / source.name)
shutil.copy2(protocol_source, RUN_DIR / 'e031-protocol-preregistered.md')

artifact_paths = sorted(
    path for path in RUN_DIR.rglob('*')
    if path.is_file() and path.name not in {'e031-artifact-manifest.json'}
    and not path.name.endswith('.tmp')
)
artifact_checksums = {
    str(path.relative_to(RUN_DIR)).replace('\\\\', '/'): sha256_file(path)
    for path in artifact_paths
}
manifest = {
    'experiment': EXPERIMENT_NAME,
    'created_at': datetime.now(UTC).isoformat(),
    'git_commit': runtime_commit,
    'runtime_image': runtime_image,
    'runtime_image_digest': runtime_image_digest,
    'api_runtime': api_runtime,
    'api_schema_validation': schema['validation'],
    'api_quality_scoring': schema['quality_scoring'],
    'model_revisions': expected_revisions,
    'diffqrcoder_revision': schema['notes']['upstream_revision'],
    'dataset_sha256': dataset_sha256,
    'candidate_pool_sha256': pool_sha256,
    'advisor_semantic_sha256': advisor_sha256,
    'advisor_file_sha256': model_file_sha256,
    'advisor_model_path': str(model_path),
    'advisor_clip_model_id': quality_scorer.model_id,
    'advisor_clip_model_revision': clip_model_revision,
    'advisor_aesthetic_weights_sha256': aesthetic_weights_sha256,
    'advisor_quality_provenance': advisor_quality_provenance,
    'verified_stage2_quality_provenance_count': len(quality_provenance_audit),
    'notebook_semantic_sha256': notebook_semantic_sha256_final,
    'protocol_document_sha256': protocol_document_sha256,
    'plan': plan.public,
    'runner': runner_summary,
    'scorer': scorer_identity,
    'source_checksums': {
        'notebook': sha256_file(notebook_source),
        'protocol_document': sha256_file(protocol_source),
        'validation_py': sha256_file(validation_source),
        'qr_verify_bridge': sha256_file(bridge_source),
        'qr_verify_package_lock': sha256_file(package_lock),
        'advisor_aesthetic_weights': sha256_file(aesthetic_weights_path),
    },
    'artifact_checksums': artifact_checksums,
    'claims': {
        'stage1_deliverable': False,
        'srmpgd_used': False,
        'phone_probability': False,
        'human_review_complete': human_review_complete,
    },
}
manifest_path = RUN_DIR / 'e031-artifact-manifest.json'
atomic_json(manifest_path, manifest)

archive_path = ARCHIVE_ROOT / f'{plan.plan_id}-{EXPERIMENT_NAME}.tar.gz'
temporary_archive = archive_path.with_suffix('.tar.gz.tmp')
with tarfile.open(temporary_archive, 'w:gz') as bundle:
    bundle.add(RUN_DIR, arcname=f'{plan.plan_id}-{EXPERIMENT_NAME}')
os.replace(temporary_archive, archive_path)
archive_sha256 = sha256_file(archive_path)
archive_sidecar = archive_path.with_suffix(archive_path.suffix + '.sha256')
atomic_text(archive_sidecar, f'{archive_sha256}  {archive_path.name}\\n')
manifest_sidecar = archive_path.with_suffix(archive_path.suffix + '.manifest.sha256')
atomic_text(manifest_sidecar, f'{sha256_file(manifest_path)}  {manifest_path.name}\\n')

for source in [archive_path, archive_sidecar, manifest_sidecar]:
    shutil.copy2(source, DOWNLOAD_ROOT / source.name)
print('Archive :', archive_path)
print('SHA-256 :', archive_sha256)
print('Copie PC :', DOWNLOAD_ROOT / archive_path.name)
"""
    ),
    markdown(
        """## Lecture finale

1. Téléchargez l'archive depuis `/workspace/downloads`.
2. Inspectez `e031-blind-review/e031-human-review.html` et remplissez le CSV sans ouvrir le
   fichier `reveal`.
3. Relancez les cellules 9 et 10 pour intégrer la revue humaine sans aucune régénération.
4. Ne réentraînez pas le conseiller sur E031 avant d'avoir archivé la décision stop/go.

La documentation de référence se trouve dans
`docs/e031-prospective-stage2-holdout.md` et l'historique dans `docs/experiment-log.md`.
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
