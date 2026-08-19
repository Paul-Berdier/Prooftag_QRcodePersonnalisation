"""Build the E028 prompt-advised exact hierarchical cascade notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "23_e028_hierarchical_prompt_advisor.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


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
        """# E028 — conseiller hiérarchique par prompt

Ce notebook met enfin le modèle de paramètres **dans chaque étage** de la chaîne :

```text
prompt ─► conseiller Stage 1 ─► 2 sources esthétiques/structurelles (jamais livrées)
                 │
                 └─► conseiller Stage 2 ─► 2 SRPG par source
                                      │
                                      └─► conseiller SR-MPGD exact par latent Stage 2
```

La production simulée essaie Stage 2 en premier et n'emploie SR-MPGD que si la porte
QR-Verify échoue. Pour apprendre proprement, la campagne calcule néanmoins les deux états de
chaque chaîne. Cela permet une comparaison contrefactuelle strictement appariée.

**Stage 1 n'est jamais un résultat livrable.** QR-Verify est une porte logicielle sur fichier,
pas une promesse de scan téléphone. L'esthétique ne départage les candidats qu'après le payload
exact, la tolérance QR et la garde de saturation.
"""
    ),
    markdown(
        """## Mode d'emploi et reprise

1. Vérifier `COLLECTION_PAYLOAD` dans la configuration.
2. Lancer **Run → Run All Cells**.
3. Après une coupure, relancer : le plan, chaque campagne et chaque CSV sont persistés sous
   `/data/e028-hierarchical/<plan-id>` ; les campagnes exportées ne sont pas recalculées.
4. Le kernel du notebook reste sur CPU. L'API Kubernetes conserve la RTX pour DiffQRCoder.
5. Les images, audits, modèles et graphiques sont copiés dans `/workspace/downloads`.
"""
    ),
    code(
        """# ruff: noqa: E402
from __future__ import annotations

import glob
import hashlib
import json
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image as NotebookImage
from IPython.display import Markdown, clear_output, display

for candidate in [Path('/app'), Path.cwd(), Path.cwd().parent]:
    if (candidate / 'prooftag_qr').is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from prooftag_qr.advisor_gallery import (
    download_advisor_gallery,
    render_advisor_contact_sheet,
    write_gallery_index,
)
from prooftag_qr.advisor_inference import (
    AdvisorInferenceRunner,
    load_advisor_inference_results,
)
from prooftag_qr.e028_hierarchical import (
    E028_PIPELINE_STATES,
    E028_POLICIES,
    audit_e028_pairing,
    build_e028_conditional_datasets,
    build_e028_hierarchical_plan,
    build_e028_holdout_prompts,
    evaluate_e028_policies,
)
from prooftag_qr.parameter_advisor import E026ParameterAdvisor, load_lab_exports
from prooftag_qr.quality_scoring import CLIPQualityScorer, project_embedding

print('Python :', sys.version.split()[0])
print('Répertoire :', Path.cwd())
"""
    ),
    markdown("## 1. Configuration figée"),
    code(
        """EXPERIMENT_NAME = 'e028-hierarchical-prompt-advisor-v1'
INPUT_GLOBS = [
    '/workspace/imports/prooftag-lab-*.csv',
    '/data/e026-input/prooftag-lab-*.csv',
    '/data/e026-week/*/exports/*.csv',
    '/data/e026-week/*/exports-recovered/*.csv',
    '/data/e026j-inference/*/exports/*.csv',
    '/data/e027-holdout/*/exports/*.csv',
]
LEGACY_PROMPT_CATALOG = {}

COLLECTION_API_URL = 'http://prooftag-qr-svc.qr-core.svc.cluster.local:8080'
COLLECTION_PAYLOAD = 'https://ptag.io/t/e028'  # remplacer par votre URL courte réelle
ERROR_CORRECTION = 'M'
QR_CONTEXT = {
    'qr_version': 3,
    'qr_mask_pattern': 4,
    'qr_module_size': 20,
    'qr_padding_px': 78,
}

# 30 prompts inconnus × 3 seeds × 13 états = 1 170 images.
HOLDOUT_PROMPT_COUNT = 30
HOLDOUT_SEEDS = (1083001, 1211001, 1327001)
STAGE1_TOP_K = 2
STAGE2_TOP_K = 2
SCAN_PROBABILITY_THRESHOLD = 0.80
QR_TOLERANCE_THRESHOLD = 0.80
SATURATION_THRESHOLD = 0.05
PROMPT_EMBEDDING_DIMENSIONS = 32
POLL_SECONDS = 15.0
RUN_E028 = True

MINIMUM_ROWS = 100
MINIMUM_PROMPT_GROUPS = 12
MINIMUM_CLASS_COUNT = 12
OUTPUT_ROOT = Path('/data/e028-hierarchical')
NOTEBOOK_RUNS = Path('/data/notebook-runs')
DOWNLOAD_ROOT = Path('/workspace/downloads')
NOTEBOOK_RUNS.mkdir(parents=True, exist_ok=True)
DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

health = json.loads(urlopen(f'{COLLECTION_API_URL}/healthz', timeout=15).read())
print('API :', health)
print('Payload :', COLLECTION_PAYLOAD)
print('Stage 1 livrable : NON')
print('États :', E028_PIPELINE_STATES)
print('Politiques :', E028_POLICIES)
"""
    ),
    markdown(
        """## 2. Réentraîner le conseiller initial sans fuite de prompt

Le modèle apprend la relation **prompt + recette → QR-Verify, tolérance, saturation, HPS,
CLIP-Aesthetic et CLIPScore**. La validation est groupée par texte de prompt : toutes les seeds
d'un prompt restent dans le même pli.
"""
    ),
    code(
        """csv_paths = sorted({path for pattern in INPUT_GLOBS for path in glob.glob(pattern)})
print('CSV trouvés :', len(csv_paths))
for path in csv_paths:
    print('-', path)
if not csv_paths:
    raise RuntimeError('Aucun export CSV trouvé dans INPUT_GLOBS.')

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
        full = quality_scorer.text_embedding(prompt)
        embedding_cache[key] = project_embedding(
            full, dimensions=PROMPT_EMBEDDING_DIMENSIONS, seed=20260721
        )
    return embedding_cache[key]

dataset = load_lab_exports(
    csv_paths,
    prompt_catalog=LEGACY_PROMPT_CATALOG,
    embedding_provider=prompt_embedding,
)
display(pd.DataFrame([dataset.audit]))
problems = []
if dataset.audit['usable_rows'] < MINIMUM_ROWS:
    problems.append(f"{dataset.audit['usable_rows']} lignes < {MINIMUM_ROWS}")
if dataset.audit['prompt_groups'] < MINIMUM_PROMPT_GROUPS:
    problems.append(f"{dataset.audit['prompt_groups']} prompts < {MINIMUM_PROMPT_GROUPS}")
if min(dataset.audit['qr_successes'], dataset.audit['qr_failures']) < MINIMUM_CLASS_COUNT:
    problems.append('classe QR-Verify minoritaire insuffisante')
if problems:
    raise RuntimeError('Dataset E028 non identifiable : ' + '; '.join(problems))

advisor = E026ParameterAdvisor(
    trees=512,
    uncertainty_penalty=0.75,
    random_state=20260819,
)
training_report = advisor.fit(
    dataset.records,
    minimum_rows=MINIMUM_ROWS,
    minimum_groups=MINIMUM_PROMPT_GROUPS,
    minimum_class_count=MINIMUM_CLASS_COUNT,
)
training_prompt_texts = sorted({record.prompt_text for record in dataset.records})
dataset_hasher = hashlib.sha256()
for record in sorted(dataset.records, key=lambda item: item.trial_id):
    dataset_hasher.update(
        json.dumps(
            {
                'trial_id': record.trial_id,
                'group_id': record.group_id,
                'parameters': record.parameters,
                'targets': record.targets,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    )
dataset_sha256 = dataset_hasher.hexdigest()
advisor_fingerprint = {
    'class': 'E026ParameterAdvisor',
    'dataset_sha256': dataset_sha256,
    'training_report': training_report,
    'feature_names': advisor.feature_names,
    'candidate_signatures': sorted(item.signature for item in dataset.candidates),
}
advisor_sha256 = hashlib.sha256(
    json.dumps(advisor_fingerprint, sort_keys=True, separators=(',', ':')).encode('utf-8')
).hexdigest()
display(pd.DataFrame([training_report]).T.rename(columns={0: 'valeur'}))
print('Conseiller SHA-256 :', advisor_sha256)
"""
    ),
    markdown(
        """## 3. Construire les chaînes conseillées avant toute génération

Pour chaque prompt, le modèle choisit deux profils Stage 1, deux Stage 2 par Stage 1, puis un
SR-MPGD par latent Stage 2. Une chaîne fixe sert de témoin. L'ordre des treize méthodes dans
chaque campagne est obligatoire : tous les Stage 1, tous les Stage 2, puis tous les SR-MPGD.
"""
    ),
    code(
        """holdout_prompts = build_e028_holdout_prompts(
    HOLDOUT_PROMPT_COUNT,
    seen_prompt_texts=training_prompt_texts,
)
plan = build_e028_hierarchical_plan(
    advisor=advisor,
    candidates=dataset.candidates,
    prompts=holdout_prompts,
    payload=COLLECTION_PAYLOAD,
    advisor_sha256=advisor_sha256,
    prompt_embedding_provider=prompt_embedding,
    seen_prompt_texts=training_prompt_texts,
    seeds=HOLDOUT_SEEDS,
    stage1_top_k=STAGE1_TOP_K,
    stage2_top_k=STAGE2_TOP_K,
    scan_probability_threshold=SCAN_PROBABILITY_THRESHOLD,
    qr_tolerance_threshold=QR_TOLERANCE_THRESHOLD,
    saturation_threshold=SATURATION_THRESHOLD,
    error_correction=ERROR_CORRECTION,
    qr_context=QR_CONTEXT,
    include_fixed_control=True,
)
expected_methods = 3 + STAGE1_TOP_K + 2 * STAGE1_TOP_K * STAGE2_TOP_K
expected_trials = HOLDOUT_PROMPT_COUNT * len(HOLDOUT_SEEDS) * expected_methods
assert plan.public['context_count'] == 90
assert plan.public['trial_count'] == expected_trials == 1170
assert all(item['method_count'] == expected_methods for item in plan.public['campaigns'])

first_states = [
    row['pipeline_state']
    for row in plan.predictions
    if row['prompt_id'] == holdout_prompts[0]['id']
]
assert first_states == sorted(
    first_states, key=lambda state: E028_PIPELINE_STATES.index(state)
)
display(pd.DataFrame([{
    'plan': plan.plan_id,
    'conseiller': advisor_sha256,
    'prompts inconnus': plan.public['prompt_count'],
    'seeds': plan.public['seed_count'],
    'contextes': plan.public['context_count'],
    'méthodes par prompt': expected_methods,
    'générations': plan.public['trial_count'],
    'candidats disponibles': plan.public['candidate_pool_counts'],
}]))
display(pd.DataFrame([
    {
        'méthode': row['plan_method_id'],
        'état': row['pipeline_state'],
        'chaîne': row['chain_id'],
        'profil choisi': row['selection_profile'],
        'P(QR)': row['predicted_qr_success'],
        'borne QR': row['predicted_qr_success_lower_bound'],
        'tolérance prédite': row['predicted_qr_tolerance'],
        'HPS prédit': row['predicted_hpsv2_1'],
        'observations de la recette': row['candidate_observations'],
    }
    for row in plan.predictions
    if row['prompt_id'] == holdout_prompts[0]['id']
]))
"""
    ),
    markdown("## 4. Génération persistante et reprenable"),
    code(
        """events = deque(maxlen=20)
started = time.monotonic()

def progress(event):
    events.append(event)
    clear_output(wait=True)
    latest = events[-1]
    display(Markdown('### Progression E028'))
    display(pd.DataFrame([{
        'événement': latest.get('event'),
        'prompt': (
            f"{latest.get('prompt_number', 0)}/"
            f"{latest.get('prompt_count', HOLDOUT_PROMPT_COUNT)}"
        ),
        'état': latest.get('status', 'running'),
        'essais': f"{latest.get('completed_trials', 0)}/{latest.get('total_trials', 0)}",
        'acceptés QR-Verify': latest.get('accepted_trials', 0),
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
    progress_callback=progress,
)
print('Plan persistant :', runner.output_dir)
print('État de reprise :', runner.state_path)
summary = runner.run() if RUN_E028 else runner.summary()
display(pd.DataFrame([summary]).T.rename(columns={0: 'valeur'}))
"""
    ),
    markdown(
        """## 5. Prouver l'appariement et rejouer les politiques de production

Le contrôle ne se contente pas des noms de méthodes : il suit les `run_id`, le hash de l'image
Stage 1, le hash du latent Stage 2 et le marqueur backend `exact_reuse`. Un résultat techniquement
généré mais non apparié arrête le notebook.
"""
    ),
    code(
        """rows = load_advisor_inference_results(runner.output_dir)
frame = pd.DataFrame(rows)
run_timestamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
RUN_DIR = NOTEBOOK_RUNS / f'{run_timestamp}-{EXPERIMENT_NAME}'
RUN_DIR.mkdir(parents=True, exist_ok=False)
frame.to_csv(RUN_DIR / 'e028-state-results.csv', index=False)

pairing_rows = audit_e028_pairing(rows)
pairing = pd.DataFrame(pairing_rows)
pairing.to_csv(RUN_DIR / 'e028-pairing-audit.csv', index=False)
generated_pairing = pairing[
    pairing.technically_generated.fillna(False).astype(bool)
]
invalid_pairing = generated_pairing[
    ~generated_pairing.complete.fillna(False).astype(bool)
]
if not invalid_pairing.empty:
    display(invalid_pairing)
    raise RuntimeError(
        f'{len(invalid_pairing)} sorties générées ne prouvent pas leur appariement exact.'
    )

report = evaluate_e028_policies(
    rows,
    qr_tolerance_threshold=QR_TOLERANCE_THRESHOLD,
    saturation_threshold=SATURATION_THRESHOLD,
)
assert all(values['stage1_deliveries'] == 0 for values in report['policies'].values())
decisions = pd.DataFrame(report['decisions'])
policy_table = pd.DataFrame([
    {'politique': name, **values} for name, values in report['policies'].items()
])
decisions.to_csv(RUN_DIR / 'e028-policy-decisions.csv', index=False)
(RUN_DIR / 'e028-policy-report.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
)
display(Markdown('### Verdict — Stage 1 n apparaît dans aucune livraison'))
display(policy_table[[
    'politique', 'contexts', 'delivery_gate_successes',
    'delivery_gate_success_rate', 'prompts_all_seeds_deliverable',
    'mean_qr_tolerance', 'mean_hpsv2_1_delivered',
    'mean_clip_aesthetic_delivered', 'mean_clip_score_delivered',
    'mean_saturation_risk_delivered', 'estimated_generation_units',
    'selected_state_counts', 'stage1_deliveries',
]])
print('Appariements exacts prouvés :', len(generated_pairing), '/', len(generated_pairing))
"""
    ),
    markdown("## 6. Graphiques QR d'abord, esthétique ensuite"),
    code(
        """order = list(E028_POLICIES)
labels = ['Fixe', 'Conseiller top-1', 'Conseiller multi-chaînes']
summary_rows = [report['policies'][name] for name in order]
colors = ['#64748b', '#3a86ff', '#22c55e']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].bar(labels, [row['delivery_gate_success_rate'] for row in summary_rows], color=colors)
axes[0].axhline(0.99, color='black', linestyle='--', label='objectif logiciel 99 %')
axes[0].set_ylim(0, 1.02)
axes[0].set_title('Payload exact + tolérance QR-Verify')
axes[0].legend()
axes[1].bar(labels, [row['mean_hpsv2_1_delivered'] or 0 for row in summary_rows], color=colors)
axes[1].set_title('HPS v2.1 après porte QR')
axes[2].bar(labels, [row['estimated_generation_units'] for row in summary_rows], color=colors)
axes[2].set_title('Coût de génération simulé')
for axis in axes:
    axis.tick_params(axis='x', rotation=18)
    axis.grid(axis='y', alpha=0.25)
fig.tight_layout()
fig.savefig(RUN_DIR / 'e028-policy-scorecard.png', dpi=170)
display(fig)
"""
    ),
    markdown(
        """## 7. Apprendre les prochains conseillers conditionnels

Ces deux datasets relient maintenant Stage 2 à **la sortie Stage 1 réellement mesurée**, puis
SR-MPGD à **la sortie Stage 2 réellement mesurée**. Ils préparent la prochaine décision en ligne,
qui pourra éviter de calculer les branches manifestement mauvaises au lieu de tout planifier à
l'avance.
"""
    ),
    code(
        """conditional = build_e028_conditional_datasets(
    rows,
    prompt_embedding_provider=prompt_embedding,
    qr_tolerance_threshold=QR_TOLERANCE_THRESHOLD,
    saturation_threshold=SATURATION_THRESHOLD,
)
conditional_status = {}
for state, state_dataset in conditional.items():
    audit = state_dataset.audit
    model = E026ParameterAdvisor(
        trees=384,
        uncertainty_penalty=0.75,
        random_state=20260819 + (1 if state == 'stage2' else 2),
    )
    try:
        state_report = model.fit(
            state_dataset.records,
            minimum_rows=60,
            minimum_groups=12,
            minimum_class_count=8,
        )
        model.save(RUN_DIR / f'e028-{state}-conditional-advisor.joblib')
        conditional_status[state] = {'trained': True, 'audit': audit, 'report': state_report}
    except ValueError as exc:
        conditional_status[state] = {
            'trained': False,
            'audit': audit,
            'reason': str(exc),
        }
    with (RUN_DIR / f'e028-{state}-conditional-dataset.jsonl').open(
        'w', encoding='utf-8'
    ) as stream:
        for record in state_dataset.records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + '\\n')
(RUN_DIR / 'e028-conditional-training-status.json').write_text(
    json.dumps(conditional_status, ensure_ascii=False, indent=2), encoding='utf-8'
)
display(pd.DataFrame([
    {'état': state, 'entraîné': value['trained'], **value['audit']}
    for state, value in conditional_status.items()
]))
"""
    ),
    markdown("## 8. Images générées et planches appariées"),
    code(
        """gallery_dir = RUN_DIR / 'e028-gallery'
gallery_entries = download_advisor_gallery(
    rows,
    api_url=COLLECTION_API_URL,
    output_dir=gallery_dir / 'images',
    timeout=30,
)
write_gallery_index(gallery_entries, gallery_dir)

def context_key(row):
    return (str(row.get('prompt_id')), int(row.get('seed') or 0))

sample_keys = set(sorted({context_key(row) for row in gallery_entries})[:8])
sample = [row for row in gallery_entries if context_key(row) in sample_keys]
sample.sort(key=lambda row: (
    context_key(row),
    E028_PIPELINE_STATES.index(row['pipeline_state']),
    str(row.get('chain_id')),
))
sample_path = render_advisor_contact_sheet(
    sample,
    title='E028 — chaînes conseillées et appariées',
    output_path=gallery_dir / 'paired-advisor-sample.png',
    columns=5,
)
display(NotebookImage(filename=str(sample_path)))

winner_ids = set(
    decisions[
        (decisions.policy == 'advisor_best_of_chains')
        & decisions.deliverable.fillna(False).astype(bool)
    ].generation_run_id.dropna()
)
winners = [row for row in gallery_entries if row.get('generation_run_id') in winner_ids]
if winners:
    winner_path = render_advisor_contact_sheet(
        winners[:60],
        title='E028 — sorties livrables du conseiller multi-chaînes',
        output_path=gallery_dir / 'advisor-deliverable-winners.png',
        columns=5,
    )
    display(NotebookImage(filename=str(winner_path)))
print('Images indexées :', len(gallery_entries))
print('Dossier :', gallery_dir / 'images')
"""
    ),
    markdown("## 9. Manifest scientifique et archive"),
    code(
        """advisor.save(RUN_DIR / 'e028-initial-prompt-parameter-advisor.joblib')
(RUN_DIR / 'training-report.json').write_text(
    json.dumps(training_report, ensure_ascii=False, indent=2), encoding='utf-8'
)
manifest = {
    'experiment': EXPERIMENT_NAME,
    'created_at': datetime.now(UTC).isoformat(),
    'plan': plan.public,
    'runner': summary,
    'advisor_sha256': advisor_sha256,
    'stage1_delivery_allowed': False,
    'production_policy': 'Stage 2 first; SR-MPGD only when Stage 2 fails the QR gate',
    'selection_order': [
        'exact_payload', 'qr_verify_tolerance', 'saturation_guard',
        'hpsv2_1', 'clip_aesthetic', 'clip_score',
    ],
    'limitations': [
        'QR-Verify is a software file gate, not a physical-phone guarantee.',
        'The paired experiment computes SR-MPGD counterfactually even when Stage 2 passes.',
        'HPS, CLIP-Aesthetic and CLIPScore are proxies and do not replace human ratings.',
        'Conditional Stage 2 and SR-MPGD advisors become usable only when their data gate passes.',
    ],
}
(RUN_DIR / 'manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
)

download_dir = DOWNLOAD_ROOT / f'e028-{plan.plan_id}'
download_dir.mkdir(parents=True, exist_ok=True)
for path in RUN_DIR.iterdir():
    if path.is_file():
        shutil.copy2(path, download_dir / path.name)
shutil.copytree(RUN_DIR / 'e028-gallery', download_dir / 'e028-gallery', dirs_exist_ok=True)
archive = shutil.make_archive(
    str(RUN_DIR), 'gztar', root_dir=RUN_DIR.parent, base_dir=RUN_DIR.name
)
download_archive = shutil.copy2(archive, DOWNLOAD_ROOT / Path(archive).name)
print('Archive téléchargeable :', download_archive)
print('Résultats visibles :', download_dir)
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
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(TARGET)
