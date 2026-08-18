"""Build the E027 paired cascade/full/SR-MPGD holdout notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "22_e027_srmpgd_policy_holdout.ipynb"


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
        """# E027 — faut-il toujours livrer SR-MPGD ?

Ce notebook réalise une comparaison appariée et reprenable sur **300 contextes jamais vus** :
100 prompts × 3 seeds. Pour chaque contexte, l'API produit dans cet ordre :

```text
Stage 1 exact ──► Stage 2 SRPG exact ──► SR-MPGD sur le latent Stage 2 exact
      │                    │                         │
      └────────────────────┴─────────────────────────┘
               même prompt, seed, QR et Stage 1
```

Trois politiques sont ensuite rejouées sur les mesures réelles :

1. **cascade** : livrer Stage 1 s'il franchit déjà la porte robuste, sinon calculer la suite ;
2. **full_lexicographic** : toujours calculer la chaîne complète, puis choisir QR exact, tolérance,
   absence de saturation, HPS, CLIP-Aesthetic et CLIPScore, dans cet ordre ;
3. **forced_srmpgd** : toujours prendre la sortie demandée SR-MPGD, même lorsque Stage 2 est mieux.

SR-MPGD conserve son itération zéro et évalue chaque itération. Une itération tachée ou saturée
ne peut donc pas écraser silencieusement le Stage 2. QR-Verify reste la seule porte QR finale.
"""
    ),
    markdown(
        """## Mode d'emploi

1. Vérifier le payload dans la cellule de configuration.
2. Lancer **Run → Run All Cells** une seule fois.
3. En cas de coupure, relancer les cellules : le plan et chaque export sont persistés sous
   `/data/e027-holdout/<plan-id>` ; les lots terminés ne sont pas recalculés.
4. Les 900 images, les CSV, le rapport JSON et les planches comparatives sont copiés dans
   `/workspace/downloads/e027-<plan-id>` et inclus dans l'archive finale.

Le notebook reste sur CPU. L'API Kubernetes conserve la RTX pour les générations.
"""
    ),
    code(
        """from __future__ import annotations

import json
import shutil
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
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
from prooftag_qr.e027_policy import (
    E027_PIPELINE_STATES,
    E027_POLICIES,
    build_e027_holdout_plan,
    e027_policy_winner_entries,
    evaluate_e027_policies,
)

print('Python :', sys.version.split()[0])
print('Répertoire :', Path.cwd())
"""
    ),
    markdown("## 1. Configuration figée"),
    code(
        """EXPERIMENT_NAME = 'e027-paired-srmpgd-policy-holdout-v1'
COLLECTION_API_URL = 'http://prooftag-qr-svc.qr-core.svc.cluster.local:8080'
COLLECTION_PAYLOAD = 'https://ptag.io/t/e027'
ERROR_CORRECTION = 'M'

# 100 prompts × 3 seeds = 300 contextes indépendants.
HOLDOUT_PROMPT_COUNT = 100
HOLDOUT_SEEDS = (743001, 857001, 971001)
PROMPTS_PER_CAMPAIGN = 20
QR_TOLERANCE_THRESHOLD = 0.80
POLL_SECONDS = 15.0
RUN_E027 = True

OUTPUT_ROOT = Path('/data/e027-holdout')
NOTEBOOK_RUNS = Path('/data/notebook-runs')
DOWNLOAD_ROOT = Path('/workspace/downloads')
NOTEBOOK_RUNS.mkdir(parents=True, exist_ok=True)
DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

health = json.loads(urlopen(f'{COLLECTION_API_URL}/healthz', timeout=15).read())
print('API :', health)
print('Payload :', COLLECTION_PAYLOAD)
print('Contextes :', HOLDOUT_PROMPT_COUNT * len(HOLDOUT_SEEDS))
print('États mesurés :', E027_PIPELINE_STATES)
print('Politiques :', E027_POLICIES)
"""
    ),
    markdown(
        """## 2. Construire le holdout et vérifier l'appariement

Le plan est déterministe. Le payload clair reste seulement en mémoire ; le plan persistant ne
contient que son SHA-256. L'ordre des méthodes est obligatoire : Stage 1, Stage 2, SR-MPGD.
Ainsi l'API réutilise le même Stage 1 puis le latent exact du Stage 2 pour SR-MPGD.
"""
    ),
    code(
        """plan = build_e027_holdout_plan(
    payload=COLLECTION_PAYLOAD,
    prompt_count=HOLDOUT_PROMPT_COUNT,
    seeds=HOLDOUT_SEEDS,
    prompts_per_campaign=PROMPTS_PER_CAMPAIGN,
    error_correction=ERROR_CORRECTION,
    qr_tolerance_threshold=QR_TOLERANCE_THRESHOLD,
)
assert plan.public['context_count'] == 300
assert plan.public['trial_count'] == 900
assert [method['id'] for method in plan.campaigns[0]['methods']] == [
    'e027_stage1', 'e027_stage2', 'e027_srmpgd'
]
display(pd.DataFrame([{
    'plan': plan.plan_id,
    'prompts': plan.public['prompt_count'],
    'seeds': plan.public['seed_count'],
    'contextes': plan.public['context_count'],
    'états générés': plan.public['trial_count'],
    'lots reprenables': plan.public['campaign_count'],
    'porte tolérance': plan.public['qr_tolerance_threshold'],
}]))
"""
    ),
    markdown("## 3. Génération réelle, persistante et reprenable"),
    code(
        """events = deque(maxlen=20)
started = time.monotonic()

def progress(event):
    events.append(event)
    clear_output(wait=True)
    latest = events[-1]
    display(Markdown('### Progression E027'))
    display(pd.DataFrame([{
        'événement': latest.get('event'),
        'lot': (
            f\"{latest.get('prompt_number', 0)}/\"
            f\"{latest.get('prompt_count', plan.public['campaign_count'])}\"
        ),
        'état': latest.get('status', 'running'),
        'essais': f\"{latest.get('completed_trials', 0)}/{latest.get('total_trials', 0)}\",
        'acceptés': latest.get('accepted_trials', 0),
        'prompt': latest.get('current_prompt_id'),
        'étape': latest.get('current_method_id'),
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
summary = runner.run() if RUN_E027 else runner.summary()
display(pd.DataFrame([summary]).T.rename(columns={0: 'valeur'}))
"""
    ),
    markdown(
        """## 4. Vérification des 300 triplets et décision lexicographique

Une absence de mesure compte comme une erreur technique, jamais comme un succès. La porte de
livraison exige le payload QR exact **et** une tolérance QR-Verify supérieure ou égale à 0,80.
À égalité QR seulement, les proxys esthétiques départagent les états.
"""
    ),
    code(
        """rows = load_advisor_inference_results(runner.output_dir)
frame = pd.DataFrame(rows)
run_timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN_DIR = NOTEBOOK_RUNS / f\"{run_timestamp}-{EXPERIMENT_NAME}\"
RUN_DIR.mkdir(parents=True, exist_ok=False)
frame.to_csv(RUN_DIR / 'e027-state-results.csv', index=False)

report = evaluate_e027_policies(
    rows,
    qr_tolerance_threshold=QR_TOLERANCE_THRESHOLD,
)
decisions = pd.DataFrame(report['decisions'])
audit = pd.DataFrame(report['group_audit'])
decisions.to_csv(RUN_DIR / 'e027-policy-decisions.csv', index=False)
audit.to_csv(RUN_DIR / 'e027-pairing-audit.csv', index=False)
(RUN_DIR / 'e027-policy-report.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
)

if report['contexts'] != 300:
    raise RuntimeError(f\"Campagne incomplète : {report['contexts']}/300 contextes présents.\")
if report['complete_contexts'] != 300:
    print('ATTENTION : des triplets sont incomplets. Voir e027-pairing-audit.csv.')

policy_table = pd.DataFrame([
    {'politique': policy, **values}
    for policy, values in report['policies'].items()
])
display(Markdown('### Verdict principal'))
display(policy_table[[
    'politique', 'contexts', 'technical_complete_contexts',
    'exact_qr_successes', 'exact_qr_success_rate',
    'delivery_gate_successes', 'delivery_gate_success_rate',
    'prompts_all_seeds_deliverable', 'prompt_all_seed_success_rate',
    'mean_qr_tolerance', 'mean_hpsv2_1', 'mean_clip_aesthetic',
    'mean_clip_score', 'mean_saturation_risk', 'estimated_generation_units',
    'selected_state_counts',
]])
display(Markdown('### Comparaisons appariées'))
display(pd.DataFrame(report['paired_comparisons']).T)
"""
    ),
    markdown("## 5. Graphiques : QR d'abord, esthétique ensuite"),
    code(
        """fig, axes = plt.subplots(1, 3, figsize=(18, 5))
order = list(E027_POLICIES)
summary_rows = [report['policies'][name] for name in order]
labels = ['Cascade', 'Complet + sélection', 'SR-MPGD forcé']

colors = ['#3a86ff', '#22c55e', '#ff006e']
axes[0].bar(
    labels,
    [row['delivery_gate_success_rate'] for row in summary_rows],
    color=colors,
)
axes[0].axhline(0.99, color='black', linestyle='--', label='objectif 99 %')
axes[0].set_ylim(0, 1.02)
axes[0].set_title('Porte QR-Verify robuste')
axes[0].tick_params(axis='x', rotation=20)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.25)

axes[1].bar(
    labels,
    [row['mean_hpsv2_1'] or 0 for row in summary_rows],
    color=colors,
)
axes[1].set_title('HPS v2.1 après porte QR')
axes[1].tick_params(axis='x', rotation=20)
axes[1].grid(axis='y', alpha=0.25)

axes[2].bar(
    labels,
    [row['estimated_generation_units'] for row in summary_rows],
    color=colors,
)
axes[2].set_title('Unités de génération estimées')
axes[2].tick_params(axis='x', rotation=20)
axes[2].grid(axis='y', alpha=0.25)

fig.tight_layout()
fig.savefig(RUN_DIR / 'e027-policy-scorecard.png', dpi=170)
display(fig)
"""
    ),
    markdown(
        """## 6. Toutes les images et planches de contrôle

Les 900 états sont téléchargés et indexés. Les planches affichent un échantillon apparié, les
échecs de la porte robuste et les contextes où la sélection complète refuse de forcer SR-MPGD.
"""
    ),
    code(
        """gallery_dir = RUN_DIR / 'e027-gallery'
gallery_entries = download_advisor_gallery(
    rows,
    api_url=COLLECTION_API_URL,
    output_dir=gallery_dir / 'images',
    timeout=30,
)
write_gallery_index(gallery_entries, gallery_dir)

def context_key(row):
    return (str(row.get('prompt_id')), int(row.get('seed') or 0))

all_keys = sorted({context_key(row) for row in gallery_entries})
sample_keys = set(all_keys[:12])
sample = [row for row in gallery_entries if context_key(row) in sample_keys]
sample.sort(key=lambda row: (context_key(row), E027_PIPELINE_STATES.index(row['pipeline_state'])))
sample_path = render_advisor_contact_sheet(
    sample,
    title='E027 — Stage 1 / Stage 2 / SR-MPGD appariés',
    output_path=gallery_dir / 'paired-state-sample.png',
    columns=3,
)
display(NotebookImage(filename=str(sample_path)))

failed_keys = {
    (str(row.prompt_id), int(row.seed))
    for row in decisions.itertuples()
    if row.policy == 'full_lexicographic' and not bool(row.deliverable)
}
failed = [row for row in gallery_entries if context_key(row) in set(sorted(failed_keys)[:12])]
if failed:
    failed.sort(
        key=lambda row: (
            context_key(row),
            E027_PIPELINE_STATES.index(row['pipeline_state']),
        )
    )
    failed_path = render_advisor_contact_sheet(
        failed,
        title='E027 — échecs de la porte robuste',
        output_path=gallery_dir / 'robust-gate-failures.png',
        columns=3,
    )
    display(NotebookImage(filename=str(failed_path)))

decision_lookup = {
    (str(row.prompt_id), int(row.seed), row.policy): row
    for row in decisions.itertuples()
}
disagreement_keys = []
for key in all_keys:
    full = decision_lookup.get((*key, 'full_lexicographic'))
    forced = decision_lookup.get((*key, 'forced_srmpgd'))
    if full and forced and full.generation_run_id != forced.generation_run_id:
        disagreement_keys.append(key)
disagreement_set = set(disagreement_keys[:12])
disagreements = [row for row in gallery_entries if context_key(row) in disagreement_set]
if disagreements:
    disagreements.sort(
        key=lambda row: (
            context_key(row),
            E027_PIPELINE_STATES.index(row['pipeline_state']),
        )
    )
    disagreement_path = render_advisor_contact_sheet(
        disagreements,
        title='E027 — cas où SR-MPGD forcé n est pas le meilleur état',
        output_path=gallery_dir / 'forced-srmpgd-disagreements.png',
        columns=3,
    )
    display(NotebookImage(filename=str(disagreement_path)))

winner_rows = e027_policy_winner_entries(gallery_entries, report['decisions'])
pd.DataFrame(winner_rows).to_csv(gallery_dir / 'policy-winners.csv', index=False)
print('Images téléchargées :', len(gallery_entries))
print('Désaccords sélection complète / SR-MPGD forcé :', len(disagreement_keys))
print('Dossier images :', gallery_dir / 'images')
"""
    ),
    markdown("## 7. Manifest et archive téléchargeable"),
    code(
        """manifest = {
    'experiment': EXPERIMENT_NAME,
    'created_at': datetime.now(timezone.utc).isoformat(),
    'plan': plan.public,
    'runner': summary,
    'report': {key: value for key, value in report.items() if key != 'decisions'},
    'result_rows': len(rows),
    'decision_rows': len(report['decisions']),
    'selection_order': [
        'qr_verify_exact_payload',
        'qr_verify_tolerance',
        'saturation_guard',
        'hpsv2_1',
        'clip_aesthetic',
        'clip_score',
        'lower_saturation',
        'lower_duration',
    ],
    'limitations': [
        'QR-Verify is a software gate and does not replace the future physical-phone holdout.',
        'CLIP-Aesthetic, CLIPScore and HPS are aesthetic proxies, not human judgments.',
        'The paired campaign computes every state; cascade compute is replayed, '
        'not physically skipped.',
        'SR-MPGD may retain iteration zero when the Stage 2 already passes its robust gate.',
    ],
}
(RUN_DIR / 'manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
)

download_dir = DOWNLOAD_ROOT / f'e027-{plan.plan_id}'
download_dir.mkdir(parents=True, exist_ok=True)
for filename in [
    'e027-state-results.csv',
    'e027-policy-decisions.csv',
    'e027-pairing-audit.csv',
    'e027-policy-report.json',
    'e027-policy-scorecard.png',
    'manifest.json',
]:
    shutil.copy2(RUN_DIR / filename, download_dir / filename)
shutil.copytree(RUN_DIR / 'e027-gallery', download_dir / 'e027-gallery', dirs_exist_ok=True)
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
