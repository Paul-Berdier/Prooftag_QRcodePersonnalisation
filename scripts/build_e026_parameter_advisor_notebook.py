"""Build the E026 prompt-to-parameters advisor notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "21_e026_prompt_parameter_advisor.ipynb"


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
        """# E026 — collecte autonome puis conseiller prompt → paramètres DiffQRCoder

Ce notebook exécute la chaîne complète : il **génère les données sur l'API GPU**, les exporte
après chaque lot, reprend automatiquement après une coupure, puis entraîne un modèle de
sélection de paramètres et recommande un top-K pour un nouveau prompt.

Priorité immuable :

1. probabilité calibrée de réussite `antfu/qr-verify` ;
2. borne basse tenant compte de l'incertitude ;
3. seulement après la porte QR : HPS v2.1, CLIP-Aesthetic et CLIPScore ;
4. à qualité comparable : saturation et durée plus faibles.

La recommandation ne certifie jamais une image. Les candidates réellement générées doivent
encore franchir QR-Verify avant livraison.

```text
prompts + recettes -> campagnes persistantes API GPU -> exports CSV
                            |
             état JSON atomique et reprise après incident
                            |
        validation groupée par prompt complètement inconnu
                            |
  P(QR valide) + incertitude + esthétique + durée + saturation
                            |
      porte de scan -> classement -> top-K configurations
                            |
       génération réelle -> QR-Verify -> retour au dataset
```
"""
    ),
    markdown(
        """## Mode d'emploi

1. Renseigner `COLLECTION_PAYLOAD` dans la configuration.
2. Exécuter **Run → Run All Cells**. La cellule 3 soumet les lots à l'API qui possède le GPU.
3. Laisser la page ouverte pour voir la progression. Fermer le navigateur ne détruit pas la
   campagne : l'API continue et l'état est écrit dans `/data/e026-week`.
4. Après une erreur, un redémarrage ou une déconnexion, relancer le même notebook et la même
   cellule : les lots terminés sont ignorés et la campagne active est retrouvée.
5. Quand la collecte se termine (ou atteint sa durée limite), les cellules suivantes chargent
   automatiquement les CSV persistants, auditent les données et entraînent le conseiller.

Le notebook E026 tourne en mode CPU afin de laisser la RTX à l'API de génération. Les anciens
CSV sans `prompt_text` peuvent encore être complétés dans `LEGACY_PROMPT_CATALOG`.
"""
    ),
    code(
        """from __future__ import annotations

import glob
import hashlib
import json
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, clear_output, display

for candidate in [Path('/app'), Path.cwd(), Path.cwd().parent]:
    if (candidate / 'prooftag_qr').is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from prooftag_qr.e026_recovery import recover_e026_exports
from prooftag_qr.parameter_advisor import E026ParameterAdvisor, load_lab_exports
from prooftag_qr.quality_scoring import CLIPQualityScorer, project_embedding
from prooftag_qr.week_campaign import WeekCampaignRunner, build_week_batches

print('Python :', sys.version.split()[0])
print('Répertoire :', Path.cwd())
"""
    ),
    markdown("## 1. Configuration explicite"),
    code(
        """EXPERIMENT_NAME = 'e026-prompt-parameter-advisor-v1'
INPUT_GLOBS = [
    '/workspace/imports/prooftag-lab-*.csv',
    '/data/e026-input/prooftag-lab-*.csv',
    '/data/e026-week/*/exports/*.csv',
    '/data/e026-week/*/exports-recovered/*.csv',
]
LEGACY_PROMPT_CATALOG = {
    # 'ancien_prompt_id': 'Texte exact du prompt ancien',
}

# Porte scientifique : ne pas abaisser pour un modèle destiné à guider la production.
MINIMUM_ROWS = 100
MINIMUM_PROMPT_GROUPS = 12
MINIMUM_CLASS_COUNT = 12
SCAN_PROBABILITY_THRESHOLD = 0.80
TOP_K = 6
PROMPT_EMBEDDING_DIMENSIONS = 32

# À modifier après entraînement pour obtenir une recommandation.
NEW_PROMPT = 'A cobalt glass greenhouse filled with white orchids, elegant editorial photograph.'
NEW_PAYLOAD_LENGTH = 28
NEW_ERROR_CORRECTION = 'M'
NEW_QR_CONTEXT = {
    'qr_version': 3,
    'qr_mask_pattern': 4,
    'qr_module_size': 20,
    'qr_padding_px': 78,
}

# Collecte intégrée. Conserver exactement ces valeurs pour reprendre le même plan.
RUN_COLLECTION = True
COLLECTION_PAYLOAD = 'https://ptag.io/t/e026w'  # remplacer par votre URL courte réelle
COLLECTION_API_URL = 'http://prooftag-qr-svc.qr-core.svc.cluster.local:8080'
COLLECTION_OUTPUT_ROOT = Path('/data/e026-week')
COLLECTION_PROMPT_COUNT = 300
COLLECTION_PROMPTS_PER_BATCH = 10
COLLECTION_SEEDS = (113001, 223001, 337001)
COLLECTION_DURATION_HOURS = 162.0
COLLECTION_POLL_SECONDS = 15.0
COLLECTION_MINIMUM_FREE_GIB = 8.0

RUN_DIR = Path('/data/notebook-runs') / (
    datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + EXPERIMENT_NAME
)
RUN_DIR.mkdir(parents=True, exist_ok=False)
Path('/workspace/imports').mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = Path('/workspace/results') / RUN_DIR.name
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=False)
print('Résultats :', RUN_DIR)
print('Téléchargements Jupyter :', DOWNLOAD_DIR)
print('Collecte persistante :', COLLECTION_OUTPUT_ROOT)
print('Ce kernel reste sur CPU ; la génération est exécutée par l API sur la RTX.')
"""
    ),
    markdown("## 2. Construire et auditer le plan de collecte"),
    code(
        """if not COLLECTION_PAYLOAD or COLLECTION_PAYLOAD.endswith('/e026w'):
    print('ATTENTION : remplacez COLLECTION_PAYLOAD par une URL courte Prooftag réelle.')

collection_batches = build_week_batches(
    COLLECTION_PAYLOAD,
    prompt_count=COLLECTION_PROMPT_COUNT,
    prompts_per_batch=COLLECTION_PROMPTS_PER_BATCH,
    seeds=COLLECTION_SEEDS,
)
collection_plan = {
    'batches': len(collection_batches),
    'prompts': sum(len(batch['prompts']) for batch in collection_batches),
    'methods': len(collection_batches[0]['methods']),
    'seeds': len(collection_batches[0]['seeds']),
    'trials': sum(
        len(batch['prompts']) * len(batch['methods']) * len(batch['seeds'])
        for batch in collection_batches
    ),
    'duration_limit_hours': COLLECTION_DURATION_HOURS,
}
display(pd.DataFrame([collection_plan]))
display(pd.DataFrame([
    {'id': method['id'], 'name': method['name'], 'output': method['output_variant']}
    for method in collection_batches[0]['methods']
]))
print('Les lots sont déterministes : mêmes paramètres = même plan et reprise du même état.')
"""
    ),
    markdown("## 3. Générer, suivre et reprendre automatiquement"),
    code(
        """progress_events = deque(maxlen=25)
progress_path = None
collection_started = time.monotonic()


def collection_progress(event):
    progress_events.append(event)
    if progress_path is not None:
        with progress_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\\n')
    clear_output(wait=True)
    latest = progress_events[-1]
    elapsed_hours = (time.monotonic() - collection_started) / 3600
    summary = {
        'événement': latest.get('event'),
        'état': latest.get('status', 'en cours'),
        'lot': (
            f"{latest.get('batch_number', '—')}/"
            f"{latest.get('batch_count', len(collection_batches))}"
        ),
        'essais': (
            f"{latest.get('completed_trials', '—')}/"
            f"{latest.get('total_trials', '—')}"
        ),
        'acceptés': latest.get('accepted_trials', '—'),
        'lots terminés': latest.get('completed_batches', 0),
        'prompt actuel': latest.get('current_prompt_id') or '—',
        'méthode actuelle': latest.get('current_method_id') or '—',
        'seed actuelle': latest.get('current_seed') or '—',
        'temps écoulé (h)': round(elapsed_hours, 2),
    }
    display(Markdown('### Progression de la collecte E026'))
    display(pd.DataFrame([summary]))
    display(pd.DataFrame(list(progress_events)[-10:]))


runner = WeekCampaignRunner(
    api_url=COLLECTION_API_URL,
    payload=COLLECTION_PAYLOAD,
    output_root=COLLECTION_OUTPUT_ROOT,
    duration_hours=COLLECTION_DURATION_HOURS,
    minimum_free_gib=COLLECTION_MINIMUM_FREE_GIB,
    poll_seconds=COLLECTION_POLL_SECONDS,
    prompt_count=COLLECTION_PROMPT_COUNT,
    prompts_per_batch=COLLECTION_PROMPTS_PER_BATCH,
    seeds=COLLECTION_SEEDS,
    progress_callback=collection_progress,
)
progress_path = runner.output_dir / 'notebook-progress.jsonl'
print('Plan :', runner.plan_id)
print('État persistant :', runner.state_path)
print('Exports persistants :', runner.exports_dir)
print('État retrouvé :', runner.state['status'])
print('Lots déjà terminés :', len(runner.state['completed_batches']), '/', len(runner.batches))

if RUN_COLLECTION:
    try:
        health = json.loads(urlopen(COLLECTION_API_URL + '/healthz', timeout=20).read())
        print('API génération :', health)
    except Exception as exc:
        raise RuntimeError(
            'API de génération indisponible. Relancer ce notebook avec '
            'scripts/notebook-remote.ps1 -Reset afin d activer le mode conseiller CPU + API GPU.'
        ) from exc
    runner.run()
    clear_output(wait=True)
    final_state = json.loads(runner.state_path.read_text(encoding='utf-8'))
    display(Markdown('### Collecte arrêtée ou terminée — état sauvegardé'))
    display(pd.DataFrame([{
        'plan': runner.plan_id,
        'état': final_state['status'],
        'lots terminés': len(final_state['completed_batches']),
        'lots prévus': len(runner.batches),
        'campagnes soumises': len(final_state['campaigns']),
        'exports CSV': len(list(runner.exports_dir.glob('*.csv'))),
        'reprise': str(runner.state_path),
    }]))
else:
    print('RUN_COLLECTION=False : collecte ignorée, entraînement sur les exports existants.')
"""
    ),
    markdown(
        """## 4. Charger les exports et fabriquer les embeddings de prompts

CLIP n'est utilisé ici que pour représenter le texte du prompt. La projection aléatoire est
déterministe et réduit l'embedding à 32 dimensions. Elle doit rester identique en entraînement
et en recommandation.
"""
    ),
    code(
        """recovery_summary = None
if RUN_COLLECTION and runner.plan_path.exists():
    recovery_summary = recover_e026_exports(
        api_url=COLLECTION_API_URL,
        plan_dir=runner.output_dir,
    )
    display(Markdown('### Récupération PostgreSQL après interruption'))
    display(pd.DataFrame([recovery_summary]))

csv_paths = sorted({path for pattern in INPUT_GLOBS for path in glob.glob(pattern)})
print('CSV trouvés :', len(csv_paths))
for path in csv_paths:
    print('-', path)

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


if csv_paths:
    dataset = load_lab_exports(
        csv_paths,
        prompt_catalog=LEGACY_PROMPT_CATALOG,
        embedding_provider=prompt_embedding,
    )
else:
    dataset = None
    print(
        'STOP données : téléverser les exports CSV dans /workspace/imports '
        'puis relancer cette cellule.'
    )

if dataset is not None:
    (RUN_DIR / 'dataset-audit.json').write_text(
        json.dumps(dataset.audit, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    display(pd.DataFrame([dataset.audit]))
"""
    ),
    markdown("## 5. Audit sans fuite et porte minimale"),
    code(
        """DATA_READY = False
if dataset is not None:
    target_frame = pd.DataFrame([
        {
            'trial_id': record.trial_id,
            'prompt_id': record.prompt_id,
            'prompt_text': record.prompt_text,
            'method_id': record.metadata.get('method_id'),
            **record.targets,
        }
        for record in dataset.records
    ])
    target_frame.to_csv(RUN_DIR / 'policy-dataset-targets.csv', index=False)
    display(target_frame.head())
    coverage = target_frame.notna().sum().sort_values(ascending=False)
    display(coverage.to_frame('labels disponibles'))

    problems = []
    if dataset.audit['usable_rows'] < MINIMUM_ROWS:
        problems.append(f"{dataset.audit['usable_rows']} lignes < {MINIMUM_ROWS}")
    if dataset.audit['prompt_groups'] < MINIMUM_PROMPT_GROUPS:
        problems.append(
            f"{dataset.audit['prompt_groups']} groupes de prompts < {MINIMUM_PROMPT_GROUPS}"
        )
    if min(dataset.audit['qr_successes'], dataset.audit['qr_failures']) < MINIMUM_CLASS_COUNT:
        problems.append(
            'classe QR-Verify minoritaire insuffisante : '
            f"succès={dataset.audit['qr_successes']}, échecs={dataset.audit['qr_failures']}"
        )
    DATA_READY = not problems
    if problems:
        print('STOP — modèle non entraîné :')
        for problem in problems:
            print('-', problem)
    else:
        print('PORTE VERTE — dataset identifiable pour un premier modèle E026.')
"""
    ),
    markdown("## 6. Entraînement et validation par prompts entièrement inconnus"),
    code(
        """advisor = None
training_report = None
if DATA_READY:
    advisor = E026ParameterAdvisor(
        trees=384,
        uncertainty_penalty=0.75,
        random_state=20260805,
    )
    training_report = advisor.fit(
        dataset.records,
        minimum_rows=MINIMUM_ROWS,
        minimum_groups=MINIMUM_PROMPT_GROUPS,
        minimum_class_count=MINIMUM_CLASS_COUNT,
    )
    advisor.save(RUN_DIR / 'prooftag-e026-parameter-advisor.joblib')
    (RUN_DIR / 'training-report.json').write_text(
        json.dumps(training_report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    display(pd.DataFrame([training_report]).T.rename(columns={0: 'valeur'}))
else:
    print('Étape ignorée tant que la porte de données n’est pas verte.')
"""
    ),
    markdown("## 7. Calibration, importance des paramètres et couverture des objectifs"),
    code(
        """if advisor is not None:
    validation = pd.DataFrame(advisor.validation_predictions)
    validation.to_csv(RUN_DIR / 'grouped-validation-predictions.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    bins = np.linspace(0, 1, 6)
    validation['bin'] = pd.cut(
        validation.calibrated_probability, bins=bins, include_lowest=True
    )
    reliability = validation.groupby('bin', observed=False).agg(
        predicted=('calibrated_probability', 'mean'),
        observed=('observed', 'mean'),
        count=('observed', 'size'),
    ).dropna()
    axes[0].plot([0, 1], [0, 1], '--', color='grey')
    axes[0].plot(reliability.predicted, reliability.observed, 'o-', color='#136f63')
    axes[0].set(xlabel='Probabilité annoncée', ylabel='Fréquence QR-Verify observée',
                title='Calibration hors prompts vus')
    axes[0].grid(alpha=0.25)

    importance = pd.DataFrame(
        advisor.feature_importances[:25], columns=['feature', 'importance']
    ).sort_values('importance')
    axes[1].barh(importance.feature, importance.importance, color='#3a86ff')
    axes[1].set(title='25 variables les plus importantes', xlabel='Importance ExtraTrees')
    axes[1].grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(RUN_DIR / 'validation-and-feature-importance.png', dpi=170)
    display(fig)
    importance.to_csv(RUN_DIR / 'feature-importance.csv', index=False)
else:
    print('Graphiques indisponibles : aucun modèle entraîné.')
"""
    ),
    markdown("## 8. Recommander les paramètres pour un nouveau prompt"),
    code(
        """recommendations = []
if advisor is not None:
    new_embedding = prompt_embedding(NEW_PROMPT)
    recommendations = advisor.recommend(
        prompt=NEW_PROMPT,
        prompt_embedding=new_embedding,
        payload_length=NEW_PAYLOAD_LENGTH,
        error_correction=NEW_ERROR_CORRECTION,
        qr_context=NEW_QR_CONTEXT,
        candidates=dataset.candidates,
        scan_probability_threshold=SCAN_PROBABILITY_THRESHOLD,
        limit=min(TOP_K, len(dataset.candidates)),
    )
    rows = []
    for item in recommendations:
        row = item.to_dict()
        candidate = row.pop('candidate')
        row.update({
            'recipe_id': candidate['id'],
            'method_id': candidate['method_id'],
            'observations': candidate['observations'],
            'configuration_json': json.dumps(
                candidate['configuration'], ensure_ascii=False, sort_keys=True
            ),
        })
        rows.append(row)
    recommendation_frame = pd.DataFrame(rows)
    recommendation_frame.to_csv(RUN_DIR / 'recommendations.csv', index=False)
    (RUN_DIR / 'recommendations.json').write_text(
        json.dumps(
            {
                'prompt': NEW_PROMPT,
                'payload_length': NEW_PAYLOAD_LENGTH,
                'error_correction': NEW_ERROR_CORRECTION,
                'qr_context': NEW_QR_CONTEXT,
                'scan_probability_threshold': SCAN_PROBABILITY_THRESHOLD,
                'recommendations': [item.to_dict() for item in recommendations],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    display(recommendation_frame[[
        'rank', 'method_id', 'scan_safe', 'predicted_qr_success',
        'qr_success_lower_bound', 'qr_success_uncertainty',
        'predicted_human_overall', 'predicted_human_aesthetic',
        'predicted_hpsv2_1', 'predicted_clip_aesthetic',
        'predicted_clip_score', 'predicted_saturation_risk',
        'predicted_duration_ms', 'observations',
    ]])
    if not any(item.scan_safe for item in recommendations):
        print(
            'ATTENTION : aucune recette ne franchit la porte probabiliste. '
            'Générer pour explorer, pas livrer.'
        )
else:
    print('Recommandation indisponible : terminer d’abord la collecte.')
"""
    ),
    markdown(
        """## 9. Lot d'apprentissage actif

Le prochain lot mélange exploitation et exploration : trois recettes au meilleur compromis sûr,
puis trois recettes très incertaines. Cela évite de répéter uniquement les configurations déjà
connues et améliore progressivement le modèle.
"""
    ),
    code(
        """if advisor is not None:
    all_predictions = advisor.recommend(
        prompt=NEW_PROMPT,
        prompt_embedding=prompt_embedding(NEW_PROMPT),
        payload_length=NEW_PAYLOAD_LENGTH,
        error_correction=NEW_ERROR_CORRECTION,
        qr_context=NEW_QR_CONTEXT,
        candidates=dataset.candidates,
        scan_probability_threshold=SCAN_PROBABILITY_THRESHOLD,
        limit=len(dataset.candidates),
    )
    exploitation = all_predictions[:3]
    used = {item.candidate.signature for item in exploitation}
    exploration = sorted(
        [item for item in all_predictions if item.candidate.signature not in used],
        key=lambda item: item.qr_success_uncertainty,
        reverse=True,
    )[:3]
    active_batch = {
        'prompt': NEW_PROMPT,
        'selection': '3 exploitation + 3 maximum uncertainty',
        'candidates': [item.to_dict() for item in exploitation + exploration],
    }
    (RUN_DIR / 'active-learning-batch.json').write_text(
        json.dumps(active_batch, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    display(pd.DataFrame([
        {
            'role': 'exploitation' if item in exploitation else 'exploration',
            'method_id': item.candidate.method_id,
            'P_qr': item.predicted_qr_success,
            'borne_basse': item.qr_success_lower_bound,
            'incertitude': item.qr_success_uncertainty,
        }
        for item in exploitation + exploration
    ]))
"""
    ),
    markdown("## 10. Manifest, limites et archive"),
    code(
        """manifest = {
    'experiment': EXPERIMENT_NAME,
    'created_at': datetime.now(timezone.utc).isoformat(),
    'input_csv': csv_paths,
    'dataset_audit': dataset.audit if dataset is not None else None,
    'data_ready': DATA_READY,
    'training_report': training_report,
    'objective_order': [
        'qr_verify_probability_lower_bound',
        'qr_verify_probability',
        'qr_verify_tolerance',
        'human_overall_if_sufficiently_labeled',
        'human_aesthetic_if_sufficiently_labeled',
        'hpsv2_1',
        'clip_aesthetic',
        'clip_score',
        'low_saturation',
        'low_duration',
    ],
    'limitations': [
        'The advisor never replaces final qr-verify validation.',
        'Recommendations are restricted to historically observed recipe configurations.',
        'Tree dispersion is a heuristic epistemic uncertainty, not a formal guarantee.',
        'CLIP-Aesthetic, CLIPScore and HPS are proxies; human ratings remain valuable labels.',
        'Seeds are sampled at generation time, not treated as a numerically predictable parameter.',
    ],
}
(RUN_DIR / 'manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
)

archive = shutil.make_archive(
    str(RUN_DIR), 'gztar', root_dir=RUN_DIR.parent, base_dir=RUN_DIR.name
)
download_archive = shutil.copy2(archive, DOWNLOAD_DIR / Path(archive).name)
print('Archive :', archive)
print('Archive téléchargeable dans Jupyter :', download_archive)
print('Modèle entraîné :', bool(advisor))
print('La prochaine décision de livraison reste une validation réelle QR-Verify.')
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
