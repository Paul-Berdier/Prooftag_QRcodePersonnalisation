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
from IPython.display import Image as NotebookImage
from IPython.display import Markdown, clear_output, display

for candidate in [Path('/app'), Path.cwd(), Path.cwd().parent]:
    if (candidate / 'prooftag_qr').is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from prooftag_qr.e026_recovery import recover_e026_exports
from prooftag_qr.advisor_inference import (
    AdvisorInferenceRunner,
    build_advisor_inference_plan,
    load_advisor_inference_results,
    select_advisor_inference_winners,
    summarize_advisor_inference_results,
)
from prooftag_qr.advisor_gallery import (
    download_advisor_gallery,
    render_advisor_contact_sheet,
    select_advisor_gallery,
    write_gallery_index,
)
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

# Rapport visuel inclus dans l'archive. La comparaison emploie le même seed.
GALLERY_COMPARISON_METHODS = (
    'diffqrcoder_stage1',
    'diffqrcoder_srpg',
    'diffqrcoder_srmpgd_robust',
    'e026w_srpg_q250_pg1',
)
GALLERY_PROMPT_COUNT = 8
GALLERY_SECTION_SIZE = 8
GALLERY_SEED = 113001

# Prompts d'inférence strictement inconnus : cinq simples puis cinq atypiques.
ADVISOR_INFERENCE_PROMPTS = [
    {
        'id': 'e026i_simple_teapot',
        'text': (
            'A handmade ceramic teapot on a linen table, soft window light, '
            'editorial photograph.'
        ),
    },
    {
        'id': 'e026i_simple_bicycle',
        'text': 'A red bicycle leaning against a pale concrete wall, clean afternoon light.',
    },
    {
        'id': 'e026i_simple_perfume',
        'text': 'A clear perfume bottle on a black pedestal, precise studio product photography.',
    },
    {
        'id': 'e026i_simple_cabin',
        'text': 'A small wooden cabin in a quiet snowy clearing at sunrise.',
    },
    {
        'id': 'e026i_simple_lemons',
        'text': 'A bowl of yellow lemons on a blue kitchen counter, natural daylight.',
    },
    {
        'id': 'e026i_atypical_whale_library',
        'text': (
            'A transparent whale-shaped library floating inside a storm cloud, '
            'cinematic concept art.'
        ),
    },
    {
        'id': 'e026i_atypical_clock_orchestra',
        'text': 'An orchestra of antique clocks performing inside an abandoned greenhouse.',
    },
    {
        'id': 'e026i_atypical_mobius',
        'text': 'A Mobius staircase woven from crimson velvet and thousands of fireflies.',
    },
    {
        'id': 'e026i_atypical_jellyfish',
        'text': 'A crystalline jellyfish serving tea in a monumental brutalist hotel lobby.',
    },
    {
        'id': 'e026i_atypical_egg_city',
        'text': 'A miniature rain-soaked city growing inside a cracked porcelain egg.',
    },
]

# Le premier prompt garde la cellule de recommandation détaillée.
NEW_PROMPT = ADVISOR_INFERENCE_PROMPTS[0]['text']
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

# Campagne réelle après entraînement : top-3 E026 contre Stage 1, mêmes trois seeds.
RUN_ADVISOR_INFERENCE = True
ADVISOR_INFERENCE_OUTPUT_ROOT = Path('/data/e026-inference')
ADVISOR_INFERENCE_TOP_K = 3
ADVISOR_INFERENCE_BASELINE = 'diffqrcoder_stage1'
ADVISOR_INFERENCE_SEEDS = (413001, 523001, 631001)
ADVISOR_INFERENCE_POLL_SECONDS = 15.0
NEW_PAYLOAD_LENGTH = len(COLLECTION_PAYLOAD)

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
            'seed': record.metadata.get('seed'),
            'generation_run_id': record.metadata.get('generation_run_id'),
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
    markdown(
        """## 8. Rapport visuel des QR générés

Le rapport ne choisit pas seulement les jolies images. Il contient quatre vues complémentaires :

- comparaison appariée des quatre recettes principales sur les mêmes prompts et le même seed ;
- meilleurs QR à la fois scannables et esthétiques ;
- images esthétiques qui échouent à QR-Verify ;
- cas proches de la frontière de décision du conseiller.

Les PNG individuels, les planches contact et un index CSV sont inclus dans l'archive finale.
Une image ancienne supprimée du stockage apparaît comme `IMAGE INDISPONIBLE`, sans interrompre
l'entraînement.
"""
    ),
    code(
        """gallery_entries = []
gallery_paths = []
gallery_dir = RUN_DIR / 'visual-gallery'
if advisor is not None:
    selected_gallery = select_advisor_gallery(
        dataset.records,
        validation_predictions=advisor.validation_predictions,
        comparison_method_ids=GALLERY_COMPARISON_METHODS,
        comparison_prompt_count=GALLERY_PROMPT_COUNT,
        preferred_seed=GALLERY_SEED,
        section_size=GALLERY_SECTION_SIZE,
    )
    gallery_entries = download_advisor_gallery(
        selected_gallery,
        api_url=COLLECTION_API_URL,
        output_dir=gallery_dir / 'images',
        timeout=30,
    )
    write_gallery_index(gallery_entries, gallery_dir)
    section_titles = {
        'comparison': 'Comparaison appariée - mêmes prompts et seed',
        'best_scannable': 'Meilleurs QR scannables et esthétiques',
        'aesthetic_failures': 'Beaux candidats qui échouent à QR-Verify',
        'uncertain': 'Cas incertains pour le conseiller',
    }
    for section, title in section_titles.items():
        section_entries = [
            entry for entry in gallery_entries if entry['section'] == section
        ]
        if not section_entries:
            continue
        sheet_path = render_advisor_contact_sheet(
            section_entries,
            title=title,
            output_path=gallery_dir / f'{section}.png',
            columns=len(GALLERY_COMPARISON_METHODS) if section == 'comparison' else 4,
        )
        gallery_paths.append(sheet_path)
        display(Markdown(f'### {title}'))
        display(NotebookImage(filename=str(sheet_path)))
    gallery_audit = json.loads(
        (gallery_dir / 'gallery-audit.json').read_text(encoding='utf-8')
    )
    display(pd.DataFrame([gallery_audit]))
else:
    gallery_audit = None
    print('Galerie indisponible : aucun modèle entraîné.')
"""
    ),
    markdown("## 9. Recommander les paramètres pour un nouveau prompt"),
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
        """## 10. Générer réellement avec les recommandations E026

Cette étape est la preuve qui manquait. Pour chacun des dix prompts jamais vus pendant
l'entraînement, le conseiller choisit ses trois recettes les plus sûres. L'API GPU génère ces
trois candidats avec exactement les mêmes seeds qu'une baseline Stage 1 fixe.

Le plan est déterministe et persistant dans `/data/e026-inference` : après une coupure, relancer
la cellule retrouve la campagne distante et les CSV déjà exportés. Chaque PNG conserve son rang
E026, la recette source, la prédiction avant génération et les scores réellement mesurés après
génération. Seul QR-Verify décide du succès QR réel.
"""
    ),
    code(
        """inference_runner = None
inference_plan = None
inference_summary = None
inference_rows = []
inference_frame = pd.DataFrame()
inference_gallery_entries = []
inference_gallery_paths = []
inference_gallery_audit = None
inference_evaluation = None

if advisor is not None:
    advisor_fingerprint = {
        'class': 'E026ParameterAdvisor',
        'random_state': advisor.random_state,
        'trees': advisor.trees,
        'uncertainty_penalty': advisor.uncertainty_penalty,
        'training_report': advisor.training_report,
        'feature_names': advisor.feature_names,
        'feature_importances': advisor.feature_importances,
        'candidate_signatures': sorted(item.signature for item in dataset.candidates),
    }
    advisor_sha256 = hashlib.sha256(
        json.dumps(
            advisor_fingerprint, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    ).hexdigest()
    training_prompt_texts = sorted({record.prompt_text for record in dataset.records})
    inference_plan = build_advisor_inference_plan(
        advisor=advisor,
        candidates=dataset.candidates,
        prompts=ADVISOR_INFERENCE_PROMPTS,
        payload=COLLECTION_PAYLOAD,
        advisor_sha256=advisor_sha256,
        prompt_embedding_provider=prompt_embedding,
        seen_prompt_texts=training_prompt_texts,
        seeds=ADVISOR_INFERENCE_SEEDS,
        top_k=ADVISOR_INFERENCE_TOP_K,
        baseline_method_id=ADVISOR_INFERENCE_BASELINE,
        scan_probability_threshold=SCAN_PROBABILITY_THRESHOLD,
        error_correction=NEW_ERROR_CORRECTION,
        qr_context=NEW_QR_CONTEXT,
    )
    display(Markdown('### Plan d inférence avant génération'))
    display(pd.DataFrame([{
        'plan': inference_plan.plan_id,
        'modèle SHA-256': advisor_sha256,
        'prompts inconnus': len(ADVISOR_INFERENCE_PROMPTS),
        'top-K conseillé': ADVISOR_INFERENCE_TOP_K,
        'seeds appariées': len(ADVISOR_INFERENCE_SEEDS),
        'images comparatives': inference_plan.public['comparison_trial_count'],
        'prérequis SRPG': inference_plan.public['prerequisite_trial_count'],
        'générations GPU totales': inference_plan.public['trial_count'],
        'baseline': ADVISOR_INFERENCE_BASELINE,
    }]))

    inference_events = deque(maxlen=25)
    inference_started = time.monotonic()

    def inference_progress(event):
        inference_events.append(event)
        clear_output(wait=True)
        latest = inference_events[-1]
        display(Markdown('### Génération réelle guidée par E026'))
        display(pd.DataFrame([{
            'événement': latest.get('event'),
            'état': latest.get('status', 'en cours'),
            'prompt': (
                f"{latest.get('prompt_number', '—')}/"
                f"{latest.get('prompt_count', len(ADVISOR_INFERENCE_PROMPTS))}"
            ),
            'essais': (
                f"{latest.get('completed_trials', '—')}/"
                f"{latest.get('total_trials', '—')}"
            ),
            'acceptés QR-Verify': latest.get('accepted_trials', '—'),
            'méthode': latest.get('current_method_id') or '—',
            'seed': latest.get('current_seed') or '—',
            'temps écoulé (min)': round((time.monotonic() - inference_started) / 60, 1),
        }]))
        display(pd.DataFrame(list(inference_events)[-10:]))

    inference_runner = AdvisorInferenceRunner(
        plan=inference_plan,
        api_url=COLLECTION_API_URL,
        output_root=ADVISOR_INFERENCE_OUTPUT_ROOT,
        poll_seconds=ADVISOR_INFERENCE_POLL_SECONDS,
        progress_callback=inference_progress,
    )
    print('Plan persistant :', inference_runner.output_dir)
    print('État de reprise :', inference_runner.state_path)
    if RUN_ADVISOR_INFERENCE:
        inference_summary = inference_runner.run()
    else:
        inference_summary = inference_runner.summary()
        print('RUN_ADVISOR_INFERENCE=False : résultats persistants uniquement.')

    inference_rows = load_advisor_inference_results(inference_runner.output_dir)
    inference_frame = pd.DataFrame(inference_rows)
    if not inference_frame.empty:
        inference_frame['prediction_error'] = (
            inference_frame.qr_success - inference_frame.predicted_qr_success
        )
        inference_frame.to_csv(RUN_DIR / 'advisor-inference-results.csv', index=False)
        display(Markdown('### Résultats réels : prédiction puis mesure'))
        display(inference_frame[[
            'prompt_id', 'role', 'advisor_rank', 'source_method_id', 'seed',
            'predicted_qr_success', 'predicted_qr_success_lower_bound',
            'qr_success', 'qr_tolerance', 'clip_aesthetic', 'clip_score',
            'hpsv2_1', 'saturation_risk', 'duration_ms', 'status',
        ]])
        comparison_frame = inference_frame[
            inference_frame.role.isin(['advisor_recommendation', 'fixed_baseline'])
        ].copy()
        recommended_frame = comparison_frame[
            comparison_frame.role == 'advisor_recommendation'
        ]
        inference_evaluation = summarize_advisor_inference_results(inference_rows)
        (RUN_DIR / 'advisor-inference-evaluation.json').write_text(
            json.dumps(inference_evaluation, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        display(Markdown('### Verdict mesuré du conseiller E026'))
        display(pd.DataFrame([inference_evaluation]).T.rename(columns={0: 'valeur'}))
        if inference_evaluation['technical_error_images']:
            display(Markdown(
                '**ALERTE : le taux principal compte les erreurs techniques comme des '
                'échecs. Le taux suffixé `_generated` ne porte que sur les images '
                'effectivement produites et ne doit pas être présenté seul.**'
            ))
        aggregate = comparison_frame.groupby(
            ['role', 'advisor_rank', 'source_method_id'], dropna=False
        ).agg(
            images_planned=('trial_id', 'size'),
            images_measured=('qr_success', 'count'),
            technical_errors=('status', lambda values: int((values == 'error').sum())),
            qr_verify_success=(
                'qr_success', lambda values: float(values.fillna(0.0).mean())
            ),
            qr_verify_success_generated=('qr_success', 'mean'),
            qr_tolerance=('qr_tolerance', 'mean'),
            clip_aesthetic=('clip_aesthetic', 'mean'),
            clip_score=('clip_score', 'mean'),
            hpsv2_1=('hpsv2_1', 'mean'),
            saturation=('saturation_risk', 'mean'),
            predicted_qr=('predicted_qr_success', 'mean'),
        ).reset_index()
        aggregate.to_csv(RUN_DIR / 'advisor-inference-aggregate.csv', index=False)
        display(Markdown('### Comparaison agrégée E026 contre baseline'))
        display(aggregate)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        labels = [
            ('baseline' if row.role == 'fixed_baseline' else f'rang {int(row.advisor_rank)}')
            + f'\\n{row.source_method_id}'
            for row in aggregate.itertuples()
        ]
        axes[0].scatter(
            aggregate.predicted_qr, aggregate.qr_verify_success,
            s=90, c='#3a86ff', edgecolors='white', linewidths=0.8,
        )
        axes[0].plot([0, 1], [0, 1], '--', color='grey')
        for label, x_value, y_value in zip(
            labels, aggregate.predicted_qr, aggregate.qr_verify_success
        ):
            axes[0].annotate(label, (x_value, y_value), fontsize=7, xytext=(4, 4),
                             textcoords='offset points')
        axes[0].set(
            xlabel='Probabilité QR prédite', ylabel='Succès QR-Verify mesuré',
            title='Prédiction contre réalité', xlim=(0, 1.02), ylim=(0, 1.02),
        )
        axes[0].grid(alpha=0.25)

        generated_frame = comparison_frame[comparison_frame.qr_success.notna()]
        colors = np.where(generated_frame.qr_success >= 0.5, '#22c55e', '#ef4444')
        axes[1].scatter(
            generated_frame.qr_tolerance, generated_frame.clip_aesthetic,
            c=colors, alpha=0.75, s=42,
        )
        axes[1].set(
            xlabel='Tolérance QR-Verify', ylabel='CLIP-Aesthetic',
            title='Scannabilité et esthétique par image',
        )
        axes[1].grid(alpha=0.25)

        positions = np.arange(len(aggregate))
        axes[2].bar(positions, aggregate.hpsv2_1, color='#8338ec')
        axes[2].set_xticks(positions, labels, rotation=55, ha='right')
        axes[2].set(ylabel='HPS v2.1 moyen', title='Préférence visuelle mesurée')
        axes[2].grid(axis='y', alpha=0.25)
        fig.tight_layout()
        scorecard_path = RUN_DIR / 'advisor-inference-scorecard.png'
        fig.savefig(scorecard_path, dpi=170)
        display(fig)

        inference_gallery_dir = RUN_DIR / 'advisor-inference-gallery'
        downloadable = [
            row for row in inference_rows
            if row.get('role') in {'advisor_recommendation', 'fixed_baseline'}
            and row.get('generation_run_id')
        ]
        inference_gallery_entries = download_advisor_gallery(
            downloadable,
            api_url=COLLECTION_API_URL,
            output_dir=inference_gallery_dir / 'images',
            timeout=30,
        )
        write_gallery_index(inference_gallery_entries, inference_gallery_dir)
        for seed in ADVISOR_INFERENCE_SEEDS:
            selected = [
                row for row in inference_gallery_entries
                if int(row.get('seed') or -1) == seed
            ]
            if not selected:
                continue
            path = render_advisor_contact_sheet(
                selected,
                title=f'E026 conseillé contre baseline - seed {seed}',
                output_path=inference_gallery_dir / f'comparison-seed-{seed}.png',
                columns=ADVISOR_INFERENCE_TOP_K + 1,
            )
            inference_gallery_paths.append(path)
            display(Markdown(f'### Comparaison réelle — seed {seed}'))
            display(NotebookImage(filename=str(path)))

        winners = select_advisor_inference_winners(inference_gallery_entries)
        if winners:
            winners_path = render_advisor_contact_sheet(
                winners,
                title='Meilleur QR mesuré pour chaque prompt inconnu',
                output_path=inference_gallery_dir / 'measured-winners.png',
                columns=5,
            )
            inference_gallery_paths.append(winners_path)
            display(Markdown('### Gagnants après QR-Verify et scores esthétiques'))
            display(NotebookImage(filename=str(winners_path)))
        inference_gallery_audit = json.loads(
            (inference_gallery_dir / 'gallery-audit.json').read_text(encoding='utf-8')
        )

        audit_dir = RUN_DIR / 'advisor-inference-audit'
        audit_dir.mkdir(exist_ok=True)
        for source in [
            inference_runner.plan_path,
            inference_runner.predictions_path,
            inference_runner.state_path,
            *sorted(inference_runner.exports_dir.glob('*.csv')),
        ]:
            shutil.copy2(source, audit_dir / source.name)
    else:
        print('Aucun essai exporté pour ce plan. Relancer cette cellule pour reprendre.')
else:
    print('Inférence impossible : le conseiller n est pas entraîné.')
"""
    ),
    markdown(
        """## 11. Lot d'apprentissage actif

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
    markdown("## 12. Manifest, limites et archive"),
    code(
        """manifest = {
    'experiment': EXPERIMENT_NAME,
    'created_at': datetime.now(timezone.utc).isoformat(),
    'input_csv': csv_paths,
    'dataset_audit': dataset.audit if dataset is not None else None,
    'data_ready': DATA_READY,
    'training_report': training_report,
    'visual_gallery': gallery_audit,
    'advisor_inference': {
        'plan': inference_plan.public if inference_plan is not None else None,
        'summary': inference_summary,
        'result_rows': len(inference_rows),
        'evaluation': inference_evaluation,
        'gallery': inference_gallery_audit,
    },
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
        'Gallery images depend on generation artifacts still being present in the API storage.',
        'Advisor inference measures generated candidates; predictions alone never certify them.',
    ],
}
(RUN_DIR / 'manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
)

# Copie visible dans l'explorateur Jupyter, en plus de l'archive complète.
for directory_name in ['visual-gallery', 'advisor-inference-gallery']:
    source = RUN_DIR / directory_name
    if source.is_dir():
        shutil.copytree(source, DOWNLOAD_DIR / directory_name, dirs_exist_ok=True)
for filename in [
    'advisor-inference-results.csv',
    'advisor-inference-aggregate.csv',
    'advisor-inference-evaluation.json',
    'advisor-inference-scorecard.png',
    'manifest.json',
]:
    source = RUN_DIR / filename
    if source.is_file():
        shutil.copy2(source, DOWNLOAD_DIR / filename)

archive = shutil.make_archive(
    str(RUN_DIR), 'gztar', root_dir=RUN_DIR.parent, base_dir=RUN_DIR.name
)
download_archive = shutil.copy2(archive, DOWNLOAD_DIR / Path(archive).name)
print('Archive :', archive)
print('Archive téléchargeable dans Jupyter :', download_archive)
print('Modèle entraîné :', bool(advisor))
print('Planches visuelles :', [str(path) for path in gallery_paths])
print('Planches inférence E026 :', [str(path) for path in inference_gallery_paths])
print('Images visibles dans Jupyter :', DOWNLOAD_DIR / 'advisor-inference-gallery')
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
