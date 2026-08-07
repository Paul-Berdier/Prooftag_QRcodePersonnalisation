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
        """# E026 — conseiller prompt → paramètres DiffQRCoder

Ce notebook entraîne un **modèle de sélection**, pas un générateur d'images. Il apprend à
prédire les résultats de recettes DiffQRCoder déjà mesurées, puis recommande un top-K pour un
nouveau prompt.

Priorité immuable :

1. probabilité calibrée de réussite `antfu/qr-verify` ;
2. borne basse tenant compte de l'incertitude ;
3. seulement après la porte QR : HPS v2.1, CLIP-Aesthetic et CLIPScore ;
4. à qualité comparable : saturation et durée plus faibles.

La recommandation ne certifie jamais une image. Les candidates réellement générées doivent
encore franchir QR-Verify avant livraison.

```text
exports CSV du laboratoire + prompt + configuration demandée
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

1. Pour créer des données, renseigner `COLLECTION_PAYLOAD`, exécuter les cellules 1 à 3 et
   récupérer les quatre JSON de campagne.
2. Arrêter Jupyter avant de lancer les campagnes dans le laboratoire Web : l'API doit récupérer
   la RTX.
3. Exporter chaque campagne avec **Exporter CSV**.
4. Relancer ce notebook et téléverser les CSV dans `/workspace/imports` depuis JupyterLab.
5. Exécuter toutes les cellules. Si la porte de données est rouge, collecter davantage de cas ;
   le notebook refuse volontairement d'entraîner un faux modèle.

Les anciens CSV sans `prompt_text` peuvent être complétés dans `LEGACY_PROMPT_CATALOG`. Les
nouveaux exports contiennent directement le prompt et la configuration exacte.
"""
    ),
    code(
        """from __future__ import annotations

import glob
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for candidate in [Path('/app'), Path.cwd(), Path.cwd().parent]:
    if (candidate / 'prooftag_qr').is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from prooftag_qr.lab import laboratory_profiles
from prooftag_qr.parameter_advisor import E026ParameterAdvisor, load_lab_exports
from prooftag_qr.quality_scoring import CLIPQualityScorer, project_embedding

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

# Utilisé uniquement pour fabriquer les JSON de collecte ; le payload n'est jamais
# enregistré dans le dataset, seulement sa longueur et son SHA-256.
COLLECTION_PAYLOAD = None  # ex. 'https://ptag.io/t/e026-pilot'

RUN_DIR = Path('/data/notebook-runs') / (
    datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + EXPERIMENT_NAME
)
RUN_DIR.mkdir(parents=True, exist_ok=False)
Path('/workspace/imports').mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = Path('/workspace/results') / RUN_DIR.name
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=False)
print('Résultats :', RUN_DIR)
print('Téléchargements Jupyter :', DOWNLOAD_DIR)
"""
    ),
    markdown("## 2. Plan de collecte E026A — 24 prompts × 10 recettes × 2 seeds"),
    code(
        """COLLECTION_PROMPTS = [
    ('simple_01', 'A single amber pear on slate, soft studio photograph.'),
    ('simple_02', 'One red paper boat floating on dark blue water, minimalist photograph.'),
    ('simple_03', 'A white porcelain cup beside a eucalyptus leaf, quiet editorial still life.'),
    ('simple_04', 'A brass key on burgundy velvet, dramatic museum lighting.'),
    ('simple_05', 'A moon-shaped lamp on an indigo wall, clean product photography.'),
    ('simple_06', 'A solitary kingfisher on a reed, pale morning mist, nature photograph.'),
    ('scene_01', 'A tiled Lisbon courtyard with citrus trees and a small fountain, warm daylight.'),
    ('scene_02', 'A winter library with arched windows, ladders and a glowing fireplace.'),
    ('scene_03', 'A mountain observatory above clouds at sunrise, cinematic landscape.'),
    ('scene_04', 'A Japanese flower market under translucent umbrellas in gentle rain.'),
    ('scene_05', 'An Art Deco railway hall with clocks, travelers and polished brass kiosks.'),
    ('scene_06', 'An underwater museum gallery with rays, coral and blue shafts of light.'),
    ('detail_01', 'An embroidered night garden with moths, irises and silver constellations.'),
    ('detail_02', 'A turquoise peacock mosaic with gold vines and opal flowers, Art Nouveau.'),
    ('detail_03', 'An illuminated manuscript forest with foxes, mushrooms and curling ivy.'),
    ('detail_04', 'A mechanical orrery made of walnut and brass, intricate astronomical detail.'),
    ('detail_05', 'A Persian carpet city seen from above, tiny gardens, canals and lanterns.'),
    ('detail_06', 'A cabinet of botanical curiosities, labeled seeds, shells and glass vials.'),
    ('atypical_01', 'A Möbius opera house folded through violet fog, impossible architecture.'),
    ('atypical_02',
     'A transparent mycelium cube growing miniature blue forests, macro photograph.'),
    ('atypical_03', 'An orchestra of ceramic insects performing inside a pomegranate.'),
    ('atypical_04', 'A crystal droplet containing a complete stormy harbor, surreal macro art.'),
    ('atypical_05', 'A recursive paper city cut from a single map, isometric shadow theatre.'),
    ('atypical_06', 'A bioluminescent archive grown from coral shelves and floating manuscripts.'),
]

profiles = {
    profile['id']: profile for profile in laboratory_profiles()
    if profile['backend'] == 'controlnet' and profile['id'] != 'diffqrcoder_auto'
}
COLLECTION_METHOD_IDS = [
    'diffqrcoder_stage1', 'diffqrcoder_srpg', 'diffqrcoder_paper_srpg',
    'diffqrcoder_srmpgd', 'diffqrcoder_srmpgd_robust',
    'diffqrcoder_srpg_s035', 'diffqrcoder_srpg_s050',
    'diffqrcoder_srpg_s080', 'diffqrcoder_qart_srpg',
]
collection_methods = []
for method_id in COLLECTION_METHOD_IDS:
    method = json.loads(json.dumps(profiles[method_id]))
    method['enabled'] = True
    collection_methods.append(method)

# Dixième recette : même chaîne publique, point intermédiaire non couvert par les profils Web.
extra = json.loads(json.dumps(profiles['diffqrcoder_srpg']))
extra['id'] = 'e026_srpg_qr750_pg1'
extra['name'] = 'E026 — SRPG QR 750 / PG 1'
extra['enabled'] = True
extra['tools']['settings']['srpg_qr_weight'] = 750.0
extra['tools']['settings']['srpg_perceptual_weight'] = 1.0
collection_methods.append(extra)

assert len(COLLECTION_PROMPTS) == 24
assert len(collection_methods) == 10
assert 6 * 2 * len(collection_methods) == 120

collection_dir = RUN_DIR / 'collection-plan'
collection_dir.mkdir()
collection_manifests = []
if COLLECTION_PAYLOAD:
    for batch_index in range(4):
        prompts = COLLECTION_PROMPTS[batch_index * 6:(batch_index + 1) * 6]
        manifest = {
            'name': f'E026A batch {batch_index + 1}/4',
            'payload': COLLECTION_PAYLOAD,
            'error_correction': 'M',
            'prompts': [
                {'id': prompt_id, 'text': text, 'negative_prompt': ''}
                for prompt_id, text in prompts
            ],
            'seeds': [83001, 93001],
            'methods': collection_methods,
            'max_attempts': 1,
        }
        path = collection_dir / f'e026a-campaign-{batch_index + 1:02d}.json'
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        shutil.copy2(path, DOWNLOAD_DIR / path.name)
        collection_manifests.append(path)
    print('Quatre campagnes prêtes :', *collection_manifests, sep='\\n- ')
    print('Arrêter Jupyter avant de les soumettre : le laboratoire doit récupérer le GPU.')
else:
    print('COLLECTION_PAYLOAD=None : aucun manifeste écrit. Le plan reste visible et auditable.')
"""
    ),
    markdown(
        """## 3. Charger les exports et fabriquer les embeddings de prompts

CLIP n'est utilisé ici que pour représenter le texte du prompt. La projection aléatoire est
déterministe et réduit l'embedding à 32 dimensions. Elle doit rester identique en entraînement
et en recommandation.
"""
    ),
    code(
        """csv_paths = sorted({path for pattern in INPUT_GLOBS for path in glob.glob(pattern)})
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
    markdown("## 4. Audit sans fuite et porte minimale"),
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
    markdown("## 5. Entraînement et validation par prompts entièrement inconnus"),
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
    markdown("## 6. Calibration, importance des paramètres et couverture des objectifs"),
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
    markdown("## 7. Recommander les paramètres pour un nouveau prompt"),
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
        """## 8. Lot d'apprentissage actif

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
    markdown("## 9. Manifest, limites et archive"),
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
