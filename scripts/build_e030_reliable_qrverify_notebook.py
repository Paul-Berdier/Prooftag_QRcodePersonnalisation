"""Build the CPU-only E030 reliable QR-Verify cascade notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "25_e030_reliable_qrverify_cascade.ipynb"


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
        """# E030 — QR-Verify répétable et cascade Stage 2 fiable

Ce notebook **ne génère aucune image et n'utilise pas le GPU**. Il reprend le dernier export
E029 v4 complet du PVC (ou une archive en secours), lit uniquement ses tableaux et ses PNG
(jamais le modèle advisor de 1,35 Go), puis :

1. refuse une archive ou une galerie incomplète ;
2. mesure chaque raster RGB unique cinq fois avec `antfu/qr-verify` ;
3. conserve l'intersection des presets réussis, donc le score le plus conservateur ;
4. simule quatre politiques, dont la chaîne de production proposée :
   **Stage 2 fixe → Stage 2 advisor alternatif → nouvelle seed** ;
5. produit les tableaux, graphes, planche des gagnants, rapport et archive traçable.

Stage 1 n'est jamais livré. SR-MPGD n'est jamais demandé. Les résultats prouvent uniquement la
stabilité du test logiciel : ils ne remplacent pas encore un holdout physique au téléphone.

La reprise est automatique : chaque raster terminé est journalisé et le cache QR-Verify est
adressé par contenu. Après une interruption, relancer **Run All Cells** continue sans rescanner
les rasters déjà validés par la même version du scorer.
"""
    ),
    code(
        """import hashlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, clear_output, display
from PIL import Image

from prooftag_qr.e030_offline import (
    discover_e029_archive,
    discover_e029_export_directory,
    e029_export_sha256,
    selective_extract_e029_archive,
    sha256_file,
    validate_rescore_journal_rows,
    validate_e029_export,
)
from prooftag_qr.policy import (
    ConservativeDeliveryGate,
    assess_stage2_candidate,
    select_stage2_cascade,
)
from prooftag_qr.validation import ConservativeQRVerifyScorer, image_raster_sha256

EXPERIMENT = 'e030-reliable-qrverify-cascade-v1'
SOURCE_ARCHIVE = os.environ.get('PROOFTAG_E030_SOURCE_ARCHIVE') or None
DOWNLOAD_ROOT = Path('/workspace/downloads')
OUTPUT_ROOT = Path('/data/e030-reliable-qrverify')
SOURCE_CACHE_ROOT = OUTPUT_ROOT / 'source-cache'
PERSISTENT_E029_ROOTS = (
    Path('/data/e029-srmpgd-raster'),
    Path('/data/notebook-runs'),
)
QR_VERIFY_CACHE = Path('/data/qr-verify-conservative-cache')
EXPECTED_PAYLOAD = os.environ.get(
    'PROOFTAG_E030_EXPECTED_PAYLOAD', 'https://ptag.io/t/e029'
)
QR_VERIFY_REPETITIONS = 5
QR_TOLERANCE_THRESHOLD = 0.80
SATURATION_THRESHOLD = 0.05

assert QR_VERIFY_REPETITIONS >= 3
for path in [DOWNLOAD_ROOT, OUTPUT_ROOT, SOURCE_CACHE_ROOT, QR_VERIFY_CACHE]:
    path.mkdir(parents=True, exist_ok=True)
print('Mode CPU hors ligne — aucune API de génération, aucun CUDA.')
"""
    ),
    markdown("## 1. Sélection automatique et validation de la source E029 v4"),
    code(
        """override = Path(SOURCE_ARCHIVE).expanduser() if SOURCE_ARCHIVE else None
if override is not None and override.is_dir():
    source_kind = 'directory_override'
    source_root = override.resolve()
    source_reference = source_root
    source_archive_sha256 = None
elif override is not None:
    source_kind = 'archive_override'
    archive_path = discover_e029_archive(DOWNLOAD_ROOT, override)
    source_reference = archive_path
    source_archive_sha256 = sha256_file(archive_path)
    source_root = SOURCE_CACHE_ROOT / source_archive_sha256[:16]
    selective_extract_e029_archive(archive_path, source_root)
else:
    try:
        source_root = discover_e029_export_directory(PERSISTENT_E029_ROOTS)
        source_kind = 'persistent_directory'
        source_reference = source_root
        source_archive_sha256 = None
    except FileNotFoundError:
        source_kind = 'archive_fallback'
        archive_path = discover_e029_archive(DOWNLOAD_ROOT)
        source_reference = archive_path
        source_archive_sha256 = sha256_file(archive_path)
        source_root = SOURCE_CACHE_ROOT / source_archive_sha256[:16]
        selective_extract_e029_archive(archive_path, source_root)

source_evidence_sha256 = e029_export_sha256(source_root)
print('Source E029 sélectionnée :', source_reference)
print('Type de source           :', source_kind)
print('SHA-256 contenu probant  :', source_evidence_sha256)
print('SHA-256 archive transport:', source_archive_sha256 or 'sans archive')
source_audit = validate_e029_export(source_root)
expected_payload_sha256 = hashlib.sha256(EXPECTED_PAYLOAD.encode('utf-8')).hexdigest()
manifest_payload_length = int(
    json.loads((source_root / 'manifest.json').read_text(encoding='utf-8'))['plan'][
        'payload_length'
    ]
)
if (
    source_audit['payload_sha256'] != expected_payload_sha256
    or manifest_payload_length != len(EXPECTED_PAYLOAD)
):
    raise RuntimeError(
        'EXPECTED_PAYLOAD ne correspond pas au payload SHA-256 de l archive E029. '
        'Configurer explicitement PROOFTAG_E030_EXPECTED_PAYLOAD avant tout rescoring.'
    )

scorer_probe = ConservativeQRVerifyScorer(
    repetitions=QR_VERIFY_REPETITIONS,
    cache_dir=QR_VERIFY_CACHE,
)
scorer_identity = {
    'engine_version': scorer_probe.engine_version,
    'scoring_version': scorer_probe.scoring_version,
    'implementation_sha256': scorer_probe.implementation_sha256,
    'repetitions': QR_VERIFY_REPETITIONS,
    'preset_count': scorer_probe.decoder.preset_count,
}
scorer_probe.close()
run_identity = {
    'experiment': EXPERIMENT,
    'source_evidence_sha256': source_evidence_sha256,
    'payload_sha256': expected_payload_sha256,
    'qr_tolerance_threshold': QR_TOLERANCE_THRESHOLD,
    'saturation_threshold': SATURATION_THRESHOLD,
    **scorer_identity,
}
run_id = hashlib.sha256(
    json.dumps(run_identity, sort_keys=True, separators=(',', ':')).encode('utf-8')
).hexdigest()[:16]
RUN_DIR = OUTPUT_ROOT / run_id
RUN_DIR.mkdir(parents=True, exist_ok=True)
(RUN_DIR / 'run-identity.json').write_text(
    json.dumps(run_identity, indent=2), encoding='utf-8'
)

display(pd.DataFrame([{**source_audit, **scorer_identity, 'run_id': run_id}]))
"""
    ),
    markdown("## 2. Inventaire complet et déduplication des rasters"),
    code(
        """manifest = json.loads((source_root / 'manifest.json').read_text(encoding='utf-8'))
state_frame = pd.read_csv(source_root / 'e029-state-results.csv')
gallery_frame = pd.read_csv(source_root / 'e029-gallery' / 'gallery-index.csv')
frame = state_frame.merge(
    gallery_frame[['trial_id', 'image_sha256', 'local_image']],
    on='trial_id', how='inner', validate='one_to_one', suffixes=('', '_gallery'),
)
if len(frame) != source_audit['expected_rows']:
    raise RuntimeError(f"Jointure E029 incomplète : {len(frame)}/{source_audit['expected_rows']}")


def strict_bool(value):
    if value is True or value == 1 or str(value).strip().casefold() in {'true', '1'}:
        return True
    if value is False or value == 0 or str(value).strip().casefold() in {'false', '0'}:
        return False
    raise ValueError(f'Valeur booléenne E029 non reconnue : {value!r}')


frame['fixed_control_bool'] = frame.fixed_control.map(strict_bool)

frame['image_path'] = frame.local_image.map(
    lambda value: str((source_root / 'e029-gallery' / str(value)).resolve())
)
work_frame = (
    frame.sort_values(['image_sha256', 'trial_id'])
    .drop_duplicates('image_sha256')
    [['image_sha256', 'image_path', 'local_image']]
    .reset_index(drop=True)
)
if len(work_frame) != source_audit['unique_rasters']:
    raise RuntimeError('Le nombre de rasters uniques a changé après la jointure.')
if frame.image_path.map(lambda value: Path(value).is_file()).sum() != len(frame):
    raise RuntimeError('Au moins une image E029 est absente après extraction sélective.')

expected_raster_sha256_by_source = {}
for item in work_frame.itertuples(index=False):
    with Image.open(item.image_path) as source_image:
        expected_raster_sha256_by_source[item.image_sha256] = image_raster_sha256(
            source_image
        )

frame.to_csv(RUN_DIR / 'e030-source-inventory.csv', index=False)
print('Générations E029 :', len(frame))
print('Rasters RGB/PNG uniques à rescanner :', len(work_frame))
display(frame.groupby(['pipeline_state', 'fixed_control_bool']).size().unstack(fill_value=0))
"""
    ),
    markdown(
        """## 3. Rescoring QR-Verify répétable, avec reprise

Le journal JSONL n'est écrit qu'après une mesure complète. Une interruption laisse au pire un
fichier temporaire de cache ignoré au redémarrage. La clé du run change automatiquement si le
bridge, ses dépendances, le protocole, le payload, le nombre de répétitions ou l'archive changent.
"""
    ),
    code(
        """JOURNAL_PATH = RUN_DIR / 'e030-rescore-results.jsonl'
PROGRESS_PATH = RUN_DIR / 'e030-rescore-progress.json'


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def journal_rows():
    if not JOURNAL_PATH.exists():
        return []
    rows = []
    for number, line in enumerate(JOURNAL_PATH.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f'Ligne JSONL partielle ignorée : {number}')
            continue
        rows.append(row)
    return validate_rescore_journal_rows(
        rows,
        run_id=run_id,
        expected_raster_sha256_by_source=expected_raster_sha256_by_source,
        payload_sha256=expected_payload_sha256,
        scorer_identity=scorer_identity,
    )


completed = {row['source_png_sha256']: row for row in journal_rows()}
started = time.monotonic()
scorer = ConservativeQRVerifyScorer(
    repetitions=QR_VERIFY_REPETITIONS,
    cache_dir=QR_VERIFY_CACHE,
)
try:
    for position, item in enumerate(work_frame.itertuples(index=False), 1):
        if item.image_sha256 in completed:
            continue
        image = Image.open(item.image_path).convert('RGB')
        try:
            score = scorer.score(image, EXPECTED_PAYLOAD).to_dict()
        except Exception as exc:
            atomic_json(PROGRESS_PATH, {
                'status': 'error', 'position': position, 'total': len(work_frame),
                'source_png_sha256': item.image_sha256,
                'error': f'{type(exc).__name__}: {exc}',
                'updated_at': datetime.now(UTC).isoformat(),
            })
            raise
        record = {
            'run_id': run_id,
            'source_png_sha256': item.image_sha256,
            'local_image': item.local_image,
            'score': score,
        }
        with JOURNAL_PATH.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\\n')
            stream.flush()
            os.fsync(stream.fileno())
        completed[item.image_sha256] = record
        progress = {
            'status': 'running', 'completed': len(completed), 'total': len(work_frame),
            'current': item.local_image, 'cache_hit': score['cache_hit'],
            'elapsed_minutes': round((time.monotonic() - started) / 60, 2),
            'updated_at': datetime.now(UTC).isoformat(),
        }
        atomic_json(PROGRESS_PATH, progress)
        clear_output(wait=True)
        display(Markdown('### Progression E030 — QR-Verify répétable'))
        display(pd.DataFrame([progress]))
finally:
    scorer.close()

if len(completed) != len(work_frame):
    raise RuntimeError(f'Rescoring incomplet : {len(completed)}/{len(work_frame)}')
atomic_json(PROGRESS_PATH, {
    'status': 'completed', 'completed': len(completed), 'total': len(work_frame),
    'updated_at': datetime.now(UTC).isoformat(),
})
print('Rescoring complet :', len(completed), 'rasters uniques')
"""
    ),
    markdown("## 4. Scores conservateurs et stabilité du moteur"),
    code(
        """records = {row['source_png_sha256']: row for row in journal_rows()}
score_rows = []
for source_hash, record in sorted(records.items()):
    score = record['score']
    score_rows.append({
        'source_png_sha256': source_hash,
        'raster_sha256': score['image_sha256'],
        'payload_sha256': score['payload_sha256'],
        'cache_key': score['cache_key'],
        'engine_version': score['engine_version'],
        'scoring_version': score['scoring_version'],
        'implementation_sha256': score['implementation_sha256'],
        'repetitions': score['repetitions'],
        'preset_count': score['preset_count'],
        'direct_exact_all_repetitions': score['direct_exact_all_repetitions'],
        'each_repetition_any_exact': score['each_repetition_any_exact'],
        'consistent_any_exact': score['consistent_any_exact'],
        'conservative_exact_presets': score['conservative_exact_presets'],
        'conservative_tolerance_score': score['conservative_tolerance_score'],
        'minimum_tolerance_score': score['minimum_tolerance_score'],
        'mean_tolerance_score': score['mean_tolerance_score'],
        'maximum_tolerance_score': score['maximum_tolerance_score'],
        'unstable_preset_count': score['unstable_preset_count'],
        'stable_preset_count': score['stable_preset_count'],
        'cache_hit': score['cache_hit'],
    })
score_frame = pd.DataFrame(score_rows)
score_frame.to_csv(RUN_DIR / 'e030-unique-raster-rescore.csv', index=False)


def observations_for(score):
    return [
        {
            'qr_success': float(bool(run['any_exact'])),
            'qr_tolerance': float(run['tolerance_score']),
            'image_sha256': score['image_sha256'],
        }
        for run in score['runs']
    ]


enriched_rows = []
for row in frame.to_dict(orient='records'):
    score = records[row['image_sha256']]['score']
    row['e029_final_image_sha256'] = row.get('final_image_sha256')
    row['final_image_sha256'] = score['image_sha256']
    row['historical_qr_success'] = row.get('qr_success')
    row['historical_qr_tolerance'] = row.get('qr_tolerance')
    row['qr_verify_observations'] = observations_for(score)
    row['qr_verify_observations_json'] = json.dumps(row['qr_verify_observations'])
    row['conservative_qr_success'] = float(score['each_repetition_any_exact'])
    row['conservative_qr_tolerance'] = score['conservative_tolerance_score']
    row['minimum_qr_tolerance'] = score['minimum_tolerance_score']
    row['mean_qr_tolerance'] = score['mean_tolerance_score']
    row['maximum_qr_tolerance'] = score['maximum_tolerance_score']
    row['unstable_preset_count'] = score['unstable_preset_count']
    row['qr_verify_implementation_sha256'] = score['implementation_sha256']
    enriched_rows.append(row)
enriched = pd.DataFrame(enriched_rows)
enriched.drop(columns=['qr_verify_observations']).to_csv(
    RUN_DIR / 'e030-enriched-candidates.csv', index=False
)

stage_summary = enriched.groupby(['pipeline_state', 'fixed_control_bool']).agg(
    rasters=('trial_id', 'count'),
    historical_tolerance=('historical_qr_tolerance', 'mean'),
    conservative_tolerance=('conservative_qr_tolerance', 'mean'),
    payload_exact_every_run=('conservative_qr_success', 'mean'),
    unstable_presets=('unstable_preset_count', 'mean'),
    clip_aesthetic=('clip_aesthetic', 'mean'),
    hpsv2_1=('hpsv2_1', 'mean'),
).reset_index()
stage_summary.to_csv(RUN_DIR / 'e030-stage-summary.csv', index=False)
display(stage_summary)
"""
    ),
    markdown(
        """## 5. Cascade de livraison simulée

La comparaison rejoue uniquement des sorties déjà calculées, dans l'ordre de seeds fixé par le
manifeste E029. Une tentative est comptée seulement si la cascade aurait eu besoin de la générer.
La porte exige cinq payloads exacts, le minimum de tolérance ≥ 0,80 et la saturation ≤ 0,05.
"""
    ),
    code(
        """gate = ConservativeDeliveryGate(
    qr_tolerance_threshold=QR_TOLERANCE_THRESHOLD,
    saturation_threshold=SATURATION_THRESHOLD,
    minimum_qr_observations=QR_VERIFY_REPETITIONS,
)
seed_order = [int(seed) for seed in manifest['plan']['seeds']]
prompt_order = [item['id'] for item in manifest['plan']['prompts']]
stage2 = enriched[enriched.pipeline_state == 'stage2'].copy()
if set(stage2.output_variant.astype(str)) != {'srpg'}:
    raise RuntimeError('E030 refuse toute sortie qui ne respecte pas le contrat Stage 2/SRPG.')
if len(stage2) != len(prompt_order) * len(seed_order) * 2:
    raise RuntimeError('La matrice Stage 2 fixe/advisor × prompt × seed est incomplète.')


def one_candidate(prompt_id, seed, fixed):
    part = stage2[
        (stage2.prompt_id == prompt_id)
        & (stage2.seed.astype(int) == int(seed))
        & (stage2.fixed_control_bool == bool(fixed))
    ]
    if len(part) != 1:
        raise RuntimeError(
            f'Candidat Stage 2 non unique : prompt={prompt_id}, seed={seed}, fixed={fixed}'
        )
    return part.iloc[0].to_dict()


policy_specs = [
    ('fixed_seed1', False, False),
    ('fixed_then_advisor_seed1', True, False),
    ('fixed_seed_retry', False, True),
    ('fixed_advisor_then_seed_retry', True, True),
]
decision_rows = []
for policy_name, use_advisor, retry_seeds in policy_specs:
    active_seeds = seed_order if retry_seeds else seed_order[:1]
    for prompt_id in prompt_order:
        attempts = 0
        selected = None
        final_decision = None
        last_seed = None
        for seed in active_seeds:
            last_seed = seed
            primary = one_candidate(prompt_id, seed, True)
            attempts += 1
            if use_advisor:
                primary_assessment = assess_stage2_candidate(primary, gate)
                alternate = None
                if not primary_assessment.deliverable:
                    alternate = one_candidate(prompt_id, seed, False)
                    attempts += 1
                decision = select_stage2_cascade(primary, alternate, gate=gate)
            else:
                assessment = assess_stage2_candidate(primary, gate)
                decision = select_stage2_cascade(primary, None, gate=gate)
                if not assessment.deliverable:
                    decision = None
            if decision is not None and decision.deliverable:
                selected = dict(decision.selected_candidate)
                final_decision = decision
                break

        selected_row = selected or {}
        selected_score = (
            assess_stage2_candidate(selected_row, gate).qr if selected else None
        )
        decision_rows.append({
            'policy': policy_name,
            'prompt_id': prompt_id,
            'deliverable': bool(selected),
            'generation_count': attempts,
            'selected_role': final_decision.selected_role if final_decision else None,
            'selected_seed': int(selected_row['seed']) if selected else None,
            'selected_method_id': selected_row.get('method_id'),
            'selected_trial_id': selected_row.get('trial_id'),
            'selected_image_path': selected_row.get('image_path'),
            'conservative_qr_success': (
                selected_score.payload_exact_all if selected_score else False
            ),
            'conservative_qr_tolerance': (
                selected_score.minimum_tolerance if selected_score else None
            ),
            'clip_aesthetic': selected_row.get('clip_aesthetic'),
            'clip_score': selected_row.get('clip_score'),
            'hpsv2_1': selected_row.get('hpsv2_1'),
            'saturation_risk': selected_row.get('saturation_risk'),
            'last_seed_considered': last_seed,
            'stage1_was_delivered': False,
            'srmpgd_was_requested': False,
        })

decisions = pd.DataFrame(decision_rows)
if decisions.stage1_was_delivered.any() or decisions.srmpgd_was_requested.any():
    raise RuntimeError('Violation du contrat E030 : Stage 1 livré ou SR-MPGD demandé.')
decisions.to_csv(RUN_DIR / 'e030-policy-decisions.csv', index=False)


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


summary_rows = []
for policy_name, part in decisions.groupby('policy', sort=False):
    successes = int(part.deliverable.sum())
    lower, upper = wilson(successes, len(part))
    winners = part[part.deliverable]
    summary_rows.append({
        'policy': policy_name,
        'delivered': successes,
        'prompts': len(part),
        'delivery_rate': successes / len(part),
        'wilson_95_low': lower,
        'wilson_95_high': upper,
        'mean_generation_count': part.generation_count.mean(),
        'maximum_generation_count': int(part.generation_count.max()),
        'mean_conservative_qr_tolerance': winners.conservative_qr_tolerance.mean(),
        'mean_clip_aesthetic': winners.clip_aesthetic.mean(),
        'mean_clip_score': winners.clip_score.mean(),
        'mean_hpsv2_1': winners.hpsv2_1.mean(),
    })
policy_summary = pd.DataFrame(summary_rows)
policy_summary.to_csv(RUN_DIR / 'e030-policy-summary.csv', index=False)
display(policy_summary)
"""
    ),
    markdown("## 6. Graphes, planche des gagnants et rapport automatique"),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
stage2_plot = enriched[enriched.pipeline_state == 'stage2']
axes[0].scatter(
    stage2_plot.historical_qr_tolerance,
    stage2_plot.conservative_qr_tolerance,
    c=np.where(stage2_plot.fixed_control_bool, '#2563eb', '#f97316'),
    alpha=0.8,
)
axes[0].plot([0, 1], [0, 1], '--', color='black', linewidth=1)
axes[0].axhline(QR_TOLERANCE_THRESHOLD, color='red', linestyle=':')
axes[0].set(xlabel='QR-Verify historique (1 passage)',
            ylabel=f'QR-Verify conservateur ({QR_VERIFY_REPETITIONS} passages)',
            title='Optimisme du score historique')
axes[0].grid(alpha=0.25)

labels = policy_summary.policy.str.replace('_', ' ')
axes[1].bar(labels, policy_summary.delivery_rate, color='#16a34a')
axes[1].errorbar(
    range(len(policy_summary)), policy_summary.delivery_rate,
    yerr=[
        policy_summary.delivery_rate - policy_summary.wilson_95_low,
        policy_summary.wilson_95_high - policy_summary.delivery_rate,
    ], fmt='none', ecolor='black', capsize=5,
)
axes[1].set_ylim(0, 1.05)
axes[1].set(ylabel='Taux de livraison conservateur', title='Politiques Stage 2')
axes[1].tick_params(axis='x', rotation=25)
axes[1].grid(axis='y', alpha=0.25)
fig.tight_layout()
fig.savefig(RUN_DIR / 'e030-score-and-policy.png', dpi=170)
display(fig)

fig, axis = plt.subplots(figsize=(9, 5))
axis.hist(score_frame.unstable_preset_count, bins=range(0, 39), color='#7c3aed')
axis.set(
    xlabel=f'Presets instables sur {QR_VERIFY_REPETITIONS} répétitions',
    ylabel='Rasters uniques', title='Instabilité réelle de QR-Verify par raster',
)
axis.grid(axis='y', alpha=0.25)
fig.tight_layout()
fig.savefig(RUN_DIR / 'e030-qrverify-instability.png', dpi=170)
display(fig)

winner_policy = 'fixed_advisor_then_seed_retry'
winners = decisions[(decisions.policy == winner_policy) & decisions.deliverable].copy()
winner_dir = RUN_DIR / 'winner-images'
winner_dir.mkdir(exist_ok=True)
for stale_winner in winner_dir.glob('*.png'):
    stale_winner.unlink()
columns = 3
rows_count = max(1, math.ceil(len(winners) / columns))
fig, axes = plt.subplots(rows_count, columns, figsize=(13, 4.4 * rows_count))
axes = np.atleast_1d(axes).ravel()
for axis, (_, row) in zip(axes, winners.iterrows(), strict=False):
    image = Image.open(row.selected_image_path).convert('RGB')
    filename = f"{row.prompt_id}-{int(row.selected_seed)}.png"
    image.save(winner_dir / filename)
    axis.imshow(image)
    axis.set_title(
        f"{row.prompt_id} · {row.selected_role} · seed {int(row.selected_seed)}\\n"
        f"QR min {row.conservative_qr_tolerance:.3f} · AES {row.clip_aesthetic:.2f}"
    )
    axis.axis('off')
for axis in axes[len(winners):]:
    axis.axis('off')
fig.tight_layout()
fig.savefig(RUN_DIR / 'e030-cascade-winners.png', dpi=160)
display(fig)

policy_records = json.loads(policy_summary.to_json(orient='records'))
best = max(
    policy_records,
    key=lambda row: (
        row['delivery_rate'], -row['mean_generation_count'],
        row.get('mean_clip_aesthetic') or -999,
    ),
)
fixed = next(row for row in policy_records if row['policy'] == 'fixed_seed1')
instability_rasters = int((score_frame.unstable_preset_count > 0).sum())
report = {
    'experiment': EXPERIMENT,
    'source_reference': str(source_reference),
    'source_kind': source_kind,
    'source_evidence_sha256': source_evidence_sha256,
    'source_archive_sha256': source_archive_sha256,
    'unique_rasters': len(score_frame),
    'qr_verify_repetitions': QR_VERIFY_REPETITIONS,
    'scorer': scorer_identity,
    'gate': {
        'payload_exact_on_every_repetition': True,
        'minimum_qr_tolerance': QR_TOLERANCE_THRESHOLD,
        'maximum_saturation_risk': SATURATION_THRESHOLD,
    },
    'unstable_rasters': instability_rasters,
    'best_policy': best,
    'fixed_seed1': fixed,
    'policies': policy_records,
    'no_gpu_generation': True,
    'stage1_delivery_allowed': False,
    'srmpgd_requested': False,
    'physical_phone_claim': False,
}
(RUN_DIR / 'e030-report.json').write_text(
    json.dumps(report, indent=2), encoding='utf-8'
)
report_markdown = f'''# Rapport E030 — QR-Verify répétable et cascade Stage 2

- Source E029 v4 : `{Path(source_reference).name}` (preuve `{source_evidence_sha256}`),
  type `{source_kind}`.
- Rasters uniques rescannés : **{len(score_frame)}** sur {len(frame)} générations.
- Répétitions QR-Verify : **{QR_VERIFY_REPETITIONS} × 37 presets par raster**.
- Rasters avec au moins un preset instable : **{instability_rasters}/{len(score_frame)}**.
- Porte : payload exact à chaque répétition, tolérance conservatrice ≥
  {QR_TOLERANCE_THRESHOLD:.2f}, saturation ≤ {SATURATION_THRESHOLD:.2f}.
- Politique la plus efficace dans ce holdout logiciel : **{best['policy']}**,
  {int(best['delivered'])}/{int(best['prompts'])} livrables, IC Wilson 95 %
  [{best['wilson_95_low']:.3f}, {best['wilson_95_high']:.3f}],
  {best['mean_generation_count']:.2f} générations Stage 2 simulées en moyenne.
- Baseline fixe, première seed : {int(fixed['delivered'])}/{int(fixed['prompts'])}.
- Stage 1 livré : **jamais**. SR-MPGD demandé : **jamais**.

## Limite

Ce résultat corrige la répétabilité du score logiciel, mais ne constitue pas une preuve de
scannabilité téléphone. La prochaine décision de production doit employer un holdout physique
séparé, sans réoptimiser les paramètres sur ces dix prompts.
'''
(RUN_DIR / 'e030-report.md').write_text(report_markdown, encoding='utf-8')
display(Markdown(report_markdown))
"""
    ),
    markdown("## 7. Provenance, checksums et archive finale"),
    code(
        """def git_commit():
    configured = os.environ.get('PROOFTAG_GIT_COMMIT', '').strip().lower()
    if re.fullmatch(r'[0-9a-f]{40}', configured):
        return configured
    try:
        discovered = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, cwd='/app', stderr=subprocess.DEVNULL
        ).strip().lower()
    except Exception:
        discovered = ''
    if not re.fullmatch(r'[0-9a-f]{40}', discovered):
        raise RuntimeError(
            'PROOFTAG_GIT_COMMIT absent : redéployer le notebook E030 avec le script versionné.'
        )
    return discovered


commit = git_commit()
runtime_image = os.environ.get('PROOFTAG_RUNTIME_IMAGE', '').strip()
if not runtime_image or commit[:12] not in runtime_image:
    raise RuntimeError(
        f'Image runtime non traçable : {runtime_image!r}; commit attendu {commit[:12]}.'
    )
runtime_image_digest = os.environ.get('PROOFTAG_RUNTIME_IMAGE_DIGEST', '').strip().lower()
if (
    len(runtime_image_digest) != 71
    or not runtime_image_digest.startswith('sha256:')
    or any(character not in '0123456789abcdef' for character in runtime_image_digest[7:])
):
    raise RuntimeError(
        'PROOFTAG_RUNTIME_IMAGE_DIGEST absent ou invalide : redéployer l image notebook E030.'
    )
validation_source = Path(inspect.getfile(ConservativeQRVerifyScorer)).resolve()
bridge_source = Path(os.environ['PROOFTAG_QR_QR_VERIFY_BRIDGE']).resolve()
package_lock = bridge_source.with_name('package-lock.json')
notebook_source = Path('/workspace/notebooks/25_e030_reliable_qrverify_cascade.ipynb')
for required in [validation_source, bridge_source, package_lock, notebook_source]:
    if not required.is_file():
        raise FileNotFoundError(f'Fichier de provenance absent : {required}')

artifact_paths = sorted(
    path for path in RUN_DIR.rglob('*')
    if path.is_file()
    and path.name not in {'e030-artifact-manifest.json'}
    and not path.name.endswith('.tmp')
)
artifact_checksums = {
    str(path.relative_to(RUN_DIR)).replace('\\\\', '/'): sha256_file(path)
    for path in artifact_paths
}
prearchive_manifest = {
    'experiment': EXPERIMENT,
    'created_at': datetime.now(UTC).isoformat(),
    'git_commit': commit,
    'runtime_image': runtime_image,
    'runtime_image_digest': runtime_image_digest,
    'source_reference': str(source_reference),
    'source_kind': source_kind,
    'source_evidence_sha256': source_evidence_sha256,
    'source_archive_sha256': source_archive_sha256,
    'source_audit': source_audit,
    'unique_rasters': len(score_frame),
    'qr_verify_repetitions': QR_VERIFY_REPETITIONS,
    'qr_verify_presets_per_repetition': scorer_identity['preset_count'],
    'scorer': {
        **scorer_identity,
        'validation_source': str(validation_source),
        'validation_source_sha256': sha256_file(validation_source),
        'bridge_source': str(bridge_source),
        'bridge_sha256': sha256_file(bridge_source),
        'package_lock_sha256': sha256_file(package_lock),
    },
    'notebook_sha256': sha256_file(notebook_source),
    'no_gpu_generation': True,
    'artifact_checksums': artifact_checksums,
    'archive_checksum_note': (
        'The archive cannot contain its own checksum. The final manifest and .sha256 '
        'sidecar next to the archive bind this payload archive by SHA-256.'
    ),
}
artifact_manifest_path = RUN_DIR / 'e030-artifact-manifest.json'
artifact_manifest_path.write_text(
    json.dumps(prearchive_manifest, indent=2), encoding='utf-8'
)
artifact_manifest_sha256 = sha256_file(artifact_manifest_path)

archive = Path(shutil.make_archive(
    str(RUN_DIR), 'gztar', root_dir=RUN_DIR.parent, base_dir=RUN_DIR.name
))
archive_sha256 = sha256_file(archive)
download_archive = shutil.copy2(archive, DOWNLOAD_ROOT / archive.name)
final_manifest = {
    **prearchive_manifest,
    'artifact_manifest_sha256': artifact_manifest_sha256,
    'archive': {
        'filename': download_archive.name,
        'sha256': archive_sha256,
        'bytes': download_archive.stat().st_size,
    },
}
final_manifest_path = OUTPUT_ROOT / f'{RUN_DIR.name}-e030-final-manifest.json'
final_manifest_path.write_text(
    json.dumps(final_manifest, indent=2), encoding='utf-8'
)
final_manifest_sha256 = sha256_file(final_manifest_path)
persistent_sidecar = OUTPUT_ROOT / f'{archive.name}.sha256'
persistent_sidecar.write_text(f'{archive_sha256}  {archive.name}\\n', encoding='ascii')
manifest_sidecar = final_manifest_path.with_suffix(final_manifest_path.suffix + '.sha256')
manifest_sidecar.write_text(
    f'{final_manifest_sha256}  {final_manifest_path.name}\\n', encoding='ascii'
)
download_manifest = shutil.copy2(
    final_manifest_path, DOWNLOAD_ROOT / final_manifest_path.name
)
archive_sidecar = shutil.copy2(
    persistent_sidecar, DOWNLOAD_ROOT / persistent_sidecar.name
)
download_manifest_sidecar = shutil.copy2(
    manifest_sidecar, DOWNLOAD_ROOT / manifest_sidecar.name
)

print('Archive téléchargeable :', download_archive)
print('Archive persistante     :', archive)
print('Manifest final persistant :', final_manifest_path)
print('Manifest téléchargeable :', download_manifest)
print('SHA-256 archive        :', archive_sha256)
print('SHA-256 manifest final :', final_manifest_sha256)
print('Sidecar archive        :', archive_sidecar)
print('Sidecar manifest       :', download_manifest_sidecar)
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
