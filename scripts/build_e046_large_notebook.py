from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "50_e046_large_advisor_dataset.ipynb"


def source(text: str) -> list[str]:
    return textwrap.dedent(text).strip().splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


CELLS = [
    markdown(
        """
        # E046 large — contrôle du dataset advisor

        Ce notebook est une **vue CPU, déterministe et en lecture seule** de
        `e046-large-advisor-dataset-v1`. Il ne lance ni diffusion, ni scoring, ni entraînement.
        Il peut être relancé pendant une campagne : les tableaux lisent seulement les promotions
        atomiques déjà présentes.

        Il montre la progression, la couverture des 256 prompts et des 13 facteurs DOE, les
        distributions WeChat/CLIP/HPS/Aesthetic, les corrélations descriptives, les fronts de
        Pareto, les taux par famille/masque/ECC, les hard negatives et les rasters des meilleurs
        et pires cas. La sélection de Phase B reste automatique.
        """
    ),
    code(
        """
        from __future__ import annotations

        import json
        import math
        import os
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from PIL import Image as PILImage
        from IPython.display import Markdown, display

        OUTPUT_ROOT = Path(os.environ.get(
            'PROOFTAG_E046_LARGE_OUTPUT_ROOT',
            '/data/e046-large-advisor-dataset-v1',
        ))
        PLAN_ID_OVERRIDE = os.environ.get(
            'PROOFTAG_E046_LARGE_PLAN_ID', ''
        ).strip() or None
        MAX_EXAMPLE_IMAGES = 24
        VALID_PRESET_THRESHOLD = 34


        def read_json(path: Path, default=None):
            if not path.is_file():
                return default
            return json.loads(path.read_text(encoding='utf-8'))


        def read_jsonl(path: Path):
            if not path.is_file():
                return []
            return [
                json.loads(line)
                for line in path.read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]


        def resolve_plan():
            latest = read_json(OUTPUT_ROOT / 'LATEST.json', {}) or {}
            if PLAN_ID_OVERRIDE:
                plan_dir = OUTPUT_ROOT / PLAN_ID_OVERRIDE
            elif latest.get('plan_dir'):
                plan_dir = Path(str(latest['plan_dir']))
            elif latest.get('plan_id'):
                plan_dir = OUTPUT_ROOT / str(latest['plan_id'])
            else:
                return latest, None, {}
            return latest, plan_dir, read_json(plan_dir / 'plan.json', {}) or {}


        latest, PLAN_DIR, plan = resolve_plan()
        if PLAN_DIR is None or not plan:
            display(Markdown(
                f'**Aucun plan E046 large trouvé sous `{OUTPUT_ROOT}`.** '
                'Créer d’abord un plan smoke ; ce notebook ne lance rien.'
            ))
        else:
            print('Plan      :', PLAN_DIR)
            print('Plan ID   :', plan.get('plan_id'))
            print('Profil    :', plan.get('profile'))
            print('Commit    :', plan.get('source_commit'))
            print('Catalogue :', plan.get('catalog_sha256'))
            print('DOE       :', plan.get('doe_sha256'))
        """
    ),
    markdown(
        """
        ## 1. Progression et reprise

        Les compteurs proviennent des marqueurs persistants. Relancer cette cellule ne déclenche
        aucun calcul GPU.
        """
    ),
    code(
        """
        def fallback_status(plan_dir: Path, current_plan: dict):
            candidates = list(current_plan.get('candidates', []))
            generated = sum(
                (
                    plan_dir / 'parents' / str(item['id'])
                    / 'GENERATION_COMPLETE.json'
                ).is_file()
                for item in candidates
            )
            scored = sum(
                (
                    plan_dir / 'parents' / str(item['id'])
                    / 'SCORING_COMPLETE.json'
                ).is_file()
                for item in candidates
            )
            selection = read_json(
                plan_dir / 'selected-refinements.json', {}
            ) or {}
            tasks = list(selection.get('tasks', []))
            refinement_planned = len(tasks) or int(
                current_plan.get('expected_refinement_count_maximum', 0)
            )
            refinement_generated = 0
            refinement_scored = 0
            terminal = 0
            for task in tasks:
                root = (
                    plan_dir / 'refinements' / str(task['candidate_id'])
                    / str(task['srmpgd_recipe_id'])
                )
                refinement_generated += int(
                    (root / 'GENERATION_COMPLETE.json').is_file()
                )
                refinement_scored += int(
                    (root / 'SCORING_COMPLETE.json').is_file()
                )
                terminal += int((root / 'TERMINAL_FAILURE.json').is_file())
            total_units = 2 * (len(candidates) + refinement_planned)
            done_units = (
                generated + scored + refinement_generated
                + refinement_scored + 2 * terminal
            )
            size_bytes = sum(
                path.stat().st_size
                for path in plan_dir.rglob('*')
                if path.is_file()
            )
            return {
                'parents_planned': len(candidates),
                'parents_generated': generated,
                'parents_scored': scored,
                'selection_complete': bool(tasks),
                'srmpgd_planned': refinement_planned,
                'srmpgd_generated': refinement_generated,
                'srmpgd_scored': refinement_scored,
                'terminal_scientific_failures': terminal,
                'percent_completion': round(
                    100 * done_units / total_units, 2
                ) if total_units else 0.0,
                'disk_usage_gib': round(size_bytes / 1024**3, 3),
                'aggregate_complete': (
                    plan_dir / 'COMPLETE.json'
                ).is_file(),
            }


        status_error = None
        if PLAN_DIR is not None and plan:
            try:
                from prooftag_qr.e046_large_campaign import (
                    status as campaign_status,
                )
                live_status = campaign_status(
                    output_root=OUTPUT_ROOT,
                    plan_id=str(plan['plan_id']),
                )
            except Exception as exc:
                status_error = f'{type(exc).__name__}: {exc}'
                live_status = fallback_status(PLAN_DIR, plan)
            display(
                pd.DataFrame([live_status]).T.rename(columns={0: 'valeur'})
            )
            if status_error:
                display(Markdown(
                    f'_Repli sur les marqueurs locaux : `{status_error}`_'
                ))
        else:
            live_status = {}
        """
    ),
    markdown(
        """
        ## 2. Observations disponibles

        Le dataset agrégé est prioritaire. S'il n'existe pas encore, le notebook rassemble les
        comparaisons parentes et SR-MPGD déjà scorées. Une tentative partielle sans marqueur
        `SCORING_COMPLETE.json` ne devient pas une observation.
        """
    ),
    code(
        """
        def load_observations(plan_dir: Path | None):
            if plan_dir is None:
                return [], 'absent'
            final_path = plan_dir / 'dataset/advisor-observations.jsonl'
            final_rows = read_jsonl(final_path)
            if final_rows:
                return final_rows, str(final_path)
            rows = []
            for path in sorted(
                (plan_dir / 'parents').glob('*/scoring/comparison.json')
            ):
                marker = path.parent.parent / 'SCORING_COMPLETE.json'
                if marker.is_file():
                    rows.extend(read_json(path, []) or [])
            for path in sorted(
                (plan_dir / 'refinements').glob(
                    '*/*/scoring/comparison.json'
                )
            ):
                marker = path.parent.parent / 'SCORING_COMPLETE.json'
                if marker.is_file():
                    rows.extend(read_json(path, []) or [])
            return rows, 'promotions partielles'


        rows, observation_source = load_observations(PLAN_DIR)
        frame = pd.DataFrame(rows)
        for column, default in {
            'source_kind': '',
            'variant': '',
            'raw': False,
            'primary_training_observation': False,
            'visual_guard_pass': False,
            'wechat_original_exact': False,
            'delivery_eligible': False,
            'observation_independence_class': '',
            'trajectory_id': None,
            'srmpgd_recipe_id': None,
            'iteration': 0,
        }.items():
            if column not in frame.columns:
                frame[column] = default

        numeric_columns = [
            'wechat_exact_presets', 'wechat_exact_rate',
            'wechat_instability', 'clip_score', 'clip_aesthetic',
            'hpsv2_1', 'module_error_rate',
            'multiobjective_prompt_score', 'qr_mask_pattern', 'iteration',
            'stage1_steps', 'stage1_guidance_scale',
            'stage1_controlnet_scale', 'control_guidance_start',
            'control_guidance_end', 'stage2_strength',
            'stage2_strength_effective', 'stage2_steps',
            'stage2_controlnet_scale', 'stage2_qr_weight',
            'stage2_perceptual_weight', 'gamma', 'latent_radius_rms',
            'srmpgd_max_iterations', 'srmpgd_lpips_weight',
        ]
        for column in numeric_columns:
            if column not in frame.columns:
                frame[column] = np.nan
            else:
                frame[column] = pd.to_numeric(
                    frame[column], errors='coerce'
                )

        if frame.empty:
            display(Markdown(
                '**Aucune observation scorée pour le moment.** '
                'La couverture planifiée reste visible ci-dessous.'
            ))
            parent_frame = frame.copy()
            srmpgd_frame = frame.copy()
        else:
            parent_frame = frame[
                frame['observation_independence_class'].eq(
                    'independent_parent'
                )
            ].copy()
            srmpgd_frame = frame[
                frame['observation_independence_class'].eq(
                    'correlated_srmpgd_checkpoint'
                )
            ].copy()
            display(pd.DataFrame([{
                'source': observation_source,
                'observations': len(frame),
                'parents indépendants': len(parent_frame),
                'checkpoints SR-MPGD corrélés': len(srmpgd_frame),
                'prompts observés': frame.get(
                    'prompt_id', pd.Series(dtype=object)
                ).nunique(),
                'familles observées': frame.get(
                    'prompt_family', pd.Series(dtype=object)
                ).nunique(),
            }]))
        """
    ),
    markdown(
        """
        ## 2 bis. Derniers parents générés en attente de scoring

        Cette galerie lit directement les promotions atomiques de Phase A. Elle permet de voir
        l'état esthétique des derniers Stage 2 dès leur génération, même avant que QR-Verify,
        CLIPScore, HPSv2.1 et CLIP-Aesthetic aient été calculés. Ces images ne sont pas encore des
        observations d'entraînement et aucun score n'est inventé.
        """
    ),
    code(
        """
        def generated_unscored_parents(plan_dir: Path | None, maximum=24):
            if plan_dir is None:
                return []
            previews = []
            for image_path in (plan_dir / 'parents').glob(
                '*/images/stage2-raw.png'
            ):
                parent_dir = image_path.parents[1]
                generation_marker = parent_dir / 'GENERATION_COMPLETE.json'
                promotion_manifest = parent_dir / 'PROMOTION_MANIFEST.json'
                scoring_marker = parent_dir / 'SCORING_COMPLETE.json'
                if (
                    not generation_marker.is_file()
                    or not promotion_manifest.is_file()
                    or scoring_marker.is_file()
                ):
                    continue
                metadata = read_json(
                    parent_dir / 'parent-metadata.json', {}
                ) or {}
                candidate = metadata.get('candidate', {}) or {}
                previews.append({
                    'candidate_id': parent_dir.name,
                    'prompt_id': candidate.get('prompt_id', '?'),
                    'config_id': candidate.get('config_id', '?'),
                    'image_path': image_path,
                    'mtime_ns': generation_marker.stat().st_mtime_ns,
                })
            return sorted(
                previews,
                key=lambda item: (
                    -item['mtime_ns'], item['candidate_id']
                ),
            )[:maximum]


        pending_previews = generated_unscored_parents(
            PLAN_DIR, MAX_EXAMPLE_IMAGES
        )
        if not pending_previews:
            display(Markdown(
                '_Aucun parent généré n’attend actuellement son scoring._'
            ))
        else:
            columns = 4
            rows_count = math.ceil(len(pending_previews) / columns)
            fig, axes = plt.subplots(
                rows_count, columns,
                figsize=(3.8 * columns, 4.2 * rows_count),
                squeeze=False,
            )
            for axis in axes.flat:
                axis.axis('off')
            for axis, item in zip(axes.flat, pending_previews):
                with PILImage.open(item['image_path']) as source_image:
                    axis.imshow(source_image.convert('RGB'))
                axis.set_title(
                    f"{item['prompt_id']} / {item['config_id']}\\n"
                    f"{item['candidate_id']}\\nscoring en attente",
                    fontsize=9,
                )
                axis.axis('off')
            fig.suptitle(
                'Derniers Stage 2 bruts promus — non scorés', fontsize=14
            )
            fig.tight_layout()
            display(fig)
            plt.close(fig)
        """
    ),
    markdown(
        """
        ## 3. Couverture des prompts et des paramètres

        La couverture planifiée est visible même avant le scoring. Les checkpoints SR-MPGD ne
        gonflent jamais le nombre de parents indépendants.
        """
    ),
    code(
        """
        planned = pd.DataFrame(
            plan.get('candidates', [])
        ) if plan else pd.DataFrame()
        recipes = pd.DataFrame(
            plan.get('parent_recipes', [])
        ) if plan else pd.DataFrame()
        if (
            not planned.empty
            and not recipes.empty
            and 'parent_recipe_id' in planned
            and 'id' in recipes
        ):
            planned = planned.merge(
                recipes.rename(columns={'id': 'parent_recipe_id'}),
                on='parent_recipe_id',
                how='left',
                suffixes=('', '_recipe'),
            )

        if planned.empty:
            display(Markdown('_Aucun candidat planifié._'))
        else:
            family_coverage = (
                planned.groupby('prompt_family', sort=True)['prompt_id']
                .nunique().rename('prompts').reset_index()
            )
            config_coverage = (
                planned.groupby('config_id', sort=True)['id']
                .count().rename('parents').reset_index()
            )
            mask_column = (
                'qr_mask_pattern'
                if 'qr_mask_pattern' in planned else None
            )
            ecc_column = (
                'error_correction'
                if 'error_correction' in planned else None
            )

            fig, axes = plt.subplots(2, 2, figsize=(16, 11))
            axes[0, 0].barh(
                family_coverage.prompt_family,
                family_coverage.prompts,
                color='#2563eb',
            )
            axes[0, 0].set(
                title='Prompts planifiés par famille',
                xlabel='prompts uniques',
            )
            axes[0, 0].invert_yaxis()
            axes[0, 1].barh(
                config_coverage.config_id,
                config_coverage.parents,
                color='#7c3aed',
            )
            axes[0, 1].set(
                title='Parents planifiés par configuration',
                xlabel='parents',
            )
            axes[0, 1].invert_yaxis()
            if mask_column:
                mask_counts = planned[mask_column].value_counts().sort_index()
                axes[1, 0].bar(
                    mask_counts.index.astype(str),
                    mask_counts.values,
                    color='#0891b2',
                )
                axes[1, 0].set(
                    title='Couverture des masques',
                    xlabel='masque',
                    ylabel='parents',
                )
            else:
                axes[1, 0].text(
                    0.5, 0.5, 'Masques indisponibles', ha='center'
                )
            if ecc_column:
                ecc_counts = planned[ecc_column].value_counts().sort_index()
                axes[1, 1].bar(
                    ecc_counts.index.astype(str),
                    ecc_counts.values,
                    color='#059669',
                )
                axes[1, 1].set(
                    title='Couverture ECC',
                    xlabel='ECC',
                    ylabel='parents',
                )
            else:
                axes[1, 1].text(
                    0.5, 0.5, 'ECC indisponible', ha='center'
                )
            for axis in axes.flat:
                axis.grid(alpha=0.2, axis='x')
            fig.tight_layout()
            plt.show()

            planned['stage2_strength_effective'] = planned[
                'stage2_strength'
            ].where(
                planned['stage2_initialization'].eq(
                    'paper_stage1_noise'
                )
            )
            factor_specs = [
                ('error_correction', 'ECC'),
                ('qr_mask_pattern', 'Masque QR'),
                ('stage1_steps', 'Pas Stage 1'),
                (
                    'stage1_guidance_scale',
                    'CFG partagé Stage 1/2',
                ),
                ('stage1_controlnet_scale', 'ControlNet Stage 1'),
                (
                    'control_guidance_start',
                    'Début contrôle partagé',
                ),
                (
                    'control_guidance_end',
                    'Fin contrôle partagée',
                ),
                ('stage2_initialization', 'Initialisation Stage 2'),
                (
                    'stage2_strength_effective',
                    'Force Stage 2 effective (papier)',
                ),
                ('stage2_steps', 'Pas Stage 2'),
                ('stage2_controlnet_scale', 'ControlNet Stage 2'),
                ('stage2_qr_weight', 'Poids QR Stage 2'),
                (
                    'stage2_perceptual_weight',
                    'Poids perceptuel Stage 2',
                ),
            ]
            fig, axes = plt.subplots(4, 4, figsize=(18, 15))
            for axis in axes.flat:
                axis.axis('off')
            factor_rows = []
            for axis, (column, label) in zip(axes.flat, factor_specs):
                series = planned[column].dropna()
                axis.axis('on')
                if column in {
                    'error_correction', 'stage2_initialization'
                }:
                    counts = series.astype(str).value_counts().sort_index()
                    axis.bar(
                        counts.index, counts.values, color='#0f766e'
                    )
                    extent = ', '.join(
                        f'{key}:{value}' for key, value in counts.items()
                    )
                else:
                    numeric = pd.to_numeric(series, errors='coerce').dropna()
                    axis.hist(
                        numeric, bins=min(12, max(2, numeric.nunique())),
                        color='#2563eb', alpha=0.82,
                    )
                    extent = (
                        f'{numeric.min():g} .. {numeric.max():g}'
                        if not numeric.empty else 'absent'
                    )
                axis.set_title(label, fontsize=10)
                axis.grid(alpha=0.2, axis='y')
                factor_rows.append({
                    'facteur': label,
                    'colonne': column,
                    'n applicable': int(series.notna().sum()),
                    'valeurs distinctes': int(series.nunique()),
                    'étendue / effectifs': extent,
                })
            fig.suptitle('Couverture effective des 13 facteurs DOE')
            fig.tight_layout()
            plt.show()
            display(pd.DataFrame(factor_rows))
        """
    ),
    markdown(
        """
        ## 4. Distributions des cibles séparées

        WeChat, CLIPScore, HPSv2.1, CLIP-Aesthetic et MER restent des cibles distinctes. Le score
        multiobjectif est un diagnostic supplémentaire, pas une substitution.
        """
    ),
    code(
        """
        if parent_frame.empty:
            display(Markdown(
                '_Les distributions apparaîtront après le scoring des parents '
                'Stage 2 raw._'
            ))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(17, 9))
            metrics = [
                (
                    'wechat_exact_presets', 'WeChat exact /37',
                    np.arange(-0.5, 38.5, 1),
                ),
                ('clip_score', 'CLIPScore', 20),
                ('hpsv2_1', 'HPSv2.1', 20),
                ('clip_aesthetic', 'CLIP-Aesthetic', 20),
                ('module_error_rate', 'MER', 20),
                ('wechat_instability', 'Presets instables', 20),
            ]
            for axis, (column, title, bins) in zip(axes.flat, metrics):
                values = pd.to_numeric(
                    parent_frame.get(column), errors='coerce'
                ).dropna()
                if values.empty:
                    axis.text(
                        0.5, 0.5, 'absent', ha='center', va='center'
                    )
                else:
                    axis.hist(
                        values, bins=bins, color='#2563eb',
                        alpha=0.82, edgecolor='white',
                    )
                    if column == 'wechat_exact_presets':
                        axis.axvline(
                            VALID_PRESET_THRESHOLD,
                            color='#dc2626',
                            linestyle='--',
                            label='gate 34/37',
                        )
                        axis.legend()
                axis.set_title(title)
                axis.grid(alpha=0.2)
            fig.tight_layout()
            plt.show()

            buckets = pd.cut(
                parent_frame['wechat_exact_presets'],
                bins=[-1, 5, 15, 25, 33, 37],
                labels=['0–5', '6–15', '16–25', '26–33', '34–37'],
            ).value_counts(sort=False)
            display(buckets.rename('parents').to_frame())
        """
    ),
    markdown(
        """
        ## 5. Corrélations descriptives

        Ces corrélations décrivent les parents observés ; elles ne prouvent ni causalité, ni
        scannabilité téléphone.
        """
    ),
    code(
        """
        correlation_columns = [
            column for column in [
                'wechat_exact_presets', 'wechat_instability',
                'clip_score', 'hpsv2_1', 'clip_aesthetic',
                'module_error_rate', 'multiobjective_prompt_score',
            ]
            if column in parent_frame.columns
        ]
        if parent_frame.empty or len(correlation_columns) < 2:
            display(Markdown(
                '_Pas encore assez de cibles numériques pour une matrice._'
            ))
        else:
            corr = parent_frame[correlation_columns].corr(
                method='pearson', min_periods=3
            )
            fig, axis = plt.subplots(figsize=(10, 8))
            image = axis.imshow(
                corr, cmap='coolwarm', vmin=-1, vmax=1
            )
            axis.set_xticks(
                range(len(corr.columns)),
                corr.columns,
                rotation=45,
                ha='right',
            )
            axis.set_yticks(range(len(corr.index)), corr.index)
            for row_index in range(len(corr.index)):
                for column_index in range(len(corr.columns)):
                    value = corr.iloc[row_index, column_index]
                    if math.isfinite(value):
                        axis.text(
                            column_index, row_index, f'{value:.2f}',
                            ha='center', va='center', fontsize=8,
                        )
            axis.set_title('Corrélations Pearson — parents indépendants')
            fig.colorbar(image, ax=axis, fraction=0.046)
            fig.tight_layout()
            plt.show()
        """
    ),
    markdown(
        """
        ## 6. Fronts de Pareto et gate final

        Le front exploratoire montre le conflit QR/qualité. Le front livrable ne contient que les
        rasters bruts qui passent la garde visuelle, l'original exact et au moins 34/37. Aucun QR
        invalide ne peut devenir gagnant final.
        """
    ),
    code(
        """
        def pareto_flags(values: np.ndarray):
            flags = np.ones(len(values), dtype=bool)
            for index, point in enumerate(values):
                dominated = (
                    np.all(values >= point, axis=1)
                    & np.any(values > point, axis=1)
                )
                dominated[index] = False
                if dominated.any():
                    flags[index] = False
            return flags


        analysis_frame = parent_frame.copy()
        visual_metrics = [
            metric for metric in (
                'clip_score', 'hpsv2_1', 'clip_aesthetic'
            )
            if metric in analysis_frame.columns
            and analysis_frame[metric].notna().any()
        ]
        if analysis_frame.empty or not visual_metrics:
            display(Markdown(
                '_Fronts indisponibles tant que les scores visuels manquent._'
            ))
        else:
            analysis_frame['visual_proxy_percentile'] = (
                analysis_frame[visual_metrics].rank(pct=True).mean(axis=1)
            )
            exploratory = analysis_frame[
                analysis_frame['raw'].fillna(False).astype(bool)
                & analysis_frame['visual_guard_pass']
                .fillna(False).astype(bool)
            ].dropna(
                subset=['wechat_exact_presets', 'visual_proxy_percentile']
            ).copy()
            deliverable = exploratory[
                exploratory['wechat_original_exact']
                .fillna(False).astype(bool)
                & exploratory['wechat_exact_presets'].ge(
                    VALID_PRESET_THRESHOLD
                )
            ].copy()

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            for axis, subset, title in (
                (
                    axes[0], exploratory,
                    'Front exploratoire sous garde visuelle',
                ),
                (
                    axes[1], deliverable,
                    'Front des candidats livrables',
                ),
            ):
                if subset.empty:
                    axis.text(
                        0.5, 0.5, 'aucun point',
                        ha='center', va='center',
                    )
                else:
                    points = subset[[
                        'wechat_exact_presets',
                        'visual_proxy_percentile',
                    ]].to_numpy(float)
                    front = pareto_flags(points)
                    axis.scatter(
                        points[:, 0], points[:, 1],
                        alpha=0.35, color='#64748b',
                    )
                    axis.scatter(
                        points[front, 0], points[front, 1],
                        color='#dc2626', label='Pareto',
                    )
                    axis.legend()
                axis.axvline(
                    VALID_PRESET_THRESHOLD,
                    color='#16a34a',
                    linestyle='--',
                )
                axis.set(
                    xlabel='WeChat exact /37',
                    ylabel='qualité visuelle — rang moyen',
                    title=title,
                )
                axis.grid(alpha=0.2)
            fig.tight_layout()
            plt.show()
        """
    ),
    markdown(
        """
        ## 7. Taux de succès par famille, masque et ECC
        """
    ),
    code(
        """
        if parent_frame.empty:
            display(Markdown('_Taux indisponibles avant scoring._'))
        else:
            success = (
                parent_frame['raw'].fillna(False).astype(bool)
                & parent_frame['visual_guard_pass'].fillna(False).astype(bool)
                & parent_frame['wechat_original_exact']
                .fillna(False).astype(bool)
                & parent_frame['wechat_exact_presets'].ge(
                    VALID_PRESET_THRESHOLD
                )
            )
            rate_frame = parent_frame.assign(automatic_success=success)
            group_specs = [
                ('prompt_family', 'Famille de prompt'),
                ('qr_mask_pattern', 'Masque QR'),
                ('error_correction', 'ECC'),
            ]
            fig, axes = plt.subplots(1, 3, figsize=(19, 6))
            tables = {}
            for axis, (column, title) in zip(axes, group_specs):
                if column not in rate_frame:
                    axis.text(0.5, 0.5, 'absent', ha='center')
                    continue
                table = (
                    rate_frame.groupby(
                        column, dropna=False, sort=True
                    )['automatic_success']
                    .agg(['sum', 'count', 'mean'])
                    .reset_index()
                )
                tables[column] = table
                axis.barh(
                    table[column].astype(str),
                    table['mean'],
                    color='#16a34a',
                )
                axis.set(
                    xlim=(0, 1), xlabel='taux', title=title
                )
                axis.grid(alpha=0.2, axis='x')
                axis.invert_yaxis()
            fig.tight_layout()
            plt.show()
            for column, table in tables.items():
                display(Markdown(f'**{column}**'))
                display(table)
        """
    ),
    markdown(
        """
        ## 8. Importance descriptive des paramètres

        Le tableau utilise une corrélation de rang. C'est un repérage descriptif avant E047,
        jamais une importance causale et jamais un modèle entraîné.
        """
    ),
    code(
        """
        parameter_specs = [
            (column, label) for column, label in [
                ('stage1_steps', 'pas Stage 1'),
                (
                    'stage1_guidance_scale',
                    'CFG partagé Stage 1/2 upstream',
                ),
                ('stage1_controlnet_scale', 'ControlNet Stage 1'),
                (
                    'control_guidance_start',
                    'début contrôle partagé Stage 1/2 upstream',
                ),
                (
                    'control_guidance_end',
                    'fin contrôle partagée Stage 1/2 upstream',
                ),
                (
                    'stage2_strength_effective',
                    'force Stage 2 effective — paper_stage1_noise',
                ),
                ('stage2_steps', 'pas Stage 2'),
                ('stage2_controlnet_scale', 'ControlNet Stage 2'),
                ('stage2_qr_weight', 'poids QR Stage 2'),
                (
                    'stage2_perceptual_weight',
                    'poids perceptuel Stage 2',
                ),
                ('qr_mask_pattern', 'masque QR'),
            ]
            if column in parent_frame.columns
        ]
        target_columns = [
            column for column in [
                'wechat_exact_presets', 'clip_score', 'hpsv2_1',
                'clip_aesthetic', 'module_error_rate',
            ]
            if column in parent_frame.columns
        ]
        importance_rows = []
        for parameter, parameter_label in parameter_specs:
            for target in target_columns:
                pair = parent_frame[[parameter, target]].apply(
                    pd.to_numeric, errors='coerce'
                ).dropna()
                if (
                    len(pair) < 4
                    or pair[parameter].nunique() < 2
                    or pair[target].nunique() < 2
                ):
                    continue
                ranked = pair.rank(method='average')
                correlation = ranked[parameter].corr(ranked[target])
                if pd.notna(correlation):
                    importance_rows.append({
                        'paramètre': parameter_label,
                        'colonne effective': parameter,
                        'cible': target,
                        'rho de rang': float(correlation),
                        '|rho|': abs(float(correlation)),
                        'n': len(pair),
                    })
        importance_frame = pd.DataFrame(importance_rows)
        if importance_frame.empty:
            display(Markdown(
                '_Pas assez de variation pour calculer les associations._'
            ))
        else:
            display(
                importance_frame.sort_values(
                    ['|rho|', 'paramètre'],
                    ascending=[False, True],
                ).head(40)
            )

        if (
            not parent_frame.empty
            and 'stage2_initialization' in parent_frame.columns
        ):
            initialization_rows = []
            for target in target_columns:
                subset = parent_frame[[
                    'stage2_initialization', target
                ]].copy()
                subset[target] = pd.to_numeric(
                    subset[target], errors='coerce'
                )
                subset = subset.dropna()
                for initialization, group in subset.groupby(
                    'stage2_initialization', sort=True
                ):
                    initialization_rows.append({
                        'facteur catégoriel': 'initialisation Stage 2',
                        'modalité': initialization,
                        'cible': target,
                        'moyenne descriptive': group[target].mean(),
                        'médiane descriptive': group[target].median(),
                        'n': len(group),
                    })
            if initialization_rows:
                display(Markdown(
                    '**Effet descriptif du facteur catégoriel '
                    '`stage2_initialization`**'
                ))
                display(pd.DataFrame(initialization_rows))
        """
    ),
    markdown(
        """
        ## 9. Hard negatives et cas utiles à l'advisor

        Les beaux QR sous 34/37 et les QR robustes mais faibles visuellement restent dans les
        exports. Ils ne sont pas des livraisons ; ils servent à apprendre la frontière.
        """
    ),
    code(
        """
        def compact_table(source_frame: pd.DataFrame, limit=20):
            columns = [
                column for column in [
                    'prompt_id', 'prompt_family', 'candidate_id',
                    'variant', 'wechat_exact_presets',
                    'wechat_original_exact', 'clip_score', 'hpsv2_1',
                    'clip_aesthetic', 'module_error_rate',
                    'visual_guard_pass',
                ]
                if column in source_frame.columns
            ]
            return source_frame[columns].head(limit)


        if (
            analysis_frame.empty
            or 'visual_proxy_percentile' not in analysis_frame
        ):
            display(Markdown('_Hard negatives indisponibles._'))
            aesthetic_failures = pd.DataFrame()
            robust_low_visual = pd.DataFrame()
        else:
            aesthetic_failures = analysis_frame[
                analysis_frame['wechat_exact_presets'].lt(
                    VALID_PRESET_THRESHOLD
                )
            ].sort_values(
                ['visual_proxy_percentile', 'wechat_exact_presets'],
                ascending=[False, True],
            )
            robust_low_visual = analysis_frame[
                analysis_frame['wechat_exact_presets'].ge(
                    VALID_PRESET_THRESHOLD
                )
            ].sort_values(
                ['visual_proxy_percentile', 'wechat_exact_presets'],
                ascending=[True, False],
            )
            display(Markdown(
                '### Esthétiques mais insuffisamment robustes'
            ))
            display(compact_table(aesthetic_failures))
            display(Markdown('### Robustes mais visuellement faibles'))
            display(compact_table(robust_low_visual))
        """
    ),
    markdown(
        """
        ## 10. Planches visuelles — meilleurs, frontières et pires

        Les images sont ouvertes depuis les chemins manifestés. Le nombre affiché est borné par
        `MAX_EXAMPLE_IMAGES` pour garder le notebook léger.
        """
    ),
    code(
        """
        def value_text(row, column, digits=3):
            value = row.get(column)
            try:
                number = float(value)
            except (TypeError, ValueError):
                return '—'
            return (
                f'{number:.{digits}f}'
                if math.isfinite(number) else '—'
            )


        def show_contact_sheet(
            source_frame: pd.DataFrame,
            title: str,
            maximum=12,
            columns=4,
        ):
            selected = source_frame.head(
                min(maximum, MAX_EXAMPLE_IMAGES)
            ).copy()
            available = []
            for _, row in selected.iterrows():
                path = Path(str(row.get('image_path', '')))
                if path.is_file():
                    available.append((row, path))
            if not available:
                display(Markdown(
                    f'**{title}** — aucun raster disponible.'
                ))
                return
            row_count = math.ceil(len(available) / columns)
            fig, axes = plt.subplots(
                row_count,
                columns,
                figsize=(4 * columns, 4.8 * row_count),
            )
            axes = np.atleast_1d(axes).reshape(row_count, columns)
            for axis in axes.flat:
                axis.axis('off')
            for axis, (row, path) in zip(axes.flat, available):
                with PILImage.open(path) as source_image:
                    axis.imshow(source_image.convert('RGB'))
                axis.set_title(
                    f"{row.get('prompt_id', '—')} / "
                    f"{row.get('variant', '—')}\\n"
                    f"QR {int(row.get('wechat_exact_presets') or 0)}/37 · "
                    f"AES {value_text(row, 'clip_aesthetic', 2)} · "
                    f"HPS {value_text(row, 'hpsv2_1', 3)}",
                    fontsize=9,
                )
                axis.axis('off')
            fig.suptitle(title, fontsize=15)
            fig.tight_layout()
            plt.show()


        if parent_frame.empty:
            display(Markdown('_Aucun raster parent scoré à afficher._'))
        else:
            eligible_images = parent_frame[
                parent_frame['delivery_eligible']
                .fillna(False).astype(bool)
            ].copy()
            eligible_images = eligible_images.sort_values(
                ['wechat_exact_presets', 'multiobjective_prompt_score'],
                ascending=[False, False],
                na_position='last',
            )
            worst_images = parent_frame.sort_values(
                ['wechat_exact_presets', 'clip_aesthetic'],
                ascending=[True, True],
                na_position='last',
            )
            show_contact_sheet(
                eligible_images,
                'Meilleurs parents automatiquement livrables',
            )
            show_contact_sheet(
                aesthetic_failures,
                'Hard negatives esthétiques mais QR faibles',
            )
            show_contact_sheet(
                worst_images,
                'Pires cas QR — conservés pour l’apprentissage',
            )
        """
    ),
    markdown(
        """
        ## 11. Trajectoires SR-MPGD corrélées

        Une trajectoire peut fournir plusieurs checkpoints, mais son unité d'indépendance reste
        `trajectory_id`. Au plus quatre trajectoires sont affichées.
        """
    ),
    code(
        """
        if srmpgd_frame.empty:
            display(Markdown(
                '_Aucun checkpoint SR-MPGD scoré pour le moment._'
            ))
        else:
            trajectory_audit = (
                srmpgd_frame.groupby('trajectory_id', dropna=False)
                .agg(
                    checkpoints=('iteration', 'count'),
                    iteration_min=('iteration', 'min'),
                    iteration_max=('iteration', 'max'),
                    parent_id=('parent_id', 'first'),
                    prompt_id=('prompt_id', 'first'),
                    recipe=('srmpgd_recipe_id', 'first'),
                )
                .reset_index()
                .sort_values('trajectory_id')
            )
            display(trajectory_audit)
            for trajectory_id in trajectory_audit.trajectory_id.head(4):
                trajectory = srmpgd_frame[
                    srmpgd_frame.trajectory_id.eq(trajectory_id)
                ].sort_values('iteration')
                show_contact_sheet(
                    trajectory,
                    f'Trajectoire {trajectory_id}',
                    maximum=12,
                    columns=4,
                )
        """
    ),
    markdown(
        """
        ## 12. Contrat de split, provenance et échecs

        Le futur entraînement doit grouper par prompt, payload, génération et trajectoire. Un
        split aléatoire ligne par ligne est interdit.
        """
    ),
    code(
        """
        split_contract = (
            read_json(PLAN_DIR / 'dataset/split-contract.json', {})
            if PLAN_DIR else {}
        )
        if not split_contract and plan:
            split_contract = plan.get('parameter_contract', {})
        display(split_contract)

        audit = {
            'parents indépendants': (
                int(parent_frame['parent_id'].nunique())
                if not parent_frame.empty and 'parent_id' in parent_frame
                else 0
            ),
            'trajectoires SR-MPGD': (
                int(srmpgd_frame['trajectory_id'].nunique())
                if not srmpgd_frame.empty else 0
            ),
            'checkpoints SR-MPGD': len(srmpgd_frame),
            'Stage 1 déclaré final': int(
                frame.get(
                    'stage1_final_eligible', pd.Series(dtype=bool)
                ).fillna(False).astype(bool).sum()
            ),
            'scene_qz déclaré final': int(
                frame.get(
                    'scene_qz_final_eligible', pd.Series(dtype=bool)
                ).fillna(False).astype(bool).sum()
            ),
            'parents primaires Stage 2 raw': (
                int(
                    parent_frame['primary_training_observation']
                    .fillna(False).astype(bool).sum()
                ) if not parent_frame.empty else 0
            ),
            'images sans SHA': (
                int(frame['image_sha256'].isna().sum())
                if not frame.empty and 'image_sha256' in frame else 0
            ),
            'latents sans SHA': (
                int(frame['latent_sha256'].isna().sum())
                if not frame.empty and 'latent_sha256' in frame else 0
            ),
        }
        display(pd.DataFrame([audit]).T.rename(columns={0: 'valeur'}))

        failures = []
        if PLAN_DIR:
            for path in sorted((PLAN_DIR / 'failures').glob('*.json')):
                item = read_json(path, {}) or {}
                item['failure_file'] = str(path)
                failures.append(item)
            for path in sorted(
                (PLAN_DIR / 'refinements').glob(
                    '*/*/TERMINAL_FAILURE.json'
                )
            ):
                item = read_json(path, {}) or {}
                item['failure_file'] = str(path)
                failures.append(item)
        if failures:
            failure_frame = pd.DataFrame(failures).drop_duplicates(
                subset=['failure_file']
            )
            display(failure_frame)
        else:
            display(Markdown('_Aucun échec promu._'))
        """
    ),
    markdown(
        """
        ## 13. Lecture finale

        - Un PASS smoke prouve l'orchestration, les hashes, les huit parents, le scoring complet,
          la conservation des négatifs et l'isolation des groupes.
        - Il ne prouve ni généralisation, ni probabilité de scan téléphone, ni qualité finale d'un
          conseiller.
        - Stage 1 reste une provenance et n'est jamais livré.
        - Les erreurs `scientific_fidelity_mismatch` restent visibles et peuvent être terminales
          localement sans invalider les tâches indépendantes. Toute autre erreur technique doit
          être corrigée avant le pilot.
        - Le pilot ne doit être planifié qu'après `verify` du smoke. Le profil full reste un choix
          opérateur explicite après validation du pilot.

        Documentation : `docs/e046-large-advisor-dataset.md`.
        """
    ),
]


NOTEBOOK = {
    "cells": CELLS,
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


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        NOTEBOOK,
        ensure_ascii=False,
        indent=1,
        sort_keys=False,
    ) + "\n"
    temporary = TARGET.with_suffix(".ipynb.tmp")
    temporary.write_text(serialized, encoding="utf-8", newline="\n")
    os.replace(temporary, TARGET)
    print(TARGET)


if __name__ == "__main__":
    main()
