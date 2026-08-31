#!/usr/bin/env python3
"""Build the deterministic CPU-only E039 review notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/34_e039_srmpgd_limiter_scanaware.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md("""# E039 — SR-MPGD : diagnostic du verrou + scan-aware

Objectif : dépasser le gagnant E038 (`hybrid r=.150`, 4 updates, γ=1000) sans sortir de la zone esthétique.

E039 compare 10 recettes sur le **même parent figé** et journalise chaque tentative de backtracking pour savoir si le vrai verrou est : rayon latent, LPIPS, core MAE ou non-amélioration de l'objectif.
"""),
    code("""from __future__ import annotations\n\nimport json\nimport os\nfrom pathlib import Path\n\nimport matplotlib.pyplot as plt\nimport pandas as pd\nfrom IPython.display import Image as DisplayImage, Markdown, display\n\nRESULTS_DIR = Path(os.environ.get(\n    \"E039_RESULTS_DIR\",\n    \"/data/e039-srmpgd-limiter-scanaware-v1\",\n))\nprint(\"E039 résultats :\", RESULTS_DIR)\n"""),
    md("## 1. Verdict"),
    code("""verdict_path = RESULTS_DIR / \"verdict.json\"\nif not verdict_path.is_file():\n    raise FileNotFoundError(\n        f\"E039 n'est pas terminé : {verdict_path} absent. \"\n        \"Lancer d'abord bash scripts/run-e039-limiter-scanaware.sh\"\n    )\nverdict = json.loads(verdict_path.read_text(encoding=\"utf-8\"))\ndisplay(Markdown(\"**E039 terminé : résultats disponibles.**\"))\ndisplay(verdict)\n"""),
    md("## 2. Classement complet — E038 contrôle + E039"),
    code("""comparison = pd.read_csv(RESULTS_DIR / \"method-comparison.csv\")\npreferred = [\n    \"method\", \"source\", \"profile\", \"max_iterations\", \"radius\", \"gamma\",\n    \"qr_verify_exact_presets\", \"ssr\", \"original_exact\",\n    \"full_module_error_count\", \"upstream_active_modules\",\n    \"lpips\", \"latent_delta_rms\", \"clip_score\", \"clip_aesthetic\", \"hpsv2_1\",\n    \"visual_guard_pass\", \"dominant_blocker\", \"accepted_updates\", \"rejected_all_iterations\",\n]\ncolumns = [c for c in preferred if c in comparison.columns]\ndisplay(comparison[columns].sort_values(\n    [\"qr_verify_exact_presets\", \"lpips\"],\n    ascending=[False, True],\n    na_position=\"last\",\n).reset_index(drop=True))\n"""),
    md("## 3. Quel garde-fou bloque réellement SR-MPGD ?"),
    code("""blockers = pd.read_csv(RESULTS_DIR / \"blocker-summary.csv\")\ndisplay(blockers.sort_values([\"profile\", \"radius\", \"max_iterations\"]).reset_index(drop=True))\n\ncols = [\n    \"rejected_by_latent_radius\",\n    \"rejected_by_lpips_budget\",\n    \"rejected_by_core_mae_budget\",\n    \"rejected_by_objective_nonincrease\",\n]\nplot_df = blockers.set_index(\"recipe\")[cols]\nax = plot_df.plot(kind=\"bar\", figsize=(16, 6))\nax.set_title(\"E039 — raisons de rejet des candidats SR-MPGD\")\nax.set_ylabel(\"nombre de candidats rejetés\")\nplt.xticks(rotation=60, ha=\"right\")\nplt.tight_layout()\nplt.show()\n"""),
    md("## 4. Toutes les images côte à côte"),
    code("""sheet = RESULTS_DIR / \"e039-all-methods-contact-sheet.png\"\ndisplay(DisplayImage(filename=str(sheet), width=1500))\n"""),
    md("## 5. Chaque recette en grand"),
    code("""e039_rows = comparison[comparison[\"source\"] == \"E039\"].copy()\nfor _, row in e039_rows.sort_values([\"profile\", \"radius\", \"max_iterations\"]).iterrows():\n    method = row[\"method\"]\n    final_path = RESULTS_DIR / method / \"images\" / f\"iteration-{int(row['max_iterations']):03d}.png\"\n    title = (\n        f\"### {method} — SSR={int(row['qr_verify_exact_presets'])}/37 \"\n        f\"MER={int(row['full_module_error_count'])}/841 LPIPS={row['lpips']:.4f} \"\n        f\"safe={row['visual_guard_pass']} blocker={row.get('dominant_blocker')}\"\n    )\n    display(Markdown(title))\n    display(DisplayImage(filename=str(final_path), width=760))\n"""),
    md("## 6. SSR ↔ esthétique"),
    code("""fig, ax = plt.subplots(figsize=(9, 6))\nfor profile, group in e039_rows.groupby(\"profile\"):\n    ax.scatter(group[\"lpips\"], group[\"qr_verify_exact_presets\"], label=profile)\n    for _, row in group.iterrows():\n        ax.annotate(str(int(row[\"max_iterations\"])), (row[\"lpips\"], row[\"qr_verify_exact_presets\"]))\nax.set_xlabel(\"LPIPS vs parent\")\nax.set_ylabel(\"QR-Verify exact presets / 37\")\nax.set_title(\"E039 — SSR en fonction de la dérive perceptuelle\")\nax.legend(title=\"profil\")\nplt.tight_layout()\nplt.show()\n"""),
    md("## 7. Effet du nombre d'updates à rayon 0.150"),
    code("""subset = e039_rows[e039_rows[\"radius\"].round(3) == 0.150]\nfig, ax = plt.subplots(figsize=(9, 6))\nfor profile, group in subset.groupby(\"profile\"):\n    group = group.sort_values(\"max_iterations\")\n    ax.plot(group[\"max_iterations\"], group[\"qr_verify_exact_presets\"], marker=\"o\", label=profile)\nax.set_xlabel(\"updates SR-MPGD\")\nax.set_ylabel(\"QR-Verify exact presets / 37\")\nax.set_title(\"E039 — les updates supplémentaires augmentent-ils encore le SSR ?\")\nax.legend()\nplt.tight_layout()\nplt.show()\n"""),
    md("## 8. Trace détaillée du gagnant"),
    code("""winner = verdict[\"research_winner\"]\ntrace = pd.read_csv(RESULTS_DIR / winner / \"trace.csv\")\npreferred_trace = [\n    \"iteration\", \"upstream_srl\", \"full_module_loss\", \"robust_loss\", \"lpips_loss\",\n    \"objective\", \"full_module_error_count\", \"upstream_active_modules\",\n    \"latent_gradient_rms\", \"raw_step_rms\", \"projected_step_rms\",\n    \"accepted_step_rms\", \"accepted_alpha\", \"latent_delta_rms\",\n    \"acceptance_reason\", \"rejected_trial_count\",\n]\ndisplay(trace[[c for c in preferred_trace if c in trace.columns]])\n\nrejections = pd.read_csv(RESULTS_DIR / winner / \"rejection-log.csv\")\ndisplay(Markdown(\"### Dernières tentatives de backtracking du gagnant\"))\ndisplay(rejections.tail(30))\n"""),
    md("## 9. E038 winner vs E039 winner"),
    code("""control = RESULTS_DIR / \"e038-control.json\"\ncontrol_data = json.loads(control.read_text(encoding=\"utf-8\"))\nwinner_row = e039_rows[e039_rows[\"method\"] == winner].iloc[0].to_dict()\nsummary = pd.DataFrame([\n    {\n        \"method\": \"E038 hybrid r=.150 i4\",\n        \"SSR /37\": control_data.get(\"qr_verify_exact_presets\"),\n        \"MER\": control_data.get(\"full_module_error_count\"),\n        \"LPIPS\": control_data.get(\"lpips\"),\n        \"original_exact\": control_data.get(\"original_exact\"),\n    },\n    {\n        \"method\": winner,\n        \"SSR /37\": winner_row.get(\"qr_verify_exact_presets\"),\n        \"MER\": winner_row.get(\"full_module_error_count\"),\n        \"LPIPS\": winner_row.get(\"lpips\"),\n        \"original_exact\": winner_row.get(\"original_exact\"),\n    },\n])\ndisplay(summary)\n"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
