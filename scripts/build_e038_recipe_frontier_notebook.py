#!/usr/bin/env python3
"""Build E038 SR-MPGD SSR/aesthetic frontier analysis notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "33_e038_srmpgd_ssr_aesthetic_frontier.ipynb"


def md(text: str, index: int) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"e038-{index:02d}",
        "metadata": {},
        "source": text.strip().splitlines(True),
    }


def code(text: str, index: int) -> dict:
    return {
        "cell_type": "code",
        "id": f"e038-{index:02d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (text.strip() + "\n").splitlines(True),
    }


cells: list[dict] = []

cells.append(md(
"""
# E038 — SR-MPGD : frontière SSR ↔ esthétique

Objectif : **trouver la recette SR-MPGD qui maximise le SSR QR-Verify sans dégrader visuellement
l'image**, avant tout test de généralisation.

Toutes les nouvelles recettes partent du **même latent parent figé** et gardent **γ=1000**.
La grille compare également les contrôles E035/E036 disponibles sur ce même parent.

Le classement E038 est volontairement lexicographique :

1. garde esthétique PASS ;
2. SSR QR-Verify maximal (`exact_presets / 37`) ;
3. décodage `original` si disponible ;
4. erreurs de modules minimales ;
5. LPIPS minimal.

E033/E034 sont des références numériques historiques mais ne sont pas présentées comme des
comparaisons pixel-à-pixel du parent E038.
""", len(cells)))

cells.append(code(
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image as DisplayImage, Markdown, display
from PIL import Image

RESULTS_DIR = Path(os.environ.get(
    "E038_RESULTS_DIR",
    "/data/e038-srmpgd-ssr-aesthetic-frontier-v1",
))
E035_DIR = Path(os.environ.get(
    "E038_E035_RESULTS_DIR",
    "/data/e035-loss-fidelity-gate-v1",
))
E036_DIR = Path(os.environ.get(
    "E038_E036_RESULTS_DIR",
    "/data/e036-gamma1000-trust-region-v1",
))

print("E038 résultats :", RESULTS_DIR)
print("E035 contrôle  :", E035_DIR)
print("E036 contrôle  :", E036_DIR)
""", len(cells)))

cells.append(md("## 1. Statut", len(cells)))
cells.append(code(
"""
verdict_path = RESULTS_DIR / "verdict.json"
E038_READY = verdict_path.is_file()

if E038_READY:
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    display(Markdown("**E038 terminé : résultats disponibles.**"))
    display(verdict)
else:
    verdict = None
    display(Markdown(
        "**E038 n'est pas encore exécuté.**  "
        "Depuis SSH : `bash scripts/run-e038-recipe-frontier.sh`.  "
        "Cette situation n'est pas une erreur du notebook."
    ))
""", len(cells)))

cells.append(md("## 2. Toutes les méthodes dans un seul tableau", len(cells)))
cells.append(code(
"""
comparison_path = RESULTS_DIR / "method-comparison.csv"
if E038_READY and comparison_path.is_file():
    comparison = pd.read_csv(comparison_path)
    preferred = [
        "method",
        "source",
        "objective_kind",
        "radius",
        "gamma",
        "qr_verify_exact_presets",
        "ssr",
        "original_exact",
        "full_module_error_count",
        "upstream_active_modules",
        "lpips",
        "latent_delta_rms",
        "mean_absolute_change",
        "clipped_pixel_ratio_increase",
        "rgb_clipped_channel_ratio_increase",
        "saturation_mean_increase",
        "clip_score",
        "clip_aesthetic",
        "hpsv2_1",
        "visual_guard_pass",
    ]
    columns = [column for column in preferred if column in comparison.columns]
    display(comparison[columns].sort_values(
        ["qr_verify_exact_presets", "lpips"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True))
else:
    comparison = pd.DataFrame()
    print("Tableau E038 indisponible tant que le Job n'est pas terminé.")
""", len(cells)))

cells.append(md("## 3. Comparaison visuelle globale", len(cells)))
cells.append(code(
"""
sheet = RESULTS_DIR / "e038-all-methods-contact-sheet.png"
if sheet.is_file():
    display(DisplayImage(filename=str(sheet)))
else:
    print("Contact sheet absent pour l'instant :", sheet)
""", len(cells)))

cells.append(md("## 4. Contrôles historiques récents en grand", len(cells)))
cells.append(code(
"""
historical_paths = [
    ("Parent FP32", E036_DIR / "parent-fp32-redecoded.png"),
    ("E035 paper — esthétique forte / SSR faible", E035_DIR / "e035_paper_srl_control/images/iteration-004.png"),
    ("E035 upstream libre — plafond SSR / image agressée", E035_DIR / "e035_upstream_code_srl/images/iteration-004.png"),
    ("E036 global r=.050", E036_DIR / "e036_gamma1000_global_trust/images/iteration-004.png"),
    ("E036 strict r=.025", E036_DIR / "e036_gamma1000_strict_trust/images/iteration-004.png"),
    ("E036 local r=.050", E036_DIR / "e036_gamma1000_local_preserve/images/iteration-004.png"),
]
for label, path in historical_paths:
    if path.is_file():
        display(Markdown(f"### {label}"))
        display(DisplayImage(filename=str(path)))
""", len(cells)))

cells.append(md("## 5. Toutes les nouvelles recettes E038 en grand", len(cells)))
cells.append(code(
"""
if not comparison.empty:
    e038_rows = comparison[comparison["source"] == "E038"].copy()
    e038_rows = e038_rows.sort_values(
        ["qr_verify_exact_presets", "visual_guard_pass", "lpips"],
        ascending=[False, False, True],
    )
    for _, row in e038_rows.iterrows():
        name = row["method"]
        path = RESULTS_DIR / name / "images/iteration-004.png"
        display(Markdown(
            f"### {name}  \n"
            f"**SSR {int(row['qr_verify_exact_presets'])}/37** — "
            f"radius={row['radius']} — LPIPS={row['lpips']:.4f} — "
            f"MER={int(row['full_module_error_count'])}/841 — "
            f"visual_safe={row.get('visual_guard_pass')}"
        ))
        if path.is_file():
            display(DisplayImage(filename=str(path)))
else:
    print("Recettes E038 non disponibles pour l'instant.")
""", len(cells)))

cells.append(md("## 6. Courbe SSR ↔ LPIPS", len(cells)))
cells.append(code(
"""
if not comparison.empty:
    subset = comparison[(comparison["source"] == "E038")].dropna(
        subset=["lpips", "qr_verify_exact_presets"]
    )
    if not subset.empty:
        plt.figure(figsize=(10, 6))
        plt.scatter(subset["lpips"], subset["qr_verify_exact_presets"])
        for _, row in subset.iterrows():
            plt.annotate(
                row["method"].replace("e038_", ""),
                (row["lpips"], row["qr_verify_exact_presets"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        plt.xlabel("LPIPS vs parent (plus bas = mieux préservé)")
        plt.ylabel("SSR QR-Verify exact / 37 (plus haut = mieux)")
        plt.title("E038 — frontière SSR / esthétique")
        plt.grid(alpha=0.25)
        plt.show()
""", len(cells)))

cells.append(md("## 7. Effet du rayon de trust-region", len(cells)))
cells.append(code(
"""
if not comparison.empty:
    radius = comparison[
        (comparison["source"] == "E038")
        & (comparison["objective_kind"] == "upstream")
    ].dropna(subset=["radius"])
    if not radius.empty:
        radius = radius.sort_values("radius")
        display(radius[[
            "method",
            "radius",
            "qr_verify_exact_presets",
            "full_module_error_count",
            "lpips",
            "latent_delta_rms",
            "visual_guard_pass",
        ]])
        plt.figure(figsize=(10, 6))
        plt.plot(radius["radius"], radius["qr_verify_exact_presets"], marker="o")
        plt.xlabel("Rayon latent RMS")
        plt.ylabel("SSR exact / 37")
        plt.title("SRL upstream seule — effet du rayon, γ=1000")
        plt.grid(alpha=0.25)
        plt.show()
""", len(cells)))

cells.append(md("## 8. Traces du gagnant et de ses voisins", len(cells)))
cells.append(code(
"""
if E038_READY and verdict and verdict.get("research_winner"):
    winner = verdict["research_winner"]
    trace_path = RESULTS_DIR / winner / "trace.csv"
    display(Markdown(f"### Gagnant E038 : `{winner}`"))
    if trace_path.is_file():
        trace = pd.read_csv(trace_path)
        columns = [
            "iteration",
            "upstream_srl",
            "full_module_loss",
            "robust_loss",
            "lpips_loss",
            "objective",
            "upstream_active_modules",
            "full_module_error_count",
            "raw_step_rms",
            "projected_step_rms",
            "accepted_step_rms",
            "accepted_alpha",
            "latent_delta_rms",
            "acceptance_reason",
        ]
        display(trace[[column for column in columns if column in trace.columns]])
else:
    print("Pas encore de gagnant E038.")
""", len(cells)))

cells.append(md("## 9. Verdict", len(cells)))
cells.append(code(
"""
if E038_READY:
    display(Markdown(
        f"**Research winner : `{verdict.get('research_winner')}`**  \n"
        f"SSR : **{verdict.get('winner_ssr_exact_presets')}/37**  \n"
        f"Original exact : **{verdict.get('winner_original_exact')}**  \n"
        f"Next action : **{verdict.get('next_action')}**"
    ))
else:
    print("Le verdict sera affiché après le Job E038.")
""", len(cells)))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
        "prooftag": {
            "experiment": "e038-srmpgd-ssr-aesthetic-frontier-v1",
            "role": "cpu-analysis-gallery",
            "gpu_runner": "python -m prooftag_qr.e038_recipe_frontier",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
body = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
OUTPUT.write_text(body, encoding="utf-8", newline="\n")
print(f"{OUTPUT} sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")
