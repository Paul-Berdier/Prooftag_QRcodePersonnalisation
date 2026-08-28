#!/usr/bin/env python3
"""Build the deterministic E037 prospective holdout notebook."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "32_e037_prospective_global_trust_holdout.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "prooftag": {
            "experiment": "e037-prospective-global-trust-mini-holdout-v1",
            "role": "cpu-review-and-audit",
            "gpu_runner": "bash scripts/run-e037-holdout.sh",
        },
    }
    notebook["cells"] = [
        markdown(
            """
# E037 — prospective mini-holdout du gagnant E036

E037 teste **sans retuning** `e036_gamma1000_global_trust` sur 10 scènes et seeds
pré-enregistrés. Le brut de chaque update conserve `gamma=1000`; la trust-region reste
exactement celle gagnante d'E036 (rayon latent 0,050, LPIPS 0,050, core MAE 0,050).

Le notebook est CPU et ne lance aucune génération. Les résultats GPU sont produits par :

```bash
bash scripts/run-e037-holdout.sh
```

Les cellules ci-dessous affichent le verdict, les métriques et **toutes les images parent/final**.
            """
        ),
        code(
            """
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from IPython.display import Image as DisplayImage, Markdown, display

RESULTS_DIR = Path(os.environ.get(
    "E037_RESULTS_DIR",
    "/data/e037-prospective-mini-holdout-v1",
))
VERDICT_PATH = RESULTS_DIR / "verdict.json"
READY = VERDICT_PATH.is_file()
print("Résultats :", RESULTS_DIR)
print("E037 terminé :", READY)
if not READY:
    display(Markdown(
        "**E037 n'est pas encore terminé.** Depuis SSH : "
        "`cd ~/apps/Prooftag_QRcodePersonnalisation && bash scripts/run-e037-holdout.sh`"
    ))
            """
        ),
        markdown("## 1. Verdict prospectif"),
        code(
            """
if READY:
    verdict = json.loads(VERDICT_PATH.read_text(encoding="utf-8"))
    assert verdict["gamma"] == 1000.0
    assert verdict["gamma_preserved"] is True
    assert verdict["case_count"] == 10
    assert verdict["production_ready"] is False
    display(verdict)
else:
    print("Verdict indisponible tant que le Job E037 n'est pas terminé.")
            """
        ),
        markdown("## 2. Tableau des dix cas"),
        code(
            """
if READY:
    summary = pd.read_csv(RESULTS_DIR / "holdout-summary.csv")
    columns = [
        "case_id",
        "seed",
        "parent_qr_verify_exact_presets",
        "final_qr_verify_exact_presets",
        "qr_verify_exact_gain",
        "parent_full_module_error_count",
        "final_full_module_error_count",
        "full_module_error_reduction",
        "final_upstream_active_modules",
        "final_lpips",
        "final_core_mae",
        "final_latent_delta_rms",
        "visual_budget_safe",
    ]
    display(summary[columns])
else:
    print("Résumé indisponible.")
            """
        ),
        markdown("## 3. Contact sheet global — 10 parents vs 10 sorties E037"),
        code(
            """
if READY:
    sheet = RESULTS_DIR / "e037-final-contact-sheet.png"
    if not sheet.is_file():
        raise FileNotFoundError(sheet)
    display(DisplayImage(filename=str(sheet)))
else:
    print("Contact sheet indisponible.")
            """
        ),
        markdown("## 4. Comparaisons visuelles cas par cas"),
        code(
            """
if READY:
    summary = pd.read_csv(RESULTS_DIR / "holdout-summary.csv")
    for row in summary.to_dict(orient="records"):
        case_id = row["case_id"]
        display(Markdown(
            f"### {case_id} — QR exact {int(row['parent_qr_verify_exact_presets'])} → "
            f"{int(row['final_qr_verify_exact_presets'])}; modules faux "
            f"{int(row['parent_full_module_error_count'])} → "
            f"{int(row['final_full_module_error_count'])}"
        ))
        comparison = RESULTS_DIR / "cases" / case_id / "comparison.png"
        if comparison.is_file():
            display(DisplayImage(filename=str(comparison)))
        else:
            print("Image manquante :", comparison)
else:
    print("Comparaisons indisponibles.")
            """
        ),
        markdown("## 5. QR-Verify et décisions par cas"),
        code(
            """
if READY:
    evidence = json.loads(
        (RESULTS_DIR / "qr-verify-evidence.json").read_text(encoding="utf-8")
    )
    summary = pd.read_csv(RESULTS_DIR / "holdout-summary.csv")
    for row in summary.to_dict(orient="records"):
        case_id = row["case_id"]
        parent_key = f"{case_id}__parent"
        final_key = f"{case_id}__final"
        display(Markdown(f"### {case_id}"))
        display({
            "parent": evidence.get(parent_key),
            "final": evidence.get(final_key),
        })
else:
    print("QR-Verify indisponible.")
            """
        ),
        markdown("## 6. Archive scientifique"),
        code(
            """
if READY:
    archive_info = json.loads(
        (RESULTS_DIR / "archive.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (RESULTS_DIR / "e037-artifact-manifest.json").read_text(encoding="utf-8")
    )
    print("Archive :", archive_info["path"])
    print("SHA-256 :", archive_info["sha256"])
    print("Fichiers manifestés :", len(manifest))
else:
    print("Archive indisponible.")
            """
        ),
    ]
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"e037-{index:02d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = nbf.writes(notebook, version=4)
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"{OUTPUT} sha256={digest}")


if __name__ == "__main__":
    build()
