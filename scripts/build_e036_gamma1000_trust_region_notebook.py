#!/usr/bin/env python3
"""Build the E036 CPU comparison notebook deterministically."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "31_e036_gamma1000_trust_region.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "prooftag": {
            "experiment": "e036-gamma1000-perceptual-trust-region-v1",
            "role": "cpu-monitor-compare-images",
            "gpu_runner": "python -m prooftag_qr.e036_trust_region",
        },
    }
    notebook["cells"] = [
        md(
            """
# E036 — γ=1000 perceptual trust-region SR-MPGD

Objectif : **garder γ=1000** et empêcher le gradient upstream de détruire l'image en
projetant/backtrackant le pas dans une région de confiance autour du parent immuable.

Comparaison affichée :

- parent FP32 `D(z0)` ;
- E035 paper ;
- E035 upstream non contraint ;
- E036 global trust-region ;
- E036 strict trust-region ;
- E036 local-preserve trust-region.

Le notebook est CPU et ne lance pas de calcul GPU. Le Job E036 se lance depuis SSH avec
`bash scripts/run-e036-trust-region.sh`.
            """
        ),
        code(
            """
from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path

import pandas as pd
from IPython.display import Image as DisplayImage, Markdown, display

PARENT_DIR = Path(os.environ.get("E036_PARENT_DIR", "/data/e035-parent-v1"))
E035_DIR = Path(os.environ.get("E036_E035_RESULTS_DIR", "/data/e035-loss-fidelity-gate-v1"))
RESULTS_DIR = Path(os.environ.get("E036_RESULTS_DIR", "/data/e036-gamma1000-trust-region-v1"))

print("Parent :", PARENT_DIR)
print("E035   :", E035_DIR)
print("E036   :", RESULTS_DIR)
            """
        ),
        md("## 1. État des artefacts"),
        code(
            """
required_parent = [
    PARENT_DIR / "parent-stage2-metadata.json",
    PARENT_DIR / "parent-stage2-latent.safetensors",
    PARENT_DIR / "parent-stage2.png",
]
for path in required_parent:
    if not path.is_file():
        raise FileNotFoundError(path)

parent_metadata = json.loads((PARENT_DIR / "parent-stage2-metadata.json").read_text(encoding="utf-8"))
print("Parent contract :", parent_metadata["contract_sha256"])

verdict_path = RESULTS_DIR / "verdict.json"
if not verdict_path.is_file():
    raise FileNotFoundError(
        f"E036 n'est pas encore exécuté ({verdict_path}). "
        "Depuis SSH : bash scripts/run-e036-trust-region.sh"
    )
verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
display(verdict)
            """
        ),
        md("## 2. Comparaison métrique"),
        code(
            """
summary = pd.read_csv(RESULTS_DIR / "branch-summary.csv")
columns = [
    "branch",
    "gamma",
    "qr_verify_exact_presets",
    "full_module_error_count",
    "full_module_error_rate",
    "upstream_srl",
    "lpips",
    "core_mae",
    "outside_active_mae",
    "latent_delta_rms",
    "policy_latent_radius_rms",
    "policy_lpips_budget",
    "policy_core_mae_budget",
]
display(summary[[column for column in columns if column in summary.columns]])
            """
        ),
        md("## 3. Contact sheet — comparaison visuelle directe"),
        code(
            """
contact_sheet = RESULTS_DIR / "e036-final-contact-sheet.png"
if not contact_sheet.is_file():
    raise FileNotFoundError(contact_sheet)
display(DisplayImage(filename=str(contact_sheet)))
            """
        ),
        md("## 4. Images individuelles en grand"),
        code(
            """
image_candidates = [
    ("Parent FP32", RESULTS_DIR / "parent-fp32-redecoded.png"),
    ("E035 paper", E035_DIR / "e035_paper_srl_control/images/iteration-004.png"),
    ("E035 upstream non contraint", E035_DIR / "e035_upstream_code_srl/images/iteration-004.png"),
    ("E036 global trust", RESULTS_DIR / "e036_gamma1000_global_trust/images/iteration-004.png"),
    ("E036 strict trust", RESULTS_DIR / "e036_gamma1000_strict_trust/images/iteration-004.png"),
    ("E036 local preserve", RESULTS_DIR / "e036_gamma1000_local_preserve/images/iteration-004.png"),
]

for label, path in image_candidates:
    if path.is_file():
        display(Markdown(f"### {label}"))
        display(DisplayImage(filename=str(path), width=736))
    else:
        print("Image absente (ignorée) :", path)
            """
        ),
        md("## 5. Traces d'optimisation γ=1000"),
        code(
            """
branches = [
    "e036_gamma1000_global_trust",
    "e036_gamma1000_strict_trust",
    "e036_gamma1000_local_preserve",
]
for branch in branches:
    path = RESULTS_DIR / branch / "trace.csv"
    frame = pd.read_csv(path)
    display(Markdown(f"### {branch}"))
    trace_columns = [
        "iteration",
        "upstream_srl",
        "upstream_active_modules",
        "full_module_error_count",
        "lpips_loss",
        "core_mae",
        "outside_active_mae",
        "latent_gradient_rms",
        "raw_step_rms",
        "projected_step_rms",
        "accepted_step_rms",
        "accepted_alpha",
        "latent_delta_rms",
        "acceptance_reason",
    ]
    display(frame[[column for column in trace_columns if column in frame.columns]])
            """
        ),
        md("## 6. QR-Verify conservateur"),
        code(
            """
qr_verify = json.loads((RESULTS_DIR / "qr-verify-evidence.json").read_text(encoding="utf-8"))
qr_rows = []
for name, evidence in qr_verify.items():
    if isinstance(evidence, dict):
        qr_rows.append(
            {
                "image": name,
                "exact_presets": evidence.get("conservative_exact_presets"),
                "repetitions": evidence.get("repetitions"),
                "preset_count": evidence.get("preset_count"),
            }
        )
display(pd.DataFrame(qr_rows))
            """
        ),
        md("## 7. Archive scientifique"),
        code(
            """
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest = []
for path in sorted(RESULTS_DIR.rglob("*")):
    if path.is_file() and path.name != "e036-artifact-manifest.json":
        manifest.append(
            {
                "path": path.relative_to(RESULTS_DIR).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
manifest_path = RESULTS_DIR / "e036-artifact-manifest.json"
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)

archive = RESULTS_DIR.parent / f"{RESULTS_DIR.name}.tar.gz"
with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as handle:
    for path in sorted(RESULTS_DIR.rglob("*")):
        if not path.is_file():
            continue
        arcname = f"{RESULTS_DIR.name}/{path.relative_to(RESULTS_DIR)}"
        info = handle.gettarinfo(str(path), arcname=arcname)
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        info.mtime = 0
        with path.open("rb") as stream:
            handle.addfile(info, stream)

print("Archive :", archive)
print("SHA-256 :", file_sha256(archive))
            """
        ),
    ]
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"e036-{index:02d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = nbf.writes(notebook, version=4)
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"{OUTPUT} sha256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    build()
