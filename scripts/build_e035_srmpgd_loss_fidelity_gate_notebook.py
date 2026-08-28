#!/usr/bin/env python3
"""Build the deterministic E035 monitoring and audit notebook."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "30_e035_srmpgd_loss_fidelity_gate.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
        "prooftag": {
            "experiment": "e035-srmpgd-loss-fidelity-gate-v1",
            "role": "cpu-monitor-and-audit",
            "gpu_runner": "python -m prooftag_qr.e035_loss_fidelity",
        },
    }
    notebook["cells"] = [
        markdown(
            """
# E035 — SR-MPGD loss fidelity gate

Cette expérience compare **une seule variable** sur le même latent Stage 2 immuable :

1. `paper_v3`, contrôle exact de la loss E034 ;
2. `upstream_code_e24ea73`, loss du code public DiffQRCoder avec centre 8×8,
   marges 0,45/0,65 et masque gaussien OpenCV.

Le notebook est volontairement CPU : il vérifie le parent, surveille le Job GPU, lit les
traces et construit l'archive scientifique. Il ne régénère jamais Stage 1 ou Stage 2.
Le verdict de décodage utilise QR-Verify sur **37 presets × 3 répétitions**.
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

from prooftag_qr.e035_parent_artifact import verify_parent_artifact

PARENT_DIR = Path(os.environ.get("E035_PARENT_DIR", "/data/e035-parent-v1"))
RESULTS_DIR = Path(os.environ.get("E035_RESULTS_DIR", "/data/e035-loss-fidelity-gate-v1"))
EXPECTED_PARENT_COMMIT = os.environ.get(
    "E035_EXPECTED_PARENT_COMMIT",
    "",
).strip()
RUNNER_COMMAND = (
    "python -m prooftag_qr.e035_loss_fidelity "
    f"--parent-dir {PARENT_DIR} --output-dir {RESULTS_DIR}"
    + (
        f" --expected-parent-commit {EXPECTED_PARENT_COMMIT}"
        if EXPECTED_PARENT_COMMIT
        else ""
    )
)
print("Parent :", PARENT_DIR)
print("Résultats :", RESULTS_DIR)
print("Commande GPU :", RUNNER_COMMAND)
            """
        ),
        markdown("## 1. Gate parent immuable"),
        code(
            """
expected_parent = {
    "qr_version": 3,
    "qr_mask_pattern": 4,
    "qr_module_size": 20,
    "qr_padding_px": 78,
    "diffqrcoder_revision": "e24ea73ee2e13c7e6e87cb422e8b11784e70ae00",
    "stage1_image_sha256": (
        "ce7066664a9d3fee982841ce30f7fbdf442e4d601818187ed05d0f1301296079"
    ),
    "stage1_file_sha256": (
        "be2ed76a2d4e3157beb3e3165a4041123ecc05b0f21d8be8c728e9f2fd12fb71"
    ),
}
if EXPECTED_PARENT_COMMIT:
    expected_parent["source_commit"] = EXPECTED_PARENT_COMMIT
parent_metadata = verify_parent_artifact(PARENT_DIR, expected=expected_parent)
assert parent_metadata["files"]["latent"]["tensor_sha256"]
assert parent_metadata["files"]["image"]["sha256"]
source = parent_metadata["source"]
method = source.get("source_method_id")
if method == "e033_public_demo_srpg_from_fixed_e034_stage1":
    assert source.get("parent_origin") == "stage2_replayed_from_exact_e034_stage1"
    assert source["stage1_image_sha256"] == (
        "ce7066664a9d3fee982841ce30f7fbdf442e4d601818187ed05d0f1301296079"
    )
    assert source["stage1_file_sha256"] == (
        "be2ed76a2d4e3157beb3e3165a4041123ecc05b0f21d8be8c728e9f2fd12fb71"
    )
    assert source["generation"]["stage1_regenerated"] is False
elif method == "e033_public_demo_srpg_exact_e034_export":
    assert source.get("parent_origin") == "exact_e034_stage2_export"
    assert source["generation"]["stage1_regenerated"] is False
else:
    raise ValueError(f"source_method_id parent non autorisé: {method!r}")
display(Markdown("**Parent vérifié : PASS**"))
display(parent_metadata)
            """
        ),
        markdown(
            """
## 2. Lancement GPU

Le parent doit d'abord être présent et vérifié. La voie préférée consiste à envelopper
le PNG et le latent E034 existants avec `export_e035_parent_artifact.py`. Le latent E034
n'étant pas présent dans l'archive fournie, le fallback contrôlé charge le **PNG Stage 1
exact observé dans E034**, vérifie son hash, exécute uniquement le Stage 2 une fois, puis
fige immédiatement le PNG et le latent :

```bash
bash scripts/deploy-e035-notebook.sh prepare
bash scripts/deploy-e035-notebook.sh capture-parent
bash scripts/deploy-e035-notebook.sh run
```

Le notebook reste disponible sur CPU et lit le PVC partagé. La cellule suivante échoue
explicitement tant que `verdict.json` n'existe pas ; elle ne fabrique aucun résultat fictif.
            """
        ),
        code(
            """
verdict_path = RESULTS_DIR / "verdict.json"
if not verdict_path.is_file():
    raise FileNotFoundError(
        f"verdict.json absent : E035 n'est pas terminé ({verdict_path}). "
        "Depuis SSH : bash scripts/deploy-e035-notebook.sh status"
    )
verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
display(verdict)
            """
        ),
        markdown("## 3. Traces appariées"),
        code(
            """
branch_ids = [
    "e035_paper_srl_control",
    "e035_upstream_code_srl",
]
frames = {}
for branch_id in branch_ids:
    trace_path = RESULTS_DIR / branch_id / "trace.csv"
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    frame = pd.read_csv(trace_path)
    frames[branch_id] = frame
    display(Markdown(f"### {branch_id}"))
    columns = [
        "iteration",
        "paper_srl",
        "upstream_srl",
        "lpips_loss",
        "objective",
        "diag_paper_center_error_count",
        "diag_upstream_margin_active_count",
        "diag_full_module_error_count",
        "diag_upstream_reference_official_loss",
        "diag_upstream_reference_local_loss",
        "diag_upstream_reference_absolute_error",
        "diag_upstream_reference_match",
        "latent_gradient_rms",
        "applied_step_rms",
        "gradient_gate_passed",
    ]
    display(frame[[column for column in columns if column in frame.columns]])
            """
        ),
        markdown("## 4. QR-Verify conservateur et verdict"),
        code(
            """
qr_verify = json.loads(
    (RESULTS_DIR / "qr-verify-evidence.json").read_text(encoding="utf-8")
)
display(qr_verify)
assert verdict.get("production_ready") is False
assert verdict.get("advisor_training_authorized") is False
print("Décision E035 :", verdict["decision"])
            """
        ),
        markdown("## 5. Inspection visuelle"),
        code(
            """
contact_sheet = RESULTS_DIR / "e035-final-contact-sheet.png"
if not contact_sheet.is_file():
    raise FileNotFoundError(contact_sheet)
display(DisplayImage(filename=str(contact_sheet)))
for branch_id in branch_ids:
    maps = sorted((RESULTS_DIR / branch_id / "diagnostic-maps").glob("iteration-004-*.png"))
    display(Markdown(f"### Cartes finales — {branch_id}"))
    for path in maps:
        display(Markdown(f"`{path.name}`"))
        display(DisplayImage(filename=str(path)))
            """
        ),
        markdown("## 6. Archive scientifique déterministe"),
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
    if path.is_file() and path.name not in {"e035-artifact-manifest.json"}:
        manifest.append(
            {
                "path": path.relative_to(RESULTS_DIR).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
manifest_path = RESULTS_DIR / "e035-artifact-manifest.json"
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
archive = RESULTS_DIR.parent / f"{RESULTS_DIR.name}.tar.gz"
with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as handle:
    for path in sorted(RESULTS_DIR.rglob("*")):
        if path.is_file():
            archive_name = (
                f"{RESULTS_DIR.name}/{path.relative_to(RESULTS_DIR)}"
            )
            info = handle.gettarinfo(str(path), arcname=archive_name)
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
        # nbformat generates random cell IDs by default. Stable IDs make the builder
        # byte-deterministic and keep Git diffs meaningful.
        cell["id"] = f"e035-{index:02d}"
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
