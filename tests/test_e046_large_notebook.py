import json
import runpy
from pathlib import Path

import IPython.display
import matplotlib
import pytest
from PIL import Image

from prooftag_qr.config import Settings
from prooftag_qr.e046_large_campaign import (
    _runtime_scientific_contract,
    scientific_plan,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/50_e046_large_advisor_dataset.ipynb"
BUILDER = ROOT / "scripts/build_e046_large_notebook.py"


def frozen_runtime_contract() -> dict:
    settings = Settings().model_copy(
        update={
            "base_model_revision": "1" * 40,
            "base_model_config_revision": "2" * 40,
            "controlnet_model_revision": "3" * 40,
            "diffqrcoder_upstream_enabled": True,
            "lab_clip_scoring_enabled": True,
            "lab_hps_scoring_enabled": True,
            "lab_quality_scoring_fail_closed": True,
        }
    )
    return _runtime_scientific_contract(settings)


def notebook_source() -> tuple[dict, str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    return payload, text


def test_e046_large_notebook_is_deterministic_read_only_analysis():
    notebook, text = notebook_source()
    namespace = runpy.run_path(str(BUILDER))

    assert notebook == namespace["NOTEBOOK"]
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 25
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(
                "".join(cell.get("source", [])),
                f"{NOTEBOOK}#cell-{index}",
                "exec",
            )

    for forbidden in (
        "subprocess",
        "kubectl",
        "generate_parent(",
        "generate_refinement(",
        "score_all_parents(",
        "torch.",
        "RUN_FULL",
    ):
        assert forbidden not in text


def test_e046_large_notebook_covers_required_scientific_views():
    _, text = notebook_source()
    normalized = " ".join(text.split())

    for phrase in (
        "Progression et reprise",
        "Derniers parents générés en attente de scoring",
        "Couverture des prompts et des paramètres",
        "Distributions des cibles séparées",
        "Corrélations descriptives",
        "Fronts de Pareto et gate final",
        "Taux de succès par famille, masque et ECC",
        "Importance descriptive des paramètres",
        "Hard negatives",
        "Planches visuelles",
        "Trajectoires SR-MPGD corrélées",
        "Contrat de split, provenance et échecs",
        "advisor-observations.jsonl",
        "SCORING_COMPLETE.json",
        "scientific_fidelity_mismatch",
        "Aucun QR invalide ne peut devenir gagnant final",
    ):
        assert " ".join(phrase.split()) in normalized

    assert "MAX_EXAMPLE_IMAGES = 24" in text
    assert "VALID_PRESET_THRESHOLD = 34" in text
    assert "independent_parent" in text
    assert "correlated_srmpgd_checkpoint" in text
    assert "PROOFTAG_E046_LARGE_PLAN_ID" in text
    assert "stage2_strength_effective" in text
    assert "Couverture effective des 13 facteurs DOE" in text
    assert "initialisation Stage 2" in text


def test_e046_large_notebook_is_available_in_cpu_offline_mode():
    name = NOTEBOOK.name
    remote = (ROOT / "scripts/notebook-remote.ps1").read_text(
        encoding="utf-8"
    )
    server = (ROOT / "scripts/notebook-server.sh").read_text(
        encoding="utf-8"
    )

    assert remote.count(name) >= 2
    assert (
        "25_e030_reliable_qrverify_cascade.ipynb|"
        "50_e046_large_advisor_dataset.ipynb"
    ) in server
    assert "offline_mode=1" in server


def test_e046_large_notebook_executes_on_a_partial_unscored_campaign(
    tmp_path: Path, monkeypatch,
):
    pytest.importorskip("pandas")
    matplotlib.use("Agg")
    plan = scientific_plan(
        profile="smoke",
        source_commit="a" * 40,
        runtime_image="prooftag-qr:test",
        runtime_image_digest="sha256:" + "b" * 64,
        runtime_scientific_contract=frozen_runtime_contract(),
    )
    plan_dir = tmp_path / plan["plan_id"]
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    (tmp_path / "LATEST.json").write_text(
        json.dumps({"plan_id": plan["plan_id"], "plan_dir": str(plan_dir)}),
        encoding="utf-8",
    )
    candidate = plan["candidates"][0]
    parent_dir = plan_dir / "parents" / candidate["id"]
    (parent_dir / "images").mkdir(parents=True)
    Image.new("RGB", (32, 32), "navy").save(
        parent_dir / "images/stage2-raw.png"
    )
    (parent_dir / "GENERATION_COMPLETE.json").write_text(
        json.dumps({"generation_complete": True}), encoding="utf-8"
    )
    (parent_dir / "PROMOTION_MANIFEST.json").write_text(
        json.dumps({"promoted": True}), encoding="utf-8"
    )
    (parent_dir / "parent-metadata.json").write_text(
        json.dumps({"candidate": candidate}), encoding="utf-8"
    )

    monkeypatch.setenv("PROOFTAG_E046_LARGE_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(IPython.display, "display", lambda *args, **kwargs: None)
    notebook, _ = notebook_source()
    namespace: dict = {"__name__": "e046_notebook_test"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{NOTEBOOK}#cell-{index}", "exec"), namespace)

    assert len(namespace["pending_previews"]) == 1
    assert namespace["pending_previews"][0]["prompt_id"] == candidate["prompt_id"]


def test_e046_large_documentation_states_counts_safety_and_operator_flow():
    protocol = (ROOT / "docs/e046-large-advisor-dataset.md").read_text(
        encoding="utf-8"
    )
    operator = (ROOT / "E046_LARGE_DATASET_A_LIRE.txt").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "notebooks/README.md").read_text(encoding="utf-8")
    log = (ROOT / "docs/experiment-log.md").read_text(encoding="utf-8")

    for text in (protocol, operator, log):
        assert "e046-large-advisor-dataset-v1" in text
        assert "/data/e046-controlled-best-generator-v1" in text
        assert "3 072" in text

    for phrase in (
        "16 familles",
        "13 facteurs",
        "Stage 2 raw",
        "34/37",
        "GroupKFold",
        "leave-one-family-out",
        "scientific_fidelity_mismatch",
        "pause-after-current",
        "cancel-current",
        "exactement 8 observations parentes",
        "278,8 heures",
        "26,0 Gio",
    ):
        assert phrase in protocol

    assert NOTEBOOK.name in protocol
    assert NOTEBOOK.name in operator
    assert NOTEBOOK.name in readme
    assert NOTEBOOK.name in log
    assert "deploy-e046-large-dataset.sh deploy" in protocol
    assert "offline-cpu" in protocol
    assert "bash scripts/run-e046-large-dataset.sh plan" in operator
    assert "bash scripts/run-e046-large-dataset.sh run" in operator
    assert "bash scripts/run-e046-large-dataset.sh verify" in operator
