"""Build the focused E029 SR-MPGD exact-raster recovery notebook."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "23_e028_hierarchical_prompt_advisor.ipynb"
TARGET = ROOT / "notebooks" / "24_e029_srmpgd_exact_raster_recovery.ipynb"


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
notebook = deepcopy(notebook)

for cell in notebook["cells"]:
    source = _source(cell)
    if cell.get("cell_type") == "markdown":
        source = source.replace("E028", "E029")
        source = source.replace(
            "conseiller hiérarchique par prompt",
            "récupération exacte du raster Stage 2",
        )
        source = source.replace(
            "Ce notebook met enfin le modèle de paramètres **dans chaque étage** de la chaîne :",
            "Ce notebook rejoue une campagne appariée courte après la correction SR-MPGD :",
        )
        source = source.replace(
            "2 sources esthétiques/structurelles",
            "une source esthétique/structurelle",
        )
        source = source.replace("2 SRPG par source", "un SRPG par source")
        source = source.replace(
            "30 prompts inconnus × 3 seeds × 13 états = 1 170 images.",
            "10 prompts de reprise × 3 seeds × 6 états = 180 images.",
        )
        source = source.replace("/data/e028-hierarchical", "/data/e029-srmpgd-raster")
        source = source.replace(
            "deux profils Stage 1, deux Stage 2 par Stage 1, puis un",
            "un profil Stage 1, un Stage 2, puis un",
        )
        source = source.replace(
            "treize méthodes",
            "six méthodes",
        )
        source = source.replace(
            "Ces deux datasets relient maintenant",
            "Ces datasets de contrôle relient",
        )
    else:
        source = source.replace(
            "    audit_e028_pairing,\n",
            "    audit_e028_pairing,\n    audit_srmpgd_iteration_zero_raster,\n",
        )
        source = source.replace(
            "EXPERIMENT_NAME = 'e028-hierarchical-prompt-advisor-v1'",
            "EXPERIMENT_NAME = 'e029-srmpgd-exact-raster-recovery-v3'",
        )
        source = source.replace(
            "COLLECTION_PAYLOAD = 'https://ptag.io/t/e028'",
            "COLLECTION_PAYLOAD = 'https://ptag.io/t/e029'",
        )
        source = source.replace("HOLDOUT_PROMPT_COUNT = 30", "HOLDOUT_PROMPT_COUNT = 10")
        source = source.replace("STAGE1_TOP_K = 2", "STAGE1_TOP_K = 1")
        source = source.replace("STAGE2_TOP_K = 2", "STAGE2_TOP_K = 1")
        source = source.replace(
            "OUTPUT_ROOT = Path('/data/e028-hierarchical')",
            "OUTPUT_ROOT = Path('/data/e029-srmpgd-raster')",
        )
        source = source.replace("RUN_E028 = True", "RUN_E029 = True")
        source = source.replace("if RUN_E028 else", "if RUN_E029 else")
        source = source.replace(
            "# 30 prompts inconnus × 3 seeds × 13 états = 1 170 images.",
            "# 10 prompts de reprise × 3 seeds × 6 états = 180 images.",
        )
        source = source.replace(
            "assert plan.public['context_count'] == 90",
            "assert plan.public['context_count'] == HOLDOUT_PROMPT_COUNT * len(HOLDOUT_SEEDS)",
        )
        source = source.replace(
            "assert plan.public['trial_count'] == expected_trials == 1170",
            "assert plan.public['trial_count'] == expected_trials == 180",
        )
        source = source.replace("Progression E028", "Progression E029")
        source = source.replace("e028-state-results.csv", "e029-state-results.csv")
        source = source.replace("e028-pairing-audit.csv", "e029-pairing-audit.csv")
        source = source.replace("e028-policy-decisions.csv", "e029-policy-decisions.csv")
        source = source.replace("e028-policy-report.json", "e029-policy-report.json")
        source = source.replace("e028-policy-scorecard.png", "e029-policy-scorecard.png")
        source = source.replace("e028-gallery", "e029-gallery")
        source = source.replace("e028-stage2", "e029-stage2")
        source = source.replace("e028-srmpgd", "e029-srmpgd")
        source = source.replace(
            "e028-initial-prompt-parameter-advisor.joblib",
            "e029-initial-prompt-parameter-advisor.joblib",
        )
        source = source.replace(
            "download_dir = DOWNLOAD_ROOT / f'e028-",
            "download_dir = DOWNLOAD_ROOT / f'e029-",
        )

        marker = "report = evaluate_e028_policies(\n"
        if marker in source:
            invariant = """iteration_zero_rows = audit_srmpgd_iteration_zero_raster(rows)
iteration_zero = pd.DataFrame(iteration_zero_rows)
iteration_zero.to_csv(
    RUN_DIR / 'e029-srmpgd-iteration-zero-raster-audit.csv', index=False
)
if iteration_zero.empty:
    raise RuntimeError(
        'E029 ne contient aucun SR-MPGD ayant conservé l itération 0 ; '
        'la régression exacte ne peut pas être prouvée.'
    )
invalid_zero = iteration_zero[
    ~iteration_zero.exact.fillna(False).astype(bool)
]
if not invalid_zero.empty:
    display(invalid_zero)
    raise RuntimeError(
        f'{len(invalid_zero)} no-op SR-MPGD ont modifié le raster Stage 2.'
    )
print(
    'No-op SR-MPGD identiques pixel pour pixel au Stage 2 :',
    len(iteration_zero), '/', len(iteration_zero),
)

"""
            source = source.replace(marker, invariant + marker)

    _set_source(cell, source)

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(TARGET)
