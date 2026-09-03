#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/47_e045_foundation_and_resilience.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        r"""
# E045 — Fondation du meilleur générateur QR adaptatif

Ce notebook **ne génère pas** de nouvelles images et **n'entraîne pas** de modèle.
Il vérifie que les données E000–E044, les paramètres et les mécanismes de reprise
sont suffisamment propres avant E046/E047.

Objectifs :

1. relire les 45 expériences sans supprimer les échecs ;
2. inventorier les artefacts réellement présents sous `/data` ;
3. construire un dataset canonique avec provenance ;
4. isoler doublons, labels contradictoires et SR-MPGD no-op ;
5. séparer QR-Verify des scans téléphone ;
6. afficher toutes les dimensions de la future pipeline ;
7. prouver la reprise sûre des générations et entraînements.

Les drapeaux restent obligatoirement :

```text
generation_campaign_authorized = false
advisor_training_authorized = false
phone_surrogate_training_authorized = false
production_ready = false
```
"""
    ),
    code(
        r"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from IPython.display import display, Markdown, Image

OUTPUT_ROOT = Path(os.environ.get(
    "PROOFTAG_E045_OUTPUT_ROOT",
    "/data/e045-foundation-v1",
))
latest = json.loads((OUTPUT_ROOT / "LATEST.json").read_text(encoding="utf-8"))
assert latest["status"] == "complete", latest
R = Path(latest["plan_dir"])
assert (R / "COMPLETE.json").is_file(), R

summary = json.loads((R / "summary.json").read_text(encoding="utf-8"))
complete = json.loads((R / "COMPLETE.json").read_text(encoding="utf-8"))
registry = json.loads((R / "experiment-registry.json").read_text(encoding="utf-8"))
parameters = json.loads((R / "parameter-space.json").read_text(encoding="utf-8"))
data_card = json.loads((R / "data-card.json").read_text(encoding="utf-8"))
recovery = json.loads((R / "recovery-runbook.json").read_text(encoding="utf-8"))
inventory_summary = json.loads((R / "artifact-inventory-summary.json").read_text(encoding="utf-8"))
observation_summary = json.loads((R / "canonical-observations-summary.json").read_text(encoding="utf-8"))
dedup_summary = json.loads((R / "deduplication-summary.json").read_text(encoding="utf-8"))
phone_dir = R / "phone-labels"
phone_pointer_path = OUTPUT_ROOT / "PHONE_LATEST.json"
if phone_pointer_path.is_file():
    phone_pointer = json.loads(phone_pointer_path.read_text(encoding="utf-8"))
    candidate_phone_dir = Path(phone_pointer["import_dir"])
    if (candidate_phone_dir / "phone-label-summary.json").is_file():
        phone_dir = candidate_phone_dir
else:
    phone_pointer = None

phone_summary = json.loads((phone_dir / "phone-label-summary.json").read_text(encoding="utf-8"))
selftest = json.loads((R / "resilience-selftest/selftest-result.json").read_text(encoding="utf-8"))

experiments_df = pd.DataFrame(registry["experiments"])
params_df = pd.DataFrame(parameters["parameters"])
artifacts_df = pd.read_csv(R / "artifact-inventory.csv", low_memory=False)
observations_df = pd.read_csv(R / "canonical-observations.csv", low_memory=False)
duplicates_df = pd.read_csv(R / "duplicate-images.csv", low_memory=False)
conflicts_df = pd.read_csv(R / "label-conflicts.csv", low_memory=False)
noops_df = pd.read_csv(R / "srmpgd-noop-images.csv", low_memory=False)
phone_images_df = pd.read_csv(phone_dir / "phone-labels-by-image.csv", low_memory=False)
phone_devices_df = pd.read_csv(phone_dir / "phone-labels-by-device.csv", low_memory=False)

print("Plan E045 :", R)
print("Commit    :", summary["source_commit"])
print("Artefacts :", inventory_summary["artifact_count"])
print("Observations :", observation_summary["canonical_observation_count"])
print("Selftest reprise :", selftest["passed"])
"""
    ),
    md("## 1. Verdict exécutif"),
    code(
        r"""
executive = pd.DataFrame([
    ["Expériences E000–E044", summary["registry"]["experiment_count"]],
    ["Paramètres canoniques", summary["registry"]["parameter_count"]],
    ["Artefacts indexés", summary["inventory"]["artifact_count"]],
    ["Images indexées", summary["inventory"]["image_count"]],
    ["Observations canoniques", summary["observations"]["canonical_observation_count"]],
    ["Candidates conseiller", summary["observations"]["eligible_parameter_advisor_count"]],
    ["Labels téléphone entraînables", summary["observations"]["eligible_phone_model_count"]],
    ["Groupes de doublons", summary["deduplication"]["duplicate_group_count"]],
    ["Conflits de labels", summary["deduplication"]["conflicting_label_group_count"]],
], columns=["Indicateur", "Valeur"])
display(executive)

flags = {
    key: complete[key]
    for key in (
        "physical_truth_available",
        "advisor_training_authorized",
        "phone_surrogate_training_authorized",
        "generation_campaign_authorized",
        "production_ready",
        "next_action",
    )
}
display(flags)
"""
    ),
    md(
        r"""
### Lecture du verdict

E045 peut être techniquement terminé tout en refusant E046/E047. C'est volontaire :
un inventaire complet ne remplace ni les captures téléphone, ni la revue de la
quarantaine, ni le gel d'un nouveau holdout.
"""
    ),
    md("## 2. E000–E044, une ligne par expérience"),
    code(
        r"""
display(experiments_df[[
    "id", "title", "decision", "training_policy", "observed"
]].style.set_properties(subset=["observed"], **{"white-space": "normal"}))
"""
    ),
    code(
        r"""
decision_counts = experiments_df["decision"].value_counts().sort_values()
plt.figure(figsize=(12, 8))
plt.barh(decision_counts.index, decision_counts.values)
plt.xlabel("Nombre d'expériences")
plt.ylabel("Décision")
plt.title("E045 — décisions tirées des expériences E000–E044")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
policy_counts = experiments_df["training_policy"].value_counts().sort_values()
plt.figure(figsize=(10, 6))
plt.barh(policy_counts.index, policy_counts.values)
plt.xlabel("Nombre d'expériences")
plt.ylabel("Politique d'utilisation")
plt.title("Quelles expériences peuvent alimenter quel apprentissage ?")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
### Règles importantes

- `evaluation_only` : les observations restent visibles, mais ne rejoignent pas le train ;
- `hard_negative_only` : utilisables pour apprendre à éviter un défaut, jamais comme
  positifs esthétiques ;
- `quarantine` : aucun entraînement tant que les collisions/labels ne sont pas corrigés ;
- `training_candidate_software_only` : QR-Verify peut être appris, mais pas le téléphone ;
- un résultat SR-MPGD identique au parent est un **no-op**, donc une sortie Stage 2.
"""
    ),
    md("## 3. Les 98 paramètres de toute la pipeline"),
    code(
        r"""
stage_counts = params_df.groupby("stage").size().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
plt.bar(stage_counts.index, stage_counts.values)
plt.ylabel("Nombre de paramètres")
plt.xlabel("Étape")
plt.title("Espace canonique E045 par étape")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

display(params_df[[
    "stage", "name", "dtype", "role", "domain", "conditional_on", "notes"
]].style.set_properties(subset=["domain", "conditional_on", "notes"], **{"white-space": "normal"}))
"""
    ),
    md(
        r"""
### Ce qui change par rapport à E026

E026 conseillait déjà plusieurs paramètres, mais E045 prépare une politique plus large :

```text
encodage QR + matrice
→ reformulation du prompt
→ seed et Stage 1
→ critique de l'image Stage 1
→ Stage 2 conditionnelle
→ critique Stage 2
→ SR-MPGD conditionnel
→ checkpoint Pareto
→ téléphone
```

L'espace n'est **jamais** évalué par produit cartésien. E046 devra utiliser des
fidélités successives et une optimisation bayésienne multiobjectif contrainte.
"""
    ),
    md("## 4. Inventaire réel des artefacts"),
    code(
        r"""
ext_counts = artifacts_df["extension"].fillna("sans extension").value_counts().head(20)
plt.figure(figsize=(11, 5))
plt.bar(ext_counts.index.astype(str), ext_counts.values)
plt.ylabel("Fichiers")
plt.xlabel("Extension")
plt.title("Artefacts E000–E044 présents sur le PVC")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
experiment_artifacts = (
    artifacts_df.groupby("experiment_id")
    .size()
    .sort_values(ascending=False)
    .head(30)
)
plt.figure(figsize=(12, 6))
plt.bar(experiment_artifacts.index.astype(str), experiment_artifacts.values)
plt.ylabel("Artefacts")
plt.xlabel("Expérience")
plt.title("Volume d'artefacts par expérience")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()

display(artifacts_df[artifacts_df["hash_status"] == "error"][
    ["experiment_id", "relative_path", "error"]
].head(100))
"""
    ),
    md("## 5. Dataset canonique et couverture"),
    code(
        r"""
if observations_df.empty:
    display(Markdown("**Aucune observation structurée n'a été extraite.**"))
else:
    counts = observations_df.groupby("experiment_id").size().sort_values(ascending=False).head(30)
    plt.figure(figsize=(12, 6))
    plt.bar(counts.index.astype(str), counts.values)
    plt.ylabel("Observations")
    plt.xlabel("Expérience")
    plt.title("Observations canoniques extraites")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.show()
"""
    ),
    code(
        r"""
eligibility_columns = [
    "eligible_parameter_advisor",
    "eligible_phone_model",
    "eligible_hard_negative",
    "evaluation_only",
]
if not observations_df.empty:
    eligibility = observations_df[eligibility_columns].fillna(0).sum().sort_values()
    plt.figure(figsize=(9, 4))
    plt.barh(eligibility.index, eligibility.values)
    plt.xlabel("Observations")
    plt.title("Éligibilité après registre et contrôles")
    plt.tight_layout()
    plt.show()
"""
    ),
    code(
        r"""
important = [
    "prompt_text", "effective_config_hash", "image_sha256",
    "qr_score", "qr_exact_presets", "original_exact",
    "clip_aesthetic", "clip_score", "hpsv2",
    "phone_attempts", "phone_successes",
]
if not observations_df.empty:
    missing = observations_df[important].isna().mean().sort_values()
    plt.figure(figsize=(10, 5))
    plt.barh(missing.index, missing.values)
    plt.xlabel("Fraction manquante")
    plt.xlim(0, 1)
    plt.title("Champs manquants dans l'historique canonique")
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## 6. QR logiciel, esthétique et divergences"),
    code(
        r"""
plot_df = observations_df.copy()
for column in ("qr_score", "clip_aesthetic", "clip_score", "hpsv2", "lpips"):
    if column in plot_df:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")

paired = plot_df.dropna(subset=["qr_score", "clip_aesthetic"]) if not plot_df.empty else plot_df
if not paired.empty:
    plt.figure(figsize=(9, 6))
    plt.scatter(paired["clip_aesthetic"], paired["qr_score"], alpha=0.45)
    plt.xlabel("CLIP-Aesthetic")
    plt.ylabel("Score QR logiciel")
    plt.title("Le compromis QR/esthétique n'est pas une relation monotone")
    plt.tight_layout()
    plt.show()
else:
    display(Markdown("Pas assez de lignes appariées QR/CLIP-Aesthetic."))
"""
    ),
    code(
        r"""
if not plot_df.empty and plot_df["qr_score"].notna().any():
    top = (
        plot_df.dropna(subset=["qr_score"])
        .sort_values(["qr_score", "clip_aesthetic"], ascending=[False, False])
        .head(100)
    )
    display(top[[
        "experiment_id", "stage", "method_id", "prompt_family",
        "qr_score", "qr_exact_presets", "original_exact",
        "clip_aesthetic", "clip_score", "hpsv2", "image_path",
        "training_policy",
    ]])
"""
    ),
    md("## 7. Doublons, conflits et SR-MPGD sans effet"),
    code(
        r"""
quality_counts = pd.Series({
    "Groupes de doublons": dedup_summary["duplicate_group_count"],
    "Conflits de labels": dedup_summary["conflicting_label_group_count"],
    "No-op SR-MPGD": dedup_summary["srmpgd_noop_group_count"],
})
plt.figure(figsize=(8, 4))
plt.bar(quality_counts.index, quality_counts.values)
plt.ylabel("Groupes")
plt.title("Problèmes de qualité détectés")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

display(Markdown("### Premiers doublons"))
display(duplicates_df.head(100))
display(Markdown("### Conflits de labels — à résoudre avant train"))
display(conflicts_df.head(100))
display(Markdown("### SR-MPGD pixel-identique à un autre état"))
display(noops_df.head(100))
"""
    ),
    md(
        r"""
Un hash identique portant des scores différents peut provenir de :

- variabilité interne à QR-Verify ;
- répétitions agrégées différemment ;
- alias de méthodes ;
- ancien bug de reporting ;
- labels téléphone réalisés dans des conditions différentes.

E045 n'écrase aucune valeur : il bloque le train jusqu'à résolution.
"""
    ),
    md("## 8. Galerie des images réellement disponibles"),
    code(
        r"""
def image_candidates(frame: pd.DataFrame, limit: int = 36):
    if frame.empty or "path" not in frame:
        return []
    images = frame[
        frame["extension"].isin([".png", ".jpg", ".jpeg", ".webp"])
        & frame["path"].notna()
    ].copy()
    # Priorité aux expériences récentes et aux fichiers final/winner/pipeline.
    images["priority"] = images["relative_path"].astype(str).str.lower().apply(
        lambda text: (
            0 if "99-final" in text or "winner" in text else
            1 if "pipeline" in text or "contact-sheet" in text else
            2
        )
    )
    images = images.sort_values(["priority", "experiment_id", "relative_path"])
    output = []
    seen = set()
    for _, row in images.iterrows():
        path = Path(str(row["path"]))
        pixel_hash = str(row.get("pixel_sha256") or path)
        if pixel_hash in seen or not path.is_file():
            continue
        seen.add(pixel_hash)
        output.append((str(row["experiment_id"]), path.name, path))
        if len(output) >= limit:
            break
    return output

gallery = image_candidates(artifacts_df, limit=36)
if not gallery:
    display(Markdown("Aucune image accessible dans l'inventaire."))
else:
    columns = 4
    rows_count = math.ceil(len(gallery) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(16, rows_count * 4))
    axes = np.array(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, (experiment, name, path) in zip(axes, gallery):
        try:
            image = PILImage.open(path).convert("RGB")
            axis.imshow(image)
            axis.set_title(f"{experiment} — {name[:35]}", fontsize=9)
        except Exception as exc:
            axis.text(0.5, 0.5, str(exc), ha="center", va="center")
        axis.axis("off")
    plt.suptitle("Échantillon d'artefacts visuels indexés", y=1.0)
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## 9. Focus E044 : prompt, gamma et téléphone"),
    code(
        r"""
e44 = observations_df[observations_df["experiment_id"] == "E044"].copy()
if e44.empty:
    display(Markdown("Aucune ligne E044 détectée dans le dataset canonique."))
else:
    for column in ("qr_score", "clip_aesthetic", "lpips"):
        e44[column] = pd.to_numeric(e44[column], errors="coerce")
    best = e44.sort_values(["qr_score", "clip_aesthetic"], ascending=[False, False]).head(50)
    display(best[[
        "prompt_family", "method_id", "seed", "qr_score",
        "qr_exact_presets", "original_exact", "clip_aesthetic",
        "lpips", "image_path"
    ]])
    display(Markdown(
        "**Attention :** les résultats E044 restent des labels logiciels. "
        "Le retour utilisateur actuel est que les images testées ne fonctionnent "
        "pas au téléphone."
    ))
"""
    ),
    md("## 10. Labels téléphone physiques"),
    code(
        r"""
display(phone_summary)
if not phone_images_df.empty and "attempts" in phone_images_df:
    display(phone_images_df.sort_values(
        ["all_devices_pass_2_of_3", "minimum_device_success_rate"],
        ascending=[False, False],
    ))
    plt.figure(figsize=(9, 5))
    plt.hist(pd.to_numeric(phone_images_df["success_rate"], errors="coerce").dropna(), bins=10)
    plt.xlabel("Taux de scan physique")
    plt.ylabel("Images")
    plt.title("Distribution des labels téléphone")
    plt.tight_layout()
    plt.show()
else:
    display(Markdown(
        "Aucune capture téléphone valide n'est encore importée. "
        f"Remplir `{R / 'inputs/phone-captures.csv'}` ou "
        "`/data/e045-phone-captures.csv`, puis utiliser "
        "`python scripts/e045-import-phone-captures.py ...`. "
        "L'import crée un dossier immuable et PHONE_LATEST.json sans modifier le plan terminé."
    ))
"""
    ),
    md(
        r"""
### Protocole minimum par image

```text
au moins 3 essais
payload exact
même appareil et même application documentés
distance / angle / luminosité conservés
2 succès sur 3 pour le premier label binaire
```

Pour le surrogate multi-appareils, les essais doivent ensuite varier
systématiquement distance, angle, éclairage et scanner.
"""
    ),
    md("## 11. Reprise : démonstration exécutée"),
    code(
        r"""
display({
    "passed": selftest["passed"],
    "transient_completed_on_attempt": selftest["transient_completed_on_attempt"],
    "oom_kind": selftest["oom_kind"],
    "oom_retryable": selftest["oom_retryable"],
    "stale_recovered_tasks": selftest["stale_recovered_tasks"],
    "invalid_promotion_rejected": selftest["invalid_promotion_rejected"],
})

task_rows = pd.DataFrame(selftest["task_summary"]["tasks"])
display(task_rows[[
    "task_id", "kind", "status", "attempt_count", "max_attempts",
    "last_error_kind", "last_error_class", "last_error_message"
]])
"""
    ),
    code(
        r"""
states = ["pending", "running", "retry_wait", "succeeded", "failed", "blocked"]
x = np.arange(len(states))
plt.figure(figsize=(12, 3))
plt.scatter(x, np.zeros_like(x), s=120)
for index, state in enumerate(states):
    plt.text(index, 0.08, state, ha="center", va="bottom")
for left, right in zip(x[:-1], x[1:]):
    plt.arrow(left + 0.12, 0, right - left - 0.24, 0,
              head_width=0.03, head_length=0.08, length_includes_head=True)
plt.ylim(-0.2, 0.35)
plt.yticks([])
plt.xticks([])
plt.title("Machine d'états persistante — les échecs ne disparaissent pas")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
### Comportement attendu en cas de casse

| Incident | Même tâche relancée ? | Action |
|---|---:|---|
| timeout / 503 / pod évincé | oui, budget borné | reprise depuis dernier checkpoint |
| CUDA OOM | **non** | nouveau batch/précision/spec_hash |
| disque plein / quota | **non** | libérer/agrandir puis nouvelle tentative explicite |
| parent ou latent absent | **non** | corriger la lignée |
| checksum/payload mismatch | **non** | mettre en quarantaine |
| worker disparu | oui | lease expiré → pending |
| résultat incomplet | non publié | reste dans `attempts/` |
"""
    ),
    md("## 12. Architecture cible E046–E049"),
    code(
        r"""
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis("off")

nodes = [
    (0.05, 0.72, 0.16, 0.14, "Intention\nutilisateur"),
    (0.27, 0.72, 0.16, 0.14, "Prompt rewrites\n+ QR matrices"),
    (0.49, 0.72, 0.16, 0.14, "Stage 1\nmulti-seed"),
    (0.71, 0.72, 0.16, 0.14, "Critique\nStage 1"),
    (0.71, 0.44, 0.16, 0.14, "Stage 2\nadaptative"),
    (0.49, 0.44, 0.16, 0.14, "Critique\nStage 2"),
    (0.27, 0.44, 0.16, 0.14, "SR-MPGD\nconditionnel"),
    (0.05, 0.44, 0.16, 0.14, "Pareto\nscannable"),
    (0.27, 0.16, 0.16, 0.14, "QR-Verify\nrépété"),
    (0.49, 0.16, 0.16, 0.14, "Banc téléphone\nmulti-appareils"),
    (0.71, 0.16, 0.16, 0.14, "Surrogate +\nactive learning"),
]
for x0, y0, width, height, label in nodes:
    rect = plt.Rectangle((x0, y0), width, height, fill=False, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x0 + width / 2, y0 + height / 2, label, ha="center", va="center")

arrows = [
    ((0.21, 0.79), (0.27, 0.79)),
    ((0.43, 0.79), (0.49, 0.79)),
    ((0.65, 0.79), (0.71, 0.79)),
    ((0.79, 0.72), (0.79, 0.58)),
    ((0.71, 0.51), (0.65, 0.51)),
    ((0.49, 0.51), (0.43, 0.51)),
    ((0.27, 0.51), (0.21, 0.51)),
    ((0.13, 0.44), (0.27, 0.30)),
    ((0.43, 0.23), (0.49, 0.23)),
    ((0.65, 0.23), (0.71, 0.23)),
]
for (x1, y1), (x2, y2) in arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops={"arrowstyle": "->", "lw": 1.5})

ax.set_xlim(0, 0.95)
ax.set_ylim(0, 1)
ax.set_title("Générateur cible : portefeuille hiérarchique, pas recette fixe", fontsize=16)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
### Séquence gelée

```text
E045  données + téléphone + résilience
E046  campagne adaptative multi-fidélité
E047  conseiller hiérarchique v2
E048  surrogate téléphone différentiable
E049  holdout prospectif entièrement nouveau
E050  fine-tuning éventuel seulement après E049
```

E031 et les prompts déjà utilisés pour concevoir ces changements ne doivent
plus devenir le test final.
"""
    ),
    md("## 13. Fichiers à revoir avant E046"),
    code(
        r"""
review = {
    "data_card": str(R / "data-card.json"),
    "conflicts": str(R / "label-conflicts.csv"),
    "duplicates": str(R / "duplicate-images.csv"),
    "srmpgd_noops": str(R / "srmpgd-noop-images.csv"),
    "phone_template": str(R / "inputs/phone-captures.csv"),
    "recovery_runbook": str(R / "recovery-runbook.json"),
    "manifest": str(R / "artifact-manifest.json"),
}
display(review)
display(data_card)
"""
    ),
    md("## 14. Manifest et intégrité"),
    code(
        r"""
manifest = json.loads((R / "artifact-manifest.json").read_text(encoding="utf-8"))
manifest_df = pd.DataFrame(manifest)
display(manifest_df)
print("Entrées manifestées :", len(manifest_df))
print("SHA du manifeste    :", complete["artifact_manifest_sha256"])
"""
    ),
    md(
        r"""
## Conclusion

E045 est une **porte de données**, pas un résultat de génération.

La prochaine action correcte est :

1. revoir les conflits et les no-op ;
2. sélectionner un lot d'images E044/E029/E031 représentatif ;
3. importer les scans téléphone réels ;
4. geler le split de développement E046 ;
5. seulement ensuite autoriser la campagne multi-fidélité.

Aucune valeur `22/37`, `37/37` ou MER faible ne remplace cette validation physique.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
        "prooftag": {
            "experiment": "e045-foundation-resilience-v1",
            "role": "cpu-audit-and-gate",
            "generation": False,
            "training": False,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(OUTPUT)
