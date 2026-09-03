#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "notebooks/48_e046_controlled_best_generator.ipynb"
ATLAS = ROOT / "notebooks/49_e046_visual_atlas.ipynb"


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


COMMON_SETUP = r"""
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
    "PROOFTAG_E046_OUTPUT_ROOT",
    "/data/e046-controlled-best-generator-v1",
))
latest = json.loads((OUTPUT_ROOT / "LATEST.json").read_text(encoding="utf-8"))
R = Path(latest["plan_dir"])
plan = json.loads((R / "plan.json").read_text(encoding="utf-8"))
verdict_path = R / "verdict.json"
complete_path = R / "COMPLETE.json"
verdict = (
    json.loads(verdict_path.read_text(encoding="utf-8"))
    if verdict_path.is_file()
    else None
)

def load_rows():
    final = R / "dataset/e046-observations.json"
    if final.is_file():
        return json.loads(final.read_text(encoding="utf-8"))

    rows = []
    for path in sorted((R / "parents").glob("*/scoring/comparison.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    for path in sorted((R / "refinements").glob("*/*/scoring/comparison.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows

rows = load_rows()
df = pd.DataFrame(rows)
for column, default in (
    ("srmpgd_recipe_id", None),
    ("row_id", None),
    ("pixel_duplicate", False),
    ("projection_was_active", False),
):
    if column not in df.columns:
        df[column] = default
print("Plan E046 :", R)
print("Profile   :", plan["profile"])
print("Status    :", latest.get("status"))
print("Rows      :", len(df))
print("Engine QR :", plan["qr_software_engine"])
"""


main_cells = [
    md(
        r"""
# E046 — Controlled Best-Generator Dataset

Cette campagne génère un **nouveau dataset propre**. Elle ne recycle pas les
306 372 anciens PNG génériques ignorés par E045.

## Règles scientifiques

- vérité QR logicielle principale : **qr-scanner-wechat via qr-verify@0.2.0** ;
- payload exact uniquement ;
- 37 presets, trois répétitions conservatrices ;
- OpenCV, ZBar et ZXing ne votent pas dans le score principal ;
- raster brut toujours conservé ;
- aucune bordure blanche/uniforme éligible comme sortie finale ;
- variante `scene_preserving` : pas de crop et cœur 580×580 octet-identique ;
- Stage 1, Stage 2, latent et tous les checkpoints SR-MPGD sont persistés ;
- le téléphone reste la vérité finale et n'est pas simulé comme acquis.

Le notebook fonctionne également en mode partiel, uniquement lorsqu'aucun Job
GPU E046 n'est actif :

```powershell
.\scripts\e046-remote.ps1 -Partial
```

Le script refuse de démarrer Jupyter pendant une génération afin de ne pas
concurrencer la RTX.
"""
    ),
    code(COMMON_SETUP),
    md("## 1. État d'avancement et contrat"),
    code(
        r"""
parent_total = len(plan["candidates"])
parent_generated = sum(
    (R / "parents" / item["id"] / "GENERATION_COMPLETE.json").is_file()
    for item in plan["candidates"]
)
parent_scored = sum(
    (R / "parents" / item["id"] / "SCORING_COMPLETE.json").is_file()
    for item in plan["candidates"]
)
selected_path = R / "selected-parents.json"
selected = (
    json.loads(selected_path.read_text(encoding="utf-8"))
    if selected_path.is_file()
    else {"selected": []}
)
refinement_tasks = [
    (item["candidate_id"], recipe["id"])
    for item in selected["selected"]
    for recipe in plan["srmpgd_recipes"]
]
refinement_generated = sum(
    (R / "refinements" / candidate / recipe / "GENERATION_COMPLETE.json").is_file()
    for candidate, recipe in refinement_tasks
)
refinement_scored = sum(
    (R / "refinements" / candidate / recipe / "SCORING_COMPLETE.json").is_file()
    for candidate, recipe in refinement_tasks
)

state = pd.DataFrame([
    ["Parents prévus", parent_total],
    ["Parents générés", parent_generated],
    ["Parents scorés", parent_scored],
    ["Parents sélectionnés", len(selected["selected"])],
    ["Trajectoires SR-MPGD prévues", len(refinement_tasks)],
    ["Trajectoires générées", refinement_generated],
    ["Trajectoires scorées", refinement_scored],
    ["Agrégation complète", complete_path.is_file()],
], columns=["Étape", "Valeur"])
display(state)
"""
    ),
    code(
        r"""
display({
    "source_commit": plan["source_commit"],
    "E045_plan_id": plan["e045_plan_id"],
    "E045_manifest_sha256": plan["e045_manifest_sha256"],
    "scientific_plan_hash": plan["scientific_plan_hash"],
    "primary_label": plan["qr_primary_label"],
    "other_decoders": plan["other_decoders_role"],
    "production_ready": False if verdict is None else verdict["production_ready"],
})
"""
    ),
    md("## 2. Prompts, familles, payloads et compositions"),
    code(
        r"""
candidate_df = pd.DataFrame(plan["candidates"])
display(candidate_df[[
    "id", "prompt_id", "prompt_family", "prompt_variant_index",
    "parent_recipe_id", "seed", "payload", "quiet_zone_hint", "prompt"
]].style.set_properties(subset=["prompt", "quiet_zone_hint"], **{"white-space": "normal"}))
"""
    ),
    code(
        r"""
family_counts = candidate_df["prompt_family"].value_counts()
plt.figure(figsize=(11, 4))
plt.bar(family_counts.index, family_counts.values)
plt.ylabel("Candidats")
plt.xlabel("Famille visuelle")
plt.title("Couverture des familles E046")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()
"""
    ),
    md("## 3. Recettes Stage 1 / Stage 2"),
    code(
        r"""
parent_recipes = pd.DataFrame(plan["parent_recipes"])
display(parent_recipes.style.set_properties(
    subset=["rationale"], **{"white-space": "normal"}
))
"""
    ),
    code(
        r"""
coverage = parent_recipes[[
    "qr_mask_pattern", "error_correction", "stage1_steps",
    "stage1_guidance_scale", "stage1_controlnet_scale",
    "stage2_initialization", "stage2_strength", "stage2_steps",
    "stage2_controlnet_scale", "stage2_qr_weight",
    "stage2_perceptual_weight",
]]
display(coverage)
"""
    ),
    code(
        r"""
fig, ax = plt.subplots(figsize=(11, 6))
scatter = ax.scatter(
    parent_recipes["stage2_qr_weight"],
    parent_recipes["stage2_perceptual_weight"],
    s=90,
)
for _, row in parent_recipes.iterrows():
    ax.annotate(
        f"m{row['qr_mask_pattern']}",
        (row["stage2_qr_weight"], row["stage2_perceptual_weight"]),
        xytext=(5, 5),
        textcoords="offset points",
    )
ax.set_xscale("log")
ax.set_xlabel("Poids SRG Stage 2")
ax.set_ylabel("Poids perceptuel Stage 2")
ax.set_title("Espace initial Stage 2, masques 0 à 7")
plt.tight_layout()
plt.show()
"""
    ),
    md("## 4. Recettes SR-MPGD — jamais forcées sur tous les parents"),
    code(
        r"""
srmpgd_recipes = pd.DataFrame(plan["srmpgd_recipes"])
display(srmpgd_recipes.style.set_properties(
    subset=["rationale"], **{"white-space": "normal"}
))
"""
    ),
    code(
        r"""
if not srmpgd_recipes.empty:
    plt.figure(figsize=(9, 5))
    plt.scatter(
        srmpgd_recipes["gamma"],
        srmpgd_recipes["latent_radius_rms"],
        s=srmpgd_recipes["max_iterations"] * 20,
    )
    for _, row in srmpgd_recipes.iterrows():
        plt.annotate(row["id"], (row["gamma"], row["latent_radius_rms"]),
                     xytext=(5, 5), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("Gamma brut")
    plt.ylabel("Rayon latent RMS")
    plt.title("Gamma, trust region et nombre d'itérations")
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## 5. Dataset disponible"),
    code(
        r"""
if df.empty:
    display(Markdown("**Aucun scoring disponible pour le moment.**"))
else:
    display(df.head(30))
    print("Colonnes :", len(df.columns))
    print("Rasters uniques :", df["image_sha256"].nunique())
"""
    ),
    md("## 6. Distribution WeChat exacte / 37"),
    code(
        r"""
if not df.empty:
    exact = pd.to_numeric(df["wechat_exact_presets"], errors="coerce").dropna()
    plt.figure(figsize=(10, 5))
    plt.hist(exact, bins=np.arange(-0.5, 38.5, 1))
    plt.xlabel("Presets exacts qr-scanner-wechat / 37")
    plt.ylabel("Images")
    plt.title("Distribution de la cible logicielle principale")
    plt.tight_layout()
    plt.show()

    buckets = pd.cut(
        exact,
        bins=[-1, 5, 15, 25, 35, 37],
        labels=["0–5", "6–15", "16–25", "26–35", "36–37"],
    ).value_counts().sort_index()
    display(buckets.to_frame("images"))
"""
    ),
    md(
        r"""
> Un score `22/37` signifie que 22 transformations logicielles ont rendu le
> payload exact à `qr-scanner-wechat`. Ce n'est pas un taux de réussite téléphone.
"""
    ),
    md("## 7. WeChat par prompt, masque, ECC et recette"),
    code(
        r"""
if not df.empty:
    best_prompt = (
        df.groupby(["prompt_id", "prompt_family"], dropna=False)["wechat_exact_presets"]
        .max()
        .sort_values(ascending=False)
        .reset_index()
    )
    display(best_prompt)

    plt.figure(figsize=(12, 5))
    plt.bar(best_prompt["prompt_id"], best_prompt["wechat_exact_presets"])
    plt.ylabel("Meilleur WeChat exact / 37")
    plt.xlabel("Prompt")
    plt.title("Sensibilité au prompt")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
"""
    ),
    code(
        r"""
if not df.empty:
    by_mask = (
        df.groupby("qr_mask_pattern")["wechat_exact_presets"]
        .agg(["count", "mean", "max", "median"])
        .reset_index()
    )
    display(by_mask)
    plt.figure(figsize=(9, 4))
    plt.bar(by_mask["qr_mask_pattern"].astype(str), by_mask["max"])
    plt.xlabel("Masque QR")
    plt.ylabel("Maximum exact / 37")
    plt.title("Couverture des huit masques légaux")
    plt.tight_layout()
    plt.show()
"""
    ),
    code(
        r"""
if not df.empty:
    by_recipe = (
        df.groupby(["parent_recipe_id", "source_kind"])["wechat_exact_presets"]
        .agg(["count", "mean", "max"])
        .sort_values("max", ascending=False)
    )
    display(by_recipe)
"""
    ),
    md("## 8. Compromis QR / esthétique"),
    code(
        r"""
if not df.empty:
    plot = df.copy()
    for column in ("wechat_exact_presets", "clip_aesthetic", "hpsv2_1",
                   "clip_score", "lpips", "module_error_rate"):
        plot[column] = pd.to_numeric(plot[column], errors="coerce")

    paired = plot.dropna(subset=["wechat_exact_presets", "clip_aesthetic"])
    plt.figure(figsize=(10, 7))
    for source, group in paired.groupby("source_kind"):
        plt.scatter(
            group["clip_aesthetic"],
            group["wechat_exact_presets"],
            alpha=0.55,
            label=source,
        )
    plt.xlabel("CLIP-Aesthetic")
    plt.ylabel("WeChat exact / 37")
    plt.title("Frontière scannabilité logicielle / esthétique")
    plt.legend()
    plt.tight_layout()
    plt.show()
"""
    ),
    code(
        r"""
if not df.empty:
    paired = plot.dropna(subset=["wechat_exact_presets", "hpsv2_1"])
    plt.figure(figsize=(10, 7))
    plt.scatter(
        paired["hpsv2_1"],
        paired["wechat_exact_presets"],
        alpha=0.55,
    )
    plt.xlabel("HPSv2")
    plt.ylabel("WeChat exact / 37")
    plt.title("WeChat exact vs préférence visuelle HPS")
    plt.tight_layout()
    plt.show()
"""
    ),
    code(
        r"""
if not df.empty:
    paired = plot.dropna(subset=["module_error_rate", "wechat_exact_presets"])
    plt.figure(figsize=(10, 7))
    plt.scatter(
        paired["module_error_rate"],
        paired["wechat_exact_presets"],
        alpha=0.5,
    )
    plt.xlabel("Module error rate")
    plt.ylabel("WeChat exact / 37")
    plt.title("MER reste un diagnostic, pas la cible")
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## 9. Quiet zone : brut contre scene-preserving"),
    code(
        r"""
if not df.empty:
    qz = df[df["quiet_zone_variant"].isin(["raw", "scene_preserving"])].copy()
    keys = [
        "candidate_id", "source_kind", "srmpgd_recipe_id", "iteration"
    ]
    paired_qz = qz.pivot_table(
        index=keys,
        columns="quiet_zone_variant",
        values=["wechat_exact_presets", "clip_aesthetic", "hpsv2_1"],
        aggfunc="first",
    )
    display(paired_qz.head(100))
"""
    ),
    code(
        r"""
if not df.empty:
    qz_delta_rows = []
    for _, group in qz.groupby(keys, dropna=False):
        if set(group["quiet_zone_variant"]) != {"raw", "scene_preserving"}:
            continue
        raw = group[group["quiet_zone_variant"] == "raw"].iloc[0]
        scene = group[group["quiet_zone_variant"] == "scene_preserving"].iloc[0]
        qz_delta_rows.append({
            "candidate_id": raw["candidate_id"],
            "source_kind": raw["source_kind"],
            "srmpgd_recipe_id": raw.get("srmpgd_recipe_id"),
            "iteration": raw["iteration"],
            "delta_wechat": scene["wechat_exact_presets"] - raw["wechat_exact_presets"],
            "delta_clip_aesthetic": scene["clip_aesthetic"] - raw["clip_aesthetic"],
            "delta_hpsv2": (
                scene["hpsv2_1"] - raw["hpsv2_1"]
                if pd.notna(scene["hpsv2_1"]) and pd.notna(raw["hpsv2_1"])
                else np.nan
            ),
            "core_same": bool(scene["core_byte_identical_to_raw"]),
            "qz_guard": scene["quiet_zone_delivery_guard_pass"],
        })
    qz_delta = pd.DataFrame(qz_delta_rows)
    display(qz_delta)
    if not qz_delta.empty:
        plt.figure(figsize=(9, 5))
        plt.hist(qz_delta["delta_wechat"], bins=np.arange(-37.5, 38.5, 1))
        plt.xlabel("Gain scene-preserving – brut, presets exacts")
        plt.ylabel("Paires")
        plt.title("Effet logiciel de la composition périphérique")
        plt.tight_layout()
        plt.show()
"""
    ),
    md(
        r"""
La variante `scene_preserving` ne colle pas un cadre uniforme. Elle part de
l'œuvre, conserve les couleurs à basse fréquence, lisse les détails locaux et
éclaircit sans crop. Le cœur QR doit garder exactement le même hash.
"""
    ),
    md("## 10. SR-MPGD : trajectoires, gamma, projection et no-op"),
    code(
        r"""
if not df.empty:
    sr = plot[plot["source_kind"] == "srmpgd"].copy()
    if sr.empty:
        display(Markdown("Aucune trajectoire SR-MPGD scorée."))
    else:
        raw_sr = sr[sr["quiet_zone_variant"] == "raw"]
        grouped = raw_sr.groupby(
            ["candidate_id", "srmpgd_recipe_id", "gamma", "iteration"],
            dropna=False,
        )["wechat_exact_presets"].max().reset_index()
        for (candidate, recipe), group in grouped.groupby(
            ["candidate_id", "srmpgd_recipe_id"]
        ):
            plt.figure(figsize=(8, 4))
            plt.plot(group["iteration"], group["wechat_exact_presets"], marker="o")
            plt.xlabel("Itération")
            plt.ylabel("WeChat exact / 37")
            plt.title(f"{candidate}\n{recipe}")
            plt.ylim(-0.5, 37.5)
            plt.tight_layout()
            plt.show()
"""
    ),
    code(
        r"""
if not df.empty and "projection_was_active" in df:
    projection = (
        df[df["source_kind"] == "srmpgd"]
        .groupby(["srmpgd_recipe_id", "gamma"], dropna=False)
        .agg(
            rows=("variant", "count"),
            projection_active=("projection_was_active", "sum"),
            max_wechat=("wechat_exact_presets", "max"),
        )
    )
    display(projection)
"""
    ),
    code(
        r"""
if not df.empty and "pixel_duplicate" in df:
    duplicate = (
        df.groupby(["source_kind", "stage"])["pixel_duplicate"]
        .agg(["count", "sum"])
    )
    display(duplicate)
"""
    ),
    md("## 11. Gardes visuelles et erreurs techniques"),
    code(
        r"""
if not df.empty:
    guard = (
        df.groupby(["source_kind", "quiet_zone_variant"])["visual_guard_pass"]
        .agg(["count", "sum", "mean"])
    )
    display(guard)
"""
    ),
    code(
        r"""
failure_paths = sorted((R / "failures").glob("*.json"))
print("Fichiers d'échec :", len(failure_paths))
for path in failure_paths[:50]:
    display(json.loads(path.read_text(encoding="utf-8")))
"""
    ),
    md("## 12. Pareto et gagnants"),
    code(
        r"""
pareto_path = R / "dataset/pareto-front.json"
if pareto_path.is_file():
    pareto_df = pd.DataFrame(json.loads(pareto_path.read_text(encoding="utf-8")))
    pareto_columns = [
        "candidate_id", "source_kind", "srmpgd_recipe_id", "variant",
        "iteration", "gamma", "wechat_exact_presets",
        "wechat_original_exact", "clip_aesthetic", "hpsv2_1",
        "clip_score", "lpips", "image_path"
    ]
    display(pareto_df.reindex(columns=pareto_columns))
else:
    display(Markdown("Pareto disponible après agrégation."))
"""
    ),
    code(
        r"""
best_path = R / "dataset/best-by-prompt.json"
if best_path.is_file():
    best_df = pd.DataFrame(json.loads(best_path.read_text(encoding="utf-8")))
    best_columns = [
        "prompt_id", "candidate_id", "source_kind", "variant",
        "srmpgd_recipe_id", "iteration", "wechat_exact_presets",
        "clip_aesthetic", "hpsv2_1", "image_path"
    ]
    display(best_df.reindex(columns=best_columns))
"""
    ),
    md("## 13. Galerie des meilleurs candidats"),
    code(
        r"""
def show_gallery(rows, title, columns=4, limit=24):
    rows = list(rows)[:limit]
    if not rows:
        display(Markdown(f"**{title} : aucune image.**"))
        return
    nrows = math.ceil(len(rows) / columns)
    fig, axes = plt.subplots(nrows, columns, figsize=(16, 4.5 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, row in zip(axes, rows):
        path = Path(str(row["image_path"]))
        if path.is_file():
            axis.imshow(PILImage.open(path).convert("RGB"))
        axis.set_title(
            f"{row['prompt_id']}\n{row['source_kind']} {row['variant']}\n"
            f"WeChat {row['wechat_exact_presets']}/37",
            fontsize=9,
        )
        axis.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if best_path.is_file():
    show_gallery(
        json.loads(best_path.read_text(encoding="utf-8")),
        "Meilleur candidat par prompt",
    )
elif not df.empty:
    fallback = (
        df[df["eligible_final"] == True]
        .sort_values(["wechat_exact_presets", "clip_aesthetic"], ascending=False)
        .drop_duplicates("prompt_id")
        .to_dict("records")
    )
    show_gallery(fallback, "Meilleurs candidats partiels")
"""
    ),
    md("## 14. Pipeline complète du gagnant"),
    code(
        r"""
if verdict is None:
    display(Markdown("Verdict final indisponible en mode partiel."))
else:
    display(verdict)
    winner_id = verdict["winner_candidate_id"]
    parent_root = R / "parents" / winner_id / "images"
    panel = [
        ("Stage 1 brut", parent_root / "stage1-raw.png"),
        ("Stage 1 scene-qz", parent_root / "stage1-scene-qz.png"),
        ("Stage 2 brut", parent_root / "stage2-raw.png"),
        ("Stage 2 scene-qz", parent_root / "stage2-scene-qz.png"),
        ("Gagnant final", R / "pipeline/99-FINAL-QR.png"),
    ]
    fig, axes = plt.subplots(1, len(panel), figsize=(22, 5))
    for axis, (label, path) in zip(axes, panel):
        if path.is_file():
            axis.imshow(PILImage.open(path).convert("RGB"))
        axis.set_title(label)
        axis.axis("off")
    plt.tight_layout()
    plt.show()
"""
    ),
    code(
        r"""
for name in (
    "best-by-prompt-contact-sheet.png",
    "pareto-contact-sheet.png",
    "phone-sample-contact-sheet.png",
):
    path = R / "pipeline" / name
    if path.is_file():
        display(Markdown(f"### {name}"))
        display(Image(filename=str(path)))
"""
    ),
    md("## 15. Préparation E047 et téléphone"),
    code(
        r"""
if verdict is not None:
    readiness = {
        "software_dataset_complete": verdict["software_dataset_complete"],
        "software_advisor_training_candidate": verdict[
            "software_advisor_training_candidate"
        ],
        "automatic_advisor_training_authorized": verdict[
            "automatic_advisor_training_authorized"
        ],
        "phone_truth_available": verdict["phone_truth_available"],
        "phone_surrogate_training_authorized": verdict[
            "phone_surrogate_training_authorized"
        ],
        "production_ready": verdict["production_ready"],
        "next_action": verdict["next_action"],
    }
    display(readiness)
"""
    ),
    md(
        r"""
### Décision attendue après revue

E047 pourra apprendre les paramètres qui maximisent **WeChat exact / 37** sous
gardes esthétiques. L'entraînement automatique reste bloqué tant que :

- les splits prompt/payload/pixels ne sont pas gelés ;
- les erreurs et no-op ne sont pas audités ;
- un échantillon représentatif n'est pas sélectionné pour le téléphone.

Le surrogate téléphone et la production restent interdits sans labels physiques.
"""
    ),
]


atlas_cells = [
    md(
        r"""
# E046 — Atlas visuel complet

Cet atlas affiche les images plutôt que de résumer uniquement les scores :

- Stage 1 brut / scene-preserving ;
- Stage 2 brut / scene-preserving ;
- chaque trajectoire SR-MPGD, toutes les itérations ;
- différences périphériques et différences dans le cœur ;
- meilleurs, échecs, no-op et Pareto.

La bordure uniforme blanche/adaptive-light n'existe pas dans les sorties E046.
"""
    ),
    code(COMMON_SETUP),
    code(
        r"""
if df.empty:
    raise RuntimeError("Aucune image E046 scorée. Ouvrir plus tard ou utiliser le notebook principal.")
for column in ("wechat_exact_presets", "clip_aesthetic", "hpsv2_1", "lpips"):
    df[column] = pd.to_numeric(df[column], errors="coerce")
"""
    ),
    md("## 1. Stage 1 / Stage 2 pour chaque parent"),
    code(
        r"""
for candidate_id, group in df[df["source_kind"] == "parent"].groupby("candidate_id"):
    order = ["stage1_raw", "stage1_scene_qz", "stage2_raw", "stage2_scene_qz"]
    records = {
        row["variant"]: row
        for _, row in group.iterrows()
    }
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for axis, variant in zip(axes, order):
        row = records.get(variant)
        if row is not None and Path(str(row["image_path"])).is_file():
            axis.imshow(PILImage.open(row["image_path"]).convert("RGB"))
            axis.set_title(
                f"{variant}\nWeChat {int(row['wechat_exact_presets'])}/37\n"
                f"AES {float(row['clip_aesthetic']):.3f}"
            )
        else:
            axis.set_title(f"{variant}\nabsent")
        axis.axis("off")
    plt.suptitle(candidate_id)
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## 2. Différence brut → scene-preserving"),
    code(
        r"""
def diff_map(left_path, right_path):
    left = np.asarray(PILImage.open(left_path).convert("RGB"), dtype=np.float32) / 255
    right = np.asarray(PILImage.open(right_path).convert("RGB"), dtype=np.float32) / 255
    return np.abs(right - left).mean(axis=2)

for candidate_id, group in df[df["source_kind"] == "parent"].groupby("candidate_id"):
    records = {row["variant"]: row for _, row in group.iterrows()}
    if "stage2_raw" not in records or "stage2_scene_qz" not in records:
        continue
    raw = records["stage2_raw"]
    scene = records["stage2_scene_qz"]
    delta = diff_map(raw["image_path"], scene["image_path"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(PILImage.open(raw["image_path"]).convert("RGB"))
    axes[0].set_title("Stage 2 brut")
    axes[1].imshow(PILImage.open(scene["image_path"]).convert("RGB"))
    axes[1].set_title("Scene-preserving")
    im = axes[2].imshow(delta)
    axes[2].set_title("Différence absolue moyenne")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    for axis in axes:
        axis.axis("off")
    plt.suptitle(
        f"{candidate_id} — cœur inchangé: {scene['core_byte_identical_to_raw']}"
    )
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## 3. Toutes les trajectoires SR-MPGD"),
    code(
        r"""
sr = df[df["source_kind"] == "srmpgd"]
if sr.empty:
    display(Markdown("Aucune trajectoire scorée."))
else:
    for (candidate_id, recipe_id, qz_variant), group in sr.groupby(
        ["candidate_id", "srmpgd_recipe_id", "quiet_zone_variant"]
    ):
        group = group.sort_values("iteration")
        columns = 5
        nrows = math.ceil(len(group) / columns)
        fig, axes = plt.subplots(nrows, columns, figsize=(18, 4.2 * nrows))
        axes = np.asarray(axes).reshape(-1)
        for axis in axes:
            axis.axis("off")
        for axis, (_, row) in zip(axes, group.iterrows()):
            path = Path(str(row["image_path"]))
            if path.is_file():
                axis.imshow(PILImage.open(path).convert("RGB"))
            axis.set_title(
                f"i{int(row['iteration'])}\n"
                f"WeChat {int(row['wechat_exact_presets'])}/37\n"
                f"LPIPS {float(row['lpips']):.4f}",
                fontsize=9,
            )
            axis.axis("off")
        plt.suptitle(f"{candidate_id} — {recipe_id} — {qz_variant}")
        plt.tight_layout()
        plt.show()
"""
    ),
    md("## 4. Courbes image + métriques pour chaque trajectoire"),
    code(
        r"""
if not sr.empty:
    raw = sr[sr["quiet_zone_variant"] == "raw"]
    for (candidate_id, recipe_id), group in raw.groupby(
        ["candidate_id", "srmpgd_recipe_id"]
    ):
        group = group.sort_values("iteration")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(
            group["iteration"],
            group["wechat_exact_presets"],
            marker="o",
            label="WeChat exact /37",
        )
        ax1.set_xlabel("Itération")
        ax1.set_ylabel("WeChat exact /37")
        ax1.set_ylim(-0.5, 37.5)
        ax2 = ax1.twinx()
        ax2.plot(
            group["iteration"],
            group["lpips"],
            marker="x",
            label="LPIPS",
        )
        ax2.set_ylabel("LPIPS")
        plt.title(f"{candidate_id} — {recipe_id}")
        fig.tight_layout()
        plt.show()
"""
    ),
    md("## 5. Top logiciel sous garde visuelle"),
    code(
        r"""
safe = df[df["eligible_final"] == True].sort_values(
    ["wechat_exact_presets", "clip_aesthetic", "hpsv2_1"],
    ascending=[False, False, False],
)
top = safe.head(32).to_dict("records")

if not top:
    display(Markdown("Aucun candidat ne passe encore la garde visuelle."))
else:
    columns = 4
    nrows = math.ceil(len(top) / columns)
    fig, axes = plt.subplots(nrows, columns, figsize=(16, 4.6 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, row in zip(axes, top):
        path = Path(str(row["image_path"]))
        if path.is_file():
            axis.imshow(PILImage.open(path).convert("RGB"))
        axis.set_title(
            f"{row['candidate_id']}\n"
            f"{row['source_kind']} {row['variant']}\n"
            f"WeChat {int(row['wechat_exact_presets'])}/37 · "
            f"AES {float(row['clip_aesthetic']):.2f}",
            fontsize=8,
        )
        axis.axis("off")
    plt.suptitle("Top E046 — logiciel et garde visuelle")
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## 6. Meilleurs esthétiques parmi les faibles scores QR"),
    code(
        r"""
low = (
    df[(df["eligible_final"] == True) & (df["wechat_exact_presets"] <= 5)]
    .sort_values(["clip_aesthetic", "hpsv2_1"], ascending=False)
    .head(24)
)
if low.empty:
    display(Markdown("Aucun exemple faible WeChat sous garde."))
else:
    columns = 4
    nrows = math.ceil(len(low) / columns)
    fig, axes = plt.subplots(nrows, columns, figsize=(16, 4.5 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, (_, row) in zip(axes, low.iterrows()):
        axis.imshow(PILImage.open(row["image_path"]).convert("RGB"))
        axis.set_title(
            f"{row['prompt_id']}\nWeChat {int(row['wechat_exact_presets'])}/37\n"
            f"AES {float(row['clip_aesthetic']):.2f}",
            fontsize=9,
        )
        axis.axis("off")
    plt.suptitle("Hard negatives esthétiques")
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## 7. Pareto complet"),
    code(
        r"""
pareto_path = R / "dataset/pareto-front.json"
if pareto_path.is_file():
    pareto = json.loads(pareto_path.read_text(encoding="utf-8"))
    columns = 4
    nrows = math.ceil(len(pareto) / columns)
    fig, axes = plt.subplots(nrows, columns, figsize=(16, 4.6 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, row in zip(axes, pareto):
        axis.imshow(PILImage.open(row["image_path"]).convert("RGB"))
        axis.set_title(
            f"{row['candidate_id']}\n{row['variant']}\n"
            f"WeChat {row['wechat_exact_presets']}/37 · "
            f"AES {float(row['clip_aesthetic']):.2f}",
            fontsize=8,
        )
        axis.axis("off")
    plt.suptitle("Front de Pareto E046")
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## 8. Gagnant final et rappel de sécurité"),
    code(
        r"""
if verdict is not None:
    final_path = R / "pipeline/99-FINAL-QR.png"
    display(Image(filename=str(final_path)))
    display({
        "winner": verdict["winner_candidate_id"],
        "variant": verdict["winner_variant"],
        "WeChat_exact": f"{verdict['winner_wechat_exact_presets']}/37",
        "original_exact": verdict["winner_wechat_original_exact"],
        "uniform_quiet_zone_replacement": verdict[
            "winner_uniform_quiet_zone_replacement"
        ],
        "phone_validated": verdict["phone_truth_available"],
        "production_ready": verdict["production_ready"],
    })
"""
    ),
    md(
        r"""
Aucun gagnant E046 ne doit être appelé « fonctionnel téléphone » avant la campagne
physique. L'atlas sert précisément à choisir les images à tester sur appareils.
"""
    ),
]


def notebook(cells: list[dict], role: str) -> dict:
    return {
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
                "experiment": "e046-controlled-best-generator-v1",
                "role": role,
                "generation": False,
                "training": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


MAIN.write_text(
    json.dumps(notebook(main_cells, "complete-audit"), ensure_ascii=False, indent=1)
    + "\n",
    encoding="utf-8",
)
ATLAS.write_text(
    json.dumps(notebook(atlas_cells, "visual-atlas"), ensure_ascii=False, indent=1)
    + "\n",
    encoding="utf-8",
)

print(MAIN)
print(ATLAS)
