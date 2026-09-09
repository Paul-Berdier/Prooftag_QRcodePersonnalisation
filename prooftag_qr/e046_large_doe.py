"""Plan d'expériences (DOE) déterministe de la campagne E046 large advisor dataset.

Le dataset advisor doit apprendre l'effet des paramètres Stage1/Stage2, pas
seulement l'effet du prompt. Chaque prompt reçoit donc les mêmes
``CONFIG_COUNT`` configurations indépendantes :

* la configuration 0 est l'ANCRE historique E044 (``m4_e044_anchor`` dans le
  catalogue E046 pilote), celle qui a produit le premier parent brutaliste très
  bon ;
* les 11 autres proviennent d'un hypercube latin (LHS) déterministe sur 13
  dimensions, choisi parmi ``LHS_TRIALS`` candidats par un critère maximin
  pénalisé par la corrélation maximale entre facteurs, puis ordonné par
  farthest-point de sorte que tout préfixe (smoke : 2, pilot : 6) reste
  espace-rempli.

Toute la génération aléatoire passe par ``random.Random(seed).random()`` dont la
séquence est garantie stable par la documentation Python. Aucun appel à
``random.shuffle`` (non garanti) ni à NumPy ``Generator`` (non garanti) n'est
utilisé. Le résultat est versionné dans ``data/e046_large_doe_v1.json`` et un
test vérifie que la reconstruction est octet-identique.

Le masque QR n'est PAS fixé par configuration : chaque configuration porte un
``mask_slot`` et le masque réel d'un couple (prompt, config) est
``(mask_slot + mask_rotation) mod 8`` avec
``mask_rotation = (design_row - 5 + 2 * (family_index - 1)) mod 8``.
La rotation est centrée sur la ligne structurale 5 de la première famille, donc
l'ancre brutaliste historique (ordinal 1, rotation 0) garde exactement son
masque 4 ; le terme de famille (multiplicateur 2) évite que le masque soit une
fonction pure de la ligne structurale à configuration fixée. Chaque famille
couvrant les 16 lignes, chaque masque reçoit exactement 384 parents dans le
profil complet ; la rotation pilot (lignes i+5 et i+13 par famille) et les quatre
prompts smoke restent eux aussi équilibrés sur les huit masques.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .resilient_experiment import stable_hash

DOE_SCHEMA = "e046-large-doe-v1"
DOE_ID = "e046-large-doe-v1"
CONFIG_COUNT = 12
PROFILE_CONFIG_COUNTS = {"smoke": 2, "pilot": 6, "full": CONFIG_COUNT}
LHS_TRIALS = 3000
CORRELATION_PENALTY = 0.5
PARENT_SEED_BASE = 460_000_000
MASK_ANCHOR_DESIGN_ROW = 5
MASK_FAMILY_MULTIPLIER = 2

DimensionKind = Literal["int", "float", "categorical"]


@dataclass(frozen=True, slots=True)
class Dimension:
    name: str
    kind: DimensionKind
    lower: float | None
    upper: float | None
    scale: Literal["linear", "log"] | None
    choices: tuple[Any, ...] | None
    decimals: int
    stage: str
    rationale: str

    def to_unit(self, value: Any) -> float:
        if self.kind == "categorical":
            assert self.choices is not None
            index = self.choices.index(value)
            return index / (len(self.choices) - 1) if len(self.choices) > 1 else 0.0
        assert self.lower is not None and self.upper is not None
        if self.scale == "log":
            return (math.log(float(value)) - math.log(self.lower)) / (
                math.log(self.upper) - math.log(self.lower)
            )
        return (float(value) - self.lower) / (self.upper - self.lower)

    def from_unit(self, unit: float) -> Any:
        if self.kind == "categorical":
            assert self.choices is not None
            index = min(len(self.choices) - 1, int(unit * len(self.choices)))
            return self.choices[index]
        assert self.lower is not None and self.upper is not None
        if self.scale == "log":
            value = math.exp(
                math.log(self.lower) + unit * (math.log(self.upper) - math.log(self.lower))
            )
        else:
            value = self.lower + unit * (self.upper - self.lower)
        if self.kind == "int":
            return int(round(value))
        return round(float(value), self.decimals)


# Bornes issues des campagnes E026-E046 (voir docs/e046-large-advisor-dataset.md,
# section « Paramètres explorés ») : elles englobent toutes les recettes du
# catalogue E046 pilote (m0..m7) et l'ancre E044, sans extrapoler au-delà des
# valeurs déjà observées comme physiquement sensées sur SD1.5 + QR ControlNet.
DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        "stage1_steps",
        "int",
        30,
        50,
        "linear",
        None,
        0,
        "stage1",
        "E046 pilote : 30 (m3) à 50 (m2).",
    ),
    Dimension(
        "stage1_guidance_scale",
        "float",
        6.0,
        8.5,
        "linear",
        None,
        2,
        "stage1",
        "E046 pilote : 6.5 (m2) à 8.2 (m6) ; la borne 6.0-8.5 encadre.",
    ),
    Dimension(
        "stage1_controlnet_scale",
        "float",
        1.0,
        1.6,
        "linear",
        None,
        3,
        "stage1",
        "E046 pilote : 1.05 (m3) à 1.55 (m2).",
    ),
    Dimension(
        "control_guidance_start",
        "float",
        0.0,
        0.08,
        "linear",
        None,
        3,
        "stage1",
        "E046 pilote : 0.0 à 0.08 (m6).",
    ),
    Dimension(
        "control_guidance_end",
        "float",
        0.90,
        1.0,
        "linear",
        None,
        3,
        "stage1",
        "E046 pilote : 0.90 (m3) à 1.0.",
    ),
    Dimension(
        "stage2_strength",
        "float",
        0.75,
        1.0,
        "linear",
        None,
        3,
        "stage2",
        "E046 pilote : 0.78 (m3) à 1.0 ; en dessous de 0.75 Stage2 ne réécrit plus assez le QR.",
    ),
    Dimension(
        "stage2_steps",
        "int",
        30,
        60,
        "linear",
        None,
        0,
        "stage2",
        "E046 pilote : 40 à 60 (m2) ; 30 ajouté pour mesurer le coût/bénéfice.",
    ),
    Dimension(
        "stage2_controlnet_scale",
        "float",
        0.90,
        1.35,
        "linear",
        None,
        3,
        "stage2",
        "E046 pilote : 0.90 (m3) à 1.35 (m7).",
    ),
    Dimension(
        "stage2_qr_weight",
        "float",
        50.0,
        500.0,
        "log",
        None,
        1,
        "stage2",
        "E046 pilote : 50 (ancre E044) à 500 (m7, poids papier SRPG) ; échelle log.",
    ),
    Dimension(
        "stage2_perceptual_weight",
        "float",
        3.0,
        22.0,
        "log",
        None,
        2,
        "stage2",
        "E046 pilote : 3 (m7) à 22 (m3) ; échelle log.",
    ),
    Dimension(
        "stage2_initialization",
        "categorical",
        None,
        None,
        None,
        ("paper_stage1_noise", "public_random"),
        0,
        "stage2",
        "Les deux initialisations Stage2 existantes du backend.",
    ),
    Dimension(
        "error_correction",
        "categorical",
        None,
        None,
        None,
        ("M", "Q"),
        0,
        "qr",
        "Version 3 : M (42 octets), Q (32) et H (24) acceptent les payloads "
        "d2 de 23 octets ; le DOE conserve les deux niveaux M/Q demandés.",
    ),
    Dimension(
        "mask_slot",
        "int",
        0,
        7,
        "linear",
        None,
        0,
        "qr",
        "Slot de masque ; le masque réel tourne avec l'ordinal du prompt.",
    ),
)

ANCHOR_CONFIG: dict[str, Any] = {
    "stage1_steps": 40,
    "stage1_guidance_scale": 7.5,
    "stage1_controlnet_scale": 1.35,
    "control_guidance_start": 0.0,
    "control_guidance_end": 1.0,
    "stage2_strength": 1.0,
    "stage2_steps": 40,
    "stage2_controlnet_scale": 1.05,
    "stage2_qr_weight": 50.0,
    "stage2_perceptual_weight": 20.0,
    "stage2_initialization": "public_random",
    "error_correction": "M",
    "mask_slot": 4,
}
ANCHOR_ID = "k00_e044_anchor"
ANCHOR_RATIONALE = (
    "Exact E044/E046 historical parent recipe (m4_e044_anchor): the configuration "
    "that produced the first very good brutalist parent. Kept verbatim as the DOE anchor."
)


def dimension_by_name(name: str) -> Dimension:
    for dimension in DIMENSIONS:
        if dimension.name == name:
            return dimension
    raise KeyError(f"unknown DOE dimension: {name}")


def _deterministic_permutation(size: int, seed: str) -> list[int]:
    """Fisher-Yates driven only by ``Random.random()`` (stable across versions)."""
    generator = random.Random(seed)
    order = list(range(size))
    for index in range(size - 1, 0, -1):
        swap = int(generator.random() * (index + 1))
        order[index], order[swap] = order[swap], order[index]
    return order


def _latin_hypercube(point_count: int, seed: str) -> list[dict[str, float]]:
    columns: dict[str, list[float]] = {}
    for dimension in DIMENSIONS:
        permutation = _deterministic_permutation(point_count, f"{seed}:{dimension.name}")
        columns[dimension.name] = [(cell + 0.5) / point_count for cell in permutation]
    return [
        {dimension.name: columns[dimension.name][index] for dimension in DIMENSIONS}
        for index in range(point_count)
    ]


def _distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return math.sqrt(
        sum((left[dimension.name] - right[dimension.name]) ** 2 for dimension in DIMENSIONS)
    )


def _max_abs_correlation(points: Sequence[Mapping[str, float]]) -> float:
    worst = 0.0
    names = [dimension.name for dimension in DIMENSIONS]
    for left, right in itertools.combinations(names, 2):
        xs = [point[left] for point in points]
        ys = [point[right] for point in points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        if var_x > 0 and var_y > 0:
            worst = max(worst, abs(cov / math.sqrt(var_x * var_y)))
    return worst


def _design_quality(points: Sequence[Mapping[str, float]]) -> tuple[float, float]:
    min_distance = min(_distance(left, right) for left, right in itertools.combinations(points, 2))
    return min_distance, _max_abs_correlation(points)


def _anchor_unit() -> dict[str, float]:
    return {
        dimension.name: dimension.to_unit(ANCHOR_CONFIG[dimension.name]) for dimension in DIMENSIONS
    }


def _select_design(
    *,
    config_count: int = CONFIG_COUNT,
    trials: int = LHS_TRIALS,
    doe_id: str = DOE_ID,
) -> dict[str, Any]:
    anchor = _anchor_unit()
    free_count = config_count - 1
    best: tuple[float, int, list[dict[str, float]], float, float] | None = None
    for trial in range(trials):
        points = _latin_hypercube(free_count, f"{doe_id}:{trial}")
        min_distance, max_corr = _design_quality([anchor, *points])
        score = min_distance - CORRELATION_PENALTY * max_corr
        if best is None or score > best[0]:
            best = (score, trial, points, min_distance, max_corr)
    assert best is not None
    _, trial, points, min_distance, max_corr = best

    # Farthest-point ordering so that any prefix of the design is space-filling.
    ordered: list[dict[str, float]] = []
    chosen: list[dict[str, float]] = [anchor]
    remaining = list(range(free_count))
    while remaining:
        pick = max(
            remaining,
            key=lambda index: (
                min(_distance(points[index], other) for other in chosen),
                -index,
            ),
        )
        ordered.append(points[pick])
        chosen.append(points[pick])
        remaining.remove(pick)
    return {
        "trial": trial,
        "min_pairwise_distance": min_distance,
        "max_abs_correlation": max_corr,
        "unit_points": ordered,
    }


def _config_id(index: int, config: Mapping[str, Any]) -> str:
    if index == 0:
        return ANCHOR_ID
    return (
        f"k{index:02d}_{config['error_correction'].lower()}"
        f"_{'paper' if config['stage2_initialization'] == 'paper_stage1_noise' else 'public'}"
        f"_qw{int(round(float(config['stage2_qr_weight'])))}"
        f"_pw{float(config['stage2_perceptual_weight']):.1f}".replace(".", "p")
    )


def build_doe() -> dict[str, Any]:
    """Return the full deterministic DOE document (pure function)."""
    design = _select_design()
    configs: list[dict[str, Any]] = []
    anchor = {
        "config_index": 0,
        "config_id": ANCHOR_ID,
        "role": "historical_anchor",
        "rationale": ANCHOR_RATIONALE,
        **ANCHOR_CONFIG,
        "unit_point": _anchor_unit(),
    }
    configs.append(anchor)
    for offset, unit_point in enumerate(design["unit_points"], start=1):
        values = {
            dimension.name: dimension.from_unit(unit_point[dimension.name])
            for dimension in DIMENSIONS
        }
        configs.append(
            {
                "config_index": offset,
                "config_id": _config_id(offset, values),
                "role": "lhs_space_filling",
                "rationale": "Deterministic maximin Latin hypercube point.",
                **values,
                "unit_point": {name: round(value, 6) for name, value in unit_point.items()},
            }
        )
    profile_config_ids = {
        profile: [config["config_id"] for config in configs[:count]]
        for profile, count in PROFILE_CONFIG_COUNTS.items()
    }
    document = {
        "schema": DOE_SCHEMA,
        "doe_id": DOE_ID,
        "method": (
            "anchor + maximin Latin hypercube (Random.random Fisher-Yates), "
            "farthest-point ordered"
        ),
        "config_count": CONFIG_COUNT,
        "lhs_trials": LHS_TRIALS,
        "correlation_penalty": CORRELATION_PENALTY,
        "selected_trial": design["trial"],
        "min_pairwise_distance_unit": round(design["min_pairwise_distance"], 6),
        "max_abs_correlation": round(design["max_abs_correlation"], 6),
        "mask_policy": (
            "actual_mask = (mask_slot + mask_rotation) mod 8; "
            f"mask_rotation = (design_row - 5 + {MASK_FAMILY_MULTIPLIER} * (family_index - 1)) mod 8"
        ),
        "parent_seed_policy": (
            f"seed = {PARENT_SEED_BASE} + (prompt_ordinal - 1) * {CONFIG_COUNT} + config_index"
        ),
        "profile_config_counts": PROFILE_CONFIG_COUNTS,
        "profile_config_ids": profile_config_ids,
        "dimensions": [
            {
                **asdict(dimension),
                "choices": list(dimension.choices) if dimension.choices else None,
            }
            for dimension in DIMENSIONS
        ],
        "configs": configs,
    }
    document["doe_sha256"] = stable_hash(
        {key: value for key, value in document.items() if key != "doe_sha256"}
    )
    return document


def config_to_parent_recipe_kwargs(
    config: Mapping[str, Any],
    *,
    qr_mask_pattern: int,
    recipe_id: str,
) -> dict[str, Any]:
    """Map a DOE config to the E046 ``ParentRecipe`` constructor fields."""
    return {
        "id": recipe_id,
        "error_correction": str(config["error_correction"]),
        "qr_mask_pattern": int(qr_mask_pattern),
        "stage1_steps": int(config["stage1_steps"]),
        "stage1_guidance_scale": float(config["stage1_guidance_scale"]),
        "stage1_controlnet_scale": float(config["stage1_controlnet_scale"]),
        "control_guidance_start": float(config["control_guidance_start"]),
        "control_guidance_end": float(config["control_guidance_end"]),
        "stage2_initialization": str(config["stage2_initialization"]),
        "stage2_strength": float(config["stage2_strength"]),
        "stage2_steps": int(config["stage2_steps"]),
        "stage2_controlnet_scale": float(config["stage2_controlnet_scale"]),
        "stage2_qr_weight": float(config["stage2_qr_weight"]),
        "stage2_perceptual_weight": float(config["stage2_perceptual_weight"]),
        "rationale": str(config.get("rationale") or ""),
    }


def prompt_mask_rotation(family_index: int, design_row: int) -> int:
    """Per-prompt mask rotation; 0 for the historical anchor (family 1, row 5)."""
    if family_index < 1 or design_row < 1:
        raise ValueError("family_index and design_row are 1-based")
    return (
        int(design_row) - MASK_ANCHOR_DESIGN_ROW
        + MASK_FAMILY_MULTIPLIER * (int(family_index) - 1)
    ) % 8


def actual_mask(mask_slot: int, mask_rotation: int) -> int:
    return (int(mask_slot) + int(mask_rotation)) % 8


def configs_for_profile(document: Mapping[str, Any], profile: str) -> tuple[dict[str, Any], ...]:
    """Return the deterministic, space-filled prefix assigned to a profile."""
    if profile not in PROFILE_CONFIG_COUNTS:
        raise ValueError(f"unknown E046 large profile: {profile}")
    count = PROFILE_CONFIG_COUNTS[profile]
    configs = tuple(dict(config) for config in document["configs"][:count])
    if len(configs) != count:
        raise RuntimeError(
            f"DOE only contains {len(configs)} configs for {profile} ({count} needed)"
        )
    return configs


def parent_seed(prompt_ordinal: int, config_index: int) -> int:
    """Unique reproducible seed for one independent Stage2 parent."""
    if prompt_ordinal < 1:
        raise ValueError("prompt_ordinal must be positive")
    if not 0 <= config_index < CONFIG_COUNT:
        raise ValueError(f"config_index must be in 0..{CONFIG_COUNT - 1}")
    return PARENT_SEED_BASE + (prompt_ordinal - 1) * CONFIG_COUNT + config_index


def validate_doe(document: Mapping[str, Any]) -> None:
    assert document["schema"] == DOE_SCHEMA
    assert document["doe_id"] == DOE_ID
    assert document["config_count"] == CONFIG_COUNT
    configs = list(document["configs"])
    assert len(configs) == CONFIG_COUNT
    assert [config["config_index"] for config in configs] == list(range(CONFIG_COUNT))
    assert configs[0]["config_id"] == ANCHOR_ID
    for key, value in ANCHOR_CONFIG.items():
        assert configs[0][key] == value, key
    assert len({config["config_id"] for config in configs}) == CONFIG_COUNT
    assert document["profile_config_counts"] == PROFILE_CONFIG_COUNTS
    for profile, count in PROFILE_CONFIG_COUNTS.items():
        assert document["profile_config_ids"][profile] == [
            config["config_id"] for config in configs[:count]
        ]
    for config in configs:
        for dimension in DIMENSIONS:
            value = config[dimension.name]
            if dimension.kind == "categorical":
                assert dimension.choices is not None and value in dimension.choices
            else:
                assert dimension.lower is not None and dimension.upper is not None
                assert dimension.lower - 1e-9 <= float(value) <= dimension.upper + 1e-9, (
                    dimension.name,
                    value,
                )
    eccs = [config["error_correction"] for config in configs]
    assert abs(eccs.count("M") - eccs.count("Q")) <= 1
    inits = [config["stage2_initialization"] for config in configs]
    assert min(inits.count(choice) for choice in ("paper_stage1_noise", "public_random")) >= 4
