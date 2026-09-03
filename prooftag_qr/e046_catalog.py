"""Catalogue scientifique E046.

E046 construit un dataset neuf et traçable. Il n'essaie pas le produit cartésien
des 98 paramètres recensés par E045 : il couvre un premier plan espace-rempli,
puis conserve assez de provenance pour qu'E047 puisse apprendre une politique.

Le score QR logiciel principal est exclusivement la sortie exacte de
qr-scanner-wechat via le pont qr-verify@0.2.0. Les autres décodeurs ne votent pas
dans le classement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .resilient_experiment import stable_hash

EXPERIMENT = "e046-controlled-best-generator-v1"
CATALOG_SCHEMA = "e046-catalog-v1"
QR_VERIFY_PRESET_COUNT = 37
QR_SOFTWARE_ENGINE = "qr-scanner-wechat via qr-verify@0.2.0"
QR_VERSION = 3
QR_MODULE_SIZE = 20
QR_PADDING_PX = 78
CANVAS_PX = 736

NEGATIVE_PROMPT = (
    "easynegative, low quality, worst quality, blurry, deformed, watermark, "
    "text, letters, logo, signature, oversaturated, clipped highlights, "
    "posterized colors, broken geometry, unreadable symbols"
)


@dataclass(frozen=True, slots=True)
class PromptSpec:
    id: str
    family: str
    payload: str
    variants: tuple[str, str, str]
    quiet_zone_hint: str


@dataclass(frozen=True, slots=True)
class ParentRecipe:
    id: str
    error_correction: Literal["M", "Q", "H"]
    qr_mask_pattern: int
    stage1_steps: int
    stage1_guidance_scale: float
    stage1_controlnet_scale: float
    control_guidance_start: float
    control_guidance_end: float
    stage2_initialization: Literal["paper_stage1_noise", "public_random"]
    stage2_strength: float
    stage2_steps: int
    stage2_controlnet_scale: float
    stage2_qr_weight: float
    stage2_perceptual_weight: float
    rationale: str


@dataclass(frozen=True, slots=True)
class SRMPGDRecipe:
    id: str
    gamma: float
    latent_radius_rms: float
    max_iterations: int
    lpips_weight: float
    lpips_budget: float
    core_mae_budget: float
    full_module_weight: float
    max_backtracks: int
    rationale: str


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    id: str
    prompt_id: str
    prompt_variant_index: int
    parent_recipe_id: str
    replicate_index: int
    seed: int
    payload: str
    prompt: str
    prompt_family: str
    quiet_zone_hint: str


PROMPTS: tuple[PromptSpec, ...] = (
    PromptSpec(
        "p01_brutalist_courtyard",
        "architectural_grid",
        "https://ptag.io/t/e46a",
        (
            "a monumental brutalist courtyard with repeated square windows, frontal symmetrical architectural photograph, soft overcast daylight, clean concrete rhythm, no lettering",
            "a quiet modernist museum atrium formed by a precise grid of concrete openings, balanced frontal composition, diffuse daylight, architectural photography, no text",
            "a geometric concrete cloister with repeated bays and a pale open sky, centered perspective, refined editorial architecture photograph, no signs or words",
        ),
        "pale concrete or open sky around the QR boundary",
    ),
    PromptSpec(
        "p02_bioluminescent_mycelium",
        "branching_high_frequency",
        "https://ptag.io/t/e46b",
        (
            "a transparent glass cube containing a living bioluminescent mycelium network, cyan branching veins and tiny amber spores, dark laboratory, surreal macro photography",
            "an elegant glowing fungal circuit branching inside clear glass, cyan filaments, subtle golden particles, black scientific studio, high-detail macro image",
            "a luminous organic network spreading through a crystal terrarium, fine blue branches and warm spores, cinematic dark botanical laboratory",
        ),
        "soft luminous halo around the QR boundary",
    ),
    PromptSpec(
        "p03_lighthouse_mist",
        "high_contrast_vertical",
        "https://ptag.io/t/e46c",
        (
            "a solitary white stone lighthouse beside a calm blue sea at sunrise, centered composition, misty sky, clean vintage travel poster without lettering",
            "a tall coastal lighthouse rising through pale morning fog, calm water and broad clean sky, centered fine-art landscape, no typography",
            "a white beacon tower on a quiet cliff at dawn, restrained blue and cream palette, large misty negative space, cinematic travel illustration without text",
        ),
        "mist or pale sky around the QR boundary",
    ),
    PromptSpec(
        "p04_cobalt_vase",
        "minimal_still_life",
        "https://ptag.io/t/e46d",
        (
            "a single cobalt blue ceramic vase holding one yellow tulip, centered on a warm cream background, soft window light, clean still-life photograph",
            "one deep-blue porcelain vase with a small golden flower, uncluttered ivory studio backdrop, gentle natural shadow, premium product photography",
            "a sculptural indigo vase and one delicate yellow bloom, calm beige wall and tabletop, minimal editorial still life, no text",
        ),
        "cream wall or tabletop around the QR boundary",
    ),
    PromptSpec(
        "p05_art_deco_lobby",
        "ornamental_symmetry",
        "https://ptag.io/t/e46e",
        (
            "an elegant art deco hotel lobby with symmetrical brass arches, emerald velvet seating and a pale marble floor, centered interior photograph, no signs",
            "a refined 1930s foyer with repeating golden geometry, dark green furniture and luminous cream stone, frontal symmetry, architectural photography",
            "a luxurious geometric entrance hall, brass lines, jade upholstery and a softly glowing marble backdrop, precise centered composition, no lettering",
        ),
        "light marble floor or wall around the QR boundary",
    ),
    PromptSpec(
        "p06_solar_greenhouse",
        "organic_grid",
        "https://ptag.io/t/e46f",
        (
            "a sunlit greenhouse filled with tomato plants, narrow wooden paths and rows of terracotta pots, balanced botanical photograph, natural geometry, soft daylight",
            "an airy glass conservatory with ordered plant beds, climbing green leaves and warm clay pots, central walkway, bright botanical editorial photo",
            "a luminous indoor garden arranged in repeating rows, lush vines, pale glass roof and a calm central path, natural documentary photography",
        ),
        "bright glass or pathway around the QR boundary",
    ),
    PromptSpec(
        "p07_ceramic_teapot",
        "product_still_life",
        "https://ptag.io/t/e46g",
        (
            "a handcrafted white ceramic teapot with a curved walnut handle on a pale linen table, warm side light, refined product photograph, no lettering",
            "a minimalist porcelain tea pot and small cup on natural fabric, soft beige studio background, subtle shadows, premium catalogue photography",
            "an elegant matte ceramic tea set on a quiet sand-colored surface, warm window light, restrained composition, no text or logos",
        ),
        "pale linen or studio wall around the QR boundary",
    ),
    PromptSpec(
        "p08_underwater_corals",
        "organic_branching",
        "https://ptag.io/t/e46h",
        (
            "a tranquil underwater coral garden with branching sea fans, turquoise water and soft sun rays, balanced nature photograph, no fish crowding the center",
            "delicate red and gold sea fans growing through clear blue water, broad luminous background, serene marine documentary image",
            "an ethereal reef of branching corals in a quiet cyan lagoon, gentle shafts of light, detailed but balanced underwater photography",
        ),
        "clear water and light rays around the QR boundary",
    ),
)


PARENT_RECIPES: tuple[ParentRecipe, ...] = (
    ParentRecipe(
        "m0_balanced_public",
        "M", 0, 40, 7.2, 1.30, 0.0, 1.0,
        "public_random", 1.0, 40, 1.05, 65.0, 18.0,
        "Balanced public Stage-2 path with mask 0.",
    ),
    ParentRecipe(
        "m1_paper_noise",
        "M", 1, 40, 7.5, 1.35, 0.0, 1.0,
        "paper_stage1_noise", 1.0, 40, 1.20, 180.0, 8.0,
        "Paper-style Stage-1 latent noising without claiming paper-exact QArt.",
    ),
    ParentRecipe(
        "m2_scan_q",
        "Q", 2, 50, 6.5, 1.55, 0.0, 1.0,
        "paper_stage1_noise", 0.85, 60, 1.25, 220.0, 7.0,
        "Higher ECC and stronger scan guidance.",
    ),
    ParentRecipe(
        "m3_aesthetic_public",
        "M", 3, 30, 8.0, 1.05, 0.05, 0.90,
        "public_random", 0.78, 40, 0.90, 55.0, 22.0,
        "Lower structural pressure and stronger perceptual preservation.",
    ),
    ParentRecipe(
        "m4_e044_anchor",
        "M", 4, 40, 7.5, 1.35, 0.0, 1.0,
        "public_random", 1.0, 40, 1.05, 50.0, 20.0,
        "Exact E044 parent recipe anchor, before any border repainting.",
    ),
    ParentRecipe(
        "m5_scan_mid",
        "M", 5, 45, 7.0, 1.45, 0.0, 0.95,
        "paper_stage1_noise", 0.90, 50, 1.15, 130.0, 10.0,
        "Intermediate scan-oriented recipe.",
    ),
    ParentRecipe(
        "m6_aesthetic_q",
        "Q", 6, 35, 8.2, 1.15, 0.08, 0.92,
        "public_random", 0.82, 45, 0.95, 85.0, 18.0,
        "Higher ECC with moderate aesthetic-oriented guidance.",
    ),
    ParentRecipe(
        "m7_paper_strong",
        "M", 7, 40, 7.5, 1.35, 0.0, 1.0,
        "paper_stage1_noise", 1.0, 40, 1.35, 500.0, 3.0,
        "Near the paper-reported SRPG weights, with binary exact target.",
    ),
)


SRMPGD_RECIPES: tuple[SRMPGDRecipe, ...] = (
    SRMPGDRecipe(
        "g250_r100_i04",
        250.0, 0.100, 4, 0.01, 0.040, 0.040, 0.10, 10,
        "Conservative low-gamma trust region.",
    ),
    SRMPGDRecipe(
        "g500_r200_i08",
        500.0, 0.200, 8, 0.01, 0.050, 0.050, 0.10, 12,
        "Best software frontier around E044, now without quiet-zone replacement.",
    ),
    SRMPGDRecipe(
        "g1000_r150_i08",
        1000.0, 0.150, 8, 0.01, 0.050, 0.050, 0.10, 12,
        "High raw gamma constrained by projection and backtracking.",
    ),
    SRMPGDRecipe(
        "g500_r150_i08",
        500.0, 0.150, 8, 0.02, 0.040, 0.040, 0.12, 12,
        "Full-profile extra branch with tighter visual preservation.",
    ),
)


PROFILE_SPECS: dict[str, dict[str, Any]] = {
    # Two prompts × three parent recipes. This validates the tournament logic.
    "smoke": {
        "prompt_variant_indices": (0,),
        "prompt_limit": 2,
        "parent_recipe_ids": (
            "m0_balanced_public",
            "m2_scan_q",
            "m3_aesthetic_public",
        ),
        "replicates_per_recipe": 1,
        "selected_parents_per_prompt": 1,
        "srmpgd_recipe_ids": ("g250_r100_i04",),
    },
    # Eight prompts × six genuinely different Stage1/Stage2 recipes. Two
    # candidates per prompt then receive three SR-MPGD trajectories each.
    "pilot": {
        "prompt_variant_indices": (0,),
        "prompt_limit": 8,
        "parent_recipe_ids": (
            "m0_balanced_public",
            "m1_paper_noise",
            "m2_scan_q",
            "m3_aesthetic_public",
            "m5_scan_mid",
            "m6_aesthetic_q",
        ),
        "replicates_per_recipe": 1,
        "selected_parents_per_prompt": 2,
        "srmpgd_recipe_ids": (
            "g250_r100_i04",
            "g500_r200_i08",
            "g1000_r150_i08",
        ),
    },
    # Eight prompts × all eight masks/recipes × two deterministic seeds.
    # This is 128 parents followed by 64 independent SR-MPGD trajectories.
    "full": {
        "prompt_variant_indices": (0,),
        "prompt_limit": 8,
        "parent_recipe_ids": tuple(recipe.id for recipe in PARENT_RECIPES),
        "replicates_per_recipe": 2,
        "selected_parents_per_prompt": 2,
        "srmpgd_recipe_ids": tuple(recipe.id for recipe in SRMPGD_RECIPES),
    },
}


def prompt_by_id(prompt_id: str) -> PromptSpec:
    prompt = next((item for item in PROMPTS if item.id == prompt_id), None)
    if prompt is None:
        raise KeyError(f"unknown E046 prompt: {prompt_id}")
    return prompt


def parent_recipe_by_id(recipe_id: str) -> ParentRecipe:
    recipe = next((item for item in PARENT_RECIPES if item.id == recipe_id), None)
    if recipe is None:
        raise KeyError(f"unknown E046 parent recipe: {recipe_id}")
    return recipe


def srmpgd_recipe_by_id(recipe_id: str) -> SRMPGDRecipe:
    recipe = next((item for item in SRMPGD_RECIPES if item.id == recipe_id), None)
    if recipe is None:
        raise KeyError(f"unknown E046 SR-MPGD recipe: {recipe_id}")
    return recipe


def build_candidates(profile: str) -> tuple[CandidateSpec, ...]:
    """Create several independent parent generations for every prompt."""
    if profile not in PROFILE_SPECS:
        raise ValueError(f"unknown E046 profile: {profile}")

    spec = PROFILE_SPECS[profile]
    candidates: list[CandidateSpec] = []
    prompts = PROMPTS[: int(spec["prompt_limit"])]
    recipe_ids = tuple(str(value) for value in spec["parent_recipe_ids"])
    replicates = int(spec["replicates_per_recipe"])

    for prompt_index, prompt in enumerate(prompts):
        for variant_index in spec["prompt_variant_indices"]:
            for recipe_position, recipe_id in enumerate(recipe_ids):
                recipe = parent_recipe_by_id(recipe_id)
                for replicate_index in range(replicates):
                    seed = (
                        72_046
                        + prompt_index * 100_003
                        + int(variant_index) * 10_007
                        + recipe_position * 1_009
                        + replicate_index * 101
                    )
                    candidate_id = (
                        f"c{len(candidates) + 1:04d}_{prompt.id}_"
                        f"v{int(variant_index)}_{recipe.id}_s{replicate_index}"
                    )
                    candidates.append(
                        CandidateSpec(
                            id=candidate_id,
                            prompt_id=prompt.id,
                            prompt_variant_index=int(variant_index),
                            parent_recipe_id=recipe.id,
                            replicate_index=replicate_index,
                            seed=seed,
                            payload=prompt.payload,
                            prompt=prompt.variants[int(variant_index)],
                            prompt_family=prompt.family,
                            quiet_zone_hint=prompt.quiet_zone_hint,
                        )
                    )
    return tuple(candidates)


def scientific_plan(
    *,
    profile: str,
    source_commit: str,
    e045_plan_id: str,
    e045_manifest_sha256: str,
) -> dict[str, Any]:
    candidates = build_candidates(profile)
    profile_spec = PROFILE_SPECS[profile]
    selected_srmpgd = [
        asdict(srmpgd_recipe_by_id(recipe_id))
        for recipe_id in profile_spec["srmpgd_recipe_ids"]
    ]
    plan = {
        "schema": CATALOG_SCHEMA,
        "experiment": EXPERIMENT,
        "profile": profile,
        "source_commit": source_commit,
        "e045_plan_id": e045_plan_id,
        "e045_manifest_sha256": e045_manifest_sha256,
        "qr_software_engine": QR_SOFTWARE_ENGINE,
        "qr_verify_preset_count": QR_VERIFY_PRESET_COUNT,
        "qr_primary_label": "exact payload only",
        "phone_truth_available": False,
        "candidates": [asdict(candidate) for candidate in candidates],
        "parent_recipes": [asdict(recipe) for recipe in PARENT_RECIPES],
        "srmpgd_recipes": selected_srmpgd,
        "expected_parent_count": len(candidates),
        "expected_prompt_count": (
            int(profile_spec["prompt_limit"])
            * len(tuple(profile_spec["prompt_variant_indices"]))
        ),
        "selected_parents_per_prompt": int(
            profile_spec["selected_parents_per_prompt"]
        ),
        "selected_parent_count": (
            int(profile_spec["selected_parents_per_prompt"])
            * int(profile_spec["prompt_limit"])
            * len(tuple(profile_spec["prompt_variant_indices"]))
        ),
        "expected_refinement_count": (
            int(profile_spec["selected_parents_per_prompt"])
            * int(profile_spec["prompt_limit"])
            * len(tuple(profile_spec["prompt_variant_indices"]))
            * len(selected_srmpgd)
        ),
        "validity_policy": {
            "final_original_exact_required": True,
            "final_minimum_exact_presets": 34,
            "ideal_minimum_exact_presets": 37,
            "refinement_minimum_exact_presets": 16,
            "meaning": (
                "WeChat is a hard software-validity gate. Prompt alignment and "
                "aesthetic scores choose the best-looking candidate inside each "
                "validity tier; beauty can never compensate for an invalid QR."
            ),
        },
        "multiobjective_policy": {
            "within_prompt_normalization": "min-max; neutral 0.5 for constants",
            "weights": {
                "wechat_robustness": 0.40,
                "clip_prompt_alignment": 0.25,
                "hps_human_preference": 0.20,
                "clip_aesthetic": 0.15,
            },
            "module_error_rate": "ascending tie-breaker and diagnostic",
            "stage1_final_eligible": False,
            "scene_preserving_final_eligible": False,
            "raw_stage2_and_raw_srmpgd_only": True,
        },
        "geometry": {
            "qr_version": QR_VERSION,
            "qr_module_size": QR_MODULE_SIZE,
            "qr_padding_px": QR_PADDING_PX,
            "canvas_px": CANVAS_PX,
            "locked_reason": (
                "E046 pilot preserves the exact 29x29 × 20 px DiffQRCoder core; "
                "version/module-size/canvas expansion remains a later controlled axis."
            ),
        },
        "quiet_zone_policy": {
            "raw_artwork_always_saved": True,
            "flat_white_or_uniform_replacement_eligible": False,
            "scene_preserving_variant": True,
            "core_must_remain_byte_identical": True,
            "no_crop": True,
        },
        "objective_priority": [
            "visual_guard_pass",
            "software_validity_tier",
            "multiobjective_prompt_score",
            "wechat_exact_presets",
            "clip_prompt_alignment",
            "hpsv2_1",
            "clip_aesthetic",
            "module_error_rate",
        ],
        "other_decoders_role": "diagnostic_only_not_in_objective",
        "production_ready": False,
    }
    plan["scientific_plan_hash"] = stable_hash(plan)
    plan["plan_id"] = plan["scientific_plan_hash"][:16]
    return plan


def catalog_document() -> dict[str, Any]:
    return {
        "schema": CATALOG_SCHEMA,
        "experiment": EXPERIMENT,
        "prompts": [asdict(item) for item in PROMPTS],
        "parent_recipes": [asdict(item) for item in PARENT_RECIPES],
        "srmpgd_recipes": [asdict(item) for item in SRMPGD_RECIPES],
        "profiles": PROFILE_SPECS,
        "qr_software_engine": QR_SOFTWARE_ENGINE,
    }


def validate_catalog() -> None:
    assert len(PROMPTS) == 8
    assert {recipe.qr_mask_pattern for recipe in PARENT_RECIPES} == set(range(8))
    assert all(recipe.error_correction in {"M", "Q", "H"} for recipe in PARENT_RECIPES)
    assert all(len(prompt.variants) == 3 for prompt in PROMPTS)
    assert all(prompt.payload.startswith("https://ptag.io/") for prompt in PROMPTS)
    assert set(PROFILE_SPECS) == {"smoke", "pilot", "full"}
    for profile, spec in PROFILE_SPECS.items():
        assert int(spec["selected_parents_per_prompt"]) >= 1
        assert int(spec["replicates_per_recipe"]) >= 1
        assert len(spec["parent_recipe_ids"]) >= 1
        assert (
            int(spec["selected_parents_per_prompt"])
            * int(spec["prompt_limit"])
            <= len(build_candidates(profile))
        )
        for parent_recipe_id in spec["parent_recipe_ids"]:
            parent_recipe_by_id(parent_recipe_id)
        for recipe_id in spec["srmpgd_recipe_ids"]:
            srmpgd_recipe_by_id(recipe_id)


validate_catalog()
