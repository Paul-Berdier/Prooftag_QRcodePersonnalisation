"""Catalogue canonique de la campagne E046 large advisor dataset.

* 16 familles visuelles × 16 prompts réellement distincts = 256 prompts ;
* chaque prompt porte un payload court unique ``https://ptag.io/d2/NNNN`` ;
* chaque famille suit la même grille structurale de 16 lignes
  (``TAG_DESIGN``), équilibrée sur 8 axes : fréquence spatiale, symétrie,
  luminosité, contraste, colorimétrie, espace négatif, densité de texture,
  composition. La géométrie dominante est portée par la famille.

Le texte des prompts vit dans ``data/e046_prompt_catalog_v2.json`` (versionné,
SHA-256 inclus dans chaque plan). Ce module charge et valide ce fichier ; le
script ``scripts/build_e046_prompt_catalog.py`` le régénère de façon
déterministe depuis ``prooftag_qr/e046_large_prompts.py``.

Le catalogue E046 pilote (``e046_catalog.py``) n'est pas modifié : la nouvelle
expérience réutilise son moteur, pas sa sémantique de plan.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .e046_large_doe import prompt_mask_rotation
from .resilient_experiment import stable_hash

EXPERIMENT = "e046-large-advisor-dataset-v1"
CATALOG_SCHEMA = "e046-prompt-catalog-v2"
CATALOG_VERSION = "v2"
PROMPT_COUNT = 256
FAMILY_COUNT = 16
PROMPTS_PER_FAMILY = 16
PROFILE_PROMPT_COUNTS = {"smoke": 4, "pilot": 32, "full": PROMPT_COUNT}
PAYLOAD_PREFIX = "https://ptag.io/d2/"
QR_VERSION = 3
QR_VERSION_3_BYTE_CAPACITY = {"L": 53, "M": 42, "Q": 32, "H": 24}

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "data" / "e046_prompt_catalog_v2.json"
DOE_PATH = REPO_ROOT / "data" / "e046_large_doe_v1.json"

FORBIDDEN_PROMPT_TERMS = (
    "typography",
    "calligraphy",
    "logo",
    "watermark",
    "signature",
    "caption",
    "brand name",
    "lettering",
    "text",
    "letters",
    "words",
    "signs",
    "labels",
    "numbers",
)
# Ces termes sont tolérés uniquement dans une clause négative finale.
NEGATIVE_CLAUSE_PATTERN = re.compile(
    r"(,\s*)?(no|without)\s+[^,]*?(text|letter|lettering|word|sign|logo|typograph|caption|label|writing|number|watermark|signature)[^,]*$",
    re.IGNORECASE,
)

TAG_AXES: dict[str, tuple[str, ...]] = {
    "spatial_frequency": ("low", "medium", "high"),
    "symmetry": ("symmetric", "asymmetric"),
    "brightness": ("light", "dark", "mixed"),
    "contrast": ("low", "high"),
    "colorfulness": ("muted", "colorful", "monochrome"),
    "negative_space": ("low", "medium", "high"),
    "texture_density": ("fine", "coarse"),
    "composition_type": ("centered", "off_center", "repetitive"),
}
GEOMETRY_TYPES = ("grid", "branching", "organic", "irregular")

# Grille structurale partagée par toutes les familles (design équilibré,
# sélectionné de façon déterministe : voir docs/e046-large-advisor-dataset.md).
TAG_DESIGN: tuple[dict[str, str], ...] = (
    {
        "spatial_frequency": "low",
        "symmetry": "symmetric",
        "brightness": "mixed",
        "contrast": "high",
        "colorfulness": "colorful",
        "negative_space": "high",
        "texture_density": "coarse",
        "composition_type": "off_center",
    },
    {
        "spatial_frequency": "low",
        "symmetry": "asymmetric",
        "brightness": "mixed",
        "contrast": "low",
        "colorfulness": "monochrome",
        "negative_space": "low",
        "texture_density": "coarse",
        "composition_type": "off_center",
    },
    {
        "spatial_frequency": "medium",
        "symmetry": "asymmetric",
        "brightness": "mixed",
        "contrast": "low",
        "colorfulness": "colorful",
        "negative_space": "medium",
        "texture_density": "fine",
        "composition_type": "centered",
    },
    {
        "spatial_frequency": "high",
        "symmetry": "symmetric",
        "brightness": "light",
        "contrast": "high",
        "colorfulness": "monochrome",
        "negative_space": "high",
        "texture_density": "coarse",
        "composition_type": "centered",
    },
    {
        "spatial_frequency": "medium",
        "symmetry": "symmetric",
        "brightness": "mixed",
        "contrast": "low",
        "colorfulness": "muted",
        "negative_space": "high",
        "texture_density": "fine",
        "composition_type": "centered",
    },
    {
        "spatial_frequency": "high",
        "symmetry": "asymmetric",
        "brightness": "light",
        "contrast": "high",
        "colorfulness": "muted",
        "negative_space": "low",
        "texture_density": "coarse",
        "composition_type": "off_center",
    },
    {
        "spatial_frequency": "low",
        "symmetry": "asymmetric",
        "brightness": "light",
        "contrast": "low",
        "colorfulness": "muted",
        "negative_space": "low",
        "texture_density": "fine",
        "composition_type": "repetitive",
    },
    {
        "spatial_frequency": "low",
        "symmetry": "symmetric",
        "brightness": "light",
        "contrast": "high",
        "colorfulness": "colorful",
        "negative_space": "medium",
        "texture_density": "fine",
        "composition_type": "repetitive",
    },
    {
        "spatial_frequency": "medium",
        "symmetry": "symmetric",
        "brightness": "dark",
        "contrast": "low",
        "colorfulness": "muted",
        "negative_space": "low",
        "texture_density": "fine",
        "composition_type": "off_center",
    },
    {
        "spatial_frequency": "low",
        "symmetry": "asymmetric",
        "brightness": "dark",
        "contrast": "low",
        "colorfulness": "colorful",
        "negative_space": "low",
        "texture_density": "coarse",
        "composition_type": "centered",
    },
    {
        "spatial_frequency": "low",
        "symmetry": "symmetric",
        "brightness": "dark",
        "contrast": "high",
        "colorfulness": "muted",
        "negative_space": "medium",
        "texture_density": "fine",
        "composition_type": "centered",
    },
    {
        "spatial_frequency": "medium",
        "symmetry": "asymmetric",
        "brightness": "dark",
        "contrast": "high",
        "colorfulness": "monochrome",
        "negative_space": "high",
        "texture_density": "coarse",
        "composition_type": "repetitive",
    },
    {
        "spatial_frequency": "high",
        "symmetry": "symmetric",
        "brightness": "dark",
        "contrast": "low",
        "colorfulness": "colorful",
        "negative_space": "medium",
        "texture_density": "coarse",
        "composition_type": "repetitive",
    },
    {
        "spatial_frequency": "high",
        "symmetry": "asymmetric",
        "brightness": "mixed",
        "contrast": "high",
        "colorfulness": "monochrome",
        "negative_space": "medium",
        "texture_density": "fine",
        "composition_type": "centered",
    },
    {
        "spatial_frequency": "high",
        "symmetry": "symmetric",
        "brightness": "light",
        "contrast": "low",
        "colorfulness": "monochrome",
        "negative_space": "low",
        "texture_density": "fine",
        "composition_type": "off_center",
    },
    {
        "spatial_frequency": "medium",
        "symmetry": "asymmetric",
        "brightness": "light",
        "contrast": "high",
        "colorfulness": "muted",
        "negative_space": "high",
        "texture_density": "coarse",
        "composition_type": "repetitive",
    },
)


@dataclass(frozen=True, slots=True)
class FamilySpec:
    family: str
    geometry_type: str
    brief: str


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        "architectural_grid",
        "grid",
        "Built architecture with repeated openings and structural rhythm.",
    ),
    FamilySpec(
        "industrial_mechanical",
        "grid",
        "Machinery and engineered systems: gears, pipes, boards, scaffolding.",
    ),
    FamilySpec(
        "geometric_ornamental",
        "grid",
        "Designed ornament: tiles, mosaics, lattices, stained glass, tessellations.",
    ),
    FamilySpec(
        "textile_pattern", "grid", "Fabric and fibre: weaves, knits, quilts, embroidery, basketry."
    ),
    FamilySpec(
        "botanical_rows",
        "grid",
        "Cultivated plant order: orchards, vineyards, terraces, greenhouse benches.",
    ),
    FamilySpec(
        "organic_branching",
        "branching",
        "Natural branching networks: trees, roots, mycelium, deltas, dendrites.",
    ),
    FamilySpec(
        "underwater_organic",
        "branching",
        "Marine life and water: reefs, kelp, sea fans, jellyfish, tide pools.",
    ),
    FamilySpec(
        "crystal_mineral",
        "branching",
        "Mineral and crystalline matter: quartz, geodes, ice, salt, agate.",
    ),
    FamilySpec(
        "minimal_product",
        "organic",
        "Product and still-life photography with one or a few objects.",
    ),
    FamilySpec(
        "glass_translucent",
        "organic",
        "Transparent and translucent materials: glass, bubbles, ice, prisms.",
    ),
    FamilySpec(
        "celestial_space",
        "organic",
        "Sky and cosmos: nebulae, planets, moons, auroras, cloudscapes.",
    ),
    FamilySpec(
        "landscape_negative_space",
        "irregular",
        "Open landscapes with large calm regions: deserts, fog, snow, seas.",
    ),
    FamilySpec(
        "macro_material",
        "irregular",
        "Extreme close-ups of matter and surfaces: rust, bark, foam, moss.",
    ),
    FamilySpec(
        "urban_night",
        "irregular",
        "Cities after dark: neon streets, light trails, skylines, platforms.",
    ),
    FamilySpec(
        "abstract_flow",
        "irregular",
        "Non-figurative fluid and generative imagery: ink, marbling, smoke.",
    ),
    FamilySpec(
        "editorial_illustration",
        "irregular",
        "Hand-made and digital illustration styles rather than photography.",
    ),
)

ANCHOR_PROMPT_TEXT = (
    "a monumental brutalist courtyard with repeated square windows, frontal "
    "symmetrical architectural photograph, soft overcast daylight, clean "
    "concrete rhythm, no lettering"
)


@dataclass(frozen=True, slots=True)
class PromptSpec:
    prompt_id: str
    ordinal: int
    family: str
    family_index: int
    design_row: int
    slug: str
    subject: str
    medium: str
    prompt: str
    payload: str
    payload_length: int
    spatial_frequency: str
    symmetry: str
    brightness: str
    contrast: str
    colorfulness: str
    negative_space: str
    texture_density: str
    geometry_type: str
    composition_type: str
    mask_rotation: int
    is_historical_anchor: bool

    @property
    def tags(self) -> dict[str, str]:
        return {axis: getattr(self, axis) for axis in TAG_AXES}


def family_by_name(name: str) -> FamilySpec:
    for family in FAMILIES:
        if family.family == name:
            return family
    raise KeyError(f"unknown E046 large family: {name}")


def payload_for_ordinal(ordinal: int) -> str:
    if not 1 <= ordinal <= PROMPT_COUNT:
        raise ValueError(f"ordinal out of range: {ordinal}")
    return f"{PAYLOAD_PREFIX}{ordinal:04d}"


def payload_fits(payload: str, error_correction: str, version: int = QR_VERSION) -> bool:
    if version != QR_VERSION:
        raise ValueError("only QR version 3 capacities are tabulated here")
    return len(payload.encode("utf-8")) <= QR_VERSION_3_BYTE_CAPACITY[error_correction]


def prompt_id_for(ordinal: int, slug: str) -> str:
    return f"d2p{ordinal:04d}_{slug}"


def _word_count(text: str) -> int:
    return len([token for token in re.split(r"\s+", text.strip()) if token])


def prompt_text_violations(prompt: str) -> list[str]:
    """Return human-readable violations of the no-text/no-logo prompt policy."""
    violations: list[str] = []
    lowered = prompt.lower()
    match = NEGATIVE_CLAUSE_PATTERN.search(prompt)
    body = prompt[: match.start()] if match else prompt
    body_lower = body.lower()
    if match is None:
        violations.append("missing closing negative clause (e.g. 'no lettering')")
    for term in FORBIDDEN_PROMPT_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", body_lower):
            violations.append(f"forbidden term outside negative clause: {term}")
    if "\n" in prompt or '"' in prompt:
        violations.append("prompt must be a single line without double quotes")
    count = _word_count(prompt)
    if not 14 <= count <= 48:
        violations.append(f"word count {count} outside 14..48")
    if lowered != lowered.encode("ascii", "ignore").decode("ascii"):
        violations.append("prompt must be ASCII")
    return violations


def _normalized_prompt(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def prompt_similarity(left: str, right: str) -> float:
    """Jaccard similarity over content words; used to detect manifest duplicates."""
    stop = {
        "a",
        "an",
        "the",
        "of",
        "and",
        "with",
        "in",
        "on",
        "no",
        "at",
        "to",
        "from",
        "by",
        "or",
        "its",
        "into",
        "over",
        "under",
        "for",
        "photograph",
        "photography",
        "text",
        "lettering",
        "words",
        "signs",
        "logos",
        "soft",
        "light",
        "clean",
    }
    tokens_left = {token for token in _normalized_prompt(left).split() if token not in stop}
    tokens_right = {token for token in _normalized_prompt(right).split() if token not in stop}
    if not tokens_left or not tokens_right:
        return 0.0
    return len(tokens_left & tokens_right) / len(tokens_left | tokens_right)


def build_prompt_specs(
    family_prompts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[PromptSpec, ...]:
    """Assemble 256 PromptSpec from per-family drafted prompts.

    ``family_prompts[family]`` is a sequence of 16 dicts with keys
    ``design_row``, ``slug``, ``subject``, ``medium``, ``prompt``. Within a family
    the prompts are ordered by design_row, except that the historical anchor
    prompt is placed first so that the global ordinal 1 (payload d2/0001,
    mask offset 0) reproduces the E044/E046 brutalist parent configuration.
    """
    specs: list[PromptSpec] = []
    ordinal = 0
    for family_index, family in enumerate(FAMILIES, start=1):
        drafted = list(family_prompts[family.family])
        if len(drafted) != PROMPTS_PER_FAMILY:
            raise ValueError(f"{family.family}: expected 16 prompts, got {len(drafted)}")
        drafted.sort(key=lambda item: int(item["design_row"]))
        anchor_items = [
            item for item in drafted if str(item["prompt"]).strip() == ANCHOR_PROMPT_TEXT
        ]
        if anchor_items:
            drafted = anchor_items + [item for item in drafted if item not in anchor_items]
        for item in drafted:
            ordinal += 1
            row = int(item["design_row"])
            if not 1 <= row <= PROMPTS_PER_FAMILY:
                raise ValueError(f"{family.family}: invalid design_row {row}")
            tags = TAG_DESIGN[row - 1]
            prompt = " ".join(str(item["prompt"]).split())
            slug = str(item["slug"]).strip().lower()
            if not re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", slug):
                raise ValueError(f"{family.family}: invalid slug {slug!r}")
            payload = payload_for_ordinal(ordinal)
            specs.append(
                PromptSpec(
                    prompt_id=prompt_id_for(ordinal, slug),
                    ordinal=ordinal,
                    family=family.family,
                    family_index=family_index,
                    design_row=row,
                    slug=slug,
                    subject=" ".join(str(item["subject"]).split()),
                    medium=str(item["medium"]),
                    prompt=prompt,
                    payload=payload,
                    payload_length=len(payload),
                    geometry_type=family.geometry_type,
                    mask_rotation=prompt_mask_rotation(family_index, row),
                    is_historical_anchor=prompt == ANCHOR_PROMPT_TEXT,
                    **tags,
                )
            )
    return tuple(specs)


def catalog_document(prompts: Sequence[PromptSpec]) -> dict[str, Any]:
    profile_prompt_ids = {
        profile: [item.prompt_id for item in prompts_for_profile(prompts, profile)]
        for profile in PROFILE_PROMPT_COUNTS
    }
    document = {
        "schema": CATALOG_SCHEMA,
        "experiment": EXPERIMENT,
        "catalog_version": CATALOG_VERSION,
        "prompt_count": len(prompts),
        "family_count": len(FAMILIES),
        "prompts_per_family": PROMPTS_PER_FAMILY,
        "payload_prefix": PAYLOAD_PREFIX,
        "qr_version": QR_VERSION,
        "qr_version_3_byte_capacity": QR_VERSION_3_BYTE_CAPACITY,
        "tag_axes": {axis: list(levels) for axis, levels in TAG_AXES.items()},
        "geometry_types": list(GEOMETRY_TYPES),
        "tag_design": [dict(row) for row in TAG_DESIGN],
        "families": [asdict(family) for family in FAMILIES],
        "anchor_prompt_text": ANCHOR_PROMPT_TEXT,
        "profile_prompt_counts": PROFILE_PROMPT_COUNTS,
        "profile_prompt_ids": profile_prompt_ids,
        "prompts": [asdict(prompt) for prompt in prompts],
    }
    document["catalog_sha256"] = stable_hash(
        {key: value for key, value in document.items() if key != "catalog_sha256"}
    )
    return document


def validate_prompts(prompts: Sequence[PromptSpec]) -> None:
    assert len(prompts) == PROMPT_COUNT, len(prompts)
    assert len({item.prompt_id for item in prompts}) == PROMPT_COUNT
    assert len({item.prompt for item in prompts}) == PROMPT_COUNT
    assert len({item.payload for item in prompts}) == PROMPT_COUNT
    assert len({item.slug for item in prompts}) == PROMPT_COUNT, "slugs must be globally unique"
    assert [item.ordinal for item in prompts] == list(range(1, PROMPT_COUNT + 1))
    families = {item.family for item in prompts}
    assert families == {family.family for family in FAMILIES}
    assert len(families) == FAMILY_COUNT
    for family in FAMILIES:
        members = [item for item in prompts if item.family == family.family]
        assert len(members) == PROMPTS_PER_FAMILY, family.family
        assert sorted(item.design_row for item in members) == list(range(1, 17)), family.family
        assert all(item.geometry_type == family.geometry_type for item in members)
    for item in prompts:
        assert item.payload == payload_for_ordinal(item.ordinal)
        assert payload_fits(item.payload, "Q") and payload_fits(item.payload, "M")
        for axis, levels in TAG_AXES.items():
            assert getattr(item, axis) in levels, (item.prompt_id, axis)
        violations = prompt_text_violations(item.prompt)
        assert not violations, (item.prompt_id, violations)
    anchors = [item for item in prompts if item.is_historical_anchor]
    assert len(anchors) == 1 and anchors[0].ordinal == 1, "anchor must be ordinal 1"
    assert anchors[0].family == "architectural_grid"
    # Manifest duplication: compare the hand-curated subject clauses rather than
    # the repeated DOE descriptors.  The latter are intentionally identical for
    # prompts sharing a design row and would otherwise create false positives.
    worst: tuple[float, str, str] | None = None
    for index, left in enumerate(prompts):
        for right in prompts[index + 1 :]:
            similarity = prompt_similarity(left.subject, right.subject)
            if worst is None or similarity > worst[0]:
                worst = (similarity, left.prompt_id, right.prompt_id)
    assert worst is not None and worst[0] < 0.60, f"near-duplicate prompts: {worst}"


def _prompts_from_document(document: Mapping[str, Any]) -> tuple[PromptSpec, ...]:
    return tuple(PromptSpec(**item) for item in document["prompts"])


def catalog_file_sha256(path: Path = CATALOG_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != CATALOG_SCHEMA:
        raise RuntimeError(f"unexpected catalog schema: {document.get('schema')}")
    expected = stable_hash(
        {key: value for key, value in document.items() if key != "catalog_sha256"}
    )
    if document.get("catalog_sha256") != expected:
        raise RuntimeError("E046 large prompt catalog content hash mismatch")
    prompts = _prompts_from_document(document)
    validate_prompts(prompts)
    if document.get("prompt_count") != PROMPT_COUNT:
        raise RuntimeError("E046 large prompt_count metadata mismatch")
    if document.get("family_count") != FAMILY_COUNT:
        raise RuntimeError("E046 large family_count metadata mismatch")
    if document.get("profile_prompt_counts") != PROFILE_PROMPT_COUNTS:
        raise RuntimeError("E046 large profile prompt counts mismatch")
    expected_profiles = {
        profile: [item.prompt_id for item in prompts_for_profile(prompts, profile)]
        for profile in PROFILE_PROMPT_COUNTS
    }
    if document.get("profile_prompt_ids") != expected_profiles:
        raise RuntimeError("E046 large profile prompt selection mismatch")
    return document


def load_prompts(path: Path = CATALOG_PATH) -> tuple[PromptSpec, ...]:
    prompts = _prompts_from_document(load_catalog(path))
    validate_prompts(prompts)
    return prompts


def load_doe(path: Path = DOE_PATH) -> dict[str, Any]:
    from .e046_large_doe import DOE_SCHEMA, validate_doe

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != DOE_SCHEMA:
        raise RuntimeError(f"unexpected DOE schema: {document.get('schema')}")
    expected = stable_hash({key: value for key, value in document.items() if key != "doe_sha256"})
    if document.get("doe_sha256") != expected:
        raise RuntimeError("E046 large DOE content hash mismatch")
    validate_doe(document)
    return document


def doe_file_sha256(path: Path = DOE_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_by_id(prompts: Sequence[PromptSpec], prompt_id: str) -> PromptSpec:
    for prompt in prompts:
        if prompt.prompt_id == prompt_id:
            return prompt
    raise KeyError(f"unknown E046 large prompt: {prompt_id}")


def family_balanced_subset(
    prompts: Sequence[PromptSpec],
    *,
    per_family: int,
) -> tuple[PromptSpec, ...]:
    """Deterministic family-balanced subset: the first ``per_family`` prompts of
    each family in catalogue order (anchor first for architectural_grid)."""
    subset: list[PromptSpec] = []
    for family in FAMILIES:
        members = [item for item in prompts if item.family == family.family]
        subset.extend(members[:per_family])
    return tuple(subset)


# Four scenes spanning the four geometry classes.  Their design-row residues
# (1, 3, 5, 7) are paired with the first two DOE mask slots (4, 5), yielding all
# eight QR masks in the eight-parent smoke profile.
# Smoke always contains the historical brutalist anchor (architectural_grid,
# design row 5 = ordinal 1, mask rotation 0) plus one prompt per remaining
# geometry type. Their mask rotations (0, 4, 6, 2) combined with the two smoke
# DOE mask slots (4, 5) cover all eight QR masks.
SMOKE_PROMPT_KEYS: tuple[tuple[str, int], ...] = (
    ("architectural_grid", 5),
    ("organic_branching", 7),
    ("minimal_product", 3),
    ("landscape_negative_space", 1),
)
# Pilot rows rotate so that architectural_grid receives rows 5 and 13: the
# anchor prompt is therefore part of every profile, and each design row still
# occurs exactly twice across the 16 families.
PILOT_ROW_OFFSETS: tuple[int, int] = (4, 12)


def prompts_for_profile(
    prompts: Sequence[PromptSpec],
    profile: str,
) -> tuple[PromptSpec, ...]:
    """Select a deterministic prompt subset for smoke, pilot, or full.

    Pilot selects two prompts per family.  Across the 16 families every design
    row occurs exactly twice, so it is family-balanced *and* structurally
    balanced rather than merely taking the first two catalogue rows.
    """
    if profile not in PROFILE_PROMPT_COUNTS:
        raise ValueError(f"unknown E046 large profile: {profile}")
    prompt_tuple = tuple(prompts)
    if profile == "full":
        selected = prompt_tuple
    elif profile == "smoke":
        lookup = {(item.family, item.design_row): item for item in prompt_tuple}
        selected = tuple(lookup[key] for key in SMOKE_PROMPT_KEYS)
    else:
        selected_list: list[PromptSpec] = []
        for family_index, family in enumerate(FAMILIES):
            rows = tuple(
                ((family_index + offset) % PROMPTS_PER_FAMILY) + 1
                for offset in PILOT_ROW_OFFSETS
            )
            members = {
                item.design_row: item for item in prompt_tuple if item.family == family.family
            }
            selected_list.extend(members[row] for row in rows)
        selected = tuple(selected_list)
    expected = PROFILE_PROMPT_COUNTS[profile]
    if len(selected) != expected or len({item.prompt_id for item in selected}) != expected:
        raise RuntimeError(f"invalid {profile} prompt subset: {len(selected)}/{expected}")
    return selected


def tag_coverage(prompts: Sequence[PromptSpec]) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {}
    for axis, levels in TAG_AXES.items():
        coverage[axis] = {level: 0 for level in levels}
        for item in prompts:
            coverage[axis][getattr(item, axis)] += 1
    coverage["geometry_type"] = {level: 0 for level in GEOMETRY_TYPES}
    for item in prompts:
        coverage["geometry_type"][item.geometry_type] += 1
    coverage["medium"] = {}
    for item in prompts:
        coverage["medium"][item.medium] = coverage["medium"].get(item.medium, 0) + 1
    return coverage
