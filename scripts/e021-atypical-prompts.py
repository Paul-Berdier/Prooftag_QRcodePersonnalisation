"""Launch paired DiffQRCoder campaigns on deliberately atypical unseen prompts."""

from __future__ import annotations

import argparse
import copy
import json
import urllib.request
from pathlib import Path
from typing import Any

PROFILE_IDS = (
    "diffqrcoder_stage1",
    "diffqrcoder_srpg",
    "diffqrcoder_srmpgd",
    "diffqrcoder_srmpgd_robust",
)
NEGATIVE_PROMPT = (
    "easynegative, text, letters, watermark, logo, signature, low quality, "
    "deformed geometry"
)

PROMPTS = (
    {
        "id": "ood01_nautilus_cutaway",
        "text": (
            "A scientific museum cutaway of a single pearlescent nautilus shell, "
            "intricate logarithmic chambers, black velvet background, macro photography."
        ),
        "negative_prompt": NEGATIVE_PROMPT,
        "set": "core",
    },
    {
        "id": "ood02_clockwork_hummingbird",
        "text": (
            "Exploded-view mechanical hummingbird made of brass gears and sapphire "
            "springs, precise technical illustration on aged cyan blueprint paper."
        ),
        "negative_prompt": NEGATIVE_PROMPT,
        "set": "core",
    },
    {
        "id": "ood03_zero_gravity_library",
        "text": (
            "An impossible library in zero gravity, books and spiral staircases floating "
            "through a vast circular room, warm cinematic light, surreal architecture."
        ),
        "negative_prompt": NEGATIVE_PROMPT,
        "set": "core",
    },
    {
        "id": "ood04_woven_archipelago",
        "text": (
            "A handwoven tapestry map of an imaginary volcanic archipelago, thick wool, "
            "indigo ocean currents, coral islands, visible textile fibers, flat composition."
        ),
        "negative_prompt": NEGATIVE_PROMPT,
        "set": "core",
    },
    {
        "id": "ood05_salt_fractals",
        "text": (
            "Top-down aerial photograph of white salt flats split by branching crimson "
            "rivers, delicate natural fractals, immense scale, no horizon."
        ),
        "negative_prompt": NEGATIVE_PROMPT,
        "set": "core",
    },
    {
        "id": "ood06_lunar_greenhouse",
        "text": (
            "Cross-section of a brutalist greenhouse on the Moon, tropical plants behind "
            "thick glass, exposed concrete, astronauts as tiny silhouettes, architectural render."
        ),
        "negative_prompt": NEGATIVE_PROMPT,
        "set": "core",
    },
    {
        "id": "ood07_white_porcelain_snow",
        "text": (
            "White porcelain origami cranes standing in fresh snow under an overcast sky, "
            "high-key monochrome photograph, extremely subtle shadows and low contrast."
        ),
        "negative_prompt": NEGATIVE_PROMPT,
        "set": "stress",
    },
    {
        "id": "ood08_obsidian_cave",
        "text": (
            "A nearly black obsidian cave illuminated by one narrow red laser beam, glossy "
            "facets barely visible in deep shadow, low-key cinematic photography."
        ),
        "negative_prompt": NEGATIVE_PROMPT,
        "set": "stress",
    },
    {
        "id": "ood09_opart_staircase",
        "text": (
            "An impossible Escher-like staircase built from repeating black and ivory arcs, "
            "precise optical-art screen print, hard edges, hypnotic geometry, no lettering."
        ),
        "negative_prompt": NEGATIVE_PROMPT,
        "set": "stress",
    },
    {
        "id": "ood10_iridescent_beetle",
        "text": (
            "Extreme macro portrait of an iridescent jewel beetle, microscopic scales, "
            "oil-slick reflections, emerald and magenta highlights, shallow depth of field."
        ),
        "negative_prompt": NEGATIVE_PROMPT,
        "set": "stress",
    },
    {
        "id": "ood11_fire_dancers_snow",
        "text": (
            "Long-exposure photograph of two fire dancers crossing a silent snowy field at "
            "night, looping amber light trails, blue moonlight, controlled motion blur."
        ),
        "negative_prompt": NEGATIVE_PROMPT,
        "set": "stress",
    },
    {
        "id": "ood12_city_inside_watch",
        "text": (
            "A complete miniature city built inside a transparent broken wristwatch, tiny "
            "streets between gears, glass reflections, dense tilt-shift diorama photography."
        ),
        "negative_prompt": NEGATIVE_PROMPT,
        "set": "stress",
    },
)


def selected_prompts(prompt_set: str) -> list[dict[str, str]]:
    prompts = [item for item in PROMPTS if prompt_set == "all" or item["set"] == prompt_set]
    return [
        {
            "id": item["id"],
            "text": item["text"],
            "negative_prompt": item["negative_prompt"],
        }
        for item in prompts
    ]


def build_manifest(
    schema: dict[str, Any],
    *,
    payload: str,
    prompt_set: str = "core",
    seeds: list[int] | None = None,
    include_reference: bool = False,
) -> dict[str, Any]:
    profiles = {item["id"]: item for item in schema["profiles"]}
    profile_ids = (("qr_reference",) if include_reference else ()) + PROFILE_IDS
    missing = [profile_id for profile_id in profile_ids if profile_id not in profiles]
    if missing:
        raise ValueError(f"Web Lab schema is missing E021 profiles: {missing}")
    methods = [copy.deepcopy(profiles[profile_id]) for profile_id in profile_ids]
    for method in methods:
        method["enabled"] = True
    prompts = selected_prompts(prompt_set)
    if not prompts:
        raise ValueError(f"No prompt found for set {prompt_set!r}")
    return {
        "name": f"E021 — prompts atypiques — {prompt_set}",
        "payload": payload,
        "error_correction": "M",
        "prompts": prompts,
        "seeds": seeds or [51001],
        "methods": methods,
        "max_attempts": 1,
    }


def _json_request(url: str, *, payload: dict[str, Any] | None = None) -> Any:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:18080")
    parser.add_argument("--payload")
    parser.add_argument("--set", choices=("core", "stress", "all"), default="core")
    parser.add_argument("--seeds", default="51001")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--list-prompts", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/e021-atypical-prompts.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_prompts:
        for prompt in selected_prompts(args.set):
            print(f"{prompt['id']} | {prompt['text']}")
        return
    if not args.payload:
        raise ValueError("--payload is required unless --list-prompts is used")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    api = args.api.rstrip("/")
    manifest = build_manifest(
        _json_request(f"{api}/v1/lab/schema"),
        payload=args.payload,
        prompt_set=args.set,
        seeds=seeds,
        include_reference=args.include_reference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trial_count = len(manifest["prompts"]) * len(manifest["seeds"]) * len(
        manifest["methods"]
    )
    print(
        f"Manifest: {args.output} — {len(manifest['prompts'])} prompts, "
        f"{len(manifest['methods'])} méthodes, {trial_count} essais"
    )
    if args.launch:
        campaign = _json_request(f"{api}/v1/lab/campaigns", payload=manifest)
        print(
            f"Campagne lancée: {campaign['id']} — "
            f"{campaign['total_trials']} essais"
        )


if __name__ == "__main__":
    main()
