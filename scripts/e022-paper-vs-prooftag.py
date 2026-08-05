"""Launch E022: paired Prooftag-safe versus paper-protocol Stage 2."""

from __future__ import annotations

import argparse
import copy
import json
import urllib.request
from pathlib import Path
from typing import Any

PROFILE_IDS = ("diffqrcoder_srpg", "diffqrcoder_paper_srpg")
NEGATIVE_PROMPT = (
    "easynegative, text, letters, watermark, logo, signature, low quality, "
    "deformed geometry, duplicate objects"
)
PROMPTS = (
    (
        "e022_simple_01_blue_vase",
        "simple",
        "A single cobalt blue ceramic vase holding one yellow tulip, centered "
        "on a warm cream background, soft window light, clean still-life photograph.",
    ),
    (
        "e022_simple_02_red_rowboat",
        "simple",
        "A small red wooden rowboat floating alone on a perfectly calm misty lake, "
        "pale mountains in the distance, minimalist landscape photograph.",
    ),
    (
        "e022_simple_03_sleeping_cat",
        "simple",
        "One orange cat curled asleep on a round moss-green cushion, plain light "
        "background, gentle children's-book illustration, simple composition.",
    ),
    (
        "e022_simple_04_lighthouse",
        "simple",
        "A solitary white stone lighthouse beside a calm blue sea at sunrise, "
        "centered composition, clean vintage travel poster, no lettering.",
    ),
    (
        "e022_simple_05_pear",
        "simple",
        "A single ripe golden pear on a dark wooden table, black background, "
        "soft studio light, restrained classical oil painting.",
    ),
    (
        "e022_atypical_01_mycelium_cube",
        "atypical",
        "A transparent glass cube containing a living bioluminescent mycelium "
        "circuit, cyan branching veins and tiny amber spores, dark laboratory, "
        "surreal macro photography.",
    ),
    (
        "e022_atypical_02_ferrofluid_aurora",
        "atypical",
        "A crown of black ferrofluid spikes levitating beneath a miniature aurora, "
        "iridescent magnetic reflections, frozen high-speed macro photograph.",
    ),
    (
        "e022_atypical_03_mobius_opera",
        "atypical",
        "An impossible brutalist opera house folded into a continuous Mobius strip, "
        "tiny spectators on concrete balconies, overcast architectural photography.",
    ),
    (
        "e022_atypical_04_crystal_droplet",
        "atypical",
        "A complete crystalline city refracted inside one suspended water droplet, "
        "prismatic towers, inverted horizon, extreme microscopy, shallow depth of field.",
    ),
    (
        "e022_atypical_05_xray_koi",
        "atypical",
        "An X-ray image of a mechanical koi fish whose bones are clockwork gears, "
        "swimming through translucent lotus roots, cyan radiograph on deep navy.",
    ),
)


def selected_prompts(family: str = "all") -> list[dict[str, str]]:
    return [
        {"id": prompt_id, "text": text, "negative_prompt": NEGATIVE_PROMPT}
        for prompt_id, prompt_family, text in PROMPTS
        if family == "all" or prompt_family == family
    ]


def build_manifest(
    schema: dict[str, Any],
    *,
    payload: str,
    prompt_family: str = "all",
    seeds: list[int] | None = None,
    include_reference: bool = False,
) -> dict[str, Any]:
    profiles = {item["id"]: item for item in schema["profiles"]}
    profile_ids = (("qr_reference",) if include_reference else ()) + PROFILE_IDS
    missing = [profile_id for profile_id in profile_ids if profile_id not in profiles]
    if missing:
        raise ValueError(f"Web Lab schema is missing E022 profiles: {missing}")
    methods = [copy.deepcopy(profiles[profile_id]) for profile_id in profile_ids]
    for method in methods:
        method["enabled"] = True
    prompts = selected_prompts(prompt_family)
    if not prompts:
        raise ValueError(f"No prompt found for family {prompt_family!r}")
    return {
        "name": f"E022 - Prooftag 65% vs protocole PDF - {prompt_family}",
        "payload": payload,
        "error_correction": "M",
        "prompts": prompts,
        "seeds": seeds or [61001],
        "methods": methods,
        "max_attempts": 1,
    }


def _json_request(url: str, payload: dict[str, Any] | None = None) -> Any:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:18080")
    parser.add_argument("--payload")
    parser.add_argument("--family", choices=("simple", "atypical", "all"), default="all")
    parser.add_argument("--seeds", default="61001")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--list-prompts", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/e022-paper-vs-prooftag.json"))
    args = parser.parse_args()
    if args.list_prompts:
        for prompt in selected_prompts(args.family):
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
        prompt_family=args.family,
        seeds=seeds,
        include_reference=args.include_reference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    trials = len(manifest["prompts"]) * len(manifest["seeds"]) * len(manifest["methods"])
    print(f"Manifest: {args.output} - {trials} trials")
    if args.launch:
        campaign = _json_request(f"{api}/v1/lab/campaigns", manifest)
        print(f"Campaign launched: {campaign['id']} - {campaign['total_trials']} trials")


if __name__ == "__main__":
    main()
