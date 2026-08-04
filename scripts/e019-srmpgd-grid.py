"""Build or launch one safe, exactly paired E019 SR-MPGD factorial batch."""

from __future__ import annotations

import argparse
import copy
import json
import urllib.request
from pathlib import Path
from typing import Any

GAMMAS = (10.0, 30.0, 100.0, 300.0, 1000.0)
ITERATIONS = (1, 2, 4, 8, 20)
LPIPS_WEIGHTS = (0.01, 0.05, 0.10, 0.25)
DEFAULT_PROMPTS = (
    {
        "id": "courtyard",
        "text": (
            "A Mediterranean courtyard with blue tiles, lemon trees and a stone "
            "fountain, warm editorial photograph."
        ),
        "negative_prompt": "easynegative",
    },
    {
        "id": "aquarium",
        "text": (
            "A monumental blue aquarium filled with sharks and rays, cinematic "
            "underwater light."
        ),
        "negative_prompt": "easynegative",
    },
    {
        "id": "station",
        "text": (
            "A grand retro-futurist railway station, clocks, glass arches and "
            "travelers."
        ),
        "negative_prompt": "easynegative",
    },
    {
        "id": "botanical",
        "text": (
            "Delicate botanical engraving with white flowers, cyan leaves and dark ink."
        ),
        "negative_prompt": "easynegative",
    },
)


def _token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def build_manifest(
    schema: dict[str, Any],
    *,
    payload: str,
    gamma: float,
    seeds: list[int],
    prompts: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    profiles = {item["id"]: item for item in schema["profiles"]}
    required = {
        "qr_reference",
        "diffqrcoder_stage1",
        "diffqrcoder_srpg",
        "diffqrcoder_srmpgd",
    }
    missing = sorted(required - profiles.keys())
    if missing:
        raise ValueError(f"Web Lab schema is missing E019 profiles: {missing}")
    methods = [
        copy.deepcopy(profiles["qr_reference"]),
        copy.deepcopy(profiles["diffqrcoder_stage1"]),
        copy.deepcopy(profiles["diffqrcoder_srpg"]),
    ]
    base = profiles["diffqrcoder_srmpgd"]
    for iterations in ITERATIONS:
        for lpips_weight in LPIPS_WEIGHTS:
            method = copy.deepcopy(base)
            method["id"] = (
                f"e019_g{int(gamma):04d}_i{iterations:02d}_"
                f"l{_token(lpips_weight)}"
            )
            method["name"] = (
                f"E019 sûr — γ {gamma:g} · {iterations} it. · "
                f"LPIPS {lpips_weight:.2f}"
            )
            method["description"] = (
                "Même latent SRPG, avec bornes E019 ; seul le triplet "
                "gamma/itérations/LPIPS change."
            )
            method["enabled"] = True
            settings = method["tools"]["settings"]
            settings["srmpgd_max_iterations"] = iterations
            settings["srmpgd_step_size"] = gamma
            settings["srmpgd_lpips_weight"] = lpips_weight
            methods.append(method)
    return {
        "name": f"E019 SR-MPGD factoriel — gamma {gamma:g}",
        "payload": payload,
        "error_correction": "M",
        "prompts": prompts or [dict(item) for item in DEFAULT_PROMPTS],
        "seeds": seeds,
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
    parser.add_argument("--payload", required=True)
    parser.add_argument("--seeds", default="51001")
    parser.add_argument("--output", type=Path, default=Path("output/e019-grid"))
    parser.add_argument(
        "--launch",
        type=float,
        choices=GAMMAS,
        help="Launch only this gamma batch. Without it, write all five manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    schema = _json_request(f"{args.api.rstrip('/')}/v1/lab/schema")
    args.output.mkdir(parents=True, exist_ok=True)
    gammas = (args.launch,) if args.launch is not None else GAMMAS
    for gamma in gammas:
        manifest = build_manifest(
            schema,
            payload=args.payload,
            gamma=gamma,
            seeds=seeds,
        )
        path = args.output / f"e019-gamma-{int(gamma):04d}.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Manifest: {path} ({len(manifest['methods'])} méthodes)")
        if args.launch is not None:
            campaign = _json_request(
                f"{args.api.rstrip('/')}/v1/lab/campaigns",
                payload=manifest,
            )
            print(f"Campagne lancée: {campaign['id']} — {campaign['total_trials']} essais")


if __name__ == "__main__":
    main()
