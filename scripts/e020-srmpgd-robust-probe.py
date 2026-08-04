"""Launch one paired E020 comparison of official and robust SR-MPGD losses."""

from __future__ import annotations

import argparse
import copy
import json
import urllib.request
from pathlib import Path
from typing import Any

PROFILE_IDS = (
    "qr_reference",
    "diffqrcoder_stage1",
    "diffqrcoder_srpg",
    "diffqrcoder_srmpgd",
    "diffqrcoder_srmpgd_robust",
)
DEFAULT_PROMPT = {
    "id": "courtyard",
    "text": (
        "A Mediterranean courtyard with blue tiles, lemon trees and a stone "
        "fountain, warm editorial photograph."
    ),
    "negative_prompt": "easynegative",
}


def build_manifest(
    schema: dict[str, Any],
    *,
    payload: str,
    prompt: dict[str, str] | None = None,
    seed: int = 51001,
) -> dict[str, Any]:
    profiles = {item["id"]: item for item in schema["profiles"]}
    missing = [profile_id for profile_id in PROFILE_IDS if profile_id not in profiles]
    if missing:
        raise ValueError(f"Web Lab schema is missing E020 profiles: {missing}")
    methods = [copy.deepcopy(profiles[profile_id]) for profile_id in PROFILE_IDS]
    for method in methods:
        method["enabled"] = True
    return {
        "name": "E020 — SR-MPGD officiel vs loss robuste",
        "payload": payload,
        "error_correction": "M",
        "prompts": [dict(prompt or DEFAULT_PROMPT)],
        "seeds": [seed],
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
    parser.add_argument("--seed", type=int, default=51001)
    parser.add_argument("--prompt-id", default=DEFAULT_PROMPT["id"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT["text"])
    parser.add_argument("--negative-prompt", default=DEFAULT_PROMPT["negative_prompt"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/e020-srmpgd-robust-probe.json"),
    )
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = args.api.rstrip("/")
    schema = _json_request(f"{api}/v1/lab/schema")
    manifest = build_manifest(
        schema,
        payload=args.payload,
        seed=args.seed,
        prompt={
            "id": args.prompt_id,
            "text": args.prompt,
            "negative_prompt": args.negative_prompt,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest: {args.output} ({len(manifest['methods'])} méthodes)")
    if args.launch:
        campaign = _json_request(f"{api}/v1/lab/campaigns", payload=manifest)
        print(
            f"Campagne lancée: {campaign['id']} — "
            f"{campaign['total_trials']} essais"
        )


if __name__ == "__main__":
    main()
