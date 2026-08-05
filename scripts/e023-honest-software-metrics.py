"""Rerun E022 unchanged with the E023 honest software metrics stack."""

from __future__ import annotations

import argparse
import importlib.util
import json
import urllib.request
from pathlib import Path
from typing import Any


def _e022_module():
    path = Path(__file__).with_name("e022-paper-vs-prooftag.py")
    spec = importlib.util.spec_from_file_location("e022_paired_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def build_manifest(
    schema: dict[str, Any],
    *,
    payload: str,
    prompt_family: str = "all",
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    e022 = _e022_module()
    manifest = e022.build_manifest(
        schema,
        payload=payload,
        prompt_family=prompt_family,
        seeds=seeds or [61001],
        include_reference=False,
    )
    manifest["name"] = (
        f"E023 - mêmes sorties, métriques logicielles honnêtes - {prompt_family}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:18080")
    parser.add_argument("--payload", required=True)
    parser.add_argument(
        "--family", choices=("simple", "atypical", "all"), default="all"
    )
    parser.add_argument("--seeds", default="61001")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/e023-honest-software-metrics.json"),
    )
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    api = args.api.rstrip("/")
    manifest = build_manifest(
        _json_request(f"{api}/v1/lab/schema"),
        payload=args.payload,
        prompt_family=args.family,
        seeds=seeds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trials = len(manifest["prompts"]) * len(seeds) * len(manifest["methods"])
    print(f"Manifest: {args.output} - {trials} essais appariés")
    print("Contrat E023: indice synthétique != taux téléphone; HPS v2.1 != scan.")
    if args.launch:
        campaign = _json_request(f"{api}/v1/lab/campaigns", manifest)
        print(f"Campagne lancée: {campaign['id']} - {campaign['total_trials']} essais")


if __name__ == "__main__":
    main()
