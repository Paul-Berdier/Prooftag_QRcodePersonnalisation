#!/usr/bin/env python3
"""Build the immutable E035 parent contract from an already exported Stage-2 state.

This command never generates Stage 1 or Stage 2. It accepts only an existing PNG and
an existing safetensors latent. The source JSON must contain the provenance fields
required by ``prooftag_qr.e035_parent_artifact``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from safetensors.torch import load_file

from prooftag_qr.e035_parent_artifact import export_parent_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--latent-key", default="latent")
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.latent.suffix != ".safetensors":
        raise ValueError("only safetensors input is accepted; pickle-based .pt is refused")
    source = json.loads(args.source_json.read_text(encoding="utf-8"))
    tensors = load_file(str(args.latent), device="cpu")
    if args.latent_key not in tensors:
        raise KeyError(
            f"latent key {args.latent_key!r} is absent; available keys: {sorted(tensors)}"
        )
    image = Image.open(args.image).convert("RGB")
    metadata = export_parent_artifact(
        args.output_dir,
        latent=tensors[args.latent_key],
        image=image,
        source=source,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
