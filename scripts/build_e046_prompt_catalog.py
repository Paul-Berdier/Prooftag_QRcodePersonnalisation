#!/usr/bin/env python3
"""Régénère de façon déterministe les artefacts versionnés E046 large :

* ``data/e046_prompt_catalog_v2.json`` depuis ``prooftag_qr/e046_large_prompts.py`` ;
* ``data/e046_large_doe_v1.json`` depuis ``prooftag_qr/e046_large_doe.py``.

Usage :

    python scripts/build_e046_prompt_catalog.py            # écrit les fichiers
    python scripts/build_e046_prompt_catalog.py --check    # échoue si différent

Le mode ``--check`` est utilisé par les tests : même code source ⇒ mêmes octets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prooftag_qr.e046_large_catalog import (  # noqa: E402
    CATALOG_PATH,
    DOE_PATH,
    build_prompt_specs,
    catalog_document,
    tag_coverage,
    validate_prompts,
)
from prooftag_qr.e046_large_doe import build_doe, validate_doe  # noqa: E402
from prooftag_qr.e046_large_prompts import FAMILY_PROMPTS  # noqa: E402


def _dump(document: dict) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    prompts = build_prompt_specs(FAMILY_PROMPTS)
    validate_prompts(prompts)
    catalog = catalog_document(prompts)
    doe = build_doe()
    validate_doe(doe)

    outputs = {
        CATALOG_PATH: _dump(catalog),
        DOE_PATH: _dump(doe),
    }
    changed = []
    for path, text in outputs.items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != text:
            changed.append(path)
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8", newline="\n")

    summary = {
        "prompt_count": len(prompts),
        "catalog_sha256_content": catalog["catalog_sha256"],
        "catalog_sha256_file": hashlib.sha256(outputs[CATALOG_PATH].encode("utf-8")).hexdigest(),
        "doe_sha256_content": doe["doe_sha256"],
        "doe_sha256_file": hashlib.sha256(outputs[DOE_PATH].encode("utf-8")).hexdigest(),
        "tag_coverage": tag_coverage(prompts),
        "changed": [str(path.relative_to(ROOT)) for path in changed],
        "mode": "check" if args.check else "write",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.check and changed:
        print(
            "catalog/DOE artefacts are stale: run scripts/build_e046_prompt_catalog.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
