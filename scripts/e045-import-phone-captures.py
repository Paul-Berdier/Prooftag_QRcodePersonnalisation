#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prooftag_qr.e045_foundation import import_phone_revision


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Importer un lot téléphone E045 dans un dossier immuable."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/e045-foundation-v1"),
    )
    args = parser.parse_args()
    print(json.dumps(
        import_phone_revision(args.input_csv, args.output_root),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
