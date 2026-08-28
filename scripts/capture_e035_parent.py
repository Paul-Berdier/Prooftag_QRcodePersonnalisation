#!/usr/bin/env python3
"""Thin executable wrapper around ``python -m prooftag_qr.e035_parent_capture``."""

from prooftag_qr.e035_parent_capture import _cli


if __name__ == "__main__":
    raise SystemExit(_cli())
