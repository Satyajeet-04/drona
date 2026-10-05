#!/usr/bin/env python3
"""Build the ten-slide capstone deck.

    PYTHONPATH=src:deck python deck/build.py

Numbers come from `artifacts/reports/*.json` via `Facts`; nothing is typed in.
If an experiment is re-run the deck moves with it, or this fails to build.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pptx import Presentation
from pptx.util import Inches

import slides10 as S
from framework import Facts

ROOT = HERE.parent
OUT = ROOT / "DRONA_capstone_deck.pptx"

# The order the batch brief mandates.
BUILDERS = [S.s01, S.s02, S.s03, S.s04, S.s05, S.s06, S.s07, S.s08, S.s09, S.s10]

TITLES = [
    "Project Title", "Team Members", "Problem Statement", "Objectives",
    "Literature Survey", "Methodology", "System Architecture", "Project Design",
    "Expected Outcomes", "Future Plan",
]


def main() -> int:
    F = Facts(ROOT)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    for i, (fn, title) in enumerate(zip(BUILDERS, TITLES), 1):
        fn(prs, F)
        print(f"  slide {i:2d}  {title}")

    prs.save(OUT)
    print(f"\n{len(BUILDERS)} slides -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
