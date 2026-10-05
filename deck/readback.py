"""Content read-back for the capstone deck.

Layout checking asks whether the shapes fit. It cannot ask whether the words are
true. That question has bitten this project six times -- a latency table in the
report, eight invented figures in another team's deck, a conformal slide reading
the wrong operating point, a slide quoting 94.70% where every other slide quoted
94.60%, an invented monthly ticket volume, and a hand-typed benchmark table.
So this script does two things the eye is bad at:

  1. scans every text run for figures that are KNOWN to be wrong, so a stale
     number cannot survive a rebuild;
  2. asserts that the figures the deck leads with actually appear, and that
     each one appears with the value the artifact holds -- not merely present.

Run:  python deck/readback.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

from pptx import Presentation

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS = ROOT / "artifacts" / "reports"

# Figures that have been measured and are WRONG. Each is a string that must
# appear nowhere in the deck. Left: why it is wrong.
STALE = {
    "69.9": "kNN-era DistilBERT latency; the measured figure is 105.5 ms",
    "36.3": "pre-correction cascade latency; measured is 33.3 ms",
    "51 ms": "an invented cascade latency from the other team's deck",
    "3.9 ms": "an invented kNN latency; measured is 27.4 ms",
    "48.2 ms": "an invented DistilBERT latency; measured is 105.5 ms",
    "94.62": "an invented DistilBERT macro-F1; measured is 94.07%",
    "94.88": "an invented cascade macro-F1; measured is 94.43%",
    "11.6": "an invented training-time figure in hours; measured is 47.2 min",
    "10.5 s/step": "an invented per-step timing, absent from the synopsis",
    "97.13": "unbacked by any artifact; the dashboard comment calling it stale",
    "93.17": "a margin-sweep cell misread as the agent's end-to-end accuracy",
    "25.0 ms": "pre-correction kNN latency; measured is 27.4 ms",
    "0.58": "an invented McNemar p-value; measured is 0.6061",
    "4.8 ms": "an invented LinearSVC latency; measured is under 0.01 s",
}

# (label, value the artifact holds, substring the deck must contain)
MUST_APPEAR = [
    ("cascade accuracy", "94.60%", "94.60%"),
    ("cascade macro-F1", "94.43%", "94.43%"),
    ("cascade latency", "33.3 ms", "33.3 ms"),
    ("cascade ECE", "0.0170", "0.0170"),
    ("distilbert accuracy", "94.91%", "94.91%"),
    ("distilbert macro-F1", "94.07%", "94.07%"),
    ("distilbert latency", "105.5 ms", "105.5 ms"),
    ("knn accuracy", "92.56%", "92.56%"),
    ("knn latency", "27.4 ms", "27.4 ms"),
    ("McNemar p", "0.6061", "0.6061"),
    ("serving threshold", "0.7635", "0.7635"),
    ("accuracy at gate", "98.24%", "98.24%"),
    ("conformal coverage", "90.32%", "90.32%"),
    ("oracle recall@15", "99.75%", "99.75%"),
    ("tail recall", "90.38%", "90.38%"),
    ("P0 recall", "95.49%", "95.49%"),
    ("test split", "1,963", "1,963"),
    ("train split", "9,158", "9,158"),
]


def slide_text(slide):
    out = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            out.append(shape.text_frame.text)
        if getattr(shape, "has_table", False) and shape.has_table:
            for row in shape.table.rows:
                for cell in row.cells:
                    out.append(cell.text)
    return "\n".join(out)


def artifact(name):
    return json.loads((REPORTS / name).read_text())


def main() -> int:
    deck = ROOT / "DRONA_capstone_deck.pptx"
    prs = Presentation(deck)
    per_slide = [slide_text(s) for s in prs.slides]
    whole = "\n".join(per_slide)

    print(f"deck: {len(prs.slides)} slides, "
          f"{prs.slide_width / 914400:.3f} x {prs.slide_height / 914400:.1f} in\n")

    # ---- 1. stale-figure scan ------------------------------------------
    print("== stale figures (must not appear) ==")
    bad = 0
    for figure, why in STALE.items():
        hits = [i + 1 for i, t in enumerate(per_slide) if figure in t]
        if hits:
            bad += 1
            print(f"  STALE  {figure!r} on slide(s) {hits} -- {why}")
    if not bad:
        print(f"  clean: none of {len(STALE)} known-wrong figures appear")

    # ---- 2. required figures present ------------------------------------
    print("\n== required figures ==")
    missing = 0
    for label, expected, needle in MUST_APPEAR:
        if needle not in whole:
            missing += 1
            print(f"  MISSING  {label}: expected {expected!r} ({needle!r})")
    if not missing:
        print(f"  all {len(MUST_APPEAR)} headline figures present")

    # ---- 3. cross-check against the artifacts --------------------------
    print("\n== cross-check against artifacts ==")
    final = artifact("06-final-comparison.json")
    conf = artifact("08-conformal.json")
    checks = [
        ("cascade accuracy", f"{final['cascade_fused']['accuracy'] * 100:.2f}%", "94.60%"),
        ("distilbert accuracy", f"{final['distilbert_finetuned']['accuracy'] * 100:.2f}%", "94.91%"),
        ("knn accuracy", f"{final['knn_retrieval']['accuracy'] * 100:.2f}%", "92.56%"),
        ("mcnemar p", f"{final['mcnemar_cascade_vs_distilbert']['p_value']:.4f}", "0.6061"),
        ("threshold", f"{conf['empirical_reference']['threshold_at_90pct']:.4f}", "0.7635"),
        ("acc at gate", f"{conf['empirical_reference']['accuracy_on_accepted_at_90pct'] * 100:.2f}%", "98.24%"),
    ]
    wrong = 0
    for label, from_artifact, needle in checks:
        status = "ok " if from_artifact == needle else "DRIFT"
        if from_artifact != needle:
            wrong += 1
        print(f"  {status} {label:22s} artifact={from_artifact:10s} deck={needle}")
    if wrong:
        print("  !! MUST_APPEAR has drifted from the artifacts -- update both")

    # ---- 4. prose claims that must not be asserted ----------------------
    print("\n== overclaims ==")
    over = 0
    for phrase, why in [
        ("significantly better", "the paired test says the opposite"),
        ("state of the art", "unsupported"),
        ("no human in the loop", "the whole design is human-in-the-loop"),
    ]:
        if phrase.lower() in whole.lower():
            over += 1
            print(f"  OVERCLAIM  {phrase!r} -- {why}")
    if not over:
        print("  none of the banned overclaims appear")

    print(f"\nREAD-BACK: {bad} stale, {missing} missing, {wrong} drift, {over} overclaim")
    return 0 if (bad + missing + wrong + over) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
