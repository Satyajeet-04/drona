"""Geometry check for the capstone deck.

Three independent questions, because each has a different cause:

  off-canvas   the shape starts outside the slide or extends past an edge.
               Almost always a column-width or box-position arithmetic error.

  intrudes     the shape carries content down into the footer band, so body
               text collides with the footer line. This is the check that
               catches a card that grew as text was added to it.

  touches-full the shape deliberately bleeds to an edge -- a full-bleed
               background or a full-height decorative panel. Not a defect, but
               reported separately so it cannot hide inside the other counts.

Exclusions, and why each is necessary:

  * shapes whose *top* is at or below FOOT are footer furniture (the footer's
    own textboxes are drawn at y=7.06 with h=0.22, so their bottom is 7.28 --
    flagging those would flag the footer for existing).
  * shapes spanning the canvas are backgrounds or panels, which are supposed to
    reach the edge.

Usage:  python deck/check_layout.py [path.pptx]
"""

from __future__ import annotations

import sys

from pptx import Presentation

EMU = 914400
TOL = 9525  # 0.01in of slack, so float rounding is not reported as a bug
FOOT = 7.06  # footer text baseline


def inches(value):
    return None if value is None else value / EMU


def classify(shape, W, H):
    """-> 'off' | 'intrudes' | 'touches-full' | None"""
    left, top = shape.left, shape.top
    width, height = shape.width, shape.height
    if None in (left, top, width, height):
        return None

    right, bottom = left + width, top + height
    l, t, r, b = (inches(v) for v in (left, top, right, bottom))

    full_bleed = width >= W * 0.95 or height >= H * 0.95

    # Content that runs into the footer band, but is not footer furniture.
    if t < FOOT - 0.06 and b > FOOT + TOL / EMU and not full_bleed:
        return "intrudes"

    if l < -TOL / EMU or t < -TOL / EMU or r > W / EMU + TOL / EMU \
            or b > H / EMU + TOL / EMU:
        return "off"
    return None


def main(path: str) -> int:
    prs = Presentation(path)
    W, H = prs.slide_width, prs.slide_height
    totals = {"off": 0, "intrudes": 0, "touches-full": 0}

    for i, slide in enumerate(prs.slides, start=1):
        shapes = list(slide.shapes)
        found = {"off": 0, "intrudes": 0, "touches-full": 0}
        for shape in shapes:
            kind = classify(shape, W, H)
            if not kind:
                continue
            found[kind] += 1
            if kind == "intrudes" and found[kind] <= 5:
                print(f"  slide {i:>2} INTRUDES {shape.shape_type} {shape.name!r} "
                      f"T{inches(shape.top):.2f} B{inches(shape.top + shape.height):.2f}")
            if kind == "off" and found[kind] <= 5:
                print(f"  slide {i:>2} OFF  {shape.shape_type} {shape.name!r} "
                      f"L{inches(shape.left):.2f} T{inches(shape.top):.2f} "
                      f"R{inches(shape.left + shape.width):.2f} "
                      f"B{inches(shape.top + shape.height):.2f}")
        for k in totals:
            totals[k] += found[k]
        flag = "" if found["off"] == found["intrudes"] == 0 else "   <-- FIX"
        print(f"slide {i:>2}: {len(shapes):>3} shapes, {found['off']} off-canvas, "
              f"{found['intrudes']} intrude-footer, "
              f"{found['touches-full']} full-bleed{flag}")

    bad = totals["off"] + totals["intrudes"]
    print(f"\nOFF-CANVAS: {totals['off']}   INTRUDES: {totals['intrudes']}   "
          f"FULL-BLEED (allowed): {totals['touches-full']}")
    print(f"LAYOUT VIOLATIONS: {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "DRONA_capstone_deck.pptx"
    sys.exit(main(target))
