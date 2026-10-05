"""Chart primitives, drawn from rectangles and textboxes.

Native PPTX charts would inherit the theme's colours, fonts and gridlines, none
of which I control from the high-level API. Drawing the marks directly keeps
every pixel accounted for and keeps the numbers as real text in the file.
"""

from __future__ import annotations

from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from framework import (
    BODY, FAINT, INK, LINE, MUTED, PAPER, SOFT, SOFT2, TEAL, TEAL_L,
    cell_border, line_of, para, textbox,
)


def blank(prs, *, dark=False, bg=None):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    fill = bg if bg is not None else (INK if dark else PAPER)
    r = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    r.fill.solid()
    r.fill.fore_color.rgb = fill
    r.line.fill.background()
    r.shadow.inherit = False
    return sl


def header(sl, kicker, title, *, n=None, sub=None, dark=False):
    """Kicker + claim-style title + rule. The title states the slide's finding."""
    ink = PAPER if dark else INK
    muted = FAINT if dark else MUTED
    rule = RGB_LIGHT if dark else LINE
    _, tf = textbox(sl, 0.75, 0.36, 11.5, 0.22)
    para(tf, kicker.upper(), size=9.5, color=TEAL if not dark else TEAL_L, bold=True, first=True)

    title_size = 21.0 if len(title) > 65 else 23.5
    title_h = 0.64 if len(title) > 65 else 0.38
    _, tf = textbox(sl, 0.75, 0.60, 11.833, title_h)
    para(tf, title, size=title_size, color=ink, bold=False, font="Georgia", line=1.04, first=True)

    y = 0.60 + title_h + 0.04
    if sub:
        _, tf = textbox(sl, 0.75, y, 11.833, 0.28)
        para(tf, sub, size=11.5, color=muted, line=1.12, first=True)
        y += 0.32
    else:
        y += 0.08
    line_of(sl, 0.75, y, 11.833, color=rule, pt=0.9)
    if n is not None:
        footer(sl, n, dark=dark)
    return y + 0.18


def footer(sl, n, *, dark=False):
    _, tf = textbox(sl, 0.75, 7.06, 6.0, 0.22)
    para(tf, "DRONA · Banking77 ticket triage · all figures from artifacts/reports/",
         size=8.5, color=FAINT if dark else FAINT, first=True)
    _, tf = textbox(sl, 10.0, 7.06, 2.583, 0.22)
    para(tf, str(n), size=8.5, color=FAINT, align=PP_ALIGN.RIGHT, first=True)


RGB_LIGHT = None  # bound below to avoid a circular import at module top


def stat(sl, x, y, w, value, label, *, sub=None, color=INK, vsize=30, dark=False,
         accent=None):
    """A number with a label under it. No card -- the type does the work."""
    _, tf = textbox(sl, x, y, w, 0.46)
    para(tf, value, size=vsize, color=color, bold=True, font="Georgia", first=True)
    _, tf = textbox(sl, x, y + 0.44, w, 0.24)
    para(tf, label, size=10, color=MUTED if not dark else FAINT, bold=True, first=True)
    if sub:
        _, tf = textbox(sl, x, y + 0.66, w, 0.40)
        para(tf, sub, size=9.5, color=FAINT, line=1.2, first=True)
    if accent:
        line_of(sl, x, y - 0.12, 0.44, color=accent, pt=2.2)


def hbar_chart(sl, x, y, w, h, rows, *, vmax=None, label_w=2.55, gap=0.16,
               fmt="{:.1f}", note_w=0.0, axis_label=None):
    """rows = [(label, value, colour, value_text_override), ...]. Zero-based."""
    vmax = vmax or max(r[1] for r in rows) * 1.02
    plot_x = x + label_w
    plot_w = w - label_w - note_w
    n = len(rows)
    row_h = (h - gap * (n - 1)) / n
    bar_h = min(row_h * 0.56, 0.40)

    for i, r in enumerate(rows):
        label, val, col = r[0], r[1], r[2]
        over = r[3] if len(r) > 3 else None
        ry = y + i * (row_h + gap)
        _, tf = textbox(sl, x, ry + (row_h - 0.24) / 2 - 0.02, label_w - 0.14, 0.30,
                        anchor=MSO_ANCHOR.MIDDLE)
        para(tf, label, size=11, color=INK, align=PP_ALIGN.RIGHT, first=True)
        bw = max(plot_w * (val / vmax), 0.02)
        bar = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(plot_x),
                                  Inches(ry + (row_h - bar_h) / 2), Inches(bw), Inches(bar_h))
        bar.adjustments[0] = 0.18
        bar.fill.solid()
        bar.fill.fore_color.rgb = col
        bar.line.fill.background()
        bar.shadow.inherit = False
        vt = over if over is not None else fmt.format(val)
        _, tf = textbox(sl, plot_x + bw + 0.09, ry + (row_h - 0.24) / 2 - 0.02,
                        plot_w - bw + note_w + 0.4, 0.30, anchor=MSO_ANCHOR.MIDDLE)
        para(tf, vt, size=11.5, color=INK, bold=True, first=True)

    base_y = y + n * (row_h + gap) - gap
    line_of(sl, plot_x, base_y, plot_w, color=LINE, pt=0.9)
    if axis_label:
        _, tf = textbox(sl, plot_x, base_y + 0.07, plot_w, 0.22)
        para(tf, axis_label, size=9, color=FAINT, first=True)
    return base_y


def vbar_chart(sl, x, y, w, h, series, *, vmax=None, vmin=0.0, fmt="{:.1f}",
               legend=None, ticks=None, gridlines=0, value_size=10.5):
    """Grouped vertical bars. series = [(name, colour, [values])], one value per category."""
    cats = ticks or [t[0] for t in (legend or [])]
    vmax = vmax if vmax is not None else max(max(s[2]) for s in series) * 1.15
    plot_h = h - 0.42
    plot_y = y
    n_c, n_s = len(cats), len(series)
    slot = w / n_c
    bw = min(slot * 0.62 / n_s, 0.55)

    for g in range(gridlines + 1):
        gy = plot_y + plot_h * g / gridlines
        line_of(sl, x, gy, w, color=LINE if g == 0 else SOFT2, pt=0.9)

    for ci, cat in enumerate(cats):
        cx = x + slot * ci
        for si, (name, col, vals) in enumerate(series):
            v = vals[ci]
            bh = plot_h * (v - vmin) / (vmax - vmin)
            bx = cx + (slot - bw * n_s) / 2 + si * bw
            b = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(bx),
                                    Inches(plot_y + plot_h - bh), Inches(bw), Inches(bh))
            b.fill.solid()
            b.fill.fore_color.rgb = col
            b.line.fill.background()
            b.shadow.inherit = False
            _, tf = textbox(sl, bx - 0.22, plot_y + plot_h - bh - 0.24, bw + 0.44, 0.22)
            para(tf, fmt.format(v), size=value_size, color=INK, bold=True,
                 align=PP_ALIGN.CENTER, first=True)
        _, tf = textbox(sl, cx, plot_y + plot_h + 0.09, slot, 0.34)
        para(tf, cat, size=10, color=MUTED, align=PP_ALIGN.CENTER, line=1.05, first=True)
    return plot_y + plot_h


def curve_chart(sl, x, y, w, h, pts, *, xlabel, ylabel, colour=TEAL,
                marker_idx=None, marker_text=None, fill_area=True, ymin=0.0, ymax=1.0,
                note=None, curve2=None, colour2=None):
    """A single line over an evenly spaced categorical axis. pts = [(x, y), ...]."""
    plot_x, plot_y = x + 0.52, y + 0.10
    plot_w, plot_h = w - 0.62, h - 0.78
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n = len(pts)

    def px(i):
        return plot_x + plot_w * i / (n - 1)

    def py(v):
        return plot_y + plot_h * (1 - (v - ymin) / (ymax - ymin))

    for g in range(5):
        gy = plot_y + plot_h * g / 4
        line_of(sl, plot_x, gy, plot_w, color=SOFT2, pt=0.7)
        _, tf = textbox(sl, x - 0.06, gy - 0.10, 0.52, 0.20)
        para(tf, f"{ymax - (ymax - ymin) * g / 4:.0%}", size=8.5, color=FAINT,
             align=PP_ALIGN.RIGHT, first=True)

    def polyline(series, col, thick):
        for i in range(n - 1):
            x1, y1 = px(i), py(series[i][1])
            x2, y2 = px(i + 1), py(series[i + 1][1])
            seg = sl.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
            seg.line.color.rgb = col
            seg.line.width = Pt(thick)
            seg.shadow.inherit = False

    if curve2 and colour2:
        polyline(curve2, colour2, 1.6)
    polyline(pts, colour, 2.4)

    if marker_idx is not None:
        mi = marker_idx
        d = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(px(mi) - 0.075), Inches(py(ys[mi]) - 0.075),
                                Inches(0.15), Inches(0.15))
        d.fill.solid()
        d.fill.fore_color.rgb = colour
        d.line.color.rgb = PAPER
        d.line.width = Pt(1.5)
        d.shadow.inherit = False
        if marker_text:
            _, tf = textbox(sl, px(mi) - 1.35, py(ys[mi]) - 0.46, 1.30, 0.34)
            para(tf, marker_text, size=9.5, color=INK, bold=True, align=PP_ALIGN.RIGHT,
                 line=1.1, first=True)

    for i, xv in enumerate(xs):
        _, tf = textbox(sl, px(i) - 0.36, plot_y + plot_h + 0.09, 0.72, 0.22)
        para(tf, str(xv), size=9, color=FAINT, align=PP_ALIGN.CENTER, first=True)

    _, tf = textbox(sl, plot_x, plot_y + plot_h + 0.30, plot_w, 0.22)
    para(tf, xlabel, size=9.5, color=MUTED, align=PP_ALIGN.CENTER, first=True)
    _, tf = textbox(sl, x - 0.06, y - 0.30, 3.0, 0.22)
    para(tf, ylabel, size=9.5, color=MUTED, first=True)
    if note:
        _, tf = textbox(sl, x, y + h - 0.06, w, 0.30)
        para(tf, note, size=9.5, color=FAINT, line=1.2, first=True)


def scatter_pair(sl, x, y, w, h, groups, *, xlabel, ylabel, xlim=(0, 1), ylim=(0, 1),
                 diag=True, notes=()):
    """groups = [(label, colour, n, x_frac, y_frac, dx, dy)] -- x_frac/y_frac are 0..1 in-plot."""
    plot_x, plot_y = x + 0.46, y
    plot_w, plot_h = w - 0.56, h - 0.62

    def px(f):
        return plot_x + plot_w * f

    def py(f):
        return plot_y + plot_h * (1 - f)

    for g in range(5):
        line_of(sl, plot_x, plot_y + plot_h * g / 4, plot_w, color=SOFT2, pt=0.7)
        line_of(sl, px(g / 4), plot_y, Pt(0.7), color=SOFT2) if False else None
    for g in range(5):
        _, tf = textbox(sl, x - 0.06, py(g / 4) - 0.09, 0.46, 0.20)
        para(tf, f"{ylim[0] + (ylim[1] - ylim[0]) * g / 4:.0%}", size=8.5, color=FAINT,
             align=PP_ALIGN.RIGHT, first=True)
        _, tf = textbox(sl, px(g / 4) - 0.30, plot_y + plot_h + 0.07, 0.60, 0.20)
        para(tf, f"{xlim[0] + (xlim[1] - xlim[0]) * g / 4:.0%}", size=8.5, color=FAINT,
             align=PP_ALIGN.CENTER, first=True)

    if diag:
        d = sl.shapes.add_connector(1, Inches(px(0)), Inches(py(0)), Inches(px(1)), Inches(py(1)))
        d.line.color.rgb = LINE
        d.line.width = Pt(1.0)
        d.shadow.inherit = False

    for label, col, n, xf, yf, dx, dy in groups:
        cx, cy = px(xf), py(yf)
        r = 0.13 + 0.30 * (n / max(g[2] for g in groups)) ** 0.5
        o = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - r), Inches(cy - r),
                                Inches(2 * r), Inches(2 * r))
        o.fill.solid()
        o.fill.fore_color.rgb = col
        o.line.color.rgb = PAPER
        o.line.width = Pt(1.4)
        o.shadow.inherit = False
        _, tf = textbox(sl, cx + dx, cy + dy, 1.9, 0.24)
        para(tf, f"{label}  {n:,}", size=9.5, color=INK, bold=True, first=True)

    _, tf = textbox(sl, plot_x, plot_y + plot_h + 0.26, plot_w, 0.22)
    para(tf, xlabel, size=9.5, color=MUTED, align=PP_ALIGN.CENTER, first=True)
    _, tf = textbox(sl, x - 0.06, y - 0.30, 4.0, 0.22)
    para(tf, ylabel, size=9.5, color=MUTED, first=True)
    return plot_x, plot_y, plot_w, plot_h, px, py


from framework import RGBColor  # noqa: E402  (placed late to keep imports tidy)

RGB_LIGHT = RGBColor(0x3A, 0x40, 0x48)
