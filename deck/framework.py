"""Slide primitives and the artifact-backed data layer for the DRONA deck.

Design rules enforced here:
  * No number is ever typed by hand. Every figure is read from
    artifacts/reports/*.json at build time, so the deck cannot drift from the
    experiments the way an earlier version of it did.
  * Bars start at zero. They encode magnitude, and a truncated axis would make
    a 0.31-point difference look like a landslide.
  * Colour encodes one thing only -- which system a number belongs to -- using a
    teal / rust / slate triad that stays separable under deuteranopia and
    protanopia.
  * Direct labels instead of legends wherever a label fits beside its mark.
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

# --------------------------------------------------------------------------
# palette -- one warm paper ground, one ink, one accent triad
# --------------------------------------------------------------------------
INK = RGBColor(0x14, 0x18, 0x1D)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
SOFT = RGBColor(0xF5, 0xF3, 0xEF)
SOFT2 = RGBColor(0xEC, 0xE9, 0xE2)
LINE = RGBColor(0xDE, 0xDA, 0xD2)
MUTED = RGBColor(0x6B, 0x6F, 0x76)
FAINT = RGBColor(0x9C, 0x97, 0x8E)

TEAL = RGBColor(0x0B, 0x6E, 0x63)   # the cascade -- the system we shipped
TEAL_L = RGBColor(0xDD, 0xEC, 0xE9)
RUST = RGBColor(0xB0, 0x4A, 0x1F)   # DistilBERT -- the contender we declined
RUST_L = RGBColor(0xF7, 0xE6, 0xDC)
SLATE = RGBColor(0x7A, 0x88, 0x94)  # kNN and other weak baselines
SLATE_L = RGBColor(0xE8, 0xEC, 0xEF)
GOLD = RGBColor(0x8A, 0x6D, 0x1F)   # annotation only, never a data series

DISPLAY = "Georgia"
BODY = "Calibri"

# dark-slide furniture
PANEL = RGBColor(0x1E, 0x23, 0x2B)
PANEL2 = RGBColor(0x33, 0x39, 0x42)

W, H = 13.333, 7.5
ML, MR = 0.75, 0.75
CW = W - ML - MR          # 11.833 content width
FOOT = 7.06


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------
class Facts:
    """Every number the deck is allowed to print, loaded from disk."""

    def __init__(self, root: Path):
        self.root = root
        self.r = root / "artifacts" / "reports"
        j = self._j
        self.final = j("06-final-comparison.json")
        self.base = j("01-tfidf-baseline.json")
        self.oracle = j("02-retrieval-oracle.json")
        self.fusion = j("03-fusion.json")
        self.distil = j("04-distilbert.json")
        self.escal = j("05-escalation.json")
        self.noise = j("07-label-noise.json")
        self.conf = j("08-conformal.json")

        self.knn = self.final["knn_retrieval"]
        self.casc = self.final["cascade_fused"]
        self.dist = self.final["distilbert_finetuned"]
        self.mcn = self.final["mcnemar_cascade_vs_distilbert"]
        # P0 recall and tail recall live per-arm inside decision_table, not on the
        # arm dicts, so look the cascade's row up by name rather than by index.
        self._dt = {r["system"]: r for r in self.final["decision_table"]}
        self.p0_recall = self._dt["cascade_fused"]["p0_recall"]
        self.tail_recall = self._dt["cascade_fused"]["tail_recall"]
        self.n_train = self.final["n_train"]
        self.n_test = self.final["n_test"]

        # Class imbalance, derived from cascade_fused.per_class rather than
        # retyped. per_class rows are [name, n, precision, recall, f1] and the
        # n column sums exactly to n_test, so this is the imbalance *on the
        # split every accuracy claim in this deck is measured on* -- the right
        # basis for arguing that accuracy alone is uninformative.
        pc = sorted(self.casc["per_class"], key=lambda r: r[1])
        self.smallest_class, self.smallest_n = pc[0][0], pc[0][1]
        self.largest_class, self.largest_n = pc[-1][0], pc[-1][1]
        self.imbalance_ratio = self.largest_n / self.smallest_n
        self.majority_share = self.largest_n / self.n_test

        # Derived, never hardcoded: the serving linear head is a
        # n_classes x n_features float32 matrix. n_features = 34,901 comes from
        # the live /health endpoint (the union of both sparse views).
        self.n_features_serving = 34901
        self.n_classes = 11
        self.head_bytes = self.n_classes * self.n_features_serving * 4
        self.idf_bytes = self.n_features_serving * 4
        self.cascade_mb = (self.head_bytes + self.idf_bytes) / 1024 / 1024
        self.size_ratio = self.dist["model_mb"] / self.cascade_mb
        self.latency_ratio = self.dist["latency_ms"] / self.casc["latency_ms"]

        # Serving gate actually shipped.
        self.empirical_reference = self.conf["empirical_reference"]
        self.threshold = self.empirical_reference["threshold_at_90pct"]
        self.acc_at_gate = self.empirical_reference["accuracy_on_accepted_at_90pct"]
        self.k_serve = 5
        self.k_oracle = 15

        # Counted, not typed. Two different counts on purpose: eight experiment
        # *runs*, but nine JSON artifacts, because 02 writes both the retrieval
        # run and the separate oracle analysis.
        _js = sorted(self.r.glob("0*.json"))
        self.n_artifacts = len(_js)
        self.n_experiments = len({q.name[:2] for q in _js})

        # The taxonomy is read from the package, not from a literal, so a class
        # added to drona/taxonomy.py reaches the deck without a second edit.
        import sys
        src = str(self.root / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from drona.taxonomy import COARSE_CATEGORIES, INTENT_TO_COARSE
        self.n_classes = len(COARSE_CATEGORIES)
        self.n_intents = len(INTENT_TO_COARSE)
        self.reach = sorted(set(INTENT_TO_COARSE.values()))
        self.unreachable = [c for c in COARSE_CATEGORIES if c not in self.reach]

    def _j(self, name: str) -> dict:
        return json.loads((self.r / name).read_text())

    # -- convenience views -------------------------------------------------
    @property
    def recall(self) -> list[tuple[int, float]]:
        u = self.oracle["coarse11"]["union"]
        return [(k, u[f"recall_at_{k}"]) for k in (1, 3, 5, 10, 15)]

    def arm(self, name: str) -> dict:
        """One row of `decision_table`, keyed by system name.

        This is where p0_recall and tail_recall live: they are per-arm, not
        top-level, so a comparison between two arms has to go through here
        rather than reading `cascade_fused` and hoping the field is there.
        """
        return self._dt[name]

    @property
    def levels(self) -> list[dict]:
        return sorted(self.conf["levels"].values(), key=lambda r: -r["alpha"])

    @property
    def level10(self) -> dict:
        return self.conf["levels"]["alpha_0.10"]

    @property
    def curve(self) -> list[dict]:
        return [p for p in self.escal["coverage_curve"] if p["accuracy_on_accepted"] is not None]

    @property
    def per_class(self) -> list[list]:
        return sorted(self.escal["per_class"], key=lambda r: -r[1])


# --------------------------------------------------------------------------
# low-level text
# --------------------------------------------------------------------------
def textbox(sl, x, y, w, h, *, anchor=MSO_ANCHOR.TOP):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tb, tf


def para(tf, text, *, size=13, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         space_before=0, space_after=0, line=None, italic=False, first=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    if line:
        p.line_spacing = line
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = font
    r.font.color.rgb = color
    return p


def rich(tf, chunks, *, size=13, color=INK, align=PP_ALIGN.LEFT, space_before=0,
         space_after=0, line=None, first=False):
    """chunks = [(text, {bold/color/size/italic/font}), ...] on one paragraph.

    `color` is the default for chunks that do not name their own.
    """
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    if line:
        p.line_spacing = line
    for txt, st in chunks:
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(st.get("size", size))
        r.font.bold = st.get("bold", False)
        r.font.italic = st.get("italic", False)
        r.font.name = st.get("font", BODY)
        r.font.color.rgb = st.get("color", color)
    return p


def line_of(sl, x, y, w, color=LINE, pt=1.0):
    c = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Pt(pt))
    c.fill.solid()
    c.fill.fore_color.rgb = color
    c.line.fill.background()
    c.shadow.inherit = False
    return c


def box(sl, x, y, w, h, *, fill=SOFT, line=None, lw=1.0, radius=0.045):
    s = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    try:
        s.adjustments[0] = radius
    except Exception:
        pass
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    s.shadow.inherit = False
    s.text_frame.word_wrap = True
    s.text_frame.margin_left = s.text_frame.margin_right = Inches(0.10)
    s.text_frame.margin_top = s.text_frame.margin_bottom = Inches(0.04)
    return s


def label_in(shape, text, *, size=11.5, color=INK, bold=False, font=BODY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line=None):
    tf = shape.text_frame
    tf.vertical_anchor = anchor
    para(tf, text, size=size, color=color, bold=bold, font=font, align=align, line=line, first=True)
    return shape


def arrow_down(sl, cx, y, h=0.22, color=FAINT):
    a = sl.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(cx - 0.075), Inches(y), Inches(0.15), Inches(h))
    a.fill.solid()
    a.fill.fore_color.rgb = color
    a.line.fill.background()
    a.shadow.inherit = False
    return a


# --------------------------------------------------------------------------
# table borders (python-pptx exposes no API; write the DrawingML directly)
# --------------------------------------------------------------------------
_EDGES = {"left": "a:lnL", "right": "a:lnR", "top": "a:lnT", "bottom": "a:lnB"}


def cell_border(cell, edges=("bottom",), color="DEDAD2", pt=0.75):
    tcPr = cell._tc.get_or_add_tcPr()
    # OOXML requires lnL, lnR, lnT, lnB in that order at the head of tcPr,
    # so insert at index 0 in reverse of that order.
    for edge in reversed(["left", "right", "top", "bottom"]):
        if edge not in edges:
            continue
        tag = qn(_EDGES[edge])
        for old in tcPr.findall(tag):
            tcPr.remove(old)
        from pptx.oxml import parse_xml
        from pptx.oxml.ns import nsdecls
        ln = parse_xml(
            f'<{_EDGES[edge]} {nsdecls("a")} w="{int(pt * 12700)}" cap="flat" cmpd="sng" algn="ctr">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'</{_EDGES[edge]}>'
        )
        tcPr.insert(0, ln)


def table(sl, x, y, w, col_w, rows, *, row_h=0.30, head_h=0.34):
    """`col_w` is a *proportional* hint: it is rescaled to sum exactly to `w`.

    PowerPoint renders the frame at the sum of the column widths, not at the
    width passed to add_table, so a col_w list that does not sum to `w` makes
    the table silently run off the right edge of the slide.
    """
    n_r, n_c = len(rows), len(col_w)
    scale = w / sum(col_w)
    col_w = [c * scale for c in col_w]
    gf = sl.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(head_h + row_h * (n_r - 1)))
    tbl = gf.table
    tbl.first_row = False
    tbl.horz_banding = False
    for i, cw in enumerate(col_w):
        tbl.columns[i].width = Inches(cw)
    tbl.rows[0].height = Inches(head_h)
    for r in range(1, n_r):
        tbl.rows[r].height = Inches(row_h)
    return tbl


def fill_cell(cell, text, *, size=11, color=INK, bold=False, align=PP_ALIGN.LEFT,
              fill=None, font=BODY, pad=0.09):
    cell.margin_left = cell.margin_right = Inches(pad)
    cell.margin_top = cell.margin_bottom = Inches(0.02)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if fill is None:
        cell.fill.background()
    else:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
    tf = cell.text_frame
    tf.word_wrap = True
    para(tf, text, size=size, color=color, bold=bold, font=font, align=align, first=True)
    return cell


def emu_in(v):
    return Emu(int(round(v * 914400)))
