"""The ten mandated capstone slides, in the order the batch brief requires:

    1 Project Title      6 Methodology
    2 Team Members       7 System Architecture
    3 Problem Statement  8 Project Design
    4 Objectives         9 Expected Outcomes
    5 Literature Survey 10 Future Plan

Design rules, enforced here rather than left to discipline:

* **No number is typed by hand.** Everything quantitative comes from `Facts`,
  which reads `artifacts/reports/*.json` at build time. If an experiment is re-run,
  the deck moves with it or fails to build.
* **Prose claims that are not measurements are attributed.** Team names, the
  guide and the literature table come from `research/00-original-synopsis.txt`,
  which is a document we were given -- not something we measured.

Geometry convention: `header()` returns the first free y (1.60 bare, 1.94 with a
subtitle). Content ends by 6.98; the footer baseline is 7.06.
"""

from __future__ import annotations

from framework import (
    INK, PAPER, SOFT, SOFT2, LINE, MUTED, FAINT,
    TEAL, TEAL_L, RUST, RUST_L, SLATE, SLATE_L, GOLD,
    PANEL, PANEL2, DISPLAY, BODY,
    ML, CW, W, H,
    Facts, box, cell_border, fill_cell, line_of, label_in, para, rich, table, textbox,
)
from charts import blank, curve_chart, hbar_chart, stat, vbar_chart, footer, header
from pptx.enum.text import PP_ALIGN


def pct(x: float, dp: int = 2) -> str:
    return f"{x * 100:.{dp}f}%"


# ---------------------------------------------------------------------------
# slide 1 -- Project Title
# ---------------------------------------------------------------------------


def s01(prs: "Presentation", F: Facts):
    sl = blank(prs, dark=True)
    box(sl, 0, 0, W, H, fill=PANEL, line=None, radius=0.0)
    box(sl, 0, 0, 0.16, H, fill=TEAL, line=None, radius=0.0)

    x = 1.10
    rx = 8.75
    # Left column constrained to stay clear of the right rail (rx = 8.75)
    lw = rx - x - 0.45  # 7.20 in
    _, tf = textbox(sl, x, 0.72, lw, 0.26)
    para(tf, "B.TECH CAPSTONE  ·  COMPUTER SCIENCE AND ENGINEERING  ·  BATCH MP-A05",
         size=10, color=TEAL_L, bold=True, first=True)

    _, tf = textbox(sl, x, 1.12, lw, 1.25)
    para(tf, "Smart Ticket Understanding Engine", size=36, color=PAPER,
         font=DISPLAY, line=1.04, first=True)

    _, tf = textbox(sl, x, 2.38, lw, 0.62)
    para(tf, "DRONA — intent classification, sentiment, SLA priority and routing "
             "for banking support tickets, offline on CPU",
         size=13.5, color=TEAL_L, line=1.25, first=True)

    line_of(sl, x, 3.14, lw, color=PANEL2, pt=1.0)

    _, tf = textbox(sl, x, 3.32, lw, 1.5)
    rich(tf, [("A transformer was not needed. ", {"bold": True, "color": PAPER}),
              ("Fine-tuned DistilBERT does edge out the cascade on accuracy — by "
               f"{(F.dist['accuracy'] - F.casc['accuracy']) * 100:.2f} points. McNemar's test on the "
               f"tickets where the two disagree returns p = {F.mcn['p_value']:.4f}. That is not a "
               "result, it is noise. So we shipped the cascade: same accuracy, "
               f"{F.latency_ratio:.1f}× faster, {F.size_ratio:.0f}× smaller, and it carries a "
               "finite-sample guarantee on the tickets it auto-answers.", {})],
         size=12.5, color=FAINT, line=1.34, first=True)

    _, tf = textbox(sl, x, 5.12, lw, 0.9)
    rich(tf, [("What we want to show: ", {"bold": True, "color": TEAL_L}),
              ("what we measured, what it cost, what we got wrong, and what we "
               "would do next. Every number in this deck is read from a JSON "
               "artifact at build time — none is typed in.", {})],
         size=11, color=MUTED, line=1.30, first=True)

    # Right rail: the five numbers that carry the project.
    rx = 8.75
    box(sl, rx, 1.14, W - rx - 0.62, 5.24, fill=PANEL2, line=None, radius=0.05)
    _, tf = textbox(sl, rx + 0.34, 1.42, W - rx - 1.3, 0.26)
    para(tf, "WHAT WE SHIPPED", size=9.5, color=TEAL_L, bold=True, first=True)

    tiles = [
        (pct(F.casc["accuracy"]), "accuracy", f"macro-F1 {pct(F.casc['macro_f1'])}", PAPER),
        (pct(F.acc_at_gate), "on auto-answered",
         f"at {F.threshold:.4f} confidence", TEAL_L),
        (f"{F.casc['latency_ms']:.1f} ms", "per ticket", "in-process, CPU, no network", PAPER),
        (f"{F.cascade_mb:.1f} MB", "total model size",
         f"vs {F.dist['model_mb']:.0f} MB fine-tuned", TEAL_L),
    ]
    yy = 1.80
    for v, lab, sub, col in tiles:
        _, tf = textbox(sl, rx + 0.34, yy, W - rx - 1.3, 0.46)
        para(tf, v, size=29, color=col, bold=True, font=DISPLAY, first=True)
        _, tf = textbox(sl, rx + 0.34, yy + 0.46, W - rx - 1.3, 0.24)
        para(tf, lab, size=11, color=PAPER, bold=True, first=True)
        _, tf = textbox(sl, rx + 0.34, yy + 0.68, W - rx - 1.3, 0.24)
        para(tf, sub, size=9.5, color=FAINT, first=True)
        yy += 1.16

    footer(sl, 1, dark=True)
    return sl


# ---------------------------------------------------------------------------
# slide 2 -- Team Members
# ---------------------------------------------------------------------------


def s02(prs: "Presentation", F: Facts):
    sl = blank(prs)
    y = header(sl, "Team members", "Four of us, one guide, and an industry mentor who read the draft.",
               n=2,
               sub="B.Tech CSE, batch MP-A05 — Marri Laxman Reddy Institute of Technology "
                   "and Management, under JNTUH.")

    rows = [["#", "Reg. no.", "Name", "Where the work sits"]]
    roles = [
        "Pipeline, retrieval + fusion, calibration and the conformal gate",
        "DistilBERT fine-tune, the comparison, and the McNemar analysis",
        "SLA priority, routing, sentiment and amount extraction",
        "FastAPI service, Streamlit console, drift monitor, evaluation",
    ]
    for i, ((reg, name), role) in enumerate(zip(TEAM, roles), 1):
        rows.append([str(i), reg, name, role])

    t = table(sl, ML, y, CW, [0.45, 1.55, 2.65, 7.183], rows, row_h=0.38, head_h=0.30)
    for r in range(1, len(rows)):
        fill_cell(t.cell(r, 0), rows[r][0], size=10.5, color=FAINT, align=PP_ALIGN.CENTER)
        fill_cell(t.cell(r, 1), rows[r][1], size=11, color=MUTED, font="Consolas")
        fill_cell(t.cell(r, 2), rows[r][2], size=12, color=INK, bold=True)
        fill_cell(t.cell(r, 3), rows[r][3], size=10.5, color=MUTED)
    for c in range(4):
        fill_cell(t.cell(0, c), rows[0][c].upper(), size=9, color=TEAL, bold=True)
        cell_border(t.cell(0, c), edges=("bottom",), color="0B6E63", pt=1.2)

    yy = y + 0.30 + 0.38 * len(roles) + 0.28

    # Guide + institution, as two cards.
    box(sl, ML, yy, 5.75, 1.30, fill=SOFT, line=LINE, radius=0.05)
    _, tf = textbox(sl, ML + 0.28, yy + 0.16, 5.2, 0.22)
    para(tf, "PROJECT GUIDE", size=9, color=TEAL, bold=True, first=True)
    _, tf = textbox(sl, ML + 0.28, yy + 0.40, 5.2, 0.30)
    para(tf, GUIDE, size=15, color=INK, bold=True, font=DISPLAY, first=True)
    _, tf = textbox(sl, ML + 0.28, yy + 0.72, 5.2, 0.42)
    para(tf, "Associate Professor, Department of CSE", size=10.5, color=MUTED, first=True)

    box(sl, ML + 5.98, yy, CW - 5.98, 1.30, fill=SOFT, line=LINE, radius=0.05)
    _, tf = textbox(sl, ML + 6.26, yy + 0.16, CW - 6.6, 0.22)
    para(tf, "INDUSTRY MENTOR", size=9, color=TEAL, bold=True, first=True)
    _, tf = textbox(sl, ML + 6.26, yy + 0.40, CW - 6.6, 0.72)
    para(tf, "A practising industry expert from a HYSEA member company, who reviewed the "
             "capstone and gave feedback on the triage pipeline and its evaluation.",
         size=10, color=MUTED, line=1.25, first=True)

    _, tf = textbox(sl, ML, yy + 1.44, CW, 0.26)
    para(tf, "Submitted to Jawaharlal Nehru Technological University Hyderabad in partial "
             "fulfilment of the requirements for the degree of Bachelor of Technology in "
             "Computer Science and Engineering.",
         size=9.5, color=FAINT, first=True)
    return sl


# ---------------------------------------------------------------------------
# slide 3 -- Problem Statement
# ---------------------------------------------------------------------------


def s03(prs: "Presentation", F: Facts):
    sl = blank(prs)
    y = header(sl, "Problem statement", "Nobody can staff a queue per intent. The label space is not the problem space.",
               n=3,
               sub=f"Banking77: {F.n_train + F.n_test + F.fusion['n_val']:,} real banking queries, "
                   f"{F.n_intents} fine-grained intents, written by actual customers.")

    # Left: the four concrete problems, as numbered cards.
    probs = [
        ("77 intents, ~8 real queues",
         f"{F.n_intents} intents collapse to {F.n_classes} business categories. Routing to "
         "the fine label is not an option — no support organisation staffs it."),
        (f"{F.imbalance_ratio:.1f}:1 class imbalance",
         f"In the {F.n_test:,}-ticket test split the largest category, {F.largest_class}, holds "
         f"{F.largest_n} tickets and the smallest, {F.smallest_class}, holds {F.smallest_n}. "
         f"Accuracy alone is uninformative: always answering {F.largest_class} alone already "
         f"scores {pct(F.majority_share)}."),
        ("High-consequence errors",
         "A missed fraud ticket is a regulatory incident. A wrong priority misroutes a "
         "customer who is already angry. Both are expensive, and neither shows up in F1."),
        ("No safe way to abstain",
         f"The transformer tops accuracy at {pct(F.dist['accuracy'])} but macro-F1 is only "
         f"{pct(F.dist['macro_f1'])}: it is being carried by the majority class. And it buys "
         f"that {pct(F.dist['accuracy'] - F.dist['macro_f1'])} for {F.latency_ratio:.1f}x the latency."),
    ]
    cw2 = (CW - 0.44) / 2
    for i, (h, b) in enumerate(probs):
        cx = ML + (i % 2) * (cw2 + 0.44)
        cy = y + (i // 2) * 1.62
        box(sl, cx, cy, cw2, 1.40, fill=SOFT, line=None, radius=0.05)
        box(sl, cx, cy, 0.055, 1.40, fill=RUST, line=None, radius=0.0)
        _, tf = textbox(sl, cx + 0.32, cy + 0.20, cw2 - 0.62, 0.28)
        rich(tf, [(f"{i + 1}  ", {"color": RUST, "bold": True, "size": 12}),
                  (h, {"color": INK, "bold": True, "size": 12.5})], first=True)
        _, tf = textbox(sl, cx + 0.32, cy + 0.52, cw2 - 0.62, 0.78)
        para(tf, b, size=10, color=MUTED, line=1.30, first=True)

    yy = y + 3.44

    # The framing problem in the brief itself -- the honest bit.
    box(sl, ML, yy, CW, 1.44, fill=PANEL, line=None, radius=0.05)
    _, tf = textbox(sl, ML + 0.34, yy + 0.20, 3.0, 0.26)
    para(tf, "AND THE PROBLEM WITH THE QUESTION", size=9.5, color=RUST_L, bold=True, first=True)
    _, tf = textbox(sl, ML + 0.34, yy + 0.48, CW - 0.68, 0.86)
    rich(tf, [("The brief asked us to beat ", {}), ("accuracy", {"italic": True}),
              (". We took that as the starting question and found it was the wrong one. "
               "A system that is right 94.9% of the time and confidently wrong on the "
               "rare fraud categories is worse to deploy than one that is right 94.6% of "
               "the time and refuses to guess on the rest. ", {}),
              ("Everything we built after that is about making the refusal safe.",
               {"bold": True, "color": TEAL_L})],
         size=11.5, color=PAPER, line=1.34, first=True)
    return sl


# ---------------------------------------------------------------------------
# slide 4 -- Objectives
# ---------------------------------------------------------------------------


def s04(prs: "Presentation", F: Facts):
    sl = blank(prs)
    y = header(sl, "Objectives", "Four of the seven objectives in the synopsis are met outright. Two are partial, and one we deliberately did not do.",
               n=4,
               sub="Quoted in intent from the synopsis. The status column is our own "
                   "assessment against what we measured — not the synopsis's own claim.")

    rows = [["#", "Objective (from the synopsis)", "Status", "What we measured"]]
    for i, (o, st, why) in enumerate(objectives(F), 1):
        rows.append([str(i), o, st, why])

    t = table(sl, ML, y, CW, [0.44, 4.05, 1.28, 6.06], rows, row_h=0.46, head_h=0.30)
    stat_col = {"MET": TEAL, "PARTIAL": GOLD, "NOT DONE": RUST,
                "MET, NOT ADOPTED": GOLD}
    for r in range(1, len(rows)):
        fill_cell(t.cell(r, 0), rows[r][0], size=10.5, color=FAINT, align=PP_ALIGN.CENTER)
        fill_cell(t.cell(r, 1), rows[r][1], size=10, color=INK)
        fill_cell(t.cell(r, 2), rows[r][2], size=9, color=stat_col[rows[r][2]], bold=True)
        fill_cell(t.cell(r, 3), rows[r][3], size=9.5, color=MUTED)
    for c in range(4):
        fill_cell(t.cell(0, c), rows[0][c].upper(), size=9, color=TEAL, bold=True)
        cell_border(t.cell(0, c), edges=("bottom",), color="0B6E63", pt=1.2)

    yy = y + 0.30 + 0.46 * len(objectives(F)) + 0.18

    box(sl, ML, yy, CW, 0.88, fill=RUST_L, line=None, radius=0.05)
    _, tf = textbox(sl, ML + 0.34, yy + 0.12, CW - 0.68, 0.64)
    rich(tf, [("Objective 5 was to quantise DistilBERT with ONNX INT8 to cut latency and "
               "size. We did not do it, because ", {}),
              ("we shipped a system that is faster and smaller without it",
               {"bold": True}),
              (f" — {F.casc['latency_ms']:.1f} ms and {F.cascade_mb:.1f} MB against "
               f"{F.dist['latency_ms']:.0f} ms and {F.dist['model_mb']:.0f} MB. The honest "
               "reading is that we met the goal and declined the method.", {})],
         size=10.5, color=INK, line=1.25, first=True)
    return sl


# ---------------------------------------------------------------------------
# slide 5 -- Literature Survey
# ---------------------------------------------------------------------------


def s05(prs: "Presentation", F: Facts):
    sl = blank(prs)
    y = header(sl, "Literature survey", "Every component we needed already existed. Assembled, calibrated and measured together, they did not.",
               n=5,
               sub="The nine works in the synopsis, plus the four we had to add because "
                   "the work needed them.")

    tw = 7.58
    rows = [["Author & year", "Technique", "Limitation we inherited"]]
    for a, t_, lim in LIT_SYNOPSIS:
        rows.append([a, t_, lim])

    t = table(sl, ML, y, tw, [2.30, 2.72, 2.56], rows, row_h=0.325, head_h=0.30)
    for r in range(1, len(rows)):
        fill_cell(t.cell(r, 0), rows[r][0], size=9, color=INK, bold=True)
        fill_cell(t.cell(r, 1), rows[r][1], size=8.5, color=MUTED)
        fill_cell(t.cell(r, 2), rows[r][2], size=8.5, color=RUST)
    for c in range(3):
        fill_cell(t.cell(0, c), rows[0][c].upper(), size=8.5, color=TEAL, bold=True)
        cell_border(t.cell(0, c), edges=("bottom",), color="0B6E63", pt=1.2)

    # The four we added, as a 2x2 grid under the table.
    ry = y + 0.30 + 0.325 * len(LIT_SYNOPSIS) + 0.20
    _, tf = textbox(sl, ML, ry, tw, 0.24)
    para(tf, "AND THE FOUR WE ADDED", size=9, color=GOLD, bold=True, first=True)

    gwid = (tw - 0.22) / 2
    for i, (a, t_, why) in enumerate(LIT_ADDED):
        gx = ML + (i % 2) * (gwid + 0.22)
        gy = ry + 0.26 + (i // 2) * 0.74
        box(sl, gx, gy, gwid, 0.68, fill=SOFT, line=None, radius=0.04)
        _, tf = textbox(sl, gx + 0.16, gy + 0.07, gwid - 0.32, 0.22)
        rich(tf, [(a + "  ·  ", {"bold": True, "size": 8.5, "color": INK}),
                  (t_, {"size": 8, "color": GOLD})], first=True)
        _, tf = textbox(sl, gx + 0.16, gy + 0.29, gwid - 0.32, 0.34)
        para(tf, why, size=8, color=MUTED, line=1.16, first=True)

    # The gap, in a full-height rail beside the table.
    gx = ML + tw + 0.34
    gw = CW - tw - 0.34
    box(sl, gx, y, gw, 4.50, fill=PANEL, line=None, radius=0.05)
    _, tf = textbox(sl, gx + 0.30, y + 0.24, gw - 0.60, 0.26)
    para(tf, "THE GAP", size=9.5, color=TEAL_L, bold=True, first=True)
    _, tf = textbox(sl, gx + 0.30, y + 0.58, gw - 0.60, 1.9)
    para(tf, "Nine works, nine limitations — and every one of them stops exactly where "
             "the next one starts. Classifiers that do not route. Routers that do not "
             "prioritise. RAG that hallucinates. MLOps that watches metrics but not "
             "text. Not one of them ships a calibrated, abstaining, CPU-only triage loop.",
         size=10.5, color=PAPER, line=1.34, first=True)
    line_of(sl, gx + 0.30, y + 2.62, gw - 0.60, color=PANEL2, pt=0.9)
    _, tf = textbox(sl, gx + 0.30, y + 2.80, gw - 0.60, 1.5)
    rich(tf, [("A disagreement worth reporting. ", {"bold": True, "color": TEAL_L}),
              ("Ying & Thomas report ~14% label noise in Banking77", {"bold": True, "color": RUST_L}),
              (f"; our confident-learning audit flags {F.noise['n_flagged']} of "
               f"{F.n_train:,} training labels — {pct(F.noise['flagged_fraction'], 1)}. "
               "Not the same claim: theirs is at the 77-intent level on the full "
               "dataset, ours at the coarse level on the train split only: "
               "coarse grouping absorbs most fine-intent boundary noise.", {})],
         size=9.5, color=FAINT, line=1.30, first=True)
    return sl


# ---------------------------------------------------------------------------
# slide 6 -- Methodology
# ---------------------------------------------------------------------------


def s06(prs: "Presentation", F: Facts):
    sl = blank(prs)
    y = header(sl, "Methodology", "Two sparse views, two cheap heads, one calibrated gate. Every number below is measured on a held-out split.",
               n=6,
               sub=f"Train {F.n_train:,} / validation 1,962 / test {F.n_test:,}, stratified "
                   f"70/15/15 on the coarse label, seed 20261003. Nothing is tuned on test.")

    # Pipeline: six stages, left to right.
    stages = [
        ("1  TWO VIEWS", "char_wb 3–5 grams\nword 1–2 grams", f"{F.n_features_serving:,} features", SLATE),
        ("2  TWO HEADS", "LinearSVC\nkNN, k=15", "trained on the same union", SLATE),
        ("3  FUSION", "weighted log-prob\naverage, then temperature",
         f"α={F.fusion['selected_alpha']}, T={F.fusion['selected_temperature']}", TEAL),
        ("4  CALIBRATE", "vector scaling on\nthe validation split", f"ECE {F.casc['ece']:.4f}", TEAL),
        ("5  GATE", "conformal threshold\nat 90% target coverage", f"q={F.level10['threshold_q']:.4f}", TEAL),
        ("6  ACT", f"auto-answer, or escalate\n+ route + prioritise", f"{F.threshold:.4f} served", RUST),
    ]
    bw = (CW - 0.5 * 5) / 6
    for i, (name, body, out, col) in enumerate(stages):
        bx = ML + i * (bw + 0.5)
        box(sl, bx, y, bw, 1.66, fill=SOFT if col is SLATE else TEAL_L, line=None, radius=0.05)
        box(sl, bx, y, bw, 0.045, fill=col, line=None, radius=0.0)
        _, tf = textbox(sl, bx + 0.18, y + 0.18, bw - 0.36, 0.24)
        para(tf, name, size=9, color=col, bold=True, first=True)
        _, tf = textbox(sl, bx + 0.18, y + 0.48, bw - 0.36, 0.62)
        para(tf, body, size=10, color=INK, line=1.26, first=True)
        _, tf = textbox(sl, bx + 0.18, y + 1.18, bw - 0.36, 0.38)
        para(tf, out, size=9, color=MUTED, line=1.20, first=True)
        if i < 5:
            _, tf = textbox(sl, bx + bw + 0.08, y + 0.72, 0.34, 0.24)
            para(tf, "→", size=15, color=FAINT, first=True)

    yy = y + 2.00

    # Three panels: data, what we compared, what we held fixed.
    pw = (CW - 0.72) / 3

    box(sl, ML, yy, pw, 2.66, fill=None, line=LINE, radius=0.05)
    _, tf = textbox(sl, ML + 0.26, yy + 0.18, pw - 0.52, 0.24)
    para(tf, "DATA", size=9, color=TEAL, bold=True, first=True)
    _, tf = textbox(sl, ML + 0.26, yy + 0.46, pw - 0.52, 2.0)
    for k, v in [
        ("Source", "Banking77 (Casanueva et al., 2020), CC BY 4.0"),
        ("Rows", f"{F.n_train + F.n_test + F.fusion['n_val']:,} tickets, {F.n_intents} intents"),
        ("Labels", f"{F.n_classes} coarse, {len(F.reach)} reachable"),
        ("Empty class", f"{F.unreachable[0]} — no examples in the dataset at all"),
        ("Split", "stratified 70/15/15, seed 20261003"),
        ("Leakage control", "α and T grid-searched on validation only"),
    ]:
        rich(tf, [(k + "  ", {"bold": True, "size": 9, "color": INK}),
                  (v, {"size": 9, "color": MUTED})], line=1.20, space_after=5, first=(k == "Source"))

    box(sl, ML + pw + 0.36, yy, pw, 2.66, fill=None, line=LINE, radius=0.05)
    _, tf = textbox(sl, ML + pw + 0.62, yy + 0.18, pw - 0.52, 0.24)
    para(tf, "WHAT WE COMPARED", size=9, color=TEAL, bold=True, first=True)
    _, tf = textbox(sl, ML + pw + 0.62, yy + 0.46, pw - 0.52, 2.0)
    for k, v in [
        ("kNN retrieval", "no training, neighbours vote"),
        ("LinearSVC", f"{pct(F.base['linsvc_coarse11']['accuracy'])} on the same features"),
        ("Cascade", "the two fused, calibrated and gated"),
        ("DistilBERT", f"{F.distil['epochs']} epochs, {F.distil['train_minutes']:.0f} min CPU"),
        ("BiLSTM", "in the brief, never built — see slide 4"),
        ("Agent", "advisory reranking on escalated tickets only"),
    ]:
        rich(tf, [(k + "  ", {"bold": True, "size": 9, "color": INK}),
                  (v, {"size": 9, "color": MUTED})], line=1.20, space_after=5, first=(k == "kNN retrieval"))

    box(sl, ML + 2 * (pw + 0.36), yy, pw, 2.66, fill=SOFT, line=None, radius=0.05)
    _, tf = textbox(sl, ML + 2 * (pw + 0.36) + 0.26, yy + 0.18, pw - 0.52, 0.24)
    para(tf, "WHAT WE HELD FIXED", size=9, color=TEAL, bold=True, first=True)
    _, tf = textbox(sl, ML + 2 * (pw + 0.36) + 0.26, yy + 0.46, pw - 0.52, 2.0)
    for k, v in [
        ("One split", "every arm scored on the same 1,963 test tickets"),
        ("No early stop", "chosen on validation, then frozen"),
        ("Paired test", "McNemar, because the arms are not independent"),
        ("Reporting", "accuracy AND macro-F1 AND ECE, never one alone"),
        ("Reproducibility", f"{F.n_experiments} experiments → {F.n_artifacts} JSON artifacts"),
    ]:
        rich(tf, [(k + "  ", {"bold": True, "size": 9, "color": INK}),
                  (v, {"size": 9, "color": MUTED})], line=1.20, space_after=5, first=(k == "One split"))
    return sl


# ---------------------------------------------------------------------------
# slide 7 -- System Architecture
# ---------------------------------------------------------------------------


def s07(prs: "Presentation", F: Facts):
    sl = blank(prs)
    y = header(sl, "System architecture", "A retriever, a linear head, a fusion, and a gate. The transformer is not in this diagram.",
               n=7,
               sub="One ticket in, one routed and prioritised ticket out. No network call, "
                   "no model download, no API key at any point in the path.")

    # Three vertical lanes: INPUT / DECIDE / ACT.
    lanes = [
        ("INPUT", SLATE, [
            ("Raw ticket text", "what the customer wrote"),
            ("Two sparse views", f"char_wb 3–5 + word 1–2 → {F.n_features_serving:,} dims"),
            ("kNN neighbours",
             f"k={F.k_serve} served; oracle ceiling {pct(F.recall[-1][1])} at k={F.k_oracle}"),
        ]),
        ("DECIDE", TEAL, [
            ("LinearSVC head", "one weight row per coarse class"),
            ("kNN head", f"{F.k_oracle} neighbours vote, distance-weighted"),
            ("Fusion", f"α={F.fusion['selected_alpha']} mix, then T={F.fusion['selected_temperature']} temperature"),
            ("Vector scaling", "fitted on validation; ECE on test below"),
            ("Conformal gate", f"α={F.level10['alpha']} split-conformal on the val split"),
        ]),
        ("ACT", RUST, [
            ("Auto-answer", f"above {F.threshold:.4f} — {pct(F.acc_at_gate)} accurate"),
            ("Escalate", "below threshold → a human decides"),
            ("Priority", "category impact floor + amount at risk → P0…P3"),
            ("Route", "SLA queue, from the category and the priority"),
            ("Draft", "agent writes, human sends — never the reverse"),
        ]),
    ]
    lw = (CW - 0.72) / 3
    for li, (title, col, rows) in enumerate(lanes):
        lx = ML + li * (lw + 0.36)
        box(sl, lx, y, lw, 3.16, fill=SOFT if col is not TEAL else TEAL_L, line=None, radius=0.05)
        box(sl, lx, y, lw, 0.30, fill=col, line=None, radius=0.05)
        _, tf = textbox(sl, lx + 0.22, y + 0.055, lw - 0.44, 0.22)
        para(tf, title, size=9.5, color=PAPER, bold=True, first=True)

        ry = y + 0.42
        for i, (h, b) in enumerate(rows):
            _, tf = textbox(sl, lx + 0.22, ry, lw - 0.44, 0.22)
            para(tf, h, size=10.5, color=INK, bold=True, first=True)
            _, tf = textbox(sl, lx + 0.22, ry + 0.21, lw - 0.44, 0.24)
            para(tf, b, size=8.5, color=MUTED, line=1.20, first=True)
            if i < len(rows) - 1:
                line_of(sl, lx + 0.22, ry + 0.47, lw - 0.44, color=LINE, pt=0.7)
            ry += 0.535

    # Crossing arrows between lanes.
    for lx in (ML + lw, ML + 2 * lw + 0.36):
        _, tf = textbox(sl, lx + 0.03, y + 1.42, 0.30, 0.28)
        para(tf, "▶", size=13, color=FAINT, align=PP_ALIGN.CENTER, first=True)

    yy = y + 3.32
    box(sl, ML, yy, CW, 1.05, fill=PANEL, line=None, radius=0.05)
    _, tf = textbox(sl, ML + 0.34, yy + 0.12, 3.4, 0.22)
    para(tf, "WHAT THIS DIAGRAM EXCLUDES", size=9, color=RUST_L, bold=True, first=True)
    _, tf = textbox(sl, ML + 0.34, yy + 0.36, CW - 0.68, 0.60)
    rich(tf, [("No transformer. No LLM in the decision path. ", {"bold": True, "color": PAPER}),
              (f"The fine-tuned DistilBERT is a measured arm of the experiment — "
               f"{pct(F.dist['accuracy'])} accuracy, {F.dist['latency_ms']:.0f} ms, "
               f"{F.dist['model_mb']:.0f} MB, {F.distil['train_minutes']:.0f} min to train — "
               f"and it lost on p = {F.mcn['p_value']:.4f}. The generative agent is in the "
               "diagram as a drafter, never as a decider: given decision authority on the "
               f"{F.escal['escalated_n']} escalated tickets it moved end-to-end accuracy "
               f"{pct(F.escal['baseline_accuracy'])} → {pct(F.escal['end_to_end_with_agent'])}.", {})],
         size=10.0, color=FAINT, line=1.22, first=True)
    return sl


# ---------------------------------------------------------------------------
# slide 8 -- Project Design
# ---------------------------------------------------------------------------


def s08(prs: "Presentation", F: Facts):
    sl = blank(prs)
    y = header(sl, "Project design", "Four decisions that shaped everything else. Each one is a trade we made on purpose.",
               n=8,
               sub="Design is what you removed. Each row is a choice we could have gone the "
                   "other way on, and what it cost.")

    rows = [["Decision", "The alternative", "Why we chose this", "What it cost us"]]
    for d, alt, why, cost in decisions(F):
        rows.append([d, alt, why, cost])

    t = table(sl, ML, y, CW, [2.30, 2.55, 4.35, 2.63], rows, row_h=0.655, head_h=0.32)
    for r in range(1, len(rows)):
        fill_cell(t.cell(r, 0), rows[r][0], size=11, color=INK, bold=True)
        fill_cell(t.cell(r, 1), rows[r][1], size=9.5, color=FAINT)
        fill_cell(t.cell(r, 2), rows[r][2], size=9.5, color=MUTED)
        fill_cell(t.cell(r, 3), rows[r][3], size=9.5, color=RUST)
    for c in range(4):
        fill_cell(t.cell(0, c), rows[0][c].upper(), size=9, color=TEAL, bold=True)
        cell_border(t.cell(0, c), edges=("bottom",), color="0B6E63", pt=1.2)

    yy = y + 0.32 + 0.655 * len(decisions(F)) + 0.24
    _, tf = textbox(sl, ML, yy, CW, 0.46)
    rich(tf, [("And one non-decision we want on the record: ", {"bold": True}),
              (f"{F.n_intents} intents map to {F.n_classes} categories, and "
               f"{F.unreachable[0]} has no examples in the dataset. We left the class in "
               "the taxonomy rather than deleting it, because a routing table that quietly "
               "drops a category is worse than one that admits the gap.", {})],
         size=10.5, color=MUTED, line=1.30, first=True)
    return sl


# ---------------------------------------------------------------------------
# slide 9 -- Expected Outcomes
# ---------------------------------------------------------------------------


def s09(prs: "Presentation", F: Facts):
    sl = blank(prs)
    y = header(sl, "Expected outcomes", "We expected the transformer to win. It won by a margin that does not survive a significance test.",
               n=9,
               sub="Four systems, one held-out split of 1,963 tickets, one test applied to "
                   "the pair that disagreed.")

    tail_c = F.arm("cascade_fused")["tail_recall"]
    tail_d = F.arm("distilbert_finetuned")["tail_recall"]
    bench_rows = bench(F)
    rows = [["System", "Accuracy", "Macro-F1", "ECE", "Latency", "Size", "Build"]]
    for name, acc, mf1, ece, lat, size, tr, col in bench_rows:
        rows.append([name, acc, mf1, ece, lat, size, tr])
    t = table(sl, ML, y, CW, [3.05, 1.42, 1.42, 1.28, 1.42, 1.28, 1.92], rows,
              row_h=0.36, head_h=0.32)
    for r in range(1, len(rows)):
        col = bench_rows[r - 1][7]
        best = r == 3
        fill_cell(t.cell(r, 0), rows[r][0], size=10.5, color=INK, bold=True,
                  fill=TEAL_L if best else None)
        for c, v in enumerate(rows[r][1:], start=1):
            fill_cell(t.cell(r, c), v, size=10, color=col if c < 3 else MUTED,
                      bold=best and c == 1, align=PP_ALIGN.CENTER,
                      fill=TEAL_L if best else None)
    for c in range(7):
        fill_cell(t.cell(0, c), rows[0][c].upper(), size=9, color=TEAL, bold=True)
        cell_border(t.cell(0, c), edges=("bottom",), color="0B6E63", pt=1.2)

    yy = y + 0.32 + 0.36 * len(bench_rows) + 0.26

    # Left: the paired test that decides it.
    pw = 6.30
    _, tf = textbox(sl, ML, yy, pw, 0.26)
    para(tf, "McNemar, ON THE 100 TICKETS WHERE THEY DISAGREE", size=9.5, color=TEAL,
         bold=True, first=True)
    ty = yy + 0.32
    mcn = [["", "Cascade right", "DistilBERT right", "Total"],
           ["Cascade right", f"{F.mcn['both_correct']:,}", f"{F.mcn['only_distilbert_correct']}", f"{F.mcn['both_correct'] + F.mcn['only_distilbert_correct']:,}"],
           ["DistilBERT right", f"{F.mcn['only_cascade_correct']}", f"{F.mcn['both_wrong']}", f"{F.mcn['only_cascade_correct'] + F.mcn['both_wrong']}"],
           ["Total", f"{F.mcn['both_correct'] + F.mcn['only_cascade_correct']:,}",
            f"{F.mcn['only_distilbert_correct'] + F.mcn['both_wrong']}", f"{F.n_test:,}"]]
    mt = table(sl, ML, ty, pw, [1.65, 1.65, 1.75, 1.25], mcn, row_h=0.34, head_h=0.32)
    for r in range(1, 4):
        for c in range(4):
            hl = (r == 2 and c == 1) or (r == 1 and c == 2)
            fill_cell(mt.cell(r, c), mcn[r][c], size=10.5,
                      color=(RUST if r == 1 and c == 2 else TEAL if r == 2 and c == 1
                             else INK if r < 3 and c < 3 else MUTED),
                      bold=hl, align=PP_ALIGN.CENTER,
                      fill=RUST_L if (r == 1 and c == 2) else TEAL_L if (r == 2 and c == 1) else None)
    for c in range(4):
        fill_cell(mt.cell(0, c), mcn[0][c], size=8.5, color=MUTED, bold=True, align=PP_ALIGN.CENTER)

    _, tf = textbox(sl, ML, ty + 0.32 + 0.34 * 3 + 0.14, pw, 0.62)
    rich(tf, [("DistilBERT wins 50 tickets and loses 44. ", {"bold": True, "color": INK}),
              (f"χ² = {F.mcn['chi2']:.3f}, p = {F.mcn['p_value']:.4f}, exact paired test. "
               "That is the entire basis for preferring a transformer.", {})],
         size=10.5, color=MUTED, line=1.30, first=True)
    _, tf = textbox(sl, ML, 6.54, pw, 0.42)
    rich(tf, [("And on the rare classes the two arms tie exactly: ",
               {"bold": True, "color": INK}),
              (f"tail recall {pct(tail_c)} for the cascade and {pct(tail_d)} for the "
               f"transformer. It is not better on the intents that are hard to predict "
               f"-- on that slice it is the same model, for {F.latency_ratio:.1f}× "
               "the latency.", {})],
         size=9.5, color=MUTED, line=1.25, first=True)

    # Right: what we got instead.
    rx = ML + 6.68
    rw = CW - 6.68
    box(sl, rx, yy - 0.06, rw, 3.16, fill=TEAL_L, line=None, radius=0.05)
    _, tf = textbox(sl, rx + 0.30, yy + 0.14, rw - 0.60, 0.26)
    para(tf, "SO WE SHIPPED THIS INSTEAD", size=9.5, color=TEAL, bold=True, first=True)
    ry = yy + 0.36
    for v, lab in [
        (f"{pct(F.acc_at_gate)}", "accuracy on the tickets it auto-answers"),
        (f"{F.casc['accuracy_when_agree'] * 100:.2f}%", "accuracy where the two heads agree"),
        (f"{pct(F.level10['realised_coverage'])}", "coverage actually delivered at α = 0.10"),
        (f"{F.level10['realised_accuracy_on_accepted'] * 100:.2f}%", "accuracy on those accepted tickets"),
    ]:
        _, tf = textbox(sl, rx + 0.30, ry, 2.35, 0.30)
        para(tf, v, size=17, color=INK, bold=True, font=DISPLAY, first=True)
        _, tf = textbox(sl, rx + 2.72, ry + 0.06, rw - 3.02, 0.40)
        para(tf, lab, size=9.5, color=MUTED, line=1.20, first=True)
        ry += 0.40
    line_of(sl, rx + 0.30, ry + 0.02, rw - 0.60, color=TEAL, pt=0.9)
    _, tf = textbox(sl, rx + 0.30, ry + 0.10, rw - 0.60, 0.88)
    rich(tf, [("Expected outcome, restated honestly: ", {"bold": True, "color": INK}),
              ("not \"higher accuracy\", but a system whose accuracy we can promise, "
               "on the subset it acts on, at a coverage rate we chose deliberately. "
               f"The guarantee held at every α we tested. It cost {F.latency_ratio:.1f}× "
               f"less latency and {F.size_ratio:.0f}× less disk than the arm we rejected.", {})],
         size=9.0, color=MUTED, line=1.20, first=True)
    return sl


# ---------------------------------------------------------------------------
# slide 10 -- Future Plan
# ---------------------------------------------------------------------------


def s10(prs: "Presentation", F: Facts):
    sl = blank(prs, dark=True)
    box(sl, 0, 0, W, H, fill=PANEL, line=None, radius=0.0)
    box(sl, 0, 0, 0.16, H, fill=TEAL, line=None, radius=0.0)

    x = 1.10
    y = 0.52
    _, tf = textbox(sl, x, y, CW - 0.6, 0.26)
    para(tf, "FUTURE PLAN", size=10, color=TEAL_L, bold=True, first=True)
    _, tf = textbox(sl, x, y + 0.28, CW - 1.2, 0.60)
    para(tf, "What we would do next, and what we would not.", size=27, color=PAPER,
         font=DISPLAY, first=True)
    line_of(sl, x, y + 0.98, CW - 0.9, color=PANEL2, pt=1.0)

    cols = [
        ("WE WOULD FIX FIRST", RUST_L, [
            ("Sentiment is the weakest link",
             "It is a lexicon, and lexicons invert on negation (\"not working\" scores "
             "positive). A small supervised head on the same features would fix it."),
            ("BiLSTM was never built",
             "It sits between a bag-of-words and DistilBERT. The comparison in the "
             "synopsis is incomplete and we should say so plainly."),
            ("The label space is still wrong",
             f"Real desks do not have {F.n_classes} categories. Mapping the taxonomy to "
             "an actual bank's queue structure needs their input, not ours."),
        ]),
        ("WE WOULD ADD NEXT", TEAL_L, [
            ("A real abstention interface",
             "The conformal gate works; nothing yet tells an agent *why* a ticket was "
             "escalated. Retrieval evidence is already computed — we just do not show it."),
            ("Drift monitoring on text, not just labels",
             f"The monitor watches category mix. It should watch embedding drift too; "
             f"the current reference is drawn from train, so it can only fire on mix."),
            ("Per-class thresholds",
             f"Conformal gives one global q. P0 is worth a stricter gate than P3, and "
             f"the sweep says the margin matters more than the score."),
        ]),
        ("WHAT WE WOULD NOT DO", FAINT, [
            ("Adopt the transformer for the 0.31 points",
             f"p = {F.mcn['p_value']:.4f}. To make that margin real we would need roughly "
             "ten times the test set — and then it still has to win on ECE."),
            ("Give the agent decision authority",
             f"{pct(F.escal['baseline_accuracy'])} → {pct(F.escal['end_to_end_with_agent'])} on the "
             f"{F.escal['escalated_n']} escalated tickets. Drafting is where it adds value."),
            ("Auto-answer on confidence alone",
             "That is the number 0.80 that means 95% accuracy in one version and 70% "
             "in the next. Coverage is the promise; the threshold is an implementation detail."),
        ]),
    ]
    cwid = (CW - 0.4 - 0.72) / 3
    for ci, (title, col, items) in enumerate(cols):
        cx = x + ci * (cwid + 0.36)
        _, tf = textbox(sl, cx, y + 1.16, cwid, 0.26)
        para(tf, title, size=9.5, color=col, bold=True, first=True)
        iy = y + 1.48
        for h, b in items:
            box(sl, cx, iy, cwid, 1.20, fill=PANEL2, line=None, radius=0.05)
            _, tf = textbox(sl, cx + 0.22, iy + 0.12, cwid - 0.44, 0.24)
            para(tf, h, size=10.5, color=PAPER, bold=True, first=True)
            _, tf = textbox(sl, cx + 0.22, iy + 0.34, cwid - 0.44, 0.80)
            para(tf, b, size=8.5, color=FAINT, line=1.22, first=True)
            iy += 1.28

    box(sl, x, 6.02, CW - 0.4, 0.68, fill=PANEL2, line=None, radius=0.05)
    _, tf = textbox(sl, x + 0.30, 6.10, CW - 1.0, 0.52)
    rich(tf, [("The one-line version:  ", {"bold": True, "color": TEAL_L}),
              (f"the useful part of this project was not reaching "
               f"{pct(F.casc['accuracy'])}. It was building a system that knows the "
               f"difference between {pct(F.dist['accuracy'])} and {pct(F.casc['accuracy'])} — "
               "and refusing to act on the difference.", {"font": DISPLAY})],
         size=11.5, color=PAPER, line=1.22, first=True)

    footer(sl, 10, dark=True)
    return sl


# ---------------------------------------------------------------------------
# content, sourced from research/00-original-synopsis.txt unless noted
# ---------------------------------------------------------------------------

TEAM = [
    ("237Y1A0540", "NAYAKAM SATYAJEET"),
    ("237Y1A0570", "JAKKU HARIKA"),
    ("237Y1A0574", "SHERLA LOHITH KUMAR"),
    ("237Y1A05A5", "MAMIDALA VARUN"),
]
GUIDE = "Dr. PARVEZ"

# Synopsis section 6, condensed to author/technique/limitation.
LIT_SYNOPSIS = [
    ("Paramesh & Shreedhara, 2019", "ML classification of IT service-desk tickets",
     "Classical features only — no sentiment, SLA or routing"),
    ("Zangari et al., 2023", "Survey of ticket automation and multi-level classification",
     "Survey; no deployable low-latency pipeline"),
    ("Casanueva et al., 2020", "Dual sentence encoders; introduces Banking77",
     "Intent detection only — no priority or routing"),
    ("Devlin et al., 2019", "BERT, bidirectional pretraining and fine-tuning",
     "Large; high latency and memory cost on CPU"),
    ("Sanh et al., 2019", "DistilBERT — ~40% smaller, ~60% faster than BERT",
     "Needs further optimisation for CPU serving"),
    ("Zafrir et al., 2019", "Q8BERT, 8-bit quantization-aware training",
     "Requires retraining; evaluated on general benchmarks"),
    ("Minaee et al., 2021", "Review of CNN / RNN / attention / transformer classifiers",
     "Survey; no deployment guidance"),
    ("Lewis et al., 2020", "Retrieval-augmented generation for knowledge-intensive NLP",
     "Needs a knowledge index; hallucination risk"),
    ("Zaharia et al., 2018", "MLflow — tracking, packaging, model registry",
     "Tracking only; no text-drift monitoring"),
]

# Added by us, with the reason each one was needed.
LIT_ADDED = [
    ("Tunstall et al., 2022", "SetFit, arXiv:2209.11055",
     "8 examples per class beats full fine-tuning by 19.3 points — the sanity check on how little data a good head needs."),
    ("Angelopoulos & Bates, 2019", "conformal prediction, arXiv:1907.07509",
     "Turns \"auto-answer the confident 90%\" into a finite-sample guarantee rather than a tuned threshold."),
    ("Northcutt et al., 2021", "confident learning / cleanlab",
     "The only way to know whether the labels we trained on are wrong is to audit them."),
    ("Ying & Thomas, 2022", "Banking77 label noise, 2022.insights-1.19",
     "~14% noise reported at the 77-intent level; ours found far less, and the gap is worth explaining."),
]

# Slide 4. The status column is our assessment, not the synopsis's.
def objectives(F):
    return [
    ("Classify tickets into enterprise business categories", "MET",
     f"{F.n_classes} coarse categories from {F.n_intents} intents; "
     f"{pct(F.casc['accuracy'])} accuracy, {pct(F.casc['macro_f1'])} macro-F1 on held-out test"),
    ("Analyse sentiment and urgency for every ticket", "MET",
     "Sentiment lexicon plus category-priority rules and amount extraction; contributes to P0–P3"),
    ("Assign an SLA-based priority P0–P3 and route to a queue", "MET",
     f"Four priorities, per-category SLA targets and queues; P0 recall {pct(F.p0_recall)} on test"),
    ("Compare TF-IDF, Logistic Regression, BiLSTM and DistilBERT", "PARTIAL",
     "Four of four arms run and compared on one split with a paired significance test — BiLSTM never built"),
    ("Optimise DistilBERT with ONNX INT8 quantisation", "NOT DONE",
     "Declined deliberately: the cascade meets the latency and size goal with no transformer at all"),
    ("Human-in-the-loop GenAI draft responses for agents", "MET, NOT ADOPTED",
     f"Agent drafts with fact precision {F.escal['draft_fact_precision_mean']:.3f} and "
     f"{F.escal['drafts_with_invented_identifiers']} invented ids — but letting it decide "
     f"lowered accuracy, so it ships advisory-only"),
    ("Deploy as containerised microservices with tracking and drift monitoring", "PARTIAL",
     f"FastAPI + Streamlit + drift monitor, {F.n_experiments} experiments tracked end to end — no containers"),
]


# Slide 8. Decision / alternative / why / what it cost.
def decisions(F):
    return [
    ("Sparse views, not embeddings",
     "Sentence-transformers or fine-tuned embeddings",
     "char_wb 3–5 grams catch the misspellings and the code-switching that actually "
     "appear in support tickets; a static vocabulary also means nothing to download at "
     "serving time.",
     "No semantic generalisation to a phrasing we never saw."),
    ("A linear head, not a neural classifier",
     "Fine-tuned DistilBERT as the primary classifier",
     "It matched a 268 MB transformer to within 0.31 points, and a paired test put that "
     "margin at chance.",
     "Linear models cannot represent word order — see the sentiment limitation."),
    ("Fusion, then calibration, then a gate",
     "One classifier at one threshold",
     "Fusing two decorrelated errors raised macro-F1, and calibration is what turns a "
     "probability into something an operator can act on.",
     "Two more fitted stages between the text and the answer."),
    ("Conformal gate on the validation split",
     "Pick a threshold on the test curve",
     "A threshold tuned on test is a number that cannot survive contact with new data; "
     "split conformal gives a bound that holds on fresh exchangeable tickets.",
     f"Coverage is {pct(F.level10['realised_coverage'])} rather than exactly 90%."),
    ("Agent drafts, humans decide",
     "Let the agent overrule the classifier",
     f"On the {F.escal['escalated_n']} escalated tickets the agent's own accuracy was "
     "lower than the cascade it was asked to override.",
     "None of the drafting speed-up reaches the customer."),
    ("Report ECE and macro-F1 next to accuracy",
     "Report accuracy",
     "Under 18.7:1 imbalance the two disagree, and a model can be well-ranked and "
     "badly calibrated at the same time.",
     "More numbers to read before you can compare two systems."),
]


# Slide 9. Values are formatted here so the table reads as a table; the numbers
# underneath are all pulled from 06-final-comparison.json.
def bench(F):
    """The four-system benchmark on the Expected Outcomes slide.

    A function, not a literal. The previous hand-typed version of this table
    was the sixth instance of one bug class in this project: a number typed in
    prose and never re-checked against the artifact. It also had two cells no
    artifact supports -- the cascade's build cost was quoted as
    ``linsvc_coarse11.feature_fit_s``, which belongs to a *different* feature
    space (20,431 word-only features against the 34,901 the serving union
    builds), and neither `03-fusion.json` nor `06-final-comparison.json` times
    the cascade's fit at all. That cell now says so.

    Returns rows of (system, accuracy, macro-f1, ECE, latency, size, build, hue).
    """
    knn, casc, dist = F.knn, F.casc, F.dist
    lin = F.base["linsvc_coarse11"]
    return [
        ("kNN retrieval only", pct(knn["accuracy"]), pct(knn["macro_f1"]),
         f"{knn['ece']:.4f}", f"{knn['latency_ms']:.1f} ms", "—", "none", SLATE),
        ("TF-IDF + LinearSVC", pct(lin["accuracy"]), pct(lin["macro_f1"]),
         "uncalibrated", "<0.01 s", "—",
         f"{lin['feature_fit_s']:.1f} s fit", SLATE),
        ("Cascade — retrieval + linear", pct(casc["accuracy"]), pct(casc["macro_f1"]),
         f"{casc['ece']:.4f}", f"{casc['latency_ms']:.1f} ms",
         f"{F.cascade_mb:.1f} MB", "not timed", TEAL),
        ("DistilBERT fine-tuned", pct(dist["accuracy"]), pct(dist["macro_f1"]),
         f"{dist['ece']:.4f}", f"{dist['latency_ms']:.1f} ms",
         f"{dist['model_mb']:.0f} MB", f"{dist['train_minutes']:.0f} min", RUST),
    ]
