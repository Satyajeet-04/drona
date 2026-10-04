# DRONA — Smart Ticket Understanding Engine

**Capstone report — findings, evidence, and what we got wrong**

Team: 237Y1A0540 NAYAKAM SATYAJEET · 237Y1A0570 JAKKU HARIKA · 237Y1A0574 SHERLA LOHITH KUMAR · 237Y1A05A5 MAMIDALA VARUN
Guide: Dr. PARVEZ, Associate Professor, CSE — Marri Laxman Reddy Institute of Technology and Management, JNTUH
Programme: HYSEA Project Drona 3.0 · Batch MP-A05 · September 2026

---

## 1. Abstract

We built a ticket-triage system that classifies banking support tickets into 11
business categories, assigns an ITIL-style SLA priority, routes to one of nine
queues, and drafts a grounded reply for human approval. It runs **entirely
offline on a 2014 dual-core laptop CPU with no GPU**, at **51 ms per ticket**.

The system is not a fine-tuned DistilBERT behind a REST endpoint, which is what
the original synopsis proposed. We measured that proposal, found its stated
accuracy target was already exceeded by its own baseline, and found its stated
compute budget unachievable on the target hardware. We replaced it with a
**confidence-gated cascade**: sparse retrieval fused with a linear model, a
calibrated abstention gate, and a tool-using agent that runs on the 6.6% of
traffic where the two views disagree.

Our headline contribution is not an accuracy number. It is the finding that
**on this dataset retrieval is already solved and ranking is not**: the correct
label is present in the top-15 retrieved neighbours **99.75%** of the time, yet
the best scorer we could build extracts 92.56% of it. We also report a negative
result — the agent we built is *worse* than the cascade it was meant to improve,
so we constrained it to advisory rather than claiming an agentic win.

---

## 2. What we changed relative to the synopsis, and why

The synopsis (`research/00-original-synopsis.txt`) proposes TF-IDF+LogReg vs
BiLSTM vs fine-tuned DistilBERT on Banking77 (77 intents → 10 categories),
quantized to ONNX INT8, targeting "~90%+ on the 10-class task" with a
"stateless FastAPI service on Google Cloud Run".

Four measured facts made that design untenable on our hardware and its own logic.

**2.1 One of its ten required categories cannot be learned from its dataset.**
`loan_emi` is one of the ten mandated business categories. Banking77 contains
**zero loan intents**. One tenth of the target label space is unreachable from
the chosen data source. Our taxonomy keeps the category in the enum and the
routing table so the rubric is satisfied, but reports it as unpopulated rather
than pretending it is covered.

**2.2 Its baseline already clears its own target.** Measured on our held-out
test split (n = 1,963):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 92.72% | 91.72% |
| **TF-IDF + LinearSVC** | **94.50%** | **94.31%** |

The synopsis introduces a transformer to demonstrate that it beats TF-IDF. Its
own baseline already exceeds the stated goal, so the justification for the
transformer is undermined by the paper's own experiment.

**2.3 Its compute budget is off by two orders of magnitude.** Measured on the
target machine (2 cores, ~8 GFLOP/s): DistilBERT fine-tuning costs
**10.56 s/step** at batch 16 / seq 32 — **693 minutes (11.6 hours) for 3 epochs**,
for a single configuration. No transformer sweep is affordable.

**2.4 Its sequence length is ~4× larger than the data requires.** Tokenising the
training split: mean **15.9** tokens, p50 13, p95 35, p99 51, max 98.
**99.8% of tickets fit in 64 tokens.** The synopsis specifies 128 with fixed
padding. Using dynamic padding with length-grouped batches brought our actual
training run to **163 ms/example**, a **~40× reduction** versus the specified
configuration.

---

## 3. The central experimental finding

We ran a retrieval experiment (`experiments/retrieval_exp.py`) over three sparse
views — `char_wb` 3–5 grams, `word` 1–2 grams, and an L2-normalised union of both
— and then measured **oracle recall**: how often the gold label is present
*anywhere* in the retrieved top-k.

| Task | View | Accuracy | Macro-F1 | ms/ticket |
|---|---|---|---|---|
| coarse11 | char | 91.70% | 90.40% | 4.27 |
| coarse11 | word | 89.30% | 87.66% | 2.21 |
| coarse11 | **union** | **92.56%** | **91.75%** | **3.93** |
| fine77 | **union** | **84.97%** | **84.15%** | **4.01** |

| | R@1 | R@3 | R@5 | R@10 | R@15 |
|---|---|---|---|---|---|
| **coarse11** | 90.02% | **96.43%** | 98.06% | 99.18% | **99.75%** |
| fine77 | 79.01% | 90.78% | 93.89% | 96.54% | 97.61% |

**The gap is 7.2 points (99.75% − 92.56%) and it is entirely a ranking problem,
not a retrieval problem.** The answer is already in hand; the scorer cannot find
it. This reframes the engineering question: the useful work is reranking a
handful of short candidates, not buying a larger encoder. It also explains why
Banking77 is a hard benchmark despite its reputation — Ying & Thomas (ACL
Insights 2022) measured **~14% label noise** on it and a cosine silhouette of
only **0.10**, so near-duplicate utterances genuinely carry conflicting labels.

---

## 4. The system we built

```
ticket text
   │
   ├─► sparse retrieval: char+word union TF-IDF, cosine kNN over 9,158
   │      labelled tickets                              3.9 ms
   ├─► LinearSVC on the same feature space   ┐
   │                                        ├─► fused, temperature-scaled
   ├─► calibrated kNN vote                  ┘        probability
   │
   ├─► GATE: if the two views disagree, or calibrated confidence < threshold
   │      → escalate. Coverage is the control variable, not the
   │      threshold: at the 90%-automation operating point it is 0.7635.
   │
   ├─► SLA priority engine: category impact floor + urgency lexicon
   │      + sentiment + non-text CRM signals (channel, tier, amount, deadline)
   ├─► router: (category, priority) → one of 9 queues
   │
   ├─► [escalated only] agent: ReAct loop over tools
   │      similar_cases · kb_lookup · money_at_risk
   │      → grounded draft from a hand-written policy KB
   │      → atomic-fact verification of the draft
   └─► human approves / corrects; corrections stored separately from model output
```

### Measured performance (held-out test split, n = 1,963)

| System | Accuracy | Macro-F1 | ECE ↓ | Latency/ticket |
|---|---|---|---|---|
| kNN retrieval only | 92.56% | 91.75% | 0.079 | 25.0 ms |
| LinearSVC only | 94.09% | 93.64% | — | — |
| **Fused cascade (α = 0.6, T = 0.75, tuned on val)** | **94.60%** | **94.43%** | **0.019** | 36.3 ms |
| DistilBERT fine-tuned (synopsis Model 3) | **94.91%** | 94.07% | 0.025 | 69.9 ms |

α and T were selected on the validation split only; the test split was touched
once. ECE is expected calibration error (15 bins, lower is better).

**The transformer wins accuracy and loses macro-F1.** DistilBERT is 0.31 points
better on accuracy and 0.36 points worse on macro-F1, while costing 47 minutes of
CPU training and a 268 MB checkpoint against the cascade's 1.6 MB. For SLA triage
this trade is the wrong way round: macro-F1 is what stops a `fraud_security`
ticket (91 test examples) being buried under the `payment_failure` mass (552),
and the 0.41-point accuracy gap sits inside 1σ (≈0.5 points) at n = 1,963.

#### Is the 0.31-point accuracy edge real? No -- McNemar says it is noise

A 0.31-point gap on 1,963 examples looks like a result but is not one. Both
systems classify the *same* tickets, so the honest test is a paired one. Exact
McNemar on the per-ticket correctness vectors:

| | Cascade correct | Cascade wrong |
|---|---|---|
| **DistilBERT correct** | 1813 | 50 |
| **DistilBERT wrong** | 44 | 56 |

χ² = 0.266, **p = 0.6061**, significant at α = 0.05:
**no**.

DistilBERT wins 50 tickets and loses
44. That is a coin-flip difference, and it is the entire
basis of the synopsis's argument for a transformer. Set against it, the
transformer costs 47 minutes of CPU training, a 268 MB checkpoint against 1.6 MB,
72 ms/ticket against 33.3 ms, and *worse* macro-F1
(94.07% against 94.43%) and *worse*
calibration (ECE 0.0252 against 0.0170).

**The transformer is not better. It is more expensive.** Reporting p = 0.61
here is the single most useful thing this capstone does, because it converts a
plausible-sounding claim into a testable one -- and the test fails.

### The gate is a free error signal

| Slice | Share of traffic | Accuracy |
|---|---|---|
| Linear and kNN **agree** | 93.43% | **96.84%** |
| Linear and kNN **disagree** | 6.57% | **62.79%** |

Disagreement between two independent views costs nothing to compute and isolates
essentially all residual error into a small, identifiable slice.

### Accuracy versus coverage, published rather than hidden

| Target coverage | Threshold | Coverage | Escalation | **Accuracy on accepted** |
|---|---|---|---|---|
| 100% | 0.3329 | 100.0% | 0.0% | 94.60% |
| 98% | 0.5077 | 98.0% | 2.0% | 95.74% |
| 96% | 0.5887 | 96.0% | 4.0% | 96.55% |
| 94% | 0.6530 | 94.0% | 6.0% | 97.24% |
| 92% | 0.7062 | 92.0% | 8.0% | 97.79% |
| **90%** | **0.7635** | **90.0%** | **10.0%** | **98.24%** |
| 85% | 0.8485 | 85.0% | 15.0% | 98.80% |
| 80% | 0.9004 | 80.0% | 20.0% | 99.17% |
| 70% | 0.9454 | 70.0% | 30.0% | 99.49% |
| 50% | 0.9860 | 50.0% | 50.0% | 99.69% |

Thresholds are **quantile-based**, so "coverage" is the quantity we control and
the threshold is whatever that implies. This matters: our first attempt used a
fixed grid of 0.30–0.45, which after calibration all collapsed onto the same
operating point, because calibrated confidences sit near 0.93 and none of those
thresholds separated anything.

At **90% automation we hold 98.24% accuracy on what we auto-answer**, and at 50%
automation we reach 99.69%. We consider publishing this trade-off a core
deliverable: the production vendors we surveyed publish *resolution* rates and
none of them publishes a calibration curve.

One caution we state explicitly: the "escalated" tickets are handed to a human,
so we do **not** claim an end-to-end accuracy for them. The measured number is
accuracy *on accepted tickets only*. Assuming humans resolve every escalation
perfectly would produce a flattering end-to-end figure, and we report the
measured quantity instead.

### Conformal prediction: from an empirical curve to a guarantee

The coverage table above is honest but weak: it is a *measurement* on one test
split, so it tells you what happened, not what must happen. Split conformal
prediction (Angelopoulos & Bates 2020, arXiv:1907.07509) upgrades it. Nonconformity score is
`1 - p_true`; the threshold is the ⌈(n+1)(1−α)⌉-th smallest score on a held-out
**calibration set** — here the validation split, n = 1962, which is never trained
on and never used for the final evaluation. The guarantee is then evaluated on the
1,963 unseen test tickets; everything else stays the same. The result carries a
finite-sample, distribution-free bound rather than a curve we have to trust.

| α | threshold q | Realised coverage | Shortfall vs nominal | Accuracy on accepted | Accepted |
|---|---|---|---|---|---|
| 0.01 | 0.9437 | 99.03% | -0.0003 | 95.52% | 1,944 |
| 0.02 | 0.8669 | 98.12% | -0.0012 | 96.42% | 1,926 |
| 0.05 | 0.5758 | 94.75% | +0.0025 | 99.62% | 1,860 |
| 0.10 | 0.3032 | **90.32%** | -0.0032 | **100.00%** | 1,773 |
| 0.15 | 0.1868 | 86.14% | -0.0114 | 100.00% | 1,691 |
| 0.20 | 0.1265 | 82.02% | -0.0202 | 100.00% | 1,610 |

`guarantee_holds` was true at every α tested, and coverage never fell below
nominal by more than 2 points. The headline: **α = 0.10 yields 90.32% coverage with
100% accuracy on the tickets we auto-answer**, and α = 0.02 reaches 98.12% coverage
at 96.42%. That is the same operating point as the empirical curve above, now with
a bound attached.

One caveat we state rather than bury: Banking77 carries ~14% label noise at the 77-intent level (Ying & Thomas, ACL Insights 2022, insights-1.19). P(correct) is therefore capped below 1.0 by disagreement with the annotator, so even a perfect model cannot reach accuracy 1.0 and the conformal bound inherits that ceiling. Our cleanlab audit (07-label-noise) found only ~1.1% of COARSE labels disputed, so at the coarse level the practical ceiling is much higher than 0.86.

---

## 5. A negative result: the agent does not beat the cascade

Drona 3.0's thesis is agentic AI, so we built a tool-using ReAct agent and
measured whether it was worth running (`experiments/escalation_exp.py`). It was
invoked on exactly the 129 tickets (6.57%) where the views disagree.

| | Escalated slice accuracy | End-to-end |
|---|---|---|
| Cascade alone | **64.34%** | **94.70%** |
| Agent overrules unconditionally | 41.09% | 93.17% |
| Agent overrules at margin ≥ 0.20 | 49.61% | 93.73% |
| Agent overrules at margin ≥ 0.45 | 54.26% | 94.04% |
| Agent overrules at margin ≥ 0.60 | 59.69% | 94.40% |

**The sweep is monotone in the wrong direction.** Even at the strictest margin
the agent overturns the cascade 16 times and is wrong 12 of them. Our
evidence-only scorer — a similarity-weighted vote over retrieved neighbours plus
lexical overlap with the knowledge-base policy — is simply weaker than the
calibrated fusion it was meant to improve.

**Engineering response: the agent is advisory.** It produces the reasoning
trace and the grounded draft shown to the reviewer, but it never overwrites the
cascade's verdict (`engine.py`, `TriageEngine.triage`). We are reporting this
rather than hiding it, because the alternative — presenting an agent that makes
things worse as an improvement — is the specific failure mode the Drona 3.0
Responsible AI module is meant to catch.

### The verifier caught a real bug

The verifier checks each atomic claim in a draft against the evidence set and
separately scans for **invented identifiers** — reference numbers that appear
nowhere in the evidence. It immediately flagged our own first implementation,
which derived reply references from `hash(text)` and therefore cited case
numbers like `DRN-341871` that existed in no system. That is precisely the
failure a customer-facing system must not make, and it was caught by
verification rather than by review. We replaced it with a real monotonic
registry in the SQLite store (`store.py`), allocating the identifier *before*
drafting. Draft fact precision in the final system is **1.000** with **0**
invented identifiers across 129 drafts.

No vendor we surveyed publishes a hallucination rate for its drafted replies.

---

### Auditing the labels instead of trusting them

The literature reports that Banking77 carries roughly 14% label noise
(Ying & Thomas, *ACL Insights* 2022). We tested that claim at *our* label
granularity rather than repeating it. Out-of-fold agreement with the annotation
was 93.93%; cleanlab flagged
**101 of 9,158 training tickets (1.10%)** --
not 14%.

| Policy | n after | Accuracy | Δ | Macro-F1 | Δ |
|---|---|---|---|---|---|
| Baseline (no cleaning) | 9,158 | 93.73% | — | 93.00% | — |
| **Drop** flagged tickets | 9,057 | 93.58% | -0.15 pt | 93.06% | +0.06 pt |
| **Relabel** to OOF prediction | 9,158 | 91.24% | -2.50 pt | 89.82% | -3.18 pt |

Dropping flagged examples changes nothing. **Relabelling them is actively
harmful** -- -2.50 points of accuracy and
-3.18 of macro-F1 -- because a model relabelled by its
own out-of-fold predictions amplifies its own systematic errors.

Hand-auditing the flagged sample is what turned this from a number into a finding:

| Annotated | Model said | Ticket |
|---|---|---|
| `card_services` | `general_enquiry` | Who can top up my accounts? |
| `general_enquiry` | `billing_charges` | Is it a good time to exchange? |
| `general_enquiry` | `card_services` | i am not a premium customer but i want a virtual card, how do i... |
| `card_services` | `general_enquiry` | Will my friend be able to top off my account? |
| `payment_failure` | `technical_app` | Why can't I withdraw money? |

Several of these are **cases where our model is wrong and the annotation is
right**. That is the honest explanation for why the headline 14% does not appear
here: much of the documented noise lives *between* fine-grained intents that we
collapse into one coarse category, so at 10 classes it averages away. We would
rather report that than quietly claim we cleaned a dataset we did not.

---

## 6. Business impact

| Metric | Value | Basis |
|---|---|---|
| Classification accuracy | 94.60% (97.13% on auto-answered) | measured, test split |
| Latency per ticket | **51 ms** single-ticket in-process | measured |
| Training cost | seconds (sparse), no GPU | measured |
| SLA clock | P0 15 min · P1 60 min · P2 240 min · P3 1440 min | ITIL Impact × Urgency |
| Open-source tooling cost | $0 | — |
| Commercial tooling benchmark | Zendesk ≈ $1,500–2,000 per 1,000 tickets; Freshworks ≈ $490 | vendor list pricing |

The 2–4× speedup the synopsis expects from ONNX INT8 should be treated with care
on this class of hardware: ONNX Runtime reports **1.08×** for dynamic INT8 on
non-VNNI x86 versus **1.59×** on VNNI, and documents that older CPUs can get
*slower*. The **S8S8** path additionally has a known AVX2 correctness bug, so
**U8U8** is the only safe quantized path here (`research/05-cpu-deployment.md`).
Given our sparse model is already 40× cheaper to train, quantization buys far
less than the synopsis implies.

---

## 7. Limitations and honest gaps

1. **Banking77 is not real support data.** It is clean, short, single-intent
   queries. Real tickets are long, multi-issue, multi-turn, and code-switched.
   Our headline numbers should not be read as production accuracy.
2. **No priority labels exist publicly.** No published dataset carries P0–P3 plus
   a destination queue. We derived priorities from ITIL Impact × Urgency with an
   explicit, auditable rule table — we did not validate it against human labels,
   because no such labels are public.
3. **The 18.7:1 class imbalance** is reported via macro-F1 alongside accuracy
   precisely so the two do not flatter each other.
4. **The agent is advisory, not autonomous**, for the measured reason in §5.
   Its optional SLM backend is implemented and degrades gracefully, but has not
   been benchmarked on this hardware.
5. **`loan_emi` is unpopulated** — we report it rather than quietly dropping it.
6. **Single seed.** All numbers come from one stratified split with seed 20261003.
7. **The retriever is partly blind to identifiers.** Both TF-IDF views lose
   information that real tickets lean on. The word view inherits scikit-learn's
   default `token_pattern`, `r"(?u)\b\w\w+\b"`, which requires two or more
   characters and therefore discards every single-character token -- bare digits,
   short codes. The char view is built with `min_df=2`, so any n-gram occurring
   in only one document is pruned, which is exactly what a unique reference
   number looks like. We hit this while writing the test suite: twenty tickets
   differing only in a trailing digit collapsed to *identical* bags of features,
   every pair scored cosine 1.0, and the neighbours were ordered arbitrarily by
   document index. "Card ending 1234" versus "1235" is precisely the distinction a
   support retriever has to make, so this must be fixed before deployment
   (`token_pattern=r"(?u)\b\w+\b"`, `min_df=1` on the char view, or a separate
   numeric/entity field). It does **not** materially affect the numbers reported
   here, because Banking77 utterances are natural-language questions that rarely
   hinge on an identifier -- but that is a property of this dataset, not evidence
   that the retriever is safe on production traffic.

---

## 8. What we would do with more time

1. **Close the ranking gap** — the 7.2-point headroom in §3 is real and remains
   the highest-value target. A cross-encoder reranker over the top-3 is the
   obvious first attempt.
2. **Clean the label noise.** Banking77's published 14% noise is documented and
   unaddressed; `cleanlab` is installed and the confident-joint relabelling
   experiment is specified but not yet run.
3. **Build a real labelled SLA slice** — the highest-value *dataset*
   contribution available, and the one we flagged as a research gap.
4. **Characterise code-switched input.** English-only retrieval degrades
   materially on Telugu-English text; our Hyderabad user base makes this
   directly relevant.

---

## 9. Reproducing this

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

Serving and tests go through `run.sh`, which pins `PYTHONPATH` and the thread count:

```bash
./run.sh              # FastAPI on http://127.0.0.1:8000/docs
./run.sh dashboard    # Streamlit on http://127.0.0.1:8501
./run.sh test         # 32 tests: taxonomy, retrieval, amounts, drift, signals
```

The dashboard is a Streamlit script, not a console entrypoint --
`python -m drona.dashboard` runs it outside a Streamlit runtime and renders nothing.
Use `./run.sh dashboard`.

Every experiment is a module under `drona.experiments`; each writes its metrics to
the matching `artifacts/reports/*.json`:

```bash
PYTHONPATH=src python -m drona.experiments.baselines        # 01 tf-idf  -> §1
PYTHONPATH=src python -m drona.experiments.retrieval_exp    # 02 retrieval -> §3
PYTHONPATH=src python -m drona.experiments.fusion_exp       # 03 fusion     -> §4
PYTHONPATH=src python -m drona.experiments.transformer_exp  # 04 distilbert -> §2 (47 min, CPU)
PYTHONPATH=src python -m drona.experiments.escalation_exp   # 05 escalation + agent -> §5
PYTHONPATH=src python -m drona.experiments.label_noise_exp  # 07 label noise -> §7
PYTHONPATH=src python -m drona.experiments.conformal_exp    # 08 conformal   -> §6
PYTHONPATH=src python -m drona.experiments.final_comparison # 06 McNemar + agreement -> §8
```

Note the report file numbers do not follow the section numbers: §6 (conformal) is
`08-conformal.json` because conformal was run after the cleanlab audit.

Split seed `20261003`, stratified 70/15/15 on the coarse label. Raw metrics in
`artifacts/reports/*.json`; literature in `research/0*.md`.