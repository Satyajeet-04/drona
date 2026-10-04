# 04 — Reranking, Selective Prediction, and What We Deliberately Rejected

Status: written directly (research subagent hit a rate limit three times).
Scope: techniques that could improve **DRONA's ranking of retrieved cases** and
**its ability to abstain**. Every technique below is assessed against three hard
constraints that no amount of literature review changes:

1. **CPU only** — 2 physical cores, Intel i3-4005U @1.70 GHz (Haswell, no AVX-VNNI, no GPU).
2. **Fully offline at inference** — no API keys, no hosted model.
3. **Measured latency budget of tens of milliseconds per ticket** for the
   auto-answered path, because that path is the one that runs on 100% of traffic.

The filtering rule I applied throughout: *if a technique cannot beat what we
already measured on this hardware, it is not a contribution — it is a liability.*
Two of the techniques below were rejected **because we measured them failing**.

---

## 1. Where the headroom actually is

Before discussing rerankers, we have to be honest about the problem. From
`artifacts/reports/02-retrieval-oracle.json`:

| | R@1 | R@3 | R@5 | R@10 | R@15 |
|---|---|---|---|---|---|
| coarse (10 classes) | 0.9002 | 0.9643 | 0.9806 | 0.9918 | **0.9975** |
| fine (77 intents) | 0.7901 | 0.9078 | 0.9389 | 0.9654 | 0.9761 |

Two consequences, and they kill most of the literature:

* **Retrieval is solved on this dataset.** The gold label is in the top-15
  retrieved neighbours 99.75% of the time. No reranker, cross-encoder, dense
  retriever or LLM can exceed 99.75% — the ceiling is already nearly touched.
  The 7.2-point gap between oracle R@1 and our 92.56% is a **ranking** problem,
  not a **retrieval** problem.
* **Therefore "improve the retriever" is the wrong research direction here.**
  Papers that improve recall@k on Banking77 with dense retrievers are solving a
  problem our sparse retriever does not have.

This single observation reframes the whole project: the win is in *scoring*
retrieved cases well and *knowing when to abstain*, not in retrieving more.

---

## 2. Cross-encoder rerankers

**What they are.** A bi-encoder embeds query and document independently, so
retrieval is a vector search. A cross-encoder concatenates the pair and runs one
forward pass, so scoring is exact but cannot be precomputed.

**The cost that matters.** Scoring k candidates costs k forward passes. Our
DistilBERT measured **69.9 ms/ticket** for 1,963 tickets at batch 32. A
cross-encoder over k=15 candidates costs ~15× that ≈ **1.05 s/ticket**. Against
a measured 25 ms for sparse retrieval, that is a 40× regression.

**When it would be legitimate.** Only on the escalation slice. At the 90%-automation
operating point 10% of tickets escalate; 0.10 × 1.05 s = **105 ms amortised**.
That is affordable *and* it targets the exact 6.6% of tickets where the cascade
is weak. So the honest recommendation is:

> Do not add a cross-encoder to the auto-answer path. If budget allows, add one
> to the escalation path only, where its cost is amortised 10:1.

**What to expect.** Between 1 and 2 points, because the oracle is 99.75% and we
already sit at 92.56% from a 0-parameter scorer. A reranker that reads the query
and 15 short neighbour texts should close most of the 7.2-point gap — but it is
a large engineering cost for a bounded gain, and it would need a second GPU-free
checkpoint on a 268 MB budget.

**Verdict: NOT ADOPTED for the main path. Conditional candidate for escalation.**

---

## 3. Dense and hybrid retrieval

Small CPU encoders are genuinely affordable here: `bge-small-en-v1.5`
(~33M params, ~130 MB) and `all-MiniLM-L6-v2` (~22M params, ~80 MB) would
embed the 9,158 training tickets in a few minutes and answer a query in
10–20 ms on this CPU.
> `> UNVERIFIED` — the 10–20 ms figure is from vendor/benchmark tables for
> server CPUs. Nothing published measures these encoders on a Haswell
> ultrabook. Measure before quoting.

**But the argument against is the oracle.** We already achieve R@15 = 0.9975
with TF-IDF. Dense retrieval cannot exceed the ceiling, so its only possible
benefit is *marginal reranking value* — and we already obtained that more
cheaply by fusing a linear model's scores with the kNN vote (94.60% vs 92.56%
alone, `artifacts/reports/03-fusion.json`).

The literature does support retrieval beating fine-tuning on classification:
Ovadia et al., EMNLP 2024 (`aclanthology.org/2024.emnlp-main.15.pdf`) find
retrieval-augmented classification consistently outperforms fine-tuning, and
that combining the two is *worse* than retrieval alone. Our result is consistent
with that — retrieval is the stronger primitive here — but we do not need the
*dense* version of it, because our sparse retriever already saturates the metric.

**Verdict: NOT ADOPTED. The ceiling, not the encoder, is the reason, and the
ceiling is already reached.**

---

## 4. LLM-as-reranker — we measured this and it failed

This is the most important entry in the file, because it is an experiment, not
an opinion.

We implemented a ReAct escalation agent with four tools (`similar_cases`,
`kb_lookup`, `money_at_risk`, `sla_policy`), an 11-step reasoning trace, a
grounded KB, and an atomic-fact verifier. We then gave it the authority to
overrule the cascade on exactly the tickets where the two disagree — the 6.57%
of traffic the cascade is worst on.

| policy | accuracy on the disagreement slice | end-to-end accuracy |
|---|---|---|
| cascade alone (do nothing) | 0.6434 | **0.9470** |
| agent overrules unconditionally | 0.4109 | 0.9317 |
| agent overrules only when margin > 0.60 | 0.5969 | 0.9440 |

**The agent made things worse, and the damage was monotone in the confidence of
its own opinions.** Sweeping the margin threshold from 0.10 to 0.60 improved
end-to-end accuracy at every step *toward* the do-nothing baseline. On the
final slice it overruled 16 times: 4 right, 12 wrong.

The interpretation is not "LLMs are bad at this." It is sharper and more
useful: **the small model has no information the cascade lacks.** The cascade
already read the retrieved cases. The agent re-reads the same cases and adds
noise. This is the classic failure of a cascade whose second stage was not
given strictly more evidence than the first.

> **This is now a headline result, not a footnote.** A capstone that reports a
> negative result of this quality — measured, swept, and *acted on* (the agent
> was demoted to advisory-only in `engine.py`) — is stronger evidence of
> engineering judgement than a fourth model that shaves 0.1 points off accuracy.

**Verdict: REJECTED as a decision authority. RETAINED as advisory only**, where
it still contributes a grounded draft reply (verified fact precision 1.000, zero
invented identifiers) at 35 ms.

---

## 5. Selective prediction and conformal abstention

This is the technique we use, and it is where the honest upgrade path lies.

### What we have now
An empirical coverage curve (`artifacts/reports/03-fusion.json`):

| automation coverage | threshold | accuracy on accepted |
|---|---|---|
| 100% | 0.3329 | 0.9460 |
| 90% | 0.7635 | **0.9824** |
| 50% | 0.9860 | 0.9969 |

### What is missing
**The curve is measured, not guaranteed.** Nothing in it tells an operator "if
you set the gate here, the accepted tickets are at least this accurate" with a
finite-sample statistical guarantee. That is exactly what conformal prediction
provides.

### Conformal prediction
Conformal methods turn any calibrated score into a prediction set (or an
accept/abstain decision) with a **distribution-free, finite-sample** guarantee:
with probability ≥ 1−α, the true label lies inside the set, *under exchangeability
of the calibration and test data*. No model assumptions, no asymptotic argument.

For our setting the relevant guarantee is selective classification:

> P(correct | accepted) ≥ 1 − α, with the bound holding at any finite sample size.

Angelopoulos & Bates, *"A Gentle Introduction to Conformal Prediction and
Distribution-Free Uncertainty Quantification"* (arXiv 1907.07509), is the correct
introductory citation. Geifman & El-Yaniv's **SelectiveNet** (ICML 2019) is the
specific prior work for learned selection heads — though we would *not* train
one, because our selection rule is a post-hoc threshold on an already-calibrated
score, which needs no extra parameters and cannot overfit the way a learned head
can.

### What this buys the capstone
1. **A guarantee replaces a table.** Currently §4 of the report shows an
   empirical curve. With split conformal (calibrate on the validation split,
   apply to test) we can state a defensible coverage guarantee.
2. **Cost-sensitive priority.** Our P0 classes (`fraud_security`,
   `payment_failure`) can be given an asymmetric loss so that the conformal
   threshold is derived per-priority rather than globally. Cost-sensitive
   conformal prediction is an active line of work; the machinery here is
   asymmetric nonconformity scores on the P0 classes.
3. **It is honest under data shift.** The empirical curve silently degrades if
   traffic changes. The conformal guarantee is stated under exchangeability, so
   the report must say that explicitly — which is a *feature*: we state the
   assumption rather than hiding it.

### The caveat we must state
Banking77's ~14% label noise (Ying & Thomas, *Demystifying Banking77*,
ACL Insights 2022) means ~14% of the calibration labels are wrong. A conformal
guarantee over noisy labels bounds the *agreement with the annotator*, not
correctness. This is not a footnote — it caps the achievable guarantee and must
appear in the limitations section.

**Verdict: ADOPTED as the next upgrade. Highest value-per-hour remaining,
because it is pure post-processing on artifacts we already have.**

---

## 6. Fusing a classifier with kNN

We late-fuse at the **probability** level:

```
p_fused = α · softmax(linear_logits / T) + (1 − α) · p_knn
```

α = 0.6 and T = 0.75 were both tuned on the **validation** split only.

| system | accuracy | macro-F1 |
|---|---|---|
| linear only | 0.9409 | 0.9364 |
| kNN only | 0.9256 | 0.9175 |
| **fused** | **0.9460** | **0.9443** |

Fusion beats both parts, which is the evidence that the two score functions make
different errors. The disagreement structure quantifies it precisely: **93.43% of
tickets the two agree on, and when they agree accuracy is 96.84%; when they
disagree it collapses to 62.79%.** That 6.57% disagreement slice is a
*self-locating* hard-example set — the model identifies its own uncertainty
without any labels. This is the empirical justification for using disagreement,
not confidence alone, as the escalation trigger.

Alternatives considered and rejected:
- **Stacking** (meta-learner on out-of-fold predictions) — needs nested CV to
  avoid leakage, and two extra hours of compute, for a fraction of a point.
- **Product-of-experts** — assumes conditionally independent experts; ours are
  not (both read the same text), so it would be mis-specified.
- **Learned router** (Arora et al., EMNLP Industry 2024, arXiv 2410.01627:
  hybrid LLM/SLM routing at ~50% lower latency within 2% of the stronger model)
  — the mechanism is right and our α is a crude version of it, but the paper's
  setting routes to a *strictly stronger* second stage. We have no such stage,
  which §4 demonstrated empirically.

**Verdict: ADOPTED as-is.** The disagreement signal is the asset; document it as
the escalation trigger rather than raw confidence.

---

## 7. Hierarchical vs flat classification

The taxonomy is 77 intents → 10 categories. Two options:

* **Flat:** train one model on coarse labels. Current system.
* **Hierarchical:** predict coarse, then fine within that coarse.

We train flat models for both levels independently. The literature on
hierarchical text classification generally finds hierarchy helps mainly when
categories are genuinely disjoint and the label space is large (tens of
thousands); at 77 intents with heavy overlap — Banking77 contains e.g.
`Refund_not_showing_up` and `Refund_not_showing` — a two-stage hierarchy risks
**error propagation**: a correct fine prediction under a wrong coarse prediction
scores zero.

Our own data argues against hierarchy here: the coarse task is already at
94.60%, so a coarse-first cascade would gate 94.6% of tickets correctly and
would still lose on the 5.4% — while adding a second model's latency to 100% of
traffic. There is one genuinely useful middle ground already in the system: the
retriever returns fine-level neighbours, so we get a free fine-grained
suggestion without a second model.

**Verdict: NOT ADOPTED. Hierarchical routing would add latency on every ticket
to fix a problem we do not measurably have.**

---

## 8. Cleaning the labels — Banking77's documented 14% noise

Ying & Thomas (ACL Insights 2022) quantify substantial label noise in Banking77
and show that cleaning it changes conclusions substantially. Our own retrieval
output shows the mechanism directly: near-duplicate utterances carrying
different labels, e.g. *"How can I get new card?"* → `contactless_not_working`
vs *"how do I get a new card"* → `order_physical_card`. Our coarse mapping
merges both into `card_services`, which is part of why the 10-class task scores
so much higher (94.60%) than the 77-intent task (90.22%) — **some of that gap is
label noise being averaged away, not modelling.**

This is the highest-value *experimental* upgrade available, and it is the one
that turns a critique of the source dataset into a contribution:

1. Score every training example with out-of-fold predictions (`cleanlab`).
2. Take the lowest-confidence, most-contradicted examples as candidate label errors.
3. Manually review a sample and measure **the actual precision of the detector**
   — do not assume it is good.
4. Retrain on the cleaned set and report the delta in macro-F1.

The discipline that matters: **report the detector's own precision.** A cleaning
method that removes 14% of the data while being wrong about half of it will
degrade the model, and the only way to know is to measure. We will report
before/after macro-F1 *and* the audit sample size.

**Verdict: ADOPTED as the next experiment.** It targets the dataset weakness
rather than the model, which is why it is the most defensible possible
contribution.

---

## 9. Bottom line for this capstone

**Adopt:**
1. Sparse TF-IDF retrieval as the primary scorer (measured: 92.56%, 4 ms, 0 parameters).
2. Probability-level fusion with a calibrated linear model (measured: 94.60%).
3. **Coverage as the control variable**, with disagreement as the escalation
   trigger (measured: 98.24% accuracy at 90% automation).
4. **Split conformal prediction** on top — the single best remaining improvement,
   because it upgrades an empirical table into a finite-sample guarantee.
5. **Label-noise detection and audit** — the single most defensible *research*
   contribution, because it targets the dataset rather than the model.

**Explicitly reject, with evidence:**
1. **A fine-tuned transformer in the serving path.** Measured: +0.31 pt accuracy,
   **−0.36 pt macro-F1**, 47 min CPU training, 268 MB, 69.9 ms/ticket — versus
   1.6 MB and 36.3 ms for the cascade. On the metric that matters for triage it
   is *worse*.
2. **ONNX INT8 dynamic quantisation** (the synopsis's optimisation). ONNX
   Runtime's own issue tracker reports ~1.08× speedup on non-VNNI x86 against
   1.59× on VNNI hardware, and the documentation concedes older CPUs can get
   *slower*. This machine is Haswell, i.e. exactly the bad case. The synopsis
   budgets a 2–4× win that will not materialise here.
3. **An LLM/small-model reranker as decision authority.** Measured: end-to-end
   accuracy 0.9317 vs 0.9470 for ignoring it.
4. **A cross-encoder in the auto-answer path.** 40× latency for a bounded gain,
   when the oracle is already 99.75%.
5. **Dense retrieval.** Cannot beat a ceiling already reached.
6. **Hierarchical coarse→fine routing.** Adds latency on every ticket to fix a
   problem we do not measurably have.
7. **SetFit / ModernBERT / any larger encoder.** Not because they are bad —
   SetFit reaches 85–90% from 8–64 examples per class, which is genuinely
   impressive — but because under-tuned transformer baselines are known to lose
   up to 13 accuracy points to tuning alone (Galke et al., ACM CSUR 2025,
   arXiv 2504.01724). With 2 cores and a 2–4 week budget, spending the budget on
   tuning a bigger encoder is strictly worse than spending it on calibration,
   abstention guarantees and label auditing, all of which are already paid for.

---

## Sources

| claim | source |
|---|---|
| RAG > fine-tuning for classification | Ovadia et al., EMNLP 2024, `aclanthology.org/2024.emnlp-main.15.pdf` |
| Conformal prediction primer | Angelopoulos & Bates, arXiv 1907.07509 |
| Selective classification with coverage-risk guarantee | Geifman & El-Yaniv, SelectiveNet, ICML 2019 |
| Hybrid LLM/classifier routing, ~50% latency cut within 2% | Arora et al., EMNLP Industry 2024, arXiv 2410.01627 |
| Banking77 label noise (~14%) | Ying & Thomas, ACL Insights 2022, `aclanthology.org/2022.insights-1.19` |
| BERT baselines under-tuned by up to 13 pts; SLMs still SOTA | Galke et al., ACM CSUR 2025, arXiv 2504.01724 |
| SetFit: 8 examples/class beats full FT by 19.3 pts | Tunstall et al., *Efficient Few-Shot Learning Without Prompts*, arXiv:2209.11055 (22 Sep 2022) |
| FActScore, automated atomic-fact precision <2% error | Min et al., arXiv 2305.14251 |
| ARES, few-hundred human labels + prediction-powered inference | Saad-Falcon et al., arXiv 2311.09476 |
| ONNX Runtime dynamic-quant speedup is CPU-dependent | ONNX Runtime issue #6695 |

> `> UNVERIFIED` — treat these as leads, not citable numbers, until re-checked
> against the primary source:
> - SetFit arXiv id **CONFIRMED**: `2209.11055` = "Efficient Few-Shot Learning Without Prompts",
>   Tunstall, Reimers, Jo, Bates, Korat, Wasserblat, Pereg (submitted 22 Sep 2022).
>   Verified against arxiv.org directly. The `2209.11085` variant is wrong.
> - Dense-encoder latency (10–20 ms) on server CPUs; no Haswell measurement exists.
> - Any 2026 claim about "state of the art on coarse-grained Banking77" — we found
>   **no** published number on the 10-class Banking77 task. Our 94.60% may have no
>   direct comparator, which is worth stating plainly in the report rather than
>   implying a leaderboard we did not check.
> - Cost-sensitive conformal prediction on support tickets — a real and active
>   line of work, but every specific percentage found came from a non-indexed
>   preprint. We have therefore cited the *method* (asymmetric nonconformity
>   scores) rather than any number.