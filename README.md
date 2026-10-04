# DRONA — Smart Ticket Understanding Engine

Banking support tickets in, a routed and prioritised ticket out. Runs offline on CPU:
no API keys, no network, no LLM in the serving path.

The interesting part is not that it works. It is **which parts of it we threw away, and why**.

---

## The headline result

We built two systems and compared them on a held-out split of 13,083 tickets.

| System | Accuracy | Macro-F1 | ECE | Latency | Size |
|---|---|---|---|---|---|
| kNN retrieval only | 92.56% | 91.75% | 0.0785 | 27.4 ms | — |
| **Cascade (retrieval + linear)** | **94.60%** | **94.43%** | **0.0170** | **33.3 ms** | **1.6 MB** |
| DistilBERT fine-tuned | 94.91% | 94.07% | 0.0252 | 105.5 ms | 268 MB |

The transformer wins on accuracy by **0.31 points**. So we ran **McNemar's test** on the
tickets where the two disagree:

| | Cascade right | DistilBERT right |
|---|---|---|
| **Cascade right** | 1,813 | 50 |
| **DistilBERT right** | 44 | 56 |

**p = 0.6061.** DistilBERT wins 50 tickets and loses 44. That is a coin flip, and it is the
entire basis for preferring a transformer.

So we shipped the cascade instead. The transformer would have cost **3.2× the latency**,
**168× the model size** and **47 minutes of CPU training** in exchange for noise.

## Three findings that changed the design

**1. Accuracy hid the calibration problem.** The cascade holds 94.6% accuracy at ECE 0.0170;
the transformer is *worse* calibrated at 0.0252 despite scoring higher. With a 18.7:1 class
imbalance accuracy and macro-F1 diverge (the transformer is worse on macro-F1), so we report
both and never accuracy alone.

**2. A confidence gate beats retraining.** Rather than chase the last 0.3 points, we put a
conformal gate in front: auto-answer the tickets the cascade is sure about, escalate the rest.
At **α = 0.10** it auto-answers **90.32%** of tickets at **100.00%** accuracy, and the
coverage guarantee held at every α from 0.01 to 0.20. Split conformal — calibrated on the
val split (n=1,962), evaluated on test (n=1,963), which it never sees.
(Angelopoulos & Bates, arXiv:1907.07509.)

**3. The labels were wrong too.** A cleanlab audit flagged **101 of 9,158** training labels
(1.1%) as suspect. We tried relabelling them: accuracy dropped **2.49 points**. We kept the
original labels and published the finding instead.

## Where it fails

We would rather show this than have you find it.

```
asdkjh qweasd zxcvbn payment failed
  → payment_failure, 97.59% confidence        (should be "other")

the mobile app keeps crashing when I open the transaction screen
  → payment_failure, 84.59% confidence        (should be technical_app)
    nearest neighbours: 0.312 / 0.255 / 0.236 similarity
```

Residual error is concentrated exactly where the model is *most* confident, and the 0.7635
threshold does not catch it. The second case is the clearer one: the retriever found nothing
good and the confidence came from the lexical anchor on the word "transaction".

A known limitation, documented in the report: the word-level TF-IDF view uses sklearn's
default `token_pattern`, which discards single-character tokens, and the char-level view uses
`min_df=2`. So "card 1234" and "card 1235" are indistinguishable to the retriever. Harmless on
Banking77's natural language — not harmless on production traffic.

## The agent is advisory-only

There is a tool-using agent with reasoning traces and a claim-level verifier. Letting it
**overrule** the cascade made end-to-end accuracy *worse*: **93.17% vs 94.70%**. So it drafts,
and a human approves. Its drafts scored **100% fact precision with zero invented identifiers**,
and the verifier blocks any reply citing a case number that does not exist.

---

## Quickstart

```bash
./install-ml.sh      # venv + torch (CPU index) + 15 packages, verified
./run.sh test        # 32 passed
./run.sh             # API      → http://localhost:8000/docs
./run.sh dashboard   # Dashboard → http://localhost:8501
```

Both services bind `0.0.0.0`, so they are reachable from another machine on the LAN at
`http://<this-host>:8501` and `http://<this-host>:8000/docs`.

One-shot, no server needed:

```bash
./run.sh "my card was declined twice and money was debited"
# payment_failure
```

### What to look at

1. **The 7 metrics** across the top of the dashboard.
2. Tab *Ingest & triage* → the sample *"my card was declined twice at checkout and the money is
   still debited"* → **Triage ticket**. Auto-answered at P0, no LLM invoked. Expand *Why this
   decision* and *Retrieved evidence*.
3. The same sample with **Amount at risk = 5000000** — priority flips to P0 and
   `amount_source` flips to `signal`. Non-text data outranks text.
4. The KYC sample → confidence 0.3837, under the 0.7635 threshold, so it **escalates** to the
   agent: reasoning trace, editable draft, and a Verification tab showing fact precision and
   invented identifiers.
5. Tab *Operations* → *"Why the numbers are what they are"*, which traces each headline metric
   back to the run that produced it.

## Layout

```
run.sh                    three entrypoints: api / dashboard / test
install-ml.sh             the only supported install path
docs/CAPSTONE_REPORT.md   438 lines, 9 sections — every number traceable to a run
research/00..05           synopsis, dataset + production landscape, reranking, CPU deployment
src/drona/                engine, cascade, retrieval, triage, calibration, store, service, dashboard
src/drona/experiments/    8 experiment modules, one per report artifact
tests/test_drona.py       32 tests, each guarding a bug we actually hit
tools/patch_report.py     regenerates the report from the artifact JSONs
artifacts/reports/        01..08 json + logs — the raw evidence
artifacts/models/cascade.joblib   1.6 MB, the model that serves
data/raw, data/processed  Banking77 subset + the 3-way split
```

### Reproducing the results

`docs/CAPSTONE_REPORT.md` §9 maps every experiment module to its artifact file and report
section. Verified: re-running an experiment reproduces its committed artifact byte-for-byte.

```bash
PYTHONPATH=src .venv/bin/python -m drona.experiments.baselines          # → 01-tfidf-baseline.json
PYTHONPATH=src .venv/bin/python -m drona.experiments.retrieval_exp      # → 02-retrieval*.json
PYTHONPATH=src .venv/bin/python -m drona.experiments.fusion_exp         # → 03-fusion.json
PYTHONPATH=src .venv/bin/python -m drona.experiments.transformer_exp    # → 04-distilbert.json  (~47 min, CPU)
PYTHONPATH=src .venv/bin/python -m drona.experiments.escalation_exp     # → 05-escalation.json
PYTHONPATH=src .venv/bin/python -m drona.experiments.final_comparison   # → 06-final-comparison.json
PYTHONPATH=src .venv/bin/python -m drona.experiments.label_noise_exp    # → 07-label-noise.json
PYTHONPATH=src .venv/bin/python -m drona.experiments.conformal_exp      # → 08-conformal.json
```

### What is *not* in this repository

`artifacts/models/distilbert-coarse/` — the 257 MB DistilBERT checkpoint. GitHub rejects any
file over 100 MB, and **serving does not need it**; the cascade loads
`artifacts/models/cascade.joblib`, which is committed. It is only required to re-run
`transformer_exp`, which refits from scratch in ~47 minutes on CPU. Every number it produced
is already in `artifacts/reports/04-distilbert.json`.

`data/drona.sqlite` — runtime state, rebuilt as tickets are triaged.

## Data

Banking77 (`banking77`), 13,083 tickets, collapsed from 77 fine-grained intents to 10 coarse
categories. One declared category, `loan_emi`, has **zero** examples in the dataset — a tenth
of the label space has no data, which is why a class can be declared and never predicted.
Splits: 9,158 train / 1,962 val / 1,963 test.

## References

- Angelopoulos & Bates, *Conformal Prediction: A Gentle Introduction*, arXiv:1907.07509
- Tunstall et al., *SetFit: Efficient Few-Shot Learning Without Prompts*, arXiv:2209.11055