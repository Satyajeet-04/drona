# DRONA — Technology Lecture

**A complete walk-through of every concept in the DRONA Smart Ticket Understanding Engine, and how each design choice differs from the alternatives it was chosen over.**

---

## How to use this document

This is not a user manual. It is a **concept-by-concept lecture**: for each idea DRONA uses, it explains (a) what the idea actually is, (b) why it is hard, (c) what the alternatives are, (d) why DRONA rejected them, and (e) what the project's own measurements say.

Every number in this lecture is traceable to a committed artifact under `artifacts/reports/`. Every "we rejected X because Y" claim comes from `research/04-reranking-and-selection.md`, which is the project's own rejection ledger.

**Hardware context for every latency figure below:** Intel Core i3-4005U @ 1.70 GHz (Haswell), 2 cores, **no AVX-VNNI**, no GPU, fully offline at inference. This constraint is not incidental — it eliminates most of the alternatives in this lecture on its own.

---

# Chapter 0 — The problem, and the whole pipeline on one page

## 0.1 What DRONA does

A bank receives a customer support ticket as free text. DRONA answers three questions:

1. **What is this ticket about?** → one of 11 categories
2. **How urgent is it?** → P0/P1/P2/P3, each with an SLA clock
3. **Who should handle it, and what should we tell the customer?** → one of 9 queues, plus a draft reply

It must do all of this in **tens of milliseconds, on a CPU, with no network calls**, and it must be able to say *"I don't know"* rather than guess.

## 0.2 The pipeline

```
 customer ticket (free text)
        │
        ├──────────────────────────────────────────────────────────┐
        │ STAGE 1  DUAL-VIEW SPARSE RETRIEVAL          (~4 ms)     │
        │   char_wb 3–5 gram TF-IDF  ─┐                            │
        │   word 1–2 gram TF-IDF     ─┴→ union → cosine → top-15  │
        │                                  neighbours               │
        ├──────────────────────────────────────────────────────────┤
        │ STAGE 2  CASCADE FUSION                        (~29 ms)   │
        │   LinearSVC  ─→ sigmoid calibrator ─→ p_linear           │
        │   kNN vote   ─→ power-3 weighting ─→ p_knn               │
        │   p_fused = 0.6·p_linear + 0.4·p_knn,  then T = 0.75    │
        ├──────────────────────────────────────────────────────────┤
        │ STAGE 3  DECISION GATE                        (<1 ms)    │
        │   accept if  views agree  AND  confidence ≥ 0.7635       │
        │   → 90% of traffic at 98.24% accuracy                   │
        │   → 10% escalated                                          │
        ├──────────────────────────────────────────────────────────┤
        │ STAGE 4  SLA PRIORITY ENGINE                  (<1 ms)    │
        │   category impact floor + urgency lexicon + sentiment    │
        │   + amount at risk + customer tier + channel + deadline  │
        │   → P0 15 min | P1 60 min | P2 240 min | P3 1440 min    │
        ├──────────────────────────────────────────────────────────┤
        │ STAGE 5  ROUTING                             (<1 ms)    │
        │   (category, priority) → queue, e.g. fraud-ops           │
        ├──────────────────────────────────────────────────────────┤
        │ STAGE 6  ADVISORY ReAct AGENT     (6.57% slice, ~35 ms) │
        │   kb_lookup · sla_policy · money_at_risk · similar_cases │
        │   → draft reply + trace + atomic-fact verification      │
        │   → NEVER overwrites the classifier's label              │
        ├──────────────────────────────────────────────────────────┤
        │ EVERYTHING  →  SQLite audit store + human review queue    │
        └──────────────────────────────────────────────────────────┘
```

## 0.3 The headline numbers

Test split held out, n = 1,963, never used for any tuning. Seed 20261003.

| System | Accuracy | Macro-F1 | ECE | Latency | Size |
|---|---|---|---|---|---|
| kNN retrieval only | 92.56% | 91.75% | 0.0785 | 27.4 ms | — |
| **Cascade (retrieval + linear)** | **94.60%** | **94.43%** | **0.0170** | **33.3 ms** | **1.6 MB** |
| DistilBERT fine-tuned | 94.91% | 94.07% | 0.0252 | 105.5 ms | 268 MB |

DistilBERT is 0.31 points better on raw accuracy. **McNemar's test on the paired disagreements: χ² = 0.266, p = 0.6061.** DistilBERT wins 50 tickets, loses 44. That is a coin flip. DRONA ships the cascade.

---

# Chapter 1 — Sparse lexical retrieval

## 1.1 TF-IDF: what the letters stand for

A ticket is turned into a vector where each coordinate is one n-gram and the value is how much that n-gram argues for this document.

**Term Frequency, sublinear:**
$$\text{tf}(t,d) = 1 + \log f_{t,d}$$

The `1 + log` matters. Without it, a ticket saying *"help help help"* gets a coordinate 3× larger than one saying *"help"* — and support tickets are full of politeness filler. Sublinear scaling compresses repetition so it stops dominating.

**Inverse Document Frequency:**
$$\text{idf}(t) = \log\frac{1 + N}{1 + n_t} + 1$$

where $N = 9{,}158$ training tickets and $n_t$ is how many contain term $t$.

Why this half exists: raw frequency cannot tell you what is *diagnostic*. Every ticket contains "account", "my", "please", "issue". Those are high-TF, low-IDF — correctly suppressed. Meanwhile "chargeback", "cvv", "swift", "atm", "unauthorized" are low-TF, high-IDF — amplified, because a ticket containing "chargeback" is genuinely unusual.

This is the entire statistical argument for the retrieval half of DRONA, and it is also why "improve the retriever with a fancier vector space" is the wrong instinct — see Chapter 11.

## 1.2 Two views, and why DRONA uses both

### The character view — `char_wb` 3–5 grams, `min_df=2`

Character n-grams taken *within word boundaries*, so "pay" becomes `" pa"`, `"pay"`, `"ay "`.

**What it buys:** morphological robustness. `"transferring"` and `"transferred"` share grams. Typos survive: `"paymnt"`, `"whithdrawal"` still overlap heavily with the correctly-spelled neighbours.

**What it costs:** it is blind to multi-word phrases. "card declined" and "declined card" produce nearly the same bag.

**The `min_df=2` trap:** pruning n-grams that appear in only one document sounds like obvious hygiene. It is in fact *deliberately catastrophic for this domain*. A unique reference number — an account number, a card last-four, a UTR — appears in exactly one document. `min_df=2` deletes precisely the features that identify a specific ticket. This is recorded as limitation #7 in the report.

### The word view — word 1–2 grams, `min_df=1`

**What it buys:** exact collocations. "virtual card", "card declined", "standing instructions" are matched as units.

**What it costs:** out-of-vocabulary fragility. A novel synonym never seen in training contributes nothing.

**The `token_pattern` trap:** scikit-learn's default is `r"(?u)\b\w\w+\b"`, which requires **two or more** characters. This silently discards single-character tokens and **bare digits**. Found while writing the test suite: twenty tickets differing only in a trailing digit collapsed to identical feature bags, every pair at cosine 1.0, and the "nearest neighbours" came back in arbitrary document-index order. The fix — `token_pattern=r"(?u)\b\w+\b"`, `min_df=1` on the char view, or a separate numeric field — is written down in the report but **not implemented**, because changing the vectoriser would invalidate every committed artifact.

### The union

Both matrices are **L2-normalised independently, then horizontally stacked, then normalised again**.

This is not cosmetic. The char view produces roughly 25,000 features; the word view far fewer. If you simply concatenated, the char block's larger L2 norm would dominate every inner product, and the word view would be decorative. Normalising each block first guarantees both views contribute equal energy to every similarity.

### Measured (`02-retrieval.json`, coarse11, n_test = 1,963)

| View | Accuracy | Macro-F1 | Query latency | Mean top-1 cosine |
|---|---|---|---|---|
| char only | 91.70% | 90.40% | 4.27 ms | 0.7008 |
| word only | 89.30% | 87.66% | 2.21 ms | 0.5759 |
| **union** | **92.56%** | **91.75%** | **3.93 ms** | 0.6120 |

The union beats char alone by +0.86 and word alone by **+3.26** — and it is *faster* than char alone, because the word view's 2.21 ms is cheaper than char's 4.27 ms and the stacking overhead is small.

## 1.3 Cosine similarity collapses to a dot product

Because both the query vector and every document vector are L2-normalised:

$$\text{cos}(q, d) = \frac{q \cdot d}{\|q\|\|d\|} = q \cdot d$$

So the whole search is one sparse matrix-vector product: $S = qX^\top$ against a 9,158 × D CSR matrix. The query has only ~20–50 non-zero coordinates (tickets are short). That single sparse BLAS call is 3.93 ms. There is no approximate index, no ANN structure, no tuning.

## 1.4 The kNN vote is case-based reasoning

Given the top-15 neighbours, the label is a similarity-weighted vote:

$$w_i = \max(s_i, 0)^3, \qquad \hat{y} = \arg\max_c \sum_{i:\, y_i = c} w_i$$

**Why the cube?** With plain linear weighting $w = s$, five mediocre neighbours at $s = 0.25$ (total 1.25) outvote one genuinely near-identical ticket at $s = 0.85$. Cubing gives $0.85^3 = 0.6141$ against $0.25^3 = 0.0156$ — a **39.3:1 ratio instead of 3.4:1**. The exponent is a noise floor: it says "only trust a neighbour if it is really similar." Banking77's mean top-1 cosine of 0.61 means the corpus is genuinely hard — the project cites Ying & Thomas (ACL Insights 2022) measuring a cosine silhouette of just **0.10** on it. In a corpus that dense, you must be ruthless about what counts as a match.

**The name for this technique is Case-Based Reasoning (CBR).** The 9,158 historical resolved tickets *are* the case base. This is not an academic curiosity — Intercom Fin, Sierra and Decagon all resolve tickets by retrieving similar resolved tickets. DRONA applies the textbook CBR algorithm with the similarity function replaced by a cosine over a dual-view TF-IDF space and the similarity metric replaced by a power-weighted vote.

## 1.5 Oracle recall — the number that reframes the project

From `02-retrieval-oracle.json`, the fraction of test tickets where the **gold label appears anywhere in the top-k retrieved neighbours**:

| k | coarse11 (union) | fine77 (union) |
|---|---|---|
| 1 | 90.02% | 79.01% |
| 3 | 96.43% | 90.78% |
| 5 | 98.06% | 93.89% |
| 10 | 99.18% | 96.54% |
| **15** | **99.75%** | **97.61%** |

And the kNN vote extracts **92.56%**.

**99.75% − 92.56% = 7.19 points of pure ranking headroom.** The correct answer is sitting in the candidate list for 19 out of every 20 tickets and the vote is throwing it away. This is the single most important measurement in the project and Chapter 11 is entirely about it.

---

# Chapter 2 — What DRONA rejected, and why

Every entry here is from `research/04-reranking-and-selection.md`, under the project's own stated filter:

> *"If a technique cannot beat what we already measured on this hardware, it is not a contribution — it is a liability."*

## 2.1 BM25

**What it is.** The standard probabilistic retrieval function. Adds two things to raw TF-IDF: non-linear term-frequency saturation $k_1$, and document-length normalisation $b$.

$$score(D,Q) = \sum_{t \in Q} IDF(t) \cdot \frac{f(t,D)(k_1+1)}{f(t,D) + k_1(1 - b + b \cdot |D|/\text{avgdl})}$$

**What DRONA gains by not using it:** length normalisation. Tickets are short — mean 15.9 tokens, median 13, p95 35. There is essentially no length variation to normalise away. $b$ is close to a no-op on this corpus.

**What DRONA loses by not using it:** nothing measurable. BM25 operates on word tokens and would have to abandon the character view. The measured cost of that abandonment is in `02-retrieval.json`: dropping char n-grams moves accuracy from **92.56% → 89.30%** (−3.26) and macro-F1 from **91.75% → 87.66%** (−4.09). That is the price of losing character-level subword robustness, and BM25 does not pay for it.

**Verdict: correctly rejected.** BM25 is a better retriever for long documents with heterogeneous lengths. DRONA's corpus is neither.

## 2.2 Dense bi-encoders (bge-small, all-MiniLM)

**What they are.** Encode query and document *independently* into a single dense vector; similarity is a dot product. Document embeddings are precomputed and indexed offline, so query cost is one forward pass.

| Model | Params | Weights | Query cost, 2 cores |
|---|---|---|---|
| `all-MiniLM-L6-v2` | ~22M | ~80 MB | ~10–30 ms |
| `bge-small-en-v1.5` | ~33M | ~130 MB | ~20–80 ms |

**What they gain:** synonymy without lexical overlap. "money not credited" retrieves "deposit missing" with no shared token.

**Why DRONA still rejected them — and the honest reason is the ceiling, not the encoder:**

1. **The ceiling.** DRONA already achieves R@15 = 99.75%. There is 0.25 points of recall left to buy. No encoder recovers what is not there.
2. **Banking77's geometry punishes density.** Cosine silhouette of 0.10 means the classes overlap heavily even in the sparse space that was *designed* to separate them. Dense vectors blur adjacent banking codes further.
3. **Cost.** ~100 MB of weights plus a ~14 MB dense index (9,158 × 384 float32) against a 1.6 MB total, for a slower query.
4. **The latency figures are `UNVERIFIED`.** They come from vendor server-CPU tables. Nothing published measures bge-small on a Haswell ultrabook. DRONA flags this in its own research notes rather than quoting it as fact.

## 2.3 Fine-tuned DistilBERT

**What it is.** Fine-tune all 66M parameters of a pretrained transformer on 9,158 training tickets to predict 11 classes.

**What it gains:** +0.31 points of raw accuracy. 94.91% vs 94.60%.

**What it costs, all measured on this hardware:**

| | Cascade | DistilBERT | Ratio |
|---|---|---|---|
| Accuracy | 94.60% | 94.91% | +0.31 (p = 0.6061) |
| **Macro-F1** | **94.43%** | **94.07%** | **−0.36** |
| ECE | **0.0170** | 0.0252 | +48% worse |
| Latency | 33.3 ms | 105.5 ms | **3.2×** |
| Disk | 1.6 MB | 268 MB | **168×** |
| CPU training | seconds | 47 min | — |

The macro-F1 result is the one that matters for triage: **DistilBERT is worse on the metric that protects minority classes**, and triage errors are asymmetric — misrouting a `fraud_security` ticket to a P3 queue costs far more than the reverse.

And the +0.31 is not real. On the 94 paired tickets where the two disagree, DistilBERT is right 50 times and wrong 44 times. **McNemar χ² = (|50−44|−1)²/(50+44) = 0.266, p = 0.6061.** You cannot distinguish this from noise.

## 2.4 Cross-encoders

**What they are.** Concatenate query and document into one sequence — `[CLS] query [SEP] doc [SEP]` — and run a single transformer pass, so self-attention sees every token pair jointly. Far more accurate than a bi-encoder, and **not precomputable**, because the embedding depends on both halves.

**The arithmetic that kills it on the main path:**

- One forward pass = **105.5 ms** per candidate on this CPU (`06-final-comparison.json` → `distilbert_finetuned.latency_ms`)
- Cross-encoding against the *full* 9,158-ticket index = 9,158 passes ≈ **16 minutes per ticket**
- Cross-encoding the *top-15* from sparse retrieval = 15 passes ≈ **1.58 s/ticket**, a **47× regression** against 33.3 ms

**Why it isn't adopted — and why it isn't dismissed.** Reranking only the top 3 on the **escalated slice** is: 10% × 3 passes × 105.5 ms ≈ **32 ms amortised**, roughly 10:1 against the sparse stage. That is affordable. But the expected *gain* is 1–2 points, because the oracle is 99.75% and DRONA already extracts 92.56% using a scorer with **zero learned parameters**.

**Verdict: rejected for the auto-answer path; documented as the correct answer for the escalation slice if the ranking gap is ever closed.**

## 2.5 LLM-as-reranker — measured, and failed

This is the most important entry in the ledger, because it was *tried*.

Running DRONA's own ReAct agent as a reranker with decision authority, on the 129 escalated tickets:

| Policy | Slice accuracy | End-to-end |
|---|---|---|
| **Cascade alone (do nothing)** | **64.34%** | **94.70%** |
| Agent overrules unconditionally | **41.09%** | 93.17% |
| Agent overrules only if margin > 0.20 | 49.61% | 93.73% |
| Agent overrules only if margin > 0.45 | 54.26% | 94.04% |
| Agent overrules only if margin > 0.60 | 59.69% | 94.40% |

**Read that table bottom-up.** Every increase in the agent's confidence threshold monotonically *improves* end-to-end accuracy — because the agent is being allowed to intervene *less*. On the final slice it overruled 16 times: **4 right, 12 wrong.**

> *"The small model has no information the cascade lacks. The cascade already read the retrieved cases; the agent re-reads the same cases and adds noise. This is the classic failure of a cascade whose second stage was not given strictly more evidence than the first."*

**This is the theoretical point that justifies the entire architecture.** A cascade is only worth building if stage 2 is *strictly more informative* than stage 1. Here it was not — so it was demoted. Chapter 9 covers what happened next.

## 2.6 Hierarchical coarse→fine classification

**What it is.** Train a coarse model, then a fine model within each coarse class. Standard in production (ServiceNow, Zendesk).

**Why DRONA rejected it — error propagation.** Banking77 contains both `Refund_not_showing_up` and `Refund_not_showing` as *separate* intents that collapse to the same coarse class. Given a correct fine prediction under a wrong coarse prediction, hierarchical scoring gives **zero**. Meanwhile coarse is already 94.60% correct, so a coarse-first cascade gates 94.6% properly and still loses the 5.4% — while adding latency to **100%** of traffic.

**The elegant compromise DRONA already has:** the retriever returns *fine-level* neighbours. Every dashboard screenshot shows the five nearest Banking77 **intents**, not just coarse categories. So DRONA gets a free fine-grained suggestion, with no second model and no added latency.

## 2.7 Ensembles DRONA did not build

| Ensemble type | Mechanism | Why not |
|---|---|---|
| **Bagging** | Train on bootstrap subsets, average | Reduces variance of *one* model class; DRONA's problem is not variance |
| **Boosting** | Sequentially fit residuals | Would add latency to every ticket for a fractional gain on a 94.6% task |
| **Stacking** | Meta-learner on out-of-fold predictions | Needs nested CV, ~2 h extra compute, fractional gain |
| **Voting** | Uniform/fixed-weight average | Ignores that the two views have *very* different reliability profiles |
| **Product of experts** | Multiply and renormalise | Assumes conditional independence — but both views read the *same text*, so it is mis-specified |
| **Learned router** | Arora et al., EMNLP Industry 2024 (arXiv 2410.01627) | Right mechanism, wrong target: routes to a *strictly stronger* second stage, and §2.5 proved no such stage exists here |

**What a cascade is instead.** Bagging, boosting and stacking all produce a prediction for *every* input. A cascade produces a prediction for *some* inputs and an escalation for the rest. DRONA combines two structurally different inductive biases — a **global parametric** margin model and a **local non-parametric** density model — and adds **selective abstention**. That third property is what no static ensemble has.

## 2.8 SetFit, ModernBERT, larger encoders

Not rejected because they are bad. SetFit genuinely reaches 85–90% from 8–64 examples per class. Rejected on *budget arithmetic*: under-tuned transformer baselines lose up to **13 accuracy points to tuning alone** (Galke et al., ACM CSUR 2025, arXiv 2504.01724), and DistilBERT already measured at **47 minutes of CPU time for a single configuration**. With 2 cores and a 2–4 week window, the marginal hour is worth far more spent on calibration, abstention guarantees and label auditing.

## 2.9 ONNX INT8 dynamic quantisation — *the trap*

The original synopsis budgeted a 2–4× speedup from INT8. It does not materialise here.

ONNX Runtime's own issue #6695 reports **1.08×** on non-VNNI x86 versus **1.59×** on VNNI, and concedes that older CPUs can get *slower*. This machine is Haswell — precisely the bad case. Worse, the S8S8 (signed-activation × signed-weight) path on AVX2 has a known correctness bug, because `vpmaddubsw` treats its first operand as unsigned. DRONA's research concludes: **U8U8 only.**

> *"The synopsis budgets a 2–4× win that will not materialise here."*

This is the clearest example of the project's method working as intended: a promising technique, checked against the actual hardware, and rejected on evidence rather than enthusiasm.

---

# Chapter 3 — The linear model

## 3.1 What a linear classifier does on sparse TF-IDF

For document vector $x \in \mathbb{R}^D$ and class $k$, the score is $f_k(x) = w_k^\top x + b_k$.

**Linguistically**, each weight $w_{k,j}$ answers: *"how strongly does the presence of feature $j$ argue for class $k$?"* — "declined", "card", "terminal" accumulate positive weight for `card_services` and `payment_failure`.

**The assumption:** features contribute additively and independently given the class. This is precisely why it cannot represent *"charged but card not received"* versus *"card received but not charged"* — the same tokens in a different order with different truth values, and an additive bag has no way to tell them apart.

**Why that is survivable here:** banking ticket vocabulary is highly domain-specific. The discriminative signal really is concentrated in a handful of surface anchors. The model does not need compositionality; it needs to recognise domain terminology, and it is superb at that.

## 3.2 LinearSVC vs LogisticRegression — and why DRONA chose hinge loss

Both build linear boundaries. They differ in what the loss function optimises, and that determines whether the output means anything.

**LogisticRegression** optimises multinomial cross-entropy:
$$P(Y=k \mid x) = \frac{e^{w_k^\top x + b_k}}{\sum_j e^{w_j^\top x + b_j}}, \qquad \mathcal{L} = -\sum_k y_k \log P(Y=k \mid x)$$
Output is bounded in [0,1] and sums to 1 — it *is* a probability.

**LinearSVC** optimises the squared hinge with L2 regularisation:
$$\mathcal{L} = \tfrac{1}{2}\|W\|_F^2 + C\sum_i \sum_{j \ne y_i} \max\left(0,\ 1 + w_j^\top x_i - w_{y_i}^\top x_i\right)$$
The loss only cares about points **inside or near the margin**. Points far beyond it contribute **zero gradient**. So the objective is purely geometric: *push the decision boundary into the widest possible gap.*

**The consequence that matters for DRONA:** because hinge loss ignores how far past the margin a point sits, `f_k(x)` is a **signed geometric distance, not a likelihood**. `decision_function` returns an N × K matrix that is unbounded (−4.8, +11.2), does not sum to 1, and where +2.0 in one class does not imply twice the odds of +1.0 in another. It is a *ranking score*. Feeding it to a softmax and calling the result a probability is how most projects end up with an uncalibrated gate.

**Why DRONA prefers it anyway:** hinge loss with L2 regularisation generally beats cross-entropy on high-dimensional sparse text. Measured: `linsvc_coarse11` **94.50%** vs `logreg_coarse11` **92.72%** (`01-tfidf-baseline.json`) — a 1.78-point gap. DRONA accepts the uncalibrated output and *fixes it properly* in Chapter 4, rather than inheriting a slightly worse model that happens to emit plausible-looking numbers.

## 3.3 `dual=True` and `max_iter=20000`

```python
LinearSVC(C=self.cfg.c, dual=True, max_iter=20000, random_state=0)   # cascade.py:113
```

**`dual=True`** — an SVM can be solved over the $D$ feature weights (primal) or over the $N$ Lagrange multipliers (dual). Here $D \approx 50{,}000$ features and $N = 9{,}158$ samples, so $D \gg N$. Under that regime the dual coordinate-descent formulation (LIBLINEAR) is substantially faster and lighter on memory. This is not a tuning flourish — it is the correct formulation for sparse high-dimensional text.

**`max_iter=20000`** — scikit-learn's default of 1,000 iterations routinely emits `ConvergenceWarning` and returns a **partially optimised** model on sparse matrices. Setting it 20× higher guarantees convergence. The cost of a silently non-converged SVM is a model whose weights are arbitrary — which would in turn poison every downstream calibration and threshold measurement.

**A note on seeds:** `cascade.py` uses `random_state=0` while `experiments/baselines.py` uses `random_state=SEED` (20261003). Both are deterministic; they simply are not the same constant, which makes cross-experiment comparison slightly noisier than it needs to be.

## 3.4 From margins to probabilities: the calibrator

`LinearSVC.decision_function` output is squashed by a **LogisticRegression calibrator fitted on the validation split** (n = 1,962). This is sigmoid/Platt-style calibration in shape — a 1-D logistic regression over the margin — and it is the explicit replacement for `CalibratedClassifierCV(cv="prefit")`, which scikit-learn deprecated. The reason for deprecation is a real hazard: `prefit` made it easy to calibrate on training data, producing a model that looked calibrated and was not. DRONA fits the calibrator on a strictly held-out split so the guarantee means something.

There is one subtlety worth naming, because it is the kind of thing that silently produces mis-ordered probability columns: **the calibrator's `classes_` may not match the SVC's.** `Cascade._reorder()` exists purely to map the calibrator's probability matrix onto the canonical `self.classes_` ordering. Without it, class index *k* means different things in the two matrices and the fusion is noise.

---

# Chapter 4 — Calibration

## 4.1 What "calibrated" means

A model is perfectly calibrated if:

$$P(\hat{Y} = Y \mid \hat{p} = p) = p \quad \forall p$$

Concretely: of 100 tickets the model labels with confidence 0.80, **exactly 80 must be correct**. Not 95. Not 60. Eighty.

## 4.2 Why models are not calibrated out of the box

Cross-entropy training pushes logits to large magnitudes — larger and larger until the loss gradient vanishes. The softmax output therefore becomes a *ranking score*, not a *frequentist probability*. It is perfectly good at ordering candidates and terrible at saying "this one is 93% likely."

## 4.3 ECE — the diagnostic

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \cdot \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

where $B_m = \{i : \hat{p}_i \in ((m-1)/M,\; m/M]\}$. DRONA uses **M = 15**.

| System | ECE |
|---|---|
| kNN only | 0.0785 |
| DistilBERT | 0.0252 |
| **Fused cascade** | **0.0170** |

The cascade is the best-calibrated of the three by a wide margin — and that is not an accident of the fusion, it is the direct consequence of the calibrated linear head being the dominant term (α = 0.6).

## 4.4 Temperature scaling

$$\hat{p}_k(T) = \frac{e^{z_k / T}}{\sum_j e^{z_j / T}}$$

$T > 1$ flattens toward uniform; $T < 1$ sharpens.

### The argmax-invariance theorem

**Claim.** $\arg\max_k \hat{p}_k(T) = \arg\max_k z_k$ for all $T > 0$.

**Proof.**
1. Divide by $T > 0$: a strictly increasing function of the original logit. So $z_a > z_b \iff z_a/T > z_b/T$.
2. Apply $\exp$: strictly increasing, so order is preserved.
3. The denominator $\sum_j e^{z_j/T}$ is a single positive scalar, identical for every class. Dividing by it scales all entries equally.

Every step is order-preserving. ∎

**The consequence is the whole reason temperature scaling is the right tool:** it **cannot change a single prediction**. It repairs the probability magnitudes and leaves the argmax — and therefore the 94.60% accuracy — completely untouched. DRONA uses α = 0.6, **T = 0.75**, both tuned on validation.

## 4.5 Why NLL and not ECE

This is the most instructive piece of engineering in the project, because DRONA got it wrong first and documented why.

NLL is a **strictly proper scoring rule**: it has a unique global minimum exactly when the predicted probabilities equal the true conditional probabilities. It is smooth and convex in $1/T$ (grid search or L-BFGS-B both work), and it has an **infinite log barrier** when a correct class is assigned probability approaching zero.

ECE cannot be optimised at all:
- its gradient is zero almost everywhere (the bin indicator is a step function),
- it depends on the arbitrary bin count $M$,
- binning artefacts produce pathological local minima.

### The failed first attempt

DRONA initially selected $T$ by minimising `mean(|confidence − correct|)` — mean absolute error. The reasoning seemed sound. It produced **ECE 0.417** with mean confidences around **0.5** while accuracy was **0.946**.

**Why MAE fails, precisely.** 94.6% of the data is correctly classified, and those terms want $\hat{p} \to 1$, i.e. $T \to 0$. The 5.4% that is wrong wants $\hat{p} \to 0$, i.e. $T \to \infty$. MAE penalises both with the same weak **linear** cost, and the 94.6% mass dominates — so flattening toward uniform "wins". NLL's log barrier makes assigning near-zero probability to a correct example catastrophically expensive, which is exactly the pressure that is missing.

**The corrupted cascade that resulted is the project's best bug story.** With mean confidence 0.53 and a threshold of τ = 0.35, *every ticket passes the gate*. Escalation never fires. The entire selective-prediction claim — the centrepiece of the capstone — is silently invalidated while every headline accuracy number still looks perfect. This is what motivated building Chapter 5's coverage-based control.

## 4.6 Calibration methods compared

| Method | Params | Preserves argmax? | Data needed | Verdict |
|---|---|---|---|---|
| Raw max-softmax | 0 | — | 0 | Overconfident; fails gating |
| **Temperature scaling** | **1** | **Yes (proved)** | ~1,000 | **Chosen** |
| Platt scaling | 2 per class | **No** — per-class intercepts can flip predictions | ~2,000+ | Used only on the single margin, not the fusion |
| Isotonic regression | Non-parametric (PAVA) | Yes | >10,000 | Creates flat discrete confidence plateaus that break thresholding |

The decisive column is the third. Temperature scaling is the only method that is *guaranteed* not to alter decisions while still fixing magnitudes — which is exactly the property a downstream abstention gate depends on.

---

# Chapter 5 — The cascade and its decision gate

## 5.1 Cascade vs ensemble

A **cascade** is a sequence of gates: run the cheap model first, and invoke the expensive stage only on the slice the cheap model cannot handle.

| | Static ensemble | **Cascade** |
|---|---|---|
| Structural heterogeneity | Usually same model class | **Global parametric + local non-parametric** |
| Output for every input | Always a prediction | **Prediction or abstention** |
| Cost | Uniform | **Concentrated on the hard slice** |
| Error signal | Softmax confidence | **View disagreement (free) + confidence** |

## 5.2 Fusion

$$p_{\text{fused}} = \alpha \cdot p_{\text{linear}} + (1-\alpha)\cdot p_{\text{knn}}$$

**α = 0.6**, tuned on the validation split (n = 1,962) alongside T = 0.75. Both are selected on validation so the test split (n = 1,963) is evaluated exactly once.

| Component | Accuracy | Macro-F1 |
|---|---|---|
| Linear only | 94.09% | 93.64% |
| kNN only | 92.56% | 91.75% |
| **Fused (α = 0.6, T = 0.75)** | **94.60%** | **94.43%** |

Fusion buys +0.51 over the linear component alone — and, more importantly, buys the *calibration* and the *disagreement signal* that the gate needs.

## 5.3 The free error signal

$$g(x) = (\neg\,\text{agree}) \;\lor\; (\hat{p}_{\text{fused}} < \tau)$$

| | Share of traffic | Accuracy |
|---|---|---|
| The two views **agree** | **93.43%** | **96.84%** |
| The two views **disagree** | **6.57%** | **62.79%** |

**Why "disagree" is such a good signal — four reasons:**

1. **It is free.** Comparing two argmax integers that the pipeline has already computed is an O(1) operation. It requires no extra matrix work, no extra forward pass, no extra latency.
2. **The inductive biases are complementary by construction.** The LinearSVC is a global hyperplane arrangement — it slices the whole space into flat regions. The kNN vote is local density — it asks "what does the neighbourhood look like." When a global linear boundary and a local density estimate land on different labels, the input is by construction near a decision boundary, in a mixed-signal region, or out of distribution. That is exactly the region where errors concentrate.
3. **Confidence is comparatively untrustworthy.** Softmax is overconfident on misclassified inputs — routinely above 90% on examples it gets wrong. Disagreement does not have this failure mode.
4. **It has a theoretical pedigree.** This is the ambiguity term in the Krogh & Vedelsby ensemble decomposition:

$$\text{Error}_{\text{ensemble}} = \overline{\text{Error}} - \text{Ambiguity}$$

Disagreement *is* the ambiguity, measured directly.

## 5.4 Coverage as the control variable

**Coverage** = the fraction of traffic the system accepts. **Risk** = the error rate on the accepted slice.

The natural instinct is to plot accuracy against a confidence threshold. DRONA's argument — and it is correct — is that **coverage is what operations actually controls.** "My team can absorb 10% escalations, so auto-answer 90%" is a capacity statement. A raw probability of 0.80 is not: it means 95% accuracy in one model version and 70% in another.

So the threshold is derived as a **quantile of the confidence distribution**, and coverage is the independent variable.

### The collapse problem

DRONA's first attempt swept a fixed grid, τ ∈ [0.30, 0.45]. After calibration, confidences cluster tightly around **0.93**. Every threshold in that range sat below essentially every confidence value, so **every threshold accepted 100% of traffic** — the whole grid collapsed to a single operating point (100% coverage, 94.60%). The sweep produced exactly one data point.

### The coverage curve

| Target coverage | Derived τ | Accuracy on accepted |
|---|---|---|
| 100% | 0.3329 | 94.60% |
| 98% | 0.5077 | 95.74% |
| 96% | 0.5887 | 96.55% |
| 94% | 0.6530 | 97.24% |
| 92% | 0.7062 | 97.79% |
| **90%** | **0.7635** | **98.24%** |
| 85% | 0.8485 | 98.80% |
| 80% | 0.9004 | 99.17% |
| 70% | 0.9454 | 99.49% |
| 50% | 0.9860 | 99.69% |

**0.7635 is the shipped threshold.** It is not a magic number; it is *the 10th percentile of the confidence distribution*, and it is stored with the model so it moves whenever the model moves.

### One implementation note worth copying

`Cascade.coverage_curve()` computes accuracy **directly on the accepted mask**. An earlier version used a cumulative sum over the confidence-sorted array — which is silently wrong whenever two confidences tie, because the running total then attributes a label to the wrong side of the tie. Sparse-text confidence distributions tie constantly.

---

# Chapter 6 — Split conformal prediction

## 6.1 The problem with empirical curves

The coverage curve in Chapter 5 is **measured, not guaranteed**. It describes what happened on one specific test split. It makes no promise about tomorrow's traffic.

## 6.2 What conformal prediction adds

Split conformal converts an empirical curve into a **finite-sample, distribution-free statistical guarantee**.

**Procedure:**

1. **Split.** Training (9,158) / calibration (1,962) / test (1,963).
2. **Assume exchangeability.** The calibration examples and a future ticket are i.i.d. draws from the same distribution $P$.
3. **Score.** Nonconformity = how wrong the prediction looks, given the truth:
$$s_i = 1 - p(Y_i \mid X_i)$$
4. **Quantile.** Sort the $n$ calibration scores and take the $\left\lceil (n+1)(1-\alpha) \right\rceil$-th smallest.

For $n = 1{,}962$ and $\alpha = 0.10$:

$$\lceil 1963 \times 0.90 \rceil = \lceil 1766.7 \rceil = 1767 \implies q = 0.3032$$

Accept the ticket if $1 - p_{\text{true}} \le q$, i.e. if $p_{\text{true}} \ge 0.6968$.

## 6.3 The guarantee

$$\mathbb{P}\bigl(Y_{n+1} \in C(X_{n+1})\bigr) \ge 1 - \alpha$$

and if the scores are continuous, tightly:

$$\mathbb{P}\bigl(Y_{n+1} \in C(X_{n+1})\bigr) \le 1 - \alpha + \frac{1}{n+1}$$

**"Distribution-free" means:** no normality assumption, no parametric form, no linear relationships — and, critically, **no assumption that the classifier is correct or even calibrated.** The guarantee comes from the exchangeability of the calibration and test data, not from the model.

This is the honest upgrade over the empirical curve: the curve says "on this split we got 90.32%"; conformal says "**at least 90%, at any finite n, distribution-free**."

## 6.4 Measured results

| α | q | Nominal coverage | Realised coverage | Shortfall | Accuracy on accepted | Accepted |
|---|---|---|---|---|---|---|
| 0.01 | 0.9437 | 99% | 99.03% | +0.03 | 95.52% | 1,944 |
| 0.02 | 0.8669 | 98% | 98.12% | +0.12 | 96.42% | 1,926 |
| 0.05 | 0.5758 | 95% | 94.75% | −0.25 | 99.62% | 1,860 |
| **0.10** | **0.3032** | **90%** | **90.32%** | **+0.32** | **100.00%** | **1,773** |
| 0.15 | 0.1868 | 85% | 86.14% | +1.14 | 100% | 1,691 |
| 0.20 | 0.1265 | 80% | 82.02% | +2.02 | 100% | 1,610 |

`guarantee_holds: true` at every α. Coverage never fell more than ~2 points below nominal.

## 6.5 The 100% number, correctly interpreted

At α = 0.10: **1,773 accepted, all 1,773 correct; 190 escalated.**

```
n = 1,963 test tickets
├── accepted / auto-answered:  1,773  (90.32% coverage)  →  1,773 correct (100.00%)
└── escalated to humans:         190  ( 9.68%)           →  model abstains
```

**Coverage is a quantity; accuracy-on-accepted is a quality. They are different axes and conflating them is the single easiest way to overclaim.**

"100% accuracy" does **not** mean the model is perfect. It means:

1. **The model bought that accuracy with abstention.** It transferred the 190 hardest tickets to humans. The other 5.4% error mass still exists — it was moved, not removed.
2. **Finite-sample variance.** Zero errors across 1,773 items is consistent with a true rate anywhere from 0% up to the nominal 10% allowance. The point estimate is not the guarantee.
3. **Label noise sets a ceiling.** See §6.6.

## 6.6 The caveat that must be stated, and is

Banking77 carries a published ~14% label noise at the 77-intent level (Ying & Thomas, ACL Insights 2022). So the conformal guarantee bounds **agreement with the annotator**, not correctness in the world.

DRONA's own cleanlab audit complicates this usefully: at the **coarse** level, only **1.10%** of labels are disputed (101 of 9,158). So the practical ceiling at 11 classes is far above 0.86. Both statements are in `08-conformal.json`'s `noise_caveat` field, and both belong in any presentation of the 100% figure.

## 6.7 The three-way comparison

| | Fixed business threshold | Temperature scaling | **Split conformal** |
|---|---|---|---|
| Goal | Ad-hoc rule | Make $P(\hat{Y}=Y) \approx \hat{p}$ | **Bound coverage at $1-\alpha$** |
| Mechanism | Static cutoff | Softmax($z/T$), minimise NLL | Quantile of $1-p_{\text{true}}$ on held-out scores |
| **Guarantee** | **None** | **None** | **Finite-sample, distribution-free** |
| Under shift | Brittle | Smooths, threshold still arbitrary | Honest — exchangeability is *stated* |

They are **complementary, not alternatives**: temperature scaling makes the scores *comparable*, conformal makes the *decision* defensible. DRONA uses both.

## 6.8 Why not SelectiveNet

Geifman & El-Yaniv (ICML 2019) train a network with a built-in reject head — the "right" answer when you have a deep model and lots of data. DRONA does not, for a specific reason: **a post-hoc threshold on an already-calibrated score has zero extra parameters and cannot overfit.** A learned rejection head is a second model fitted on the same data; a quantile is arithmetic. On 2 CPU cores with 2–4 weeks, spending the budget on a second fitted model to replace a quantile would be indefensible.

The research notes also identify where conformal *would* earn its keep: **cost-sensitive nonconformity** — assigning P0 fraud tickets a harsher penalty so the guarantee applies to the errors that actually cost money, rather than uniformly.

---

# Chapter 7 — SLA priority

## 7.1 ITIL: Impact × Urgency

$$\text{Priority} = f(\text{Impact}, \text{Urgency})$$

| Priority | Meaning | SLA response |
|---|---|---|
| **P0** | Critical — active breach, mass financial loss | **15 min** |
| **P1** | High — major disruption, individual account | **60 min** |
| **P2** | Medium — standard disruption, no immediate exposure | **240 min** (4 h) |
| **P3** | Low — informational, general enquiry | **1440 min** (24 h) |

## 7.2 Why priority is *not* a text-classification problem

This is the most common mistake in applied triage, and it is worth being blunt about. A text classifier — TF-IDF, DistilBERT, or an LLM — learns *what the ticket is about*. It has no access to:

1. **Category impact floor** — `fraud_security` is inherently riskier than `general_enquiry`
2. **Urgency lexicon** — "immediately", "unauthorized", "emergency"
3. **Sentiment / distress** — a calm customer reporting a breach is still an emergency
4. **Customer tier** — enterprise, VIP, standard
5. **Amount at risk** — ₹5,00,000 vs ₹50 is a different event
6. **Channel** — live chat carries different urgency than async email
7. **SLA deadline proximity** — minutes to breach

None of these are in the text. A classifier *cannot* see them. Treating priority as a 4-class text classifier guarantees you lose signals 4–7 entirely.

## 7.3 DRONA's design: a deterministic multi-signal rule engine

```python
base = {4: "P0", 3: "P1", 2: "P2", 1: "P3"}[CATEGORY_IMPACT[category]]
# urgency can escalate; sentiment can escalate; nothing can de-escalate
if amount >= 100_000 and base in ("P2", "P3"):  → escalate to P1
if tier == "enterprise" and base == "P3":      → escalate to P2
if sla_deadline_minutes < 30 and impact >= 3:   → escalate to P0
```

Every decision records **why**: `amount_source ∈ {signal, text_extracted, none}`, `escalated_by ∈ {urgency, amount_at_risk, customer_tier, sla_deadline}`.

## 7.4 The monotonicity property — a one-way safety ratchet

> **Urgency, sentiment, tier or financial risk may escalate a ticket. Nothing may ever de-escalate it below its category impact floor.**

This is deliberate, and the reasoning is a cost-asymmetry argument:

- **Under-escalation** (fraud treated as P3) → financial loss, regulatory exposure, reputational damage.
- **Over-escalation** (P2 treated as P1) → slightly more agent workload for four hours. Zero harm.

The asymmetry is roughly three orders of magnitude. So the rule engine is built to be *eager to escalate and structurally unable to quieten*.

Concretely: a customer who politely writes *"I noticed an unauthorized transfer of ₹5,00,000 from my account"* has neutral sentiment and calm phrasing. Monotonicity guarantees the tone cannot downgrade it. The `fraud_security` impact floor (4) and the extracted amount force P0 immediately.

## 7.5 Indian amount extraction, and the silent failure mode

The extractor must natively handle:

| Form | Example | Value |
|---|---|---|
| Indian grouping | `₹1,50,000` | 150000 |
| Western grouping | `Rs. 2,50,750` | 250750 |
| Lakh | `1.5 lakh` | 150000 |
| Crore | `2 crore` | 20000000 |
| K suffix | `45k` | 45000 |
| Currency symbol | `₹`, `INR`, `Rs` | — |

**Two implementation details that matter:**

1. **Ordering.** The `_NUMBER` grouped alternative is listed **first** in the pattern so `Rs 45000` cannot truncate to `450`.
2. **Largest wins.** When several amounts appear, the largest is returned — a customer listing a ₹200 service fee and a ₹4,00,000 disputed transaction means the transaction.

**The silent failure mode is what makes this dangerous.** A regex that expects `$100,000` and meets `Rs 50,00,000` returns **0.0** without raising, without warning, without logging. The consequence chain:

> ₹5,00,000 treated as ₹0 → the amount escalation rule never fires → a critical ticket sits at P3 → an SLA breach that nobody was alerted to.

This is why `test_extract_amount` is parametrized over all seven forms in the test suite.

## 7.6 Alternatives considered

| Architecture | Why rejected |
|---|---|
| **Learned priority classifier** | Needs massive historical P0–P3 data that does not exist publicly; priority depends on dynamic CRM state, not invariant text, so a text model overfits agent noise and cannot be safety-audited |
| **Pure keyword score** | Brittle false positives (*"this is **not** urgent"*) and structurally blind to non-text triggers — a quietly worded ticket from an enterprise VIP |
| **LLM prompting** | 2,000–5,000 ms latency vs <1 ms; non-deterministic output; can hallucinate parameters; cannot be integrated with a low-latency SLA clock |

## 7.7 The honest limitation

**No public dataset carries true P0–P3 SLA labels with destination queues.** So while DRONA's *classification* is rigorously evaluated on 1,963 held-out tickets, **the priority rule table is an unvalidated expert heuristic derived from ITIL.**

The project's own research notes (`02-dataset-landscape.md`) identify a Kaggle set — `tobiasbueck/multilingual-customer-support-tickets`, ~10,000 rows with **Priority, queue and type** columns — as the one public dataset found that would make this table empirically checkable. It was found during research and **never used**. This is the highest-value, lowest-cost unexploited item in the entire project.

---

# Chapter 8 — The ReAct agent

## 8.1 What ReAct is

**ReAct** = *Reasoning + Acting*. A loop of:

```
Thought   →  "The customer says unauthorized transfer. I should check the SLA policy
              and the amount at risk before drafting."
Action    →  sla_policy(priority="P0")
Observation → {"sla_minutes": 15, "escalate_to": "fraud-ops"}
Thought   →  "P0 confirms fraud-ops. Now the monetary exposure."
Action    →  money_at_risk(ticket_id=...)
Observation → {"amount": 500000, "currency": "INR"}
Thought   →  "Now find comparable resolved cases for the reply template."
Action    →  similar_cases(category="fraud_security", k=5)
Observation → [ ... 5 retrieved tickets ... ]
Decision  →  draft reply + category proposal + trace
```

## 8.2 Memory, state, trace — three different things

| Term | What it is in DRONA | Where |
|---|---|---|
| **Memory** | Short-term execution context: prior thoughts, tool inputs, observations | `AgentTrace`, per-ticket |
| **State** | Structured runtime object: ticket id, text, priority, retrieved evidence, extracted claims, verification flags | `TriageResult` in `engine.py` |
| **Trace** | Full Thought/Action/Observation log, rendered in the dashboard | `result.trace` |

Confusing these three is the most common way agent architectures get described vaguely. All three exist in DRONA, and all three are persisted — which is what makes the agent auditable rather than magical.

## 8.3 Four tools

| Tool | Purpose |
|---|---|
| `kb_lookup` | Category policy, action, escalation target, customer-facing language |
| `sla_policy` | Response clock for the assigned priority |
| `money_at_risk` | Extracted monetary exposure |
| `similar_cases` | Retrieved neighbours from the same index the cascade used |

The tool set is deliberately small and every tool is **read-only and total** — no tool can mutate state. This bounds the blast radius of a hallucination from "customer data loss" to "a wrong sentence in a draft a human will read."

## 8.4 Bounded risk: classifier vs agent

**A classifier's risk is bounded.** Its output space is 11 discrete labels. Every possible failure is a cell in a confusion matrix, measurable with accuracy, macro-F1, ECE and a threshold.

**An agent's risk is unbounded.** Its output space is English prose containing *actionable customer promises*. An invented case number sends a customer looking for a record that does not exist. An unsupported refund promise creates a financial liability. A misstated SLA creates a contractual exposure.

That asymmetry is the entire justification for the architecture in Chapter 9.

## 8.5 Atomic-fact verification

Following FActScore (arXiv 2305.14251), the draft is decomposed into atomic claims:

> *"Your transaction of ₹500 under case DRN-102 was refunded."*

splits into three independently checkable atoms:
1. a ₹500 transaction occurred,
2. case ID DRN-102 is correct,
3. it was refunded.

$$\text{Fact Precision} = \frac{\text{supported claims}}{\text{total claims}}$$

**Measured: 1.000** across all 129 drafts. **Invented identifiers: 0.**

### The distinction that makes this useful

| Failure type | Severity | Action |
|---|---|---|
| **Unsupported claim** — uncorroborated prose | Warning | Flag, human reads the draft anyway |
| **Invented identifier** — a reference that exists nowhere | **Blocker** | `releasable = False` |

An invented identifier is categorically worse than vague prose, because it **destroys the audit trail**: it breaks downstream lookups, it misleads the customer, and it produces a record that can never be reconciled.

**The bug that produced one:** an earlier version derived the reply's reference number from `hash(text)`, inventing case IDs like `DRN-341871`. The verifier caught it. The fix is structural — `store.next_id()` allocates a real, persisted ID from SQLite **before** any reply referencing it is drafted, so the number in the draft is guaranteed to resolve.

`verify()` also scans with a dedicated pattern for identifier shapes:

```python
id_pattern = r"\b(?:[a-z]{2,4}-\d{4,}|[A-Z]{2,}\d{5,}|\d{6,})\b"
```

Invented identifiers must match **exactly** — a near-miss is not a pass.

## 8.6 Why next-token likelihood scoring, not free generation

The optional local SLM backend (`Qwen/Qwen2.5-0.5B-Instruct`, fp32, `torch.set_num_threads(2)`, loaded lazily, degrades gracefully) is used for exactly one thing: `_llm_pick()` scores candidate **label tokens** by next-token likelihood.

1. **The output space is bounded** to valid class tokens. No syntax errors, no format drift, no out-of-vocabulary tokens.
2. **It yields a true softmax distribution**, which plugs directly into thresholding, conformal bounds and uncertainty estimation. Free generation yields a string you then have to trust.
3. **One forward pass** instead of an autoregressive loop. On a CPU this is decisive.

And when the SLM does have a preference, it gets a **+0.15 nudge**, never an override. Decision score is `0.6 × similarity-support + 0.4 × min(lexical_overlap, 1)` — the retrieval evidence dominates by construction.

---

# Chapter 9 — The measured negative result, and why it was published

## 9.1 What was measured

On the 129 escalated tickets (6.57% of traffic):

| Policy | Slice accuracy | End-to-end |
|---|---|---|
| **Cascade alone** | **64.34%** | **94.70%** |
| Agent overrules unconditionally | 41.09% | 93.17% |
| margin ≥ 0.20 | 49.61% | 93.73% |
| margin ≥ 0.45 | 54.26% | 94.04% |
| margin ≥ 0.60 | 59.69% | 94.40% |

Read bottom-up again: **every increase in the agent's confidence threshold improves the result**, because it is being permitted to intervene *less*. At the final setting it overruled 16 times — **4 right, 12 wrong**.

## 9.2 The architectural lesson

> *"This is the classic failure of a cascade whose second stage was not given strictly more evidence than the first."*

A cascade is worth nothing if stage 2 is not strictly more informative than stage 1. Here the agent re-read the same retrieved cases the cascade had already read, on the same text, with a weaker model — and added noise. The cascade's stage 2 **is the human reviewer**, and the human has information neither model has.

## 9.3 The decision: advisory, not disabled

The agent was **not removed**. It was demoted:

```python
result.agent_proposal = decision     # recorded
# result.category is never overwritten
```

| | Advisory (shipped) | Autonomous (rejected) |
|---|---|---|
| Authority | Draft + recommend; human approves | Direct execution |
| Labour cost | Higher review burden | Lower |
| Governance liability | Low | High |
| Required fail-safes | Traces, claim verification, editable draft | Conformal bounds, drift monitoring, automated rollback |

**Why this is the scientifically correct decision, not a retreat:** architecture should be governed by held-out evidence, not by paradigm preference. Demoting the agent preserved end-to-end accuracy *while still delivering generative assistance* — draft replies at 1.000 fact precision, zero invented identifiers, ~35 ms, off the critical latency path.

An agent that cannot be trusted to decide is still extremely valuable as a drafter. Refusing to use it at all would have thrown away the strongest Responsible-AI evidence in the project.

## 9.4 An unclaimed result worth ~150×

`research/03-production-landscape.md` predicted a full CPU agentic triage loop at **5,000–13,000 ms**, based on Phi-4-mini at 5,460 ms tool-call latency plus 3,000–8,000 ms of drafting — and called publishing that number "a legitimate research contribution."

**DRONA's measured agent latency is 34.99 ms.**

That is roughly **150× faster** than the literature figure, and the reason is structural: a hand-written policy agent with a deterministic tool layer, using a tiny SLM only for label-token scoring, rather than an LLM generating free prose. **The report does not currently claim this.** It is a genuine, measured, publishable contribution sitting unclaimed.

---

# Chapter 10 — MLOps and Responsible AI

## 10.1 The audit store, and why overrides are stored separately

Schema (`data/drona.sqlite`):

```
tickets(ticket_id PK, text, category, priority, queue, confidence, escalated,
        latency_ms, created_at, reviewed, approved,
        agent_category, agent_priority,      ← model's proposal
        human_category, human_priority,      ← the human's answer
        draft_reply, trace_json, verify_json)
feedback(...)
```

**Human overrides go in different columns from model output.** Three reasons:

1. **Ground truth is preserved.** Overwriting the model's prediction destroys the record of what the model *said*, which makes historical accuracy, calibration curves and drift analysis uncomputable after the fact.
2. **Feedback loops are prevented.** In-place replacement means a retraining pipeline can consume its own predictions as labels — a circular loop that silently amplifies the model's own systematic errors. (This is not hypothetical: DRONA measured exactly this failure in Chapter 12.)
3. **Accountability is legible.** You can always answer "was this automated or human-handled?"

## 10.2 Data drift vs concept drift

| | Definition | Banking example |
|---|---|---|
| **Data drift** (covariate shift) | $P(X)$ changes, $P(Y \mid X)$ constant | Customers start using new product terminology after a launch |
| **Concept drift** | $P(Y \mid X)$ changes | A policy change means "refund request" now routes to Fraud, not Billing |

DRONA detects **data drift**: per-category share in the live ticket stream vs the reference share in the training split, flagging $|\Delta| > 0.05$, with a **`min_live = 30` guard** so the statistic is not computed on a sample too small to be meaningful.

### The best bug in `store.py`

An earlier version computed **both sides from the `tickets` table** — comparing live share against live share. Every delta was identically zero. The monitor could never fire.

> *"A drift monitor that can never fire is worse than none."*

Because it reports "no drift" with the same confidence as a working monitor, it converts vigilance into false assurance. This is the same failure class as the stale-threshold bug in Chapter 4: **a monitoring component that looks healthy while measuring nothing.**

## 10.3 Determinism vs reproducibility

| | Meaning |
|---|---|
| **Determinism** | Same code + same inputs + same environment → bit-identical output |
| **Reproducibility** | An independent run, possibly on different hardware → statistically equivalent results |

The fixed seed **20261003** buys:
- an identical 70/15/15 stratified split every time,
- byte-for-byte identical artifacts `01`–`08` (the README asserts this was verified),
- reproducible runs of all 32 tests.

It does **not** buy:
- cross-platform invariance (float non-associativity, AVX2 vs VNNI instruction paths, thread scheduling),
- statistical robustness — **one seed is one sample.** The report states this as limitation #6.

## 10.4 Regression tests that guard real bugs

32 tests, each traceable to a failure actually hit:

| Test guards | The bug |
|---|---|
| `test_taxonomy_covers_every_intent` | Taxonomy silently losing coverage of an intent |
| `test_knn_returns_the_identical_text_it_was_given` | Indexing `sims.data` (the flat 1-D buffer) by row instead of slicing the sparse row |
| Threshold-vs-calibration test | A calibrated model served with a threshold tuned against a *different* calibration |
| Parametrized `test_extract_amount` (7 cases) | Amount extraction silently returning 0 |
| Drift test | A drift monitor whose reference is the live data |

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q     # 32 passed in 10.42s
```

> **Note:** bare `pytest tests/` from the repo root **fails** with `ModuleNotFoundError: No module named 'drona'`. `run.sh` exports `PYTHONPATH=src` first, so `bash run.sh test` works — but a naive `pytest` does not. Worth a `conftest.py` or `pyproject.toml` entry so an evaluator's first command succeeds.

## 10.5 Human-in-the-loop

The Streamlit console (`run.sh dashboard`) shows queues sorted by SLA priority. For escalated tickets it displays the prediction, confidence, retrieval evidence, the full agent trace, and an **editable draft**. Agents review, edit, approve or correct; the correction is written to `human_category` / `human_priority`, and the original model output is preserved beside it.

**What is missing, and named as missing in `research/03`:** **shadow mode / canary deployment.** Shadow mode runs a new model alongside production on live traffic while serving only the old model's output — risk-free benchmarking on real data. Canary sends 5% of traffic to the new version. DRONA has an audit store and a review loop but no parallel shadow evaluation. Its absence is justified (single-node CPU, no load balancer) and stated honestly rather than glossed.

## 10.6 Dashboard facts are read from artifacts

`_load_facts()` reads headline numbers **out of the artifact JSONs at runtime**, never as literals. The comment explains why, and it is worth reading verbatim:

> *"the console kept displaying 97.13% auto-answer accuracy from the old mis-calibrated cascade (T=3.0) after the calibration work had moved the real operating point to 98.24% at 90% coverage. A dashboard that displays a stale constant looks authoritative and is wrong, which is worse than showing nothing."*

**That comment is also the smoking gun for a live inconsistency in the report.** 97.13% is the *pre-calibration* number. It was correctly purged from the dashboard but **survives in §6 of the capstone report and in the README business-impact text.** The dashboard fix was applied; the prose fix was not.

---

# Chapter 11 — The 7.2-point gap, and the failure the gate cannot catch

## 11.1 Separating retrieval from ranking

| Stage | Job | Status |
|---|---|---|
| **Retrieval** (candidate generation) | Get the gold label into a short candidate list | **Solved** — R@15 = 99.75% |
| **Ranking** (candidate scoring) | Order those 15 so the right one wins the vote | **7.19 points of headroom** |

**What not to build:** a bigger dense index, FAISS, a stronger encoder. Searching 9,158 tickets already takes 27.4 ms and already finds the answer 99.75% of the time. Papers reporting improved recall@k on Banking77 with dense retrievers are solving a problem this sparse retriever does not have.

**What to build:** a **reranker over the top 3–5 only.** That is the correct shape of the problem and Chapter 2.4 gives the exact arithmetic for why it is affordable *on the escalation slice* and not on the main path.

## 11.2 Why near-duplicate utterances break the vote

The failure mechanism is visible directly in the retrieval output:

> *"How can I get new card?"* → retrieved `contactless_not_working`
> *"how do I get a new card"* → retrieved `order_physical_card`

Two almost identical strings, two different fine intents, both mapping to `card_services` at the coarse level. The vote cannot distinguish them because **there is nothing in the surface form to distinguish them.**

This also explains part of why the coarse task scores 94.60% while the 77-intent task scores 90.22%: **some of that 4-point gap is label noise being averaged away, not modelling skill.** This should be stated explicitly whenever the two numbers are compared.

## 11.3 The residual failure the gate cannot catch

> *"the mobile app keeps crashing when I open the transaction screen"*

| | |
|---|---|
| True category | `technical_app` |
| Predicted | `payment_failure` |
| Confidence | **84.59%** |
| Nearest-neighbour similarities | **0.312, 0.255, 0.236** |
| Gate threshold | 0.7635 → **passed**, auto-answered |

```
"the mobile app keeps crashing when I open the transaction screen"
        │
        ├─► retrieval: [0.312, 0.255, 0.236]   ← POOR. nothing matched.
        ├─► linear model anchors on "transaction"
        └─► payment_failure @ 84.59% → gate PASSED
```

**The structural cause.** Softmax probability measures the **relative distribution across candidate classes** — it does not measure **distance to the training distribution**. The retriever found essentially nothing (max cosine 0.312), and the gate is blind to that fact because it reads only the classifier's confidence.

**And gibberish is worse:**

> *"asdkjh qweasd zxcvbn payment failed"* → `payment_failure` at **97.59%**

The isolated token "payment" fires a 97.59%-confident wrong answer. This is precisely the failure mode that motivates the 11th `other` class.

## 11.4 Why the `other` class exists

A closed-world classifier assumes every input belongs to one of $C$ classes. Given out-of-distribution input, it is *forced* to choose — and **it chooses wrong with full confidence**, because a softmax has no way to express "none of these."

This is a documented structured-output design rule: **always give a categorical field an escape value.** DRONA's `other` class is that escape hatch. It exists so that gibberish, non-banking requests and unhandled topics have somewhere legitimate to go, instead of being force-fitted into `payment_failure` and poisoning that queue's SLA clock.

**This is also why DRONA defines 11 classes while only 10 are populated.** The 11th is `loan_emi`, mandated by the original synopsis but having **zero examples in Banking77**. It is retained because the rubric requires it, and it is documented as unpopulated rather than quietly deleted.

---

# Chapter 12 — Label noise, and a result that contradicted the literature

## 12.1 The published claim

Ying & Thomas (ACL Insights 2022, `2022.insights-1.19`) report **~14% label noise** in Banking77 at the 77-intent level, and report that cleaning it yields **+4.5% F1 and +8% ARI**. The project's research notes carry this forward as an open contribution: *"14% label noise published but unaddressed."*

## 12.2 What DRONA measured

Using **cleanlab** — score every training example out-of-fold, flag those whose out-of-fold prediction most contradicts the annotation:

- Out-of-fold agreement with annotation: **93.93%**
- Flagged: **101 of 9,158 = 1.10%**

**Not 14%. Roughly a twelfth of it.**

## 12.3 Two cleaning policies, both tested

| Policy | n | Accuracy | Δ | Macro-F1 | Δ |
|---|---|---|---|---|---|
| Baseline | 9,158 | 93.73% | — | 93.00% | — |
| **Drop flagged** | 9,057 | 93.58% | **−0.15** | 93.06% | **+0.06** |
| **Relabel to OOF prediction** | 9,158 | 91.24% | **−2.50** | 89.82% | **−3.18** |

**Dropping the flagged examples slightly hurt. Relabelling them was actively, substantially harmful.**

## 12.4 Why relabelling is destructive

A model relabelled by its own out-of-fold predictions **amplifies its own systematic errors**. The 101 "noisy" examples cleanlab found are disproportionately the ones the model finds genuinely hard — ambiguous intents, boundary cases. Overwriting their human labels with the model's guesses destroys exactly the information that makes them hard, and the next training run is worse.

This is the same failure as in-place human-override storage in Chapter 10, arriving through a different door. **It is the strongest single argument in the project for keeping model output and ground truth in separate fields.**

## 12.5 The honest reading

The 14% figure is a **77-intent** measurement. DRONA's audit is a **coarse 11-class** measurement. These are not the same quantity, and the coarse mapping merges many genuinely-distinct fine intents into one label — which mechanically *reduces* measured disagreement. Both statements can be true.

**DRONA kept the original labels and published the finding** rather than quietly applying a policy that made the numbers worse. And `research/04` §8 records the methodological lesson with unusual clarity:

> *"Manually review a sample and measure the detector's actual precision — do not assume it is good. Report the detector's own precision."*

---

# Chapter 13 — CPU deployment

## 13.1 FP32 vs INT8

| | Bits | Bytes | Range |
|---|---|---|---|
| **Float32** (IEEE 754) | 32 | 4 | 1 sign + 8 exponent + 23 mantissa |
| **INT8 signed** | 8 | 1 | −128 … 127 |
| **INT8 unsigned** | 8 | 1 | 0 … 255 |

A 66M-parameter DistilBERT: **268 MB** in FP32 → **~67 MB** in INT8. Exactly 4× smaller, a 75% reduction.

## 13.2 What ONNX is for

ONNX is a framework-agnostic computation-graph standard (DAGs of standard ops). Two reasons to export:

1. **Decouple training from serving.** Train in PyTorch, serve from ONNX Runtime — or C++, or a runtime with no GIL.
2. **Graph-level optimisation.** Constant folding, dead-node elimination, operator fusion (`Add`+`Gelu`, LayerNorm chains) into single optimised kernels.

## 13.3 Dynamic vs static quantisation

| | Weights | Activations | Calibration data |
|---|---|---|---|
| **Dynamic** | INT8 offline | FP32 stored; min/max computed **per forward pass**, then quantised on the fly | Not needed |
| **Static** | INT8 offline | INT8 offline, with fixed scale/zero-point | **Needed** — a representative dataset is run through to record activation histograms |

Dynamic is simpler and needs no calibration set. Static eliminates the per-inference min/max computation — which is precisely the overhead that eats the speedup on CPUs without integer dot-product instructions.

## 13.4 VNNI vs non-VNNI — where the speedup lives

**VNNI** (Intel DL Boost, Cascade Lake / Ice Lake) provides `VPDPBUSD`: an INT8 × INT8 → INT32 dot-product-and-accumulate in **one instruction**. Measured **1.59×** speedup.

**Non-VNNI x86 (AVX2 only, i.e. Haswell):** no single instruction exists. The runtime must unpack 8-bit integers into wider registers, issue multiple AVX2 multiply-adds, and repack. Add the dynamic-quantization min/max overhead per layer and the instruction overhead **cancels the savings**: **1.08×**. On older architectures it can be *slower than FP32*.

## 13.5 The AVX2 correctness bug

`vpmaddubsw` — the core AVX2 integer multiply-add — treats its **first operand as unsigned (U8)** and its **second as signed (S8)**.

A pipeline attempting **S8S8** (signed activations × signed weights) on AVX2 hits signed-overflow and saturation bugs, producing **numerically corrupted output** — a silently wrong classifier, which is worse than a crash.

**Recommendation, recorded in the project: U8U8 only.** Activations are zero-point-shifted into unsigned range and the hardware semantics match the arithmetic.

**The lesson:** quantisation is not a single decision. Format, calibration mode, *and* hardware instruction path must be chosen together, and the combination must be verified numerically — not assumed.

## 13.6 Batch-size-1 inference is memory-bandwidth-bound

At batch 1, a transformer is **not compute-bound** — it is bound by DRAM bandwidth. Every parameter must be fetched from main memory into cache to process a 16–35 token snippet. DistilBERT moves **268 MB across the memory bus for every single ticket**; the CPU spends far longer waiting on RAM than executing FLOPs.

Which is exactly why size is the latency lever:

| | Fused cascade | DistilBERT |
|---|---|---|
| Size | **1.6 MB** — fits in L3 cache | 268 MB FP32 / 67 MB INT8 |
| Per-ticket | **33.3 ms** | 105.5 ms |
| Macro-F1 | **94.43%** | 94.07% |
| ECE | **0.0170** | 0.0252 |
| CPU training | **2.8 s** fit, measured on the 20,431-feature word-only space; the 34,901-feature serving union is not separately timed | 47 min (`04-distilbert.json` → `train_minutes` 47.18) |
| Paired test | p = 0.6061 — statistically identical | |

**DRONA's architecture wins on the axis that actually binds** — memory traffic — and ties on the axis that is usually assumed to matter — accuracy.

---

# Chapter 14 — Everything rejected, on one page

| Technique | What it is | Measured verdict |
|---|---|---|
| **BM25** | TF saturation + length normalisation | Length normalisation is a no-op on 15.9-token tickets; losing char n-grams costs −3.26 acc / −4.09 F1 |
| **Dense bi-encoder** (bge-small, MiniLM) | Independent embeddings + vector search | R@15 already 99.75%; 10–20× slower; ~100 MB; Banking77 silhouette 0.10 |
| **Fine-tuned DistilBERT** | Full fine-tune, 66M params | +0.31 acc (p = 0.6061), **−0.36 macro-F1**, +48% ECE, 3.2× latency, 168× size, 47 min training |
| **Cross-encoder reranker** | Joint query+doc encoding | 1.58 s/ticket on top-15 = 47× regression. Affordable *only* on the escalation slice |
| **LLM-as-reranker** | SLM re-scores retrieved cases | **41.09% vs 64.34%. End-to-end 93.17% vs 94.70%. Monotonically worse.** |
| **Hierarchical coarse→fine** | Two-stage classifier | Error propagation on overlapping intents; adds latency to 100% of traffic |
| **Bagging / boosting** | Variance reduction | Wrong problem — not a variance issue |
| **Stacking** | Meta-learner on OOF preds | Nested CV, ~2 h extra compute, fractional gain |
| **Product of experts** | Multiply + renormalise | Assumes conditional independence; both views read the same text |
| **Learned router** (arXiv 2410.01627) | Adaptive cascade | Needs a *strictly stronger* stage 2; §2.5 proved none exists |
| **SetFit / ModernBERT** | Efficient few-shot / modern encoder | Good techniques; 13 accuracy points lost to under-tuning (arXiv 2504.01724), unaffordable at 47 min/config |
| **ONNX INT8 dynamic** | 8-bit quantisation | **1.08×** on non-VNNI Haswell, not the budgeted 2–4×; S8S8 correctness bug → U8U8 only |
| **SelectiveNet** (ICML 2019) | Learned reject head | A quantile has 0 parameters and cannot overfit |
| **Free-form LLM generation** | Autoregressive drafting | Unbounded risk; unbounded latency on CPU |
| **Learned priority model** | Supervised P0–P3 classifier | No public labels exist; text cannot see tier/amount/channel |
| **LLM priority prompting** | Prompt the model | 2,000–5,000 ms vs <1 ms; non-deterministic |
| **Shadow mode / canary** | Parallel evaluation, traffic ramp | Needs a load balancer; absent, and stated as absent |
| **Cleanlab relabelling** | Overwrite labels with OOF preds | **−2.50 accuracy, −3.18 macro-F1** |

---

# Chapter 15 — Corrections and caveats

## 15.1 Numbers that disagree across documents

These should be reconciled before submission — a mentor reading two sections will notice.

| Quantity | Values in circulation | Note |
|---|---|---|
| **Cascade latency** | 33.3 ms (README), 36.3 ms (report §4), 51 ms (report §1/§6) | Three figures; the 51 ms is explicitly *end-to-end in-process*, the others are component — but the report does not say so |
| **Auto-answered accuracy** | 97.13% (README/§6), 98.24% (coverage curve), 100% (conformal α=0.10) | **97.13% is the pre-calibration number** — `dashboard.py`'s own comment identifies it as stale. Different methods, but nothing labels them |
| **Cascade accuracy** | 94.60% (README/fusion), 94.70% (escalation exp) | Different evaluation paths; should be reconciled or footnoted |
| **Agent slice accuracy** | 41.09% (margin 0.0), 59.69% (margin 0.6), 0.59690 (`slice_agent_accuracy`) | **Not a contradiction** — `research/04` §4 shows these are different *policies*. The JSON **field name** is misleading, not the value |
| **DistilBERT training** | "3 epochs / batch 16 / seq 32 → 693 min" (§2.3) vs `max_len 64, batch 32, epochs 2, train_minutes 47.18` (`04-distilbert.json`) | §2.3's figure is the **pre-optimisation projection**, not the shipped run. Two different experiments described as one |
| **LinearSVC "94.50 vs 94.09"** | Not a contradiction | 94.50% = standalone baseline (`01-tfidf-baseline.json`); 94.09% = the cascade's linear component (`03-fusion.json`). Confusingly labelled |
| **§8 vs §5** | §8 says the relabelling experiment "is specified but not yet run"; §5 reports its result (−2.50) | **Stale future-work item** |
| **§9 section map** | `label_noise_exp → §7`, `final_comparison → §8` | §7 is limitations, §8 is future work. The material is in §4 |
| **Reproduction commands** | §9 `python -m drona.experiments.*` vs README `PYTHONPATH=src .venv/bin/python -m ...` | README's is the working one |

## 15.2 Design debts recorded but not fixed

- **Retriever blindness to identifiers** — sklearn's default `token_pattern r"(?u)\b\w\w+\b"` drops bare digits and single characters; `min_df=2` on the char view prunes unique reference numbers. Fix would be `token_pattern=r"(?u)\b\w+\b"`, `min_df=1`, or a separate numeric/entity field. Changing it would invalidate every committed artifact.
- **`store.next_id()` is not truly monotonic** — it derives the ID from `COUNT(*)`, so a deleted row or rolled-back insert causes reuse. The collision loop only guards rows that still exist.
- **`retrieval.save_index()` is a stub** that only creates a directory.
- **Bare `pytest tests/` fails** without `PYTHONPATH=src`.
- **Single seed (20261003)** — one sample, not a distribution.

## 15.3 Claims from the project's own research that are marked UNVERIFIED

Stated plainly in the source documents, and repeated here because unverified numbers should never travel into a presentation:

- Dense-encoder CPU latency (10–20 ms) — vendor **server**-CPU tables; no Haswell measurement exists.
- RAGAS faithfulness agreement rate — no published number.
- ARES / RAGTruth human agreement.
- CPU cost per 1,000 tickets at scale.
- Full agentic CPU loop latency from the literature.
- Cost-sensitive conformal percentages — all from non-indexed preprints, so only the *method* is cited.
- **No published accuracy exists for the 10-class Banking77 task at all.** 94.60% may have no direct comparator. `research/04` says this should be stated plainly "rather than implying a leaderboard we did not check."
- Several figures in `research/03` trace to SEO/content-farm blogs rather than vendor documentation or peer-reviewed venues. The project records this about itself.

---

# Chapter 16 — Reproducing everything

```bash
cd ~/drona
bash install-ml.sh          # venv + CPU torch + 15 packages
bash run.sh test            # 32 tests
bash run.sh api             # FastAPI on 0.0.0.0:8000  →  /docs
bash run.sh dashboard       # Streamlit on 0.0.0.0:8501
```

Experiments (from the README; note `PYTHONPATH=src` is required):

```bash
PYTHONPATH=src .venv/bin/python -m drona.experiments.baselines
PYTHONPATH=src .venv/bin/python -m drona.experiments.retrieval_exp
PYTHONPATH=src .venv/bin/python -m drona.experiments.fusion_exp
PYTHONPATH=src .venv/bin/python -m drona.experiments.transformer_exp   # ~47 min CPU
PYTHONPATH=src .venv/bin/python -m drona.experiments.escalation_exp
PYTHONPATH=src .venv/bin/python -m drona.experiments.final_comparison
PYTHONPATH=src .venv/bin/python -m drona.experiments.label_noise_exp
PYTHONPATH=src .venv/bin/python -m drona.experiments.conformal_exp
```

Each writes `artifacts/reports/0N-*.json`. `tools/patch_report.py` regenerates report sections from those artifacts — the mechanism that keeps prose and measurement in sync, and the one the report currently only half-uses.

---

# Chapter 17 — The argument, in one paragraph

DRONA's thesis is that on a CPU-only budget, **the scarce resource is not model capacity — it is knowing when not to answer.** Every design decision follows from that. It builds a retrieval stage that already contains the right answer 99.75% of the time; a linear classifier because it beats logistic regression by 1.78 points and trains in seconds; a sigmoid calibrator fitted on held-out data because hinge-loss margins are not probabilities; temperature scaling fitted by NLL because it provably cannot change a single prediction; a coverage-based gate because operations controls automation rate, not confidence; a conformal bound because an empirical curve is a measurement, not a guarantee; a monotonic rule engine for priority because under-escalation costs a thousand times more than over-escalation; and an agent that is **measured, demoted to advisory, and published** because a 0.5B model re-reading the same retrieved cases adds noise, not information. The transformer was built, benchmarked, and rejected on a paired statistical test — and that negative result is the project's strongest evidence that the architecture was chosen by measurement rather than by fashion.

---

*Compiled from the DRONA repository at commit `20c943b`. Every figure traces to `artifacts/reports/*.json`. Every rejection traces to `research/04-reranking-and-selection.md`.*