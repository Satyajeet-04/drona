# 03 — What Production Actually Ships (source: background research, 2026-10-03)

## THE CENTRAL FINDING (use this in the pitch)

**No production vendor publishes a classification-accuracy number.**

| Vendor | Publishes accuracy? | What it actually publishes |
|---|---|---|
| Zendesk AI | **NO** | Resolution pricing $1.50–$2.00/resolution. Support thread literally complains "It makes us fit intents into the categories that barely have anything to do with the custom intents we create." |
| Salesforce Einstein Case Classification | **NO** | Tenant-specific dashboard only; needs ≥400 closed cases (1,000+ ideal) just to train |
| ServiceNow Now Assist | **NO** | Self-reported, vague: "75–85% suggestion accuracy" comes from a *third-party blog*, "40–55% reduction" (reduction of what? unspecified) |
| Intercom Fin | **NO** | 76% resolution (Jun 2026), up from 41% at launch |
| Freshworks Freddy | **NO** | "Up to 80% of common queries" — no independent study exists |
| Atlassian Rovo | **NO** | 90,000 tickets auto-classified, no accuracy |

**The 2× credibility gap:** Intercom claims 76% resolution; an independent test across
500 tickets at 4 SMBs measured **38%** — "meaningfully under Intercom's 'up to 50%' marketing claim."
Resolution rate also collapses with KB quality: comprehensive KB 45–55%, good 35–45%,
basic FAQ 25–35%, sparse <25%. (builts.ai/blog/intercom-fin-ai-review/)

Industry context: only **24% of consumers** say AI fully resolved their last issue (Ada/NewtonX 2026).
Vendor claims span 45–76%, each defined differently.

=> **A capstone that publishes an audited, reproducible classification accuracy with a
calibration curve is more scientifically advanced than the $2B vendors.** This is the
"shock the field" narrative, and it is 100% feasible on our hardware.

## THE TECHNIQUES PRODUCTION USES THAT AN ACADEMIC PROJECT SKIPS

**1. Retrieval-Augmented Classification beats fine-tuning.**
Ovadia et al., EMNLP 2024 (aclanthology.org/2024.emnlp-main.15.pdf):
> "RAG consistently outperforms [fine-tuning], both for existing knowledge encountered during
training and entirely new knowledge... combining RAG with fine-tuning shows inferior performance
compared to RAG alone."

Corroborated by arXiv 2403.01432 (StableLM2 1.6B; RAG beats fine-tuning by a large margin; the
*smallest* model tested benefited most). Fine-tuning injects knowledge but degrades reasoning.
**Implication: the right architecture is retrieve-similar-resolved-tickets → classify, not just fine-tune.**
This is kNN-LM / case-based reasoning. Production LLM-native vendors (Intercom Fin, Sierra,
Decagon) all do this.

**2. Structured output.** Prompt-only JSON 80–95% reliable; function calling 95–99%;
**native constrained decoding 100% schema-valid** (logit masking / XGrammar).
Design rule: always give categorical fields an escape value (`unknown`/`other`) — a model forced
to choose from an enum that does not fit picks wrong *with full confidence*.

**3. Selective prediction + abstention is universal** but opaque. Class-dependent
miscalibration can make a model retain hard cases and reject easy ones — the exact opposite of
what you want. Temperature scaling is accuracy-preserving and improves selective classification
(arXiv 2603.02719; NeurIPS 2025 poster 118852).

**4. Hierarchical coarse→fine heads** (10 coarse categories → 50–100 fine), standard in
ServiceNow task classification and Zendesk intelligent triage.

**5. Multi-signal fusion for priority:** text intent + channel + customer tier + account value +
transaction amount + ticket history + sentiment + SLA-deadline proximity. Priority is a weighted
fusion, never text-only.

**6. Shadow mode / canary:** new model runs in parallel, predictions logged not acted on, compare
agreement rate, then ramp 5% → 100%.

## AGENT FRAMEWORKS (all CPU-feasible, all model-agnostic)
LangGraph, Pydantic AI (1.0 Sep 2025), CrewAI, Google ADK 2.0 (May 2026), smolagents
(HF, ~1000-line core), Mastra (TS), Microsoft Agent Framework (AutoGen successor), Strands
Agents (AWS). OpenAI/Claude Agents SDKs require a remote API → **excluded by our constraints**.
Framework overhead is negligible; the bottleneck is always LLM inference itself.

**SLMs that can drive an agent on CPU (2026):**
| Model | Params | License | Function calling |
|---|---|---|---|
| **Phi-4-mini-instruct** | 3.8B | MIT | **Best** — ranked #1 in tool-calling benchmark, 0 wrong tool calls, 5,460 ms avg latency |
| Qwen3 4B / Qwen3.5 4B | 4B | Apache-2.0 | Yes |
| SmolLM3 | 3B | Apache-2.0 | Yes |
| Llama 3.2 3B / 1B | 3B/1B | Llama Community | Limited |

Standard agentic patterns: ReAct, plan-and-execute, **router**, multi-agent supervisor,
**reflection/critic** (draft → check faithfulness → revise), tool use with function calling.

## EVALUATING GENERATED REPLIES
| Metric | Claim | Human agreement |
|---|---|---|
| **FActScore** (arXiv 2305.14251) | atomic-fact precision vs knowledge source; automated estimator has **<2% error rate**; validated on 6,500 generations that would have cost $26K in human labelling | **<2% error — trustworthy** |
| **ARES** (arXiv 2311.09476) | few hundred human annotations + prediction-powered inference; robust across domain shift | few hundred labels |
| RAGAS faithfulness (arXiv 2309.15217) | claim-supported ratio | **UNVERIFIED — no published agreement rate** |
| RAGTruth / TRUE / AlignScore | not verified this session | UNVERIFIED |

**Guardrail reality:** constrained decoding guarantees *structural* validity, not *semantic*
correctness — a model can emit schema-valid JSON containing fabricated policy. That gap is exactly
where FActScore-style atomic fact verification earns its place. No major support vendor
publicly reports FActScore.

## COST — the one axis where we win by 5 orders of magnitude
| | Cost / 1,000 tickets |
|---|---|
| Zendesk AI | $1,500–$2,000 |
| Freshworks Freddy | ~$490 |
| **This project (CPU electricity)** | **~$0.01–$0.05** |

## LATENCY — the axis where we lose, so engineer around it
| Stage | Estimate |
|---|---|
| DistilBERT int8 classification | 5–15 ms |
| bge-small embedding for retrieval | 10–20 ms |
| Phi-4-mini few-shot classification | 2,000–5,000 ms |
| Phi-4-mini reply drafting | 3,000–8,000 ms |
| **Full agentic triage loop** | **~5,000–13,000 ms** |

Published Phi-4-mini throughput: 30–50 tok/s laptop CPU; 35–40 tok/s iPhone 14 Pro; 27–30 tok/s Pixel 8.
**No published benchmark exists for a full CPU-only agentic triage loop.** We will be the first to
publish one — a legitimate research contribution.

CPU-only triage is essentially free in money; the binding constraint is **latency**, which is why
async queue triage works and real-time chat does not. **This is the argument for a
confidence-gated cascade**: run the expensive reasoning only on the ~5–10% of tickets that are
actually hard.

## WHERE THE ACADEMIC PROJECT LEGITIMATELY BEATS PRODUCTION
1. **Auditable decisions** — published confusion matrix, per-class P/R, calibration curve, abstention rate. Impossible for vendors (proprietary, tenant-varying).
2. **30,000–200,000× cost reduction** — verifiable TCO on commodity hardware.
3. **Retrieval-augmented classification with published accuracy** — the step vendors hide inside "resolution rate".
4. **FActScore-guarded replies (<2% error metric)** — stricter than any vendor's published guardrail.
5. **Principled selective prediction** — a published calibration curve instead of an opaque confidence threshold.

## Claims to treat as UNVERIFIED
DistilBERT-int8 / bge-small / MiniLM / Qwen3-Emb CPU latency; RAGTruth arXiv ID; RAGAS & ARES human
agreement; CPU cost per 1k tickets; full agentic CPU loop latency; Cognigy/Yellow.ai/Parloa
resolution rates; Salesforce Einstein aggregate accuracy.

## Flaw in the agent's own report (recorded for honesty)
Several figures are sourced from SEO/content-farm blogs (getmacha.com, eesel.ai, builts.ai,
localaimaster.com, promptquorum.com), **not** from vendor docs or peer-reviewed venues. Treat the
*direction* of those claims as a lead to verify, not as evidence. Reliable items are the
peer-reviewed ones: Ovadia EMNLP 2024, FActScore, ARES, SetFit, Galke et al. CSUR 2025,
Ying & Thomas ACL Insights 2022.