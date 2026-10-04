# 02 — Dataset Landscape (source: background research agent, 2026-10-03)

> **CRITICAL READ-THROUGH NOTE:** Sections 2 and 6 recommend GPT-4/Llama-3 synthetic data
> generation. **This is INFEASIBLE** — operator constraint #1 forbids any paid API and
> constraint #4 confirms no free LLM endpoint and no GPU. The "RECOMMENDED DATA STRATEGY"
> below is therefore *partially void*. Section 7 (GAPS AND OPPORTUNITIES) is the part that
> survived and it is the strategically valuable part. See `03-*.md` for the offline-safe
> data strategy actually adopted.

## 1. Intent-classification benchmarks

| Name | #Examples | #Classes | Real/Syn | License | URL | Notes |
|---|---|---|---|---|---|---|
| **Banking77** | 13,083 | 77 | Real | CC BY 4.0 | huggingface.co/datasets/PolyAI/banking77 | Train 10,003 / test 3,080. **14% label noise** (Ying & Thomas 2022, aclanthology.org/2022.insights-1.19.pdf). Cleaning → +4.5% F1, +8% ARI. Cosine silhouette 0.10. |
| **CLINC150** | 22,500 (15K/3K/4.5K) | 150 + OOS | Real | CC BY 4.0 | github.com/clinc/oos-eval | 150 intents, 10 domains, **out-of-scope class** — the canonical OOD benchmark. Silhouette 0.21. |
| **HWU64** | 23,475 | 64 | Real | CC BY 4.0 | huggingface.co/datasets/DeepPavlov/hwu64 | Personal-assistant + task-oriented. |
| **MASSIVE** | 1,132,644 | 60 | Synthetic (paraphrased from SLURP) | CC BY 4.0 | huggingface.co/datasets/AmazonScience/massive | **51 languages**, 18 domains, parallel. Not real tickets. |
| **SNIPS** | 14,484 | 7 | Real | CC BY 4.0 | github.com/snipsco/nlu-benchmark | Small but clean. |
| **ATIS** | 5,871 | 26 | Real | **LDC paid** | catalog.ldc.upenn.edu/LDC93S5 | Outdated airline domain. |
| **MIntRec2.0** | 15,040 (1,245 dialogues) | 30 | Real multi-turn | CC BY 4.0 | github.com/thu-coai/MIntRec2.0 | ICLR 2024. Real customer-support conversations. |
| **MICCS** | ~30,000 | 13 parent + 17 leaf | Real (logistics) | CC0 | arxiv.org/abs/2603.23172 | 2026. **Hierarchical** intent labels. |
| **Taskmaster-1/2/3** | 13.2K / 17.3K / 25K+ | — | Crowdsourced | CC BY 4.0 | github.com/google-research-datasets/Taskmaster | Conversational, not ticket-specific. |
| **SGD** | 16,142 | 16 services | Template-synthetic | CC BY 4.0 | github.com/google-research-datasets/dstc8-schema-guided-dialogue | Multi-domain dialogue. |
| **Wolof Banking Speech** | — | 77 | Translated | — | proceedings.neurips.cc/paper_files/paper/2025/file/b8946cbd97fcc555a28810aee63e525a-Paper-Datasets_and_Benchmarks_Track.pdf | NeurIPS 2025 D&B. Banking77 translated to Wolof. |

### Financial / banking beyond Banking77
FinCausal (~1,000, causality), FiQA-SA (1,173 binary sentiment), Financial PhraseBank (4,840, 3-class). None are ticket data.
**Indian fintech support tickets: NONE public.** PhonePe, Razorpay, Paytm, Zerodha all release nothing.

### Kaggle support-ticket sets
| Name | # | Labels | License | URL |
|---|---|---|---|---|
| Customer IT Support — Ticket (tobiasbueck) | ~10,000 | **Priority, queue, type** | CC BY 4.0 | kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets |
| Customer Support Ticket Dataset (suraj520) | ~8,000 | type, channel, status | CC0 | kaggle.com/datasets/suraj520/customer-support-ticket-dataset |
| IT Service Ticket Classification (adisongoh) | 47,837 | binary urgent/not | CC0 | kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset |
| Customer Support Tickets CRM (ajverse) | 20,000 | synthetic CRM | CC0 | kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset |

**tobiasbueck is the only public dataset found carrying real priority + queue labels.** That is directly valuable for the P0–P3 and routing objectives.

## 2. Synthetic data generation (partly infeasible — see note)
- Self-Instruct (Wang 2022, arxiv.org/abs/2212.10560); Evol-Instruct / WizardLM (arXiv 2304.12244); Distilabel (argilla-io.github.io/distilabel); AnyClassifier (huggingface.co/blog/kenhktsui/anyclassifier).
- **Synthetic ≠ real**: 6K GPT-3 synthetic SST-2 → 76% vs 88% from 3K human labels (−12%), arxiv.org/html/2503.14023v1. But augmentation at the *low-data* regime helps +3–26% at 100 real samples, diminishing past ~1,000.
- Counter-case: clinical NLP, an 8B student trained purely on synthetic reached 89.30%, beating its own 70B teacher (nature.com/articles/s41746-025-01681-4).
- **Soft-label (distribution) distillation beats hard-label**, gap widens with class count (link.springer.com/article/10.1007/s10462-025-11423-3).
- Validation practice: human-label agreement (Cohen's κ), MinHash/SimHash or embedding-cosine dedup, class-collapse monitoring, cleanlab scoring (cleanlab.ai/blog/learn/studio-synthetic-data/).

## 3. Indian / Telugu / code-switched
- **No public Telugu customer-support ticket dataset. No Telugu/Hinglish intent-classification benchmark.**
- Code-switching costs English-only classifiers: Telugu-English sentiment 80.22% (aclanthology.org/2026.mme-main.3.pdf); code-mixed classification 72.37% / 68.90% (isca-archive.org/interspeech_2025/biswas25_interspeech.pdf). General literature: **5–15% degradation**.
- Indic encoders: IndicBERT ALBERT 18M/12 langs (huggingface.co/ai4bharat/indic-bert); IndicBERT v2 BERT 278M/24 langs/20.9B tokens; IndicBERT v3 4B/14 langs/35B tokens; Sarvam-1 2B; Sarvam-105B; IndicSBERT sentence-embeddings (github.com/AI4Bharat/indicnlp_catalog); mBERT 178M/104; XLM-R 550M/100.
- Best open choice for Telugu: **IndicBERT v2** (278M) — though XLM-R is stronger and larger.

## 4. Priority / sentiment labelling
- ITIL defines **Priority = Impact × Urgency**, P0 critical → P3 low (axelos.com/certifications/itil-service-management). Zendesk and Jira use configurable priority schemes; neither publishes an ML method.
- **No academic paper found on how firms build P0–P3 labels — it is proprietary.** We will have to define and document our own, defensibly.
- Sentiment reference sets available offline: SST-2 (67,349), SST-5 (11,855), Financial PhraseBank (4,840, finance-specific 3-class), FiQA-SA (1,173).

## 5. Dataset size needed for >95% on a 10-class coarse task
| Classes | Samples/class | Total | Expected acc. |
|---|---|---|---|
| 10 | 100 | 1,000 | 85–90% (few-shot regime) |
| 10 | 500 | 5,000 | 90–93% |
| 10 | 1,000 | 10,000 | 93–95% |
| 10 | 2,000 | 20,000 | 95%+ |
| 77 | 130 | 10,003 | ~94% (actual Banking77 BERT) |
Also: SetFit+ModernBERT reaches 85–90% with **8–64 samples/class** (moshewasserblat.medium.com/new-results-on-setfit-modernbert-for-text-classification-with-few-shot-training-53c154df7c0e).

## 7. GAPS AND OPPORTUNITIES — the part that survived
| Gap | Opportunity |
|---|---|
| **No Indian fintech support-ticket dataset exists** | Create one. Genuine novel contribution, perfectly matched to a Telangana audience. |
| **No public P0–P3 SLA priority dataset** | Create one using ITIL Impact×Urgency, documented and validated. |
| **No Telugu customer-support dataset** | Create one — high novelty for a HYSEA Drona capstone. |
| **No code-switched intent-classification benchmark** | Create one — publishable research contribution. |
| **14% label noise in the canonical dataset is published but unaddressed** | Fix it, measure the delta, and publish the cleaned version. |

These five gaps are, collectively, the strongest available "shock the field" narrative that
also happens to be **fully feasible offline on 2 CPU cores in 2–4 weeks**.