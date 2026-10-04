# 01 — What DRONA Actually Is (source: https://hysea.in/project-drona/)

Fetched 2026-10-03. Facts only, no filler.

## Identity
**HYSEA Project Drona 3.0** — a *faculty development* programme of the **HYSEA Industry &
Academia Council** (Hyderabad Software Enterprises Association).

Partners: **HYSEA** + **TASK** (Telangana Academy for Skill and Knowledge) + **Infosys Springboard**.
Registration deadline for the 3.0 faculty cohort: **September 17, 2026** (already passed — user is enrolled).
Mentors are practising industry experts from HYSEA member companies who **evaluate the capstone and give feedback**.

## Why this reframes the project
The Smart Ticket Understanding Engine synopsis in
`00-original-synopsis.pdf` is **not** a hackathon brief. It is the
**industry-assigned capstone project** that a Drona faculty participant submits for
evaluation and certification.

Therefore the real audience is:
1. An **industry mentor** from a HYSEA member company (Cigniti, TCS, Infosys, Wipro, etc.)
2. **Dr. Parvez** (project guide, Associate Professor, CSE, MLRITM)
3. **Dr. N. Y. V. Subba Reddy / MLRITM HOD** — institution-level
4. **HYSEA + TASK + Infosys Springboard** as credentialing bodies

Judging is therefore weighted toward: **industry relevance, business impact, business
sensitivity, program-management discipline, and use of current technology** — explicitly
listed under "What's in it for the faculty": *Business Sense & Customer Centricity*,
*Program Management with focus on Professional Skills Development*.

## Drona 3.0 curriculum — what the cohort was explicitly taught
Track 1 — Foundation for AI (4 wks, Aug 4 – Sep 1): AI types, ML/DL, NLP pipeline, CNN/RNN.
Track 2 — Citizen Data Science (10 wks, Sep 2 – Nov 10): Python, NumPy/Pandas, Matplotlib/
Seaborn/Plotly, scikit-learn, EDA (incl. Simpson's paradox), classification algorithms,
ensemble techniques, clustering, ANN.
Track 3 — Generative AI (9 wks, Nov 10 – Jan 12):
- Discriminative vs Generative; transformer pretraining; GPT products
- Prompt Engineering — primer, principles, RE guidelines, text completion
- Responsible & Ethical AI — principles, governance, explainability, human-in-loop
- **LLMs, SLMs, RAG** — build a retrieval-augmented generation system; LLM significance and
  architecture; **limitations of LLMs**
- Applied Deep Learning for NLP, unsupervised learning, CNN/RNN for text
- **AI Application Development Lifecycle** (AIADL), SDLC integration
- **MLOps** — theoretical aspects, factors affecting ML models, stages in MLOps
- **Agentic AI** — aspects of agentic AI; build agents with ChatGPT and Zapier

## The strategic consequence
The synopsis's technical core — **TF-IDF+LogReg vs BiLSTM vs DistilBERT + ONNX INT8 on
Banking77 collapsed 77→10 labels** — is a **2021-era** recipe. Every item in Track 3 sits
*strictly above* it in the curriculum the same faculty cohort was just taught.

Submitting the synopsis design as-is means submitting a design that a reviewer can score as
"below the level of the course we just finished."

The Drona 3.0 thesis statement is **"Agentic AI, emerging technologies, leadership, and
industry application."** A capstone that is not agentic is off-thesis.

## What must therefore be true of the winning solution
- **Not** a single-shot classifier behind a REST endpoint.
- Must demonstrate a **closed loop / autonomous execution** with tools, memory and state.
- Must use **GenAI + RAG + SLM** in the way Track 3 taught, and show awareness of
  **LLM limitations** (hallucination) — measured, not asserted.
- Must show **Responsible AI**: explainability, human-in-the-loop, auditability.
- Must show **MLOps**: tracking, versioning, drift, reproducible pipelines.
- Must show **business/customer centricity**: SLA math, cost per ticket, agent-hours saved,
  CSAT impact — quantified.
- Must run on **CPU-only** hardware (this machine; see 00-VERIFIED-SYSTEM-SPECS).

## Scoring-relevant open questions for the operator
1. Is there a separate **live review/presentation date** that fixes the deadline?
2. Is the **capstone evaluated by the HYSEA mentor in person**? PowerPoint or live demo?
3. Are **external paid APIs permitted** (Claude/GPT/Gemini) or must inference be fully local/offline?
4. Is a **live demo** expected? (CPU-only i3 machine — demo must not need a GPU.)