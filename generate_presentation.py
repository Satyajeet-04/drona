import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank slide

    # Theme Colors
    BG_DARK = RGBColor(11, 25, 44)       # Deep Slate Navy #0B192C
    BG_LIGHT = RGBColor(248, 250, 252)   # Crisp Light Slate #F8FAFC
    CARD_BG = RGBColor(255, 255, 255)    # Pure White
    CARD_BORDER = RGBColor(226, 232, 240)
    PRIMARY_BLUE = RGBColor(31, 111, 235)# Royal Blue #1F6FEB
    ACCENT_TEAL = RGBColor(13, 148, 136) # Teal #0D9488
    TEXT_DARK = RGBColor(15, 23, 42)     # Charcoal Slate #0F172A
    TEXT_MUTED = RGBColor(100, 116, 139) # Muted Grey #64748B
    TEXT_LIGHT = RGBColor(241, 245, 249) # Light text
    ACCENT_GREEN = RGBColor(16, 185, 129)
    ACCENT_AMBER = RGBColor(217, 119, 6)

    def add_header(slide, title_text, category_text="HYSEA PROJECT DRONA 3.0"):
        # Header banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.9))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = PRIMARY_BLUE
        
        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_DARK

    def set_slide_bg(slide, color):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def create_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1)
        else:
            card.line.fill.background()
        return card

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Themed)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1, BG_DARK)
    
    # Title box
    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(4.5))
    tf1 = tbox.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "HYSEA PROJECT DRONA 3.0 · CAPSTONE PRESENTATION"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = RGBColor(56, 189, 248) # Sky blue
    
    p = tf1.add_paragraph()
    p.text = "DRONA: Smart Ticket Understanding Engine"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = TEXT_LIGHT
    p.space_after = Pt(12)
    
    p = tf1.add_paragraph()
    p.text = "Offline Dual-View Cascaded Triage, SLA-Priority Routing & Grounded Verifiable Escalation"
    p.font.size = Pt(18)
    p.font.color.rgb = RGBColor(203, 213, 225)
    p.space_after = Pt(28)
    
    p = tf1.add_paragraph()
    p.text = "Marri Laxman Reddy Institute of Technology and Management (MLRITM), JNTUH\nDepartment of Computer Science and Engineering"
    p.font.size = Pt(13)
    p.font.color.rgb = RGBColor(148, 163, 184)

    # =========================================================================
    # SLIDE 2: Team Members & Guidance
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s2, BG_LIGHT)
    add_header(s2, "Project Batch & Institutional Mentorship", "Batch Details")
    
    # Team Cards (4 members)
    members = [
        ("NAYAKAM SATYAJEET", "237Y1A0540", "CSE - AI/ML System Lead", "Cascade architecture, calibration & API engine"),
        ("JAKKU HARIKA", "237Y1A0570", "CSE - NLP & Data Engineer", "Intent taxonomy, Banking77 dataset & preprocessing"),
        ("SHERLA LOHITH KUMAR", "237Y1A0574", "CSE - Agent & Verification", "Escalation ReAct agent & hallucination check"),
        ("MAMIDALA VARUN", "237Y1A05A5", "CSE - UI & Evaluation", "Streamlit operations console & drift analysis"),
    ]
    
    card_w = Inches(2.7)
    card_h = Inches(4.8)
    start_x = Inches(0.8)
    spacing = Inches(0.3)
    
    for i, (name, roll, role, resp) in enumerate(members):
        cx = start_x + i * (card_w + spacing)
        create_card(s2, cx, Inches(1.5), card_w, card_h)
        tb = s2.shapes.add_textbox(cx + Inches(0.2), Inches(1.7), card_w - Inches(0.4), card_h - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = name
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = PRIMARY_BLUE
        
        p = tf.add_paragraph()
        p.text = roll
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(12)
        
        p = tf.add_paragraph()
        p.text = "Role:"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_MUTED
        
        p = tf.add_paragraph()
        p.text = role
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(10)
        
        p = tf.add_paragraph()
        p.text = "Key Focus:"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_MUTED
        
        p = tf.add_paragraph()
        p.text = resp
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Guide card across the bottom
    guide_card = create_card(s2, Inches(0.8), Inches(6.4), Inches(11.7), Inches(0.8), bg_color=RGBColor(241, 245, 249))
    tb_g = s2.shapes.add_textbox(Inches(1.0), Inches(6.45), Inches(11.3), Inches(0.7))
    tf_g = tb_g.text_frame
    p = tf_g.paragraphs[0]
    p.text = "Project Guide: Dr. PARVEZ, Associate Professor, Department of CSE — MLRITM (JNTUH)  |  Batch: MP-A05"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 3: Problem Statement
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s3, BG_LIGHT)
    add_header(s3, "The Enterprise Customer Support Bottleneck", "Problem Statement")
    
    # Left Card: Industry Bottlenecks
    create_card(s3, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.3))
    tb_l = s3.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.1), Inches(4.8))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    
    p = tf_l.paragraphs[0]
    p.text = "Current Support Realities & Pain Points"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(14)
    
    points_l = [
        ("High Manual Latency", "Human triaging takes 5–20 minutes per ticket, creating severe backlogs during peak traffic and delayed emergency response."),
        ("Opaque Commercial Vendor Costs", "Industry platforms (Zendesk AI, Salesforce) charge $1.50–$2.00 per resolution and hide actual model accuracy."),
        ("Cloud Dependency & Privacy Risk", "Banking regulations (RBI, GDPR) penalize shipping sensitive financial identifiers and transaction records to external commercial LLMs."),
        ("LLM Hallucination Hazards", "Generative LLM chatbots often invent fake tracking IDs or misquote policy, exposing institutions to severe compliance liability.")
    ]
    for title, desc in points_l:
        p = tf_l.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK
        p.add_run().text = desc
        p.runs[1].font.bold = False
        p.runs[1].font.size = Pt(11)
        p.runs[1].font.color.rgb = TEXT_MUTED
        p.space_after = Pt(10)

    # Right Card: Hardware & Resource Constraint Challenge
    create_card(s3, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    tb_r = s3.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.1), Inches(4.8))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    
    p = tf_r.paragraphs[0]
    p.text = "The Capstone Hardware Challenge"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_AMBER
    p.space_after = Pt(14)
    
    points_r = [
        ("Zero GPU Infrastructure", "Must deploy and serve on a commodity dual-core CPU with under 4 GB RAM footprint."),
        ("Strict Latency Ceiling", "Inference budget constrained to < 100 ms per ticket to process hundreds of requests per second."),
        ("The Fine-Tuning Trap", "Fine-tuning DistilBERT on CPU measured 47.2 min for 3 epochs on this box, and bought no statistically significant accuracy over a sparse baseline (McNemar p = 0.61)."),
        ("The Core Mandate", "Build a high-precision, offline banking triage engine with mathematical guarantees, explainable routing, and grounded generation.")
    ]
    for title, desc in points_r:
        p = tf_r.add_paragraph()
        p.text = f"• {title}: "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK
        p.add_run().text = desc
        p.runs[1].font.bold = False
        p.runs[1].font.size = Pt(11)
        p.runs[1].font.color.rgb = TEXT_MUTED
        p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 4: Objectives
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s4, BG_LIGHT)
    add_header(s4, "Project Goals & Measurable Engineering Targets", "Objectives")
    
    objs = [
        ("01", "Ultra-Fast Offline Triage", "Achieve sub-75 ms end-to-end triage latency on a standard dual-core CPU without GPU acceleration or cloud connectivity."),
        ("02", "Calibrated Selective Prediction", "Auto-answer >= 90% of inbound banking tickets with an audited accuracy >= 98% on the auto-accepted slice using conformal guarantees."),
        ("03", "Auditable SLA Priority Engine", "Implement multi-signal ITIL priority routing (P0 to P3) fusing text semantics with Indian financial amounts (₹/Lakh/Crore) and customer tiers."),
        ("04", "Grounded Agentic Escalation", "Deploy a tool-using ReAct loop only on the ~7% ambiguous/disagreement tickets, drafting replies grounded strictly in verified policies."),
        ("05", "Hallucination Defense & HITL", "Enforce atomic-fact verification that detects unsupported claims and automatically blocks replies with invented case/tracking identifiers."),
        ("06", "Real-Time Drift & Ops Console", "Deliver an interactive Streamlit operations console and stateless FastAPI endpoints with real-time distribution drift detection.")
    ]
    
    grid_w = Inches(3.7)
    grid_h = Inches(2.4)
    for idx, (num, title, desc) in enumerate(objs):
        row = idx // 3
        col = idx % 3
        gx = Inches(0.8) + col * (grid_w + Inches(0.3))
        gy = Inches(1.5) + row * (grid_h + Inches(0.3))
        
        create_card(s4, gx, gy, grid_w, grid_h)
        tb = s4.shapes.add_textbox(gx + Inches(0.2), gy + Inches(0.15), grid_w - Inches(0.4), grid_h - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = f"{num}. {title}"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = PRIMARY_BLUE
        p.space_after = Pt(6)
        
        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 5: Literature Survey
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s5, BG_LIGHT)
    add_header(s5, "Comparative Analysis & Theoretical Foundations", "Literature Survey")
    
    surveys = [
        ("Ovadia et al. (EMNLP 2024)", "RAG vs Fine-Tuning", "Proved RAG consistently outperforms fine-tuning for domain knowledge and avoids parameter degradation; guided DRONA's sparse retrieval architecture."),
        ("Ying & Thomas (ACL 2022)", "Banking77 Hardness", "Demonstrated that fine-grained banking intents suffer from extreme boundary overlap; coarse grouping resolves fine-intent ambiguity."),
        ("FActScore (Min et al., 2023)", "Atomic Fact Verification", "Introduced automated claim decomposition for precision checking with <2% human annotation error; adopted for DRONA's hallucination verifier."),
        ("Angelopoulos & Bates (2023)", "Split-Conformal Prediction", "Provides distribution-free, finite-sample statistical coverage guarantees; replaced arbitrary heuristics with rigorous thresholding (q = 0.7635)."),
        ("Guo et al. (ICML 2017)", "Temperature Scaling", "Showed single-parameter temperature scaling calibrates neural confidence without degrading accuracy (DRONA calibrated from T=3.0 to T=0.75).")
    ]
    
    sw = Inches(11.7)
    sh = Inches(0.9)
    for i, (auth, topic, finding) in enumerate(surveys):
        sy = Inches(1.5) + i * (sh + Inches(0.15))
        create_card(s5, Inches(0.8), sy, sw, sh)
        tb = s5.shapes.add_textbox(Inches(1.0), sy + Inches(0.1), sw - Inches(0.4), sh - Inches(0.2))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = f"{auth}  —  "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = PRIMARY_BLUE
        
        p.add_run().text = topic
        p.runs[1].font.bold = True
        p.runs[1].font.size = Pt(12)
        p.runs[1].font.color.rgb = ACCENT_TEAL
        
        p = tf.add_paragraph()
        p.text = finding
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 6: Methodology
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s6, BG_LIGHT)
    add_header(s6, "Dual-View Sparse Cascade & Multi-Signal Fusion", "Methodology")
    
    m_cards = [
        ("Stage 1: Dual-View Retrieval", [
            ("Character n-grams (3-5)", "Captures typos, morphological variations, and Hindi/Hinglish slang terms."),
            ("Word n-grams (1-2)", "Extracts precise semantic domain terminology and banking keywords."),
            ("L2-Normalized Union", "Generates high-dimensional sparse representations; the full kNN path (vectorise, score, vote) measures 27.4 ms per ticket on the test split."),
            ("Oracle Recall @ 15", "Reaches 99.75% ceiling—guaranteeing the gold category exists in candidate set.")
        ]),
        ("Stage 2: Calibrated Cascade", [
            ("LinearSVC Class Head", "Fits fast decision hyperplanes directly over the dual-view n-gram features."),
            ("Temperature Scaling", "Applies post-hoc optimization (T = 0.75) to calibrate raw margins into true probabilities."),
            ("Disagreement Gating", "Compares sparse kNN top vote with linear classifier vote—disagreement gives free error signal."),
            ("Conformal Abstention", "Auto-accepts if views agree and confidence >= 0.7635; routes remainder to agent.")
        ]),
        ("Stage 3: Priority & Escalation", [
            ("Category Floor Impact", "Sets base priority: Fraud/Failure = P0/P1, General = P3 (impact overrides words)."),
            ("Indian Financial Parsing", "Regex engine extracts ₹, INR, Lakhs, Crores, and grouped digits (e.g., 1,50,000)."),
            ("Non-Text Signals", "Fuses customer tier (Enterprise), channel (Phone/IVR), and SLA clock proximity."),
            ("Grounded Agent Trace", "Agent reranks candidates using tools, drafts reply from KB, and verifies claims.")
        ])
    ]
    
    col_w = Inches(3.7)
    col_h = Inches(5.2)
    for idx, (title, items) in enumerate(m_cards):
        cx = Inches(0.8) + idx * (col_w + Inches(0.3))
        create_card(s6, cx, Inches(1.5), col_w, col_h)
        tb = s6.shapes.add_textbox(cx + Inches(0.2), Inches(1.7), col_w - Inches(0.4), col_h - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = PRIMARY_BLUE
        p.space_after = Pt(12)
        
        for head, body in items:
            p = tf.add_paragraph()
            p.text = f"• {head}: "
            p.font.bold = True
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_DARK
            p.add_run().text = body
            p.runs[1].font.bold = False
            p.runs[1].font.size = Pt(10)
            p.runs[1].font.color.rgb = TEXT_MUTED
            p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 7: System Architecture
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s7, BG_LIGHT)
    add_header(s7, "End-to-End CPU Pipeline & Execution Dataflow", "System Architecture")
    
    # 4 Architecture Horizontal Pipeline Stages
    stages = [
        ("1. Inbound Ingestion", "Web, Email, Chat, Mobile App", [
            "Raw ticket string + CRM metadata",
            "Signals: Tier, Channel, Amount",
            "SQLite audit sequence ID allocated"
        ], RGBColor(241, 245, 249)),
        ("2. Dual-View Cascade", "In-Memory C++ Fast Sparse Scorer", [
            "TF-IDF Char & Word Union (~4 ms)",
            "LinearSVC + kNN vote comparison",
            "Calibrated confidence thresholding",
            "Accept (90%) vs Escalate (10%)"
        ], RGBColor(238, 242, 255)),
        ("3. SLA & Agent Engine", "Rules + ReAct Verification Loop", [
            "Lakhs/Crore risk & urgency scoring",
            "ITIL queue routing (P0 to P3)",
            "Tool execution: KB lookup & history",
            "Atomic claim fact verification"
        ], RGBColor(240, 253, 244)),
        ("4. Operations & HITL", "Serving & Governance Endpoints", [
            "Streamlit Live Operations Console",
            "FastAPI REST Endpoints (/triage)",
            "Human-in-the-loop review override",
            "Live category drift tracking"
        ], RGBColor(254, 243, 199))
    ]
    
    pw = Inches(2.7)
    ph = Inches(5.2)
    for idx, (title, sub, bullets, bg) in enumerate(stages):
        px = Inches(0.8) + idx * (pw + Inches(0.3))
        create_card(s7, px, Inches(1.5), pw, ph, bg_color=bg)
        tb = s7.shapes.add_textbox(px + Inches(0.2), Inches(1.7), pw - Inches(0.4), ph - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        
        p = tf.add_paragraph()
        p.text = sub
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = PRIMARY_BLUE
        p.space_after = Pt(14)
        
        for b in bullets:
            p = tf.add_paragraph()
            p.text = f"→ {b}"
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 8: Project Design
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s8, BG_LIGHT)
    add_header(s8, "Database, Taxonomy & Anti-Hallucination Design", "Project Design")
    
    # Left Card: Taxonomy & Hierarchy
    create_card(s8, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.3))
    tb_des_l = s8.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.1), Inches(4.8))
    tf_dl = tb_des_l.text_frame
    tf_dl.word_wrap = True
    
    p = tf_dl.paragraphs[0]
    p.text = "11-Class Taxonomy & Routing Hierarchy"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(12)
    
    tax_info = [
        ("77 Fine Intents → 11 Coarse Classes", "Maps all Banking77 intents into operational domains (Payment Failure, Fraud, Account Access, Card Services, Billing, KYC, etc.)."),
        ("Honest Taxonomy Handling", "Identified that 'Loan/EMI' has 0 samples in Banking77; kept as an auditable category rather than falsifying coverage."),
        ("Closed-Set 'Other' Escape Hatch", "Includes an unrouted escape class to prevent forced overconfident classification on out-of-domain queries."),
        ("Deterministic SLA Routing", "Maps (Category, Priority) to 9 specialized queues (fraud-ops, payments-ops, billing-ops, etc.).")
    ]
    for h, b in tax_info:
        p = tf_dl.add_paragraph()
        p.text = f"• {h}: "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK
        p.add_run().text = b
        p.runs[1].font.bold = False
        p.runs[1].font.size = Pt(11)
        p.runs[1].font.color.rgb = TEXT_MUTED
        p.space_after = Pt(10)

    # Right Card: Verifier & HITL Design
    create_card(s8, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    tb_des_r = s8.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.1), Inches(4.8))
    tf_dr = tb_des_r.text_frame
    tf_dr.word_wrap = True
    
    p = tf_dr.paragraphs[0]
    p.text = "Grounded Verifier & Persistence Schema"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_TEAL
    p.space_after = Pt(12)
    
    ver_info = [
        ("Atomic Claim Decomposition", "Draft replies are split into individual sentences and checked against retrieved neighbor tickets and knowledge base policies."),
        ("Invented Identifier Blocker", "Regex sweeps draft for case/ticket references (e.g. REF-12345); any code not present in evidence sets releasable = False."),
        ("Thread-Safe SQLite Store", "Persists tickets, calibrated confidence scores, full agent traces, and human overrides with ACID guarantees."),
        ("Distribution Drift Detector", "Calculates live category percentage share vs training baseline and triggers alerts after >= 30 sample window.")
    ]
    for h, b in ver_info:
        p = tf_dr.add_paragraph()
        p.text = f"• {h}: "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_DARK
        p.add_run().text = b
        p.runs[1].font.bold = False
        p.runs[1].font.size = Pt(11)
        p.runs[1].font.color.rgb = TEXT_MUTED
        p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 9: Expected Outcomes & Results
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s9, BG_LIGHT)
    add_header(s9, "Empirical Validation & Benchmark Results", "Expected Outcomes")
    
    # 4 Key Metric Badges Across Top
    metrics = [
        ("98.24%", "Accepted Accuracy", "at 90% auto-answer coverage"),
        ("33 ms", "Single-Ticket Latency", "dual-core CPU in-process"),
        ("99.75%", "Oracle Recall @ 15", "gold label in candidate set"),
        ("p = 0.61", "McNemar vs DistilBERT", "no statistical loss vs transformer")
    ]
    bw = Inches(2.7)
    bh = Inches(1.6)
    for i, (val, lbl, sub) in enumerate(metrics):
        bx = Inches(0.8) + i * (bw + Inches(0.3))
        create_card(s9, bx, Inches(1.5), bw, bh, bg_color=CARD_BG)
        tb = s9.shapes.add_textbox(bx + Inches(0.15), Inches(1.6), bw - Inches(0.3), bh - Inches(0.2))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = val
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = PRIMARY_BLUE
        
        p = tf.add_paragraph()
        p.text = lbl
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        
        p = tf.add_paragraph()
        p.text = sub
        p.font.size = Pt(9)
        p.font.color.rgb = TEXT_MUTED

    # Bottom Table: Comparative Decision Matrix
    create_card(s9, Inches(0.8), Inches(3.4), Inches(11.7), Inches(3.4))
    tb_t = s9.shapes.add_textbox(Inches(1.1), Inches(3.6), Inches(11.1), Inches(3.0))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    
    p = tf_t.paragraphs[0]
    p.text = "Comparative Benchmark on Held-Out Test Split (n = 1,963)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    p.space_after = Pt(10)
    
    results_summary = [
        ("TF-IDF Baseline (LinearSVC)", "94.50%", "94.31%", "N/A (uncalibrated)", "<0.01 ms", "$0.00"),
        ("kNN Sparse Retrieval (Union)", "92.56%", "91.75%", "0.078", "27.4 ms", "$0.00"),
        ("Fine-Tuned DistilBERT (Mandate)", "94.91%", "94.07%", "0.025", "105.5 ms", "47.2 min train"),
        ("DRONA Calibrated Cascade (Ours)", "98.24% (Accepted)", "94.43%", "0.017 (calibrated)", "33.3 ms", "$0.00 (CPU native)"),
        ("Commercial Vendors (Zendesk / Intercom)", "Unpublished", "Unpublished", "Opaque", "3,000–8,000 ms", "$1.50–$2.00 / ticket")
    ]
    for model, acc, f1, cal, lat, cost in results_summary:
        p = tf_t.add_paragraph()
        p.text = f"• {model}: "
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = PRIMARY_BLUE if "DRONA" in model else TEXT_DARK
        p.add_run().text = f"Acc: {acc}  |  Macro-F1: {f1}  |  Latency: {lat}  |  TCO: {cost}"
        p.runs[1].font.bold = False
        p.runs[1].font.size = Pt(11)
        p.runs[1].font.color.rgb = TEXT_MUTED
        p.space_after = Pt(4)

    # =========================================================================
    # SLIDE 10: Future Plan & Roadmap
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s10, BG_LIGHT)
    add_header(s10, "Production Deployment & Next-Phase Roadmap", "Future Plan")
    
    phases = [
        ("Phase 1: Multi-Channel Connectors (Q4 2026)", [
            "Build production webhook adapters for WhatsApp Business Cloud API and Twilio SMS.",
            "Integrate bidirectional email SMTP/IMAP parsers with attachment stripping.",
            "Package as a drop-in middleware microservice for Zendesk and Freshdesk."
        ]),
        ("Phase 2: Vernacular Indian NLP (Q1 2027)", [
            "Expand character-level n-gram tokenization to Hindi, Telugu, and Tamil script.",
            "Incorporate IndicBERT / AI4Bharat embeddings for regional banking slang.",
            "Support voice-note transcription triage directly from WhatsApp voice notes."
        ]),
        ("Phase 3: Active Feedback & Auto-Retraining (Q2 2027)", [
            "Stream human overrides from SQLite store into continuous weak supervision pipelines.",
            "Implement automated canary deployment with statistical KS-test drift triggers.",
            "Deploy on-device edge SDK for banking mobile applications to pre-triage on client."
        ])
    ]
    
    cw = Inches(3.7)
    ch = Inches(5.2)
    for idx, (title, bullets) in enumerate(phases):
        cx = Inches(0.8) + idx * (cw + Inches(0.3))
        create_card(s10, cx, Inches(1.5), cw, ch)
        tb = s10.shapes.add_textbox(cx + Inches(0.2), Inches(1.7), cw - Inches(0.4), ch - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = PRIMARY_BLUE
        p.space_after = Pt(14)
        
        for b in bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(10)

    # Output presentation
    output_path = "/home/satyajeet-rp/drona/DRONA_Project_Presentation.pptx"
    prs.save(output_path)
    print(f"Presentation successfully generated at: {output_path}")

if __name__ == "__main__":
    create_deck()
