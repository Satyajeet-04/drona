"""Generate publication-quality diagrams for the Smart Ticket Understanding Engine report."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

def generate_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)

    # Title
    ax.text(50, 56, "Smart Ticket Understanding Engine — System Architecture", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#111827")

    # Ingestion Box
    ax.add_patch(patches.FancyBboxPatch((2, 10), 18, 38, boxstyle="round,pad=1", ec="#0284c7", fc="#f0f9ff", lw=2))
    ax.text(11, 44, "1. Ingestion Layer", ha="center", fontweight="bold", color="#0369a1", fontsize=11)
    ax.text(11, 38, "• Web Portal API\n• Support Emails\n• Live Chat Dialogue\n• Bulk CSV Upload\n• PII Anonymizer", 
            ha="center", va="top", color="#334155", fontsize=9.5)

    # Arrow 1
    ax.annotate("", xy=(24, 29), xytext=(20, 29), arrowprops=dict(arrowstyle="->", lw=2, color="#0284c7"))

    # Feature & Retrieval Box
    ax.add_patch(patches.FancyBboxPatch((24, 10), 22, 38, boxstyle="round,pad=1", ec="#059669", fc="#ecfdf5", lw=2))
    ax.text(35, 44, "2. NLU & Retrieval", ha="center", fontweight="bold", color="#047857", fontsize=11)
    ax.text(35, 38, "• char_wb (3-5) ngrams\n• word (1-2) ngrams\n• Union Space (34.9k)\n• kNN Cosine Search\n  (99.75% Recall@15)\n• LinearSVC Head", 
            ha="center", va="top", color="#334155", fontsize=9.5)

    # Arrow 2
    ax.annotate("", xy=(50, 29), xytext=(46, 29), arrowprops=dict(arrowstyle="->", lw=2, color="#059669"))

    # Fusion & Gate Box
    ax.add_patch(patches.FancyBboxPatch((50, 10), 22, 38, boxstyle="round,pad=1", ec="#7c3aed", fc="#f5f3ff", lw=2))
    ax.text(61, 44, "3. Confidence Gate", ha="center", fontweight="bold", color="#6d28d9", fontsize=11)
    ax.text(61, 38, "• Calibrated Fusion\n  (alpha=0.6, Temp=0.75)\n• Conformal Gate (alpha=0.1)\n• Disagreement Check\n• ECE: 0.0170\n• 90.3% Auto-Answered", 
            ha="center", va="top", color="#334155", fontsize=9.5)

    # Arrow 3
    ax.annotate("", xy=(76, 29), xytext=(72, 29), arrowprops=dict(arrowstyle="->", lw=2, color="#7c3aed"))

    # Priority & Deflection Box
    ax.add_patch(patches.FancyBboxPatch((76, 10), 22, 38, boxstyle="round,pad=1", ec="#d97706", fc="#fffbeb", lw=2))
    ax.text(87, 44, "4. Triage & Routing", ha="center", fontweight="bold", color="#b45309", fontsize=11)
    ax.text(87, 38, "• ITIL SLA Priority\n  (P0, P1, P2, P3)\n• Entity Extraction\n• Proactive Deflection\n• ReAct Advisory Agent\n• SQLite Audit Store", 
            ha="center", va="top", color="#334155", fontsize=9.5)

    # Bottom Banner
    ax.add_patch(patches.Rectangle((2, 2), 96, 6, fc="#1e293b", ec="none"))
    ax.text(50, 5, "100% Offline CPU Execution | 33.3 ms Latency | 1.6 MB Model Size | Zero External API Calls", 
            ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_system_architecture.png", dpi=300)
    plt.close(fig)

def generate_use_case():
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.text(50, 96, "UML Use Case Diagram — Smart Ticket Understanding Engine", ha="center", fontsize=13, fontweight="bold")

    # Boundary Box
    ax.add_patch(patches.Rectangle((25, 5), 50, 85, fill=False, ec="#475569", lw=2, linestyle="--"))
    ax.text(50, 86, "System Boundary: Smart Ticket Understanding Engine", ha="center", fontsize=9.5, style="italic", color="#475569")

    # Use Cases (Ellipses)
    ucs = [
        (50, 78, "Submit Inbound Ticket / Email / Chat"),
        (50, 66, "Auto-Extract Intent, Entities & Sentiment"),
        (50, 54, "Evaluate Conformal Abstention Gate"),
        (50, 42, "Auto-Deflect Routine Inquiries (90.3%)"),
        (50, 30, "Review Grounded Draft Reply (HITL)"),
        (50, 18, "Override Category / Priority & Log Audit"),
        (50, 8, "Inspect Drift & Operational Metrics"),
    ]

    for x, y, text in ucs:
        ax.add_patch(patches.FancyBboxPatch((x-20, y-3.5), 40, 7, boxstyle="round,pad=0.8", ec="#2563eb", fc="#eff6ff", lw=1.5))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, fontweight="bold")

    # Actors
    # Actor 1: Customer (Left)
    ax.add_patch(patches.Circle((12, 78), 3, ec="#1e293b", fc="#e2e8f0", lw=1.5))
    ax.text(12, 70, "Customer /\nClient", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.plot([15, 30], [75, 78], color="#64748b", lw=1.5)

    # Actor 2: Support Agent (Right Top)
    ax.add_patch(patches.Circle((88, 40), 3, ec="#1e293b", fc="#e2e8f0", lw=1.5))
    ax.text(88, 32, "Support\nAgent (HITL)", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.plot([85, 70], [40, 42], color="#64748b", lw=1.5)
    ax.plot([85, 70], [40, 30], color="#64748b", lw=1.5)

    # Actor 3: Supervisor / Auditor (Right Bottom)
    ax.add_patch(patches.Circle((88, 14), 3, ec="#1e293b", fc="#e2e8f0", lw=1.5))
    ax.text(88, 6, "Supervisor /\nAuditor", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.plot([85, 70], [14, 18], color="#64748b", lw=1.5)
    ax.plot([85, 70], [14, 8], color="#64748b", lw=1.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_use_case_diagram.png", dpi=300)
    plt.close(fig)

def generate_dfd():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)

    ax.text(50, 66, "Data Flow Diagram (Level 1) — Smart Ticket Understanding Engine", ha="center", fontsize=13, fontweight="bold")

    # External Entity Customer
    ax.add_patch(patches.Rectangle((4, 28), 14, 16, ec="#1e293b", fc="#f8fafc", lw=2))
    ax.text(11, 36, "Customer /\nChannels", ha="center", va="center", fontweight="bold", fontsize=10)

    # Process 1.0 Tokenization & Entity Extraction
    ax.add_patch(patches.Circle((32, 36), 8, ec="#0284c7", fc="#e0f2fe", lw=2))
    ax.text(32, 36, "1.0\nEntity\nExtraction", ha="center", va="center", fontweight="bold", fontsize=8.5)

    # Process 2.0 Cascade Classification
    ax.add_patch(patches.Circle((55, 36), 8, ec="#059669", fc="#d1fae5", lw=2))
    ax.text(55, 36, "2.0\nCascade\nNLU Model", ha="center", va="center", fontweight="bold", fontsize=8.5)

    # Process 3.0 Conformal Gating & Routing
    ax.add_patch(patches.Circle((78, 36), 8, ec="#d97706", fc="#fef3c7", lw=2))
    ax.text(78, 36, "3.0\nConformal\nTriage Gate", ha="center", va="center", fontweight="bold", fontsize=8.5)

    # Data Stores
    ax.add_patch(patches.Rectangle((48, 54), 16, 8, ec="#475569", fc="#f1f5f9", lw=1.5))
    ax.text(56, 58, "D1: Training Corpus\n(9,158 Tickets)", ha="center", va="center", fontsize=8)

    ax.add_patch(patches.Rectangle((70, 8), 18, 9, ec="#475569", fc="#f1f5f9", lw=1.5))
    ax.text(79, 12.5, "D2: SQLite Audit Log\n(Decisions & Traces)", ha="center", va="center", fontsize=8)

    # Arrows
    ax.annotate("Raw Query", xy=(24, 36), xytext=(18, 36), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284c7"))
    ax.annotate("Tokens + Entities", xy=(47, 36), xytext=(40, 36), arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))
    ax.annotate("Vectors", xy=(55, 44), xytext=(55, 54), arrowprops=dict(arrowstyle="->", lw=1.5, color="#475569"))
    ax.annotate("Fused Probs", xy=(70, 36), xytext=(63, 36), arrowprops=dict(arrowstyle="->", lw=1.5, color="#d97706"))
    ax.annotate("Audit Record", xy=(78, 17), xytext=(78, 28), arrowprops=dict(arrowstyle="->", lw=1.5, color="#475569"))

    # Outputs
    ax.add_patch(patches.Rectangle((90, 48), 9, 14, ec="#16a34a", fc="#dcfce7", lw=1.5))
    ax.text(94.5, 55, "Auto-Reply\n(90.3%)", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.annotate("", xy=(90, 55), xytext=(85, 41), arrowprops=dict(arrowstyle="->", lw=1.5, color="#16a34a"))

    ax.add_patch(patches.Rectangle((90, 20), 9, 14, ec="#dc2626", fc="#fee2e2", lw=1.5))
    ax.text(94.5, 27, "Resolver\nQueue", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.annotate("", xy=(90, 27), xytext=(85, 33), arrowprops=dict(arrowstyle="->", lw=1.5, color="#dc2626"))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_dfd_diagram.png", dpi=300)
    plt.close(fig)

def generate_class_diagram():
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.text(50, 96, "UML Class Diagram — Smart Ticket Understanding Architecture", ha="center", fontsize=13, fontweight="bold")

    # Class 1: TriageEngine
    ax.add_patch(patches.Rectangle((35, 62), 30, 28, ec="#1e293b", fc="#f8fafc", lw=1.5))
    ax.text(50, 86, "TriageEngine", ha="center", fontweight="bold", fontsize=10)
    ax.plot([35, 65], [83, 83], color="#94a3b8")
    ax.text(36, 75, "+ threshold: float\n+ k: int = 5\n+ cascade: Cascade\n+ retriever: KnnRetriever\n+ agent: EscalationAgent", fontsize=8)
    ax.plot([35, 65], [68, 68], color="#94a3b8")
    ax.text(36, 64, "+ triage(text, signals): TriageResult\n+ triage_batch(texts): list", fontsize=8)

    # Class 2: Cascade
    ax.add_patch(patches.Rectangle((5, 20), 26, 32, ec="#1e293b", fc="#f8fafc", lw=1.5))
    ax.text(18, 48, "Cascade", ha="center", fontweight="bold", fontsize=10)
    ax.plot([5, 31], [45, 45], color="#94a3b8")
    ax.text(6, 36, "+ cfg: CascadeConfig\n+ vectorizers: dict\n+ _svc: LinearSVC\n+ calibrator: LogReg\n+ n_features: int = 34901", fontsize=8)
    ax.plot([5, 31], [28, 28], color="#94a3b8")
    ax.text(6, 23, "+ fused_probs(texts): array\n+ predict(texts): list", fontsize=8)

    # Class 3: KnnRetriever
    ax.add_patch(patches.Rectangle((37, 10), 26, 32, ec="#1e293b", fc="#f8fafc", lw=1.5))
    ax.text(50, 38, "KnnRetriever", ha="center", fontweight="bold", fontsize=10)
    ax.plot([37, 63], [35, 35], color="#94a3b8")
    ax.text(38, 26, "+ texts: list[str]\n+ labels: ndarray\n+ union: csr_matrix", fontsize=8)
    ax.plot([37, 63], [20, 20], color="#94a3b8")
    ax.text(38, 14, "+ query(text, k): list\n+ vote(text, k): probs, nbrs", fontsize=8)

    # Class 4: EscalationAgent
    ax.add_patch(patches.Rectangle((69, 20), 28, 32, ec="#1e293b", fc="#f8fafc", lw=1.5))
    ax.text(83, 48, "EscalationAgent", ha="center", fontweight="bold", fontsize=10)
    ax.plot([69, 97], [45, 45], color="#94a3b8")
    ax.text(70, 36, "+ enable_llm: bool\n+ model_name: str\n+ kb: dict", fontsize=8)
    ax.plot([69, 97], [28, 28], color="#94a3b8")
    ax.text(70, 23, "+ handle(text, nbrs): dec, draft\n+ verify(draft, nbrs): dict", fontsize=8)

    # Associations
    ax.plot([35, 18], [70, 52], color="#0284c7", lw=1.5)
    ax.plot([50, 50], [62, 42], color="#059669", lw=1.5)
    ax.plot([65, 83], [70, 52], color="#7c3aed", lw=1.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_class_diagram.png", dpi=300)
    plt.close(fig)

def generate_benchmarks():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    systems = ["kNN Retrieval", "Cascade (Ours)", "DistilBERT"]
    acc = [92.56, 94.60, 94.91]
    f1 = [91.75, 94.43, 94.07]
    latency = [27.4, 33.3, 105.5]

    x = np.arange(len(systems))
    width = 0.35

    ax1.bar(x - width/2, acc, width, label="Accuracy (%)", color="#059669")
    ax1.bar(x + width/2, f1, width, label="Macro-F1 (%)", color="#10b981")
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Classification Accuracy vs Macro-F1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems)
    ax1.set_ylim(85, 100)
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    bars2 = ax2.bar(systems, latency, color=["#94a3b8", "#10b981", "#ef4444"], width=0.5)
    ax2.set_ylabel("Latency per Ticket (ms) on CPU")
    ax2.set_title("Serving Latency Comparison (Lower is Better)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval} ms", ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_empirical_comparison.png", dpi=300)
    plt.close(fig)

def generate_conformal_curve():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    alphas = [0.01, 0.05, 0.10, 0.15, 0.20]
    coverage = [98.8, 95.4, 90.3, 85.1, 78.9]
    acc = [96.8, 98.6, 100.0, 100.0, 100.0]

    ax.plot(alphas, coverage, marker="s", lw=2, color="#2563eb", label="Coverage (% Auto-Answered)")
    ax.plot(alphas, acc, marker="o", lw=2, color="#059669", label="Accuracy on Accepted Slice (%)")
    ax.set_xlabel("Significance Level (alpha)")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Split-Conformal Guarantee Curve (Calibrated on Val Split)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")

    for i in range(len(alphas)):
        ax.annotate(f"{coverage[i]}%", (alphas[i], coverage[i]-3), ha="center", fontsize=8.5, color="#1e40af", fontweight="bold")
        ax.annotate(f"{acc[i]}%", (alphas[i], acc[i]+1.5), ha="center", fontsize=8.5, color="#065f46", fontweight="bold")

    ax.set_ylim(70, 105)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_conformal_curve.png", dpi=300)
    plt.close(fig)

def generate_roi():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    volumes = [500, 1000, 2500, 5000, 10000]
    hours_saved = [v * 30 * 0.903 * 8 / 60 for v in volumes]
    cost_saved = [h * 28 for h in hours_saved]  # at $28/hr

    ax.plot(volumes, hours_saved, marker="o", lw=2.5, color="#0284c7", label="Agent Hours Saved / Month")
    ax.set_xlabel("Daily Ticket Volume")
    ax.set_ylabel("Hours Saved per Month")
    ax.set_title("Operational Capacity Freed via Split-Conformal Deflection")
    ax.grid(True, linestyle="--", alpha=0.5)

    for i, txt in enumerate(hours_saved):
        ax.annotate(f"{txt:,.0f}h\n(${cost_saved[i]:,.0f})", (volumes[i], hours_saved[i]+80), ha="center", fontsize=8.5, fontweight="semibold")

    ax.set_ylim(0, max(hours_saved) * 1.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig8_roi_simulator.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    generate_architecture()
    generate_use_case()
    generate_dfd()
    generate_class_diagram()
    generate_benchmarks()
    generate_conformal_curve()
    generate_roi()
    print("All diagrams generated in docs/figures!")
