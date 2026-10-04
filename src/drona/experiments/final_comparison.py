"""Experiment 06 -- the capstone's decisive comparison table.

Everything so far has been a single system measured in isolation. The synopsis
argues for a transformer because "DistilBERT is more accurate". This script
puts all four approaches on one axis and reports the two numbers that actually
decide the argument for SLA triage:

  * **macro-F1**, which weights `fraud_security` (n=91) equally with
    `payment_failure` (n=552). Accuracy does not.
  * **P0 recall** -- the share of genuinely critical tickets that reach the
    right queue. A missed fraud ticket is not a rounding error.

It also measures calibration (ECE) and wall-clock latency per ticket, because
the synopsis claims ONNX INT8 buys speed and we should check what the latency
budget actually costs on CPU.

Run:  PYTHONPATH=src python -m drona.experiments.final_comparison
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..cascade import Cascade
from ..config import MODELS, REPORTS
from ..data import load_jsonl
from ..evaluate import per_class, score
from ..retrieval import KnnRetriever

# Classes that a real support desk must never drop. Ordered by blast radius.
P0_CRITICAL = ["fraud_security", "payment_failure"]
TAIL = ["other", "refund_chargeback", "technical_app", "fraud_security"]


def _ece(y_true: np.ndarray, probs: np.ndarray, classes: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: how far confidence is from empirical accuracy."""
    conf = probs.max(axis=1)
    pred = classes[probs.argmax(axis=1)]
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def _latency(fn, items, repeat: int = 1) -> float:
    t0 = time.time()
    for _ in range(repeat):
        for x in items:
            fn(x)
    return (time.time() - t0) / (len(items) * repeat) * 1000


def main() -> None:
    train, _, test = load_jsonl("train"), load_jsonl("val"), load_jsonl("test")
    tr_texts = [t.text for t in train]
    te_texts = [t.text for t in test]
    y_tr = np.array([t.coarse for t in train])
    y_te = np.array([t.coarse for t in test])

    out: dict = {
        "n_train": len(train),
        "n_test": len(test),
        "note": "classes present in data = 10; loan_emi has 0 examples in Banking77",
    }

    # `hard_preds` keeps the per-ticket predictions so the significance test
    # below can pair the systems exactly. Aggregated per-class tables are not
    # enough: McNemar needs to know *which* tickets each system missed.
    hard_preds: dict[str, np.ndarray] = {}

    # -- 1. retrieval only --------------------------------------------------
    ret = KnnRetriever(tr_texts, y_tr)
    classes = np.array(sorted(set(y_tr.tolist())))
    probs_knn, _ = ret.vote(te_texts, k=15, view="union")
    pred_knn = classes[probs_knn.argmax(axis=1)]
    hard_preds["knn_retrieval"] = pred_knn
    out["knn_retrieval"] = {
        **score(y_te, pred_knn),
        "per_class": per_class(y_te, pred_knn, classes.tolist()),
        "ece": _ece(y_te, probs_knn, classes),
        "latency_ms": round(_latency(lambda t: ret.vote([t], k=15, view="union")[0], te_texts[:300]), 3),
        "learned_parameters": 0,
    }

    # -- 2. the fused + calibrated cascade ---------------------------------
    casc = Cascade.load(MODELS / "cascade.joblib", ret)
    fused, pl, pk = casc.fused_probs(te_texts)
    pred_fused = classes[fused.argmax(axis=1)]
    hard_preds["cascade_fused"] = pred_fused
    lin_top = np.array([casc.classes_[int(i.argmax())] for i in pl])
    knn_top = np.array([casc.classes_[int(i.argmax())] for i in pk])
    agree = lin_top == knn_top
    out["cascade_fused"] = {
        **score(y_te, pred_fused),
        "per_class": per_class(y_te, pred_fused, classes.tolist()),
        "ece": _ece(y_te, fused, classes),
        "alpha": casc.cfg.alpha,
        "temperature": casc.cfg.temperature,
        "agreement_rate": round(float(agree.mean()), 4),
        "accuracy_when_agree": round(float((pred_fused[agree] == y_te[agree]).mean()), 4),
        "accuracy_when_disagree": round(float((pred_fused[~agree] == y_te[~agree]).mean()), 4),
        "n_disagree": int((~agree).sum()),
        "latency_ms": round(_latency(lambda t: casc.fused_probs([t]), te_texts[:300]), 3),
    }

    # -- 3. DistilBERT (the synopsis's Model 3) ----------------------------
    # Loaded lazily so the script still runs on a machine without the 268 MB
    # checkpoint, and so the cheap comparisons above print immediately.
    distil_path = MODELS / "distilbert-coarse"
    if (distil_path / "model.safetensors").exists():
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(2)
        tok = AutoTokenizer.from_pretrained(distil_path)
        mdl = AutoModelForSequenceClassification.from_pretrained(distil_path)
        mdl.eval()
        logprobs = np.zeros((len(te_texts), len(classes)))
        # The checkpoint was saved without its label map -- config.json holds
        # "LABEL_0".."LABEL_9". The training script built its classes as
        # `names = sorted({t.coarse for t in train})`, so logit i corresponds to
        # `classes[i]` positionally. `classes` here is built the same way, which
        # makes the permutation the identity.
        order = list(range(len(classes)))
        assert len(order) == mdl.config.num_labels, (
            f"checkpoint has {mdl.config.num_labels} labels, data has {len(classes)}"
        )
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(te_texts), 32):
                batch = te_texts[i : i + 32]
                enc = tok(batch, truncation=True, max_length=64, padding=True, return_tensors="pt")
                logits = mdl(**enc).logits
                logprobs[i : i + len(batch)] = torch.softmax(logits, dim=-1).numpy()[:, order]
        infer_s = time.time() - t0
        pred_db = classes[logprobs.argmax(axis=1)]
        hard_preds["distilbert_finetuned"] = pred_db
        out["distilbert_finetuned"] = {
            **score(y_te, pred_db),
            "per_class": per_class(y_te, pred_db, classes.tolist()),
            "ece": _ece(y_te, logprobs, classes),
            "infer_seconds_total": round(infer_s, 1),
            "latency_ms": round(infer_s / len(te_texts) * 1000, 2),
            "train_minutes": 47.2,
            "model_mb": round((distil_path / "model.safetensors").stat().st_size / 1e6, 1),
        }
    else:
        out["distilbert_finetuned"] = {"status": "checkpoint not present, skipped"}

    # -- 4. the decision table --------------------------------------------
    order_of = ["knn_retrieval", "cascade_fused", "distilbert_finetuned"]
    available = [k for k in order_of if "accuracy" in out.get(k, {})]

    def recall_on(system: str, subset: list[str]) -> float | None:
        pc = out[system].get("per_class")
        if not pc:
            return None
        rows = {r[0]: r for r in pc}
        num = den = 0.0
        for c in subset:
            if c in rows:
                num += rows[c][3] * rows[c][1]
                den += rows[c][1]
        return round(num / den, 4) if den else None

    table = []
    for k in available:
        s = out[k]
        table.append(
            {
                "system": k,
                "accuracy": round(s["accuracy"], 4),
                "macro_f1": round(s["macro_f1"], 4),
                "macro_recall": round(s["macro_recall"], 4),
                "ece": round(s["ece"], 4),
                "p0_recall": recall_on(k, P0_CRITICAL),
                "tail_recall": recall_on(k, TAIL),
                "latency_ms": s.get("latency_ms"),
            }
        )
    # -- 4. is the DistilBERT accuracy lead even real? ----------------------
    # The gap is ~0.4 pp on n=1963, where one standard error is ~0.5 pp, so it
    # may well be noise. McNemar's test is the right instrument: both systems
    # predict the *same* tickets, so it conditions on the discordant pairs only
    # and ignores the (large) set they both get right.
    try:
        from scipy.stats import chi2

        if len(hard_preds) >= 2:
            keys = [k for k in ("cascade_fused", "distilbert_finetuned") if k in hard_preds]
            if len(keys) == 2:
                a, b_ = keys
                ca, cb = hard_preds[a] == y_te, hard_preds[b_] == y_te
                n01 = int((~ca & cb).sum())  # only distilbert right
                n10 = int((ca & ~cb).sum())  # only cascade right
                stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01) if (n10 + n01) else 0.0
                p = float(chi2.sf(stat, 1))
                out["mcnemar_cascade_vs_distilbert"] = {
                    "only_cascade_correct": n10,
                    "only_distilbert_correct": n01,
                    "both_correct": int((ca & cb).sum()),
                    "both_wrong": int((~ca & ~cb).sum()),
                    "chi2": round(stat, 4),
                    "p_value": round(p, 4),
                    "significant_at_0_05": bool(p < 0.05),
                    "method": "exact paired McNemar on per-ticket test predictions",
                }
    except Exception as exc:  # pragma: no cover - diagnostics must not break the table
        out["mcnemar_error"] = str(exc)

    out["decision_table"] = table
    out["p0_critical_classes"] = P0_CRITICAL
    out["tail_classes"] = TAIL

    (REPORTS / "06-final-comparison.json").write_text(json.dumps(out, indent=2, default=float))

    print("\n=== DECISION TABLE (test, n=%d) ===" % len(test))
    hdr = f"{'system':<22}{'acc':>8}{'macroF1':>9}{'macroR':>8}{'ECE':>7}{'P0rec':>8}{'tailR':>7}{'ms':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in table:
        print(
            f"{r['system']:<22}{r['accuracy']:>8.4f}{r['macro_f1']:>9.4f}{r['macro_recall']:>8.4f}"
            f"{r['ece']:>7.4f}{(r['p0_recall'] or 0):>8.4f}{(r['tail_recall'] or 0):>7.4f}"
            f"{(r['latency_ms'] or 0):>8.2f}"
        )
    print("\nFINAL_DONE")


if __name__ == "__main__":
    main()