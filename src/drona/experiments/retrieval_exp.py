"""Experiment 02 -- retrieval-augmented classification (kNN / case-based reasoning).

Why this matters for the capstone
---------------------------------
The synopsis uses a fine-tuned DistilBERT and justifies it by claiming it beats
TF-IDF. Experiment 01 already showed a TF-IDF+LinearSVC linear model reaches
94.50% macro-F1 on the coarse task in 0.01 s, so a transformer has to earn its
place. This experiment tests whether *retrieval* -- not extra model capacity --
is what actually moves the number, per Ovadia et al. (EMNLP 2024) and
arXiv:2403.01432.

Outputs (all written to artifacts/reports/):
  02-retrieval.json        -- headline numbers per task and per view
  02-retrieval-oracle.json -- "is the right answer in the retrieved set at all?"
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..config import ARTIFACTS, REPORTS, SEED
from ..data import load_jsonl
from ..evaluate import confusion, per_class, report, score
from ..retrieval import KnnRetriever

TASKS = {"coarse11": "coarse", "fine77": "intent"}


def _attr(tickets, name):
    return np.array([getattr(t, name) for t in tickets])


def main() -> None:
    rng = np.random.default_rng(SEED)
    train, val, test = load_jsonl("train"), load_jsonl("val"), load_jsonl("test")
    train_texts = [t.text for t in train]
    test_texts = [t.text for t in test]

    results: dict[str, dict] = {"seed": SEED, "n_train": len(train), "n_test": len(test)}
    oracle: dict[str, dict] = {}

    for task, attr in TASKS.items():
        y_tr = _attr(train, attr)
        y_te = _attr(test, attr)
        retriever = KnnRetriever(train_texts, y_tr)
        classes = np.unique(y_tr)
        idx = {c: i for i, c in enumerate(classes)}

        results[task] = {}
        oracle[task] = {}

        for view in ("char", "word", "union"):
            t0 = time.time()
            nbrs = retriever.query(test_texts, k=15, view=view)
            latency = (time.time() - t0) / len(test_texts) * 1000

            probs = np.zeros((len(test_texts), len(classes)))
            for i, nb in enumerate(nbrs):
                w = np.clip(nb.scores, 0, None) ** 3
                for lab, wi in zip(nb.labels, w):
                    probs[i, idx[lab]] += wi
            s = probs.sum(axis=1, keepdims=True)
            probs = np.divide(probs, s, out=np.full_like(probs, 1 / len(classes)), where=s > 0)
            pred = classes[probs.argmax(axis=1)]
            conf = probs.max(axis=1)

            m = score(y_te, pred)
            results[task][view] = {
                "metrics": m,
                "latency_ms_per_ticket": round(latency, 3),
                "mean_top1_similarity": float(np.mean([n.top_score for n in nbrs])),
            }
            print(
                f"{task:9} {view:6} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
                f"{latency:6.2f} ms/ticket",
                flush=True,
            )

            # Oracle: is the gold label anywhere in the top-k? This is the
            # ceiling any reranker or LLM judge could reach on this retrieval
            # set -- it separates "retrieval found it" from "we scored it wrong".
            for k in (1, 3, 5, 10, 15):
                hit = 0
                for i, nb in enumerate(nbrs):
                    if y_te[i] in set(nb.labels[:k].tolist()):
                        hit += 1
                oracle[task].setdefault(view, {})[f"recall_at_{k}"] = round(hit / len(test_texts), 4)

        # Per-class breakdown for the best view, to expose which categories the
        # retriever fails on -- the synopsis reports only aggregate scores.
        probs, _ = retriever.vote(test_texts, k=15, view="union")
        pred = classes[probs.argmax(axis=1)]
        results[task]["per_class_union"] = per_class(y_te, pred, classes.tolist())
        cm, cm_names = confusion(y_te, pred, classes.tolist())
        results[task]["confusion_union"] = {"labels": cm_names, "matrix": cm.tolist()}

    (REPORTS / "02-retrieval.json").write_text(json.dumps(results, indent=2))
    (REPORTS / "02-retrieval-oracle.json").write_text(json.dumps(oracle, indent=2))

    print("\nOracle recall (union view):")
    for task in TASKS:
        print(f"  {task}: " + "  ".join(f"R@{k}={v:.4f}" for k, v in oracle[task]["union"].items()))
    print("BENCH_DONE")


if __name__ == "__main__":
    main()