"""Model 1 of the synopsis: TF-IDF (word 1-2 grams) + Logistic Regression.

Run as a module:  PYTHONPATH=src python -m drona.experiments.baselines
This is the number every other model must beat, and the number the synopsis
predicts will be beaten by DistilBERT.
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from .. import data, evaluate
from .. import taxonomy as tx
from ..config import REPORTS, SEED

CV = dict(C=1.0, max_iter=2000, random_state=SEED)


def build_vectorizer(word_ngram=(1, 2)):
    """Word n-grams per the synopsis, plus char n-grams for typo robustness.

    Tickets are noisy free text with typos, so the char view (3-5 grams) is
    what catches "pendng", "trasfer", "chrage".
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=word_ngram,
                                 min_df=2, max_features=50000, sublinear_tf=True,
                                 strip_accents="unicode", lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, max_features=50000, sublinear_tf=True,
                                 lowercase=True)),
    ])


def run():
    train, _, test = (data.load_jsonl(n) for n in ("train", "val", "test"))
    assert train and test, "run data.dump first"
    iv = data.intent_vocab(train + test)
    inames = [None] * len(iv)
    for k, v in iv.items():
        inames[v] = k
    cnames = tx.COARSE_CATEGORIES

    Xtr, Xte = data.corpus(train), data.corpus(test)
    ytr_f = data.fine_labels(train, iv)
    yte_f = data.fine_labels(test, iv)
    ytr_c = data.coarse_labels(train)
    yte_c = data.coarse_labels(test)

    results = {}

    for clf_name, clf in [
        ("logreg", LogisticRegression(**CV)),
        ("linsvc", LinearSVC(C=1.0, random_state=SEED, max_iter=5000)),
    ]:
        for task, ytr, yte, names in [
            ("coarse11", ytr_c, yte_c, cnames),
            ("fine77", ytr_f, yte_f, inames),
        ]:
            vec = build_vectorizer()
            t0 = time.time()
            A = vec.fit_transform(Xtr)
            B = vec.transform(Xte)
            t_feat = time.time() - t0
            clf.fit(A, ytr)
            t0 = time.time()
            pred = clf.predict(B)
            t_pred = time.time() - t0
            m = evaluate.report(yte, pred, names, f"TF-IDF+{clf_name} -> {task}")
            m["n_features"] = int(A.shape[1])
            m["feature_fit_s"] = round(t_feat, 2)
            m["predict_1963_s"] = round(t_pred, 2)
            results[f"{clf_name}_{task}"] = m

    out = REPORTS / "01-tfidf-baseline.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return results


if __name__ == "__main__":
    run()