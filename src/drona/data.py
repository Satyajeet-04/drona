"""Dataset construction for Banking77 -> coarse enterprise taxonomy."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from . import taxonomy as tx
from .config import PROCESSED, RAW, SEED, TEST_FRAC, VAL_FRAC


@dataclass(frozen=True)
class Ticket:
    text: str
    intent: str
    coarse: str


def _read_csv(path):
    """Banking77 ships as a CSV whose text field contains commas."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("text") or "").strip()
            intent = (row.get("category") or "").strip()
            if text and intent:
                out.append((text, intent))
    return out


def load_banking77():
    """Return (train, test) as lists of Ticket using the canonical splits."""
    intents = json.load(open(RAW / "categories.json"))
    tx.validate(intents)  # fail loudly if the taxonomy drifts from the data

    train = [Ticket(t, i, tx.coarse_of(i)) for t, i in _read_csv(RAW / "train.csv")]
    test = [Ticket(t, i, tx.coarse_of(i)) for t, i in _read_csv(RAW / "test.csv")]
    return train, test


def resplit(train, test, val_frac=VAL_FRAC, test_frac=TEST_FRAC):
    """
    Combine canonical train+test and re-split 70/15/15 stratified by coarse
    category, per the synopsis. Stratifying on the *coarse* label (not the fine
    one) is what the synopsis actually specifies, and it guarantees every
    priority-bearing category appears in every split.
    """
    allt = list(train) + list(test)
    idx = list(range(len(allt)))
    tr_idx, rest = train_test_split(
        idx, test_size=val_frac + test_frac, random_state=SEED,
        stratify=[t.coarse for t in allt],
    )
    rel_val = test_frac / (val_frac + test_frac)
    va_idx, te_idx = train_test_split(
        rest, test_size=rel_val, random_state=SEED,
        stratify=[allt[i].coarse for i in rest],
    )
    return (
        [allt[i] for i in tr_idx],
        [allt[i] for i in va_idx],
        [allt[i] for i in te_idx],
    )


def corpus(tickets):
    return [t.text for t in tickets]


def fine_labels(tickets, vocab=None):
    if vocab is None:
        vocab = {i: n for n, i in enumerate(sorted({t.intent for t in tickets}))}
    return [vocab[t.intent] for t in tickets]


def coarse_labels(tickets):
    return [tx.COARSE_ID[t.coarse] for t in tickets]


def intent_vocab(tickets):
    return {i: n for n, i in enumerate(sorted({t.intent for t in tickets}))}


def dump(tickets, name):
    """Persist a split as JSONL so later stages do not re-parse CSVs."""
    path = PROCESSED / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for t in tickets:
            fh.write(json.dumps({"text": t.text, "intent": t.intent, "coarse": t.coarse}) + "\n")
    return path


def load_jsonl(name):
    import csv as _csv
    path = PROCESSED / f"{name}.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return [Ticket(**json.loads(l)) for l in fh if l.strip()]
    return []