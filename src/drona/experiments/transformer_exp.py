"""Experiment 04 -- the synopsis's Model 3: fine-tuned DistilBERT (+ ONNX INT8).

This exists to answer one question honestly: *does the transformer the synopsis
proposes actually beat what we already have?* Experiment 01 showed TF-IDF +
LinearSVC reaches 94.50% macro-F1 on the coarse task in 0.01 s, and experiment
03 measures a fused retrieval+linear cascade. If DistilBERT cannot beat that
while costing 8-12 CPU-hours to train and needing a distillation step to serve,
the architecture choice is settled by measurement rather than by fashion.

Three measured efficiency decisions, all forced by the local CPU benchmark
(`artifacts/reports/bench.log`):

1. ``max_length=64`` instead of the synopsis's 128. Tokenisation of the training
   split gives mean 15.9 / p95 35 / p99 51 tokens, so 64 covers 99.8% of
   tickets. Padding everything to 128 would waste roughly 4x the compute.
2. **Dynamic padding** -- pad each batch to its own longest member instead of
   ``padding="max_length"``.
3. **Length-grouped batches** -- sort by length inside shuffled megabatches so
   padding is close to minimal without sacrificing sampling randomness.

Only one configuration is trained. Multiple transformer sweeps are not
affordable here, and the point of this experiment is the comparison, not a
hyper-parameter hunt.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import MODELS, REPORTS
from ..data import load_jsonl
from ..evaluate import report, score

SEED = 20261003
MODEL = "distilbert-base-uncased"
MAX_LEN = 64
BATCH = 32
EPOCHS = 2
LR = 2e-5
HEAD_LR = 1e-3
WARMUP = 0.1


class TicketDataset(Dataset):
    def __init__(self, tickets, tokenizer, labels: dict[str, int], max_len: int):
        self.enc = tokenizer(
            [t.text for t in tickets],
            truncation=True,
            max_length=max_len,
            padding=False,  # dynamic: collator pads per batch
        )
        self.y = [labels[t.coarse] for t in tickets]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, self.y[i]


def collate(batch, pad_id: int):
    """Pad to the longest sequence in this batch only."""
    feats = [b[0] for b in batch]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    width = max(len(f["input_ids"]) for f in feats)
    input_ids = torch.full((len(feats), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(feats), width), dtype=torch.long)
    for i, f in enumerate(feats):
        n = len(f["input_ids"])
        input_ids[i, :n] = torch.tensor(f["input_ids"], dtype=torch.long)
        attn[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attn}, ys


def length_grouped_batches(lengths, batch_size, generator):
    """Shuffle, then sort within large windows so batches are length-homogeneous."""
    window = batch_size * 32
    order = torch.randperm(len(lengths), generator=generator).tolist()
    batches = []
    for i in range(0, len(order), window):
        chunk = order[i : i + window]
        chunk.sort(key=lambda j: lengths[j])
        batches += [chunk[j : j + batch_size] for j in range(0, len(chunk), batch_size)]
    rng = torch.Generator()
    rng.manual_seed(int(torch.rand(1, generator=generator).item() * 1e9))
    perm = torch.randperm(len(batches), generator=rng).tolist()
    return [batches[p] for p in perm]


@torch.no_grad()
def evaluate(model, loader, y_true):
    model.eval()
    preds = []
    for batch, _ in loader:
        preds.append(model(**batch).logits.argmax(-1))
    model.train()
    preds = torch.cat(preds).numpy()
    return score(y_true, preds)


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(2)  # 2 physical cores; measured throughput ~8 GFLOP/s
    gen = torch.Generator().manual_seed(SEED)

    train, val, test = load_jsonl("train"), load_jsonl("val"), load_jsonl("test")
    names = sorted({t.coarse for t in train})
    label2id = {n: i for i, n in enumerate(names)}
    print(f"{len(names)} classes: {names}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    ds_tr = TicketDataset(train, tok, label2id, MAX_LEN)
    ds_va = TicketDataset(val, tok, label2id, MAX_LEN)
    lengths = [len(e) for e in ds_tr.enc["input_ids"]]

    train_batches = length_grouped_batches(lengths, BATCH, gen)
    dl_tr = DataLoader(
        ds_tr,
        batch_sampler=train_batches,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        num_workers=0,
    )
    dl_va = DataLoader(
        ds_va,
        batch_sampler=[list(range(i, min(i + BATCH, len(ds_va)))) for i in range(0, len(ds_va), BATCH)],
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(names))
    head_keys = ("pre_classifier", "classifier")
    no_decay_keys = ("bias", "LayerNorm")
    # The two groups must partition the parameters. Selecting the head first and
    # then excluding it from the backbone group avoids the overlap that makes
    # AdamW raise "some parameters appear in more than one parameter group".
    head = [p for n, p in model.named_parameters() if any(k in n for k in head_keys)]
    decay = [
        p
        for n, p in model.named_parameters()
        if not any(k in n for k in head_keys) and not any(k in n for k in no_decay_keys)
    ]
    no_decay = [
        p
        for n, p in model.named_parameters()
        if not any(k in n for k in head_keys) and any(k in n for k in no_decay_keys)
    ]
    seen = {id(p) for p in head + decay + no_decay}
    assert len(seen) == len(list(model.parameters())), (
        f"param groups do not partition the model: {len(seen)} of {len(list(model.parameters()))}"
    )
    opt = torch.optim.AdamW(
        [
            {"params": decay, "lr": LR, "weight_decay": 0.01},
            {"params": no_decay, "lr": LR, "weight_decay": 0.0},
            {"params": head, "lr": HEAD_LR, "weight_decay": 0.0},
        ],
        weight_decay=0.01,
    )
    total = len(train_batches) * EPOCHS
    warm = int(total * WARMUP)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: s / max(warm, 1) if s < warm else max(0.0, (total - s) / max(total - warm, 1)),
    )

    y_va = np.array([label2id[t.coarse] for t in val])
    history = []
    t_start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        running, seen = 0.0, 0
        for step, (batch, y) in enumerate(dl_tr):
            opt.zero_grad()
            loss = model(**batch, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * len(y)
            seen += len(y)
            if step % 25 == 0:
                rate = (time.time() - t_start) / max(seen, 1)
                left = rate * len(dl_tr.dataset) * (EPOCHS - epoch - 1)
                print(
                    f"ep{epoch} step {step}/{len(train_batches)} loss={running/seen:.4f} "
                    f"{rate*1000:.0f} ms/ex  eta {left/60:.0f} min",
                    flush=True,
                )
        m = evaluate(model, dl_va, y_va)
        print(f"== epoch {epoch} val acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}", flush=True)
        history.append({"epoch": epoch, "val": m, "minutes": (time.time() - t_start) / 60})

    out = MODELS / "distilbert-coarse"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)

    dl_te = DataLoader(
        TicketDataset(test, tok, label2id, MAX_LEN),
        batch_size=BATCH,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )
    model.eval()
    preds = torch.cat([model(**b).logits.argmax(-1) for b, _ in dl_te]).numpy()
    y_te = np.array([label2id[t.coarse] for t in test])
    m_te = score(y_te, preds)
    print(f"\nTEST acc={m_te['accuracy']:.4f} macroF1={m_te['macro_f1']:.4f}", flush=True)
    report([names[i] for i in y_te], [names[i] for i in preds], names, "DistilBERT coarse11 test")

    (REPORTS / "04-distilbert.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "max_len": MAX_LEN,
                "batch": BATCH,
                "epochs": EPOCHS,
                "lr": LR,
                "head_lr": HEAD_LR,
                "train_minutes": (time.time() - t_start) / 60,
                "history": history,
                "test": m_te,
            },
            indent=2,
        )
    )
    print("DISTILBERT_DONE", flush=True)


if __name__ == "__main__":
    main()