from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PokerChunkDataset, collate_batch, load_public_benchmark
from .features import FeatureVectorizer
from .model import ChunkTransformerClassifier
from .tokenizer import EventTokenizer


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    probs = []
    labels = []
    losses = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            y = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, features)
            loss = criterion(logits, y)
            p = torch.sigmoid(logits).detach().cpu().numpy()

            probs.extend(p.tolist())
            labels.extend(y.detach().cpu().numpy().tolist())
            losses.append(float(loss.item()))

    labels_arr = np.asarray(labels, dtype=np.float32)
    probs_arr = np.asarray(probs, dtype=np.float32).clip(1e-6, 1 - 1e-6)
    preds = (probs_arr >= 0.5).astype(np.int32)

    out = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(labels_arr, preds)) if len(labels_arr) else 0.0,
        "log_loss": float(log_loss(labels_arr, probs_arr, labels=[0, 1])) if len(set(labels)) > 1 else 0.0,
        "mean_prob": float(probs_arr.mean()) if len(probs_arr) else 0.0,
    }
    if len(set(labels)) > 1:
        out["roc_auc"] = float(roc_auc_score(labels_arr, probs_arr))
        out["pr_auc"] = float(average_precision_score(labels_arr, probs_arr))
    else:
        out["roc_auc"] = 0.0
        out["pr_auc"] = 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to public_miner_benchmark.json.gz")
    parser.add_argument("--out", default="artifacts/p44_chunk_detector.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_samples, val_samples = load_public_benchmark(args.data, seed=args.seed)
    if not train_samples or not val_samples:
        raise RuntimeError("Dataset loading failed: train or validation split is empty.")
    print(f"Train samples: {len(train_samples)} | Validation samples: {len(val_samples)}")

    train_chunks = [s.chunk for s in train_samples]
    tokenizer = EventTokenizer(max_len=args.max_len).fit(train_chunks, min_freq=1, max_vocab=8000)
    vectorizer = FeatureVectorizer().fit(train_chunks)

    train_ds = PokerChunkDataset(train_samples, tokenizer, vectorizer)
    val_ds = PokerChunkDataset(val_samples, tokenizer, vectorizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    model = ChunkTransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        feature_dim=len(vectorizer.feature_names),
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    # Handles class imbalance if present.
    labels = np.asarray([s.label for s in train_samples], dtype=np.float32)
    pos = labels.sum()
    neg = len(labels) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask, features)
            print(logits)
            print(y)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running.append(float(loss.item()))

        metrics = evaluate(model, val_loader, device)
        metrics["train_loss"] = float(np.mean(running)) if running else 0.0
        metrics["epoch"] = epoch
        history.append(metrics)
        print(json.dumps(metrics, indent=2))

        # PR-AUC is useful when bot/human imbalance exists. ROC-AUC is fallback.
        selection_metric = metrics.get("pr_auc", 0.0) + metrics.get("roc_auc", 0.0)
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
        "tokenizer": tokenizer.state_dict(),
        "vectorizer": vectorizer.state_dict(),
        "threshold": 0.5,
        "history": history,
        "training_data": str(args.data),
    }
    torch.save(artifact, out_path)
    print(f"Saved model artifact to: {out_path}")


if __name__ == "__main__":
    main()
