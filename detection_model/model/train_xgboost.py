from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier

from .dataset import augment_chunk_windows, load_public_benchmark
from .features import FeatureVectorizer
from .hierarchical_dataset import (
    HierarchicalPokerChunkDataset,
    hierarchical_collate_batch,
)
from .hierarchical_model import HierarchicalChunkClassifier
from .hierarchical_tokenizer import HandActionTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost on hierarchical Poker44 chunk embeddings."
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--torch-model", required=True)
    parser.add_argument("--out", default="artifacts/p44_xgb_detector.joblib")

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--augment-windows", action="store_true")
    parser.add_argument("--augment-validation-windows", action="store_true")
    parser.add_argument("--window-hands", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--keep-short-window-chunks", action="store_true")

    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)

    return parser.parse_args()


def load_torch_artifact(
    path: str | Path,
    device: torch.device,
) -> Tuple[HierarchicalChunkClassifier, HandActionTokenizer, FeatureVectorizer, Dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Torch model artifact not found: {path}")

    artifact = torch.load(path, map_location=device)

    tokenizer = HandActionTokenizer.from_state_dict(artifact["tokenizer"])
    vectorizer = FeatureVectorizer.from_state_dict(artifact["vectorizer"])

    model_config = artifact["model_config"]

    model = HierarchicalChunkClassifier(**model_config)
    model.load_state_dict(artifact["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, vectorizer, artifact


@torch.no_grad()
def extract_xgb_matrix(
    model: HierarchicalChunkClassifier,
    tokenizer: HandActionTokenizer,
    vectorizer: FeatureVectorizer,
    samples,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = HierarchicalPokerChunkDataset(
        samples=samples,
        tokenizer=tokenizer,
        vectorizer=vectorizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda b: hierarchical_collate_batch(
            b,
            pad_id=tokenizer.pad_id,
        ),
    )

    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for batch in tqdm(loader, desc="extract embeddings"):
        action_ids = batch["action_ids"].to(device)
        action_mask = batch["action_mask"].to(device)
        hand_mask = batch["hand_mask"].to(device)
        features = batch["features"].to(device)
        labels = batch["labels"].detach().cpu().numpy().astype(np.int32)

        chunk_embedding = model.extract_chunk_embedding(
            action_ids=action_ids,
            action_mask=action_mask,
            hand_mask=hand_mask,
        )

        emb_np = chunk_embedding.detach().cpu().numpy().astype(np.float32)
        feat_np = features.detach().cpu().numpy().astype(np.float32)

        # Final XGBoost input:
        # neural chunk embedding + engineered chunk features
        x = np.concatenate([emb_np, feat_np], axis=1)

        all_x.append(x)
        all_y.append(labels)

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)

    return X, y


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    preds = (scores >= threshold).astype(np.int32)

    metrics: Dict[str, Any] = {
        "count": int(len(y_true)),
        "threshold": float(threshold),
        "human_count": int((y_true == 0).sum()),
        "bot_count": int((y_true == 1).sum()),
        "score_min": float(scores.min()) if len(scores) else 0.0,
        "score_max": float(scores.max()) if len(scores) else 0.0,
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "accuracy": float(accuracy_score(y_true, preds)),
    }

    cm = confusion_matrix(y_true, preds, labels=[0, 1])

    metrics["confusion_matrix"] = {
        "tn_human_pred_human": int(cm[0, 0]),
        "fp_human_pred_bot": int(cm[0, 1]),
        "fn_bot_pred_human": int(cm[1, 0]),
        "tp_bot_pred_bot": int(cm[1, 1]),
    }

    if len(set(y_true.tolist())) > 1:
        clipped = np.clip(scores, 1e-6, 1 - 1e-6)

        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
        metrics["log_loss"] = float(log_loss(y_true, clipped, labels=[0, 1]))
        metrics["brier"] = float(brier_score_loss(y_true, scores))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["log_loss"] = None
        metrics["brier"] = None

    return metrics


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Loading hierarchical torch model: {args.torch_model}")
    model, tokenizer, vectorizer, torch_artifact = load_torch_artifact(
        args.torch_model,
        device=device,
    )

    train_samples, val_samples = load_public_benchmark(args.data)

    if args.augment_windows:
        before = len(train_samples)
        train_samples = augment_chunk_windows(
            train_samples,
            window_hands=args.window_hands,
            stride=args.window_stride,
            keep_short_chunks=args.keep_short_window_chunks,
        )
        print(f"Train windows: {before} -> {len(train_samples)}")

    if args.augment_validation_windows:
        before = len(val_samples)
        val_samples = augment_chunk_windows(
            val_samples,
            window_hands=args.window_hands,
            stride=args.window_stride,
            keep_short_chunks=args.keep_short_window_chunks,
        )
        print(f"Validation windows: {before} -> {len(val_samples)}")

    print("Extracting train matrix...")
    X_train, y_train = extract_xgb_matrix(
        model=model,
        tokenizer=tokenizer,
        vectorizer=vectorizer,
        samples=train_samples,
        device=device,
        batch_size=args.batch_size,
    )

    print("Extracting validation matrix...")
    X_val, y_val = extract_xgb_matrix(
        model=model,
        tokenizer=tokenizer,
        vectorizer=vectorizer,
        samples=val_samples,
        device=device,
        batch_size=args.batch_size,
    )

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Train labels human/bot:", int((y_train == 0).sum()), int((y_train == 1).sum()))
    print("Val labels human/bot:", int((y_val == 0).sum()), int((y_val == 1).sum()))

    scale_pos_weight = float((y_train == 0).sum()) / max(1.0, float((y_train == 1).sum()))

    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=44,
    )

    print("Training XGBoost...")
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    train_scores = clf.predict_proba(X_train)[:, 1]
    val_scores = clf.predict_proba(X_val)[:, 1]

    train_metrics = compute_metrics(y_train, train_scores, args.threshold)
    val_metrics = compute_metrics(y_val, val_scores, args.threshold)

    print("\n=== Train metrics ===")
    print(json.dumps(train_metrics, indent=2))

    print("\n=== Validation metrics ===")
    print(json.dumps(val_metrics, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "xgb_model": clf,
        "torch_model_path": str(args.torch_model),
        "threshold": float(args.threshold),
        "window_hands": int(args.window_hands),
        "window_stride": int(args.window_stride),
        "xgb_input_dim": int(X_train.shape[1]),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "feature_description": "concat(chunk_embedding, engineered_chunk_features)",
    }

    joblib.dump(payload, out_path)

    print(f"\nSaved XGBoost detector to: {out_path}")


if __name__ == "__main__":
    main()