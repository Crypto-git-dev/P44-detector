from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .action_vectorizer import ActionVectorizer
from .dataset import (
    augment_chunk_prefixes,
    augment_chunk_windows,
    load_public_benchmark,
)
from .features import FeatureVectorizer
from .hierarchical_dataset import (
    HierarchicalPokerChunkDataset,
    hierarchical_collate_batch,
)
from .hierarchical_model import HierarchicalChunkClassifier


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(
    path: Optional[str],
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    if not path:
        return None

    path_obj = Path(path).expanduser()

    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path_obj}")

    print(f"Loading checkpoint from: {path_obj}")

    checkpoint = torch.load(path_obj, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint should be dict, got: {type(checkpoint)}")

    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    return checkpoint


def backup_existing_file(path: Path, overwrite: bool) -> None:
    if overwrite or not path.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}.backup_{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)

    print(f"Existing checkpoint backed up to: {backup_path}")


def apply_checkpoint_config(
    args: argparse.Namespace,
    checkpoint: Optional[Dict[str, Any]],
) -> None:
    if checkpoint is None:
        return

    config = checkpoint.get("model_config") or {}

    if not isinstance(config, dict) or not config:
        return

    print("Using architecture config from checkpoint:")
    print(json.dumps(config, indent=2, default=str))

    args.max_actions_per_hand = int(
        config.get("max_actions_per_hand", args.max_actions_per_hand)
    )

    args.max_hands = int(
        config.get("max_hands", args.max_hands)
    )

    args.d_model = int(
        config.get("d_model", args.d_model)
    )

    args.layers = int(
        config.get("n_layers", args.layers)
    )

    args.heads = int(
        config.get("n_heads", args.heads)
    )

    args.chunk_layers = int(
        config.get("chunk_layers", config.get("gru_layers", args.chunk_layers))
    )

    args.dropout = float(
        config.get("dropout", args.dropout)
    )


def fit_or_load_action_vectorizer(
    checkpoint: Optional[Dict[str, Any]],
    train_chunks: List[List[Dict[str, Any]]],
    args: argparse.Namespace,
) -> ActionVectorizer:
    if checkpoint is not None and "action_vectorizer" in checkpoint:
        action_vectorizer = ActionVectorizer.from_state_dict(
            checkpoint["action_vectorizer"]
        )
        print("Loaded ActionVectorizer from checkpoint")
        return action_vectorizer

    action_vectorizer = ActionVectorizer(
        max_actions_per_hand=args.max_actions_per_hand,
    )

    action_vectorizer.fit(
        train_chunks,
        min_freq=args.min_freq,
    )

    print("Fitted ActionVectorizer from training chunks")
    print("Street vocab size:", action_vectorizer.street_vocab_size)
    print("Action type vocab size:", action_vectorizer.action_type_vocab_size)
    print("Seat vocab size:", action_vectorizer.seat_vocab_size)
    print("Numeric dim:", action_vectorizer.numeric_dim)

    return action_vectorizer


def fit_or_load_feature_vectorizer(
    checkpoint: Optional[Dict[str, Any]],
    train_chunks: List[List[Dict[str, Any]]],
) -> FeatureVectorizer:
    if checkpoint is not None and "vectorizer" in checkpoint:
        vectorizer = FeatureVectorizer.from_state_dict(checkpoint["vectorizer"])
        print("Loaded FeatureVectorizer from checkpoint")
        return vectorizer

    vectorizer = FeatureVectorizer()
    vectorizer.fit(train_chunks)

    print("Fitted FeatureVectorizer from training chunks")
    print("Feature dim:", len(vectorizer.feature_names))

    return vectorizer


def make_model(
    action_vectorizer: ActionVectorizer,
    feature_vectorizer: FeatureVectorizer,
    args: argparse.Namespace,
) -> HierarchicalChunkClassifier:
    if args.d_model % args.heads != 0:
        raise ValueError(
            f"d_model must be divisible by heads. "
            f"Got d_model={args.d_model}, heads={args.heads}"
        )

    return HierarchicalChunkClassifier(
        street_vocab_size=action_vectorizer.street_vocab_size,
        action_type_vocab_size=action_vectorizer.action_type_vocab_size,
        seat_vocab_size=action_vectorizer.seat_vocab_size,
        numeric_dim=action_vectorizer.numeric_dim,
        feature_dim=len(feature_vectorizer.feature_names),
        max_actions_per_hand=args.max_actions_per_hand,
        max_hands=args.max_hands,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        chunk_layers=args.chunk_layers,
        dropout=args.dropout,
        pad_id=0,
    )


def batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def forward_model(
    model: HierarchicalChunkClassifier,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    logits = model(
        action_cat=batch["action_cat"],
        action_num=batch["action_num"],
        action_mask=batch["action_mask"],
        hand_mask=batch["hand_mask"],
        features=batch["features"],
    )

    return logits.view(-1)


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    preds = (scores >= threshold).astype(np.int32)

    metrics: Dict[str, Any] = {
        "count": int(len(labels)),
        "threshold": float(threshold),
        "human_count": int((labels == 0).sum()),
        "bot_count": int((labels == 1).sum()),
        "score_min": float(scores.min()) if len(scores) else 0.0,
        "score_max": float(scores.max()) if len(scores) else 0.0,
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "score_std": float(scores.std()) if len(scores) else 0.0,
        "accuracy": float(accuracy_score(labels, preds)) if len(labels) else 0.0,
    }

    human_scores = scores[labels == 0]
    bot_scores = scores[labels == 1]

    metrics["human_score_mean"] = (
        float(human_scores.mean()) if len(human_scores) else None
    )

    metrics["bot_score_mean"] = (
        float(bot_scores.mean()) if len(bot_scores) else None
    )

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    metrics["confusion_matrix"] = {
        "tn_human_pred_human": int(cm[0, 0]),
        "fp_human_pred_bot": int(cm[0, 1]),
        "fn_bot_pred_human": int(cm[1, 0]),
        "tp_bot_pred_bot": int(cm[1, 1]),
    }

    if len(set(labels.tolist())) > 1:
        clipped = np.clip(scores, 1e-6, 1 - 1e-6)

        metrics["log_loss"] = float(log_loss(labels, clipped, labels=[0, 1]))
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        metrics["log_loss"] = 0.0
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    return metrics


@torch.no_grad()
def evaluate(
    model: HierarchicalChunkClassifier,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    threshold: float,
) -> Dict[str, Any]:
    model.eval()

    losses: List[float] = []
    labels_all: List[float] = []
    scores_all: List[float] = []

    for batch in loader:
        batch = batch_to_device(batch, device)

        logits = forward_model(model, batch)
        labels = batch["labels"].view(-1)

        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)

        losses.append(float(loss.item()))
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        scores_all.extend(probs.detach().cpu().numpy().tolist())

    labels_np = np.asarray(labels_all, dtype=np.int32)
    scores_np = np.asarray(scores_all, dtype=np.float32).clip(1e-6, 1 - 1e-6)

    metrics = compute_metrics(
        labels=labels_np,
        scores=scores_np,
        threshold=threshold,
    )

    metrics["loss"] = float(np.mean(losses)) if losses else 0.0

    return metrics


def train_one_epoch(
    model: HierarchicalChunkClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float,
    epoch: int,
    end_epoch: int,
) -> float:
    model.train()

    running_losses: List[float] = []

    for batch in tqdm(loader, desc=f"epoch {epoch}/{end_epoch}"):
        batch = batch_to_device(batch, device)

        labels = batch["labels"].view(-1)

        optimizer.zero_grad(set_to_none=True)

        logits = forward_model(model, batch)
        loss = criterion(logits, labels)

        loss.backward()

        if max_grad_norm and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        running_losses.append(float(loss.item()))

    return float(np.mean(running_losses)) if running_losses else 0.0


def save_artifact(
    out_path: Path,
    model: HierarchicalChunkClassifier,
    optimizer: torch.optim.Optimizer,
    action_vectorizer: ActionVectorizer,
    vectorizer: FeatureVectorizer,
    args: argparse.Namespace,
    history: List[Dict[str, Any]],
    epoch: int,
    best_metric: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "architecture": "structured_action_transformer_hand_gru_chunk",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": dict(model.config),
        "action_vectorizer": action_vectorizer.state_dict(),
        "vectorizer": vectorizer.state_dict(),
        "threshold": float(args.threshold),
        "history": history,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "training_data": str(args.data),
        "train_args": vars(args),
    }

    torch.save(artifact, out_path)
    print(f"Saved model artifact to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train structured-action hierarchical Poker44 chunk-level bot detector."
    )

    parser.add_argument("--data", required=True, help="Path to public_miner_benchmark.json.gz")
    parser.add_argument("--out", default="artifacts/p44_action_vector_gru_window4.pt")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--max-actions-per-hand", type=int, default=64)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1, help="Action-level Transformer layers")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--chunk-layers", type=int, default=1, help="GRU/chunk encoder layers")
    parser.add_argument("--dropout", type=float, default=0.30)

    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--augment-prefixes", action="store_true")
    parser.add_argument("--min-prefix-hands", type=int, default=4)
    parser.add_argument("--max-prefixes-per-chunk", type=int, default=32)

    parser.add_argument("--augment-windows", action="store_true")
    parser.add_argument("--augment-validation-windows", action="store_true")
    parser.add_argument("--window-hands", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--keep-short-window-chunks", action="store_true")

    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument(
        "--calibrate-visible-actions",
        action="store_true",
        help="Apply validator-style deterministic visible-action sampling during training.",
    )

    parser.add_argument(
        "--visible-action-window-size",
        type=int,
        default=8,
        help="Number of visible actions per hand, matching validator sampling.",
    )

    parser.add_argument(
        "--calibrate-validation-visible-actions",
        action="store_true",
        help="Also apply validator-style visible-action sampling to validation dataset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    print(f"Using device: {device}")

    checkpoint = load_checkpoint(args.resume_from, device)
    apply_checkpoint_config(args, checkpoint)

    train_samples, val_samples = load_public_benchmark(
        args.data,
        seed=args.seed,
    )

    if not train_samples or not val_samples:
        raise RuntimeError("Dataset loading failed: train or validation split is empty.")

    print(f"Original train chunks: {len(train_samples)}")
    print(f"Original validation chunks: {len(val_samples)}")

    if args.augment_prefixes and args.augment_windows:
        raise ValueError(
            "Use either --augment-prefixes or --augment-windows, not both."
        )

    if args.augment_prefixes:
        before = len(train_samples)

        train_samples = augment_chunk_prefixes(
            train_samples,
            min_prefix_hands=args.min_prefix_hands,
            max_prefixes_per_chunk=args.max_prefixes_per_chunk,
            include_full_chunk=True,
        )

        print(
            f"Prefix-augmented train chunks: {before} -> {len(train_samples)} "
            f"(min_prefix_hands={args.min_prefix_hands}, "
            f"max_prefixes_per_chunk={args.max_prefixes_per_chunk})"
        )

    if args.augment_windows:
        before = len(train_samples)

        train_samples = augment_chunk_windows(
            train_samples,
            window_hands=args.window_hands,
            stride=args.window_stride,
            keep_short_chunks=args.keep_short_window_chunks,
        )

        print(
            f"Sliding-window train chunks: {before} -> {len(train_samples)} "
            f"(window_hands={args.window_hands}, stride={args.window_stride})"
        )

    if args.augment_validation_windows:
        before = len(val_samples)

        val_samples = augment_chunk_windows(
            val_samples,
            window_hands=args.window_hands,
            stride=args.window_stride,
            keep_short_chunks=args.keep_short_window_chunks,
        )

        print(
            f"Sliding-window validation chunks: {before} -> {len(val_samples)} "
            f"(window_hands={args.window_hands}, stride={args.window_stride})"
        )

    if not train_samples:
        raise RuntimeError("No training samples after augmentation.")

    if not val_samples:
        raise RuntimeError("No validation samples after augmentation.")

    train_chunks = [sample.chunk for sample in train_samples]

    action_vectorizer = fit_or_load_action_vectorizer(
        checkpoint=checkpoint,
        train_chunks=train_chunks,
        args=args,
    )

    vectorizer = fit_or_load_feature_vectorizer(
        checkpoint=checkpoint,
        train_chunks=train_chunks,
    )

    train_ds = HierarchicalPokerChunkDataset(
        samples=train_samples,
        action_vectorizer=action_vectorizer,
        feature_vectorizer=vectorizer,
        max_hands=args.max_hands,
        calibrate_visible_actions=args.calibrate_visible_actions,
        visible_action_window_size=args.visible_action_window_size,
        recompute_features_after_calibration=True,
    )

    val_ds = HierarchicalPokerChunkDataset(
        samples=val_samples,
        action_vectorizer=action_vectorizer,
        feature_vectorizer=vectorizer,
        max_hands=args.max_hands,
        calibrate_visible_actions=args.calibrate_validation_visible_actions,
        visible_action_window_size=args.visible_action_window_size,
        recompute_features_after_calibration=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda b: hierarchical_collate_batch(
            b,
            cat_pad_id=0,
            numeric_dim=action_vectorizer.numeric_dim,
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda b: hierarchical_collate_batch(
            b,
            cat_pad_id=0,
            numeric_dim=action_vectorizer.numeric_dim,
        ),
    )

    print("Final train chunks:", len(train_ds))
    print("Validation chunks:", len(val_ds))
    print("Batch size:", args.batch_size)
    print("Batches per epoch:", len(train_loader))
    print("Optimizer steps:", len(train_loader) * args.epochs)

    print("ActionVectorizer:")
    print("  street_vocab_size:", action_vectorizer.street_vocab_size)
    print("  action_type_vocab_size:", action_vectorizer.action_type_vocab_size)
    print("  seat_vocab_size:", action_vectorizer.seat_vocab_size)
    print("  numeric_dim:", action_vectorizer.numeric_dim)

    print("FeatureVectorizer:")
    print("  feature_dim:", len(vectorizer.feature_names))

    model = make_model(
        action_vectorizer=action_vectorizer,
        feature_vectorizer=vectorizer,
        args=args,
    ).to(device)

    if checkpoint is not None:
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("model")
            or checkpoint.get("state_dict")
        )

        if state is None:
            raise KeyError(f"No model state found in checkpoint. Keys={list(checkpoint.keys())}")

        model.load_state_dict(state)
        print("Loaded model weights from checkpoint")

    labels = np.asarray([sample.label for sample in train_samples], dtype=np.float32)
    pos = float(labels.sum())
    neg = float(len(labels) - pos)

    if args.no_pos_weight:
        pos_weight = None
        print("Using BCEWithLogitsLoss without pos_weight")
    else:
        pos_weight_value = neg / max(pos, 1.0)
        pos_weight = torch.tensor([pos_weight_value], device=device)
        print(
            f"Class balance: human={int(neg)}, bot={int(pos)}, "
            f"pos_weight={pos_weight_value:.4f}"
        )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    history: List[Dict[str, Any]] = []
    best_metric = -1.0

    if checkpoint is not None:
        if isinstance(checkpoint.get("history"), list):
            history = checkpoint["history"]

        if isinstance(checkpoint.get("best_metric"), (int, float)):
            best_metric = float(checkpoint["best_metric"])

        if not args.fine_tune:
            opt_state = checkpoint.get("optimizer_state_dict")

            if opt_state is not None:
                try:
                    optimizer.load_state_dict(opt_state)
                    print("Loaded optimizer state from checkpoint")
                except Exception as exc:
                    print(
                        "Could not load optimizer state. "
                        f"Fresh optimizer will be used. Error: {exc}"
                    )

            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            print(f"Resume mode: starting from epoch {start_epoch}")
        else:
            print("Fine-tune mode: loaded weights, fresh optimizer.")

    out_path = Path(args.out).expanduser()
    backup_existing_file(out_path, overwrite=args.overwrite)

    best_state = None
    final_epoch = start_epoch - 1
    end_epoch = start_epoch + args.epochs - 1

    for epoch in range(start_epoch, start_epoch + args.epochs):
        final_epoch = epoch

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_grad_norm=args.max_grad_norm,
            epoch=epoch,
            end_epoch=end_epoch,
        )

        metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            threshold=args.threshold,
        )

        metrics["train_loss"] = float(train_loss)
        metrics["epoch"] = int(epoch)

        history.append(metrics)

        print(json.dumps(metrics, indent=2))

        roc_auc = metrics.get("roc_auc", 0.0)
        pr_auc = metrics.get("pr_auc", 0.0)

        selection_metric = float(roc_auc or 0.0) + float(pr_auc or 0.0)

        if selection_metric > best_metric:
            best_metric = selection_metric

            best_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }

            print(f"New best metric: {best_metric:.6f}")

            save_artifact(
                out_path=out_path,
                model=model,
                optimizer=optimizer,
                action_vectorizer=action_vectorizer,
                vectorizer=vectorizer,
                args=args,
                history=history,
                epoch=epoch,
                best_metric=best_metric,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    save_artifact(
        out_path=out_path,
        model=model,
        optimizer=optimizer,
        action_vectorizer=action_vectorizer,
        vectorizer=vectorizer,
        args=args,
        history=history,
        epoch=final_epoch,
        best_metric=best_metric,
    )

    print("Training finished.")
    print(f"Best metric: {best_metric:.6f}")
    print(f"Saved artifact: {out_path}")


if __name__ == "__main__":
    main()