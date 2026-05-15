from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from .dataset import ChunkSample
from .features import FeatureVectorizer
from .action_vectorizer import ActionVectorizer

import copy
import hashlib
from typing import Optional

def _sample_visible_indices(
    total: int,
    *,
    window_size: int,
    seed_parts: list[str],
    actions: Optional[list[dict]] = None,
) -> list[int]:
    """
    Match validator-style deterministic visible-action sampling.

    Keeps:
      - first action
      - last action
      - at least one action from each street bucket when possible
      - deterministic sampled middle actions

    Important:
      This preserves chronological order by returning sorted indices.
    """

    if total <= 1:
        return [0] * max(1, window_size)

    if total <= window_size:
        return list(range(total))

    seed = "|".join(seed_parts).encode("utf-8", errors="ignore")

    def _sort_key(index: int, extra: str = "") -> bytes:
        return hashlib.sha256(
            seed + f":{index}:{extra}".encode("utf-8")
        ).digest()

    picked = {0, total - 1}

    if actions:
        street_buckets: dict[str, list[int]] = {}

        for idx in range(1, total - 1):
            action = actions[idx] if idx < len(actions) else {}
            street = str(action.get("street", "") or "preflop").lower()
            street_buckets.setdefault(street, []).append(idx)

        for street in sorted(street_buckets.keys()):
            if len(picked) >= window_size:
                break

            ordered = sorted(
                street_buckets[street],
                key=lambda idx: _sort_key(idx, street),
            )

            if ordered:
                picked.add(ordered[0])

    middle = [
        idx
        for idx in range(1, total - 1)
        if idx not in picked
    ]

    ordered_middle = sorted(middle, key=_sort_key)

    for idx in ordered_middle:
        if len(picked) >= window_size:
            break

        picked.add(idx)

    return sorted(picked)


def calibrate_hand_visible_actions(
    hand: dict,
    *,
    window_size: int,
    seed_parts: list[str],
) -> dict:
    """
    Apply validator-style visible-action sampling to one hand.

    This creates a copied hand with sampled actions.
    Original hand is not mutated.
    """

    if not isinstance(hand, dict):
        return hand

    actions = hand.get("actions") or []

    if not isinstance(actions, list) or not actions:
        return hand

    indices = _sample_visible_indices(
        total=len(actions),
        window_size=window_size,
        seed_parts=seed_parts,
        actions=actions,
    )

    sampled_actions = []

    for idx in indices:
        if 0 <= idx < len(actions):
            sampled_actions.append(actions[idx])

    new_hand = copy.deepcopy(hand)
    new_hand["actions"] = sampled_actions

    return new_hand


def calibrate_chunk_visible_actions(
    chunk: list[dict],
    *,
    window_size: int,
    chunk_id: str,
) -> list[dict]:
    """
    Apply validator-style visible-action sampling to every hand in one chunk.
    """

    calibrated: list[dict] = []

    for hand_idx, hand in enumerate(chunk):
        actions = hand.get("actions") or [] if isinstance(hand, dict) else []

        seed_parts = [
            str(chunk_id),
            f"hand_{hand_idx}",
            f"actions_{len(actions)}",
        ]

        calibrated.append(
            calibrate_hand_visible_actions(
                hand,
                window_size=window_size,
                seed_parts=seed_parts,
            )
        )

    return calibrated

class HierarchicalPokerChunkDataset(Dataset):
    def __init__(
        self,
        samples,
        action_vectorizer,
        feature_vectorizer,
        max_hands: int,
        calibrate_visible_actions: bool = False,
        visible_action_window_size: int = 8,
        recompute_features_after_calibration: bool = True,
    ):
        self.samples = samples
        self.action_vectorizer = action_vectorizer
        self.feature_vectorizer = feature_vectorizer
        self.max_hands = int(max_hands)

        self.calibrate_visible_actions = bool(calibrate_visible_actions)
        self.visible_action_window_size = int(visible_action_window_size)
        self.recompute_features_after_calibration = bool(
            recompute_features_after_calibration
        )

        chunks = [sample.chunk for sample in samples]
        self.feature_matrix = feature_vectorizer.transform(chunks)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        chunk = sample.chunk
        chunk_id = getattr(sample, "chunk_id", None) or f"sample_{idx}"

        if self.calibrate_visible_actions:
            chunk = calibrate_chunk_visible_actions(
                chunk,
                window_size=self.visible_action_window_size,
                chunk_id=str(chunk_id),
            )

        action_cat, action_num = self.action_vectorizer.encode_chunk(
            chunk,
            max_hands=self.max_hands,
        )

        if self.calibrate_visible_actions and self.recompute_features_after_calibration:
            features = self.feature_vectorizer.transform([chunk])[0]
        else:
            features = self.feature_matrix[idx]

        return {
            "action_cat": action_cat,
            "action_num": action_num,
            "features": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(float(sample.label), dtype=torch.float32),
            "num_hands": torch.tensor(len(action_cat), dtype=torch.long),
        }


def hierarchical_collate_batch(
    batch: List[Dict[str, Any]],
    cat_pad_id: int = 0,
    numeric_dim: int = 18,
) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)

    max_hands = max(len(item["action_cat"]) for item in batch)
    max_actions = max(
        len(hand_actions)
        for item in batch
        for hand_actions in item["action_cat"]
    )

    max_hands = max(1, max_hands)
    max_actions = max(1, max_actions)

    action_cat = torch.full(
        size=(batch_size, max_hands, max_actions, 3),
        fill_value=cat_pad_id,
        dtype=torch.long,
    )

    action_num = torch.zeros(
        size=(batch_size, max_hands, max_actions, numeric_dim),
        dtype=torch.float32,
    )

    action_mask = torch.zeros(
        size=(batch_size, max_hands, max_actions),
        dtype=torch.bool,
    )

    hand_mask = torch.zeros(
        size=(batch_size, max_hands),
        dtype=torch.bool,
    )

    features = torch.stack([item["features"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    for batch_idx, item in enumerate(batch):
        cat_hands = item["action_cat"]
        num_hands = item["action_num"]

        for hand_idx, (cat_rows, num_rows) in enumerate(zip(cat_hands, num_hands)):
            if hand_idx >= max_hands:
                break

            length = min(len(cat_rows), max_actions)

            if length <= 0:
                continue

            action_cat[batch_idx, hand_idx, :length, :] = torch.tensor(
                cat_rows[:length],
                dtype=torch.long,
            )

            action_num[batch_idx, hand_idx, :length, :] = torch.tensor(
                num_rows[:length],
                dtype=torch.float32,
            )

            action_mask[batch_idx, hand_idx, :length] = True
            hand_mask[batch_idx, hand_idx] = True

    return {
        "action_cat": action_cat,
        "action_num": action_num,
        "action_mask": action_mask,
        "hand_mask": hand_mask,
        "features": features,
        "labels": labels,
    }