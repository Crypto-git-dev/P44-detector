from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from .dataset import ChunkSample
from .features import FeatureVectorizer
from .action_vectorizer import ActionVectorizer


class HierarchicalPokerChunkDataset(Dataset):
    def __init__(
        self,
        samples: List[ChunkSample],
        action_vectorizer: ActionVectorizer,
        feature_vectorizer: FeatureVectorizer,
        max_hands: int,
    ):
        self.samples = samples
        self.action_vectorizer = action_vectorizer
        self.feature_vectorizer = feature_vectorizer
        self.max_hands = int(max_hands)

        chunks = [sample.chunk for sample in samples]
        self.feature_matrix = feature_vectorizer.transform(chunks)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        action_cat, action_num = self.action_vectorizer.encode_chunk(
            sample.chunk,
            max_hands=self.max_hands,
        )

        return {
            "action_cat": action_cat,
            "action_num": action_num,
            "features": torch.tensor(self.feature_matrix[idx], dtype=torch.float32),
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