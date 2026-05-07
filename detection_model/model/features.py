from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np


class FeatureVectorizer:
    """Small deterministic chunk-level feature vectorizer."""

    def __init__(self):
        self.feature_names: List[str] = [
            "num_hands",
            "avg_actions_per_hand",
            "std_actions_per_hand",
            "avg_players",
            "avg_streets",
            "showdown_rate",
            "fold_rate",
            "call_rate",
            "check_rate",
            "bet_rate",
            "raise_rate",
            "blind_rate",
            "avg_amount_bb",
            "std_amount_bb",
            "avg_pot_before",
            "avg_pot_after",
            "action_entropy",
        ]
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def _entropy(self, counts: Counter) -> float:
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        probs = np.asarray([v / total for v in counts.values() if v > 0], dtype=np.float32)
        return float(-(probs * np.log(probs + 1e-12)).sum())

    def transform_one_raw(self, chunk: List[Dict[str, Any]]) -> np.ndarray:
        num_hands = len(chunk)
        if num_hands == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        action_counts_per_hand = []
        players_per_hand = []
        streets_per_hand = []
        showdown_flags = []
        normalized_amounts = []
        pot_before_values = []
        pot_after_values = []
        action_counter: Counter[str] = Counter()

        for hand in chunk:
            actions = hand.get("actions") or []
            players = hand.get("players") or []
            streets = hand.get("streets") or []
            outcome = hand.get("outcome") or {}

            action_counts_per_hand.append(len(actions))
            players_per_hand.append(len(players))
            streets_per_hand.append(len(streets))
            showdown_flags.append(1.0 if outcome.get("showdown") else 0.0)

            for action in actions:
                action_type = str(action.get("action_type") or "unknown").lower()
                action_counter[action_type] += 1
                normalized_amounts.append(float(action.get("normalized_amount_bb") or 0.0))
                pot_before_values.append(float(action.get("pot_before") or 0.0))
                pot_after_values.append(float(action.get("pot_after") or 0.0))

        meaningful = sum(action_counter.get(k, 0) for k in ["fold", "call", "check", "bet", "raise"])
        denom = max(1, meaningful)
        blind_total = action_counter.get("small_blind", 0) + action_counter.get("big_blind", 0)
        amount_arr = np.asarray(normalized_amounts or [0.0], dtype=np.float32)
        pot_before_arr = np.asarray(pot_before_values or [0.0], dtype=np.float32)
        pot_after_arr = np.asarray(pot_after_values or [0.0], dtype=np.float32)

        values = np.asarray([
            float(num_hands),
            float(np.mean(action_counts_per_hand)),
            float(np.std(action_counts_per_hand)),
            float(np.mean(players_per_hand)),
            float(np.mean(streets_per_hand)),
            float(np.mean(showdown_flags)),
            action_counter.get("fold", 0) / denom,
            action_counter.get("call", 0) / denom,
            action_counter.get("check", 0) / denom,
            action_counter.get("bet", 0) / denom,
            action_counter.get("raise", 0) / denom,
            blind_total / max(1, sum(action_counter.values())),
            float(amount_arr.mean()),
            float(amount_arr.std()),
            float(pot_before_arr.mean()),
            float(pot_after_arr.mean()),
            self._entropy(action_counter),
        ], dtype=np.float32)

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return values

    def fit(self, chunks: List[List[Dict[str, Any]]]) -> "FeatureVectorizer":
        raw = np.vstack([self.transform_one_raw(chunk) for chunk in chunks]).astype(np.float32)
        self.mean_ = raw.mean(axis=0)
        self.std_ = raw.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
        raw = np.vstack([self.transform_one_raw(chunk) for chunk in chunks]).astype(np.float32)
        if self.mean_ is None or self.std_ is None:
            return raw
        return ((raw - self.mean_) / self.std_).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.feature_names = list(state.get("feature_names") or self.feature_names)
        mean = state.get("mean")
        std = state.get("std")
        self.mean_ = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std_ = None if std is None else np.asarray(std, dtype=np.float32)

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "FeatureVectorizer":
        obj = cls()
        obj.load_state_dict(state)
        return obj
