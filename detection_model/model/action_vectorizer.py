from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import math
import numpy as np


class ActionVectorizer:
    """
    Converts each poker action into:
      - categorical ids: street_id, action_type_id, actor_seat_id
      - numeric feature vector

    One hand becomes:
      cat_ids:      [num_actions, 3]
      numeric_feats:[num_actions, numeric_dim]
    """

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, max_actions_per_hand: int = 64):
        self.max_actions_per_hand = int(max_actions_per_hand)

        self.street_to_id = {
            self.PAD: 0,
            self.UNK: 1,
        }

        self.action_type_to_id = {
            self.PAD: 0,
            self.UNK: 1,
        }

        self.seat_to_id = {
            self.PAD: 0,
            self.UNK: 1,
        }

        self.numeric_feature_names = [
            "action_order_norm",
            "amount_log",
            "raise_to_log",
            "call_to_log",
            "normalized_amount_bb_log",
            "pot_before_log",
            "pot_after_log",
            "pot_delta_log",
            "amount_to_pot",
            "raise_to_pot",
            "call_to_pot",
            "has_amount",
            "has_raise_to",
            "has_call_to",
            "pot_increased",
            "pot_decreased",
            "is_forced_action",
            "is_money_action",
        ]

    @property
    def numeric_dim(self) -> int:
        return len(self.numeric_feature_names)

    @property
    def street_vocab_size(self) -> int:
        return len(self.street_to_id)

    @property
    def action_type_vocab_size(self) -> int:
        return len(self.action_type_to_id)

    @property
    def seat_vocab_size(self) -> int:
        return len(self.seat_to_id)

    def normalize_text(self, value: Any, default: str = "unknown") -> str:
        if value is None:
            return default

        text = str(value).strip().lower()

        if not text:
            return default

        return (
            text.replace("-", "_")
            .replace(" ", "_")
            .replace("/", "_")
        )

    def safe_float(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return default

        try:
            if isinstance(value, str):
                value = (
                    value.replace(",", "")
                    .replace("€", "")
                    .replace("$", "")
                    .replace("£", "")
                )
            return float(value)
        except Exception:
            return default

    def safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return default

    def is_present(self, value: Any) -> bool:
        if value is None:
            return False

        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
            return False

        return True

    def log1p_value(self, value: float) -> float:
        value = max(0.0, float(value))
        return math.log1p(value)

    def bounded_ratio(self, numerator: float, denominator: float, cap: float = 10.0) -> float:
        if denominator <= 0:
            return 0.0

        ratio = numerator / denominator
        ratio = max(0.0, min(float(ratio), cap))

        # Scale to roughly [0, 1]
        return ratio / cap

    def fit(
        self,
        chunks: List[List[Dict[str, Any]]],
        min_freq: int = 1,
    ) -> "ActionVectorizer":
        street_counter = Counter()
        action_counter = Counter()
        seat_counter = Counter()

        for chunk in chunks:
            for hand in chunk:
                actions = hand.get("actions") or []

                for action in actions:
                    street = self.normalize_text(action.get("street"))
                    action_type = self.normalize_text(action.get("action_type"))
                    seat = self.safe_int(action.get("actor_seat"), default=0)

                    street_counter[street] += 1
                    action_counter[action_type] += 1
                    seat_counter[f"seat_{seat}" if seat > 0 else "seat_unknown"] += 1

        for value, count in street_counter.items():
            if count >= min_freq and value not in self.street_to_id:
                self.street_to_id[value] = len(self.street_to_id)

        for value, count in action_counter.items():
            if count >= min_freq and value not in self.action_type_to_id:
                self.action_type_to_id[value] = len(self.action_type_to_id)

        for value, count in seat_counter.items():
            if count >= min_freq and value not in self.seat_to_id:
                self.seat_to_id[value] = len(self.seat_to_id)

        return self

    def encode_action(
        self,
        action: Dict[str, Any],
        action_index: int,
        total_actions: int,
    ) -> tuple[list[int], list[float]]:
        street = self.normalize_text(action.get("street"))
        action_type = self.normalize_text(action.get("action_type"))
        seat = self.safe_int(action.get("actor_seat"), default=0)
        seat_key = f"seat_{seat}" if seat > 0 else "seat_unknown"

        street_id = self.street_to_id.get(street, self.street_to_id[self.UNK])
        action_type_id = self.action_type_to_id.get(action_type, self.action_type_to_id[self.UNK])
        seat_id = self.seat_to_id.get(seat_key, self.seat_to_id[self.UNK])

        amount = self.safe_float(action.get("amount"), default=0.0)
        raise_to = self.safe_float(action.get("raise_to"), default=0.0)
        call_to = self.safe_float(action.get("call_to"), default=0.0)
        normalized_amount_bb = self.safe_float(action.get("normalized_amount_bb"), default=0.0)
        pot_before = self.safe_float(action.get("pot_before"), default=0.0)
        pot_after = self.safe_float(action.get("pot_after"), default=0.0)

        has_raise_to = 1.0 if self.is_present(action.get("raise_to")) else 0.0
        has_call_to = 1.0 if self.is_present(action.get("call_to")) else 0.0
        has_amount = 1.0 if amount > 0 else 0.0

        pot_delta = pot_after - pot_before

        forced_actions = {
            "small_blind",
            "big_blind",
            "ante",
            "straddle",
            "bring_in",
        }

        money_actions = {
            "small_blind",
            "big_blind",
            "ante",
            "straddle",
            "bring_in",
            "call",
            "bet",
            "raise",
            "all_in",
            "allin",
        }

        is_forced_action = 1.0 if action_type in forced_actions else 0.0
        is_money_action = 1.0 if action_type in money_actions else 0.0

        if total_actions <= 1:
            action_order_norm = 0.0
        else:
            action_order_norm = action_index / max(1, total_actions - 1)

        numeric = [
            action_order_norm,
            self.log1p_value(amount),
            self.log1p_value(raise_to),
            self.log1p_value(call_to),
            self.log1p_value(normalized_amount_bb),
            self.log1p_value(pot_before),
            self.log1p_value(pot_after),
            self.log1p_value(abs(pot_delta)),
            self.bounded_ratio(amount, pot_before),
            self.bounded_ratio(raise_to, pot_before),
            self.bounded_ratio(call_to, pot_before),
            has_amount,
            has_raise_to,
            has_call_to,
            1.0 if pot_delta > 0 else 0.0,
            1.0 if pot_delta < 0 else 0.0,
            is_forced_action,
            is_money_action,
        ]

        cat_ids = [street_id, action_type_id, seat_id]

        return cat_ids, numeric

    def encode_hand(self, hand: Dict[str, Any]) -> tuple[list[list[int]], list[list[float]]]:
        actions = hand.get("actions") or []

        cat_rows: list[list[int]] = []
        num_rows: list[list[float]] = []

        for idx, action in enumerate(actions[: self.max_actions_per_hand]):
            cat_ids, numeric = self.encode_action(
                action=action,
                action_index=idx,
                total_actions=len(actions),
            )

            cat_rows.append(cat_ids)
            num_rows.append(numeric)

        if not cat_rows:
            cat_rows.append([0, 0, 0])
            num_rows.append([0.0 for _ in range(self.numeric_dim)])

        return cat_rows, num_rows

    def encode_chunk(
        self,
        chunk: List[Dict[str, Any]],
        max_hands: int,
    ) -> tuple[list[list[list[int]]], list[list[list[float]]]]:
        chunk_cat: list[list[list[int]]] = []
        chunk_num: list[list[list[float]]] = []

        for hand in chunk[:max_hands]:
            cat_rows, num_rows = self.encode_hand(hand)
            chunk_cat.append(cat_rows)
            chunk_num.append(num_rows)

        if not chunk_cat:
            chunk_cat.append([[0, 0, 0]])
            chunk_num.append([[0.0 for _ in range(self.numeric_dim)]])

        return chunk_cat, chunk_num

    def state_dict(self) -> Dict[str, Any]:
        return {
            "max_actions_per_hand": self.max_actions_per_hand,
            "street_to_id": dict(self.street_to_id),
            "action_type_to_id": dict(self.action_type_to_id),
            "seat_to_id": dict(self.seat_to_id),
            "numeric_feature_names": list(self.numeric_feature_names),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.max_actions_per_hand = int(
            state.get("max_actions_per_hand", self.max_actions_per_hand)
        )

        self.street_to_id = {
            str(k): int(v)
            for k, v in state["street_to_id"].items()
        }

        self.action_type_to_id = {
            str(k): int(v)
            for k, v in state["action_type_to_id"].items()
        }

        self.seat_to_id = {
            str(k): int(v)
            for k, v in state["seat_to_id"].items()
        }

        self.numeric_feature_names = list(
            state.get("numeric_feature_names", self.numeric_feature_names)
        )

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "ActionVectorizer":
        obj = cls(
            max_actions_per_hand=int(state.get("max_actions_per_hand", 64))
        )
        obj.load_state_dict(state)
        return obj