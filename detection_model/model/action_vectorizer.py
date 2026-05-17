from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import math


class ActionVectorizer:
    """
    Converts each poker action into compact, essential model inputs:
      - categorical ids: street_id, action_type_id, actor_seat_id
      - numeric action context features

    The hand/chunk structure is preserved for the hierarchical encoder. Wide
    per-seat hand features are intentionally not generated here; table/stack
    summaries live in FeatureVectorizer where they are easier for XGBoost to use.
    """

    PAD = "<PAD>"
    UNK = "<UNK>"

    FORCED_ACTIONS = {"small_blind", "big_blind", "ante", "straddle", "bring_in"}
    MONEY_ACTIONS = FORCED_ACTIONS | {"call", "bet", "raise", "all_in", "allin"}
    AGGRESSIVE_ACTIONS = {"bet", "raise", "all_in", "allin"}
    PASSIVE_ACTIONS = {"check", "call"}

    def __init__(self, max_actions_per_hand: int = 64, max_table_seats: int = 9):
        self.max_actions_per_hand = max(1, int(max_actions_per_hand))
        self.max_table_seats = max(1, int(max_table_seats))

        self.street_to_id = {self.PAD: 0, self.UNK: 1}
        self.action_type_to_id = {self.PAD: 0, self.UNK: 1}
        self.seat_to_id = {self.PAD: 0, self.UNK: 1}

        self.numeric_feature_names = [
            # order / money / pot context
            "action_order_norm",
            "amount_bb_scaled",
            "raise_to_bb_scaled",
            "call_to_bb_scaled",
            "pot_before_bb_scaled",
            "pot_after_bb_scaled",
            "pot_delta_bb_signed",
            "amount_to_pot",
            "raise_to_pot",
            "call_to_pot",
            # presence flags
            "has_amount",
            "has_raise_to",
            "has_call_to",
            # action semantics
            "is_forced_action",
            "is_money_action",
            "is_aggressive_action",
            "is_passive_action",
            "is_all_in",
            # table/actor context
            "actor_seat_norm",
            "actor_stack_bb_scaled",
            "amount_to_actor_stack",
            "street_progress",
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
        return text.replace("-", "_").replace(" ", "_").replace("/", "_")

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
                    .strip()
                )
            out = float(value)
            return out if math.isfinite(out) else default
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

    def capped_ratio(self, numerator: float, denominator: float, cap: float = 10.0) -> float:
        if denominator <= 0:
            return 0.0
        ratio = max(0.0, min(float(numerator) / float(denominator), cap))
        return ratio / cap

    def scaled_bb(self, value: float, cap: float = 200.0) -> float:
        value = max(0.0, float(value))
        if cap <= 0:
            return 0.0
        return min(math.log1p(value) / math.log1p(cap), 1.0)

    def parse_seat_from_uid(self, value: Any) -> int:
        if value is None:
            return 0
        text = str(value).strip().lower()
        if text.startswith("seat_"):
            return self.safe_int(text.replace("seat_", ""), default=0)
        return 0

    def _infer_hand_max_seats_raw(self, hand: Dict[str, Any]) -> int:
        if not isinstance(hand, dict):
            return self.max_table_seats

        metadata = hand.get("metadata") or {}
        explicit = self.safe_int(metadata.get("max_seats"), default=0)
        if explicit > 0:
            return explicit

        seats: List[int] = []
        for key in ("hero_seat", "button_seat"):
            seat = self.safe_int(metadata.get(key), default=0)
            if seat > 0:
                seats.append(seat)

        for player in hand.get("players") or []:
            if not isinstance(player, dict):
                continue
            seat = self.safe_int(player.get("seat"), default=0) or self.parse_seat_from_uid(player.get("player_uid"))
            if seat > 0:
                seats.append(seat)

        for action in hand.get("actions") or []:
            if not isinstance(action, dict):
                continue
            seat = self.safe_int(action.get("actor_seat"), default=0)
            if seat > 0:
                seats.append(seat)

        return max(seats) if seats else self.max_table_seats

    def get_hand_max_seats(self, hand: Dict[str, Any]) -> int:
        raw = self._infer_hand_max_seats_raw(hand)
        return max(1, min(int(raw), self.max_table_seats))

    def get_players_by_seat(self, hand: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        players_by_seat: Dict[int, Dict[str, Any]] = {}
        if not isinstance(hand, dict):
            return players_by_seat

        max_seats = self.get_hand_max_seats(hand)
        for player in hand.get("players") or []:
            if not isinstance(player, dict):
                continue
            seat = self.safe_int(player.get("seat"), default=0) or self.parse_seat_from_uid(player.get("player_uid"))
            if 1 <= seat <= max_seats:
                players_by_seat[seat] = player
        return players_by_seat

    def fit(self, chunks: List[List[Dict[str, Any]]], min_freq: int = 1) -> "ActionVectorizer":
        street_counter: Counter[str] = Counter()
        action_counter: Counter[str] = Counter()
        seat_counter: Counter[str] = Counter()
        observed_max_seats = self.max_table_seats

        for chunk in chunks:
            if not isinstance(chunk, list):
                continue
            for hand in chunk:
                if not isinstance(hand, dict):
                    continue
                observed_max_seats = max(observed_max_seats, self._infer_hand_max_seats_raw(hand))
                for action in hand.get("actions") or []:
                    if not isinstance(action, dict):
                        continue
                    street = self.normalize_text(action.get("street"))
                    action_type = self.normalize_text(action.get("action_type"))
                    seat = self.safe_int(action.get("actor_seat"), default=0)

                    street_counter[street] += 1
                    action_counter[action_type] += 1
                    seat_counter[f"seat_{seat}" if seat > 0 else "seat_unknown"] += 1

        self.max_table_seats = max(1, int(observed_max_seats))

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
        *,
        bb: float,
        max_seats: int,
        actor_stack: float,
    ) -> Tuple[List[int], List[float]]:
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
        pot_delta = pot_after - pot_before

        amount_bb = normalized_amount_bb if normalized_amount_bb > 0 else (amount / bb if amount > 0 and bb > 0 else 0.0)
        raise_to_bb = raise_to / bb if raise_to > 0 and bb > 0 else 0.0
        call_to_bb = call_to / bb if call_to > 0 and bb > 0 else 0.0
        pot_before_bb = pot_before / bb if pot_before > 0 and bb > 0 else 0.0
        pot_after_bb = pot_after / bb if pot_after > 0 and bb > 0 else 0.0
        pot_delta_bb = pot_delta / bb if bb > 0 else 0.0
        actor_stack_bb = actor_stack / bb if actor_stack > 0 and bb > 0 else 0.0

        action_order_norm = 0.0 if total_actions <= 1 else action_index / max(1, total_actions - 1)
        street_progress = {"preflop": 0.25, "flop": 0.50, "turn": 0.75, "river": 1.00}.get(street, 0.0)

        feature_map: Dict[str, float] = {
            "action_order_norm": float(action_order_norm),
            "amount_bb_scaled": self.scaled_bb(amount_bb),
            "raise_to_bb_scaled": self.scaled_bb(raise_to_bb),
            "call_to_bb_scaled": self.scaled_bb(call_to_bb),
            "pot_before_bb_scaled": self.scaled_bb(pot_before_bb),
            "pot_after_bb_scaled": self.scaled_bb(pot_after_bb),
            "pot_delta_bb_signed": max(-1.0, min(1.0, pot_delta_bb / 200.0)),
            "amount_to_pot": self.capped_ratio(amount, pot_before),
            "raise_to_pot": self.capped_ratio(raise_to, pot_before),
            "call_to_pot": self.capped_ratio(call_to, pot_before),
            "has_amount": 1.0 if amount > 0 else 0.0,
            "has_raise_to": 1.0 if self.is_present(action.get("raise_to")) else 0.0,
            "has_call_to": 1.0 if self.is_present(action.get("call_to")) else 0.0,
            "is_forced_action": 1.0 if action_type in self.FORCED_ACTIONS else 0.0,
            "is_money_action": 1.0 if action_type in self.MONEY_ACTIONS else 0.0,
            "is_aggressive_action": 1.0 if action_type in self.AGGRESSIVE_ACTIONS else 0.0,
            "is_passive_action": 1.0 if action_type in self.PASSIVE_ACTIONS else 0.0,
            "is_all_in": 1.0 if action_type in {"all_in", "allin"} else 0.0,
            "actor_seat_norm": seat / max(1, max_seats) if seat > 0 else 0.0,
            "actor_stack_bb_scaled": self.scaled_bb(actor_stack_bb),
            "amount_to_actor_stack": self.capped_ratio(amount, actor_stack, cap=1.0),
            "street_progress": street_progress,
        }

        cat_ids = [street_id, action_type_id, seat_id]
        numeric = [float(feature_map.get(name, 0.0)) for name in self.numeric_feature_names]
        return cat_ids, numeric

    def encode_hand(self, hand: Dict[str, Any]) -> Tuple[List[List[int]], List[List[float]]]:
        actions = hand.get("actions") or []
        metadata = hand.get("metadata") or {}
        bb = self.safe_float(metadata.get("bb"), default=0.0)
        max_seats = self.get_hand_max_seats(hand)
        players_by_seat = self.get_players_by_seat(hand)

        cat_rows: List[List[int]] = []
        num_rows: List[List[float]] = []

        for idx, action in enumerate(actions[: self.max_actions_per_hand]):
            if not isinstance(action, dict):
                continue
            actor_seat = self.safe_int(action.get("actor_seat"), default=0)
            actor = players_by_seat.get(actor_seat) or {}
            actor_stack = self.safe_float(actor.get("starting_stack"), default=0.0)

            cat_ids, numeric = self.encode_action(
                action=action,
                action_index=idx,
                total_actions=len(actions),
                bb=bb,
                max_seats=max_seats,
                actor_stack=actor_stack,
            )
            cat_rows.append(cat_ids)
            num_rows.append(numeric)

        if not cat_rows:
            cat_rows.append([0, 0, 0])
            num_rows.append([0.0 for _ in range(self.numeric_dim)])

        return cat_rows, num_rows

    def encode_chunk(self, chunk: List[Dict[str, Any]], max_hands: int) -> Tuple[List[List[List[int]]], List[List[List[float]]]]:
        chunk_cat: List[List[List[int]]] = []
        chunk_num: List[List[List[float]]] = []

        if not isinstance(chunk, list):
            return [[[0, 0, 0]]], [[[0.0 for _ in range(self.numeric_dim)]]]

        for hand in chunk[:max_hands]:
            if not isinstance(hand, dict):
                continue
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
            "max_table_seats": self.max_table_seats,
            "street_to_id": dict(self.street_to_id),
            "action_type_to_id": dict(self.action_type_to_id),
            "seat_to_id": dict(self.seat_to_id),
            "numeric_feature_names": list(self.numeric_feature_names),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.max_actions_per_hand = int(state.get("max_actions_per_hand", self.max_actions_per_hand))
        self.max_table_seats = int(state.get("max_table_seats", self.max_table_seats))
        self.street_to_id = {str(k): int(v) for k, v in state["street_to_id"].items()}
        self.action_type_to_id = {str(k): int(v) for k, v in state["action_type_to_id"].items()}
        self.seat_to_id = {str(k): int(v) for k, v in state["seat_to_id"].items()}
        self.numeric_feature_names = list(state.get("numeric_feature_names", self.numeric_feature_names))

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "ActionVectorizer":
        obj = cls(
            max_actions_per_hand=int(state.get("max_actions_per_hand", 64)),
            max_table_seats=int(state.get("max_table_seats", 9)),
        )
        obj.load_state_dict(state)
        return obj
