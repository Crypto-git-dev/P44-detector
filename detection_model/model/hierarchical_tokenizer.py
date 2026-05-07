from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional


class HandActionTokenizer:
    """
    Poker44 hierarchical tokenizer.

    Expected input:

        chunk: List[hand_dict]
        hand: Dict with keys like:
            metadata
            players
            streets
            actions
            outcome

        action:
            {
              "action_id": "1",
              "street": "preflop",
              "actor_seat": 2,
              "action_type": "small_blind",
              "amount": 0.008,
              "raise_to": null,
              "call_to": null,
              "normalized_amount_bb": 0.4,
              "pot_before": 0.0,
              "pot_after": 0.008
            }

    Output:

        encode_chunk(chunk) -> List[List[int]]

        one chunk
            -> many hands
            -> each hand is a list of action/event token ids

    Label is NOT handled here.
    Label belongs to the whole chunk/window.
    """

    PAD = "<PAD>"
    UNK = "<UNK>"
    NO_ACTION = "<NO_ACTION>"
    HAND_START = "<HAND_START>"
    HAND_END = "<HAND_END>"

    FORCED_ACTIONS = {
        "small_blind",
        "big_blind",
        "ante",
        "straddle",
        "bring_in",
    }

    VOLUNTARY_ACTIONS = {
        "fold",
        "check",
        "call",
        "bet",
        "raise",
        "all_in",
        "all-in",
        "allin",
    }

    MONEY_ACTIONS = {
        "small_blind",
        "big_blind",
        "ante",
        "straddle",
        "bring_in",
        "call",
        "bet",
        "raise",
        "all_in",
        "all-in",
        "allin",
    }

    def __init__(
        self,
        max_hands: int = 128,
        max_actions_per_hand: int = 64,
    ):
        self.max_hands = int(max_hands)
        self.max_actions_per_hand = int(max_actions_per_hand)

        self.token_to_id: Dict[str, int] = {
            self.PAD: 0,
            self.UNK: 1,
            self.NO_ACTION: 2,
            self.HAND_START: 3,
            self.HAND_END: 4,
        }

        self.id_to_token: Dict[int, str] = {
            idx: token for token, idx in self.token_to_id.items()
        }

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _add_token(self, token: str) -> None:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def fit(
        self,
        chunks: List[List[Dict[str, Any]]],
        min_freq: int = 1,
        max_vocab: int = 20000,
    ) -> "HandActionTokenizer":
        counter: Counter[str] = Counter()

        for chunk in chunks:
            if not isinstance(chunk, list):
                continue

            for hand in chunk:
                if not isinstance(hand, dict):
                    continue

                for token in self.hand_to_tokens(hand):
                    counter[token] += 1

        for token, count in counter.most_common(max_vocab):
            if count >= min_freq:
                self._add_token(token)

        return self

    # ------------------------------------------------------------------
    # Basic normalization helpers
    # ------------------------------------------------------------------

    def normalize_text(self, value: Any, default: str = "unknown") -> str:
        if value is None:
            return default.upper()

        text = str(value).strip().lower()

        if not text:
            return default.upper()

        text = (
            text.replace("-", "_")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("'", "")
            .replace('"', "")
        )

        return text.upper()

    def safe_float(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return default

        try:
            if isinstance(value, str):
                value = value.replace(",", "").replace("€", "").replace("$", "").replace("£", "")
            return float(value)
        except Exception:
            return default

    def safe_int(self, value: Any, default: int = 0) -> int:
        if value is None:
            return default

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

    # ------------------------------------------------------------------
    # Bucket helpers
    # ------------------------------------------------------------------

    def ratio_bucket(self, ratio: float, prefix: str) -> str:
        if ratio < 0:
            return f"{prefix}_NEGATIVE"
        if ratio == 0:
            return f"{prefix}_ZERO"
        if ratio < 0.05:
            return f"{prefix}_LT_005"
        if ratio < 0.10:
            return f"{prefix}_005_010"
        if ratio < 0.25:
            return f"{prefix}_010_025"
        if ratio < 0.33:
            return f"{prefix}_025_033"
        if ratio < 0.50:
            return f"{prefix}_033_050"
        if ratio < 0.66:
            return f"{prefix}_050_066"
        if ratio < 0.75:
            return f"{prefix}_066_075"
        if ratio < 1.00:
            return f"{prefix}_075_100"
        if ratio < 1.25:
            return f"{prefix}_100_125"
        if ratio < 1.50:
            return f"{prefix}_125_150"
        if ratio < 2.00:
            return f"{prefix}_150_200"
        if ratio < 3.00:
            return f"{prefix}_200_300"
        if ratio < 5.00:
            return f"{prefix}_300_500"

        return f"{prefix}_OVER_500"

    def bb_bucket(self, value_bb: float, prefix: str) -> str:
        if value_bb < 0:
            return f"{prefix}_NEGATIVE"
        if value_bb == 0:
            return f"{prefix}_ZERO"
        if value_bb < 0.25:
            return f"{prefix}_LT_025BB"
        if value_bb < 0.50:
            return f"{prefix}_025_050BB"
        if value_bb < 1:
            return f"{prefix}_050_1BB"
        if value_bb < 2:
            return f"{prefix}_1_2BB"
        if value_bb < 3:
            return f"{prefix}_2_3BB"
        if value_bb < 5:
            return f"{prefix}_3_5BB"
        if value_bb < 8:
            return f"{prefix}_5_8BB"
        if value_bb < 12:
            return f"{prefix}_8_12BB"
        if value_bb < 20:
            return f"{prefix}_12_20BB"
        if value_bb < 40:
            return f"{prefix}_20_40BB"
        if value_bb < 80:
            return f"{prefix}_40_80BB"

        return f"{prefix}_OVER_80BB"

    def count_bucket(self, count: int, prefix: str) -> str:
        if count <= 0:
            return f"{prefix}_0"
        if count == 1:
            return f"{prefix}_1"
        if count == 2:
            return f"{prefix}_2"
        if count == 3:
            return f"{prefix}_3"
        if count == 4:
            return f"{prefix}_4"
        if count == 5:
            return f"{prefix}_5"
        if count == 6:
            return f"{prefix}_6"
        if count <= 9:
            return f"{prefix}_7_9"
        if count <= 15:
            return f"{prefix}_10_15"
        if count <= 30:
            return f"{prefix}_16_30"

        return f"{prefix}_OVER_30"

    def actor_bucket(self, action: Dict[str, Any]) -> str:
        seat = self.safe_int(action.get("actor_seat"), default=0)

        if seat <= 0:
            return "SEAT_UNKNOWN"
        if seat <= 9:
            return f"SEAT_{seat}"

        return "SEAT_10_PLUS"

    def action_order_bucket(self, action_index: int, total_actions: int) -> str:
        if total_actions <= 0:
            return "ACTION_ORDER_UNKNOWN"

        if action_index == 0:
            return "ACTION_ORDER_FIRST"

        if action_index == total_actions - 1:
            return "ACTION_ORDER_LAST"

        ratio = action_index / max(1, total_actions - 1)

        if ratio < 0.33:
            return "ACTION_ORDER_EARLY"
        if ratio < 0.66:
            return "ACTION_ORDER_MIDDLE"

        return "ACTION_ORDER_LATE"

    def street_token(self, street: Any) -> str:
        street_name = self.normalize_text(street)

        known = {
            "PREFLOP",
            "FLOP",
            "TURN",
            "RIVER",
            "SHOWDOWN",
        }

        if street_name in known:
            return f"STREET_{street_name}"

        return "STREET_UNKNOWN"

    def action_type_token(self, action_type: Any) -> str:
        action = str(action_type or "unknown").strip().lower().replace("-", "_").replace(" ", "_")

        aliases = {
            "all-in": "all_in",
            "allin": "all_in",
            "small blind": "small_blind",
            "big blind": "big_blind",
            "uncalled bet return": "uncalled_bet_return",
        }

        action = aliases.get(action, action)

        return f"ACTION_{self.normalize_text(action)}"

    # ------------------------------------------------------------------
    # Amount / betting context
    # ------------------------------------------------------------------

    def amount_tokens(self, action: Dict[str, Any]) -> List[str]:
        """
        Convert amount fields into robust betting tokens.

        Handles:
            amount
            normalized_amount_bb
            pot_before
            pot_after
            raise_to
            call_to
        """

        tokens: List[str] = []

        action_type_raw = str(action.get("action_type") or "unknown").strip().lower()
        action_type = action_type_raw.replace("-", "_").replace(" ", "_")

        amount = self.safe_float(action.get("amount"), default=0.0)
        normalized_bb = self.safe_float(action.get("normalized_amount_bb"), default=0.0)
        pot_before = self.safe_float(action.get("pot_before"), default=0.0)
        pot_after = self.safe_float(action.get("pot_after"), default=0.0)

        action_prefix = self.normalize_text(action_type)

        if action_type in {"fold", "check"}:
            tokens.append("AMOUNT_NOT_REQUIRED")
            tokens.append(f"{action_prefix}_NO_MONEY")
            return tokens

        if amount <= 0:
            tokens.append("AMOUNT_ZERO_OR_MISSING")
            tokens.append(f"{action_prefix}_AMOUNT_ZERO_OR_MISSING")
        else:
            tokens.append("HAS_AMOUNT")
            tokens.append(self.bb_bucket(normalized_bb, "AMOUNT_BB"))

            if pot_before > 0:
                amount_to_pot = amount / pot_before
                pot_bucket = self.ratio_bucket(amount_to_pot, "AMOUNT_TO_POT")
                tokens.append(pot_bucket)
                tokens.append(f"{action_prefix}_{pot_bucket}")
            else:
                tokens.append("AMOUNT_NO_POT_CONTEXT")
                tokens.append(f"{action_prefix}_AMOUNT_NO_POT_CONTEXT")

        if pot_before > 0:
            tokens.append(self.bb_bucket(pot_before / max(normalized_bb / amount, 1e-9) if amount > 0 and normalized_bb > 0 else 0.0, "POT_BEFORE_EST_BB"))
        else:
            tokens.append("POT_BEFORE_ZERO_OR_MISSING")

        if pot_after > 0 and pot_before > 0:
            pot_growth = pot_after / pot_before
            tokens.append(self.ratio_bucket(pot_growth, "POT_AFTER_TO_BEFORE"))
        elif pot_after > 0:
            tokens.append("POT_AFTER_EXISTS_NO_BEFORE")
        else:
            tokens.append("POT_AFTER_ZERO_OR_MISSING")

        raise_to = action.get("raise_to")
        call_to = action.get("call_to")

        if self.is_present(raise_to):
            tokens.append("HAS_RAISE_TO")
            tokens.extend(
                self.value_context_tokens(
                    value=raise_to,
                    pot_before=pot_before,
                    normalized_amount_bb=normalized_bb,
                    amount=amount,
                    prefix="RAISE_TO",
                )
            )
        else:
            tokens.append("NO_RAISE_TO")

        if self.is_present(call_to):
            tokens.append("HAS_CALL_TO")
            tokens.extend(
                self.value_context_tokens(
                    value=call_to,
                    pot_before=pot_before,
                    normalized_amount_bb=normalized_bb,
                    amount=amount,
                    prefix="CALL_TO",
                )
            )
        else:
            tokens.append("NO_CALL_TO")

        return tokens

    def value_context_tokens(
        self,
        value: Any,
        pot_before: float,
        normalized_amount_bb: float,
        amount: float,
        prefix: str,
    ) -> List[str]:
        tokens: List[str] = []

        value_f = self.safe_float(value, default=0.0)

        if value_f <= 0:
            return [f"{prefix}_ZERO_OR_INVALID"]

        if pot_before > 0:
            ratio = value_f / pot_before
            tokens.append(self.ratio_bucket(ratio, f"{prefix}_TO_POT"))
        else:
            tokens.append(f"{prefix}_NO_POT_CONTEXT")

        # Estimate value in BB if normalized_amount_bb is available for amount.
        if amount > 0 and normalized_amount_bb > 0:
            bb_size = amount / normalized_amount_bb
            value_bb = value_f / max(bb_size, 1e-9)
            tokens.append(self.bb_bucket(value_bb, f"{prefix}_BB"))
        else:
            tokens.append(f"{prefix}_NO_BB_CONTEXT")

        return tokens

    # ------------------------------------------------------------------
    # Hand-level context
    # ------------------------------------------------------------------

    def metadata_tokens(self, hand: Dict[str, Any]) -> List[str]:
        tokens: List[str] = []

        metadata = hand.get("metadata") or {}

        game_type = metadata.get("game_type")
        limit_type = metadata.get("limit_type")
        max_seats = metadata.get("max_seats")
        hand_ended = metadata.get("hand_ended_on_street")
        button_seat = metadata.get("button_seat")
        hero_seat = metadata.get("hero_seat")
        sb = self.safe_float(metadata.get("sb"), default=0.0)
        bb = self.safe_float(metadata.get("bb"), default=0.0)
        ante = self.safe_float(metadata.get("ante"), default=0.0)

        if game_type:
            tokens.append(f"GAME_{self.normalize_text(game_type)}")

        if limit_type:
            tokens.append(f"LIMIT_{self.normalize_text(limit_type)}")

        if max_seats is not None:
            max_seats_i = self.safe_int(max_seats)
            if max_seats_i <= 0:
                tokens.append("MAX_SEATS_UNKNOWN")
            elif max_seats_i <= 10:
                tokens.append(f"MAX_SEATS_{max_seats_i}")
            else:
                tokens.append("MAX_SEATS_10_PLUS")
        else:
            tokens.append("MAX_SEATS_UNKNOWN")

        if hand_ended:
            tokens.append(f"ENDED_{self.normalize_text(hand_ended)}")
        else:
            tokens.append("ENDED_UNKNOWN")

        if button_seat is not None:
            button_i = self.safe_int(button_seat)
            if button_i > 0 and button_i <= 10:
                tokens.append(f"BUTTON_SEAT_{button_i}")
            else:
                tokens.append("BUTTON_SEAT_UNKNOWN")
        else:
            tokens.append("BUTTON_SEAT_UNKNOWN")

        if hero_seat is not None:
            tokens.append("HAS_HERO_SEAT")
        else:
            tokens.append("NO_HERO_SEAT")

        if bb > 0 and sb > 0:
            tokens.append(self.ratio_bucket(sb / bb, "SB_TO_BB"))
        else:
            tokens.append("BLINDS_UNKNOWN")

        if ante > 0 and bb > 0:
            tokens.append(self.ratio_bucket(ante / bb, "ANTE_TO_BB"))
        else:
            tokens.append("NO_ANTE_OR_UNKNOWN")

        return tokens

    def player_tokens(self, hand: Dict[str, Any]) -> List[str]:
        """
        Tokenize full per-player state.

        This does NOT only create extracted summary features like avg/min/max stack.
        Instead, it emits seat-level player state tokens:

            PLAYER_SEAT_1_PRESENT
            PLAYER_SEAT_1_STACK_BB_200_300BB
            PLAYER_SEAT_1_HOLE_CARDS_HIDDEN
            PLAYER_SEAT_1_SHOWED_FALSE
            PLAYER_SEAT_1_IS_HERO
            PLAYER_SEAT_1_IS_BUTTON

        Notes:
            - We do not tokenize raw player_uid because it can cause memorization.
            - We do not tokenize exact raw stack floats; we bucket stack size in BB.
            - We keep player order/seat state explicit.
        """

        tokens: List[str] = []

        players = hand.get("players") or []
        metadata = hand.get("metadata") or {}

        bb = self.safe_float(metadata.get("bb"), default=0.0)
        max_seats = self.safe_int(metadata.get("max_seats"), default=0)
        hero_seat = self.safe_int(metadata.get("hero_seat"), default=0)
        button_seat = self.safe_int(metadata.get("button_seat"), default=0)

        tokens.append(self.count_bucket(len(players), "PLAYER_COUNT"))

        # Build lookup by seat.
        players_by_seat: Dict[int, Dict[str, Any]] = {}

        for player in players:
            if not isinstance(player, dict):
                continue

            seat = self.safe_int(player.get("seat"), default=0)

            if seat > 0:
                players_by_seat[seat] = player

        # If max_seats is missing, infer from visible seats.
        if max_seats <= 0:
            max_seats = max(players_by_seat.keys(), default=0)

        # Cap to avoid huge vocab for strange data.
        max_seats = min(max_seats, 10)

        if max_seats <= 0:
            tokens.append("PLAYER_SEATS_UNKNOWN")
            return tokens

        for seat in range(1, max_seats + 1):
            seat_prefix = f"PLAYER_SEAT_{seat}"
            player = players_by_seat.get(seat)

            if player is None:
                tokens.append(f"{seat_prefix}_EMPTY")
                continue

            tokens.append(f"{seat_prefix}_PRESENT")

            # Stack state.
            starting_stack = self.safe_float(player.get("starting_stack"), default=0.0)

            if starting_stack > 0 and bb > 0:
                stack_bb = starting_stack / bb
                tokens.append(self.bb_bucket(stack_bb, f"{seat_prefix}_STACK"))
            elif starting_stack > 0:
                tokens.append(f"{seat_prefix}_STACK_EXISTS_NO_BB_CONTEXT")
            else:
                tokens.append(f"{seat_prefix}_STACK_UNKNOWN")

            # Hole cards visibility state.
            hole_cards = player.get("hole_cards")

            if hole_cards:
                tokens.append(f"{seat_prefix}_HOLE_CARDS_VISIBLE")

                if isinstance(hole_cards, list):
                    tokens.append(self.count_bucket(len(hole_cards), f"{seat_prefix}_HOLE_CARD_COUNT"))
                else:
                    tokens.append(f"{seat_prefix}_HOLE_CARDS_NON_LIST")
            else:
                tokens.append(f"{seat_prefix}_HOLE_CARDS_HIDDEN")

            # Showdown/show state.
            if bool(player.get("showed_hand")):
                tokens.append(f"{seat_prefix}_SHOWED_TRUE")
            else:
                tokens.append(f"{seat_prefix}_SHOWED_FALSE")

            # Hero/button role state.
            if hero_seat == seat:
                tokens.append(f"{seat_prefix}_IS_HERO")
            else:
                tokens.append(f"{seat_prefix}_NOT_HERO")

            if button_seat == seat:
                tokens.append(f"{seat_prefix}_IS_BUTTON")
            else:
                tokens.append(f"{seat_prefix}_NOT_BUTTON")

        return tokens

    def street_summary_tokens(self, hand: Dict[str, Any]) -> List[str]:
        tokens: List[str] = []

        streets = hand.get("streets") or []

        if not streets:
            tokens.append("NO_BOARD_STREETS")
            return tokens

        seen_streets = set()

        for street in streets:
            if not isinstance(street, dict):
                continue

            street_name = self.normalize_text(street.get("street"))
            seen_streets.add(street_name)

            board_cards = street.get("board_cards") or []
            tokens.append(f"HAS_BOARD_{street_name}")
            tokens.append(self.count_bucket(len(board_cards), f"BOARD_CARDS_{street_name}"))

        for name in ("FLOP", "TURN", "RIVER"):
            if name not in seen_streets:
                tokens.append(f"NO_BOARD_{name}")

        return tokens

    def outcome_tokens(self, hand: Dict[str, Any]) -> List[str]:
        tokens: List[str] = []

        outcome = hand.get("outcome") or {}
        metadata = hand.get("metadata") or {}

        bb = self.safe_float(metadata.get("bb"), default=0.0)
        total_pot = self.safe_float(outcome.get("total_pot"), default=0.0)
        rake = self.safe_float(outcome.get("rake"), default=0.0)

        if outcome.get("showdown"):
            tokens.append("OUTCOME_SHOWDOWN")
        else:
            tokens.append("OUTCOME_NO_SHOWDOWN")

        reason = outcome.get("result_reason")
        if reason:
            tokens.append(f"RESULT_{self.normalize_text(reason)}")
        else:
            tokens.append("RESULT_UNKNOWN")

        winners = outcome.get("winners") or []
        payouts = outcome.get("payouts") or {}

        tokens.append(self.count_bucket(len(winners), "WINNER_COUNT"))

        if isinstance(payouts, dict):
            tokens.append(self.count_bucket(len(payouts), "PAYOUT_COUNT"))
        else:
            tokens.append("PAYOUT_COUNT_UNKNOWN")

        if total_pot > 0 and bb > 0:
            tokens.append(self.bb_bucket(total_pot / bb, "TOTAL_POT"))
        else:
            tokens.append("TOTAL_POT_UNKNOWN")

        if rake > 0 and total_pot > 0:
            tokens.append(self.ratio_bucket(rake / total_pot, "RAKE_TO_POT"))
        else:
            tokens.append("NO_RAKE_OR_UNKNOWN")

        return tokens

    # ------------------------------------------------------------------
    # Action tokenization
    # ------------------------------------------------------------------

    def action_tokens(
        self,
        action: Dict[str, Any],
        action_index: int,
        total_actions: int,
    ) -> List[str]:
        tokens: List[str] = []

        raw_action_type = str(action.get("action_type") or "unknown").strip().lower()
        action_type = raw_action_type.replace("-", "_").replace(" ", "_")

        tokens.append(self.street_token(action.get("street")))
        tokens.append(self.action_type_token(action_type))
        tokens.append(self.actor_bucket(action))
        tokens.append(self.action_order_bucket(action_index, total_actions))

        if action_type in self.FORCED_ACTIONS:
            tokens.append("FORCED_ACTION")
            tokens.append(f"FORCED_{self.normalize_text(action_type)}")

            # Keep forced amount context, but mark it separately.
            for token in self.amount_tokens(action):
                tokens.append(f"FORCED_{token}")

            return tokens

        if action_type in self.VOLUNTARY_ACTIONS:
            tokens.append("VOLUNTARY_ACTION")
        else:
            tokens.append("OTHER_ACTION")

        if action_type in {"fold", "check"}:
            tokens.append("NON_MONEY_DECISION")
        elif action_type in self.MONEY_ACTIONS:
            tokens.append("MONEY_DECISION")
        else:
            tokens.append("UNKNOWN_MONEY_CONTEXT")

        amount_tokens = self.amount_tokens(action)
        tokens.extend(amount_tokens)

        # Action-specific amount tokens help separate:
        # BET_AMOUNT_TO_POT_050_066 vs CALL_AMOUNT_TO_POT_050_066
        action_prefix = self.normalize_text(action_type)
        for token in amount_tokens:
            if (
                token.startswith("AMOUNT_TO_POT")
                or token.startswith("AMOUNT_BB")
                or token.startswith("RAISE_TO")
                or token.startswith("CALL_TO")
            ):
                tokens.append(f"{action_prefix}_{token}")

        return tokens

    # ------------------------------------------------------------------
    # Public encoding methods
    # ------------------------------------------------------------------

    def hand_to_tokens(self, hand: Dict[str, Any]) -> List[str]:
        tokens: List[str] = [self.HAND_START]

        if not isinstance(hand, dict):
            return [self.HAND_START, "INVALID_HAND", self.HAND_END]

        tokens.extend(self.metadata_tokens(hand))
        tokens.extend(self.player_tokens(hand))
        tokens.extend(self.street_summary_tokens(hand))
        tokens.extend(self.outcome_tokens(hand))

        actions = hand.get("actions") or []
        tokens.append(self.count_bucket(len(actions), "ACTION_COUNT"))

        if not actions:
            tokens.append(self.NO_ACTION)
        else:
            for action_index, action in enumerate(actions):
                if not isinstance(action, dict):
                    tokens.append("INVALID_ACTION")
                    continue

                tokens.extend(
                    self.action_tokens(
                        action=action,
                        action_index=action_index,
                        total_actions=len(actions),
                    )
                )

        tokens.append(self.HAND_END)

        return tokens

    def encode_hand(self, hand: Dict[str, Any]) -> List[int]:
        tokens = self.hand_to_tokens(hand)

        if(len(tokens) > self.max_actions_per_hand):
            print("Limited token length for hand. Original tokens length:", len(tokens))

        # Keep HAND_END if truncating.
        if len(tokens) > self.max_actions_per_hand:
            tokens = tokens[: max(1, self.max_actions_per_hand - 1)] + [self.HAND_END]

        ids = [
            self.token_to_id.get(token, self.unk_id)
            for token in tokens
        ]

        if not ids:
            ids = [self.token_to_id[self.NO_ACTION]]

        return ids

    def encode_chunk(self, chunk: List[Dict[str, Any]]) -> List[List[int]]:
        encoded: List[List[int]] = []

        if not isinstance(chunk, list):
            return [[self.token_to_id[self.NO_ACTION]]]

        for hand in chunk[: self.max_hands]:
            encoded.append(self.encode_hand(hand))

        if not encoded:
            encoded.append([self.token_to_id[self.NO_ACTION]])

        return encoded

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "max_hands": self.max_hands,
            "max_actions_per_hand": self.max_actions_per_hand,
            "token_to_id": dict(self.token_to_id),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.max_hands = int(state.get("max_hands", self.max_hands))
        self.max_actions_per_hand = int(
            state.get("max_actions_per_hand", self.max_actions_per_hand)
        )

        token_to_id = state.get("token_to_id")
        if not isinstance(token_to_id, dict):
            raise ValueError("Tokenizer state missing token_to_id dict.")

        self.token_to_id = {
            str(token): int(idx)
            for token, idx in token_to_id.items()
        }

        self.id_to_token = {
            int(idx): str(token)
            for token, idx in self.token_to_id.items()
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "HandActionTokenizer":
        obj = cls(
            max_hands=int(state.get("max_hands", 128)),
            max_actions_per_hand=int(state.get("max_actions_per_hand", 64)),
        )
        obj.load_state_dict(state)
        return obj