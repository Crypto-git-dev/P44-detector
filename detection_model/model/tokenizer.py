from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .features import extract_event_tokens_from_chunk


class EventTokenizer:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, token_to_id: Dict[str, int] | None = None, max_len: int = 512):
        self.max_len = max_len
        self.token_to_id = token_to_id or {self.PAD: 0, self.UNK: 1}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def fit(self, chunks: List[List[Dict[str, Any]]], min_freq: int = 1, max_vocab: int = 5000) -> "EventTokenizer":
        counter: Counter[str] = Counter()
        for chunk in chunks:
            counter.update(extract_event_tokens_from_chunk(chunk, max_events=self.max_len))
        most_common = [tok for tok, c in counter.most_common(max_vocab) if c >= min_freq]
        self.token_to_id = {self.PAD: 0, self.UNK: 1}
        for tok in most_common:
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.token_to_id)
        return self

    def encode(self, chunk: List[Dict[str, Any]]) -> List[int]:
        tokens = extract_event_tokens_from_chunk(chunk, max_events=self.max_len)
        ids = [self.token_to_id.get(tok, self.token_to_id[self.UNK]) for tok in tokens]
        return ids[: self.max_len]

    def state_dict(self) -> Dict[str, Any]:
        return {"token_to_id": self.token_to_id, "max_len": self.max_len}

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "EventTokenizer":
        return cls(token_to_id=dict(state["token_to_id"]), max_len=int(state.get("max_len", 512)))
