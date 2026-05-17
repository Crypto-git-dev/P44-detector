from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class HierarchicalChunkClassifier(nn.Module):
    """
    First-model hierarchical architecture restored as the neural backbone:

        action categorical embeddings + numeric projection
            -> Transformer over actions inside each hand using a CLS token
            -> GRU over hands in the chunk
            -> attention pooling into one chunk embedding

    The production/final classifier is XGBoost trained on:
        concat(chunk_embedding, engineered_chunk_features)

    The small torch head below is only an auxiliary training probe, so the encoder
    can learn useful embeddings before XGBoost is fitted by train_hierarchical.py.
    """

    def __init__(
        self,
        street_vocab_size: int,
        action_type_vocab_size: int,
        seat_vocab_size: int,
        numeric_dim: int,
        feature_dim: int,
        max_actions_per_hand: int = 64,
        max_hands: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        chunk_layers: int = 1,
        gru_layers: int | None = None,
        bidirectional_gru: bool = True,
        dropout: float = 0.30,
        pad_id: int = 0,
        **_: object,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads. Got d_model={d_model}, n_heads={n_heads}")
        if gru_layers is None:
            gru_layers = chunk_layers

        self.pad_id = int(pad_id)
        self.street_vocab_size = int(street_vocab_size)
        self.action_type_vocab_size = int(action_type_vocab_size)
        self.seat_vocab_size = int(seat_vocab_size)
        self.numeric_dim = int(numeric_dim)
        self.feature_dim = int(feature_dim)
        self.max_actions_per_hand = int(max_actions_per_hand)
        self.max_hands = int(max_hands)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.chunk_layers = int(chunk_layers)
        self.gru_layers = int(gru_layers)
        self.bidirectional_gru = bool(bidirectional_gru)
        self.dropout_p = float(dropout)

        self.config: Dict[str, int | float | bool] = {
            "street_vocab_size": self.street_vocab_size,
            "action_type_vocab_size": self.action_type_vocab_size,
            "seat_vocab_size": self.seat_vocab_size,
            "numeric_dim": self.numeric_dim,
            "feature_dim": self.feature_dim,
            "max_actions_per_hand": self.max_actions_per_hand,
            "max_hands": self.max_hands,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "chunk_layers": self.chunk_layers,
            "gru_layers": self.gru_layers,
            "bidirectional_gru": self.bidirectional_gru,
            "dropout": self.dropout_p,
            "pad_id": self.pad_id,
        }

        self.street_embedding = nn.Embedding(street_vocab_size, d_model, padding_idx=pad_id)
        self.action_type_embedding = nn.Embedding(action_type_vocab_size, d_model, padding_idx=pad_id)
        self.seat_embedding = nn.Embedding(seat_vocab_size, d_model, padding_idx=pad_id)

        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.action_position_embedding = nn.Embedding(max_actions_per_hand, d_model)
        self.action_input_norm = nn.LayerNorm(d_model)
        self.action_input_dropout = nn.Dropout(dropout)

        self.hand_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.hand_cls_token, mean=0.0, std=0.02)

        hand_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.hand_encoder = nn.TransformerEncoder(hand_encoder_layer, num_layers=n_layers)
        self.hand_norm = nn.LayerNorm(d_model)

        if self.bidirectional_gru:
            gru_hidden_size = max(1, d_model // 2)
        else:
            gru_hidden_size = d_model

        self.hand_gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden_size,
            num_layers=self.gru_layers,
            dropout=dropout if self.gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=self.bidirectional_gru,
        )

        gru_output_dim = gru_hidden_size * (2 if self.bidirectional_gru else 1)
        self.gru_projection = nn.Linear(gru_output_dim, d_model) if gru_output_dim != d_model else nn.Identity()
        self.gru_norm = nn.LayerNorm(d_model)

        self.hand_attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        if feature_dim > 0:
            self.feature_projection = nn.Sequential(
                nn.Linear(feature_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            probe_input_dim = d_model * 2
        else:
            self.feature_projection = None
            probe_input_dim = d_model

        # Auxiliary probe used only while training the torch backbone. The final
        # saved classifier is the XGBoost head in the artifact.
        self.aux_classifier = nn.Linear(probe_input_dim, 1)

    def embed_actions(
        self,
        flat_action_cat: torch.Tensor,
        flat_action_num: torch.Tensor,
        flat_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        street_ids = flat_action_cat[:, :, 0].clamp(min=0, max=self.street_vocab_size - 1)
        action_type_ids = flat_action_cat[:, :, 1].clamp(min=0, max=self.action_type_vocab_size - 1)
        seat_ids = flat_action_cat[:, :, 2].clamp(min=0, max=self.seat_vocab_size - 1)

        street_emb = self.street_embedding(street_ids)
        action_type_emb = self.action_type_embedding(action_type_ids)
        seat_emb = self.seat_embedding(seat_ids)
        numeric_emb = self.numeric_projection(flat_action_num)

        positions = torch.arange(flat_action_cat.shape[1], device=flat_action_cat.device).unsqueeze(0)
        positions = positions.clamp(max=self.action_position_embedding.num_embeddings - 1)
        pos_emb = self.action_position_embedding(positions)

        x = street_emb + action_type_emb + seat_emb + numeric_emb + pos_emb
        x = self.action_input_norm(x)
        x = self.action_input_dropout(x)
        return x

    def encode_hands(
        self,
        action_cat: torch.Tensor,
        action_num: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_hands, max_actions, cat_dim = action_cat.shape
        if cat_dim != 3:
            raise ValueError(f"Expected action_cat last dim = 3, got {cat_dim}")
        if action_num.shape[-1] != self.numeric_dim:
            raise ValueError(f"Expected action_num last dim = {self.numeric_dim}, got {action_num.shape[-1]}")

        flat_action_cat = action_cat.reshape(batch_size * max_hands, max_actions, 3)
        flat_action_num = action_num.reshape(batch_size * max_hands, max_actions, self.numeric_dim)
        flat_action_mask = action_mask.reshape(batch_size * max_hands, max_actions)

        empty_rows = ~flat_action_mask.any(dim=1)
        if empty_rows.any():
            flat_action_cat = flat_action_cat.clone()
            flat_action_num = flat_action_num.clone()
            flat_action_mask = flat_action_mask.clone()
            flat_action_cat[empty_rows, 0, :] = self.pad_id
            flat_action_num[empty_rows, 0, :] = 0.0
            flat_action_mask[empty_rows, 0] = True

        x = self.embed_actions(flat_action_cat, flat_action_num, flat_action_mask)

        cls_token = self.hand_cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        cls_mask = torch.ones((flat_action_mask.size(0), 1), dtype=torch.bool, device=flat_action_mask.device)
        extended_action_mask = torch.cat([cls_mask, flat_action_mask], dim=1)

        encoded_actions = self.hand_encoder(x, src_key_padding_mask=~extended_action_mask)
        hand_embeddings = self.hand_norm(encoded_actions[:, 0, :])
        return hand_embeddings.reshape(batch_size, max_hands, self.d_model)

    def encode_chunk(self, hand_embeddings: torch.Tensor, hand_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_hands, _ = hand_embeddings.shape
        empty_chunks = ~hand_mask.any(dim=1)
        if empty_chunks.any():
            hand_mask = hand_mask.clone()
            hand_mask[empty_chunks, 0] = True

        hand_lengths = hand_mask.long().sum(dim=1).clamp(min=1)
        packed = pack_padded_sequence(
            hand_embeddings,
            lengths=hand_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.hand_gru(packed)
        gru_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=max_hands)

        contextual_hands = self.gru_projection(gru_outputs)
        contextual_hands = self.gru_norm(contextual_hands)

        attention_logits = self.hand_attention(contextual_hands).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~hand_mask, -1e9)
        attention_weights = torch.softmax(attention_logits, dim=1) * hand_mask.float()
        attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        return torch.sum(contextual_hands * attention_weights.unsqueeze(-1), dim=1)

    def extract_chunk_embedding(
        self,
        action_cat: torch.Tensor,
        action_num: torch.Tensor,
        action_mask: torch.Tensor,
        hand_mask: torch.Tensor,
    ) -> torch.Tensor:
        hand_embeddings = self.encode_hands(
            action_cat=action_cat,
            action_num=action_num,
            action_mask=action_mask,
        )
        return self.encode_chunk(hand_embeddings=hand_embeddings, hand_mask=hand_mask)

    def forward(
        self,
        action_cat: torch.Tensor,
        action_num: torch.Tensor,
        action_mask: torch.Tensor,
        hand_mask: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        chunk_embedding = self.extract_chunk_embedding(
            action_cat=action_cat,
            action_num=action_num,
            action_mask=action_mask,
            hand_mask=hand_mask,
        )

        if self.feature_projection is not None:
            feature_embedding = self.feature_projection(features)
            probe_input = torch.cat([chunk_embedding, feature_embedding], dim=-1)
        else:
            probe_input = chunk_embedding

        return self.aux_classifier(probe_input).squeeze(-1)
