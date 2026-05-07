from __future__ import annotations

import torch
import torch.nn as nn


class ChunkTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        max_len: int = 512,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "feature_dim": feature_dim,
            "max_len": max_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
        }
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        seq_len = min(seq_len, self.config["max_len"])
        input_ids = input_ids[:, :seq_len]
        attention_mask = attention_mask[:, :seq_len]

        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos)

        # PyTorch expects True for padding positions in src_key_padding_mask.
        padding_mask = ~attention_mask
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Masked mean pooling.
        mask_float = attention_mask.float().unsqueeze(-1)
        pooled = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)

        feature_repr = self.feature_mlp(features)
        joined = torch.cat([pooled, feature_repr], dim=-1)
        logits = self.classifier(joined).squeeze(-1)
        return logits
