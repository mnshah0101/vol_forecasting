"""
Transformer Model
=================
Encoder-only transformer for single-horizon RV prediction.
Projects input features to d_model, adds learned positional encoding,
then passes through standard transformer encoder layers.
Pools the output and maps to a scalar forecast.
"""

import math
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        lookback: int = 1800,
        n_features: int = 50,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        pool: str = 'mean',
        head_units: list[int] = [32],
        max_horizon: int = 0,
    ):
        super().__init__()
        self.lookback = lookback
        self.pool_mode = pool

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # Horizon embedding (variable prediction horizon)
        self.horizon_embed = None
        if max_horizon > 0:
            self.horizon_embed = nn.Embedding(max_horizon + 1, d_model)

        # Learned positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # MLP head
        d = d_model
        layers = []
        for units in head_units:
            layers.extend([nn.Linear(d, units), nn.GELU(), nn.Dropout(dropout)])
            d = units
        layers.append(nn.Linear(d, 1))
        self.head = nn.Sequential(*layers)

        print(f"Transformer: {num_layers} layers, d_model={d_model}, "
              f"nhead={nhead}, ff={dim_feedforward}, pool={pool}, "
              f"params={sum(p.numel() for p in self.parameters()):,}")

    @property
    def embedding_dim(self) -> int:
        return self.input_proj.out_features

    def forward(self, x: torch.Tensor, horizon=None, return_embedding: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, L, F)
            horizon: (B,) int tensor of prediction horizons, or None
            return_embedding: if True, return (pred, embedding) tuple
        Returns:
            (B,) scalar forecast, or (pred, embedding) if return_embedding
        """
        # Project features to d_model
        x = self.input_proj(x)  # (B, L, d_model)
        if horizon is not None and self.horizon_embed is not None:
            h_embed = self.horizon_embed(horizon)  # (B, d_model)
            x = x + h_embed.unsqueeze(1)           # broadcast to (B, L, d_model)
        x = x + self.pos_embed

        # Transformer encoder
        x = self.encoder(x)  # (B, L, d_model)
        x = self.norm(x)

        # Pool sequence dimension
        if self.pool_mode == 'last':
            emb = x[:, -1, :]  # (B, d_model)
        elif self.pool_mode == 'first':
            emb = x[:, 0, :]
        else:  # mean
            emb = x.mean(dim=1)  # (B, d_model)

        pred = self.head(emb).squeeze(1)
        if return_embedding:
            return pred, emb
        return pred
