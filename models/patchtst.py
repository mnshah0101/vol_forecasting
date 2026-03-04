"""
PatchTST Model
===============
Patch Time Series Transformer for single-horizon RV prediction.

Based on Nie et al. (2023) "A Time Series is Worth 64 Words".
Splits the lookback window into non-overlapping (or overlapping) patches,
embeds each patch, adds positional encoding, and processes with a
transformer encoder. The pooled output maps to a scalar forecast.
"""

import math
import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(
        self,
        lookback: int = 1800,
        n_features: int = 50,
        patch_len: int = 60,
        stride: int = 60,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        head_units: list[int] = [32],
        max_horizon: int = 0,
    ):
        super().__init__()
        self.lookback = lookback
        self.patch_len = patch_len
        self.stride = stride

        # Number of patches
        self.n_patches = (lookback - patch_len) // stride + 1

        # Patch embedding: flatten each patch then project
        patch_dim = patch_len * n_features
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.embed_norm = nn.LayerNorm(d_model)

        # Horizon embedding (variable prediction horizon)
        self.horizon_embed = None
        if max_horizon > 0:
            self.horizon_embed = nn.Embedding(max_horizon + 1, d_model)

        # Learned positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

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

        print(f"PatchTST: {self.n_patches} patches (len={patch_len}, stride={stride}), "
              f"{num_layers} layers, d_model={d_model}, nhead={nhead}, "
              f"params={sum(p.numel() for p in self.parameters()):,}")

    @property
    def embedding_dim(self) -> int:
        return self.patch_embed.out_features

    def forward(self, x: torch.Tensor, horizon=None, return_embedding: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, L, F)
            horizon: (B,) int tensor of prediction horizons, or None
            return_embedding: if True, return (pred, embedding) tuple
        Returns:
            (B,) scalar forecast, or (pred, embedding) if return_embedding
        """
        B = x.shape[0]

        # Extract patches: (B, n_patches, patch_len, F)
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, n_patches, F, patch_len)
        patches = patches.permute(0, 1, 3, 2)               # (B, n_patches, patch_len, F)
        patches = patches.reshape(B, self.n_patches, -1)     # (B, n_patches, patch_len * F)

        # Embed patches
        x = self.patch_embed(patches)    # (B, n_patches, d_model)
        x = self.embed_norm(x)
        if horizon is not None and self.horizon_embed is not None:
            h_embed = self.horizon_embed(horizon)  # (B, d_model)
            x = x + h_embed.unsqueeze(1)           # broadcast to (B, n_patches, d_model)
        x = x + self.pos_embed

        # Transformer encoder
        x = self.encoder(x)  # (B, n_patches, d_model)
        x = self.norm(x)

        # Pool: mean over patches
        emb = x.mean(dim=1)  # (B, d_model)

        pred = self.head(emb).squeeze(1)
        if return_embedding:
            return pred, emb
        return pred
