"""
GRU Model
=========
Gated Recurrent Unit for single-horizon RV prediction.
Processes the lookback window sequentially, then maps the final
hidden state to a scalar forecast via an MLP head.
"""

import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(
        self,
        lookback: int = 1800,
        n_features: int = 50,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        head_units: list[int] = [32],
        max_horizon: int = 0,
    ):
        super().__init__()
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Horizon embedding + input projection (variable prediction horizon)
        self.horizon_embed = None
        self.input_proj = None
        if max_horizon > 0:
            self.input_proj = nn.Linear(n_features, hidden_size)
            self.horizon_embed = nn.Embedding(max_horizon + 1, hidden_size)
            gru_input_size = hidden_size
        else:
            gru_input_size = n_features

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        d = hidden_size * (2 if bidirectional else 1)

        layers = []
        for units in head_units:
            layers.extend([nn.Linear(d, units), nn.ReLU(), nn.Dropout(dropout)])
            d = units
        layers.append(nn.Linear(d, 1))
        self.head = nn.Sequential(*layers)

        print(f"GRU: {num_layers} layers, hidden={hidden_size}, "
              f"bidir={bidirectional}, head={head_units}, "
              f"params={sum(p.numel() for p in self.parameters()):,}")

    @property
    def embedding_dim(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, x: torch.Tensor, horizon=None, return_embedding: bool = False) -> torch.Tensor:
        # Run entirely in fp32 — GRU is sequential so bf16 saves nothing,
        # and cuBLAS bf16 GEMMs fail on the fused B*L internal dimensions
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            if self.input_proj is not None:
                x = self.input_proj(x)
            if horizon is not None and self.horizon_embed is not None:
                h_embed = self.horizon_embed(horizon)
                x = x + h_embed.unsqueeze(1)

            _, h_n = self.gru(x)

            if self.bidirectional:
                h = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                h = h_n[-1]

            pred = self.head(h).squeeze(1)
            if return_embedding:
                return pred, h
            return pred
