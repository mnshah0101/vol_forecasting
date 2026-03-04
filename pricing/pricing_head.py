"""
Learned Pricing Head
====================
Small MLP that maps (backbone embedding + market features) → logit for P(S_T > K).

Trained end-to-end alongside the backbone via BCEWithLogitsLoss.  The market
features mirror the Black-Scholes internals so the head can learn corrections
on top of a BS-like baseline:

    log_moneyness  = log(spot / strike)
    forecasted_var = predicted forward RV (with gradients from RV head)
    sqrt_var       = sqrt(forecasted_var)
    d2             = (log_moneyness - forecasted_var / 2) / sqrt_var
    tte_norm       = log(tte_seconds + 1) / log(901)
"""

import torch
import torch.nn as nn


class PricingHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_units: list[int] = [64, 32],
        dropout: float = 0.1,
    ):
        super().__init__()
        input_dim = embedding_dim + 5  # embedding + 5 market features

        layers = []
        d = input_dim
        for units in hidden_units:
            layers.extend([nn.Linear(d, units), nn.ReLU(), nn.Dropout(dropout)])
            d = units
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

        print(f"PricingHead: input_dim={input_dim}, hidden={hidden_units}, "
              f"dropout={dropout}, "
              f"params={sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self,
        embedding: torch.Tensor,     # (B, D) backbone embedding
        spot: torch.Tensor,          # (B,) current spot price
        strike: torch.Tensor,        # (B,) strike price
        forecasted_var: torch.Tensor, # (B,) predicted forward variance (with grads)
        tte_seconds: torch.Tensor,   # (B,) time to expiry in seconds
    ) -> torch.Tensor:
        """Returns raw logits (B,) — apply sigmoid externally or use BCEWithLogitsLoss."""
        eps = 1e-8

        log_moneyness = torch.log(spot / (strike + eps))
        sqrt_var = torch.sqrt(forecasted_var + eps)
        d2 = (log_moneyness - forecasted_var / 2) / (sqrt_var + eps)
        tte_norm = torch.log(tte_seconds + 1) / torch.log(torch.tensor(901.0, device=embedding.device))

        # Stack market features: (B, 5)
        market = torch.stack([log_moneyness, forecasted_var, sqrt_var, d2, tte_norm], dim=1)

        # Concatenate with embedding: (B, D+5)
        x = torch.cat([embedding, market], dim=1)

        return self.mlp(x).squeeze(1)  # (B,)
