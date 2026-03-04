"""
N-HiTS Model
=============
Neural Hierarchical Interpolation for Time Series forecasting.
Single-horizon RV prediction with identity basis.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


HORIZON_MAP = {
    '1s': 1, '5s': 5, '10s': 10, '30s': 30,
    '1m': 60, '3m': 180, '5m': 300,
    '10m': 600, '15m': 900, '30m': 1800, '1h': 3600,
}


def parse_horizon(h: str) -> tuple[str, int]:
    if h in HORIZON_MAP:
        return h, HORIZON_MAP[h]
    try:
        secs = int(h)
        for label, val in HORIZON_MAP.items():
            if val == secs:
                return label, secs
        return f'{secs}s', secs
    except ValueError:
        raise ValueError(f"Cannot parse horizon '{h}'. Use e.g. '5m', '300', '1h'")


class _IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self._scalar_forecast = (forecast_size == 1)

    def forward(self, theta: torch.Tensor):
        backcast = theta[:, :self.backcast_size]
        if self._scalar_forecast:
            return backcast, theta[:, self.backcast_size:self.backcast_size + 1]
        knots = theta[:, self.backcast_size:].unsqueeze(1)
        if knots.shape[-1] == self.forecast_size:
            forecast = knots
        else:
            forecast = F.interpolate(knots, size=self.forecast_size, mode='linear')
        return backcast, forecast.squeeze(1)


class NHiTSBlock(nn.Module):
    def __init__(self, lookback, n_features, n_theta, pool_kernel, basis,
                 mlp_units=[256, 128], dropout=0.1, activation='ReLU'):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=True)
        pooled_len = int(math.ceil(lookback / pool_kernel))
        input_dim = pooled_len * n_features

        activ = getattr(nn, activation)()
        layers = [nn.Linear(input_dim, mlp_units[0])]
        for i in range(len(mlp_units) - 1):
            layers.append(nn.Linear(mlp_units[i], mlp_units[i + 1]))
            layers.append(activ)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(mlp_units[-1], n_theta))
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, x):
        B = x.shape[0]
        x_pool = self.pool(x.permute(0, 2, 1))
        theta = self.layers(x_pool.reshape(B, -1))
        return self.basis(theta)


class NHiTS(nn.Module):
    """N-HiTS with residual/static separation to avoid full-tensor clones."""

    def __init__(self, lookback=900, n_features=50, forecast_size=1,
                 n_blocks=[1, 1, 1],
                 mlp_units=[[256, 128], [128, 64], [256, 64]],
                 n_pool_kernel_size=[300, 60, 1],
                 n_freq_downsample=[300, 60, 1],
                 dropout=0.1, activation='ReLU',
                 max_horizon=0, d_model=0):
        super().__init__()
        self.lookback = lookback
        self.forecast_size = forecast_size

        # Horizon embedding + input projection (variable prediction horizon)
        self.horizon_embed = None
        self.input_proj = None
        if max_horizon > 0:
            proj_dim = d_model if d_model > 0 else n_features
            self.input_proj = nn.Linear(n_features, proj_dim)
            self.horizon_embed = nn.Embedding(max_horizon + 1, proj_dim)
            block_features = proj_dim
        else:
            block_features = n_features

        self.blocks = nn.ModuleList()
        for s in range(len(n_blocks)):
            n_forecast_knots = max(forecast_size // n_freq_downsample[s], 1)
            n_theta = lookback + n_forecast_knots
            basis = _IdentityBasis(lookback, forecast_size)
            for _ in range(n_blocks[s]):
                self.blocks.append(NHiTSBlock(
                    lookback, block_features, n_theta, n_pool_kernel_size[s],
                    basis, mlp_units[s], dropout, activation,
                ))

        print(f"N-HiTS: {len(self.blocks)} blocks, "
              f"pool={n_pool_kernel_size}, freq_ds={n_freq_downsample}, "
              f"mlp={mlp_units}, forecast_size={forecast_size}")

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError("NHiTS has no single embedding bottleneck (additive block architecture)")

    def forward(self, x, horizon=None, return_embedding: bool = False):
        if return_embedding:
            raise NotImplementedError(
                "NHiTS has no single embedding bottleneck (additive block architecture). "
                "Pricing head is not supported with NHiTS."
            )
        # cuBLAS bfloat16 has matrix dimension limits; run in float32
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            if self.input_proj is not None:
                x = self.input_proj(x)
            if horizon is not None and self.horizon_embed is not None:
                h_embed = self.horizon_embed(horizon)  # (B, D)
                x = x + h_embed.unsqueeze(1)           # broadcast to (B, L, D)

            level = x[:, -1, 0]
            forecast = level.unsqueeze(1).expand(-1, self.forecast_size)

            x_flipped = x.flip(dims=(1,))
            residual = x_flipped[:, :, 0]
            static_feats = x_flipped[:, :, 1:]

            for block in self.blocks:
                x_input = torch.cat([residual.unsqueeze(2), static_feats], dim=2)
                backcast, block_forecast = block(x_input)
                residual = residual - backcast
                forecast = forecast + block_forecast

            return forecast.squeeze(1)
