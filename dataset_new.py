"""
GPU-Resident Dataset & Batching (aligned with pipeline_v2)
===========================================================
All data lives on GPU. Batching is just index slicing — zero CPU involvement.

Features : all 257 columns from pipeline_v2 (Tiers 1-3 + HAR + cyclical + log transforms).
Targets  : computed on-the-fly via prefix sums of per-second RV,
           giving second-granular horizon resolution (0–900s).

At 1s aggregation, RV_t = r_t² (single squared return per second),
so cumsum(RV) = cumsum(r²) — no separate log-return column needed.
"""

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from vol_forecasting.data_pull.pipeline_new import PipelineConfig, get_feature_columns


# ---------------------------------------------------------------------------
# Constants — must match pipeline_v2 output
# ---------------------------------------------------------------------------

DEFAULT_INSTRUMENTS = ["s", "p"]
MEASURES = ["rv", "rsv_pos", "rsv_neg", "bpv"]


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def get_feature_cols(cfg: PipelineConfig | None = None) -> list[str]:
    """Full ordered list of model feature columns from pipeline_v2.

    Returns all 257 features: HAR rolling windows, basis, spread, jump,
    signed returns, drawdown, RV term structure, microprice, VPIN, OBI,
    MODWT, cross-correlation, vol-of-vol, TSRV, RK, RQ, EWMA, Roll,
    Parkinson, roughness, spectral bands, Fourier magnitudes, calendar,
    autocorrelation, skew/kurt, cyclical time, and log transforms.
    """
    if cfg is None:
        cfg = PipelineConfig()
    return get_feature_columns(cfg)


def get_supporting_cols(
    instruments: list[str] | None = None,
) -> list[str]:
    """Supporting columns kept in the dataframe but not used as features."""
    instruments = instruments or DEFAULT_INSTRUMENTS
    cols = ["received_time"]
    for inst in instruments:
        cols.extend([
            f"{inst}_rv", f"{inst}_rsv_pos", f"{inst}_rsv_neg", f"{inst}_bpv",
            f"{inst}_mid", f"{inst}_seas_factor",
        ])
    return cols


# ---------------------------------------------------------------------------
# Prefix-sum builder (runs once on CPU before GPU transfer)
# ---------------------------------------------------------------------------

def precompute_prefix_sums(
    df: pl.DataFrame,
    instruments: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Cumulative sum of per-second RV → prefix array of length T+1.

    prefix[0] = 0
    prefix[k] = sum(rv[0] .. rv[k-1])

    Forward RV over [a, b) = prefix[b] - prefix[a].
    """
    instruments = instruments or DEFAULT_INSTRUMENTS
    prefix_rvs: dict[str, np.ndarray] = {}
    for inst in instruments:
        rv = df.get_column(f"{inst}_rv").to_numpy().astype(np.float64)
        rv = np.nan_to_num(rv, nan=0.0)
        prefix = np.zeros(len(rv) + 1, dtype=np.float64)
        np.cumsum(rv, out=prefix[1:])
        prefix_rvs[inst] = prefix.astype(np.float32)
    return prefix_rvs


def build_prefix_stack(
    prefix_rvs: dict[str, np.ndarray],
    instruments: list[str] | None = None,
) -> np.ndarray:
    """Stack per-instrument prefix arrays → (P, T+1) array."""
    instruments = instruments or DEFAULT_INSTRUMENTS
    return np.stack([prefix_rvs[p] for p in instruments], axis=0)


# ---------------------------------------------------------------------------
# GPU Dataset
# ---------------------------------------------------------------------------

class GPUDataset:
    """All data lives on GPU.  Batching is just index slicing.

    Features come from pipeline_v2's 257 precomputed columns.
    Targets are computed on-the-fly from prefix sums at *second* granularity:

        target = log( prefix[t + h + 1] - prefix[t + 1] + eps )

    where h is sampled uniformly from [min_horizon, max_horizon] (seconds).
    """

    def __init__(
        self,
        features: np.ndarray,           # (T, F) float32
        prefix_stack: np.ndarray,       # (P, T+1) float32
        lookback: int,
        stride: int,
        device: torch.device,
        variable_horizon: bool = True,
        min_horizon: int = 1,
        max_horizon: int = 900,
        prod_idx: int = 0,              # default 0 = 's' (spot)
        log_target: bool = True,        # False for raw_mse / raw_huber
        mid_prices: np.ndarray | None = None,  # (T,) float32 — spot mid prices
    ):
        self.lookback = lookback
        self.device = device
        self.variable_horizon = variable_horizon
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.prod_idx = prod_idx
        self.log_target = log_target
        self.target_clip: float | None = None
        self.eps = 1e-12

        # Move to GPU once
        self.features = torch.as_tensor(features, dtype=torch.float32, device=device)
        self.prefix_stack = torch.as_tensor(prefix_stack, dtype=torch.float32, device=device)

        # Mid-prices for pricing head (optional)
        if mid_prices is not None:
            self.mid_prices = torch.as_tensor(mid_prices, dtype=torch.float32, device=device)
        else:
            self.mid_prices = None

        # Valid indices: need lookback history AND enough forward data
        all_idx = torch.arange(lookback, len(features), device=device)
        max_valid_t = self.prefix_stack.shape[1] - max_horizon - 2
        valid = all_idx[all_idx <= max_valid_t]
        self._all_valid_indices = valid          # full pool for resampling
        self.indices = valid[::stride]           # deterministic initial subset

        self._compute_target_stats()

    # ------------------------------------------------------------------ #
    def resample(self):
        """Randomly pick len(self.indices) positions from the full valid pool.

        Target stats (tgt_mean, tgt_std) are NOT recomputed — they stay
        fixed from the initial deterministic subset.
        """
        n = len(self.indices)
        perm = torch.randperm(len(self._all_valid_indices), device=self.device)[:n]
        self.indices = self._all_valid_indices[perm]

    # ------------------------------------------------------------------ #
    def _compute_target_stats(self):
        """Compute target mean/std at midpoint horizon for normalisation."""
        mid_h = (self.min_horizon + self.max_horizon) // 2
        t = self.indices
        rv_forward = (
            self.prefix_stack[self.prod_idx, t + mid_h + 1]
            - self.prefix_stack[self.prod_idx, t + 1]
        )
        if self.log_target:
            targets = torch.log(rv_forward + self.eps)
        else:
            targets = rv_forward
        self.tgt_mean = targets.mean().item()
        self.tgt_std = targets.std().item() + 1e-8

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.indices)

    # ------------------------------------------------------------------ #
    def get_batch(self, batch_idx: torch.Tensor):
        """Pure GPU index gather — no Python loops, no CPU involvement.

        Returns
        -------
        x     : (B, L, F)
        pw    : (B, P, L)   prefix window sums
        ps    : (B, P, 1)   prefix start values
        y     : (B,)
        h     : (B,) int    sampled horizons (seconds)
        t_idx : (B,) int    raw time indices (for pricing head mid-price lookups)
        """
        t = self.indices[batch_idx]                          # (B,)
        starts = t - self.lookback                           # (B,)
        L = self.lookback
        B = t.shape[0]

        # (B, L) offset matrix
        offsets = (
            starts.unsqueeze(1)
            + torch.arange(L, device=self.device).unsqueeze(0)
        )  # (B, L)

        # Gather features → (B, L, F)
        x = self.features[offsets]

        # Prefix window: positions [start+1 .. t] for cumulative log-RV
        pw_offsets = offsets + 1                              # (B, L)
        pw = self.prefix_stack[:, pw_offsets.long()]          # (P, B, L)
        pw = pw.permute(1, 0, 2)                             # (B, P, L)

        ps = self.prefix_stack[:, starts.long()]              # (P, B)
        ps = ps.permute(1, 0).unsqueeze(2)                   # (B, P, 1)

        # Sample random horizon per sample (seconds)
        if self.variable_horizon:
            h = torch.randint(
                self.min_horizon, self.max_horizon + 1, (B,), device=self.device
            )
        else:
            # Fixed horizon = max_horizon for all samples
            h = torch.full((B,), self.max_horizon, device=self.device, dtype=torch.long)

        # target: log or raw depending on loss mode
        rv_forward = (
            self.prefix_stack[self.prod_idx, (t + h + 1).long()]
            - self.prefix_stack[self.prod_idx, (t + 1).long()]
        )
        if self.log_target:
            y = torch.log(rv_forward + self.eps)
        else:
            y = rv_forward
        y = (y - self.tgt_mean) / self.tgt_std
        if self.target_clip is not None:
            y = y.clamp(-self.target_clip, self.target_clip)

        return x, pw, ps, y, h, t


# ---------------------------------------------------------------------------
# Batch Iterator
# ---------------------------------------------------------------------------

class BatchIterator:
    """GPU-side batch iterator with optional shuffling.  Replaces DataLoader."""

    def __init__(
        self,
        dataset: GPUDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n = len(dataset)

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.n, device=self.dataset.device)
        else:
            perm = torch.arange(self.n, device=self.dataset.device)

        for i in range(0, self.n, self.batch_size):
            batch_idx = perm[i : i + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            yield self.dataset.get_batch(batch_idx)


# ---------------------------------------------------------------------------
# Batch Augmenter
# ---------------------------------------------------------------------------

class BatchAugmenter(nn.Module):
    """Batched GPU augmentation: static positional features + cumulative
    log-RV from prefix sums over the lookback window.

    Parameters
    ----------
    lookback : int
        Number of historical time steps.
    n_instruments : int
        Number of instruments (2 for pipeline_v2: spot + perp).
    """

    def __init__(self, lookback: int, n_instruments: int = 2):
        super().__init__()
        L = lookback
        self.n_instruments = n_instruments
        k = torch.arange(L, dtype=torch.float32)
        static = torch.stack(
            [
                k / L,
                (L - k) / L,
                torch.sqrt((L - k) / L),
                torch.log1p(L - k),
            ],
            dim=1,
        )  # (L, 4)
        self.register_buffer("static_pos", static)
        self.eps = 1e-12

    def forward(
        self,
        x: torch.Tensor,
        prefix_window: torch.Tensor,
        prefix_start: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, L, F)     — 257 features from pipeline_v2
            prefix_window:  (B, P, L)     — P=2 instruments (spot, perp)
            prefix_start:   (B, P, 1)
        Returns:
            (B, L, F + 4 + P)            — 257 + 4 + 2 = 263
        """
        B, L, _ = x.shape
        cum_rv = torch.log(prefix_window - prefix_start + self.eps)  # (B, P, L)
        cum_rv = cum_rv.permute(0, 2, 1)                             # (B, L, P)
        static = self.static_pos.unsqueeze(0).expand(B, -1, -1)      # (B, L, 4)
        return torch.cat([x, static, cum_rv], dim=2)


# ---------------------------------------------------------------------------
# Builder helper
# ---------------------------------------------------------------------------

def build_dataset(
    df: pl.DataFrame,
    lookback: int,
    stride: int,
    device: torch.device,
    variable_horizon: bool = True,
    min_horizon: int = 1,
    max_horizon: int = 900,
    prod_idx: int = 0,
    cfg: PipelineConfig | None = None,
) -> GPUDataset:
    """Convenience: pipeline_v2 output DataFrame → GPUDataset.

    Parameters
    ----------
    df : pl.DataFrame
        Output of pipeline_v2.run_pipeline() — contains all 257 features,
        base RV columns, and FFF seasonality factors.
    lookback : int
        Number of historical rows (seconds) the model sees.
    stride : int
        Subsample valid indices (1 = every second, 60 = every minute, etc.).
    max_horizon : int
        Max forward horizon in seconds (default 900 = 15 minutes).
    prod_idx : int
        Index into instruments for the target (0=spot, 1=perp).
    cfg : PipelineConfig | None
        Pipeline config. If None, uses default PipelineConfig().
    """
    if cfg is None:
        cfg = PipelineConfig()
    instruments = cfg.instruments

    # Features: all 257 columns from pipeline_v2
    feat_cols = get_feature_columns(cfg)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} expected feature columns in dataframe. "
            f"First 5: {missing[:5]}"
        )
    features = df.select(feat_cols).to_numpy().astype(np.float32)

    # Prefix sums from per-second RV
    prefix_rvs = precompute_prefix_sums(df, instruments)
    prefix_stack = build_prefix_stack(prefix_rvs, instruments)

    return GPUDataset(
        features=features,
        prefix_stack=prefix_stack,
        lookback=lookback,
        stride=stride,
        device=device,
        variable_horizon=variable_horizon,
        min_horizon=min_horizon,
        max_horizon=max_horizon,
        prod_idx=prod_idx,
    )
