"""
GPU-Resident Dataset & Batching
================================
All data lives on GPU. Batching is just index slicing — zero CPU involvement.

Features : precomputed rolling-window + cyclical time columns from pipeline.
Targets  : computed on-the-fly via prefix sums of per-second RV,
           giving second-granular horizon resolution (0–900s).

At 1s aggregation, RV_t = r_t² (single squared return per second),
so cumsum(RV) = cumsum(r²) — no separate log-return column needed.
"""

import numpy as np
import polars as pl
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants — must match pipeline output
# ---------------------------------------------------------------------------

PRODUCTS = ["s", "p"]
MEASURES = ["rv", "rsv_pos", "rsv_neg", "bpv"]
ROLLING_WINDOWS = [5, 30, 60, 1440, 2048, 4096, 8192, 16384]

CYCLICAL_COMPONENTS = [
    "month", "day_of_week", "day_of_month", "hour", "minute", "second",
]


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def get_rolling_feature_cols(
    products: list[str] | None = None,
    measures: list[str] | None = None,
    windows: list[int] | None = None,
) -> list[str]:
    """Rolling window feature columns: {prod}_{measure}_w{W}."""
    products = products or PRODUCTS
    measures = measures or MEASURES
    windows = windows or ROLLING_WINDOWS
    return [
        f"{prod}_{meas}_w{w}"
        for prod in products
        for meas in measures
        for w in windows
    ]


def get_cyclical_feature_cols(
    components: list[str] | None = None,
) -> list[str]:
    """Cyclical sin/cos columns: time_{comp}_sin, time_{comp}_cos."""
    components = components or CYCLICAL_COMPONENTS
    cols = []
    for comp in components:
        cols.extend([f"time_{comp}_sin", f"time_{comp}_cos"])
    return cols


def get_feature_cols(
    products: list[str] | None = None,
    measures: list[str] | None = None,
    windows: list[int] | None = None,
    cyclical_components: list[str] | None = None,
) -> list[str]:
    """Full ordered list of model feature columns."""
    return (
        get_rolling_feature_cols(products, measures, windows)
        + get_cyclical_feature_cols(cyclical_components)
    )


def get_supporting_cols(
    products: list[str] | None = None,
) -> list[str]:
    """Supporting columns kept in the dataframe but not used as features."""
    products = products or PRODUCTS
    cols = ["received_time", "mow"]
    for prod in products:
        cols.extend([
            f"{prod}_rv", f"{prod}_rsv_pos", f"{prod}_rsv_neg", f"{prod}_bpv",
            f"{prod}_mid", f"{prod}_seas_factor",
        ])
    return cols


# ---------------------------------------------------------------------------
# Prefix-sum builder (runs once on CPU before GPU transfer)
# ---------------------------------------------------------------------------

def precompute_prefix_sums(
    df: pl.DataFrame,
    products: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Cumulative sum of per-second RV → prefix array of length T+1.

    At 1s aggregation RV_t = r_t², so cumsum(RV) gives the same result
    as cumsum(r²) without needing a separate log-return column.

    prefix[0] = 0
    prefix[k] = sum(rv[0] .. rv[k-1])

    Forward RV over [a, b) = prefix[b] - prefix[a].
    """
    products = products or PRODUCTS
    prefix_rvs: dict[str, np.ndarray] = {}
    for prod in products:
        rv = df.get_column(f"{prod}_rv").to_numpy().astype(np.float64)
        rv = np.nan_to_num(rv, nan=0.0)
        prefix = np.zeros(len(rv) + 1, dtype=np.float64)
        np.cumsum(rv, out=prefix[1:])
        prefix_rvs[prod] = prefix.astype(np.float32)
    return prefix_rvs


def build_prefix_stack(
    prefix_rvs: dict[str, np.ndarray],
    products: list[str] | None = None,
) -> np.ndarray:
    """Stack per-product prefix arrays → (P, T+1) array."""
    products = products or PRODUCTS
    return np.stack([prefix_rvs[p] for p in products], axis=0)


# ---------------------------------------------------------------------------
# GPU Dataset
# ---------------------------------------------------------------------------

class GPUDataset:
    """All data lives on GPU.  Batching is just index slicing.

    Features come from precomputed rolling-window + cyclical columns.
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
    log-RV from prefix sums over the lookback window."""

    def __init__(self, lookback: int, n_products: int = 4):
        super().__init__()
        L = lookback
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
            x:              (B, L, F)     — 140 features from pipeline
            prefix_window:  (B, P, L)     — P=4 products
            prefix_start:   (B, P, 1)
        Returns:
            (B, L, F + 4 + P)            — 140 + 4 + 4 = 148
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
    products: list[str] | None = None,
) -> GPUDataset:
    """Convenience: pipeline output DataFrame → GPUDataset.

    Parameters
    ----------
    df : pl.DataFrame
        Output of pipeline.run_pipeline() — contains rolling window features,
        cyclical time features, base RV columns, and seasonality factors.
    lookback : int
        Number of historical rows (seconds) the model sees.
    stride : int
        Subsample valid indices (1 = every second, 60 = every minute, etc.).
    max_horizon : int
        Max forward horizon in seconds (default 900 = 15 minutes).
    prod_idx : int
        Index into PRODUCTS for the target instrument (0=s, 1=p, 2=e, 3=d).
    """
    products = products or PRODUCTS

    # Features: rolling windows + cyclical time
    feat_cols = get_feature_cols(products)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} expected feature columns in dataframe. "
            f"First 5: {missing[:5]}"
        )
    features = df.select(feat_cols).to_numpy().astype(np.float32)

    # Prefix sums from per-second RV
    prefix_rvs = precompute_prefix_sums(df, products)
    prefix_stack = build_prefix_stack(prefix_rvs, products)

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
