"""
Volatility Forecasting — Feature Construction Pipeline
=======================================================
Steps:
  1. Load per-instrument parquets, merge on floored 1s timestamp
  2. Compute midprice → log returns → aggregate to 1s features (RV, RSV±, BPV)
  3. Train/test split; compute minute-of-week (MOW) averages of RV on train
  4. Wavelet-smooth MOW averages: log → wavelet → keep coarse bands → exp
  5. Normalize seasonality curve (divide by its mean)
  6. Divide all second-level features by their mapped seasonality value
  7. Rolling windows of each feature: sum → sqrt → log
  8. Cyclical sin/cos time features
  9. Save feature parquet (targets computed on the fly at train/inference time)

All constants are parameterised in PipelineConfig.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pywt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════

# Aggregation granularity (seconds) — features are computed at this resolution
AGG_FREQ_S = 1

# Forecast horizons: 0 to MAX_HORIZON_MIN minutes (in seconds at AGG_FREQ_S)
MAX_HORIZON_MIN = 15
MAX_HORIZON_S = MAX_HORIZON_MIN * 60   # 900

# Rolling window widths (in units of AGG_FREQ_S = seconds)
ROLLING_WINDOWS = [5, 30, 60, 1440, 2048, 4096, 8192, 16384]

# Seasonality resolution: minute-of-week (10,080 slots)
SEASONALITY_PERIODS_PER_WEEK = 7 * 24 * 60   # 10,080

# Wavelet settings for seasonality smoothing
WAVELET = "db4"
WAVELET_LEVEL = None          # None → pywt auto
WAVELET_KEEP_BANDS = 3        # keep N coarsest coefficient arrays

# Train / test split
TRAIN_FRAC = 0.8

# Cyclical time components to encode as sin/cos
CYCLICAL_TIME_COMPONENTS = [
    "month", "day_of_week", "day_of_month", "hour", "minute", "second",
]

# Numerical
EPS = 1e-12


# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    # ── Frequencies ───────────────────────────────────────
    base_freq_s: int = 1              # resample raw ticks to this (seconds)
    agg_freq_s: int = AGG_FREQ_S     # feature aggregation granularity

    # ── Instruments (column prefixes in merged dataframe) ─
    instruments: list[str] = field(default_factory=lambda: ["s", "p", "e", "d"])

    # ── Price column suffixes per instrument ──────────────
    bid_col: str = "p_bid_0_price"
    ask_col: str = "p_ask_0_price"

    # ── Horizons ─────────────────────────────────────────
    max_horizon_min: int = MAX_HORIZON_MIN
    max_horizon_s: int = MAX_HORIZON_S

    # ── Rolling-window widths (agg_freq units = seconds) ─
    rolling_windows: list[int] = field(default_factory=lambda: list(ROLLING_WINDOWS))

    # ── Train / test ─────────────────────────────────────
    train_frac: float = TRAIN_FRAC

    # ── Wavelet seasonality ──────────────────────────────
    wavelet: str = WAVELET
    wavelet_level: Optional[int] = WAVELET_LEVEL
    wavelet_keep_bands: int = WAVELET_KEEP_BANDS

    # ── Cyclical time features ───────────────────────────
    cyclical_time_components: list[str] = field(
        default_factory=lambda: list(CYCLICAL_TIME_COMPONENTS)
    )

    # ── Numerical ────────────────────────────────────────
    eps: float = EPS

    # ── IO ───────────────────────────────────────────────
    ts_col: str = "received_time"

    # ── Per-instrument input files ───────────────────────
    input_map: dict[str, str] = field(default_factory=dict)

    # ── Derived ──────────────────────────────────────────
    @property
    def seasonality_periods_per_week(self) -> int:
        return SEASONALITY_PERIODS_PER_WEEK

    @property
    def base_features(self) -> list[str]:
        """The four second-level variance features per instrument."""
        return ["rv", "rsv_pos", "rsv_neg", "bpv"]

    @property
    def cyclical_feature_names(self) -> list[str]:
        names = []
        for comp in self.cyclical_time_components:
            names.extend([f"time_{comp}_sin", f"time_{comp}_cos"])
        return names

    @property
    def n_rolling_features(self) -> int:
        return len(self.instruments) * len(self.base_features) * len(self.rolling_windows)

    @property
    def n_model_features(self) -> int:
        return self.n_rolling_features + len(self.cyclical_feature_names)


# ═══════════════════════════════════════════════════════════
# Step 0 — Load separate parquets and merge on timestamp
# ═══════════════════════════════════════════════════════════

def load_and_merge(cfg: PipelineConfig) -> pl.DataFrame:
    """
    Load one parquet per instrument.  Each file has unprefixed columns
    (received_time, bid_0_price, ask_0_price, …).  We prefix all non-ts
    columns with the instrument tag and outer-join on the floored timestamp.
    """
    ts = cfg.ts_col
    freq_str = f"{cfg.base_freq_s}s"
    frames: list[pl.DataFrame] = []

    for inst, path in cfg.input_map.items():
        print(f"  Loading {inst} ← {path}")
        raw = pl.read_parquet(path)

        raw = raw.with_columns(pl.col(ts).cast(pl.Datetime("us")))

        # Floor to base freq and keep last tick per period
        raw = (
            raw.with_columns(pl.col(ts).dt.truncate(freq_str).alias("__ts"))
            .sort("__ts", ts)
            .group_by("__ts")
            .agg([pl.col(c).last() for c in raw.columns if c != ts and c != "__ts"])
            .rename({"__ts": ts})
            .sort(ts)
        )

        rename_map = {c: f"{inst}_{c}" for c in raw.columns if c != ts}
        raw = raw.rename(rename_map)
        frames.append(raw)

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, on=ts, how="full", coalesce=True)

    merged = merged.sort(ts)

    price_cols = [c for c in merged.columns if c != ts]
    merged = merged.with_columns(
        [pl.col(c).forward_fill().backward_fill() for c in price_cols]
    )

    return merged


# ═══════════════════════════════════════════════════════════
# Step 1 — Returns + aggregate to 1s features
# ═══════════════════════════════════════════════════════════

def compute_agg_features(df: pl.DataFrame, cfg: PipelineConfig) -> pl.DataFrame:
    """
    From base-freq prices → midprice → log returns → aggregate to agg_freq:
      RV   = Σ r²
      RSV+ = Σ r² · 𝟙(r>0)
      RSV− = Σ r² · 𝟙(r<0)
      BPV  = (π/2) · Σ |rₜ|·|rₜ₋₁|
    """
    ts = cfg.ts_col
    freq_str = f"{cfg.agg_freq_s}s"

    # Midprice + log returns at base freq
    mid_exprs, ret_exprs = [], []
    for inst in cfg.instruments:
        bid = f"{inst}_{cfg.bid_col}"
        ask = f"{inst}_{cfg.ask_col}"
        mid = f"{inst}_mid"
        ret = f"{inst}_ret"
        mid_exprs.append(((pl.col(bid) + pl.col(ask)) / 2).alias(mid))
        ret_exprs.append(pl.col(mid).log().diff().alias(ret))

    df = df.with_columns(mid_exprs).with_columns(ret_exprs)

    # BPV helper: |r_t| * |r_{t-1}| within same period
    agg_floor = pl.col(ts).dt.truncate(freq_str)
    df = df.with_columns(agg_floor.alias("__agg_floor"))

    bpv_exprs = []
    for inst in cfg.instruments:
        ret = f"{inst}_ret"
        abs_ret = pl.col(ret).abs()
        abs_ret_lag = abs_ret.shift(1)
        same_period = pl.col("__agg_floor") == pl.col("__agg_floor").shift(1)
        bpv_pair = pl.when(same_period).then(abs_ret * abs_ret_lag).otherwise(None)
        bpv_exprs.append(bpv_pair.alias(f"{inst}_bpv_pair"))

    df = df.with_columns(bpv_exprs)

    # Aggregate
    agg_ops = [pl.col("__agg_floor").first().alias(ts)]
    for inst in cfg.instruments:
        ret = f"{inst}_ret"
        r2 = pl.col(ret).pow(2)
        agg_ops.extend([
            r2.sum().alias(f"{inst}_rv"),
            r2.filter(pl.col(ret) > 0).sum().alias(f"{inst}_rsv_pos"),
            r2.filter(pl.col(ret) < 0).sum().alias(f"{inst}_rsv_neg"),
            (pl.col(f"{inst}_bpv_pair").sum() * (np.pi / 2)).alias(f"{inst}_bpv"),
            pl.col(f"{inst}_mid").last().alias(f"{inst}_mid"),
        ])

    df_agg = df.group_by("__agg_floor").agg(agg_ops).sort(ts).drop("__agg_floor")
    return df_agg


# ═══════════════════════════════════════════════════════════
# Step 2 — Minute-of-week index + train/test split
# ═══════════════════════════════════════════════════════════

def add_mow_index(df: pl.DataFrame, cfg: PipelineConfig) -> pl.DataFrame:
    """
    Add `mow` column: minute-of-week (0 .. 10,079).
    Each second maps to its enclosing minute slot.
    """
    ts = cfg.ts_col
    mow_expr = (
        (pl.col(ts).dt.weekday() - 1) * 24 * 60     # Monday=0
        + pl.col(ts).dt.hour() * 60
        + pl.col(ts).dt.minute()
    )
    return df.with_columns(mow_expr.cast(pl.Int32).alias("mow"))


def train_test_split(
    df: pl.DataFrame, cfg: PipelineConfig
) -> tuple[pl.DataFrame, pl.DataFrame]:
    n = df.height
    split_idx = int(n * cfg.train_frac)
    return df[:split_idx], df[split_idx:]


# ═══════════════════════════════════════════════════════════
# Step 3 — Cyclical sin/cos time features
# ═══════════════════════════════════════════════════════════

_CYCLICAL_SPEC: dict[str, tuple[str, int]] = {
    "month":        ("month",   12),
    "day_of_week":  ("weekday",  7),
    "day_of_month": ("day",     31),
    "hour":         ("hour",    24),
    "minute":       ("minute",  60),
    "second":       ("second",  60),
}


def add_cyclical_time_features(
    df: pl.DataFrame, cfg: PipelineConfig
) -> pl.DataFrame:
    """sin(2π · value / period) and cos(2π · value / period) per component."""
    ts = cfg.ts_col
    exprs: list[pl.Expr] = []

    for comp in cfg.cyclical_time_components:
        if comp not in _CYCLICAL_SPEC:
            raise ValueError(
                f"Unknown cyclical component '{comp}'. "
                f"Valid: {list(_CYCLICAL_SPEC.keys())}"
            )
        accessor, period = _CYCLICAL_SPEC[comp]
        raw_val = getattr(pl.col(ts).dt, accessor)().cast(pl.Float64)
        angle = raw_val * (2.0 * np.pi / period)
        exprs.append(angle.sin().alias(f"time_{comp}_sin"))
        exprs.append(angle.cos().alias(f"time_{comp}_cos"))

    return df.with_columns(exprs)


# ═══════════════════════════════════════════════════════════
# Step 4 — Wavelet-smoothed seasonality from MOW RV averages
# ═══════════════════════════════════════════════════════════

def _wavelet_smooth(values: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """log → wavelet decompose → zero high-freq bands → reconstruct → exp."""
    log_vals = np.log(values + cfg.eps)

    coeffs = pywt.wavedec(log_vals, cfg.wavelet, level=cfg.wavelet_level)
    for i in range(cfg.wavelet_keep_bands, len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])

    reconstructed = pywt.waverec(coeffs, cfg.wavelet)[: len(values)]
    return np.exp(reconstructed)


def compute_seasonality(
    df_train: pl.DataFrame, cfg: PipelineConfig
) -> dict[str, np.ndarray]:
    """
    For each instrument, compute minute-of-week average RV on train set,
    wavelet-smooth, and normalise so mean = 1.

    Returns dict  inst → np.ndarray of shape (10_080,)
    """
    ppw = cfg.seasonality_periods_per_week   # 10,080
    seasonality: dict[str, np.ndarray] = {}

    for inst in cfg.instruments:
        rv_col = f"{inst}_rv"
        mow_avg = (
            df_train.group_by("mow")
            .agg(pl.col(rv_col).mean().alias("rv_mean"))
            .sort("mow")
        )

        full_mow = pl.DataFrame({"mow": np.arange(ppw, dtype=np.int32)})
        mow_avg = full_mow.join(mow_avg, on="mow", how="left")
        raw = mow_avg.get_column("rv_mean").to_numpy().astype(np.float64)

        # Interpolate gaps
        mask = np.isfinite(raw)
        if not mask.all():
            idx = np.arange(len(raw))
            raw[~mask] = np.interp(idx[~mask], idx[mask], raw[mask])

        smoothed = _wavelet_smooth(raw, cfg)
        smoothed = smoothed / smoothed.mean()   # normalise

        seasonality[inst] = smoothed

    return seasonality


# ═══════════════════════════════════════════════════════════
# Step 5 — Deseasonalise features
# ═══════════════════════════════════════════════════════════

def deseasonalise(
    df: pl.DataFrame,
    seasonality: dict[str, np.ndarray],
    cfg: PipelineConfig,
) -> pl.DataFrame:
    """
    Divide every base feature by the corresponding instrument's seasonality
    value mapped via the `mow` column.  Also stores `{inst}_seas_factor`
    for downstream reseasonalisation.
    """
    mow = df.get_column("mow").to_numpy()

    exprs: list[pl.Expr] = []
    for inst in cfg.instruments:
        seas_curve = seasonality[inst]
        seas_mapped = pl.Series(f"__{inst}_seas", seas_curve[mow])

        for feat in cfg.base_features:
            col = f"{inst}_{feat}"
            exprs.append((pl.col(col) / seas_mapped).alias(col))

        # Store mapped seasonality for reseasonalisation at inference time
        exprs.append(
            pl.Series(f"{inst}_seas_factor", seas_curve[mow])
            .alias(f"{inst}_seas_factor")
        )

    return df.with_columns(exprs)


# ═══════════════════════════════════════════════════════════
# Step 6 — Rolling-window features:  sum → √ → log
# ═══════════════════════════════════════════════════════════

def compute_rolling_features(
    df: pl.DataFrame,
    cfg: PipelineConfig,
) -> pl.DataFrame:
    """
    For every (instrument × base_feature × window), compute:
        log( sqrt( rolling_sum(feature, window) ) )
    """
    eps = cfg.eps
    exprs: list[pl.Expr] = []

    for inst in cfg.instruments:
        for feat in cfg.base_features:
            col = f"{inst}_{feat}"
            for w in cfg.rolling_windows:
                alias = f"{inst}_{feat}_w{w}"
                exprs.append(
                    (pl.col(col).rolling_sum(w) + eps).sqrt().log().alias(alias)
                )

    return df.with_columns(exprs)


# ═══════════════════════════════════════════════════════════
# On-the-fly target computation (not stored in parquet)
# ═══════════════════════════════════════════════════════════

def compute_target_for_horizon(
    df: pl.DataFrame,
    cfg: PipelineConfig,
    inst: str,
    horizon_s: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute target for a single (instrument, horizon) pair on the fly.

    Returns:
        target_deseas_log : log of forward sum of deseasonalised RV
        target_deseas     : forward sum of deseasonalised RV (level)
        fwd_seas_sum      : sum of seasonality factors over forward window
                            (for reseasonalisation)
    """
    rv = df.get_column(f"{inst}_rv")
    seas = df.get_column(f"{inst}_seas_factor")

    target_deseas = rv.rolling_sum(horizon_s).shift(-horizon_s).to_numpy()
    target_deseas_log = np.log(target_deseas + cfg.eps)
    fwd_seas_sum = seas.rolling_sum(horizon_s).shift(-horizon_s).to_numpy()

    return target_deseas_log, target_deseas, fwd_seas_sum


# ═══════════════════════════════════════════════════════════
# Feature column names
# ═══════════════════════════════════════════════════════════

def get_feature_columns(cfg: PipelineConfig) -> list[str]:
    """Return the full list of model feature column names."""
    cols = []
    for inst in cfg.instruments:
        for feat in cfg.base_features:
            for w in cfg.rolling_windows:
                cols.append(f"{inst}_{feat}_w{w}")

    cols.extend(cfg.cyclical_feature_names)
    return cols


# ═══════════════════════════════════════════════════════════
# On-the-fly model fitting (OLS)
# ═══════════════════════════════════════════════════════════

def fit_horizon_model(
    df_train: pl.DataFrame,
    cfg: PipelineConfig,
    horizon_s: int,
    inst: str,
):
    """
    Fit OLS for a single (instrument, horizon) pair.
    Target is computed on the fly.

    Returns model dict or None if insufficient data.
    """
    feature_cols = get_feature_columns(cfg)

    target_log, _, _ = compute_target_for_horizon(df_train, cfg, inst, horizon_s)
    target_series = pl.Series("__target", target_log)

    subset = (
        df_train.select(feature_cols)
        .with_columns(target_series.alias("__target"))
        .drop_nulls()
        .drop_nans()
    )
    if subset.height < len(feature_cols) + 1:
        return None

    X = subset.select(feature_cols).to_numpy()
    y = subset.get_column("__target").to_numpy()

    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    X_aug = np.hstack([ones, X])
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

    y_hat = X_aug @ beta
    residuals = y - y_hat
    sigma2 = np.var(residuals, ddof=X_aug.shape[1])

    return {
        "horizon_s": horizon_s,
        "inst": inst,
        "feature_cols": feature_cols,
        "beta": beta,
        "sigma2": sigma2,
        "residuals": residuals,
    }


# ═══════════════════════════════════════════════════════════
# Predict + reseasonalise (no smearing — just exp)
# ═══════════════════════════════════════════════════════════

def predict_and_reseasonalise(
    df: pl.DataFrame,
    model: dict,
    cfg: PipelineConfig,
) -> np.ndarray:
    """
    Generate RV predictions: exp(log_yhat) * (fwd_seas_sum / horizon).
    No smearing — just exponentiate.
    """
    inst = model["inst"]
    horizon_s = model["horizon_s"]
    feature_cols = model["feature_cols"]
    beta = model["beta"]

    X = df.select(feature_cols).to_numpy()
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    X_aug = np.hstack([ones, X])
    log_yhat = X_aug @ beta

    yhat = np.exp(log_yhat)

    _, _, fwd_seas = compute_target_for_horizon(df, cfg, inst, horizon_s)
    return yhat * (fwd_seas / horizon_s)


# ═══════════════════════════════════════════════════════════
# Full pipeline — build feature table only
# ═══════════════════════════════════════════════════════════

def run_pipeline(
    output_path: str | Path,
    cfg: PipelineConfig | None = None,
) -> pl.DataFrame:
    """
    End-to-end: load → features → seasonality → deseasonalise → rolling → save.
    No targets or models — those are computed on the fly.
    """
    if cfg is None:
        cfg = PipelineConfig()

    output_path = Path(output_path)

    print(f"[1/7] Loading & merging {len(cfg.input_map)} instruments "
          f"(resampled to {cfg.base_freq_s}s) ...")
    df = load_and_merge(cfg)
    print(f"       Merged rows: {df.height:,}")

    print(f"[2/7] Computing {cfg.agg_freq_s}s features (RV, RSV±, BPV) ...")
    df = compute_agg_features(df, cfg)
    df = add_mow_index(df, cfg)
    print(f"       Agg rows: {df.height:,}")

    print("[3/7] Train/test split + wavelet seasonality ...")
    df_train, _ = train_test_split(df, cfg)
    seasonality = compute_seasonality(df_train, cfg)
    for inst, s in seasonality.items():
        print(f"       {inst} seasonality: min={s.min():.4f}  max={s.max():.4f}  "
              f"mean≈{s.mean():.4f}")

    print("[4/7] Deseasonalising ...")
    df = deseasonalise(df, seasonality, cfg)

    print(f"[5/7] Computing rolling-window features "
          f"(windows={cfg.rolling_windows}) ...")
    df = compute_rolling_features(df, cfg)

    print(f"[6/7] Adding cyclical time features "
          f"({', '.join(cfg.cyclical_time_components)}) ...")
    df = add_cyclical_time_features(df, cfg)

    # ── Column summary ────────────────────────────────────
    feature_cols = get_feature_columns(cfg)
    print(f"\n       Model features:  {len(feature_cols)}")
    print(f"       Rolling:         {cfg.n_rolling_features}")
    print(f"       Cyclical:        {len(cfg.cyclical_feature_names)}")
    print(f"       Total columns:   {df.width}")

    # ── Save ──────────────────────────────────────────────
    print(f"\n[7/7] Saving ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    print(f"✓ Saved feature table → {output_path}  "
          f"({df.height:,} rows × {df.width} cols)")

    seas_path = output_path.with_name(output_path.stem + "_seasonality.npz")
    np.savez(seas_path, **seasonality)
    print(f"✓ Saved seasonality   → {seas_path}")

    return df


# ═══════════════════════════════════════════════════════════
# Model fitting & serialisation
# ═══════════════════════════════════════════════════════════

def fit_and_save_models(
    df: pl.DataFrame,
    cfg: PipelineConfig,
    output_dir: str | Path,
    horizon_seconds: list[int] | None = None,
) -> dict:
    """
    Fit OLS for every (instrument, horizon) pair on the training split.
    Horizons default to every 60s from 60 to max_horizon_s.

    Returns dict[(inst, horizon_s)] → model_dict
    """
    import pickle

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if horizon_seconds is None:
        horizon_seconds = list(range(60, cfg.max_horizon_s + 1, 60))

    df_train, _ = train_test_split(df, cfg)

    models: dict[tuple[str, int], dict] = {}
    for inst in cfg.instruments:
        print(f"  Fitting {inst} ({len(horizon_seconds)} horizons) ...")
        for h_s in horizon_seconds:
            model = fit_horizon_model(df_train, cfg, h_s, inst=inst)
            if model is None:
                print(f"    h={h_s:5d}s  SKIPPED (insufficient data)")
                continue
            # Store only what's needed — drop raw residuals
            del model["residuals"]
            models[(inst, h_s)] = model
        n_fit = sum(1 for k in models if k[0] == inst)
        print(f"    → {n_fit} models for {inst}")

    bundle = {
        "models": models,
        "config": {
            "agg_freq_s": cfg.agg_freq_s,
            "instruments": cfg.instruments,
            "max_horizon_s": cfg.max_horizon_s,
            "rolling_windows": cfg.rolling_windows,
            "cyclical_time_components": cfg.cyclical_time_components,
            "eps": cfg.eps,
            "train_frac": cfg.train_frac,
        },
    }

    path = output_dir / "models.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved {len(models)} models → {path}")

    return models


# ═══════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """OOS R² = 1 − SS_res / SS_tot."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def fit_and_evaluate(
    df: pl.DataFrame,
    cfg: PipelineConfig,
    output_dir: str | Path,
    horizon_seconds: list[int] | None = None,
):
    """
    Fit OLS per (instrument, horizon) on train, predict on test,
    compute R² in log-space and raw level-space (no smearing).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if horizon_seconds is None:
        horizon_seconds = list(range(60, cfg.max_horizon_s + 1, 60))

    df_train, df_test = train_test_split(df, cfg)

    records = []
    for inst in cfg.instruments:
        print(f"\n{'='*60}")
        print(f"  Instrument: {inst}")
        print(f"{'='*60}")

        for h_s in horizon_seconds:
            model = fit_horizon_model(df_train, cfg, h_s, inst=inst)
            if model is None:
                print(f"  h={h_s:5d}s  SKIPPED")
                continue

            feature_cols = model["feature_cols"]
            beta = model["beta"]

            # Test predictions (log-space)
            X_test = df_test.select(feature_cols).to_numpy()
            ones = np.ones((X_test.shape[0], 1), dtype=np.float64)
            log_yhat = np.hstack([ones, X_test]) @ beta

            # Test targets (computed on the fly)
            y_log, y_deseas, fwd_seas = compute_target_for_horizon(
                df_test, cfg, inst, h_s
            )

            r2_log = _r_squared(y_log, log_yhat)

            # Raw level: exp(log_yhat) * (fwd_seas / h_s)
            yhat_raw = np.exp(log_yhat) * (fwd_seas / h_s)
            y_raw = y_deseas * (fwd_seas / h_s)
            r2_raw = _r_squared(y_raw, yhat_raw)

            # Train R²
            X_tr = df_train.select(feature_cols).to_numpy()
            y_tr_log, _, _ = compute_target_for_horizon(df_train, cfg, inst, h_s)
            log_yhat_tr = np.hstack(
                [np.ones((X_tr.shape[0], 1)), X_tr]
            ) @ beta
            r2_train = _r_squared(y_tr_log, log_yhat_tr)

            h_min = h_s / 60
            print(f"  h={h_s:5d}s ({h_min:5.1f}m)  "
                  f"R²_log={r2_log:+.4f}  R²_raw={r2_raw:+.4f}  "
                  f"σ²={model['sigma2']:.2f}")

            records.append({
                "inst": inst,
                "horizon_s": h_s,
                "horizon_min": h_min,
                "r2_log_train": r2_train,
                "r2_log_test": r2_log,
                "r2_raw_test": r2_raw,
                "sigma2": model["sigma2"],
                "n_features": len(feature_cols),
            })

    summary = pl.DataFrame(records)
    summary.write_csv(output_dir / "r2_summary.csv")

    # ── Plots ─────────────────────────────────────────────
    with PdfPages(output_dir / "analysis.pdf") as pdf:

        # R² vs horizon (log-space)
        fig, ax = plt.subplots(figsize=(10, 5))
        for inst in cfg.instruments:
            sub = summary.filter(pl.col("inst") == inst).sort("horizon_s")
            h = sub.get_column("horizon_min").to_numpy()
            ax.plot(h, sub.get_column("r2_log_test").to_numpy(),
                    label=f"{inst} (test)", marker=".", markersize=3)
            ax.plot(h, sub.get_column("r2_log_train").to_numpy(),
                    label=f"{inst} (train)", linestyle="--", alpha=0.4)
        ax.set_xlabel("Horizon (minutes)")
        ax.set_ylabel("R²")
        ax.set_title("Log-space R² vs horizon")
        ax.legend(fontsize=7, ncol=2)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # R² vs horizon (raw level)
        fig, ax = plt.subplots(figsize=(10, 5))
        for inst in cfg.instruments:
            sub = summary.filter(pl.col("inst") == inst).sort("horizon_s")
            h = sub.get_column("horizon_min").to_numpy()
            ax.plot(h, sub.get_column("r2_raw_test").to_numpy(),
                    label=inst, marker=".", markersize=3)
        ax.set_xlabel("Horizon (minutes)")
        ax.set_ylabel("R²")
        ax.set_title("Raw level-space R² vs horizon (no smearing)")
        ax.legend(fontsize=7)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # Train vs Test scatter
        fig, ax = plt.subplots(figsize=(7, 7))
        for inst in cfg.instruments:
            sub = summary.filter(pl.col("inst") == inst)
            ax.scatter(
                sub.get_column("r2_log_train").to_numpy(),
                sub.get_column("r2_log_test").to_numpy(),
                label=inst, alpha=0.6, s=30,
            )
        lims = [-0.5, 1.0]
        ax.plot(lims, lims, "k--", linewidth=0.8, label="y=x")
        ax.set_xlabel("Train R² (log)")
        ax.set_ylabel("Test R² (log)")
        ax.set_title("Overfit diagnostic: Train vs Test R²")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # σ² vs horizon
        fig, ax = plt.subplots(figsize=(10, 5))
        for inst in cfg.instruments:
            sub = summary.filter(pl.col("inst") == inst).sort("horizon_s")
            h = sub.get_column("horizon_min").to_numpy()
            ax.plot(h, sub.get_column("sigma2").to_numpy(),
                    label=inst, marker=".", markersize=3)
        ax.set_xlabel("Horizon (minutes)")
        ax.set_ylabel("σ²")
        ax.set_title("Residual variance (log-space)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    print(f"\n✓ Saved analysis.pdf → {output_dir / 'analysis.pdf'}")

    # Console summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for inst in cfg.instruments:
        sub = summary.filter(pl.col("inst") == inst)
        r2 = sub.get_column("r2_log_test").to_numpy()
        valid = np.isfinite(r2)
        print(f"\n  {inst}:  ({valid.sum()}/{len(horizon_seconds)} horizons)")
        print(f"    R²_log (test):  mean={np.nanmean(r2):.4f}  "
              f"median={np.nanmedian(r2):.4f}  "
              f"[{np.nanmin(r2):.4f}, {np.nanmax(r2):.4f}]")
        r2_raw = sub.get_column("r2_raw_test").to_numpy()
        print(f"    R²_raw (test):  mean={np.nanmean(r2_raw):+.4f}  "
              f"median={np.nanmedian(r2_raw):+.4f}")

    return summary


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Volatility feature pipeline (1s resolution)",
        epilog="Example: python pipeline.py "
               "--inputs s:data/spot.parquet p:data/perp.parquet "
               "e:data/eth.parquet d:data/doge.parquet "
               "--output data/features.parquet",
    )
    parser.add_argument(
        "--inputs", required=True, nargs="+",
        help="Instrument inputs as PREFIX:PATH pairs",
    )
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--base-freq", type=int, default=1)
    parser.add_argument("--agg-freq", type=int, default=AGG_FREQ_S)
    parser.add_argument("--max-horizon-min", type=int, default=MAX_HORIZON_MIN)
    parser.add_argument(
        "--rolling-windows", nargs="+", type=int,
        default=ROLLING_WINDOWS,
    )
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--wavelet", default=WAVELET)
    parser.add_argument("--wavelet-keep-bands", type=int, default=WAVELET_KEEP_BANDS)
    parser.add_argument(
        "--cyclical-time", nargs="+",
        default=CYCLICAL_TIME_COMPONENTS,
    )
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument(
        "--eval", action="store_true",
        help="Run model fitting + evaluation after feature construction",
    )
    parser.add_argument(
        "--eval-horizon-step", type=int, default=60,
        help="Evaluation horizon step in seconds (default: 60 = every minute)",
    )

    args = parser.parse_args()

    input_map = {}
    for spec in args.inputs:
        if ":" not in spec:
            parser.error(f"Invalid input spec '{spec}' — expected PREFIX:PATH")
        prefix, path = spec.split(":", 1)
        input_map[prefix] = path

    cfg = PipelineConfig(
        base_freq_s=args.base_freq,
        agg_freq_s=args.agg_freq,
        instruments=list(input_map.keys()),
        input_map=input_map,
        max_horizon_min=args.max_horizon_min,
        max_horizon_s=args.max_horizon_min * 60,
        rolling_windows=args.rolling_windows,
        train_frac=args.train_frac,
        wavelet=args.wavelet,
        wavelet_keep_bands=args.wavelet_keep_bands,
        cyclical_time_components=args.cyclical_time,
        eps=args.eps,
    )

    df = run_pipeline(args.output, cfg)

    if args.eval:
        print("\n" + "═" * 70)
        print("  RUNNING EVALUATION")
        print("═" * 70)
        horizon_seconds = list(
            range(args.eval_horizon_step, cfg.max_horizon_s + 1, args.eval_horizon_step)
        )
        eval_dir = Path(args.output).parent / "eval"
        fit_and_evaluate(df, cfg, eval_dir, horizon_seconds)
