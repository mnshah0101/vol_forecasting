"""
Standalone LightGBM model for pipeline_new (370-feature pipeline).

Usage:
    module load anaconda3 && conda activate base
    python -m vol_forecasting.models.lightgbm_v2 \
        --data /path/to/output.parquet \
        --horizon 15m \
        --product s \
        --pricer-path pricing/configs/pricer.json \
        --jsu-params-path pricing/configs/jsu.csv

Point-in-time tabular model: uses the 370 pipeline_new features at t
to predict log(forward RV) over [t+1, t+1+horizon].
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Feature columns — inline copy of pipeline_new.get_feature_columns()
# to keep this script standalone (no vol_forecasting import needed).
# ---------------------------------------------------------------------------

INSTRUMENTS = ["s", "p"]
BASE_FEATURES = ["rv", "rsv_pos", "rsv_neg", "bpv"]
ROLLING_WINDOWS = [5, 30, 60, 1440, 2048, 3600, 4096, 8192, 14400, 16384]
MODWT_SCALES = [6, 8, 10, 13, 15, 17]


def _feature_columns() -> list[str]:
    """All 370 pipeline_new feature columns."""
    s, p = "s", "p"
    insts = INSTRUMENTS
    cols: list[str] = []

    # Tier 1: rolling windows
    for inst in insts:
        for feat in BASE_FEATURES:
            for w in ROLLING_WINDOWS:
                cols.append(f"{inst}_{feat}_w{w}")

    # Basis & spread
    for w in [5, 60, 1440, 8192]:
        cols.append(f"basis_level_w{w}")
    for w in [60, 1440, 8192]:
        cols.append(f"basis_vol_w{w}")
    for lag in [60, 300]:
        cols.append(f"basis_change_{lag}s")
    cols.append("basis_zscore_1h")

    for inst in insts:
        for w in [5, 60, 1440]:
            cols.append(f"{inst}_spread_w{w}")
    cols.append("s_spread_zscore_5m")

    # Jump, asymmetry, downside
    for inst in insts:
        for w in [60, 1440, 8192]:
            cols.append(f"{inst}_jump_ratio_w{w}")
    for inst in insts:
        for w in [60, 1440, 8192]:
            cols.append(f"{inst}_asym_impact_w{w}")
    for inst in insts:
        for w in [60, 1440, 8192]:
            cols.append(f"{inst}_downside_share_w{w}")

    # Signed returns, drawdown, RV term structure
    for inst in insts:
        for w in [30, 60, 1440]:
            cols.append(f"{inst}_signed_ret_w{w}")
    for inst in insts:
        for w in [1440, 8192, 16384]:
            cols.append(f"{inst}_drawdown_w{w}")
    for inst in insts:
        cols.append(f"{inst}_rv_term_slope")
        cols.append(f"{inst}_rv_term_curvature")

    # Microprice returns & acceleration
    for label in ["1s", "5s", "10s", "30s", "60s", "300s"]:
        cols.append(f"microprice_ret_{label}")
    cols.extend(["ret_accel_1s_5s", "ret_accel_5s_30s", "ret_accel_30s_300s"])

    # VPIN, tick intensity
    cols.extend(["vpin_5m", "tick_intensity_ratio"])

    # OBI
    for inst in insts:
        cols.append(f"{inst}_obi_delta_5s")
        cols.append(f"{inst}_obi_autocorr_1m")
    cols.append("obi_divergence")
    for inst in insts:
        cols.append(f"{inst}_spread_obi_interaction")

    # MODWT
    for inst in insts:
        for scale in MODWT_SCALES:
            cols.append(f"{inst}_modwt_d{scale}")

    # Cross-correlation, vol-of-vol
    for w in [60, 1440, 8192]:
        cols.append(f"sp_corr_w{w}")
    for inst in insts:
        for w in [60, 1440]:
            cols.append(f"{inst}_vol_of_vol_w{w}")

    # TSRV
    for inst in insts:
        for label in ["1m", "5m", "15m"]:
            cols.append(f"{inst}_tsrv_{label}")
    for inst in insts:
        for label in ["1m", "5m"]:
            cols.append(f"{inst}_tsrv_rv_ratio_{label}")

    # RK, RQ
    for inst in insts:
        for label in ["5m", "15m"]:
            cols.append(f"{inst}_rk_{label}")
    for inst in insts:
        for label in ["1m", "5m", "15m"]:
            cols.append(f"{inst}_rq_{label}")

    # EWMA
    for inst in insts:
        for hl in ["5s", "30s"]:
            cols.append(f"{inst}_ewma_var_{hl}")

    # Roll measure, Parkinson
    for inst in [s, p]:
        for label in ["1m", "5m"]:
            cols.append(f"{inst}_roll_measure_{label}")
    for inst in [s, p]:
        for label in ["30s", "60s"]:
            cols.append(f"{inst}_parkinson_{label}")

    # Perp lead
    cols.append("perp_lead_1s")

    # Roughness, vol accel
    for inst in insts:
        for label in ["5s", "30s", "60s", "300s"]:
            cols.append(f"{inst}_roughness_{label}")
        cols.append(f"{inst}_vol_accel_5s")

    # Spectral bands
    for inst in insts:
        for w_label in ["4k", "8k"]:
            cols.append(f"{inst}_spec_hf_lf_{w_label}")
            cols.append(f"{inst}_spec_mf_lf_{w_label}")
            cols.append(f"{inst}_spec_hf_mf_{w_label}")
            cols.append(f"{inst}_spec_trend_share_{w_label}")

    # Fourier magnitudes
    for inst in [s, p]:
        for period in ["8h", "24h", "168h"]:
            cols.append(f"{inst}_fourier_mag_{period}")

    # Calendar / cyclical
    cols.extend([
        "sin_funding", "cos_funding",
        "monthly_expiry_days", "quarterly_expiry_days",
        "weekend_proximity", "epoch_4h_phase",
    ])

    # Autocorrelation, skew/kurt
    for inst in insts:
        for label in ["5m", "15m"]:
            cols.append(f"{inst}_ret_autocorr_{label}")
    for inst in insts:
        for label in ["5m", "15m"]:
            cols.append(f"{inst}_skew_{label}")
            cols.append(f"{inst}_kurt_{label}")

    cols.extend(["sp_ret_corr_5m", "spread_rv_corr_15m"])

    # Cyclical time encodings
    cols.extend([
        "sin_diurnal", "cos_diurnal",
        "sin_weekly", "cos_weekly",
        "sin_hour", "cos_hour",
    ])

    # Log transforms
    for inst in insts:
        for label in ["1m", "5m", "15m"]:
            for prefix in ["tsrv_", "rq_"]:
                cols.append(f"log_{inst}_{prefix}{label}")
        for label in ["5m", "15m"]:
            cols.append(f"log_{inst}_rk_{label}")
        for hl in ["5s", "30s"]:
            cols.append(f"log_{inst}_ewma_var_{hl}")
        for label in ["1m", "5m"]:
            cols.append(f"log_{inst}_roll_measure_{label}")
        for label in ["30s", "60s"]:
            cols.append(f"log_{inst}_parkinson_{label}")

    # ── New features ──────────────────────────────────────────
    # TSRV 1h/4h
    for inst in insts:
        for label in ["1h", "4h"]:
            cols.append(f"{inst}_tsrv_{label}")
    # Downside share w3600
    for inst in insts:
        cols.append(f"{inst}_downside_share_w3600")
    # RQ 1h
    for inst in insts:
        cols.append(f"{inst}_rq_1h")
    # Raw instantaneous spread in bps
    for inst in insts:
        cols.append(f"{inst}_spread_bps")
    # Signed jump w900, w3600
    for inst in insts:
        for w in [900, 3600]:
            cols.append(f"{inst}_signed_jump_w{w}")
    # Realized jump level w3600
    for inst in insts:
        cols.append(f"{inst}_rj_w3600")
    # Realized covariance
    for label in ["1m", "5m"]:
        cols.append(f"sp_rcov_{label}")
        cols.append(f"sp_rcov_corr_ratio_{label}")
    # Signed volume imbalance
    for inst in insts:
        for label in ["w60", "w300", "w900"]:
            cols.append(f"{inst}_signed_vol_imbalance_{label}")
    # Exchange session seasonality
    for session in [
        "tokyo_open", "tokyo_close", "shanghai_open", "shanghai_close",
        "london_open", "london_close", "nyse_open", "nyse_close",
        "cme_open", "cme_close",
    ]:
        cols.append(f"session_{session}")
    # Deribit expiry calendar
    cols.extend([
        "deribit_weekly_expiry_hours", "deribit_monthly_expiry_hours",
        "deribit_weekly_expiry_proximity", "deribit_monthly_expiry_proximity",
    ])
    # Funding rate cycle
    cols.extend([
        "funding_cycle_phase", "funding_cycle_sin", "funding_cycle_cos",
        "funding_time_proximity",
    ])
    # Volatility phase
    for inst in insts:
        for label in ["5m", "15m"]:
            cols.append(f"{inst}_vol_phase_{label}")
    # Cumulative returns
    for inst in insts:
        for label in ["10s", "60s"]:
            cols.append(f"{inst}_cum_ret_{label}")
    for label in ["10s", "60s"]:
        cols.append(f"ps_cum_ret_ratio_{label}")
    # Log transforms for 1h/4h estimators
    for inst in insts:
        for label in ["1h", "4h"]:
            cols.append(f"log_{inst}_tsrv_{label}")
        cols.append(f"log_{inst}_rq_1h")

    # ── Cross-exchange features (Coinbase spot) ──
    cols.append("coinbase_available")
    cols.append("sc_mid_delta")
    for w in [5, 30, 60, 300]:
        cols.append(f"sc_mid_delta_w{w}")
    for w in [60, 300]:
        cols.append(f"sc_mid_delta_vol_w{w}")
    cols.append("sc_obi_imbalance")
    for w in [60, 300]:
        cols.append(f"sc_obi_imbalance_w{w}")
    cols.append("sc_spread_ratio")
    for w in [60, 300]:
        cols.append(f"sc_spread_ratio_w{w}")
    for w in [60, 300, 900, 3600]:
        cols.append(f"sc_rv_spread_w{w}")
    for w in [60, 300]:
        cols.append(f"sc_vwap_ratio_w{w}")
    cols.append("sc_vwap_delta")
    for w in [60, 300]:
        cols.append(f"sc_vwap_delta_w{w}")

    return cols


# ---------------------------------------------------------------------------
# Horizon parsing
# ---------------------------------------------------------------------------

HORIZON_MAP = {
    "1s": 1, "5s": 5, "10s": 10, "30s": 30,
    "1m": 60, "3m": 180, "5m": 300,
    "10m": 600, "15m": 900, "30m": 1800, "1h": 3600,
}


def parse_horizon(h: str) -> tuple[str, int]:
    if h in HORIZON_MAP:
        return h, HORIZON_MAP[h]
    secs = int(h)
    for label, val in HORIZON_MAP.items():
        if val == secs:
            return label, secs
    return f"{secs}s", secs


# ---------------------------------------------------------------------------
# Data loading & target construction
# ---------------------------------------------------------------------------

# Extra columns needed for pricer eval (beyond the 370 features)
PRICER_EXTRA_COLS = ["received_time", "s_mid", "p_mid", "s_seas_factor", "p_seas_factor"]


def load_data(path: str, feature_cols: list[str], product: str, horizon_s: int,
              train_frac: float, val_frac: float, stride: int,
              data_frac: float = 1.0, target_clip: float | None = None,
              need_pricer_cols: bool = False):
    """Load parquet, extract features and targets, split train/val/test.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test, stats, df_or_None
        df is returned (with pricer columns) only when need_pricer_cols=True.
    """
    print(f"Loading {path} ...")
    t0 = time.time()

    if data_frac < 1.0:
        needed_cols = list(set(feature_cols + [f"{product}_rv"]))
        if need_pricer_cols:
            needed_cols = list(set(needed_cols + PRICER_EXTRA_COLS))
        n_total = pl.scan_parquet(path).select(pl.len()).collect().item()
        n_rows = int(n_total * data_frac)
        df = pl.scan_parquet(path).select(
            [c for c in needed_cols if c in pl.scan_parquet(path).collect_schema().names()]
        ).head(n_rows).collect()
        print(f"  Loaded {df.height:,}/{n_total:,} rows ({data_frac:.0%}) x {df.width} cols in {time.time()-t0:.1f}s")
    else:
        df = pl.read_parquet(path)
        print(f"  Loaded {df.height:,} rows x {df.width} cols in {time.time()-t0:.1f}s")

    # Validate feature columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature columns. First 5: {missing[:5]}")

    # Extract features
    X_all = df.select(feature_cols).to_numpy().astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Build prefix sums for target construction
    rv_col = f"{product}_rv"
    rv = df.get_column(rv_col).to_numpy().astype(np.float64)
    rv = np.nan_to_num(rv, nan=0.0)
    prefix = np.zeros(len(rv) + 1, dtype=np.float64)
    np.cumsum(rv, out=prefix[1:])

    # Target: log(sum of RV over [t+1, t+1+horizon])
    eps = 1e-12
    n = len(X_all)
    y_all = np.full(n, np.nan, dtype=np.float64)
    valid_end = n - horizon_s
    for t in range(valid_end):
        fwd_rv = prefix[t + 1 + horizon_s] - prefix[t + 1]
        y_all[t] = np.log(fwd_rv + eps)

    # Split
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    # Normalize features using train stats
    feat_mean = np.nanmean(X_all[:n_train], axis=0, keepdims=True)
    feat_std = np.nanstd(X_all[:n_train], axis=0, keepdims=True) + 1e-8
    X_all = (X_all - feat_mean) / feat_std

    # Normalize targets using train stats
    train_y = y_all[:n_train]
    valid_train = ~np.isnan(train_y)
    tgt_mean = float(np.mean(train_y[valid_train]))
    tgt_std = float(np.std(train_y[valid_train])) + 1e-8
    y_all = (y_all - tgt_mean) / tgt_std

    # Target clipping (in normalized space)
    if target_clip is not None:
        not_nan = ~np.isnan(y_all)
        y_all[not_nan] = np.clip(y_all[not_nan], -target_clip, target_clip)
        print(f"  Target clipping: [-{target_clip}, {target_clip}] (normalized)")

    # Build valid index masks (no NaN targets), apply stride
    def _split(X, y, start, end, apply_stride):
        idx = np.arange(start, end)
        valid = ~np.isnan(y[idx])
        idx = idx[valid]
        if apply_stride and stride > 1:
            idx = idx[::stride]
        return X[idx], y[idx]

    X_train, y_train = _split(X_all, y_all, 0, n_train, True)
    X_val, y_val = _split(X_all, y_all, n_train, n_train + n_val, False)
    X_test, y_test = _split(X_all, y_all, n_train + n_val, n, False)

    print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")
    print(f"  Target stats (train): mean={tgt_mean:.4f}  std={tgt_std:.4f}")

    stats = {
        "feat_mean": feat_mean, "feat_std": feat_std,
        "tgt_mean": tgt_mean, "tgt_std": tgt_std,
        "n_train": n_train, "n_val": n_val,
        "X_all": X_all,  # kept for pricer eval (row-level predictions)
    }

    df_out = df if need_pricer_cols else None
    return X_train, y_train, X_val, y_val, X_test, y_test, stats, df_out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X, y, label=""):
    y_hat = model.predict(X)
    mse = float(np.mean((y - y_hat) ** 2))
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    mae = float(np.mean(np.abs(y - y_hat)))
    print(f"  {label:6s}  MSE={mse:.6f}  R²={r2:.4f}  MAE={mae:.6f}")
    return {"mse": mse, "r2": r2, "mae": mae}


# ---------------------------------------------------------------------------
# Pricer evaluation (BS Fourier + JSU baseline)
# ---------------------------------------------------------------------------

def pricer_eval(
    lgbm_model,
    X_all: np.ndarray,          # (N, F) normalized features for full dataset
    df: pl.DataFrame,           # full dataframe (needs received_time, mid, seas_factor)
    n_train: int, n_val: int,
    tgt_mean: float, tgt_std: float,
    product: str,
    horizon_s: int,
    pricer=None,                # BSFourierPricer or None
    jsu_pricer=None,            # JSUPricer or None
    period_len_min: int = 15,
) -> dict:
    """Period-based pricer evaluation on the val set.

    For each period_len_min-minute period in the val set:
      - strike = spot at period start
      - realized = 1 if end-of-period spot > strike, else 0
      - at each minute boundary, predict forward RV for remaining seconds
        using a linear scaling approximation: RV_h ≈ RV_H * (h / H)
        where H = training horizon, h = remaining time
      - feed to pricer → P(S_T > K)
      - compute Brier score vs realized
    """
    df_val = df[n_train:n_train + n_val]
    mid_col = f"{product}_mid"
    seas_col = f"{product}_seas_factor"

    if mid_col not in df_val.columns:
        print("  Pricer eval: mid column not found, skipping")
        return {}

    spot_all = df_val.get_column(mid_col).to_numpy().astype(np.float64)
    seas_all = (df_val.get_column(seas_col).to_numpy().astype(np.float64)
                if seas_col in df_val.columns
                else np.ones(len(df_val), dtype=np.float64))

    ts = df_val.get_column("received_time")
    second = ts.dt.second().to_numpy()
    minute = ts.dt.minute().to_numpy().astype(np.int32)
    mip = minute % period_len_min  # minute-in-period
    period_start = ts.dt.truncate(f"{period_len_min}m")

    # Identify complete periods
    ps_series = period_start.alias("__ps")
    period_info = (
        df_val.with_columns(ps_series)
        .group_by("__ps")
        .agg([
            pl.col(mid_col).first().alias("__strike"),
            pl.col(mid_col).last().alias("__end_price"),
            pl.col(mid_col).count().alias("__n"),
        ])
    )
    expected = period_len_min * 60
    period_info = period_info.filter(pl.col("__n") >= int(expected * 0.9))

    ps_list = period_info.get_column("__ps").to_list()
    strike_arr = period_info.get_column("__strike").to_numpy()
    end_arr = period_info.get_column("__end_price").to_numpy()
    strike_map = dict(zip(ps_list, strike_arr))
    end_map = dict(zip(ps_list, end_arr))
    complete_set = set(ps_list)

    if not complete_set:
        print("  Pricer eval: no complete periods found")
        return {}

    # Select evaluation rows: minute boundaries within complete periods
    ps_val = period_start.to_list()
    in_complete = np.array([p in complete_set for p in ps_val])
    is_min_bdy = second == 0
    max_h_sec = period_len_min * 60
    has_fwd = np.arange(len(df_val)) + max_h_sec < n_val

    valid = is_min_bdy & has_fwd & in_complete
    valid_idx = np.where(valid)[0]  # val-local indices

    if len(valid_idx) == 0:
        print("  Pricer eval: no valid evaluation points")
        return {}

    # Per valid row: remaining horizon, strike, realized
    mip_valid = mip[valid_idx]
    horizons = ((period_len_min - mip_valid) * 60).astype(np.int64)
    horizons = np.clip(horizons, 1, max_h_sec)

    strikes = np.array([strike_map[ps_val[i]] for i in valid_idx])
    end_prices = np.array([end_map[ps_val[i]] for i in valid_idx])
    realized = (end_prices > strikes).astype(np.float64)
    spot_valid = spot_all[valid_idx]
    seas_valid = seas_all[valid_idx]

    brier_naive = float(np.mean((realized.mean() - realized) ** 2))
    result = {
        "brier_naive": brier_naive,
        "base_rate": float(realized.mean()),
        "n_predictions": int(len(valid_idx)),
        "n_periods": len(complete_set),
    }

    # LightGBM predictions at each evaluation point (val-local → global index)
    global_idx = valid_idx + n_train
    X_eval = X_all[global_idx]
    pred_norm = lgbm_model.predict(X_eval)

    # Denormalize → deseasonalized forward RV (level)
    pred_log = pred_norm * tgt_std + tgt_mean
    rv_deseas_full = np.exp(pred_log)  # RV for the training horizon

    # Scale to the actual remaining horizon:  RV_h ≈ RV_H * (h / H)
    rv_deseas = rv_deseas_full * (horizons.astype(np.float64) / horizon_s)

    # Reseasonalize
    forecasted_var = rv_deseas * seas_valid

    # BSFourier eval
    if pricer is not None:
        tte = horizons.astype(np.float64)
        p_cal, p_bs = pricer.probability(
            spot=spot_valid, strike=strikes,
            forecasted_var=forecasted_var, tte_seconds=tte,
            return_bs_raw=True,
        )
        p_cal = p_cal.ravel()
        p_bs = p_bs.ravel()

        brier_cal = float(np.mean((p_cal - realized) ** 2))
        brier_bs = float(np.mean((p_bs - realized) ** 2))
        brier_skill = 1 - brier_cal / brier_naive if brier_naive > 0 else float("nan")

        per_h_brier = {}
        for h in sorted(np.unique(horizons)):
            m = horizons == h
            if m.sum() >= 10:
                per_h_brier[int(h)] = float(np.mean((p_cal[m] - realized[m]) ** 2))

        result.update({
            "brier_cal": brier_cal,
            "brier_bs": brier_bs,
            "brier_skill": brier_skill,
            "per_horizon_brier": per_h_brier,
        })
        print(f"  BSFourier  Brier: {brier_cal:.4f} (skill {brier_skill:+.3f})  "
              f"BS raw: {brier_bs:.4f}  naive: {brier_naive:.4f}")

    # JSU eval
    if jsu_pricer is not None:
        tte_ms = horizons.astype(np.float64) * 1000.0
        p_jsu = jsu_pricer.probability(
            spot=spot_valid, strike=strikes, tte_ms=tte_ms,
            forecasted_var=forecasted_var,
        ).ravel()

        brier_jsu = float(np.mean((p_jsu - realized) ** 2))
        jsu_skill = 1 - brier_jsu / brier_naive if brier_naive > 0 else float("nan")

        per_h_brier_jsu = {}
        for h in sorted(np.unique(horizons)):
            m = horizons == h
            if m.sum() >= 10:
                per_h_brier_jsu[int(h)] = float(np.mean((p_jsu[m] - realized[m]) ** 2))

        result.update({
            "brier_jsu": brier_jsu,
            "jsu_skill": jsu_skill,
            "per_horizon_brier_jsu": per_h_brier_jsu,
        })
        print(f"  JSU        Brier: {brier_jsu:.4f} (skill {jsu_skill:+.3f})")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LightGBM for dataset v2")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to output.parquet from pipeline_v2")
    parser.add_argument("--horizon", type=str, default="15m",
                        help="Forecast horizon (e.g. 15m, 900, 5m)")
    parser.add_argument("--product", type=str, default="s",
                        choices=["s", "p"], help="Target product")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=60,
                        help="Subsample training rows (1=every sec, 60=every min)")
    parser.add_argument("--data-frac", type=float, default=1.0,
                        help="Fraction of data to load (0.1 = first 10%%, saves memory)")
    parser.add_argument("--target-clip", type=float, default=None,
                        help="Clip normalized targets to [-x, x] (e.g. 5.0)")
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--pricer-path", type=str, default=None,
                        help="Path to BSFourierPricer JSON config")
    parser.add_argument("--jsu-params-path", type=str, default=None,
                        help="Path to JSU params CSV")
    parser.add_argument("--pricer-period-min", type=int, default=15,
                        help="Period length in minutes for pricer eval")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save model + results (default: runs/<id>)")
    args = parser.parse_args()

    h_label, h_secs = parse_horizon(args.horizon)
    feature_cols = _feature_columns()
    need_pricer = args.pricer_path is not None or args.jsu_params_path is not None
    print(f"LightGBM | product={args.product} | horizon={h_label} ({h_secs}s)")
    print(f"Features: {len(feature_cols)}")
    if args.target_clip is not None:
        print(f"Target clip: {args.target_clip}")

    # Load pricers
    pricer = None
    if args.pricer_path is not None:
        try:
            from vol_forecasting.pricing.bs_fourier_pricer import BSFourierPricer
            pricer = BSFourierPricer.from_json(args.pricer_path)
            print(f"BSFourier pricer: {pricer} (period={args.pricer_period_min}m)")
        except Exception as e:
            print(f"WARNING: Could not load BSFourier pricer: {e}")

    jsu_pricer = None
    if args.jsu_params_path is not None:
        try:
            from vol_forecasting.pricing.jsu_pricer import JSUPricer
            jsu_pricer = JSUPricer.from_csv(args.jsu_params_path)
            print(f"JSU pricer: {jsu_pricer}")
        except Exception as e:
            print(f"WARNING: Could not load JSU pricer: {e}")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, stats, df = load_data(
        args.data, feature_cols, args.product, h_secs,
        args.train_frac, args.val_frac, args.stride, args.data_frac,
        target_clip=args.target_clip,
        need_pricer_cols=need_pricer,
    )

    # Run directory
    run_id = uuid.uuid4().hex[:8]
    save_dir = Path(args.save_dir) if args.save_dir else Path(f"runs/lgbm_{run_id}")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_id} → {save_dir}")

    # LightGBM datasets
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, free_raw_data=False)

    params = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "n_jobs": -1,
        "verbose": -1,
        "seed": 42,
    }

    print(f"\nTraining ({args.n_estimators} rounds, early_stop={args.early_stopping}) ...")
    t0 = time.time()

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=args.early_stopping),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    elapsed = time.time() - t0
    best_iter = model.best_iteration
    print(f"Training done in {elapsed:.1f}s  (best_iteration={best_iter})")

    # Evaluate
    print("\nEvaluation:")
    train_metrics = evaluate(model, X_train, y_train, "Train")
    val_metrics = evaluate(model, X_val, y_val, "Val")
    test_metrics = evaluate(model, X_test, y_test, "Test")

    # Pricer evaluation
    pricer_metrics = {}
    if (pricer is not None or jsu_pricer is not None) and df is not None:
        print("\nPricer evaluation (val set):")
        try:
            pricer_metrics = pricer_eval(
                model, stats["X_all"], df,
                stats["n_train"], stats["n_val"],
                stats["tgt_mean"], stats["tgt_std"],
                args.product, h_secs,
                pricer=pricer, jsu_pricer=jsu_pricer,
                period_len_min=args.pricer_period_min,
            )
        except Exception as e:
            import traceback
            print(f"  Pricer eval failed: {e}")
            traceback.print_exc()

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    fi = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    print(f"\nTop 20 features (gain):")
    for name, gain in fi[:20]:
        print(f"  {gain:12.1f}  {name}")

    # Save
    model_path = save_dir / f"lgbm_{args.product}_{h_label}.txt"
    model.save_model(str(model_path))
    print(f"\nModel saved: {model_path}")

    # Serialize pricer metrics (drop non-serializable keys)
    pricer_out = {k: v for k, v in pricer_metrics.items()
                  if not isinstance(v, np.ndarray)}

    results = {
        "run_id": run_id,
        "model": "LightGBM",
        "product": args.product,
        "horizon": h_label,
        "horizon_s": h_secs,
        "best_iteration": best_iter,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "pricer": pricer_out,
        "top_features": [{"name": n, "gain": float(g)} for n, g in fi[:50]],
        "params": params,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "stride": args.stride,
        "target_clip": args.target_clip,
        "tgt_mean": stats["tgt_mean"],
        "tgt_std": stats["tgt_std"],
        "elapsed_s": elapsed,
    }
    results_path = save_dir / f"results_{args.product}_{h_label}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
