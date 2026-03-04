"""
pricer_eval.py — Evaluate volatility models through the BSFourierPricer lens
═══════════════════════════════════════════════════════════════════════════════

Workflow:
  1. Load feature parquet + saved models from the pipeline
  2. Load BSFourierPricer from fourier_params.json
  3. For each "period" (e.g. hour), set strike = spot at period start
  4. At each minute within the period, use the appropriate horizon model
     (horizon = remaining minutes) to forecast forward RV
  5. Feed (spot, strike, forecasted_var) to the pricer → P(S_T > K)
  6. Compare against realised outcome (binary: did end-of-period price > strike?)
  7. Produce calibration plots, Brier scores, PnL analysis

Usage:
  python pricer_eval.py \\
      --features  data/features.parquet \\
      --models    data/models/models.pkl \\
      --pricer    fourier_params.json \\
      --inst      s \\
      --output    data/pricer_eval \\
      --smearing  none \\
      --period    60
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl


# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    features_path: str = ""
    models_path: str = ""
    pricer_path: str = ""
    seasonality_path: str = ""
    output_dir: str = "pricer_eval"

    # Which instrument to evaluate
    inst: str = "s"

    # Period length in agg_freq units (minutes if agg_freq=60s)
    # At minute n of the period, horizon = period_len - n
    period_len: int = 60

    # Smearing for RV retransformation
    smearing: str = "none"   # none | normal | duan_winsor

    # Train fraction (must match pipeline)
    train_frac: float = 0.8

    # Evaluate on: "test", "train", "all"
    eval_split: str = "test"

    # Price column names
    bid_col: str = "p_bid_0_price"
    ask_col: str = "p_ask_0_price"
    ts_col: str = "received_time"


# ═══════════════════════════════════════════════════════════
# Load models + pricer
# ═══════════════════════════════════════════════════════════

def load_models(path: str | Path) -> dict:
    """Load the models bundle saved by pipeline.fit_and_save_models."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"  Loaded {len(bundle['models'])} models from {path}")
    print(f"  Config: {bundle['config']}")
    return bundle


def load_pricer(path: str | Path):
    """Load BSFourierPricer. Import here to keep dependency optional."""
    from bs_fourier_pricer import BSFourierPricer
    pricer = BSFourierPricer.from_json(path)
    print(f"  Loaded pricer: {pricer}")
    return pricer


# ═══════════════════════════════════════════════════════════
# Predict forward RV for a single horizon model
# ═══════════════════════════════════════════════════════════

def predict_rv(
    df: pl.DataFrame,
    model: dict,
    smearing: str = "none",
) -> np.ndarray:
    """
    Predict deseasionalised forward RV (in level space) for every row.

    Returns exp(log_yhat) optionally adjusted by smearing.
    This is Σ(log diff)² for the forward horizon — exactly what
    BSFourierPricer.probability expects as forecasted_var.
    """
    feature_cols = model["feature_cols"]
    beta = model["beta"]

    X = df.select(feature_cols).to_numpy()
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    X_aug = np.hstack([ones, X])
    log_yhat = X_aug @ beta

    if smearing == "normal":
        return np.exp(log_yhat + 0.5 * model["sigma2"])
    elif smearing == "duan_winsor":
        return np.exp(log_yhat) * model["duan_winsor_factor"]
    else:
        return np.exp(log_yhat)


# ═══════════════════════════════════════════════════════════
# Core evaluation
# ═══════════════════════════════════════════════════════════

def run_eval(ecfg: EvalConfig):
    """
    Full evaluation: load data → assign periods → predict per-minute
    probabilities → score against realisations.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = Path(ecfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────
    print("[1/5] Loading data...")
    df = pl.read_parquet(ecfg.features_path)
    bundle = load_models(ecfg.models_path)
    models = bundle["models"]
    pcfg = bundle["config"]

    pricer = load_pricer(ecfg.pricer_path)

    # ── Split ─────────────────────────────────────────────
    n = df.height
    split_idx = int(n * ecfg.train_frac)
    if ecfg.eval_split == "test":
        df = df[split_idx:]
        label = "test"
    elif ecfg.eval_split == "train":
        df = df[:split_idx]
        label = "train"
    else:
        label = "all"
    print(f"  Evaluating on {label}: {df.height:,} rows")

    ts = ecfg.ts_col
    inst = ecfg.inst
    period_len = ecfg.period_len

    # ── Mid price (for spot) ──────────────────────────────
    mid_col = f"{inst}_mid"
    if mid_col in df.columns:
        df = df.rename({mid_col: "spot"})
    else:
        bid_col = f"{inst}_{ecfg.bid_col}"
        ask_col = f"{inst}_{ecfg.ask_col}"
        if bid_col in df.columns:
            df = df.with_columns(
                ((pl.col(bid_col) + pl.col(ask_col)) / 2).alias("spot")
            )
        else:
            raise ValueError(
                f"Cannot find price columns for {inst}. "
                f"Available: {[c for c in df.columns if inst in c][:10]}"
            )

    # ── Assign periods ────────────────────────────────────
    # Periods are non-overlapping windows of `period_len` rows.
    # Within a period, minute_in_period goes 0..period_len-1.
    # At minute m, remaining horizon = period_len - m.
    print("[2/5] Assigning periods...")
    timestamps = df.get_column(ts)
    minute_of_hour = timestamps.dt.minute().to_numpy()
    hour_block = timestamps.dt.truncate(f"{period_len}m")

    df = df.with_columns([
        hour_block.alias("period_start"),
        pl.Series("minute_in_period", minute_of_hour % period_len),
    ])

    # End-of-period price and strike (spot at period start)
    period_agg = (
        df.group_by("period_start")
        .agg([
            pl.col("spot").first().alias("strike"),
            pl.col("spot").last().alias("end_price"),
            pl.col("spot").count().alias("period_rows"),
        ])
    )
    df = df.join(period_agg, on="period_start", how="left")

    # Only keep complete periods
    df = df.filter(pl.col("period_rows") == period_len)

    # Realised binary outcome: did price end above strike?
    df = df.with_columns(
        (pl.col("end_price") > pl.col("strike")).cast(pl.Float64).alias("realised")
    )

    print(f"  Complete periods: {df.select('period_start').n_unique()}")
    print(f"  Rows: {df.height:,}")

    # ── Per-row predictions ───────────────────────────────
    print("[3/5] Generating predictions...")
    spot = df.get_column("spot").to_numpy()
    strike = df.get_column("strike").to_numpy()
    mip = df.get_column("minute_in_period").to_numpy()
    realised = df.get_column("realised").to_numpy()

    # Seasonality factor for reseasonalisation
    seas_col = f"{inst}_seas_factor"
    has_seas = seas_col in df.columns
    if has_seas:
        seas_factor = df.get_column(seas_col).to_numpy()
    else:
        seas_factor = np.ones(df.height, dtype=np.float64)

    prob_cal = np.full(df.height, np.nan, dtype=np.float64)
    prob_bs = np.full(df.height, np.nan, dtype=np.float64)
    fcast_var = np.full(df.height, np.nan, dtype=np.float64)

    # Group by horizon (= period_len - minute_in_period)
    for m in range(period_len):
        h = period_len - m
        model_key = (inst, h)
        if model_key not in models:
            continue

        model = models[model_key]
        mask = mip == m

        if mask.sum() == 0:
            continue

        df_slice = df.filter(pl.col("minute_in_period") == m)

        # Check all feature columns exist
        missing = [c for c in model["feature_cols"] if c not in df_slice.columns]
        if missing:
            print(f"  WARNING: h={h} missing features: {missing[:3]}...")
            continue

        # Predict deseasonalised RV (level space)
        rv_deseas = predict_rv(df_slice, model, smearing=ecfg.smearing)

        # Reseasonalise: multiply by seasonality factor.
        # For forward-looking RV over h minutes, use the forward seas sum
        # if available, otherwise approximate with current seas_factor.
        fwd_seas_col = f"{inst}_fwd_seas_h{h}"
        if fwd_seas_col in df_slice.columns:
            fwd_s = df_slice.get_column(fwd_seas_col).to_numpy()
            forecasted_var = rv_deseas * (fwd_s / h)
        else:
            s_f = df_slice.get_column(seas_col).to_numpy() if has_seas else 1.0
            forecasted_var = rv_deseas * s_f

        # TTE in seconds
        agg_freq_s = pcfg.get("agg_freq_s", 60) if isinstance(pcfg, dict) else 60
        tte_s = float(h * agg_freq_s)

        # Price via BSFourierPricer
        spot_slice = df_slice.get_column("spot").to_numpy()
        strike_slice = df_slice.get_column("strike").to_numpy()

        p_c, p_b = pricer.probability(
            spot=spot_slice,
            strike=strike_slice,
            forecasted_var=forecasted_var,
            tte_seconds=tte_s,
            return_bs_raw=True,
        )

        # Write back into full arrays
        idx = np.where(mask)[0]
        prob_cal[idx] = p_c.ravel()
        prob_bs[idx] = p_b.ravel()
        fcast_var[idx] = forecasted_var.ravel()

    df = df.with_columns([
        pl.Series("prob_cal", prob_cal),
        pl.Series("prob_bs", prob_bs),
        pl.Series("forecasted_var", fcast_var),
    ])

    # Drop rows we couldn't predict
    valid = np.isfinite(prob_cal)
    print(f"  Valid predictions: {valid.sum():,} / {df.height:,}")
    df_valid = df.filter(pl.Series("__valid", valid))

    # ── Scoring ───────────────────────────────────────────
    print("[4/5] Scoring...")
    y = df_valid.get_column("realised").to_numpy()
    p_cal = df_valid.get_column("prob_cal").to_numpy()
    p_bs = df_valid.get_column("prob_bs").to_numpy()
    mip_v = df_valid.get_column("minute_in_period").to_numpy()

    # Brier score: mean((p - y)²), lower is better
    brier_cal = np.mean((p_cal - y) ** 2)
    brier_bs = np.mean((p_bs - y) ** 2)
    brier_naive = np.mean((y.mean() - y) ** 2)

    # Log loss
    eps = 1e-8
    ll_cal = -np.mean(y * np.log(np.clip(p_cal, eps, 1-eps))
                      + (1-y) * np.log(np.clip(1-p_cal, eps, 1-eps)))
    ll_bs = -np.mean(y * np.log(np.clip(p_bs, eps, 1-eps))
                     + (1-y) * np.log(np.clip(1-p_bs, eps, 1-eps)))
    ll_naive = -np.mean(y * np.log(np.clip(y.mean(), eps, 1-eps))
                        + (1-y) * np.log(np.clip(1-y.mean(), eps, 1-eps)))

    # Per-minute Brier
    minute_brier_cal = {}
    minute_brier_bs = {}
    for m in range(period_len):
        mask = mip_v == m
        if mask.sum() < 10:
            continue
        minute_brier_cal[m] = np.mean((p_cal[mask] - y[mask]) ** 2)
        minute_brier_bs[m] = np.mean((p_bs[mask] - y[mask]) ** 2)

    print(f"\n{'='*60}")
    print(f"  PRICER EVALUATION  ({label} set, inst={inst}, "
          f"smearing={ecfg.smearing})")
    print(f"{'='*60}")
    print(f"  Periods:     {df_valid.select('period_start').n_unique():,}")
    print(f"  Predictions: {valid.sum():,}")
    print(f"  Base rate:   {y.mean():.4f} "
          f"(fraction of periods ending above strike)")
    print(f"\n  {'Metric':<20} {'Calibrated':>12} {'BS raw':>12} {'Naive':>12}")
    print(f"  {'─'*56}")
    print(f"  {'Brier score':<20} {brier_cal:>12.6f} {brier_bs:>12.6f} "
          f"{brier_naive:>12.6f}")
    print(f"  {'Log loss':<20} {ll_cal:>12.6f} {ll_bs:>12.6f} "
          f"{ll_naive:>12.6f}")
    print(f"  {'Brier skill (vs N)':<20} "
          f"{1 - brier_cal/brier_naive:>12.4f} "
          f"{1 - brier_bs/brier_naive:>12.4f} {'—':>12}")

    # ── Save predictions ──────────────────────────────────
    df_valid.write_parquet(output_dir / "predictions.parquet")

    # ── Plots ─────────────────────────────────────────────
    print("[5/5] Plotting...")

    with PdfPages(output_dir / "pricer_eval.pdf") as pdf:

        # 1. Calibration curve (reliability diagram)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, probs, name in [(axes[0], p_cal, "Calibrated"),
                                 (axes[1], p_bs, "BS raw")]:
            n_bins = 20
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_idx = np.digitize(probs, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)

            bin_means = np.array([
                y[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else np.nan
                for i in range(n_bins)
            ])
            bin_counts = np.array([
                (bin_idx == i).sum() for i in range(n_bins)
            ])

            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
            ax.bar(bin_centres, bin_counts / bin_counts.sum(), width=0.04,
                   alpha=0.2, color="grey", label="Density")
            ax2 = ax.twinx()
            valid_bins = np.isfinite(bin_means)
            ax2.plot(bin_centres[valid_bins], bin_means[valid_bins],
                     "o-", color="tab:blue", markersize=4)
            ax2.set_ylabel("Observed frequency")
            ax2.set_ylim(-0.05, 1.05)
            ax.set_xlabel(f"Predicted probability ({name})")
            ax.set_title(f"{name}: Calibration diagram")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # 2. Brier score by minute-in-period
        fig, ax = plt.subplots(figsize=(12, 5))
        mins = sorted(minute_brier_cal.keys())
        cal_vals = [minute_brier_cal[m] for m in mins]
        bs_vals = [minute_brier_bs[m] for m in mins]
        naive_line = brier_naive

        ax.plot(mins, cal_vals, "o-", label="Calibrated", markersize=3)
        ax.plot(mins, bs_vals, "s-", label="BS raw", markersize=3, alpha=0.6)
        ax.axhline(naive_line, color="grey", linestyle=":", label="Naive")
        ax.set_xlabel("Minute in period (0 = period start)")
        ax.set_ylabel("Brier score (lower = better)")
        ax.set_title("Brier score by minute within period")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # 3. Probability distribution by outcome
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, probs, name in [(axes[0], p_cal, "Calibrated"),
                                 (axes[1], p_bs, "BS raw")]:
            ax.hist(probs[y == 1], bins=50, alpha=0.6, density=True,
                    label="End > strike", color="tab:green")
            ax.hist(probs[y == 0], bins=50, alpha=0.6, density=True,
                    label="End ≤ strike", color="tab:red")
            ax.set_xlabel(f"Predicted probability ({name})")
            ax.set_ylabel("Density")
            ax.set_title(f"{name}: Probability by outcome")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # 4. Probability over time within periods (sample 20 periods)
        period_starts = df_valid.get_column("period_start").unique().sort()
        n_sample = min(20, len(period_starts))
        sample_periods = period_starts.sample(n_sample, seed=42).sort()

        fig, axes = plt.subplots(4, 5, figsize=(20, 12), sharex=True, sharey=True)
        axes_flat = axes.flatten()

        for i, ps in enumerate(sample_periods.to_list()):
            if i >= len(axes_flat):
                break
            period_df = df_valid.filter(pl.col("period_start") == ps).sort(ts)
            m = period_df.get_column("minute_in_period").to_numpy()
            pc = period_df.get_column("prob_cal").to_numpy()
            pb = period_df.get_column("prob_bs").to_numpy()
            outcome = period_df.get_column("realised").to_numpy()[0]

            ax = axes_flat[i]
            ax.plot(m, pc, label="Cal", color="tab:blue", linewidth=1.5)
            ax.plot(m, pb, label="BS", color="tab:orange",
                    linewidth=1, alpha=0.6)
            ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":")
            bg = "#d5f5e3" if outcome > 0.5 else "#f5b7b1"
            ax.set_facecolor(bg)
            ax.set_title(f"{str(ps)[:16]}", fontsize=7)
            if i == 0:
                ax.legend(fontsize=6)

        for ax in axes_flat:
            ax.set_ylim(-0.05, 1.05)
        fig.suptitle("Probability evolution within sample periods "
                     "(green=above strike, red=below)", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # 5. Forecasted variance distribution by minute
        fv_valid = df_valid.get_column("forecasted_var").to_numpy()
        fig, ax = plt.subplots(figsize=(12, 5))
        for m in [0, 15, 30, 45, 55]:
            mask = mip_v == m
            if mask.sum() < 10:
                continue
            ax.hist(np.log10(fv_valid[mask] + 1e-20), bins=50, alpha=0.4,
                    density=True, label=f"minute={m} (h={period_len-m})")
        ax.set_xlabel("log₁₀(forecasted_var)")
        ax.set_ylabel("Density")
        ax.set_title("Forecasted variance distribution by minute-in-period")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # 6. Cumulative PnL from simple strategy
        # Strategy: bet proportionally to (p - 0.5) each period
        fig, ax = plt.subplots(figsize=(14, 5))

        # For each row: position = p_cal - 0.5,  payoff = realised - 0.5
        position = p_cal - 0.5
        payoff = y - 0.5
        pnl_per_row = position * payoff
        cumulative_pnl = np.cumsum(pnl_per_row)

        position_bs = p_bs - 0.5
        pnl_bs = position_bs * payoff
        cum_pnl_bs = np.cumsum(pnl_bs)

        ax.plot(cumulative_pnl, label="Calibrated", linewidth=1)
        ax.plot(cum_pnl_bs, label="BS raw", linewidth=1, alpha=0.6)
        ax.set_xlabel("Row index")
        ax.set_ylabel("Cumulative PnL (proportional betting)")
        ax.set_title("Cumulative PnL: bet ∝ (p − 0.5)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sharpe annotation
        sharpe_cal = (np.mean(pnl_per_row) / np.std(pnl_per_row)
                      * np.sqrt(252 * 24) if np.std(pnl_per_row) > 0 else 0)
        sharpe_bs = (np.mean(pnl_bs) / np.std(pnl_bs)
                     * np.sqrt(252 * 24) if np.std(pnl_bs) > 0 else 0)
        ax.text(0.02, 0.95,
                f"Sharpe (ann.): Cal={sharpe_cal:.2f}  BS={sharpe_bs:.2f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    print(f"\n✓ Saved pricer_eval.pdf    → {output_dir / 'pricer_eval.pdf'}")
    print(f"✓ Saved predictions.parquet → {output_dir / 'predictions.parquet'}")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate vol models via BSFourierPricer",
    )
    parser.add_argument("--features", required=True,
                        help="Feature parquet from pipeline")
    parser.add_argument("--models", required=True,
                        help="models.pkl from pipeline")
    parser.add_argument("--pricer", required=True,
                        help="fourier_params.json for BSFourierPricer")
    parser.add_argument("--inst", default="s",
                        help="Instrument prefix (default: s)")
    parser.add_argument("--output", default="pricer_eval",
                        help="Output directory")
    parser.add_argument("--smearing", default="none",
                        choices=["none", "normal", "duan_winsor"],
                        help="Smearing method for RV retransformation")
    parser.add_argument("--period", type=int, default=60,
                        help="Period length in agg-freq units (default: 60)")
    parser.add_argument("--split", default="test",
                        choices=["train", "test", "all"],
                        help="Which data split to evaluate on")
    parser.add_argument("--train-frac", type=float, default=0.8)

    args = parser.parse_args()

    ecfg = EvalConfig(
        features_path=args.features,
        models_path=args.models,
        pricer_path=args.pricer,
        inst=args.inst,
        output_dir=args.output,
        smearing=args.smearing,
        period_len=args.period,
        eval_split=args.split,
        train_frac=args.train_frac,
    )

    run_eval(ecfg)
