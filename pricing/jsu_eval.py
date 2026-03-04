"""
jsu_eval.py — Evaluate JSU pricer (+ optional BSFourier comparison)
═══════════════════════════════════════════════════════════════════════════════

The JSU pricer is self-contained: it only needs spot, strike, and tte_ms.
No vol models required.

Optionally, if BSFourier pricer + vol models are supplied, we compare
both pricers side-by-side on the same data.

Usage:
  # JSU only
  python jsu_eval.py \\
      --data       data/features.parquet \\
      --jsu-params params1.csv \\
      --inst       s \\
      --period     15 \\
      --output     data/jsu_eval

  # JSU + BSFourier comparison
  python jsu_eval.py \\
      --data         data/features.parquet \\
      --jsu-params   params1.csv \\
      --bs-params    fourier_params.json \\
      --models       data/models/models.pkl \\
      --inst         s \\
      --period       15 \\
      --smearing     none \\
      --output       data/jsu_eval
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl


# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    data_path: str = ""
    jsu_params_path: str = ""

    # Optional BSFourier comparison
    bs_params_path: Optional[str] = None
    models_path: Optional[str] = None
    smearing: str = "none"

    output_dir: str = "jsu_eval"
    inst: str = "s"

    # Period length in minutes — must match JSU param range
    # (JSU params1.csv covers 200ms–900s = 15 min)
    period_min: int = 15

    ts_col: str = "received_time"
    train_frac: float = 0.8
    eval_split: str = "test"
    jsu_strategy: str = "clamp"


# ═══════════════════════════════════════════════════════════
# Scoring helpers
# ═══════════════════════════════════════════════════════════

def _brier(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() < 1:
        return np.nan
    return float(np.mean((p[mask] - y[mask]) ** 2))


def _log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-8) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() < 1:
        return np.nan
    y, p = y[mask], np.clip(p[mask], eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 20):
    edges = np.linspace(0, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    obs = np.array([y[idx == i].mean() if (idx == i).sum() > 0 else np.nan
                    for i in range(n_bins)])
    counts = np.array([(idx == i).sum() for i in range(n_bins)])
    return centres, obs, counts


# ═══════════════════════════════════════════════════════════
# BSFourier prediction helper (optional)
# ═══════════════════════════════════════════════════════════

def _predict_bs(df, models, bs_pricer, inst, period_min, agg_freq_s, smearing):
    """Predict BSFourier probabilities for each row using the matching horizon model."""
    n = df.height
    prob_cal = np.full(n, np.nan, dtype=np.float64)
    prob_raw = np.full(n, np.nan, dtype=np.float64)
    mip = df.get_column("minute_in_period").to_numpy()

    for m in range(period_min):
        h = period_min - m
        key = (inst, h)
        if key not in models:
            continue
        model = models[key]
        mask = mip == m
        if mask.sum() == 0:
            continue

        df_slice = df.filter(pl.col("minute_in_period") == m)
        if any(c not in df_slice.columns for c in model["feature_cols"]):
            continue

        # Predict log RV → level space
        X = df_slice.select(model["feature_cols"]).to_numpy()
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        log_yhat = np.hstack([ones, X]) @ model["beta"]

        if smearing == "normal":
            rv = np.exp(log_yhat + 0.5 * model["sigma2"])
        elif smearing == "duan_winsor":
            rv = np.exp(log_yhat) * model["duan_winsor_factor"]
        else:
            rv = np.exp(log_yhat)

        # Reseasonalise
        fwd_col = f"{inst}_fwd_seas_h{h}"
        if fwd_col in df_slice.columns:
            fwd_s = df_slice.get_column(fwd_col).to_numpy()
            fvar = rv * (fwd_s / h)
        else:
            seas_col = f"{inst}_seas_factor"
            sf = df_slice.get_column(seas_col).to_numpy() if seas_col in df_slice.columns else 1.0
            fvar = rv * sf

        tte_s = float(h * agg_freq_s)
        pc, pb = bs_pricer.probability(
            spot=df_slice.get_column("spot").to_numpy(),
            strike=df_slice.get_column("strike").to_numpy(),
            forecasted_var=fvar,
            tte_seconds=tte_s,
            return_bs_raw=True,
        )
        idx = np.where(mask)[0]
        prob_cal[idx] = pc.ravel()
        prob_raw[idx] = pb.ravel()

    return prob_cal, prob_raw


# ═══════════════════════════════════════════════════════════
# Main eval
# ═══════════════════════════════════════════════════════════

def run_eval(cfg: EvalConfig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────
    print("[1/6] Loading...")
    df = pl.read_parquet(cfg.data_path)

    from jsu_pricer import JSUPricer
    jsu = JSUPricer.from_csv(cfg.jsu_params_path,
                              short_tte_strategy=cfg.jsu_strategy)

    has_bs = cfg.bs_params_path is not None and cfg.models_path is not None
    bs_pricer, bs_models, pcfg = None, None, None
    if has_bs:
        from bs_fourier_pricer import BSFourierPricer
        bs_pricer = BSFourierPricer.from_json(cfg.bs_params_path)
        with open(cfg.models_path, "rb") as f:
            bundle = pickle.load(f)
        bs_models = bundle["models"]
        pcfg = bundle["config"]
        print(f"  BSFourier: {bs_pricer}, {len(bs_models)} vol models")

    # ── 2. Split + spot ───────────────────────────────────
    n = df.height
    split_idx = int(n * cfg.train_frac)
    if cfg.eval_split == "test":
        df = df[split_idx:]
    elif cfg.eval_split == "train":
        df = df[:split_idx]
    print(f"  Split: {cfg.eval_split}, {df.height:,} rows")

    inst = cfg.inst
    mid_col = f"{inst}_mid"
    if mid_col not in df.columns:
        raise ValueError(f"{mid_col} not in columns. Re-run pipeline to include midprices.")
    df = df.rename({mid_col: "spot"})

    # ── 3. Assign periods ─────────────────────────────────
    print("[2/6] Assigning periods...")
    ts = cfg.ts_col
    period_min = cfg.period_min
    timestamps = df.get_column(ts)

    period_start = timestamps.dt.truncate(f"{period_min}m")
    period_start_epoch = period_start.dt.epoch("ms")
    ts_epoch = timestamps.dt.epoch("ms")
    minute_in_period = ((ts_epoch - period_start_epoch) / 60_000).cast(pl.Int32)

    df = df.with_columns([
        period_start.alias("period_start"),
        minute_in_period.alias("minute_in_period"),
    ])

    period_agg = (
        df.group_by("period_start").agg([
            pl.col("spot").first().alias("strike"),
            pl.col("spot").last().alias("end_price"),
            pl.col("spot").count().alias("period_rows"),
        ])
    )
    df = df.join(period_agg, on="period_start", how="left")
    df = df.filter(pl.col("period_rows") == period_min)
    df = df.with_columns(
        (pl.col("end_price") > pl.col("strike")).cast(pl.Float64).alias("realised")
    )

    n_periods = df.select("period_start").n_unique()
    print(f"  Complete periods: {n_periods:,}, rows: {df.height:,}")

    # ── 4. Predictions ────────────────────────────────────
    print("[3/6] JSU predictions...")
    spot = df.get_column("spot").to_numpy()
    strike = df.get_column("strike").to_numpy()
    mip = df.get_column("minute_in_period").to_numpy()
    y = df.get_column("realised").to_numpy()

    # JSU: tte_ms = remaining minutes × 60,000
    tte_ms = (period_min - mip).astype(np.float64) * 60_000.0
    prob_jsu = jsu.probability(spot=spot, strike=strike, tte_ms=tte_ms)
    print(f"  JSU valid: {np.isfinite(prob_jsu).sum():,}")

    # BSFourier (optional)
    prob_bs_cal, prob_bs_raw = None, None
    if has_bs:
        print("[4/6] BSFourier predictions...")
        agg_freq_s = pcfg["agg_freq_s"] if isinstance(pcfg, dict) else 60
        prob_bs_cal, prob_bs_raw = _predict_bs(
            df, bs_models, bs_pricer, inst, period_min, agg_freq_s, cfg.smearing,
        )
        print(f"  BSFourier valid: {np.isfinite(prob_bs_cal).sum():,}")
    else:
        print("[4/6] BSFourier: skipped")

    # ── 5. Scoring ────────────────────────────────────────
    print("[5/6] Scoring...")

    brier_naive = _brier(y, np.full_like(y, y.mean()))
    ll_naive = _log_loss(y, np.full_like(y, y.mean()))

    # All pricers to evaluate
    pricers = {"JSU": prob_jsu}
    if has_bs:
        pricers["BS_cal"] = prob_bs_cal
        pricers["BS_raw"] = prob_bs_raw

    scores = {}
    for name, p in pricers.items():
        mask = np.isfinite(p)
        if mask.sum() < 10:
            continue
        b = _brier(y[mask], p[mask])
        ll = _log_loss(y[mask], p[mask])
        b_naive_sub = _brier(y[mask], np.full(mask.sum(), y[mask].mean()))
        scores[name] = {
            "brier": b, "logloss": ll,
            "skill": 1 - b / b_naive_sub,
            "n": int(mask.sum()),
        }

    # Per-minute Brier
    minute_brier = {}
    for name, p in pricers.items():
        mb = {}
        for m in range(period_min):
            mask = (mip == m) & np.isfinite(p)
            if mask.sum() < 10:
                continue
            mb[m] = _brier(y[mask], p[mask])
        minute_brier[name] = mb

    # Console output
    print(f"\n{'='*70}")
    print(f"  PRICER EVALUATION  ({cfg.eval_split}, inst={inst}, "
          f"period={period_min}m, {n_periods:,} periods)")
    print(f"{'='*70}")
    print(f"  Base rate: {y.mean():.4f}")
    print(f"\n  {'Model':<15} {'Brier':>10} {'LogLoss':>10} "
          f"{'Skill':>10} {'N':>10}")
    print(f"  {'─'*55}")
    print(f"  {'Naive':<15} {brier_naive:>10.6f} {ll_naive:>10.6f} "
          f"{'—':>10} {len(y):>10,}")
    for name, s in scores.items():
        print(f"  {name:<15} {s['brier']:>10.6f} {s['logloss']:>10.6f} "
              f"{s['skill']:>+10.4f} {s['n']:>10,}")

    # Per-minute summary
    for name, mb in minute_brier.items():
        if not mb:
            continue
        vals = np.array(list(mb.values()))
        best_m = min(mb, key=mb.get)
        worst_m = max(mb, key=mb.get)
        print(f"\n  {name} by minute:  best={best_m} ({mb[best_m]:.6f})  "
              f"worst={worst_m} ({mb[worst_m]:.6f})  "
              f"mean={vals.mean():.6f}")

    # ── 6. Plots ──────────────────────────────────────────
    print("[6/6] Plotting...")
    colors = {"JSU": "#e67e22", "BS_cal": "#2980b9", "BS_raw": "#95a5a6"}
    hour_of_day = df.get_column(ts).dt.hour().to_numpy()

    # Save predictions
    out_cols = [pl.Series("prob_jsu", prob_jsu)]
    if has_bs:
        out_cols += [pl.Series("prob_bs_cal", prob_bs_cal),
                     pl.Series("prob_bs_raw", prob_bs_raw)]
    df.with_columns(out_cols).write_parquet(output_dir / "predictions.parquet")

    with PdfPages(output_dir / "pricer_eval.pdf") as pdf:

        # ── 1. Calibration diagrams ──────────────────────
        n_p = len(pricers)
        fig, axes = plt.subplots(1, n_p, figsize=(6 * n_p, 6), squeeze=False)
        for i, (name, p) in enumerate(pricers.items()):
            mask = np.isfinite(p)
            centres, obs, counts = _calibration_bins(y[mask], p[mask])
            ax = axes.flat[i]
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
            ax.bar(centres, counts / max(counts.sum(), 1), width=0.04,
                   alpha=0.15, color="grey")
            ax2 = ax.twinx()
            v = np.isfinite(obs)
            ax2.plot(centres[v], obs[v], "o-",
                     color=colors.get(name, "tab:blue"), markersize=4)
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_ylabel("Observed frequency")
            ax.set_xlabel("Predicted probability")
            ax.set_title(f"{name}: Calibration")
            ax.set_xlim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── 2. Brier by minute (all pricers overlaid) ────
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, mb in minute_brier.items():
            if not mb:
                continue
            mins = sorted(mb.keys())
            ax.plot(mins, [mb[m] for m in mins], "o-", label=name,
                    markersize=3, color=colors.get(name))
        ax.axhline(brier_naive, color="grey", linestyle=":", label="Naive")
        ax.set_xlabel("Minute in period (0 = start)")
        ax.set_ylabel("Brier score (lower = better)")
        ax.set_title(f"Brier score by minute within {period_min}m period")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── 3. Probability distributions by outcome ──────
        fig, axes = plt.subplots(1, n_p, figsize=(6 * n_p, 5), squeeze=False)
        for i, (name, p) in enumerate(pricers.items()):
            mask = np.isfinite(p)
            ax = axes.flat[i]
            ax.hist(p[mask & (y == 1)], bins=50, alpha=0.6, density=True,
                    label="S_T > K", color="tab:green")
            ax.hist(p[mask & (y == 0)], bins=50, alpha=0.6, density=True,
                    label="S_T ≤ K", color="tab:red")
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Density")
            ax.set_title(f"{name}: Discrimination")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── 4. Brier heatmap: minute × hour-of-day ──────
        for name, p in pricers.items():
            mask = np.isfinite(p)
            if mask.sum() < 100:
                continue

            matrix = np.full((period_min, 24), np.nan)
            for m in range(period_min):
                for h in range(24):
                    sel = mask & (mip == m) & (hour_of_day == h)
                    if sel.sum() < 5:
                        continue
                    matrix[m, h] = _brier(y[sel], p[sel])

            fig, ax = plt.subplots(figsize=(14, 6))
            vmin = np.nanmin(matrix) if np.any(np.isfinite(matrix)) else 0
            vmax = np.nanmax(matrix) if np.any(np.isfinite(matrix)) else 0.3
            im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                           vmin=vmin, vmax=vmax)
            ax.set_xlabel("Hour of day (UTC)")
            ax.set_ylabel("Minute in period")
            ax.set_title(f"{name}: Brier — Minute × Hour-of-Day")
            ax.set_xticks(range(24))
            plt.colorbar(im, ax=ax, label="Brier", shrink=0.8)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # ── 5. Sample period traces ──────────────────────
        period_starts = df.get_column("period_start").unique().sort()
        n_sample = min(20, len(period_starts))
        sample_ps = period_starts.sample(n_sample, seed=42).sort()

        fig, axes = plt.subplots(4, 5, figsize=(20, 12),
                                 sharex=True, sharey=True)
        for i, ps in enumerate(sample_ps.to_list()):
            if i >= 20:
                break
            ax = axes.flat[i]
            pdata = df.filter(pl.col("period_start") == ps).sort(ts)
            m_arr = pdata.get_column("minute_in_period").to_numpy()
            spot_arr = pdata.get_column("spot").to_numpy()
            strike_arr = pdata.get_column("strike").to_numpy()
            outcome = pdata.get_column("realised").to_numpy()[0]

            # JSU trace
            tte_local = (period_min - m_arr).astype(np.float64) * 60_000
            p_jsu = jsu.probability(spot_arr, strike_arr, tte_local)
            ax.plot(m_arr, p_jsu, color="#e67e22", linewidth=1.5, label="JSU")

            # BSFourier trace (recompute per slice to avoid index gymnastics)
            if has_bs:
                agg_freq_s = pcfg["agg_freq_s"] if isinstance(pcfg, dict) else 60
                bs_trace = np.full_like(m_arr, np.nan, dtype=np.float64)
                for m_val in np.unique(m_arr):
                    h = period_min - int(m_val)
                    key = (inst, h)
                    if key not in bs_models:
                        continue
                    model = bs_models[key]
                    sl = pdata.filter(pl.col("minute_in_period") == m_val)
                    if any(c not in sl.columns for c in model["feature_cols"]):
                        continue
                    X = sl.select(model["feature_cols"]).to_numpy()
                    ones = np.ones((X.shape[0], 1))
                    log_yh = np.hstack([ones, X]) @ model["beta"]
                    rv = np.exp(log_yh)
                    fwd_col = f"{inst}_fwd_seas_h{h}"
                    if fwd_col in sl.columns:
                        fv = rv * (sl.get_column(fwd_col).to_numpy() / h)
                    else:
                        fv = rv
                    pc, _ = bs_pricer.probability(
                        sl.get_column("spot").to_numpy(),
                        sl.get_column("strike").to_numpy(),
                        fv, tte_seconds=float(h * agg_freq_s),
                        return_bs_raw=True,
                    )
                    row_mask = m_arr == m_val
                    bs_trace[row_mask] = pc.ravel()

                valid_bs = np.isfinite(bs_trace)
                if valid_bs.any():
                    ax.plot(m_arr[valid_bs], bs_trace[valid_bs],
                            color="#2980b9", linewidth=1, alpha=0.7,
                            label="BS_cal")

            # Spot normalised (secondary axis)
            ax_r = ax.twinx()
            moneyness = (spot_arr / strike_arr[0] - 1) * 100
            ax_r.plot(m_arr, moneyness, color="black", linewidth=0.5,
                      alpha=0.4, linestyle="--")
            ax_r.axhline(0, color="black", linewidth=0.3, alpha=0.3)
            ax_r.set_ylabel("M%", fontsize=5)
            ax_r.tick_params(labelsize=5)

            ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":")
            bg = "#d5f5e3" if outcome > 0.5 else "#f5b7b1"
            ax.set_facecolor(bg)
            ax.set_title(f"{str(ps)[:16]}", fontsize=7)
            if i == 0:
                ax.legend(fontsize=6)

        for ax in axes.flat:
            ax.set_ylim(-0.05, 1.05)
        fig.suptitle(f"Probability traces — sample {period_min}m periods "
                     f"(green bg = above strike, red bg = below)", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── 6. JSU vs BSFourier head-to-head ─────────────
        if has_bs:
            both = np.isfinite(prob_jsu) & np.isfinite(prob_bs_cal)
            if both.sum() > 100:

                # Scatter
                fig, ax = plt.subplots(figsize=(7, 7))
                n_pts = min(5000, both.sum())
                rng = np.random.default_rng(42)
                idx = rng.choice(np.where(both)[0], n_pts, replace=False)
                c = np.where(y[idx] > 0.5, "#2ecc71", "#e74c3c")
                ax.scatter(prob_jsu[idx], prob_bs_cal[idx], c=c,
                           alpha=0.3, s=8, edgecolors="none")
                ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
                ax.set_xlabel("JSU probability")
                ax.set_ylabel("BSFourier probability")
                ax.set_title("JSU vs BSFourier (green=above, red=below)")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)

                # Brier difference bar chart
                fig, ax = plt.subplots(figsize=(12, 5))
                for m in range(period_min):
                    mask = (mip == m) & both
                    if mask.sum() < 10:
                        continue
                    b_jsu = _brier(y[mask], prob_jsu[mask])
                    b_bs = _brier(y[mask], prob_bs_cal[mask])
                    diff = b_bs - b_jsu   # positive = JSU wins
                    color = "#e67e22" if diff > 0 else "#2980b9"
                    ax.bar(m, diff, color=color, edgecolor="white",
                           linewidth=0.3)
                ax.axhline(0, color="grey", linewidth=0.5)
                ax.set_xlabel("Minute in period")
                ax.set_ylabel("Brier(BS) − Brier(JSU)")
                ax.set_title("Brier difference (orange = JSU better, "
                             "blue = BS better)")
                ax.grid(True, axis="y", alpha=0.3)
                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)

        # ── 7. JSU param sensitivity ─────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ATM probability vs TTE
        tte_sweep = np.arange(1, period_min + 1) * 60_000.0
        mean_spot = np.nanmean(spot)
        atm_probs = jsu.probability(
            np.full_like(tte_sweep, mean_spot),
            np.full_like(tte_sweep, mean_spot),
            tte_sweep,
        )
        axes[0].plot(tte_sweep / 60_000, atm_probs, "o-",
                     color="#e67e22", markersize=4)
        axes[0].axhline(0.5, color="grey", linewidth=0.5, linestyle=":")
        axes[0].set_xlabel("TTE (minutes)")
        axes[0].set_ylabel("P(S > K) at ATM")
        axes[0].set_title("JSU ATM probability vs TTE")
        axes[0].grid(True, alpha=0.3)

        # Moneyness smile at different TTEs
        for tte_m in [1, 5, 10, period_min]:
            tte_v = float(tte_m * 60_000)
            pcts = np.linspace(-2, 2, 100)
            strikes_sweep = mean_spot * (1 + pcts / 100)
            probs = jsu.probability(
                np.full_like(strikes_sweep, mean_spot),
                strikes_sweep,
                np.full_like(strikes_sweep, tte_v),
            )
            axes[1].plot(pcts, probs, label=f"TTE={tte_m}m")
        axes[1].axhline(0.5, color="grey", linewidth=0.5, linestyle=":")
        axes[1].axvline(0, color="grey", linewidth=0.5, linestyle=":")
        axes[1].set_xlabel("Moneyness (%)")
        axes[1].set_ylabel("P(S > K)")
        axes[1].set_title("JSU probability smile")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ── 8. Sharpness: prediction spread over time ────
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, p in pricers.items():
            mask = np.isfinite(p)
            std_by_min = []
            for m in range(period_min):
                sel = mask & (mip == m)
                std_by_min.append(np.std(p[sel]) if sel.sum() > 10 else np.nan)
            ax.plot(range(period_min), std_by_min, "o-", label=name,
                    markersize=3, color=colors.get(name))
        ax.set_xlabel("Minute in period")
        ax.set_ylabel("Std of predictions")
        ax.set_title("Sharpness: prediction spread by minute "
                     "(higher = more decisive)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    print(f"\n✓ Saved pricer_eval.pdf     → {output_dir / 'pricer_eval.pdf'}")
    print(f"✓ Saved predictions.parquet → {output_dir / 'predictions.parquet'}")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate JSU pricer (+ optional BSFourier comparison)")
    parser.add_argument("--data", required=True,
                        help="Feature parquet (needs {inst}_mid column)")
    parser.add_argument("--jsu-params", required=True,
                        help="params1.csv for JSU pricer")
    parser.add_argument("--jsu-strategy", default="clamp",
                        choices=["clamp", "interpolate", "intrinsic"])

    parser.add_argument("--bs-params", default=None,
                        help="fourier_params.json (omit to skip)")
    parser.add_argument("--models", default=None,
                        help="models.pkl (required with --bs-params)")
    parser.add_argument("--smearing", default="none",
                        choices=["none", "normal", "duan_winsor"])

    parser.add_argument("--inst", default="s")
    parser.add_argument("--period", type=int, default=15)
    parser.add_argument("--output", default="jsu_eval")
    parser.add_argument("--split", default="test",
                        choices=["train", "test", "all"])
    parser.add_argument("--train-frac", type=float, default=0.8)

    args = parser.parse_args()

    if args.bs_params and not args.models:
        parser.error("--models required when --bs-params is provided")

    ecfg = EvalConfig(
        data_path=args.data,
        jsu_params_path=args.jsu_params,
        bs_params_path=args.bs_params,
        models_path=args.models,
        smearing=args.smearing,
        output_dir=args.output,
        inst=args.inst,
        period_min=args.period,
        eval_split=args.split,
        train_frac=args.train_frac,
        jsu_strategy=args.jsu_strategy,
    )

    run_eval(ecfg)
