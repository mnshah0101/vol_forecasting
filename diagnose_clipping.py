"""Diagnose target clipping for MoE GRU config."""
import numpy as np
import polars as pl
import torch

from vol_forecasting.dataset import (
    precompute_prefix_sums, build_prefix_stack, get_feature_cols,
)

# ── Config (matching moe_gru_config.yaml) ──
TARGET_CLIP = 5.0
PRODUCTS = ["s", "p", "e"]
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
LOOKBACK = 3600
STRIDE = 30
MIN_H = 1
MAX_H = 900
PROD_IDX = 0  # 's' = spot
LOG_TARGET = False  # raw_mse → log_target=False
EPS = 1e-12

print("Loading parquet...")
df = pl.read_parquet("/tmp/features.parquet")
n = len(df)
n_train = int(n * TRAIN_FRAC)
n_val = int(n * VAL_FRAC)
print(f"Total rows: {n:,} | Train: {n_train:,} | Val: {n_val:,}")

print("Computing prefix sums...")
prefix_rvs = precompute_prefix_sums(df, PRODUCTS)
prefix_stack = build_prefix_stack(prefix_rvs, PRODUCTS)

# ── Compute target stats at midpoint horizon (same as GPUDataset) ──
mid_h = (MIN_H + MAX_H) // 2
print(f"Midpoint horizon for normalization: {mid_h}s")

# Train indices (same as GPUDataset)
all_idx = np.arange(LOOKBACK, n_train)
prefix_train_end = min(n_train + MAX_H + 2, prefix_stack.shape[1])
max_valid_t = prefix_train_end - MAX_H - 2
train_idx = all_idx[all_idx <= max_valid_t][::STRIDE]

rv_mid = prefix_stack[PROD_IDX, train_idx + mid_h + 1] - prefix_stack[PROD_IDX, train_idx + 1]
if LOG_TARGET:
    tgt_mid = np.log(rv_mid + EPS)
else:
    tgt_mid = rv_mid
tgt_mean = float(np.mean(tgt_mid))
tgt_std = float(np.std(tgt_mid)) + 1e-8
print(f"Target stats (from train, h={mid_h}): mean={tgt_mean:.6f}, std={tgt_std:.6f}")

# ── Analyze clipping across all horizons for train set ──
print(f"\n{'='*70}")
print("TRAIN SET - Clipping analysis across horizons")
print(f"{'='*70}")

horizons_to_check = [1, 5, 30, 60, 150, 300, 450, 600, 900]
total_clipped_above = 0
total_clipped_below = 0
total_samples = 0

for h in horizons_to_check:
    valid_mask = train_idx + h + 1 < prefix_stack.shape[1]
    t = train_idx[valid_mask]
    rv = prefix_stack[PROD_IDX, t + h + 1] - prefix_stack[PROD_IDX, t + 1]
    if LOG_TARGET:
        y = np.log(rv + EPS)
    else:
        y = rv
    y_norm = (y - tgt_mean) / tgt_std

    above = np.sum(y_norm > TARGET_CLIP)
    below = np.sum(y_norm < -TARGET_CLIP)
    n_h = len(y_norm)
    pct_above = 100.0 * above / n_h if n_h else 0
    pct_below = 100.0 * below / n_h if n_h else 0
    pct_total = pct_above + pct_below

    total_clipped_above += above
    total_clipped_below += below
    total_samples += n_h

    print(f"  h={h:4d}s | n={n_h:>8,} | "
          f"clip_above={pct_above:6.2f}% | clip_below={pct_below:6.2f}% | "
          f"total_clipped={pct_total:6.2f}% | "
          f"y_norm: min={y_norm.min():.2f} max={y_norm.max():.2f} "
          f"mean={y_norm.mean():.2f} std={y_norm.std():.2f}")

pct_above_all = 100.0 * total_clipped_above / total_samples
pct_below_all = 100.0 * total_clipped_below / total_samples
print(f"\n  OVERALL (uniform horizon mix): "
      f"clip_above={pct_above_all:.2f}% | clip_below={pct_below_all:.2f}% | "
      f"total={pct_above_all + pct_below_all:.2f}%")

# ── Same analysis for val set ──
print(f"\n{'='*70}")
print("VAL SET - Clipping analysis across horizons")
print(f"{'='*70}")

val_start = n_train
val_end = n_train + n_val
prefix_val_end = min(val_end + MAX_H + 2, prefix_stack.shape[1])

all_val_idx = np.arange(LOOKBACK, n_val)  # val-local
max_valid_val = prefix_val_end - n_train - MAX_H - 2
val_idx = all_val_idx[all_val_idx <= max_valid_val][::STRIDE]

total_clipped_above_v = 0
total_clipped_below_v = 0
total_samples_v = 0

for h in horizons_to_check:
    # val-local indices, prefix is also val-local slice
    pfx_val = prefix_stack[:, n_train:prefix_val_end]
    valid_mask = val_idx + h + 1 < pfx_val.shape[1]
    t = val_idx[valid_mask]
    rv = pfx_val[PROD_IDX, t + h + 1] - pfx_val[PROD_IDX, t + 1]
    if LOG_TARGET:
        y = np.log(rv + EPS)
    else:
        y = rv
    # Use TRAIN stats for normalization (matching trainer.py)
    y_norm = (y - tgt_mean) / tgt_std

    above = np.sum(y_norm > TARGET_CLIP)
    below = np.sum(y_norm < -TARGET_CLIP)
    n_h = len(y_norm)
    pct_above = 100.0 * above / n_h if n_h else 0
    pct_below = 100.0 * below / n_h if n_h else 0

    total_clipped_above_v += above
    total_clipped_below_v += below
    total_samples_v += n_h

    print(f"  h={h:4d}s | n={n_h:>8,} | "
          f"clip_above={pct_above:6.2f}% | clip_below={pct_below:6.2f}% | "
          f"total_clipped={pct_above + pct_below:6.2f}% | "
          f"y_norm: min={y_norm.min():.2f} max={y_norm.max():.2f} "
          f"mean={y_norm.mean():.2f} std={y_norm.std():.2f}")

pct_above_v = 100.0 * total_clipped_above_v / total_samples_v
pct_below_v = 100.0 * total_clipped_below_v / total_samples_v
print(f"\n  OVERALL (uniform horizon mix): "
      f"clip_above={pct_above_v:.2f}% | clip_below={pct_below_v:.2f}% | "
      f"total={pct_above_v + pct_below_v:.2f}%")

# ── Distribution stats at key percentiles ──
print(f"\n{'='*70}")
print("DISTRIBUTION of normalized targets (train, all horizons combined)")
print(f"{'='*70}")

# Sample uniformly across horizons like training does
rng = np.random.default_rng(42)
n_sample = min(500_000, len(train_idx))
sample_idx = rng.choice(train_idx, size=n_sample, replace=False)
sample_h = rng.integers(MIN_H, MAX_H + 1, size=n_sample)

rv_sample = prefix_stack[PROD_IDX, sample_idx + sample_h + 1] - prefix_stack[PROD_IDX, sample_idx + 1]
if LOG_TARGET:
    y_sample = np.log(rv_sample + EPS)
else:
    y_sample = rv_sample
y_norm_sample = (y_sample - tgt_mean) / tgt_std

percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
pct_vals = np.percentile(y_norm_sample, percentiles)
print("Percentiles of normalized targets:")
for p, v in zip(percentiles, pct_vals):
    flag = " *** CLIPPED" if abs(v) > TARGET_CLIP else ""
    print(f"  {p:6.1f}%ile → {v:+8.3f}{flag}")

clipped_total = np.sum(np.abs(y_norm_sample) > TARGET_CLIP)
print(f"\nSampled {n_sample:,} targets with uniform random horizons [{MIN_H}, {MAX_H}]:")
print(f"  Clipped: {clipped_total:,} / {n_sample:,} = {100*clipped_total/n_sample:.3f}%")
print(f"  Above +{TARGET_CLIP}: {np.sum(y_norm_sample > TARGET_CLIP):,} ({100*np.sum(y_norm_sample > TARGET_CLIP)/n_sample:.3f}%)")
print(f"  Below -{TARGET_CLIP}: {np.sum(y_norm_sample < -TARGET_CLIP):,} ({100*np.sum(y_norm_sample < -TARGET_CLIP)/n_sample:.3f}%)")

# ── Impact on R² ──
print(f"\n{'='*70}")
print("IMPACT on R² (what does clipping do to reported metrics?)")
print(f"{'='*70}")

y_clipped = np.clip(y_norm_sample, -TARGET_CLIP, TARGET_CLIP)

ss_tot_raw = np.sum((y_norm_sample - y_norm_sample.mean())**2)
ss_tot_clipped = np.sum((y_clipped - y_clipped.mean())**2)
print(f"  SS_tot (unclipped): {ss_tot_raw:.2f}")
print(f"  SS_tot (clipped):   {ss_tot_clipped:.2f}")
print(f"  Variance reduction: {100*(1 - ss_tot_clipped/ss_tot_raw):.2f}%")

# Show what a mean predictor gives for R²=0 baseline
mean_pred = np.full_like(y_clipped, y_clipped.mean())
ss_res_mean = np.sum((y_clipped - mean_pred)**2)
print(f"  R² with mean predictor (should be ~0): {1 - ss_res_mean/ss_tot_clipped:.6f}")
