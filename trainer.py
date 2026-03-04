"""
Training Loop
=============
Train/eval epochs and the full training pipeline.
Model-agnostic: the model class is passed in via model_cls.
"""

import json
import time
import importlib
import urllib.request
import uuid
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from vol_forecasting import dataset as dataset_v1
from vol_forecasting import dataset_new as dataset_v2


def _load_dataset_module(version: str):
    """Return the correct dataset module based on version string."""
    if version == "v2":
        return dataset_v2
    return dataset_v1


def _resolve_cls(target: str):
    """Resolve a dotted path like 'vol_forecasting.models.nhits.NHiTS' to a class."""
    module_path, cls_name = target.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _parse_horizon(h: str) -> tuple[str, int]:
    """Import-free horizon parsing (duplicated from model to keep trainer independent)."""
    horizon_map = {
        '1s': 1, '5s': 5, '10s': 10, '30s': 30,
        '1m': 60, '3m': 180, '5m': 300,
        '10m': 600, '15m': 900, '30m': 1800, '1h': 3600,
    }
    if h in horizon_map:
        return h, horizon_map[h]
    try:
        secs = int(h)
        for label, val in horizon_map.items():
            if val == secs:
                return label, secs
        return f'{secs}s', secs
    except ValueError:
        raise ValueError(f"Cannot parse horizon '{h}'. Use e.g. '5m', '300', '1h'")


def _ntfy(channel: str | None, msg: str, *, title: str | None = None,
          tags: str | None = None, priority: str = "default"):
    """Send a push notification via ntfy.sh. Silently no-ops on failure."""
    if not channel:
        return
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{channel}",
            data=msg.encode(),
            method="POST",
        )
        if title:
            req.add_header("Title", title)
        if tags:
            req.add_header("Tags", tags)
        req.add_header("Priority", priority)
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # never let notifications break training


# ---------------------------------------------------------------------------
# Pricer evaluation for NN models
# ---------------------------------------------------------------------------

@torch.no_grad()
def _pricer_eval(
    model,
    feat_gpu: torch.Tensor,     # (N_val, F) float32 on GPU (val-local)
    pfx_gpu: torch.Tensor,      # (P, N_val+?) float32 on GPU (val-local)
    df: pl.DataFrame,           # full DataFrame
    n_train: int, n_val: int,
    lookback: int,
    n_products: int,
    tgt_mean: float, tgt_std: float,
    prod: str, prod_idx: int,
    pricer,                     # BSFourierPricer instance (or None)
    jsu_pricer=None,            # JSUPricer baseline (optional, model-free)
    pricing_head=None,          # PricingHead instance (or None)
    device: torch.device = None,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    period_len_min: int = 15,
    batch_size: int = 4096,
    log_target: bool = True,
    augmenter_cls=None,
) -> dict:
    """Period-based pricer evaluation on the val set (GPU-accelerated).

    For each period_len_min-minute period in the val set:
      - strike = spot at period start
      - realized = 1 if end-of-period spot > strike, else 0
      - at each minute boundary, predict forward RV for remaining seconds
      - feed to BSFourierPricer → P(S_T > K)
      - compute Brier score vs realized

    Model and tensors should already be on GPU. Augmenter is created internally.
    Returns dict of metrics or {} on failure.
    """
    if augmenter_cls is None:
        augmenter_cls = dataset_v1.BatchAugmenter
    if device is None:
        device = feat_gpu.device
    model.eval()
    augmenter = augmenter_cls(lookback, n_products).to(device)

    df_val = df[n_train:n_train + n_val]
    mid_col = f"{prod}_mid"
    seas_col = f"{prod}_seas_factor"

    if mid_col not in df_val.columns:
        return {}

    spot_all = df_val.get_column(mid_col).to_numpy().astype(np.float64)
    seas_all = (df_val.get_column(seas_col).to_numpy().astype(np.float64)
                if seas_col in df_val.columns
                else np.ones(len(df_val), dtype=np.float64))

    ts = df_val.get_column("received_time")
    second = ts.dt.second().to_numpy()
    minute = ts.dt.minute().to_numpy().astype(np.int32)  # polars returns Int8 → overflow in (15 - mip)*60
    mip = minute % period_len_min  # minute-in-period
    period_start = ts.dt.truncate(f"{period_len_min}m")

    # ── Identify complete periods ──
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
        return {}

    # ── Select evaluation rows: minute boundaries, enough lookback ──
    ps_val = period_start.to_list()
    in_complete = np.array([p in complete_set for p in ps_val])
    is_min_bdy = second == 0
    has_lb = np.arange(len(df_val)) >= lookback
    max_h_sec = period_len_min * 60
    has_fwd = np.arange(len(df_val)) + max_h_sec < n_val

    valid = is_min_bdy & has_lb & has_fwd & in_complete
    valid_idx = np.where(valid)[0]  # val-local indices

    if len(valid_idx) == 0:
        return {}

    # Per valid row: horizon, strike, realized
    mip_valid = mip[valid_idx]
    horizons = ((period_len_min - mip_valid) * 60).astype(np.int64)

    # Clamp horizons to model's embedding range
    max_h = model.horizon_embed.num_embeddings - 1
    oob = (horizons < 1) | (horizons > max_h)
    if oob.any():
        print(f"    Pricer eval: clamping {oob.sum()} horizons out of [1, {max_h}] "
              f"(min={horizons.min()}, max={horizons.max()})")
    horizons = np.clip(horizons, 1, max_h)

    strikes = np.array([strike_map[ps_val[i]] for i in valid_idx])
    end_prices = np.array([end_map[ps_val[i]] for i in valid_idx])
    realized = (end_prices > strikes).astype(np.float64)
    spot_valid = spot_all[valid_idx]
    seas_valid = seas_all[valid_idx]

    # ── Common stats ──
    brier_naive = float(np.mean((realized.mean() - realized) ** 2))
    result = {
        'brier_naive': brier_naive,
        'base_rate': float(realized.mean()),
        'n_predictions': int(len(valid_idx)),
        'n_periods': len(complete_set),
        'realized': realized,
    }

    # ── Model inference (shared for BSFourier + learned head) ──
    need_inference = (pricer is not None) or (pricing_head is not None) or (jsu_pricer is not None)
    if need_inference:
        L = lookback
        arange_L = torch.arange(L, device=device)

        rv_pred = np.full(len(valid_idx), np.nan)
        # Collect learned head logits if pricing_head is available
        learned_logits = np.full(len(valid_idx), np.nan) if pricing_head is not None else None
        if pricing_head is not None:
            pricing_head.eval()

        spot_valid_gpu = torch.as_tensor(spot_valid, dtype=torch.float32, device=device)
        strikes_gpu = torch.as_tensor(strikes, dtype=torch.float32, device=device)

        unique_h = np.unique(horizons)
        pbar = tqdm(unique_h, desc="  [pricer]", leave=False)
        for h in pbar:
            h_mask = horizons == h
            h_idx = valid_idx[h_mask]  # val-local indices for this horizon
            pbar.set_postfix(h=int(h), n=len(h_idx))

            for s in range(0, len(h_idx), batch_size):
                e = min(s + batch_size, len(h_idx))
                batch_vi = torch.as_tensor(h_idx[s:e], dtype=torch.long, device=device)
                B = batch_vi.shape[0]

                starts = batch_vi - L
                offsets = starts.unsqueeze(1) + arange_L.unsqueeze(0)  # (B, L)

                x = feat_gpu[offsets]                                       # (B, L, F)
                pw = pfx_gpu[:, (offsets + 1).long()].permute(1, 0, 2)     # (B, P, L)
                ps = pfx_gpu[:, starts.long()].permute(1, 0).unsqueeze(2)

                hb = torch.full((B,), int(h), dtype=torch.long, device=device)

                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    xb = augmenter(x, pw, ps)
                    if pricing_head is not None:
                        pred, embedding = model(xb, horizon=hb, return_embedding=True)
                    else:
                        pred = model(xb, horizon=hb)

                pred_cpu = pred.float().cpu().numpy()

                # Denormalize → deseasonalized forward RV (level)
                pred_dn = pred_cpu * tgt_std + tgt_mean
                rv_deseas = np.exp(pred_dn) if log_target else pred_dn

                batch_pos = np.where(h_mask)[0][s:e]
                rv_pred[batch_pos] = rv_deseas

                # Learned head logits
                if pricing_head is not None:
                    pred_dn_gpu = pred.float() * tgt_std + tgt_mean
                    if log_target:
                        fvar_gpu = torch.exp(pred_dn_gpu.clamp(max=20))
                    else:
                        fvar_gpu = pred_dn_gpu
                    fvar_gpu = fvar_gpu.clamp(min=1e-8)
                    batch_pos_global = np.where(h_mask)[0][s:e]
                    spot_batch = spot_valid_gpu[batch_pos_global]
                    strike_batch = strikes_gpu[batch_pos_global]
                    logits = pricing_head(embedding.float(), spot_batch, strike_batch, fvar_gpu, hb.float())
                    learned_logits[batch_pos] = logits.cpu().numpy()

    # ── Reseasonalize RV predictions (shared by BSFourier + JSU) ──
    forecasted_var = None
    if need_inference:
        forecasted_var = rv_pred * seas_valid

    # ── BSFourier eval (needs model + pricer) ──
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
        brier_skill = 1 - brier_cal / brier_naive if brier_naive > 0 else float('nan')

        per_h_brier = {}
        for h in sorted(np.unique(horizons)):
            m = horizons == h
            if m.sum() >= 10:
                per_h_brier[int(h)] = float(np.mean((p_cal[m] - realized[m]) ** 2))

        result.update({
            'brier_cal': brier_cal,
            'brier_bs': brier_bs,
            'brier_skill': brier_skill,
            'per_horizon_brier': per_h_brier,
            'p_cal': p_cal,
            'p_bs': p_bs,
        })

    # ── JSU eval (uses model RV when available) ──
    if jsu_pricer is not None:
        tte_ms = horizons.astype(np.float64) * 1000.0
        p_jsu = jsu_pricer.probability(
            spot=spot_valid, strike=strikes, tte_ms=tte_ms,
            forecasted_var=forecasted_var,
        ).ravel()

        brier_jsu = float(np.mean((p_jsu - realized) ** 2))
        jsu_skill = 1 - brier_jsu / brier_naive if brier_naive > 0 else float('nan')

        per_h_brier_jsu = {}
        for h in sorted(np.unique(horizons)):
            m = horizons == h
            if m.sum() >= 10:
                per_h_brier_jsu[int(h)] = float(np.mean((p_jsu[m] - realized[m]) ** 2))

        result.update({
            'brier_jsu': brier_jsu,
            'jsu_skill': jsu_skill,
            'per_horizon_brier_jsu': per_h_brier_jsu,
            'p_jsu': p_jsu,
        })

    # ── Learned pricing head eval ──
    if pricing_head is not None and learned_logits is not None:
        from scipy.special import expit
        p_learned = expit(learned_logits)

        brier_learned = float(np.mean((p_learned - realized) ** 2))
        learned_skill = 1 - brier_learned / brier_naive if brier_naive > 0 else float('nan')

        per_h_brier_learned = {}
        for h in sorted(np.unique(horizons)):
            m = horizons == h
            if m.sum() >= 10:
                per_h_brier_learned[int(h)] = float(np.mean((p_learned[m] - realized[m]) ** 2))

        result.update({
            'brier_learned': brier_learned,
            'learned_skill': learned_skill,
            'per_horizon_brier_learned': per_h_brier_learned,
            'p_learned': p_learned,
        })

    return result


def _calibration_plot(p_cal, p_bs, realized, brier_cal, brier_bs, epoch, save_path,
                      p_jsu=None, brier_jsu=None,
                      p_learned=None, brier_learned=None,
                      p_lgbm_bs=None, brier_lgbm_bs=None,
                      p_lgbm_embed=None, brier_lgbm_embed=None):
    """Save a reliability diagram (calibration plot) to *save_path*.

    Side-by-side subplots for each available pricer, each with grey density
    bars and a coloured observed-frequency line.
    """
    panels = [
        (p_cal, "Calibrated", brier_cal, "tab:blue"),
        (p_bs, "BS raw", brier_bs, "tab:blue"),
    ]
    if p_jsu is not None and brier_jsu is not None:
        panels.append((p_jsu, "JSU baseline", brier_jsu, "#e67e22"))
    if p_learned is not None and brier_learned is not None:
        panels.append((p_learned, "Learned Head", brier_learned, "tab:green"))
    if p_lgbm_bs is not None and brier_lgbm_bs is not None:
        panels.append((p_lgbm_bs, "LightGBM-BS", brier_lgbm_bs, "tab:red"))
    if p_lgbm_embed is not None and brier_lgbm_embed is not None:
        panels.append((p_lgbm_embed, "LightGBM-Embed", brier_lgbm_embed, "tab:purple"))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    for ax, (probs, name, brier, color) in zip(axes, panels):
        n_bins = 20
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_idx = np.digitize(probs, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        bin_means = np.array([
            realized[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else np.nan
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
                 "o-", color=color, markersize=4)
        ax2.set_ylabel("Observed frequency")
        ax2.set_ylim(-0.05, 1.05)
        ax.set_xlabel(f"Predicted probability ({name})")
        ax.set_title(f"{name}: Calibration diagram  (Brier={brier:.4f})")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Epoch {epoch}", fontsize=13)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Train / eval epochs
# ---------------------------------------------------------------------------

def _compute_loss(pred, target, loss_mode, *, loss_alpha=1.0,
                   tgt_mean=0.0, tgt_std=1.0):
    """Compute loss based on mode: log_mse / raw_mse use MSE, raw_huber uses Huber,
    mixed_mse blends log-space and level-space MSE via loss_alpha."""
    if loss_mode == 'raw_huber':
        return F.huber_loss(pred, target, delta=1.0)
    if loss_mode == 'raw_rmse':
        return torch.sqrt(F.mse_loss(pred, target) + 1e-8)
    if loss_mode == 'mixed_mse':
        log_loss = F.mse_loss(pred, target)
        # denormalize back to log-space, then exp to level-space
        pred_lv = torch.exp(pred * tgt_std + tgt_mean)
        tgt_lv = torch.exp(target * tgt_std + tgt_mean)
        level_loss = F.mse_loss(pred_lv, tgt_lv)
        return loss_alpha * log_loss + (1.0 - loss_alpha) * level_loss
    return F.mse_loss(pred, target)


def _denorm_to_level(pred_norm, tgt_norm, tgt_mean, tgt_std, log_target):
    """Denormalize predictions/targets back to level-space (raw RV)."""
    pred_dn = pred_norm.float() * tgt_std + tgt_mean
    y_dn = tgt_norm.float() * tgt_std + tgt_mean
    if log_target:
        return torch.exp(pred_dn), torch.exp(y_dn)
    return pred_dn, y_dn


def _train_epoch(model, augmenter, batches, optimizer, scaler, use_amp, amp_dtype,
                 epochs, epoch, loss_mode='log_mse', loss_alpha=1.0,
                 pricing_head=None, dataset=None, tgt_mean=0.0, tgt_std=1.0,
                 bce_weight=0.1, period_len=900, log_target=True, clip_params=None):
    model.train()
    if pricing_head is not None:
        pricing_head.train()
    total_n = 0
    ss_res = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    bce_sum = 0.0
    bce_n = 0

    pbar = tqdm(batches, desc=f"  Epoch {epoch:3d}/{epochs} [train]", leave=False)
    for x, pw, ps, yb, hb, t_idx in pbar:
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            xb = augmenter(x, pw, ps)

            # ── Pricing head: need embedding + gradients through forecasted_var ──
            if pricing_head is not None and dataset is not None and dataset.mid_prices is not None:
                pred, embedding = model(xb, horizon=hb, return_embedding=True)
            else:
                pred = model(xb, horizon=hb)

            batch_loss = _compute_loss(pred, yb, loss_mode,
                                       loss_alpha=loss_alpha,
                                       tgt_mean=tgt_mean, tgt_std=tgt_std)
            loss = batch_loss
            if hasattr(model, 'aux_loss') and model.aux_loss is not None:
                loss = loss + model.aux_loss

            # ── BCE loss from pricing head ──
            if pricing_head is not None and dataset is not None and dataset.mid_prices is not None:
                # Denormalize pred → forecasted_var (WITH gradients for end-to-end)
                pred_dn = pred * tgt_std + tgt_mean
                if log_target:
                    forecasted_var = torch.exp(pred_dn.clamp(max=20))
                else:
                    forecasted_var = pred_dn
                # Variance must be positive for sqrt/d2 in pricing head
                forecasted_var = forecasted_var.clamp(min=1e-8)

                # Period-open strike: spot at t + h - period_len
                strike_idx = (t_idx + hb - period_len).clamp(min=0)
                strike = dataset.mid_prices[strike_idx.long()]
                spot_t = dataset.mid_prices[t_idx.long()]
                spot_end = dataset.mid_prices[(t_idx + hb).clamp(max=len(dataset.mid_prices) - 1).long()]

                binary_label = (spot_end > strike).float()
                logits = pricing_head(embedding.float(), spot_t, strike, forecasted_var, hb.float())
                bce_loss = F.binary_cross_entropy_with_logits(logits, binary_label)
                loss = loss + bce_weight * bce_loss

                with torch.no_grad():
                    bce_sum += bce_loss.item() * x.shape[0]
                    bce_n += x.shape[0]

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if clip_params is not None:
            nn.utils.clip_grad_norm_(clip_params, 1.0)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = x.shape[0]
        total_n += bs
        with torch.no_grad():
            ss_res += ((pred.detach() - yb) ** 2).sum().item()
            sum_y += yb.sum().item()
            sum_y2 += (yb ** 2).sum().item()
        pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

    avg_mse = ss_res / total_n
    global_mean = sum_y / total_n
    ss_tot = sum_y2 - total_n * global_mean ** 2
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    avg_bce = bce_sum / bce_n if bce_n > 0 else float('nan')
    return avg_mse, r2, avg_bce


@torch.no_grad()
def _eval_epoch(model, augmenter, batches, use_amp, amp_dtype,
                tgt_mean=0.0, tgt_std=1.0, loss_mode='log_mse', log_target=True,
                loss_alpha=1.0):
    model.eval()
    total_loss = 0.0
    total_n = 0
    # normalized-space stats
    ss_res = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    # level-space stats
    lv_ss_res = 0.0
    lv_sum_y = 0.0
    lv_sum_y2 = 0.0

    pbar = tqdm(batches, desc="  [val]", leave=False)
    for x, pw, ps, yb, hb, _t_idx in pbar:
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            xb = augmenter(x, pw, ps)
            pred = model(xb, horizon=hb)

        bs = x.shape[0]
        total_loss += _compute_loss(pred, yb, loss_mode,
                                    loss_alpha=loss_alpha,
                                    tgt_mean=tgt_mean, tgt_std=tgt_std).item() * bs
        total_n += bs
        ss_res += ((pred - yb) ** 2).sum().item()
        sum_y += yb.sum().item()
        sum_y2 += (yb ** 2).sum().item()

        # level space
        pred_lv, y_lv = _denorm_to_level(pred, yb, tgt_mean, tgt_std, log_target)
        lv_ss_res += ((pred_lv - y_lv) ** 2).sum().item()
        lv_sum_y += y_lv.sum().item()
        lv_sum_y2 += (y_lv ** 2).sum().item()
        pbar.set_postfix(loss=f"{total_loss / total_n:.4f}")

    loss = total_loss / total_n
    global_mean = sum_y / total_n
    ss_tot = sum_y2 - total_n * global_mean ** 2
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    lv_mean = lv_sum_y / total_n
    lv_ss_tot = lv_sum_y2 - total_n * lv_mean ** 2
    lv_r2 = 1 - lv_ss_res / lv_ss_tot if lv_ss_tot > 0 else float('nan')
    return loss, r2, lv_r2


def _build_model(model_cls, *, device, **model_kwargs):
    """Instantiate and move model to device."""
    return model_cls(**model_kwargs).to(device)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    df: pl.DataFrame,
    model_target: str = 'vol_forecasting.models.nhits.NHiTS',
    model_kwargs: dict | None = None,
    products: list[str] = ['s', 'p', 'e'],
    horizon: str = '15m',
    lookback: int = 3600,
    stride: int = 1,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    epochs: int = 30,
    patience: int = 5,
    device: str = 'cuda',
    use_compile: bool = False,
    variable_horizon: dict | None = None,
    ntfy_channel: str | None = None,
    pricer_path: str | None = None,
    jsu_params_path: str | None = None,
    pricer_period_min: int = 15,
    run_dir: str | None = None,
    loss_mode: str = 'log_mse',
    loss_alpha: float = 1.0,
    target_clip: float | None = None,
    pricing_head_enabled: bool = False,
    pricing_head_kwargs: dict | None = None,
    bce_weight: float = 0.1,
    pricing_head_period_len: int = 900,
    dataset_version: str = 'v1',
    lgbm_pricer_enabled: bool = False,
    lgbm_pricer_use_realized_var: bool = False,
) -> dict:
    ds = _load_dataset_module(dataset_version)
    GPUDataset = ds.GPUDataset
    BatchIterator = ds.BatchIterator
    BatchAugmenter = ds.BatchAugmenter
    precompute_prefix_sums = ds.precompute_prefix_sums
    build_prefix_stack = ds.build_prefix_stack
    get_feature_cols = ds.get_feature_cols

    model_cls = _resolve_cls(model_target)
    h_label, h_secs = _parse_horizon(horizon)
    if model_kwargs is None:
        model_kwargs = {}

    # ── Loss mode ──
    valid_modes = ('log_mse', 'raw_mse', 'raw_huber', 'raw_rmse', 'mixed_mse')
    if loss_mode not in valid_modes:
        raise ValueError(f"loss_mode must be one of {valid_modes}, got '{loss_mode}'")
    log_target = loss_mode.startswith('log') or loss_mode == 'mixed_mse'
    print(f"Loss mode: {loss_mode} (log_target={log_target})"
          + (f", loss_alpha={loss_alpha}" if loss_mode == 'mixed_mse' else "")
          + (f", target_clip={target_clip}" if target_clip else ""))

    vh_enabled = variable_horizon and variable_horizon.get('enabled', False)
    if vh_enabled:
        vh_min = variable_horizon.get('min_horizon', 1)
        vh_max = variable_horizon.get('max_horizon', 900)
    else:
        vh_min = h_secs
        vh_max = h_secs

    # ── Run directory ──
    run_id = uuid.uuid4().hex[:8]
    if run_dir is None:
        run_dir = f"runs/{run_id}"
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_id} → {run_path}")

    # Save run config
    run_config = {
        'run_id': run_id,
        'model_target': model_target,
        'model_kwargs': model_kwargs,
        'products': products,
        'horizon': horizon,
        'lookback': lookback,
        'stride': stride,
        'train_frac': train_frac,
        'val_frac': val_frac,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs': epochs,
        'patience': patience,
        'variable_horizon': variable_horizon,
        'dataset_version': dataset_version,
        'loss_mode': loss_mode,
        'loss_alpha': loss_alpha,
        'target_clip': target_clip,
        'lgbm_pricer': {
            'enabled': lgbm_pricer_enabled,
            'use_realized_var': lgbm_pricer_use_realized_var,
        },
    }
    (run_path / "config.json").write_text(json.dumps(run_config, indent=2))

    # ── Load pricers (optional) ──
    pricer = None
    if pricer_path is not None:
        try:
            from vol_forecasting.pricing.bs_fourier_pricer import BSFourierPricer
            pricer = BSFourierPricer.from_json(pricer_path)
            print(f"BSFourier pricer: {pricer} (period={pricer_period_min}m)")
        except Exception as e:
            print(f"WARNING: Could not load BSFourier pricer from {pricer_path}: {e}")

    jsu_pricer = None
    if jsu_params_path is not None:
        try:
            from vol_forecasting.pricing.jsu_pricer import JSUPricer
            jsu_pricer = JSUPricer.from_csv(jsu_params_path)
            print(f"JSU pricer: {jsu_pricer}")
        except Exception as e:
            print(f"WARNING: Could not load JSU pricer from {jsu_params_path}: {e}")

    # ── Pricing head validation ──
    if pricing_head_enabled and 'nhits' in model_target.lower():
        raise ValueError("Pricing head is not supported with NHiTS (no embedding bottleneck)")
    if pricing_head_kwargs is None:
        pricing_head_kwargs = {}

    print(f"Model: {model_target}")
    if vh_enabled:
        print(f"Variable horizon: {vh_min}–{vh_max}s")
    else:
        print(f"Forecast horizon: {h_label} ({h_secs}s)")
    if pricing_head_enabled:
        print(f"Pricing head: enabled (bce_weight={bce_weight}, period_len={pricing_head_period_len}s)")
    if lgbm_pricer_enabled:
        src = 'realized' if lgbm_pricer_use_realized_var else 'model'
        print(f"LightGBM pricer: enabled (var_source={src})")

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    dev = torch.device(device)

    # ── Feature columns (shared across all products) ──
    print(f"getting feature cols (dataset {dataset_version})")
    if dataset_version == 'v2':
        feature_cols = get_feature_cols()  # v2 uses PipelineConfig defaults
    else:
        feature_cols = get_feature_cols(products)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} expected feature columns. "
            f"First 5: {missing[:5]}"
        )
    print('exctracting and normalizing features')
    # ── Extract & normalize features (once, before product loop) ──
    features = df.select(feature_cols).to_numpy().astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    print('splitting')
    n = len(features)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    feat_mean = np.nanmean(features[:n_train], axis=0, keepdims=True)
    feat_std = np.nanstd(features[:n_train], axis=0, keepdims=True) + 1e-8
    features = (features - feat_mean) / feat_std

    # ── Prefix sums (once, shared across products) ──
    prefix_rvs = precompute_prefix_sums(df, products)
    prefix_stack = build_prefix_stack(prefix_rvs, products)

    # ── Mid-prices for pricing head (per-product, extracted once) ──
    mid_prices_all = {}
    if pricing_head_enabled:
        for prod in products:
            mid_col = f"{prod}_mid"
            if mid_col in df.columns:
                mid_prices_all[prod] = df.get_column(mid_col).to_numpy().astype(np.float32)
            else:
                print(f"WARNING: {mid_col} not found — pricing head disabled for {prod}")

    # ── Augmented feature dimensions ──
    n_window_feats = 4 + len(products)
    total_features = len(feature_cols) + n_window_feats
    print(f"\nFeatures: {len(feature_cols)} pipeline + {n_window_feats} augmented = {total_features}")
    print(f"Data: {n:,} rows | Train: {n_train:,} | Val: {n_val:,} | Test: {n - n_train - n_val:,}")

    results = {}

    for prod in products:
        prod_idx = products.index(prod)

        print(f"\n{'='*60}")
        horizon_desc = f"variable {vh_min}–{vh_max}s" if vh_enabled else h_label
        print(f"Training {model_cls.__name__} for {prod} @ {horizon_desc}")
        print(f"{'='*60}")

        full_model_kwargs = dict(lookback=lookback, n_features=total_features, **model_kwargs)
        full_model_kwargs['max_horizon'] = vh_max

        # ── GPU-resident datasets ──
        t0_gpu = time.time()

        prefix_train_end = min(n_train + vh_max + 2, prefix_stack.shape[1])
        prefix_val_end = min(n_train + n_val + vh_max + 2, prefix_stack.shape[1])

        # Mid-prices for this product (or None)
        prod_mid = mid_prices_all.get(prod)

        train_gds = GPUDataset(
            features[:n_train],
            prefix_stack[:, :prefix_train_end],
            lookback, stride, dev,
            variable_horizon=vh_enabled, min_horizon=vh_min, max_horizon=vh_max,
            prod_idx=prod_idx, log_target=log_target,
            mid_prices=prod_mid[:n_train] if prod_mid is not None else None,
        )
        val_gds = GPUDataset(
            features[n_train:n_train + n_val],
            prefix_stack[:, n_train:prefix_val_end],
            lookback, stride, dev,
            variable_horizon=vh_enabled, min_horizon=vh_min, max_horizon=vh_max,
            prod_idx=prod_idx, log_target=log_target,
            mid_prices=prod_mid[n_train:n_train + n_val] if prod_mid is not None else None,
        )
        test_gds = GPUDataset(
            features[n_train + n_val:],
            prefix_stack[:, n_train + n_val:],
            lookback, 1, dev,
            variable_horizon=vh_enabled, min_horizon=vh_min, max_horizon=vh_max,
            prod_idx=prod_idx, log_target=log_target,
            mid_prices=prod_mid[n_train + n_val:] if prod_mid is not None else None,
        )

        # Use training target stats for val/test to prevent data leakage
        tgt_mean = train_gds.tgt_mean
        tgt_std = train_gds.tgt_std
        val_gds.tgt_mean = tgt_mean
        val_gds.tgt_std = tgt_std
        test_gds.tgt_mean = tgt_mean
        test_gds.tgt_std = tgt_std

        # Target clipping (in normalized space, so clip_val is in std units)
        if target_clip is not None:
            train_gds.target_clip = target_clip
            val_gds.target_clip = target_clip
            test_gds.target_clip = target_clip

        gpu_mb = sum(
            ds.features.nbytes + ds.prefix_stack.nbytes
            for ds in (train_gds, val_gds, test_gds)
        ) / 1e6
        print(f"GPU transfer: {time.time() - t0_gpu:.2f}s, ~{gpu_mb:.0f} MB")
        print(f"Train: {len(train_gds):,}  Val: {len(val_gds):,}  Test: {len(test_gds):,}")

        train_batches = BatchIterator(train_gds, batch_size, shuffle=True, drop_last=True)
        val_batches = BatchIterator(val_gds, batch_size * 2, shuffle=False, drop_last=False)
        test_batches = BatchIterator(test_gds, batch_size * 2, shuffle=False, drop_last=False)

        # ── Model + Augmenter ──
        augmenter = BatchAugmenter(lookback, len(products)).to(dev)
        model = _build_model(model_cls, device=dev, **full_model_kwargs)

        if use_compile and hasattr(torch, 'compile'):
            model = torch.compile(model, mode='max-autotune')
            augmenter = torch.compile(augmenter)
            print("torch.compile enabled (max-autotune)")

        # ── Pricing head (optional) ──
        pricing_head = None
        if pricing_head_enabled and train_gds.mid_prices is not None:
            from vol_forecasting.pricing.pricing_head import PricingHead
            raw_model = getattr(model, '_orig_mod', model)
            pricing_head = PricingHead(
                embedding_dim=raw_model.embedding_dim,
                **pricing_head_kwargs,
            ).to(dev)

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ph_param_count = sum(p.numel() for p in pricing_head.parameters()) if pricing_head else 0
        print(f"Parameters: {param_count:,}" + (f" + {ph_param_count:,} (pricing head)" if ph_param_count else ""))

        model_name = model_cls.__name__
        _ntfy(ntfy_channel,
              f"{model_name} | {prod} @ {horizon_desc}\n"
              f"Params: {param_count:,} | Epochs: {epochs} | BS: {batch_size}",
              title=f"Training started: {prod}", tags="rocket")

        # Combine model + pricing head params into one optimizer
        all_params = list(model.parameters())
        if pricing_head is not None:
            all_params += list(pricing_head.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        use_amp = (device == 'cuda')
        amp_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

        # ── Train ──
        best_val_loss = float('inf')
        best_state = None
        best_ph_state = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_gds.resample()

            train_loss, train_r2, train_bce = _train_epoch(
                model, augmenter, train_batches, optimizer, scaler,
                use_amp, amp_dtype, epochs, epoch, loss_mode=loss_mode,
                loss_alpha=loss_alpha,
                pricing_head=pricing_head, dataset=train_gds,
                tgt_mean=tgt_mean, tgt_std=tgt_std,
                bce_weight=bce_weight, period_len=pricing_head_period_len,
                log_target=log_target,
                clip_params=all_params if pricing_head is not None else None,
            )

            val_loss, val_r2, val_lv_r2 = _eval_epoch(
                model, augmenter, val_batches, use_amp, amp_dtype,
                tgt_mean, tgt_std, loss_mode=loss_mode, log_target=log_target,
                loss_alpha=loss_alpha,
            )

            scheduler.step()
            elapsed = time.time() - t0

            # ── Checkpoint (before pricer eval to avoid CUDA context poisoning) ──
            improved = val_loss < best_val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_ph_state = ({k: v.cpu().clone() for k, v in pricing_head.state_dict().items()}
                                 if pricing_head is not None else None)
                no_improve = 0
            else:
                no_improve += 1

            # ── Pricer eval on val set (GPU-accelerated) ──
            pricer_metrics = {}
            if pricer is not None or jsu_pricer is not None or pricing_head is not None:
                try:
                    raw_model = getattr(model, '_orig_mod', model)
                    pricer_metrics = _pricer_eval(
                        raw_model,
                        val_gds.features, val_gds.prefix_stack,
                        df, n_train, n_val, lookback, len(products),
                        tgt_mean, tgt_std,
                        prod, prod_idx, pricer,
                        jsu_pricer=jsu_pricer,
                        pricing_head=pricing_head,
                        device=dev, use_amp=use_amp, amp_dtype=amp_dtype,
                        period_len_min=pricer_period_min, batch_size=batch_size,
                        log_target=log_target,
                        augmenter_cls=BatchAugmenter,
                    )
                    if pricer_metrics and 'p_cal' in pricer_metrics:
                        _calibration_plot(
                            pricer_metrics['p_cal'],
                            pricer_metrics['p_bs'],
                            pricer_metrics['realized'],
                            pricer_metrics['brier_cal'],
                            pricer_metrics['brier_bs'],
                            epoch,
                            run_path / prod / f"calibration_epoch_{epoch:03d}.png",
                            p_jsu=pricer_metrics.get('p_jsu'),
                            brier_jsu=pricer_metrics.get('brier_jsu'),
                            p_learned=pricer_metrics.get('p_learned'),
                            brier_learned=pricer_metrics.get('brier_learned'),
                        )
                except Exception as e:
                    import traceback
                    print(f"    Pricer eval failed: {e}")
                    traceback.print_exc()

            # ── LightGBM pricers (per-epoch, retrained on current embeddings) ──
            lgbm_results = {}
            if lgbm_pricer_enabled and pricer is not None:
                try:
                    from vol_forecasting.pricing.lgbm_pricer import train_lgbm_pricers
                    raw_model = getattr(model, '_orig_mod', model)
                    lgbm_results = train_lgbm_pricers(
                        raw_model,
                        train_gds.features, train_gds.prefix_stack,
                        val_gds.features, val_gds.prefix_stack,
                        df, n_train, n_val,
                        lookback, len(products),
                        tgt_mean, tgt_std,
                        prod, prod_idx, pricer,
                        device=dev,
                        use_amp=use_amp, amp_dtype=amp_dtype,
                        period_len_min=pricer_period_min,
                        batch_size=batch_size,
                        log_target=log_target,
                        use_realized_var=lgbm_pricer_use_realized_var,
                        augmenter_cls=BatchAugmenter,
                    )
                    if lgbm_results and pricer_metrics and 'p_cal' in pricer_metrics:
                        _calibration_plot(
                            lgbm_results['p_cal'],
                            lgbm_results['p_bs'],
                            lgbm_results['realized'],
                            lgbm_results['brier_cal'],
                            lgbm_results['brier_bs'],
                            epoch,
                            run_path / prod / f"calibration_lgbm_epoch_{epoch:03d}.png",
                            p_lgbm_bs=lgbm_results.get('p_lgbm_bs'),
                            brier_lgbm_bs=lgbm_results.get('brier_lgbm_bs'),
                            p_lgbm_embed=lgbm_results.get('p_lgbm_embed'),
                            brier_lgbm_embed=lgbm_results.get('brier_lgbm_embed'),
                        )
                except Exception as e:
                    import traceback
                    print(f"    LightGBM pricers failed: {e}")
                    traceback.print_exc()

            # ── Print epoch summary ──
            brier_str = ""
            if pricer_metrics and 'brier_cal' in pricer_metrics:
                brier_str = (f" | Brier: {pricer_metrics['brier_cal']:.4f} "
                             f"(skill {pricer_metrics['brier_skill']:+.3f})")
            if pricer_metrics and 'brier_jsu' in pricer_metrics:
                brier_str += (f" JSU: {pricer_metrics['brier_jsu']:.4f} "
                              f"(skill {pricer_metrics['jsu_skill']:+.3f})")
            if pricer_metrics and 'brier_learned' in pricer_metrics:
                brier_str += (f" | Learned: {pricer_metrics['brier_learned']:.4f} "
                              f"(skill {pricer_metrics['learned_skill']:+.3f})")
            if lgbm_results and 'brier_lgbm_bs' in lgbm_results:
                brier_str += (f" | LGBM-BS: {lgbm_results['brier_lgbm_bs']:.4f} "
                              f"(skill {lgbm_results['lgbm_bs_skill']:+.3f})")
            if lgbm_results and 'brier_lgbm_embed' in lgbm_results:
                brier_str += (f" | LGBM-E: {lgbm_results['brier_lgbm_embed']:.4f} "
                              f"(skill {lgbm_results['lgbm_embed_skill']:+.3f})")

            bce_str = f" BCE: {train_bce:.4f}" if not np.isnan(train_bce) else ""
            loss_label = 'Huber' if loss_mode == 'raw_huber' else 'RMSE' if loss_mode == 'raw_rmse' else 'MSE'
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train {loss_label}: {train_loss:.6f} R²: {train_r2:.4f}{bce_str} | "
                  f"Val {loss_label}: {val_loss:.6f} R²: {val_r2:.4f} lvR²: {val_lv_r2:.4f}{brier_str} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

            # ── MoE routing summary (every 5 epochs + first/last) ──
            raw_model = getattr(model, '_orig_mod', model)
            if hasattr(raw_model, 'routing_summary') and (epoch == 1 or epoch == epochs or epoch % 5 == 0):
                print(f"  Router (epoch {epoch}):")
                print(raw_model.routing_summary())

            # ── Save epoch metrics to run directory ──
            epoch_metrics = {
                'epoch': epoch,
                'product': prod,
                'train_mse': train_loss,
                'train_r2': train_r2,
                'train_bce': train_bce if not np.isnan(train_bce) else None,
                'val_mse': val_loss,
                'val_r2': val_r2,
                'val_level_r2': val_lv_r2,
                'lr': scheduler.get_last_lr()[0],
                'elapsed_s': elapsed,
            }
            if pricer_metrics:
                epoch_metrics['pricer'] = {
                    k: v for k, v in pricer_metrics.items()
                    if k not in ('p_cal', 'p_bs', 'p_jsu', 'p_learned', 'realized')
                }
            if lgbm_results:
                epoch_metrics['lgbm'] = {
                    k: v for k, v in lgbm_results.items()
                    if k not in ('lgbm_bs_model', 'lgbm_embed_model',
                                 'p_lgbm_bs', 'p_lgbm_embed',
                                 'realized', 'p_bs', 'p_cal', 'horizons')
                }

            epoch_dir = run_path / prod
            epoch_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = epoch_dir / f"epoch_{epoch:03d}.json"
            metrics_path.write_text(json.dumps(epoch_metrics, indent=2))

            if improved:
                # Save best checkpoint to run directory
                ckpt = {
                    'epoch': epoch,
                    'state_dict': best_state,
                    'val_loss': best_val_loss,
                    'feat_mean': feat_mean,
                    'feat_std': feat_std,
                    'tgt_mean': tgt_mean,
                    'tgt_std': tgt_std,
                    'feature_cols': feature_cols,
                    'model_kwargs': full_model_kwargs,
                    'loss_mode': loss_mode,
                }
                if best_ph_state is not None:
                    ckpt['pricing_head_state'] = best_ph_state
                    ckpt['pricing_head_kwargs'] = pricing_head_kwargs
                if pricer_metrics:
                    ckpt['pricer_metrics'] = pricer_metrics
                if lgbm_results:
                    lgbm_ckpt = {}
                    for mname in ['lgbm_bs', 'lgbm_embed']:
                        key = f'{mname}_model'
                        if key in lgbm_results:
                            lgbm_ckpt[mname] = lgbm_results[key].model_to_string()
                    ckpt['lgbm_models'] = lgbm_ckpt
                    ckpt['lgbm_metrics'] = {
                        k: v for k, v in lgbm_results.items()
                        if k not in ('lgbm_bs_model', 'lgbm_embed_model',
                                     'p_lgbm_bs', 'p_lgbm_embed',
                                     'realized', 'p_bs', 'p_cal', 'horizons')
                    }
                torch.save(ckpt, epoch_dir / "best.pt")
                print(f"    Saved best → {epoch_dir / 'best.pt'}")
            else:
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    _ntfy(ntfy_channel,
                          f"Epoch {epoch}/{epochs} | {prod} @ {horizon_desc}\n"
                          f"Val MSE: {val_loss:.6f} R²: {val_r2:.4f}\n"
                          f"Early stopping (patience={patience})",
                          title=f"Early stop: {prod} ep {epoch}",
                          tags="stop_sign")
                    break

            _ntfy(ntfy_channel,
                  f"Epoch {epoch}/{epochs} | {prod} @ {horizon_desc}\n"
                  f"Train MSE: {train_loss:.6f} R²: {train_r2:.4f}\n"
                  f"Val   MSE: {val_loss:.6f} R²: {val_r2:.4f}\n"
                  f"{'** new best **' if improved else f'no improve {no_improve}/{patience}'} | {elapsed:.1f}s",
                  title=f"{prod} ep {epoch}/{epochs}",
                  tags="chart_with_upwards_trend" if improved else "hourglass")

        # ── Test (per-horizon evaluation for deterministic R²) ──
        model_raw = _build_model(model_cls, device=dev, **full_model_kwargs)
        model_raw.load_state_dict(best_state)
        model_raw.eval()

        augmenter_raw = BatchAugmenter(lookback, len(products)).to(dev)

        eval_horizons = [1, 60, 300, 600, 900] if vh_enabled else [vh_max]
        per_horizon_r2 = {}
        per_horizon_lv_r2 = {}
        agg_ss_res = 0.0
        agg_ss_tot = 0.0
        agg_lv_ss_res = 0.0
        agg_lv_ss_tot = 0.0
        agg_n = 0

        for eval_h in eval_horizons:
            test_gds_h = GPUDataset(
                features[n_train + n_val:],
                prefix_stack[:, n_train + n_val:],
                lookback, 1, dev,
                variable_horizon=False, min_horizon=eval_h, max_horizon=eval_h,
                prod_idx=prod_idx, log_target=log_target,
            )
            test_gds_h.tgt_mean = tgt_mean
            test_gds_h.tgt_std = tgt_std
            test_batches_h = BatchIterator(test_gds_h, batch_size * 2, shuffle=False, drop_last=False)

            all_preds, all_targets = [], []
            with torch.no_grad():
                for x, pw, ps, yb, hb, _t_idx in test_batches_h:
                    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                        xb = augmenter_raw(x, pw, ps)
                        all_preds.append(model_raw(xb, horizon=hb).float().cpu())
                    all_targets.append(yb.cpu())

            preds = torch.cat(all_preds).numpy()
            actuals = torch.cat(all_targets).numpy()

            # denormalized space
            preds_dn = preds * tgt_std + tgt_mean
            actuals_dn = actuals * tgt_std + tgt_mean

            h_ss_res = float(np.sum((actuals_dn - preds_dn) ** 2))
            h_ss_tot = float(np.sum((actuals_dn - np.mean(actuals_dn)) ** 2))
            h_r2 = 1 - h_ss_res / h_ss_tot if h_ss_tot > 0 else float('nan')
            per_horizon_r2[eval_h] = h_r2

            # level space
            if log_target:
                preds_lv = np.exp(preds_dn)
                actuals_lv = np.exp(actuals_dn)
            else:
                preds_lv = preds_dn
                actuals_lv = actuals_dn
            h_lv_ss_res = float(np.sum((actuals_lv - preds_lv) ** 2))
            h_lv_ss_tot = float(np.sum((actuals_lv - np.mean(actuals_lv)) ** 2))
            h_lv_r2 = 1 - h_lv_ss_res / h_lv_ss_tot if h_lv_ss_tot > 0 else float('nan')
            per_horizon_lv_r2[eval_h] = h_lv_r2

            agg_ss_res += h_ss_res
            agg_ss_tot += h_ss_tot
            agg_lv_ss_res += h_lv_ss_res
            agg_lv_ss_tot += h_lv_ss_tot
            agg_n += len(preds)

            del test_gds_h, test_batches_h

        mse = agg_ss_res / agg_n
        r2 = 1 - agg_ss_res / agg_ss_tot if agg_ss_tot > 0 else float('nan')
        lv_r2 = 1 - agg_lv_ss_res / agg_lv_ss_tot if agg_lv_ss_tot > 0 else float('nan')

        print(f"\n  Test ({prod} @ {horizon_desc}):")
        for eh in eval_horizons:
            print(f"    h={eh:4d}s  R²={per_horizon_r2[eh]:.4f}  lvR²={per_horizon_lv_r2[eh]:.4f}")
        print(f"    Aggregate MSE={mse:.6f}  R²={r2:.4f}  lvR²={lv_r2:.4f}")
        if lgbm_results:
            lgbm_str = " | ".join(
                f"{n}: {lgbm_results[f'brier_{n}']:.4f}"
                for n in ['lgbm_bs', 'lgbm_embed']
                if f'brier_{n}' in lgbm_results
            )
            if lgbm_str:
                print(f"    LightGBM Brier: {lgbm_str}")

        _ntfy(ntfy_channel,
              f"{model_name} | {prod} @ {horizon_desc}\n"
              f"MSE: {mse:.6f} | R²: {r2:.4f} | lvR²: {lv_r2:.4f}\n"
              f"Per-h R²: {per_horizon_r2}\n"
              f"Per-h lvR²: {per_horizon_lv_r2}\n"
              f"Best val MSE: {best_val_loss:.6f}",
              title=f"Test done: {prod} R²={r2:.4f} lvR²={lv_r2:.4f}",
              tags="white_check_mark", priority="high")

        # Free GPU dataset memory before next product
        del train_gds, val_gds, train_batches, val_batches, test_batches
        torch.cuda.empty_cache()

        prod_results = {
            'model': model_raw,
            'state_dict': best_state,
            'feat_mean': feat_mean,
            'feat_std': feat_std,
            'tgt_mean': tgt_mean,
            'tgt_std': tgt_std,
            'feature_cols': feature_cols,
            'horizon': h_label,
            'horizon_secs': h_secs,
            'best_val_loss': best_val_loss,
            'test_r2': r2,
            'test_level_r2': lv_r2,
            'test_mse': mse,
            'per_horizon_r2': per_horizon_r2,
            'per_horizon_level_r2': per_horizon_lv_r2,
        }
        if lgbm_results:
            for name in ['lgbm_bs', 'lgbm_embed']:
                bk = f'brier_{name}'
                sk = f'{name}_skill'
                if bk in lgbm_results:
                    prod_results[bk] = lgbm_results[bk]
                    prod_results[sk] = lgbm_results[sk]
        results[prod] = prod_results
        if vh_enabled:
            results[prod]['variable_horizon'] = {
                'min_horizon': vh_min,
                'max_horizon': vh_max,
            }

    print(f"\nRun directory: {run_path}")
    return results
