"""
LightGBM Pricing Models
=======================
Train LightGBM models to predict P(S_T > K) using:
  1. LightGBM-BS:    5 market features (same inputs as Black-Scholes)
  2. LightGBM-Embed: backbone embeddings + market features + p_bs

Both models are trained *once* after neural training completes (using best
checkpoint), not per-epoch.
"""

import numpy as np
import polars as pl
import torch
from tqdm import tqdm


def _collect_pricing_data(
    model,
    feat_gpu: torch.Tensor,
    pfx_gpu: torch.Tensor,
    df_slice: pl.DataFrame,
    lookback: int,
    n_products: int,
    tgt_mean: float,
    tgt_std: float,
    prod: str,
    prod_idx: int,
    pricer,
    device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    period_len_min: int = 15,
    batch_size: int = 4096,
    log_target: bool = True,
    use_realized_var: bool = False,
    augmenter_cls=None,
) -> dict:
    """Collect period-based pricing features from a data split.

    Mirrors ``_pricer_eval``'s period identification and model inference but
    returns raw feature arrays instead of computing Brier scores.

    Returns
    -------
    dict with keys: embeddings, market_features, p_bs, p_cal, labels, horizons
    (or empty dict on failure).
    """
    from vol_forecasting import dataset as dataset_v1

    if augmenter_cls is None:
        augmenter_cls = dataset_v1.BatchAugmenter

    model.eval()
    augmenter = augmenter_cls(lookback, n_products).to(device)

    mid_col = f"{prod}_mid"
    seas_col = f"{prod}_seas_factor"

    if mid_col not in df_slice.columns:
        return {}

    n_split = len(df_slice)
    spot_all = df_slice.get_column(mid_col).to_numpy().astype(np.float64)
    seas_all = (
        df_slice.get_column(seas_col).to_numpy().astype(np.float64)
        if seas_col in df_slice.columns
        else np.ones(n_split, dtype=np.float64)
    )

    ts = df_slice.get_column("received_time")
    second = ts.dt.second().to_numpy()
    minute = ts.dt.minute().to_numpy().astype(np.int32)
    mip = minute % period_len_min
    period_start = ts.dt.truncate(f"{period_len_min}m")

    # ── Identify complete periods ──
    ps_series = period_start.alias("__ps")
    period_info = (
        df_slice.with_columns(ps_series)
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
    has_lb = np.arange(n_split) >= lookback
    max_h_sec = period_len_min * 60
    has_fwd = np.arange(n_split) + max_h_sec < n_split

    valid = is_min_bdy & has_lb & has_fwd & in_complete
    valid_idx = np.where(valid)[0]

    if len(valid_idx) == 0:
        return {}

    # Per valid row: horizon, strike, realized
    mip_valid = mip[valid_idx]
    horizons = ((period_len_min - mip_valid) * 60).astype(np.int64)

    # Clamp horizons to model's embedding range
    max_h = model.horizon_embed.num_embeddings - 1
    horizons = np.clip(horizons, 1, max_h)

    strikes = np.array([strike_map[ps_val[i]] for i in valid_idx])
    end_prices = np.array([end_map[ps_val[i]] for i in valid_idx])
    realized = (end_prices > strikes).astype(np.float64)
    spot_valid = spot_all[valid_idx]
    seas_valid = seas_all[valid_idx]

    # ── Model inference ──
    L = lookback
    arange_L = torch.arange(L, device=device)

    if use_realized_var:
        valid_idx_t = torch.as_tensor(valid_idx, dtype=torch.long, device=device)
        horizons_t = torch.as_tensor(horizons, dtype=torch.long, device=device)
        rv_true = (
            pfx_gpu[prod_idx, (valid_idx_t + horizons_t + 1).long()]
            - pfx_gpu[prod_idx, (valid_idx_t + 1).long()]
        )
        rv_pred = rv_true.float().cpu().numpy()
    else:
        rv_pred = np.full(len(valid_idx), np.nan)

    all_embeddings = []
    embed_positions = []

    unique_h = np.unique(horizons)
    for h in tqdm(unique_h, desc="  [lgbm collect]", leave=False):
        h_mask = horizons == h
        h_idx = valid_idx[h_mask]

        for s in range(0, len(h_idx), batch_size):
            e = min(s + batch_size, len(h_idx))
            batch_vi = torch.as_tensor(h_idx[s:e], dtype=torch.long, device=device)
            B = batch_vi.shape[0]

            starts = batch_vi - L
            offsets = starts.unsqueeze(1) + arange_L.unsqueeze(0)

            x = feat_gpu[offsets]
            pw = pfx_gpu[:, (offsets + 1).long()].permute(1, 0, 2)
            ps = pfx_gpu[:, starts.long()].permute(1, 0).unsqueeze(2)

            hb = torch.full((B,), int(h), dtype=torch.long, device=device)

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    xb = augmenter(x, pw, ps)
                    pred, embedding = model(xb, horizon=hb, return_embedding=True)

            batch_pos = np.where(h_mask)[0][s:e]
            if not use_realized_var:
                pred_cpu = pred.float().cpu().numpy()
                pred_dn = pred_cpu * tgt_std + tgt_mean
                rv_deseas = np.exp(pred_dn) if log_target else pred_dn
                rv_pred[batch_pos] = rv_deseas

            all_embeddings.append(embedding.float().cpu().numpy())
            embed_positions.append(batch_pos)

    # Reconstruct embeddings in order
    embed_dim = all_embeddings[0].shape[1]
    embeddings = np.full((len(valid_idx), embed_dim), np.nan)
    for emb, pos in zip(all_embeddings, embed_positions):
        embeddings[pos] = emb

    # ── Compute market features ──
    forecasted_var = rv_pred * seas_valid
    eps = 1e-8
    log_moneyness = np.log(spot_valid / (strikes + eps))
    sqrt_var = np.sqrt(np.clip(forecasted_var, eps, None))
    d2 = (log_moneyness - forecasted_var / 2) / (sqrt_var + eps)
    tte_norm = np.log(horizons.astype(np.float64) + 1) / np.log(901.0)

    market_features = np.column_stack([
        log_moneyness, forecasted_var, sqrt_var, d2, tte_norm,
    ])

    # ── Compute BS probabilities ──
    tte = horizons.astype(np.float64)
    p_cal, p_bs = pricer.probability(
        spot=spot_valid, strike=strikes,
        forecasted_var=forecasted_var, tte_seconds=tte,
        return_bs_raw=True,
    )
    p_bs = p_bs.ravel()
    p_cal = p_cal.ravel()

    return {
        'embeddings': embeddings,           # (N, D)
        'market_features': market_features, # (N, 5)
        'p_bs': p_bs,                       # (N,)
        'p_cal': p_cal,                     # (N,)
        'labels': realized,                 # (N,)
        'horizons': horizons,               # (N,)
        'variance_source': 'realized' if use_realized_var else 'model',
    }


def train_lgbm_pricers(
    model,
    train_feat_gpu, train_pfx_gpu,
    val_feat_gpu, val_pfx_gpu,
    df, n_train, n_val,
    lookback, n_products,
    tgt_mean, tgt_std,
    prod, prod_idx,
    pricer,
    device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    period_len_min: int = 15,
    batch_size: int = 4096,
    log_target: bool = True,
    use_realized_var: bool = False,
    augmenter_cls=None,
) -> dict:
    """Train LightGBM-BS and LightGBM-Embed pricers on best-checkpoint data.

    Returns dict with trained models, val Brier scores, and val predictions
    (or empty dict on failure).
    """
    import lightgbm as lgb

    var_source = 'realized' if use_realized_var else 'model'
    print(f"\n  === LightGBM Pricers ({prod}, var_source={var_source}) ===")

    common = dict(
        lookback=lookback, n_products=n_products,
        tgt_mean=tgt_mean, tgt_std=tgt_std,
        prod=prod, prod_idx=prod_idx, pricer=pricer,
        device=device, use_amp=use_amp, amp_dtype=amp_dtype,
        period_len_min=period_len_min, batch_size=batch_size,
        log_target=log_target, use_realized_var=use_realized_var,
        augmenter_cls=augmenter_cls,
    )

    # ── Collect train data ──
    print("  Collecting train data...")
    train_data = _collect_pricing_data(
        model, train_feat_gpu, train_pfx_gpu, df[:n_train], **common,
    )
    if not train_data:
        print("  WARNING: No train data collected for LightGBM")
        return {}

    # ── Collect val data ──
    print("  Collecting val data...")
    val_data = _collect_pricing_data(
        model, val_feat_gpu, val_pfx_gpu, df[n_train:n_train + n_val], **common,
    )
    if not val_data:
        print("  WARNING: No val data collected for LightGBM")
        return {}

    embed_dim = train_data['embeddings'].shape[1]
    print(f"  Train: {len(train_data['labels']):,} samples, "
          f"Val: {len(val_data['labels']):,} samples, "
          f"Embed dim: {embed_dim}")

    # ── Build feature matrices ──
    # LightGBM-BS: 5 market features
    X_train_bs = train_data['market_features']
    X_val_bs = val_data['market_features']

    # LightGBM-Embed: D embedding dims + 5 market features + p_bs
    X_train_embed = np.column_stack([
        train_data['embeddings'],
        train_data['market_features'],
        train_data['p_bs'],
    ])
    X_val_embed = np.column_stack([
        val_data['embeddings'],
        val_data['market_features'],
        val_data['p_bs'],
    ])

    y_train = train_data['labels']
    y_val = val_data['labels']

    lgb_params = {
        "objective": "mse",
        "metric": "mse",
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    }

    brier_naive = float(np.mean((y_train.mean() - y_val) ** 2))
    results = {}

    for name, X_tr, X_v in [
        ("lgbm_bs", X_train_bs, X_val_bs),
        ("lgbm_embed", X_train_embed, X_val_embed),
    ]:
        print(f"\n  Training {name} ({X_tr.shape[1]} features)...")
        dtrain = lgb.Dataset(X_tr, label=y_train)
        dval = lgb.Dataset(X_v, label=y_val, reference=dtrain)

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]

        booster = lgb.train(
            lgb_params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=callbacks,
        )

        p_val = booster.predict(X_v)
        brier = float(np.mean((p_val - y_val) ** 2))
        skill = 1 - brier / brier_naive if brier_naive > 0 else float('nan')

        print(f"  {name}: Brier={brier:.4f} (skill={skill:+.3f}), "
              f"best_iter={booster.best_iteration}")

        results[f'{name}_model'] = booster
        results[f'brier_{name}'] = brier
        results[f'{name}_skill'] = skill
        results[f'p_{name}'] = p_val

    # Include val-set BS/calibrated predictions for calibration plot
    results['realized'] = y_val
    results['horizons'] = val_data['horizons']
    results['p_bs'] = val_data['p_bs']
    results['p_cal'] = val_data['p_cal']
    results['brier_bs'] = float(np.mean((val_data['p_bs'] - y_val) ** 2))
    results['brier_cal'] = float(np.mean((val_data['p_cal'] - y_val) ** 2))
    results['variance_source'] = val_data.get('variance_source', var_source)

    return results
