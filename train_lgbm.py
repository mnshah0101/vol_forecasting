"""Standalone LightGBM pricer trainer.

Trains LightGBM pricers from a saved run checkpoint + data, without
rerunning neural training.

Usage:
    python -m vol_forecasting.train_lgbm \
        --run-dir runs/abc123 \
        --data /tmp/features.parquet \
        --pricer-path vol_forecasting/pricing/configs/fourier.json \
        --product s \
        --data-frac 0.3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from vol_forecasting.trainer import (
    _resolve_cls,
    _load_dataset_module,
    _calibration_plot,
)
from vol_forecasting.pricing.bs_fourier_pricer import BSFourierPricer
from vol_forecasting.pricing.lgbm_pricer import train_lgbm_pricers


def _normalize_state_dict_keys(state_dict: dict) -> tuple[dict, bool]:
    """Strip common wrapper prefixes from checkpoint keys."""
    prefixes = ("_orig_mod.", "module.")
    changed = False
    out = {}
    for k, v in state_dict.items():
        nk = k
        for pfx in prefixes:
            if nk.startswith(pfx):
                nk = nk[len(pfx):]
        if nk != k:
            changed = True
        out[nk] = v
    return out, changed


def main():
    parser = argparse.ArgumentParser(description="Standalone LightGBM pricer trainer")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("--data", required=True, help="Path to features parquet file")
    parser.add_argument(
        "--val-data",
        default=None,
        help="Optional separate parquet for LightGBM validation "
             "(uses full file as validation set)",
    )
    parser.add_argument("--pricer-path", required=True, help="Path to fourier.json")
    parser.add_argument("--product", default=None,
                        help="Which product to train (default: all from config)")
    parser.add_argument("--data-frac", type=float, default=1.0,
                        help="Use tail fraction of data (default: 1.0)")
    parser.add_argument("--val-data-frac", type=float, default=1.0,
                        help="Use tail fraction of --val-data (default: 1.0)")
    parser.add_argument(
        "--val-on-forecast-train",
        action="store_true",
        help="Train LGBM on full --data and validate on full forecast train split "
             "(first train_frac chunk of --data)",
    )
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--lgbm-var-source",
        choices=["model", "realized"],
        default=None,
        help="Variance source for LightGBM pricing features "
             "(default: run config if available)",
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir)

    # ── 1. Load config ──
    config = json.loads((run_path / "config.json").read_text())
    model_target = config["model_target"]
    lookback = config["lookback"]
    train_frac = config["train_frac"]
    val_frac = config["val_frac"]
    dataset_version = config.get("dataset_version", "v1")
    loss_mode = config.get("loss_mode", "log_mse")
    variable_horizon = config.get("variable_horizon")
    products = config["products"]
    horizon = config.get("horizon", "15m")
    batch_size = args.batch_size or config.get("batch_size", 4096)
    lgbm_cfg = config.get("lgbm_pricer", {})
    cfg_use_realized_var = bool(lgbm_cfg.get("use_realized_var", False))
    if args.lgbm_var_source is None:
        use_realized_var = cfg_use_realized_var
    else:
        use_realized_var = (args.lgbm_var_source == "realized")

    log_target = loss_mode.startswith("log") or loss_mode == "mixed_mse"

    vh_enabled = variable_horizon and variable_horizon.get("enabled", False)
    if vh_enabled:
        vh_min = variable_horizon.get("min_horizon", 1)
        vh_max = variable_horizon.get("max_horizon", 900)
    else:
        from vol_forecasting.trainer import _parse_horizon
        _, h_secs = _parse_horizon(horizon)
        vh_min = h_secs
        vh_max = h_secs

    # Products to train
    train_products = [args.product] if args.product else products

    # ── 2. Load dataset module ──
    ds = _load_dataset_module(dataset_version)
    GPUDataset = ds.GPUDataset
    BatchAugmenter = ds.BatchAugmenter
    precompute_prefix_sums = ds.precompute_prefix_sums
    build_prefix_stack = ds.build_prefix_stack

    # ── 3. Load + preprocess parquet(s) ──
    def _load_preprocessed_parquet(path: str, frac: float, tag: str) -> pl.DataFrame:
        print(f"Loading {tag} data from {path}")
        out = pl.read_parquet(path)

        # Drop GARCH columns
        garch_cols = [c for c in out.columns if "GARCH" in c]
        out = out.drop(garch_cols)

        # Clip HAR columns to [-30, 0]
        har_cols = [c for c in out.columns if "HAR_log" in c]
        out = out.with_columns([pl.col(c).clip(-30, 0) for c in har_cols])

        # Optional tail slice
        if frac < 1.0:
            keep = int(len(out) * frac)
            out = out.tail(keep)
            print(f"Using tail {frac:.0%} of {tag} data: {len(out):,} rows")
        return out

    df = _load_preprocessed_parquet(args.data, args.data_frac, "train")
    df_val_external = None
    if args.val_data is not None:
        df_val_external = _load_preprocessed_parquet(
            args.val_data, args.val_data_frac, "val"
        )
        if args.val_on_forecast_train:
            print("WARNING: --val-data provided; ignoring --val-on-forecast-train")

    # ── 4. Load pricer ──
    pricer = BSFourierPricer.from_json(args.pricer_path)
    print(f"BSFourier pricer: {pricer}")
    print(f"LightGBM variance source: {'realized' if use_realized_var else 'model'}")

    dev = torch.device(args.device)
    use_amp = args.device == "cuda"
    amp_dtype = torch.bfloat16

    # ── 5. Prefix sums (shared across products) ──
    prefix_rvs_train = precompute_prefix_sums(df, products)
    prefix_stack_train = build_prefix_stack(prefix_rvs_train, products)
    prefix_stack_val_external = None
    if df_val_external is not None:
        prefix_rvs_val = precompute_prefix_sums(df_val_external, products)
        prefix_stack_val_external = build_prefix_stack(prefix_rvs_val, products)

    for prod in train_products:
        print(f"\n{'=' * 60}")
        print(f"Training LightGBM pricers for product: {prod}")
        print(f"{'=' * 60}")

        prod_dir = run_path / prod
        ckpt_path = prod_dir / "best.pt"
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}, skipping")
            continue

        # ── 6. Load checkpoint ──
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        state_dict = ckpt["state_dict"]
        model_kwargs = ckpt["model_kwargs"]
        feat_mean = ckpt["feat_mean"]
        feat_std = ckpt["feat_std"]
        tgt_mean = ckpt["tgt_mean"]
        tgt_std = ckpt["tgt_std"]
        feature_cols = ckpt["feature_cols"]

        # ── 7. Extract & normalize features ──
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} feature columns. First 5: {missing[:5]}"
            )

        features = df.select(feature_cols).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        n = len(features)

        # Normalize with checkpoint stats
        features = (features - feat_mean) / feat_std

        # ── 8. Build train/val splits and GPU datasets ──
        prod_idx = products.index(prod)
        stride = 1

        if df_val_external is not None:
            # Train on full --data; validate on full --val-data.
            train_features = features
            n_train = len(train_features)

            missing_val = [c for c in feature_cols if c not in df_val_external.columns]
            if missing_val:
                raise ValueError(
                    f"Missing {len(missing_val)} val feature columns. "
                    f"First 5: {missing_val[:5]}"
                )
            val_features = df_val_external.select(feature_cols).to_numpy().astype(np.float32)
            val_features = np.nan_to_num(val_features, nan=0.0, posinf=0.0, neginf=0.0)
            val_features = (val_features - feat_mean) / feat_std
            n_val = len(val_features)

            prefix_train_end = min(n_train + vh_max + 2, prefix_stack_train.shape[1])
            prefix_val_end = min(n_val + vh_max + 2, prefix_stack_val_external.shape[1])

            train_prefix = prefix_stack_train[:, :prefix_train_end]
            val_prefix = prefix_stack_val_external[:, :prefix_val_end]

            df_lgbm = pl.concat([df, df_val_external], how="vertical")
            lgbm_n_train = len(df)
            lgbm_n_val = len(df_val_external)
            split_desc = "train=full --data | val=full --val-data"

        elif args.val_on_forecast_train:
            # Train on full --data; validate on full forecast-train split.
            n_forecast_train = int(n * train_frac)
            if n_forecast_train <= 0:
                raise ValueError("Forecast train split is empty; check train_frac/data size")

            train_features = features
            val_features = features[:n_forecast_train]
            n_train = len(train_features)
            n_val = len(val_features)

            prefix_train_end = min(n_train + vh_max + 2, prefix_stack_train.shape[1])
            prefix_val_end = min(n_val + vh_max + 2, prefix_stack_train.shape[1])

            train_prefix = prefix_stack_train[:, :prefix_train_end]
            val_prefix = prefix_stack_train[:, :prefix_val_end]

            df_val_slice = df[:n_forecast_train]
            df_lgbm = pl.concat([df, df_val_slice], how="vertical")
            lgbm_n_train = len(df)
            lgbm_n_val = len(df_val_slice)
            split_desc = "train=full --data | val=full forecast-train split"

        else:
            # Original behavior from run config fractions.
            n_train = int(n * train_frac)
            n_val = int(n * val_frac)
            train_features = features[:n_train]
            val_features = features[n_train:n_train + n_val]

            prefix_train_end = min(n_train + vh_max + 2, prefix_stack_train.shape[1])
            prefix_val_end = min(n_train + n_val + vh_max + 2, prefix_stack_train.shape[1])

            train_prefix = prefix_stack_train[:, :prefix_train_end]
            val_prefix = prefix_stack_train[:, n_train:prefix_val_end]

            df_lgbm = df
            lgbm_n_train = n_train
            lgbm_n_val = n_val
            split_desc = "train=train_frac | val=val_frac (forecast-style contiguous)"

        train_gds = GPUDataset(
            train_features,
            train_prefix,
            lookback, stride, dev,
            variable_horizon=vh_enabled, min_horizon=vh_min, max_horizon=vh_max,
            prod_idx=prod_idx, log_target=log_target,
        )
        val_gds = GPUDataset(
            val_features,
            val_prefix,
            lookback, stride, dev,
            variable_horizon=vh_enabled, min_horizon=vh_min, max_horizon=vh_max,
            prod_idx=prod_idx, log_target=log_target,
        )
        # Use training target stats for val
        val_gds.tgt_mean = tgt_mean
        val_gds.tgt_std = tgt_std

        print(f"  Split mode: {split_desc}")
        print(f"  Data rows: train_src={len(df):,} | val_src={len(df_val_external) if df_val_external is not None else len(df):,}")
        print(f"  LGBM rows: train={n_train:,} | val={n_val:,}")

        # ── 9. Reconstruct model ──
        model_cls = _resolve_cls(model_target)
        model = model_cls(**model_kwargs).to(dev)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            normalized_state_dict, changed = _normalize_state_dict_keys(state_dict)
            if not changed:
                raise
            print("  Detected wrapped checkpoint keys; stripping prefixes and retrying load")
            model.load_state_dict(normalized_state_dict)
        model.eval()

        # ── 10. Train LightGBM pricers ──
        lgbm_results = train_lgbm_pricers(
            model,
            train_gds.features, train_gds.prefix_stack,
            val_gds.features, val_gds.prefix_stack,
            df_lgbm, lgbm_n_train, lgbm_n_val,
            lookback, len(products),
            tgt_mean, tgt_std,
            prod, prod_idx, pricer,
            device=dev,
            use_amp=use_amp, amp_dtype=amp_dtype,
            batch_size=batch_size,
            log_target=log_target,
            use_realized_var=use_realized_var,
            augmenter_cls=BatchAugmenter,
        )

        # ── 11. Print results ──
        print(f"\n  Results for {prod}:")
        print(f"    Variance source:    {lgbm_results.get('variance_source', 'model')}")
        print(f"    BS raw Brier:      {lgbm_results['brier_bs']:.4f}")
        print(f"    Calibrated Brier:  {lgbm_results['brier_cal']:.4f}")
        if "brier_lgbm_bs" in lgbm_results:
            print(f"    LightGBM-BS Brier: {lgbm_results['brier_lgbm_bs']:.4f}"
                  f"  (skill {lgbm_results['lgbm_bs_skill']:+.3f})")
        if "brier_lgbm_embed" in lgbm_results:
            print(f"    LightGBM-Embed Brier: {lgbm_results['brier_lgbm_embed']:.4f}"
                  f"  (skill {lgbm_results['lgbm_embed_skill']:+.3f})")

        # ── 12. Save LightGBM models ──
        lgbm_save = {}
        for mname in ["lgbm_bs", "lgbm_embed"]:
            key = f"{mname}_model"
            if key in lgbm_results:
                lgbm_save[mname] = lgbm_results[key].model_to_string()
        lgbm_save["metrics"] = {
            k: v for k, v in lgbm_results.items()
            if k not in ("lgbm_bs_model", "lgbm_embed_model",
                         "p_lgbm_bs", "p_lgbm_embed",
                         "realized", "p_bs", "p_cal", "horizons")
        }

        save_path = prod_dir / "lgbm_pricers.pt"
        torch.save(lgbm_save, save_path)
        print(f"  Saved LightGBM models → {save_path}")

        # ── 13. Save calibration plot ──
        plot_path = prod_dir / "calibration_lgbm_standalone.png"
        _calibration_plot(
            lgbm_results["p_cal"],
            lgbm_results["p_bs"],
            lgbm_results["realized"],
            lgbm_results["brier_cal"],
            lgbm_results["brier_bs"],
            epoch=0,
            save_path=plot_path,
            p_lgbm_bs=lgbm_results.get("p_lgbm_bs"),
            brier_lgbm_bs=lgbm_results.get("brier_lgbm_bs"),
            p_lgbm_embed=lgbm_results.get("p_lgbm_embed"),
            brier_lgbm_embed=lgbm_results.get("brier_lgbm_embed"),
        )
        print(f"  Saved calibration plot → {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
