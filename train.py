"""
Training Entry Point (Hydra)
============================
Usage:
    python -m vol_forecasting.train
    python -m vol_forecasting.train data.path=other.parquet training.horizon=5m
    python -m vol_forecasting.train model.dropout=0.3 training.epochs=50
    python -m vol_forecasting.train model._target_=vol_forecasting.models.other.OtherModel
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import polars as pl
import torch

from vol_forecasting.trainer import train


@hydra.main(version_base=None, config_path="configs", config_name="nhits_config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = cfg.device if torch.cuda.is_available() else 'cpu'

    print("Loading data...")
    df = pl.read_parquet(cfg.data.path)

    har_cols = [c for c in df.columns if 'HAR_log' in c]
    garch_cols = [c for c in df.columns if 'GARCH' in c]
    df = df.drop(garch_cols)
    df = df.with_columns([pl.col(c).clip(-30, 0) for c in har_cols])

    # ── Optional contiguous tail slice ──
    data_frac = cfg.data.get('data_frac', 1.0)
    if data_frac < 1.0:
        keep = int(len(df) * data_frac)
        df = df.tail(keep)
        print(f"data_frac={data_frac}: kept last {keep:,} rows of {keep / data_frac:,.0f}")

    print(f"Data shape: {df.shape}")

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_target = model_cfg.pop('_target_')
    lookback = model_cfg.pop('lookback')

    vh_cfg = OmegaConf.to_container(cfg.training.variable_horizon, resolve=True) \
        if 'variable_horizon' in cfg.training else None

    # ── Pricing head config ──
    ph_cfg = OmegaConf.to_container(cfg.training.pricing_head, resolve=True) \
        if 'pricing_head' in cfg.training else {}

    # ── LightGBM pricer config ──
    lgbm_cfg = OmegaConf.to_container(cfg.training.lgbm_pricer, resolve=True) \
        if 'lgbm_pricer' in cfg.training else {}

    results = train(
        df,
        model_target=model_target,
        model_kwargs=model_cfg,
        products=list(cfg.data.products),
        horizon=cfg.training.horizon,
        lookback=lookback,
        stride=cfg.training.stride,
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs,
        patience=cfg.training.patience,
        device=device,
        use_compile=cfg.training.use_compile,
        variable_horizon=vh_cfg,
        ntfy_channel=cfg.get('ntfy_channel', None),
        pricer_path=cfg.get('pricer_path', None),
        jsu_params_path=cfg.get('jsu_params_path', None),
        pricer_period_min=cfg.get('pricer_period_min', 15),
        run_dir=cfg.get('run_dir', None),
        loss_mode=cfg.training.get('loss_mode', 'log_mse'),
        loss_alpha=cfg.training.get('loss_alpha', 1.0),
        target_clip=cfg.training.get('target_clip', None),
        pricing_head_enabled=ph_cfg.get('enabled', False),
        pricing_head_kwargs=ph_cfg.get('kwargs', {}),
        bce_weight=ph_cfg.get('bce_weight', 0.1),
        pricing_head_period_len=ph_cfg.get('period_len', 900),
        dataset_version=cfg.data.get('dataset_version', 'v1'),
        lgbm_pricer_enabled=lgbm_cfg.get('enabled', False),
        lgbm_pricer_use_realized_var=lgbm_cfg.get('use_realized_var', False),
    )

    model_name = model_target.rsplit('.', 1)[-1].lower()
    vh_active = vh_cfg and vh_cfg.get('enabled', False)
    if vh_active:
        save_path = cfg.save_path or f'{model_name}_vh_{vh_cfg["min_horizon"]}_{vh_cfg["max_horizon"]}.pt'
    else:
        save_path = cfg.save_path or f'{model_name}_{cfg.training.horizon}.pt'
    loss_mode = cfg.training.get('loss_mode', 'log_mse')
    save_dict = {}
    for prod, r in results.items():
        save_dict[prod] = {
            'state_dict': r['state_dict'],
            'feat_mean': r['feat_mean'],
            'feat_std': r['feat_std'],
            'tgt_mean': r['tgt_mean'],
            'tgt_std': r['tgt_std'],
            'feature_cols': r['feature_cols'],
            'horizon': r['horizon'],
            'horizon_secs': r['horizon_secs'],
            'test_r2': r['test_r2'],
            'loss_mode': loss_mode,
        }
        if 'variable_horizon' in r:
            save_dict[prod]['variable_horizon'] = r['variable_horizon']
    torch.save(save_dict, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
