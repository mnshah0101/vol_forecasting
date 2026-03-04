"""
bs_fourier_pricer.py
────────────────────────────────────────────────────────────────────────────────
Inference module: Black-Scholes + Fourier calibration → ITM probability.

Primary interface
─────────────────
    from bs_fourier_pricer import BSFourierPricer

    pricer = BSFourierPricer.from_json("fourier_params.json")

    prob = pricer.probability(
        spot                    = 88_825.0,
        strike                  = 89_000.0,
        forecasted_var          = 0.000342,   # your model output — see below
        tte_seconds             = 450.0,      # seconds remaining in the window
    )

What to pass as forecasted_var
───────────────────────────────
Pass your model's predicted sum of squared log-differences for the REMAINING
time window:

    forecasted_var = Σ ( log(p_{i+1}/p_i) )²   summed over remaining TTE

This is already the total variance in BS units — no annualisation, no TTE
scaling required.  It is exactly what goes into d2:

    d2 = [ log(S0/K)  −  forecasted_var / 2 ] / sqrt(forecasted_var)
    P  = N(d2)

tte_seconds is accepted as a second parameter and is used only to:
  1. Guard against division by zero at expiry (forecasted_var → 0)
  2. Provide context for diagnostic output

If your forecast is already guaranteed > 0 you can pass tte_seconds=None.

Relationship between forecasted_var and annualised vol
───────────────────────────────────────────────────────
For reference, the mapping is:

    forecasted_var  =  σ_ann²  ×  tte_seconds / 31_557_600

So a 60% annualised vol with 450 seconds remaining gives:
    forecasted_var  =  0.60²  ×  450 / 31_557_600  =  0.000171

The helpers total_variance() and ann_vol_from_forecasted_var() convert between
these representations if needed.

Black-Scholes model
───────────────────
    d2  = [ log(S0/K) − forecasted_var/2 ] / sqrt(forecasted_var)
    P   = N(d2)

    Zero-drift, zero-rate formulation appropriate for short-horizon crypto.
    At expiry (forecasted_var → 0): P → 1 if S0 > K, else P → 0.

Fourier calibration
───────────────────
    features(p)  = [1, sin(2πp), cos(2πp), ..., sin(2πNp), cos(2πNp)]
    P_calibrated = clip( P_BS + coef @ features(P_BS), 0, 1 )

    Coefficients are produced by bs_fourier_calibration.py and stored in
    fourier_params.json.

Loading params
──────────────
    pricer = BSFourierPricer.from_json("fourier_params.json")
    pricer = BSFourierPricer.from_dict({"n_harmonics": 5, "coef": [...]})
    pricer = BSFourierPricer(n_harmonics=5, coef=np.array([...]))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union, Optional

import numpy as np
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# Types and constants
# ─────────────────────────────────────────────────────────────────────────────

Numeric = Union[float, int, np.ndarray]

_EPS              = 1e-12
_SECONDS_PER_YEAR = 365.25 * 24 * 3600   # 31_557_600


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _arr(x: Numeric) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=np.float64))


def _bs(spot: np.ndarray,
         strike: np.ndarray,
         total_var: np.ndarray) -> np.ndarray:
    """
    N(d2) probability that S_T > K under zero-drift lognormal dynamics.

    Handles total_var ≈ 0 (at/near expiry) via the clamp — the result
    converges to the intrinsic binary value:
        1  if spot > strike
        0  if spot < strike
       0.5 if spot == strike
    """
    log_m   = np.log(np.clip(spot, _EPS, None) / np.clip(strike, _EPS, None))
    sq_var  = np.sqrt(np.clip(total_var, _EPS, None))
    d2      = (log_m - total_var / 2.0) / sq_var
    return norm.cdf(d2)


def _fourier_features(p: np.ndarray, n_harmonics: int) -> np.ndarray:
    """
    Feature matrix for calibration correction.
    Columns: [1, sin(2πp), cos(2πp), ..., sin(2πNp), cos(2πNp)]
    Shape: (len(p), 2*n_harmonics + 1)
    """
    cols = [np.ones(len(p))]
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * p
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    return np.column_stack(cols)


# ─────────────────────────────────────────────────────────────────────────────
# BSFourierPricer
# ─────────────────────────────────────────────────────────────────────────────

class BSFourierPricer:
    """
    Black-Scholes digital-call pricer with Fourier calibration correction.

    Parameters
    ──────────
    n_harmonics : int
        Number of Fourier harmonics used when fitting.
        Coefficient vector length must equal 2 * n_harmonics + 1.

    coef : array-like of length 2*n_harmonics+1
        Ridge regression coefficients from bs_fourier_calibration.py.
    """

    def __init__(self, n_harmonics: int, coef: np.ndarray):
        self.n_harmonics = int(n_harmonics)
        self.coef        = np.asarray(coef, dtype=np.float64).ravel()
        expected         = 2 * self.n_harmonics + 1
        if len(self.coef) != expected:
            raise ValueError(
                f"coef has {len(self.coef)} elements; "
                f"expected {expected} (= 2 × {self.n_harmonics} + 1)")

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "BSFourierPricer":
        """Load from fourier_params.json produced by bs_fourier_calibration.py."""
        data = json.loads(Path(path).read_text())
        return cls(n_harmonics=int(data["n_harmonics"]),
                   coef=np.array(data["coef"], dtype=np.float64))

    @classmethod
    def from_dict(cls, d: dict) -> "BSFourierPricer":
        return cls(n_harmonics=d["n_harmonics"], coef=np.array(d["coef"]))

    # ── Core ─────────────────────────────────────────────────────────────────

    def probability(self,
                     spot:           Numeric,
                     strike:         Numeric,
                     forecasted_var: Numeric,
                     tte_seconds:    Optional[Numeric] = None,
                     return_bs_raw:  bool = False,
                    ) -> Union[np.ndarray, tuple]:
        """
        Calibrated P(S_T > K).

        Parameters
        ──────────
        spot           : current mid price  (scalar or array)
        strike         : contract strike    (scalar or array)
        forecasted_var : your model's predicted Σ(log diff)² for remaining TTE.
                         This is total variance in BS units — pass directly,
                         no annualisation or TTE scaling needed.
        tte_seconds    : seconds remaining in the contract window (optional).
                         Used only to floor forecasted_var to a minimum of
                         tte_seconds × _MIN_VAR_PER_SECOND so the pricer
                         degrades gracefully as TTE → 0.
                         Pass None to skip the floor entirely.
        return_bs_raw  : if True, return (p_calibrated, p_bs_raw) tuple.

        Returns
        ───────
        np.ndarray of calibrated probabilities in [0, 1]
        (or tuple (calibrated, raw) if return_bs_raw=True)
        """
        s   = _arr(spot)
        k   = _arr(strike)
        tv  = _arr(forecasted_var)

        # Optional: floor total_var so it never goes to zero before expiry.
        # Uses a very small per-second minimum (1-tick move of ~0.1 bps² / s).
        if tte_seconds is not None:
            tte = _arr(tte_seconds)
            _MIN_VAR_PER_SEC = 1e-10
            tv  = np.maximum(tv, tte * _MIN_VAR_PER_SEC)

        p_bs  = _bs(s, k, tv)
        phi   = _fourier_features(np.clip(p_bs, 0.0, 1.0), self.n_harmonics)
        delta = phi @ self.coef
        p_cal = np.clip(p_bs + delta, 0.0, 1.0)

        if return_bs_raw:
            return p_cal, p_bs
        return p_cal

    def bs_probability(self,
                        spot:           Numeric,
                        strike:         Numeric,
                        forecasted_var: Numeric) -> np.ndarray:
        """Raw uncalibrated BS probability (no Fourier correction)."""
        return _bs(_arr(spot), _arr(strike), _arr(forecasted_var))

    def calibration_correction(self, bs_prob: Numeric) -> np.ndarray:
        """
        Additive Fourier correction for a vector of raw BS probabilities.
        p_calibrated = clip(bs_prob + correction, 0, 1)
        """
        p = np.clip(_arr(bs_prob), 0.0, 1.0)
        return _fourier_features(p, self.n_harmonics) @ self.coef

    # ── DataFrame convenience ─────────────────────────────────────────────────

    def probability_from_df(self,
                             df,
                             spot_col:    str = "spot",
                             strike_col:  str = "strike",
                             var_col:     str = "forecasted_var",
                             tte_col:     Optional[str] = "tte_seconds",
                            ) -> np.ndarray:
        """
        Price an entire DataFrame in one call. Accepts polars or pandas.

        Default column names: spot, strike, forecasted_var, tte_seconds.
        Set tte_col=None to skip the TTE floor.
        """
        def _col(name):
            try:
                import polars as pl
                if isinstance(df, pl.DataFrame):
                    return df[name].to_numpy()
            except ImportError:
                pass
            return df[name].values

        tte = _col(tte_col) if (tte_col and tte_col in df.columns) else None
        return self.probability(
            spot           = _col(spot_col),
            strike         = _col(strike_col),
            forecasted_var = _col(var_col),
            tte_seconds    = tte,
        )

    def __repr__(self) -> str:
        return (f"BSFourierPricer("
                f"n_harmonics={self.n_harmonics}, "
                f"coef_norm={np.linalg.norm(self.coef):.6f})")


# ─────────────────────────────────────────────────────────────────────────────
# Variance unit helpers
# ─────────────────────────────────────────────────────────────────────────────

def total_variance(ann_vol: Numeric, tte_seconds: Numeric) -> np.ndarray:
    """
    Convert annualised vol + TTE in seconds to total variance.

        forecasted_var = ann_vol² × tte_seconds / 31_557_600

    Useful if your vol source is annualised and you want to convert to the
    units expected by BSFourierPricer.probability().

    Example
    ───────
    >>> total_variance(ann_vol=0.60, tte_seconds=450)
    0.0001712...   # use this as forecasted_var
    """
    return _arr(ann_vol) ** 2 * _arr(tte_seconds) / _SECONDS_PER_YEAR


def ann_vol_from_forecasted_var(forecasted_var: Numeric,
                                  tte_seconds: Numeric) -> np.ndarray:
    """
    Recover implied annualised vol from forecasted_var and TTE.

        ann_vol = sqrt( forecasted_var × 31_557_600 / tte_seconds )
    """
    tv  = _arr(forecasted_var)
    tte = _arr(tte_seconds)
    return np.sqrt(tv * _SECONDS_PER_YEAR / np.clip(tte, _EPS, None))


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BS+Fourier pricer smoke test")
    parser.add_argument("--params",        required=True,
                        help="Path to fourier_params.json")
    parser.add_argument("--spot",          type=float, default=88_825.0)
    parser.add_argument("--strike",        type=float, default=88_825.0)
    parser.add_argument("--forecasted-var",type=float, default=None,
                        help="Σ(log diff)² for remaining TTE (total variance). "
                             "If omitted, derived from --ann-vol and --tte.")
    parser.add_argument("--ann-vol",       type=float, default=0.60,
                        help="Annualised vol — used only if --forecasted-var not given")
    parser.add_argument("--tte",           type=float, default=900.0,
                        help="Seconds remaining in window (default: 900 = full 15 min)")
    args = parser.parse_args()

    pricer = BSFourierPricer.from_json(args.params)

    fv = (args.forecasted_var if args.forecasted_var is not None
          else float(total_variance(args.ann_vol, args.tte).ravel()[0]))

    p_cal, p_bs = pricer.probability(
        args.spot, args.strike, fv, tte_seconds=args.tte, return_bs_raw=True)

    implied_vol = float(ann_vol_from_forecasted_var(fv, args.tte).ravel()[0])

    print(f"\nBS+Fourier Pricer  —  {pricer}")
    print(f"  Params file:         {args.params}")
    print(f"  Spot:                {args.spot:>12,.2f}")
    print(f"  Strike:              {args.strike:>12,.2f}")
    print(f"  TTE:                 {args.tte:>9.1f}s  ({args.tte/60:.2f} min)")
    print(f"  forecasted_var:      {fv:>12.8f}  [Σ(log diff)² for remaining TTE]")
    print(f"  Implied ann vol:     {implied_vol:>11.2%}")
    print(f"  P_BS  (raw):         {p_bs.item():>12.6f}")
    print(f"  P_cal (calibrated):  {p_cal.item():>12.6f}")
    print(f"  Correction:          {(p_cal - p_bs).item():>+12.6f}")

    # # Moneyness sweep
    # print(f"\n  Sweep over strikes — same forecasted_var={fv:.8f}, TTE={args.tte:.0f}s:")
    # print(f"  {'Strike':>10}  {'Moneyness':>10}  {'P_BS':>8}  {'P_cal':>8}  {'delta':>8}")
    # print(f"  {'─'*52}")
    # for pct in [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]:
    #     k = args.spot * (1 + pct / 100)
    #     pc, pb = pricer.probability(args.spot, k, fv, tte_seconds=args.tte,
    #                                  return_bs_raw=True)
    #     print(f"  {k:>10,.1f}  {pct:>+9.1f}%  {pb.item():>8.4f}  "
    #           f"{pc.item():>8.4f}  {(pc-pb).item():>+8.4f}")

    # # TTE sweep (same forecasted_var scaled proportionally)
    # print(f"\n  Sweep over TTE — spot={args.spot:.0f}, strike={args.strike:.0f}, "
    #       f"ann_vol={implied_vol:.0%}:")
    # print(f"  {'TTE (s)':>8}  {'TTE (min)':>9}  {'fcast_var':>10}  "
    #       f"{'P_BS':>8}  {'P_cal':>8}")
    # print(f"  {'─'*52}")
    # for tte_s in [900, 600, 300, 120, 60, 30, 10, 1]:
    #     fv_s = float(total_variance(implied_vol, tte_s).ravel()[0])
    #     pc, pb = pricer.probability(args.spot, args.strike, fv_s,
    #                                  tte_seconds=tte_s, return_bs_raw=True)
    #     print(f"  {tte_s:>8.0f}  {tte_s/60:>9.2f}  {fv_s:>10.8f}  "
    #           f"{pb.item():>8.4f}  {pc.item():>8.4f}")
