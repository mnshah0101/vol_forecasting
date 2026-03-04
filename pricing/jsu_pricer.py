"""
jsu_pricer.py
────────────────────────────────────────────────────────────────────────────────
Johnson SU pricer for BTC 15-minute binary options.

Each row in params1.csv contains a JSU(a, b, loc, scale) fit to the
log-return distribution at a specific time-to-expiry bucket (200ms steps,
200ms–900s). Given the current spot price, a strike, and the TTE in
milliseconds, this module:

  1. Looks up (or interpolates) the JSU parameters for that TTE bucket
  2. Computes P(S_T > K) = 1 - JSU.cdf( log(K/S) )
  3. Applies no post-hoc correction (params are the model)

Handling degenerate short-TTE fits
────────────────────────────────────
Fits below ~16s TTE have scale values of 1e-25 to 1e-12 — the MLE
optimizer collapsed because there is essentially no price movement in
200ms windows. These rows are detected (scale < MIN_VALID_SCALE) and
handled by one of three strategies (configurable):

  "clamp"       — use the nearest valid TTE params (default, conservative)
  "interpolate" — linearly interpolate a and b from valid neighbours;
                  extrapolate scale as scale_ref * sqrt(tte / tte_ref)
  "intrinsic"   — return the binary intrinsic value (1 if S>K, 0 if S<K)
                  for TTE below the valid threshold

Usage
─────
    from jsu_pricer import JSUPricer

    pricer = JSUPricer.from_csv("params1.csv")

    # Scalar
    prob = pricer.probability(spot=88825.0, strike=89000.0, tte_ms=450_000)

    # Vectorised
    probs = pricer.probability(
        spot    = np.array([88825, 88825]),
        strike  = np.array([89000, 88500]),
        tte_ms  = np.array([450_000, 60_000]),
    )

    # From a polars/pandas DataFrame
    probs = pricer.probability_from_df(df,
        spot_col="s_mid", strike_col="strike", tte_ms_col="tte_ms")

Interpolation detail
────────────────────
When interpolate=True, the pricer builds smooth cubic splines over a, b, loc
and a sqrt-scaled version of scale vs TTE for all valid rows, then evaluates
at the exact requested TTE. This gives a continuous probability surface
instead of a step function over 200ms buckets.

params1.csv columns used: tte_ms, a, b, loc, scale
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union, Optional, Literal

import numpy as np
from scipy.stats import johnsonsu
from scipy.interpolate import CubicSpline

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MIN_VALID_SCALE = 1e-4    # scale below this → degenerate MLE fit
EPS             = 1e-12
Numeric         = Union[float, int, np.ndarray]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _arr(x: Numeric) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=np.float64))


def _jsu_itm_prob(log_moneyness: np.ndarray,
                   a: np.ndarray, b: np.ndarray,
                   loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    P(S_T > K) = P(log(S_T/S_0) > log(K/S_0))
               = 1 - JSU.cdf(log(K/S_0))
               = JSU.sf(log_moneyness)

    where log_moneyness = log(K / spot).

    Vectorised: all inputs must be same-length arrays.
    """
    return johnsonsu.sf(log_moneyness, a, b, loc=loc, scale=scale)


# ─────────────────────────────────────────────────────────────────────────────
# JSUPricer
# ─────────────────────────────────────────────────────────────────────────────

class JSUPricer:
    """
    Johnson SU binary options pricer backed by per-TTE fitted parameters.

    Parameters
    ──────────
    tte_ms  : np.ndarray, shape (N,)  — TTE bucket centres in milliseconds
    a       : np.ndarray, shape (N,)  — JSU skewness parameter
    b       : np.ndarray, shape (N,)  — JSU shape parameter
    loc     : np.ndarray, shape (N,)  — JSU location (≈ mean log-return)
    scale   : np.ndarray, shape (N,)  — JSU scale (≈ σ√TTE in log-return units)
    valid   : np.ndarray bool (N,)    — True where MLE fit is not degenerate
    short_tte_strategy : "clamp" | "interpolate" | "intrinsic"
    """

    def __init__(self,
                  tte_ms:   np.ndarray,
                  a:        np.ndarray,
                  b:        np.ndarray,
                  loc:      np.ndarray,
                  scale:    np.ndarray,
                  valid:    np.ndarray,
                  short_tte_strategy: Literal["clamp","interpolate","intrinsic"] = "clamp"):

        self.tte_ms   = tte_ms.astype(np.float64)
        self.a        = a.astype(np.float64)
        self.b        = b.astype(np.float64)
        self.loc      = loc.astype(np.float64)
        self.scale    = scale.astype(np.float64)
        self.valid    = valid.astype(bool)
        self.strategy = short_tte_strategy

        self._min_valid_tte = float(self.tte_ms[self.valid].min())
        self._max_valid_tte = float(self.tte_ms[self.valid].max())

        # Build lookup structures
        self._build_lookup()

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_csv(cls,
                  path: Union[str, Path],
                  short_tte_strategy: Literal["clamp","interpolate","intrinsic"] = "clamp",
                  min_valid_scale: float = MIN_VALID_SCALE) -> "JSUPricer":
        """
        Load from params1.csv (or any CSV with columns: tte_ms, a, b, loc, scale).

        Rows where scale < min_valid_scale are flagged as degenerate and
        handled according to short_tte_strategy.
        """
        try:
            import polars as pl
            df    = pl.read_csv(path)
            tte   = df["tte_ms"].to_numpy().astype(np.float64)
            a     = df["a"].to_numpy().astype(np.float64)
            b     = df["b"].to_numpy().astype(np.float64)
            loc   = df["loc"].to_numpy().astype(np.float64)
            scale = df["scale"].to_numpy().astype(np.float64)
        except ImportError:
            import pandas as pd
            df    = pd.read_csv(path)
            tte   = df["tte_ms"].to_numpy().astype(np.float64)
            a     = df["a"].to_numpy().astype(np.float64)
            b     = df["b"].to_numpy().astype(np.float64)
            loc   = df["loc"].to_numpy().astype(np.float64)
            scale = df["scale"].to_numpy().astype(np.float64)

        # Sort by TTE ascending (should already be, but be safe)
        order = np.argsort(tte)
        tte, a, b, loc, scale = tte[order], a[order], b[order], loc[order], scale[order]

        valid = scale >= min_valid_scale

        n_valid = valid.sum()
        n_degen = (~valid).sum()
        print(f"  Loaded {len(tte):,} TTE buckets from {path}")
        print(f"  Valid: {n_valid:,}  (TTE {tte[valid].min():.0f}ms – {tte[valid].max():.0f}ms)")
        print(f"  Degenerate (scale<{min_valid_scale:.0e}): {n_degen:,}  "
              f"(TTE {tte[~valid].min():.0f}ms – {tte[~valid].max():.0f}ms)  "
              f"→ strategy='{short_tte_strategy}'")

        return cls(tte, a, b, loc, scale, valid, short_tte_strategy)

    @classmethod
    def from_dict(cls, d: dict, **kwargs) -> "JSUPricer":
        """Construct from a dict with keys: tte_ms, a, b, loc, scale."""
        return cls(
            tte_ms = np.array(d["tte_ms"]),
            a      = np.array(d["a"]),
            b      = np.array(d["b"]),
            loc    = np.array(d["loc"]),
            scale  = np.array(d["scale"]),
            valid  = np.array(d.get("valid", np.ones(len(d["tte_ms"]), dtype=bool))),
            **kwargs,
        )

    # ── Lookup / interpolation ────────────────────────────────────────────────

    def _build_lookup(self):
        """
        Pre-build all lookup structures used at inference time.

        For "clamp": store valid arrays sorted by TTE for searchsorted.
        For "interpolate": build CubicSpline objects over valid TTE range.
        """
        v = self.valid
        self._v_tte   = self.tte_ms[v]
        self._v_a     = self.a[v]
        self._v_b     = self.b[v]
        self._v_loc   = self.loc[v]
        self._v_scale = self.scale[v]

        if self.strategy == "interpolate":
            # Splines for a, b, loc vs TTE
            self._sp_a   = CubicSpline(self._v_tte, self._v_a,   extrapolate=True)
            self._sp_b   = CubicSpline(self._v_tte, self._v_b,   extrapolate=True)
            self._sp_loc = CubicSpline(self._v_tte, self._v_loc, extrapolate=True)
            # Scale ~ sqrt(TTE): fit spline to scale / sqrt(TTE), extrapolate in that space
            sqrt_tte       = np.sqrt(self._v_tte)
            scale_norm     = self._v_scale / sqrt_tte
            self._sp_scale_norm = CubicSpline(self._v_tte, scale_norm, extrapolate=True)

    def _lookup_params(self, tte_ms_arr: np.ndarray) -> tuple:
        """
        For each requested TTE, return (a, b, loc, scale) arrays.

        Handles:
          - TTE within valid range: interpolate or nearest-bucket lookup
          - TTE below valid range: clamp / interpolate / intrinsic flag
          - TTE above valid range: clamp to max valid TTE
        """
        n = len(tte_ms_arr)

        if self.strategy == "interpolate":
            # Evaluate splines everywhere; mark sub-threshold for intrinsic fallback
            t  = np.clip(tte_ms_arr, self._min_valid_tte, self._max_valid_tte)
            a  = self._sp_a(t)
            b  = np.clip(self._sp_b(t), 0.01, None)   # b must be positive
            lo = self._sp_loc(t)
            sc = self._sp_scale_norm(t) * np.sqrt(t)
            sc = np.clip(sc, EPS, None)
            return a, b, lo, sc

        else:
            # "clamp" or "intrinsic": nearest-neighbour on valid TTE array
            # Clamp requested TTE into valid range for lookup
            t_clamped = np.clip(tte_ms_arr, self._min_valid_tte, self._max_valid_tte)
            idx       = np.searchsorted(self._v_tte, t_clamped, side="left")
            idx       = np.clip(idx, 0, len(self._v_tte) - 1)
            # Pick closest neighbour
            idx_l     = np.clip(idx - 1, 0, len(self._v_tte) - 1)
            dl        = np.abs(self._v_tte[idx_l] - t_clamped)
            dr        = np.abs(self._v_tte[idx]   - t_clamped)
            best      = np.where(dl <= dr, idx_l, idx)

            return (self._v_a[best], self._v_b[best],
                    self._v_loc[best], self._v_scale[best])

    # ── Core inference ────────────────────────────────────────────────────────

    def probability(self,
                     spot:   Numeric,
                     strike: Numeric,
                     tte_ms: Numeric,
                     forecasted_var: Optional[Numeric] = None,
                    ) -> np.ndarray:
        """
        P(S_T > K): calibrated ITM probability using per-TTE JSU parameters.

        Parameters
        ──────────
        spot   : current mid price  (scalar or array)
        strike : contract strike    (scalar or array)
        tte_ms : time to expiry in milliseconds  (scalar or array)
        forecasted_var : predicted forward RV over the horizon (scalar or array).
            When provided, replaces the JSU's fitted `scale` parameter with
            sqrt(forecasted_var), injecting the model's vol prediction while
            keeping the JSU shape (a, b) and location from the historical fit.

        Returns
        ───────
        np.ndarray of probabilities in [0, 1]

        Notes
        ─────
        - For TTE below ~16s (degenerate fit zone), behaviour depends on
          short_tte_strategy:
            "clamp"       → use nearest valid params (16s params)
            "interpolate" → extrapolate smoothly toward intrinsic
            "intrinsic"   → return 1.0 if spot>strike else 0.0
        - All inputs broadcast: pass scalar TTE with array spot/strike,
          or all three as matching arrays.
        """
        s   = _arr(spot)
        k   = _arr(strike)
        t   = _arr(tte_ms)

        # Broadcast to common length
        s, k, t = np.broadcast_arrays(s, k, t)
        s, k, t = s.copy(), k.copy(), t.copy()

        a, b, loc, scale = self._lookup_params(t)

        # Override scale with model's predicted vol if provided
        if forecasted_var is not None:
            fv = _arr(forecasted_var)
            scale = np.sqrt(np.clip(fv, EPS, None))

        log_m = np.log(np.clip(k, EPS, None) / np.clip(s, EPS, None))
        probs = _jsu_itm_prob(log_m, a, b, loc, scale)

        # For "intrinsic" strategy: override sub-threshold TTE rows
        if self.strategy == "intrinsic":
            sub_thresh = t < self._min_valid_tte
            if sub_thresh.any():
                intrinsic = (s[sub_thresh] >= k[sub_thresh]).astype(np.float64)
                probs     = probs.copy()
                probs[sub_thresh] = intrinsic

        return np.clip(probs, 0.0, 1.0)

    # ── DataFrame convenience ─────────────────────────────────────────────────

    def probability_from_df(self,
                             df,
                             spot_col:   str = "spot",
                             strike_col: str = "strike",
                             tte_ms_col: str = "tte_ms",
                            ) -> np.ndarray:
        """
        Price an entire DataFrame in one call. Accepts polars or pandas.
        Returns np.ndarray of probabilities aligned with the DataFrame rows.
        """
        def _col(name):
            try:
                import polars as pl
                if isinstance(df, pl.DataFrame):
                    return df[name].to_numpy()
            except ImportError:
                pass
            return df[name].values

        return self.probability(
            spot   = _col(spot_col),
            strike = _col(strike_col),
            tte_ms = _col(tte_ms_col),
        )

    # ── Parameter inspection ─────────────────────────────────────────────────

    def params_at(self, tte_ms: float) -> dict:
        """
        Return the JSU parameters that will be used for a given TTE.
        Useful for inspection and debugging.
        """
        t = _arr(tte_ms)
        a, b, loc, scale = self._lookup_params(t)
        is_valid = float(tte_ms) >= self._min_valid_tte
        return {
            "tte_ms":   float(tte_ms),
            "a":        float(a[0]),
            "b":        float(b[0]),
            "loc":      float(loc[0]),
            "scale":    float(scale[0]),
            "valid":    is_valid,
            "strategy": self.strategy if not is_valid else "direct",
        }

    def __repr__(self) -> str:
        return (f"JSUPricer(buckets={len(self.tte_ms):,}, "
                f"valid={self.valid.sum():,}, "
                f"strategy='{self.strategy}', "
                f"tte_range={self._min_valid_tte:.0f}–{self._max_valid_tte:.0f}ms)")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────────────────────────────────────

def price(spot: Numeric, strike: Numeric, tte_ms: Numeric,
           params_csv: Union[str, Path],
           strategy: Literal["clamp","interpolate","intrinsic"] = "clamp",
          ) -> np.ndarray:
    """
    One-liner: load params and return probability in a single call.

    prob = price(spot=88825, strike=89000, tte_ms=450_000,
                 params_csv="params1.csv")
    """
    return JSUPricer.from_csv(params_csv, strategy).probability(spot, strike, tte_ms)


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JSU pricer smoke test")
    parser.add_argument("--params",   required=True, help="Path to params1.csv")
    parser.add_argument("--spot",     type=float, default=88_825.0)
    parser.add_argument("--strike",   type=float, default=88_825.0)
    parser.add_argument("--tte-ms",   type=float, default=450_000.0,
                        help="TTE in milliseconds (default: 450000 = 7.5 min)")
    parser.add_argument("--strategy", default="clamp",
                        choices=["clamp","interpolate","intrinsic"])
    args = parser.parse_args()

    print(f"\nLoading params from {args.params} ...")
    pricer = JSUPricer.from_csv(args.params, short_tte_strategy=args.strategy)
    print(f"  {pricer}\n")

    # Single price
    prob = pricer.probability(args.spot, args.strike, args.tte_ms)
    p    = pricer.params_at(args.tte_ms)
    print(f"Single price:")
    print(f"  Spot:     {args.spot:>12,.2f}")
    print(f"  Strike:   {args.strike:>12,.2f}")
    print(f"  TTE:      {args.tte_ms/1000:>9.1f}s  ({args.tte_ms/60000:.2f} min)")
    print(f"  JSU a:    {p['a']:>12.6f}")
    print(f"  JSU b:    {p['b']:>12.6f}")
    print(f"  JSU loc:  {p['loc']:>12.8f}")
    print(f"  JSU scale:{p['scale']:>12.8f}")
    print(f"  P(S>K):   {prob.item():>12.6f}")

    # Moneyness sweep
    print(f"\n  Moneyness sweep — TTE={args.tte_ms/1000:.0f}s, spot={args.spot:.0f}:")
    print(f"  {'Strike':>10}  {'Moneyness':>10}  {'P(S>K)':>10}")
    print(f"  {'─'*34}")
    for pct in [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]:
        k  = args.spot * (1 + pct / 100)
        pb = pricer.probability(args.spot, k, args.tte_ms)
        print(f"  {k:>10,.1f}  {pct:>+9.1f}%  {pb.item():>10.5f}")

    # TTE sweep (ATM)
    print(f"\n  TTE sweep — ATM (spot=strike={args.spot:.0f}):")
    print(f"  {'TTE (ms)':>10}  {'TTE (s)':>8}  {'a':>8}  {'b':>8}  "
          f"{'scale':>10}  {'P(S>K)':>10}  {'valid':>6}")
    print(f"  {'─'*70}")
    for tte_s in [900, 600, 300, 120, 60, 30, 16, 10, 5, 1]:
        tte_ms = tte_s * 1000
        pb     = pricer.probability(args.spot, args.spot, tte_ms)
        p2     = pricer.params_at(tte_ms)
        print(f"  {tte_ms:>10.0f}  {tte_s:>8.0f}  {p2['a']:>8.4f}  {p2['b']:>8.4f}  "
              f"{p2['scale']:>10.6f}  {pb.item():>10.5f}  "
              f"{'✓' if p2['valid'] else '✗ '+args.strategy:>6}")
