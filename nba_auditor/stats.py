"""Statistics for return-series audits.

All Sharpe ratios inside this module are *per period* (not annualized) unless a name says `_ann`.
Kurtosis is the plain (non-excess) fourth standardized moment, so a normal distribution has 3,
matching the notation in Bailey & López de Prado.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def moments(x: np.ndarray) -> tuple[float, float, float, float]:
    """Mean, standard deviation (ddof=1), skewness and kurtosis (non-excess, population moments)."""
    x = np.asarray(x, float)
    mu = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    c = x - mu
    m2 = float((c**2).mean())
    if m2 <= 0:
        return mu, sd, 0.0, 3.0
    skew = float((c**3).mean() / m2**1.5)
    kurt = float((c**4).mean() / m2**2)
    return mu, sd, skew, kurt


def sharpe(x: np.ndarray) -> float:
    mu, sd, _, _ = moments(x)
    return mu / sd if sd and sd > 0 else float("nan")


def newey_west_lags(n: int) -> int:
    """Rule-of-thumb bandwidth floor(4 * (n/100)^(2/9)) (Newey & West 1994)."""
    return max(1, int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def newey_west_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Bartlett kernel) t-stat of the mean of x."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = newey_west_lags(n)
    e = x - x.mean()
    lr_var = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1 - k / (lags + 1)
        lr_var += 2 * w * (e[k:] @ e[:-k]) / n
    if lr_var <= 0:
        return float("nan")
    return float(x.mean() / math.sqrt(lr_var / n))


def hac_bandwidth(x: np.ndarray) -> int:
    """Bartlett bandwidth that can see overlapping returns.

    The Newey-West rule of thumb (7 lags at T=1260) cannot see a 20-day overlap. We also count the run
    of consecutive significantly positive autocorrelations (rho_k > 2/sqrt(T)) starting at lag 1 and use
    twice that run, capped at T/4. For independent returns the rule of thumb wins.
    """
    x = np.asarray(x, float)
    T = len(x)
    base = newey_west_lags(T)
    max_lag = max(1, min(T // 5, 260))
    rho = autocorrelations(x, max_lag)
    thr = 2.0 / math.sqrt(T)
    run = 0
    while run < max_lag and rho[run] > thr:
        run += 1
    return int(min(max(base, 2 * run), max(1, T // 4)))


def variance_ratio(x: np.ndarray, lags: int) -> float:
    """Long-run (Bartlett/HAC) variance divided by the plain variance; ~1 for independent returns."""
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    g0 = e @ e / n
    if g0 <= 0:
        return 1.0
    lr = g0
    for k in range(1, min(lags, n - 1) + 1):
        lr += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(lr / g0)


def autocorrelations(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, float)
    e = x - x.mean()
    d = e @ e
    if d <= 0:
        return np.zeros(max_lag)
    return np.array([(e[k:] @ e[:-k]) / d for k in range(1, max_lag + 1)])


def lo_sharpe(x: np.ndarray, periods_per_year: float, max_lag: int | None = None) -> dict:
    """Lo-style (2002) adjustment: Lo's IID standard error, scaled by his autocorrelation correction.

    * SE(SR) under IID returns: sqrt((1 + SR^2 / 2) / T)   (Lo 2002, eq. 9 for normal returns).
    * Annualizing over q periods: SR(q) = eta(q) * SR, with
      eta(q) = q / sqrt(q + 2 * sum_{k=1}^{q-1} (q - k) * rho_k)   (Lo 2002, eq. 15).
      Autocorrelations beyond `max_lag` (default: the Newey-West bandwidth) are set to zero, because
      hundreds of noisy sample autocorrelations would dominate the sum.
    * The Lo-adjusted t-stat is the IID t-stat scaled by eta(q) / sqrt(q), i.e. by the same
      autocorrelation correction applied to the annualized Sharpe.
    """
    x = np.asarray(x, float)
    T = len(x)
    sr = sharpe(x)
    q = max(1, int(round(periods_per_year)))
    if max_lag is None:
        max_lag = newey_west_lags(T)
    L = min(max_lag, q - 1, T - 2)
    rho = autocorrelations(x, L) if L > 0 else np.zeros(0)
    s = q + 2 * sum((q - k) * rho[k - 1] for k in range(1, L + 1))
    eta = q / math.sqrt(s) if s > 0 else float("nan")
    se_iid = math.sqrt((1 + 0.5 * sr**2) / T)
    t_iid = sr / se_iid
    factor = eta / math.sqrt(q) if q > 0 and not math.isnan(eta) else float("nan")
    return {
        "sharpe_ann_naive": sr * math.sqrt(periods_per_year),
        "sharpe_ann_lo": sr * eta * math.sqrt(periods_per_year / q) if not math.isnan(eta) else float("nan"),
        "lo_eta_over_sqrt_q": factor,
        "t_iid": t_iid,
        "t_lo": t_iid * factor,
        "autocorr_lags_used": L,
        "rho_1": float(rho[0]) if len(rho) else 0.0,
    }


def expected_max_sharpe(n_trials: int, sr_variance: float, sr_mean: float = 0.0) -> float:
    """E[max of n_trials Sharpe estimates] ~ mean + sqrt(V) * ((1-g) Z^-1(1-1/N) + g Z^-1(1-1/(N e))).

    Bailey & López de Prado (2014), eq. (2) / appendix eq. (6). Returns `sr_mean` for N <= 1.
    """
    if n_trials <= 1:
        return sr_mean
    n = float(n_trials)
    z = (1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n) + EULER_GAMMA * norm.ppf(1 - 1.0 / (n * math.e))
    return float(sr_mean + math.sqrt(sr_variance) * z)


def _sr_denominator(sr: float, skew: float, kurt: float) -> float:
    v = 1 - skew * sr + (kurt - 1) / 4.0 * sr**2
    return math.sqrt(v) if v > 0 else float("nan")


def probabilistic_sharpe(sr: float, sr_benchmark: float, T: int, skew: float, kurt: float) -> float:
    """PSR(SR*) = Z[(SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)]   (Bailey & López de Prado 2012)."""
    d = _sr_denominator(sr, skew, kurt)
    if T < 2 or math.isnan(d):
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(T - 1) / d))


def deflated_sharpe(sr: float, T: int, skew: float, kurt: float, n_trials: int, sr_variance: float) -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014, eq. 2).

    sr, sr_variance: per-period Sharpe of the selected strategy and the variance of the per-period
    Sharpe ratios across all trials. DSR = PSR(SR0) with SR0 = expected max Sharpe of n_trials
    zero-skill strategies.
    """
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return {"sr0": sr0, "dsr": probabilistic_sharpe(sr, sr0, T, skew, kurt)}


def min_track_record_length(sr: float, sr_benchmark: float, skew: float, kurt: float, prob: float = 0.95) -> float:
    """MinTRL = 1 + (1 - g3 SR + (g4 - 1)/4 SR^2) (Z_prob / (SR - SR*))^2 observations.

    Bailey & López de Prado (2012), eq. (11) and appendix A.3. Infinite if SR <= SR*.
    """
    if not (sr > sr_benchmark):
        return float("inf")
    v = 1 - skew * sr + (kurt - 1) / 4.0 * sr**2
    return float(1 + v * (norm.ppf(prob) / (sr - sr_benchmark)) ** 2)


def max_drawdown(returns: np.ndarray) -> dict:
    """Max drawdown of the compounded equity curve, and the longest underwater spell (in periods).

    Indices refer to positions in `returns`; a peak index of -1 means the starting capital.
    """
    r = np.asarray(returns, float)
    if len(r) == 0:
        return {"max_drawdown": 0.0, "peak_index": -1, "trough_index": -1,
                "longest_underwater_periods": 0, "longest_underwater_end_index": -1, "currently_underwater": False}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        eq_full = np.concatenate([[1.0], np.cumprod(1 + r)])
        run_peak = np.maximum.accumulate(eq_full)
        dd = eq_full[1:] / run_peak[1:] - 1
    trough = int(np.argmin(dd))
    peak = int(np.argmax(eq_full[: trough + 2])) - 1
    underwater = dd < -1e-12
    longest, cur, end_longest = 0, 0, -1
    for i, u in enumerate(underwater):
        cur = cur + 1 if u else 0
        if cur > longest:
            longest, end_longest = cur, i
    return {
        "max_drawdown": float(dd.min()),
        "peak_index": peak,
        "trough_index": trough,
        "longest_underwater_periods": int(longest),
        "longest_underwater_end_index": int(end_longest),
        "currently_underwater": bool(underwater[-1]),
    }


def gaussian_top_share(mean: float, sd: float, n: int, frac: float = 0.05) -> float:
    """Expected share of total PnL from the top `frac` of periods if returns were iid normal.

    Expected sum of the top k = k * (mean + sd * phi(z) / frac) with z = Z^-1(1 - frac); total = n * mean.
    """
    if mean <= 0 or n <= 0:
        return float("nan")
    k = max(1, int(math.ceil(frac * n)))
    z = norm.ppf(1 - frac)
    top = k * (mean + sd * norm.pdf(z) / frac)
    return float(top / (n * mean))
