"""Reproduce the calibration table in README.md: verdict frequencies on synthetic series.

Run: python scripts/calibration.py   (a few minutes). Seeds are fixed.
"""
import collections
import warnings

import numpy as np
import pandas as pd

from nba_auditor import audit_returns

warnings.filterwarnings("ignore")
IDX = pd.bdate_range("2012-01-02", periods=2600)
T = 1260


def market(s, n):
    """An independent market series used as the benchmark (the strategies below have no market exposure)."""
    return np.random.default_rng(90_000 + s).normal(0.0003, 0.01, n)


def run(label, gen, n, with_benchmark=True, **kw):
    """Without a benchmark the best verdict is 'Some red flags' by design, so rows get one unless stated."""
    c = collections.Counter()
    for s in range(n):
        x = gen(s)
        idx = IDX[:len(x)]
        b = pd.Series(market(s, len(x)), index=idx) if with_benchmark else None
        c[audit_returns(pd.Series(x, index=idx), benchmark=b, **kw).verdict] += 1
    pct = {k: f"{100 * c[k] / n:.0f}%" for k in ("No red flags found", "Some red flags", "Major red flags")}
    print(f"{label:58s} {pct}")


def best(s):
    r = np.random.default_rng(10_000 + s).normal(0, 0.01, (T, 100))
    return r[:, np.argmax(r.mean(0) / r.std(0))]


def overlap(s):
    e = np.random.default_rng(40_000 + s).normal(0, 0.01, T + 19)
    return np.convolve(e, np.ones(20), "valid") / 20


def ar1(s, phi=0.3):
    e = np.random.default_rng(50_000 + s).normal(0, 0.01, T)
    x = np.empty(T)
    x[0] = e[0]
    for i in range(1, T):
        x[i] = phi * x[i - 1] + e[i]
    return x


def skewed(s, sign=1, sr=0.7):
    rng = np.random.default_rng(60_000 + s)
    base = rng.normal(0, 0.006, T)
    jumps = (rng.random(T) < 0.02) * rng.exponential(0.04, T)
    x = sign * (jumps - jumps.mean()) + base
    return x - x.mean() + sr * x.std() / np.sqrt(252)


if __name__ == "__main__":
    run("noise, unselected, n_trials=1 (300 runs)", lambda s: np.random.default_rng(s).normal(0, 0.01, T), 300,
        n_trials=1, turnover=0.05)
    run("best of 100 noise, n_trials=100 (200 runs)", best, 200, n_trials=100, turnover=0.05)
    run("best of 100 noise, n_trials=1 (understated) (200 runs)", best, 200, n_trials=1, turnover=0.05)
    run("noise, 20-day overlapping returns, n_trials=10 (200 runs)", overlap, 200, n_trials=10, turnover=0.05)
    run("noise, AR(1) phi=0.3, n_trials=10 (200 runs)", ar1, 200, n_trials=10, turnover=0.05)
    for sr in (0.5, 1.0, 1.5):
        mu = sr * 0.01 / np.sqrt(252)
        run(f"planted SR {sr}, 5y, n_trials=10 (200 runs)",
            lambda s, mu=mu: np.random.default_rng(20_000 + s).normal(mu, 0.01, T), 200, n_trials=10, turnover=0.05)
    run("planted SR 0.7, positive skew, n_trials=1 (200 runs)", lambda s: skewed(s, 1), 200, n_trials=1, turnover=0.05)
    run("2x leveraged index (Sharpe ~0.75), NO benchmark (200 runs)",
        lambda s: 2 * market(s, 2520) + 0.12 / 252 - 0.0003 * 2, 200, with_benchmark=False, n_trials=1, turnover=0.0)
    run("2x leveraged index, benchmark = the index (200 runs)",
        lambda s: 2 * market(s, 2520) + 0.12 / 252 - 0.0003 * 2, 200, n_trials=1, turnover=0.0)
    run("planted SR 0.7, negative skew, n_trials=1 (200 runs)", lambda s: skewed(s, -1), 200, n_trials=1, turnover=0.05)
