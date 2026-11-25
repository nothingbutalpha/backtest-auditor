"""A momentum strategy as a chat assistant might write it when asked for
"a z-score momentum strategy with a parameter sweep". It contains four common backtest bugs.
Don't fix them here: walkthrough.py audits this file as-is.

The prices are synthetic: a random walk with no predictable component, so any edge this
backtest reports is produced by the bugs, not by the market.
"""
import numpy as np
import pandas as pd


def make_prices(seed: int = 7, n_days: int = 1260) -> pd.DataFrame:
    """Five years of synthetic daily closes for one made-up asset (no vendor data)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days, name="date")
    log_rets = rng.normal(0.0003, 0.012, n_days)
    return pd.DataFrame({"close": 100 * np.exp(np.cumsum(log_rets))}, index=dates)


def backtest(df: pd.DataFrame, lookback: int, entry_z: float) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["mom"] = df["close"].pct_change(lookback)
    # Normalize momentum to a z-score.
    df["z"] = (df["mom"] - df["mom"].mean()) / df["mom"].std()
    df["signal"] = np.where(df["z"] > entry_z, 1, np.where(df["z"] < -entry_z, -1, 0))
    df["strategy"] = df["signal"] * df["ret"]
    return df


def run() -> dict:
    prices = make_prices()
    results = []
    for lookback in [5, 10, 20, 40, 60]:
        for entry_z in [0.5, 1.0, 1.5]:
            bt = backtest(prices, lookback, entry_z)
            sharpe = bt["strategy"].mean() / bt["strategy"].std() * np.sqrt(252)
            results.append((sharpe, lookback, entry_z, bt))
    sharpe, lookback, entry_z, best = max(results, key=lambda r: r[0])
    print(f"Best: lookback={lookback}, entry_z={entry_z}, Sharpe={sharpe:.2f}")
    return {
        "returns": best["strategy"].dropna(),
        "turnover": best["signal"].diff().abs().fillna(0).mean(),
        "benchmark": best["ret"].dropna(),
        "n_trials": len(results),
    }


if __name__ == "__main__":
    run()
