"""The same momentum idea with the four bugs in leaky_strategy.py fixed.

1. Same-bar signal and fill (LA07): the position is decided at today's close and held over the
   *next* bar, so it earns tomorrow's return: `position = signal.shift(1)`.
2. Full-sample z-score (LA09): normalize with a trailing window that only uses past data.
3. Parameter sweep scored on the full sample (LA11): we don't sweep again here. One setting,
   fixed in advance, and we report the sweep we already ran (15 variants) as `n_trials`.
4. No costs (CO01): charge 10 bps per unit of one-way turnover.

Same synthetic random-walk prices as the leaky version, so there is no real edge to find.
"""
import numpy as np
import pandas as pd

from leaky_strategy import make_prices

LOOKBACK = 20
ENTRY_Z = 1.0
Z_WINDOW = 252
COST_BPS = 10.0
N_TRIALS = 15  # every variant tried so far, including the sweep in leaky_strategy.py


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["mom"] = df["close"].pct_change(LOOKBACK)
    roll = df["mom"].rolling(Z_WINDOW, min_periods=Z_WINDOW)
    df["z"] = (df["mom"] - roll.mean()) / roll.std()
    df["signal"] = np.where(df["z"] > ENTRY_Z, 1, np.where(df["z"] < -ENTRY_Z, -1, 0))
    df["position"] = df["signal"].shift(1).fillna(0)
    df["turnover"] = df["position"].diff().abs().fillna(0)
    df["cost"] = df["turnover"] * COST_BPS / 10_000
    df["gross"] = df["position"] * df["ret"]
    df["strategy"] = df["gross"] - df["cost"]
    return df.iloc[Z_WINDOW + LOOKBACK:]


def run() -> dict:
    bt = backtest(make_prices())
    sharpe = bt["strategy"].mean() / bt["strategy"].std() * np.sqrt(252)
    print(f"lookback={LOOKBACK}, entry_z={ENTRY_Z}, net Sharpe={sharpe:.2f}")
    # Hand the auditor gross returns plus turnover; it charges the cost (costs_bps) and shows a cost grid.
    return {
        "returns": bt["gross"],
        "turnover": bt["turnover"],
        "benchmark": bt["ret"],
        "n_trials": N_TRIALS,
        "costs_bps": COST_BPS,
    }


if __name__ == "__main__":
    run()
