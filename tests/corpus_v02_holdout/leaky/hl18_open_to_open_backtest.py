# expect: LA07
# Position decided on today's close earns the return from today's open to the next open.
import numpy as np
import pandas as pd


def backtest(df: pd.DataFrame, lookback: int = 5, fee: float = 0.0005) -> pd.Series:
    mom = df["close"] / df["close"].shift(lookback) - 1
    sig = np.sign(mom).fillna(0)
    oo = df["open"].shift(-1) / df["open"] - 1
    net = sig * oo - fee * sig.diff().abs().fillna(0)
    return net.dropna()


if __name__ == "__main__":
    data = pd.read_csv("dia.csv", index_col="date", parse_dates=True)
    r = backtest(data)
    print("Sharpe", r.mean() / r.std() * np.sqrt(252))
