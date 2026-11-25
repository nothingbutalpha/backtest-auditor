# expect: LA09:medium
import pandas as pd


def backtest(df, q=0.8, cost=0.0005):
    df = df.copy()
    df["rsi_like"] = df["close"].diff().rolling(14).mean()
    upper = df["rsi_like"].quantile(q)
    df["signal"] = (df["rsi_like"] > upper).astype(int)
    df["strategy"] = df["signal"].shift(1) * df["close"].pct_change() - cost * df["signal"].diff().abs()
    return df
