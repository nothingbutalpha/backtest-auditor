# expect: none
import pandas as pd


def backtest(df, q=0.8, cost=0.0005):
    df = df.copy()
    df["osc"] = df["close"].diff().rolling(14).mean()
    df["upper"] = df["osc"].rolling(252).quantile(q).shift(1)
    df["signal"] = (df["osc"] > df["upper"]).astype(int)
    df["strategy"] = df["signal"].shift(1) * df["close"].pct_change() - cost * df["signal"].diff().abs()
    return df
