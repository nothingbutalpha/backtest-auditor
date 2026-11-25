# expect: LA02
# Feature builder sorts by timestamp descending, so shift(1) and rolling pull the following bars.
import pandas as pd


def build_features(df):
    df = df.sort_values("timestamp", ascending=False).copy()
    df["prev_close"] = df["close"].shift(1)
    df["gap"] = df["open"] / df["prev_close"] - 1
    df["hi_5"] = df["high"].shift(1).rolling(5).max()
    df["breakout"] = df["close"] > df["hi_5"]
    return df.sort_values("timestamp")


def backtest(df, fee=0.0005):
    f = build_features(df)
    sig = ((f["gap"] > 0.01) & f["breakout"]).astype(float)
    pos = sig.shift(1).fillna(0)
    ret = f["close"].pct_change().fillna(0)
    return (pos * ret - fee * pos.diff().abs().fillna(0)).sum()
