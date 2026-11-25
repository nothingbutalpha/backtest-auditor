# expect: none
import pandas as pd


def f(df):
    df["mom"] = df["close"].pct_change(20)
    exp = df["mom"].expanding(min_periods=252)
    df["z"] = (df["mom"] - exp.mean().shift(1)) / exp.std().shift(1)
    df["signal"] = (df["z"] > 1).astype(int)
    return df
