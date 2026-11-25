# expect: LA09:medium
import pandas as pd


def f(df):
    df["mom"] = df["close"].pct_change(20)
    df["z"] = (df["mom"] - df["mom"].mean()) / df["mom"].std()
    df["signal"] = (df["z"] > 1).astype(int)
    return df
