# expect: LA02:high
import pandas as pd


def build(df):
    df.sort_values("date", ascending=False, inplace=True)
    df["ma50"] = df["close"].rolling(50).mean()
    df["trend"] = (df["close"] > df["ma50"]).astype(int)
    return df
