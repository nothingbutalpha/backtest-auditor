# expect: LA09:medium
import pandas as pd


def make_signals(df, n_bins=10):
    df["vol"] = df["close"].pct_change().rolling(21).std()
    df["vol_bin"] = pd.qcut(df["vol"], n_bins, labels=False)
    df["signal"] = (df["vol_bin"] <= 2).astype(int)
    return df
