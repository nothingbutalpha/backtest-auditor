# expect: none
# Looks like full-sample normalisation inside a def, but uses expanding mean/std lagged by one bar.
import numpy as np
import pandas as pd


def make_features(df):
    out = pd.DataFrame(index=df.index)
    vol = df["volume"]
    out["vol_z"] = (vol - vol.expanding(60).mean().shift(1)) / vol.expanding(60).std().shift(1)
    ret = df["close"].pct_change()
    out["ret_z"] = (ret - ret.expanding(60).mean().shift(1)) / ret.expanding(60).std().shift(1)
    return out


def backtest(df, fee=0.0005):
    f = make_features(df)
    sig = ((f["vol_z"] > 2) & (f["ret_z"] < -2)).astype(float)
    pos = sig.shift(1).fillna(0)
    net = pos * df["close"].pct_change().fillna(0) - fee * pos.diff().abs().fillna(0)
    return net.mean() / net.std() * np.sqrt(252)
