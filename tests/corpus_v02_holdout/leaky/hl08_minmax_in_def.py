# expect: LA09
# Feature scaling inside make_features divides by full-sample max/min of the df passed in.
import pandas as pd


def make_features(df):
    out = pd.DataFrame(index=df.index)
    out["vol_n"] = df["volume"] / df["volume"].max()
    rng = df["close"].rolling(14)
    out["stoch"] = (df["close"] - rng.min()) / (rng.max() - rng.min())
    atr = (df["high"] - df["low"]).rolling(14).mean()
    out["atr_n"] = (atr - atr.min()) / (atr.max() - atr.min())
    return out


def backtest(df, fee=0.0005):
    f = make_features(df)
    sig = ((f["vol_n"] > 0.3) & (f["stoch"] < 0.2) & (f["atr_n"] < 0.5)).astype(float)
    pos = sig.shift(1).fillna(0)
    ret = df["close"].pct_change().fillna(0)
    return (pos * ret - fee * pos.diff().abs().fillna(0)).cumsum()
