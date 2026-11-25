# expect: LA09:medium
import pandas as pd


def standardize(s):
    return (s - s.mean()) / s.std()


df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["mom"] = df["close"].pct_change(60)
df["mom_z"] = standardize(df["mom"])
df["signal"] = (df["mom_z"] > 0.5).astype(int)
df["pnl"] = df["signal"].shift(1) * df["close"].pct_change() - 0.0005 * df["signal"].diff().abs()
