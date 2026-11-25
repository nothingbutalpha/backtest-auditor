# expect: none
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_parquet("fx_daily_features.parquet")
df["fwd_ret_3d"] = df["close"].shift(-3) / df["close"] - 1
df["y"] = (df["fwd_ret_3d"] > 0).astype(int)
df = df.dropna()
COLS = ["carry", "mom_60", "rv_20"]
n = int(len(df) * 0.75)
gb = GradientBoostingClassifier().fit(df[COLS].iloc[:n], df["y"].iloc[:n])
print(gb.score(df[COLS].iloc[n + 3:], df["y"].iloc[n + 3:]))
