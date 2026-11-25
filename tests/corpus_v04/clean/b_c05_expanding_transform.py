# expect: none
import pandas as pd

df = pd.read_csv("gc_futures.csv", index_col=0, parse_dates=True)
df["trend"] = df["close"].pct_change(60)
df["carry"] = df["basis"].rolling(5).mean()
scaled = df[["trend", "carry"]].transform(lambda s: (s - s.expanding(250).mean()) / s.expanding(250).std())
df["position"] = (0.5 * scaled["trend"] + 0.5 * scaled["carry"]).clip(-1, 1).shift(1)
df["pnl"] = df["position"] * df["close"].pct_change() - 0.0002 * df["position"].diff().abs()
print(df["pnl"].sum())
