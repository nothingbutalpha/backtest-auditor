# expect: none
# Class 3: panel version, next-open fills per ticker.
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
g = df.groupby("ticker")
df["mom"] = g["close"].pct_change(20)
df["entry_px"] = g["open"].shift(-1)
df["exit_px"] = g["open"].shift(-2)
df["oo_ret"] = df["exit_px"] / df["entry_px"] - 1
df["position"] = (df.groupby("date")["mom"].rank(pct=True) > 0.9).astype(float)
df["pnl"] = df["position"] * df["oo_ret"] - 0.001 * df.groupby("ticker")["position"].diff().abs()
daily = df.groupby("date")["pnl"].mean()
print(daily.mean() * 252)
