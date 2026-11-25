# expect: none
# Class 2: IC computed in a loop with scipy, forward returns only on the right-hand side of the metric.
import pandas as pd
from scipy.stats import spearmanr

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
g = df.groupby("ticker")["close"]
df["signal"] = g.pct_change(20)
df["fwd_ret"] = g.shift(-20) / df["close"] - 1
rows = []
for d, day in df.dropna().groupby("date"):
    rho, p = spearmanr(day["signal"], day["fwd_ret"])
    rows.append((d, rho))
ic = pd.DataFrame(rows, columns=["date", "ic"]).set_index("date")["ic"]
print(ic.mean(), (ic > 0).mean())
