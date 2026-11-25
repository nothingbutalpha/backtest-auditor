# expect: none
# Class 2 (added after the repo rescan found this pattern in our internal research code): a long-format
# frame that stores the forward return as its own column for an IC computation. "long" means long format here.
import pandas as pd

closes = pd.read_parquet("closes.parquet")
score = closes.pct_change(20)
fwd = closes.shift(-5) / closes - 1
ew = fwd.mean(axis=1)
excess = fwd.sub(ew, axis=0)
long = pd.DataFrame({"score": score.stack(), "excess_ret": excess.stack()}).dropna().reset_index()
long.columns = ["date", "ticker", "score", "excess_ret"]
ic = long.groupby("date").apply(lambda d: d["score"].corr(d["excess_ret"], method="spearman"))
print(ic.mean())
