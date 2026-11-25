# expect: SB01@5
import pandas as pd

wide = pd.read_parquet("asx_close.parquet")
wide = wide.loc["2006":"2022"].dropna(axis=1)
month = wide.resample("ME").last()
mom = month.shift(1) / month.shift(12) - 1
nxt = month.pct_change().shift(-1)
top = mom.rank(axis=1, pct=True) > 0.8
print(nxt[top].mean(axis=1).mean() * 12)
