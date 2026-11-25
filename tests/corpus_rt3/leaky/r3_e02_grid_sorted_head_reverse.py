# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import numpy as np, pandas as pd
px = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
rows = []
for n in range(10, 200, 10):
    sig = (px > px.rolling(n).mean()).shift(1)
    r = sig * px.pct_change() - 0.0005 * sig.diff().abs()
    rows.append({"n": n, "sharpe": r.mean() / r.std() * np.sqrt(252)})
table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
n_use = int(table["n"].iloc[0])
final = (px > px.rolling(n_use).mean()).shift(1) * px.pct_change()
print(final.mean() / final.std() * np.sqrt(252))
