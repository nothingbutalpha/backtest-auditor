# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np, pandas as pd
px = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
rows = []
for n in range(10, 200, 10):
    sig = (px > px.rolling(n).mean()).shift(1)
    r = sig * px.pct_change() - 0.0005 * sig.diff().abs()
    rows.append((n, r.mean() / r.std() * np.sqrt(252)))
table = pd.DataFrame(rows, columns=["n", "sharpe"])
print(table.to_string())  # robustness surface: every variant shown, none selected
