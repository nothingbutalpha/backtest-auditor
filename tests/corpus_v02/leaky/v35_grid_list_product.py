# expect: LA11:medium
import itertools
import numpy as np
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
grid = list(itertools.product([5, 10, 20], [50, 100, 200]))
rows = []
for fast, slow in grid:
    pos = (df["close"].rolling(fast).mean() > df["close"].rolling(slow).mean()).astype(int).shift(1)
    r = pos * df["close"].pct_change() - 0.0005 * pos.diff().abs()
    rows.append({"fast": fast, "slow": slow, "sharpe": r.mean() / r.std() * np.sqrt(252)})
table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
print(table.head(1))
