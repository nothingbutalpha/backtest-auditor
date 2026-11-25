# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("panel.parquet")
last = df.groupby("ticker")["date"].max()
end = df["date"].max()
alive = (last == end)
print(f"{(~alive).sum()} of {len(alive)} names delist before {end:%Y-%m-%d}; all kept in the backtest")
assert (~alive).sum() > 0, "panel looks survivor-only"
