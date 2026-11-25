# expect: SB01:medium@6
import pandas as pd

LAST = pd.Timestamp("2024-06-28")
panel = pd.read_parquet("jp_equities.parquet")
panel = panel.groupby("code").filter(lambda g: g["date"].max() >= LAST)
panel = panel.sort_values(["code", "date"])
panel["mom"] = panel.groupby("code")["close"].pct_change(120).shift(1)
print(panel.groupby("date")["mom"].mean().tail())
