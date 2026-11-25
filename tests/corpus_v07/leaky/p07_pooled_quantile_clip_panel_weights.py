# expect: LA09@5
import pandas as pd

panel = pd.read_parquet("quality_panel.parquet")
lo, hi = panel["gross_prof"].quantile([0.005, 0.995])
panel["gp"] = panel["gross_prof"].clip(lo, hi)
panel["w"] = panel.groupby("date")["gp"].transform(lambda s: (s - s.mean()) / s.abs().sum())
ret = (panel["w"] * panel["ret_fwd_1m"]).groupby(panel["date"]).sum()
print(ret.mean() * 12)
