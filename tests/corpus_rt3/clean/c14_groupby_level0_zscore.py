# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
panel = pd.read_parquet("panel.parquet").set_index(["date", "ticker"]).sort_index()
panel["value"] = panel["book"] / panel["mktcap"]
g = panel.groupby(level=0)["value"]
panel["signal"] = (panel["value"] - g.transform("mean")) / g.transform("std")
panel["position"] = panel.groupby(level=1)["signal"].shift(1)
