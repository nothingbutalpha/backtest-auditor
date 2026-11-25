# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
panel = pd.read_parquet("panel.parquet").set_index(["date", "ticker"]).sort_index()
panel["mom"] = panel.groupby(level="ticker")["close"].pct_change(126)
panel["signal"] = panel.groupby(level="date")["mom"].rank(pct=True)
panel["position"] = panel.groupby(level="ticker")["signal"].shift(1)
