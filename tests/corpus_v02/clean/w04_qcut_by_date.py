# expect: none
import pandas as pd
panel = pd.read_csv("panel.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
panel["mom"] = panel.groupby("ticker")["close"].pct_change(60)
panel["quintile"] = panel.groupby("date")["mom"].transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
panel["position"] = (panel["quintile"] == 4).astype(int)
panel["next_ret"] = panel.groupby("ticker")["close"].pct_change().shift(-1)
panel["strategy"] = panel["position"] * panel["next_ret"] - 0.001 * panel.groupby("ticker")["position"].diff().abs()
