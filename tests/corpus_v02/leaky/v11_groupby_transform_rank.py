# expect: LA09:medium
import pandas as pd
panel = pd.read_csv("panel.csv", parse_dates=["date"])
panel["mom"] = panel.groupby("ticker")["close"].pct_change(20)
panel["mom_pct"] = panel.groupby("ticker")["mom"].transform(lambda s: s.rank(pct=True))
panel["signal"] = (panel["mom_pct"] > 0.8).astype(int)
