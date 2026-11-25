import pandas as pd
panel = pd.read_csv("panel_pit.csv", parse_dates=["date"])
panel["ret"] = panel.groupby("ticker")["close"].pct_change()
panel["mom"] = panel.groupby("ticker")["close"].pct_change(20)
panel["z"] = panel.groupby("date")["mom"].transform(lambda s: (s - s.mean()) / s.std())
panel["signal"] = (panel["z"] > 1).astype(int)
panel["held"] = panel.groupby("ticker")["signal"].shift(1)
panel["strat"] = panel["held"] * panel["ret"] - panel.groupby("ticker")["held"].diff().abs() * 0.001
