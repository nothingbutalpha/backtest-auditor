import pandas as pd
panel = pd.read_csv("panel_pit.csv", parse_dates=["date"])
panel["ret"] = panel.groupby("ticker")["close"].pct_change()
panel["z"] = panel.groupby("ticker")["ret"].transform(lambda s: (s - s.rolling(60).mean()) / s.rolling(60).std())
panel["signal"] = (panel["z"] < -2).astype(int)
panel["held"] = panel.groupby("ticker")["signal"].shift(1)
panel["strat"] = panel["held"] * panel["ret"] - 0.001 * panel.groupby("ticker")["held"].diff().abs()
