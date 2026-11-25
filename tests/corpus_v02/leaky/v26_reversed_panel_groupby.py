# expect: LA02:high
import pandas as pd
panel = pd.read_csv("panel.csv", parse_dates=["date"])
panel = panel.sort_values(["ticker", "date"], ascending=[True, False])
panel["vol20"] = panel.groupby("ticker")["close"].transform(lambda s: s.pct_change().rolling(20).std())
panel["signal"] = (panel["vol20"] < 0.01).astype(int)
