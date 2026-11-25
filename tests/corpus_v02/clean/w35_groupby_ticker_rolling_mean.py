# expect: none
import pandas as pd
panel = pd.read_csv("panel.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
panel["ret"] = panel.groupby("ticker")["close"].pct_change()
panel["avg60"] = panel.groupby("ticker")["ret"].transform(lambda s: s.rolling(60).mean())
panel["signal"] = ((panel["ret"] - panel["avg60"]) < -0.02).astype(int)
