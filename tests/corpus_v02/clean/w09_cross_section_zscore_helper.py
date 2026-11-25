# expect: none
import pandas as pd


def zscore(s):
    return (s - s.mean()) / s.std()


panel = pd.read_csv("panel.csv", parse_dates=["date"])
panel["mom"] = panel.groupby("ticker")["close"].pct_change(20)
panel["mom_z"] = panel.groupby("date")["mom"].transform(zscore)
panel["signal"] = (panel["mom_z"] > 1).astype(int)
