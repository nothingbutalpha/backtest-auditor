# expect: none
import pandas as pd


def score_cross_section(day):
    day = day.copy()
    day["z"] = (day["mom"] - day["mom"].mean()) / day["mom"].std()
    day["long"] = (day["z"] > 1).astype(int)
    return day


panel = pd.read_csv("panel.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
panel["mom"] = panel.groupby("ticker")["close"].pct_change(20)
frames = []
for date, day in panel.groupby("date"):
    frames.append(score_cross_section(day))
scored = pd.concat(frames)
