# expect: none
# Looks like a full-sample rank, but ranks are cross-sectional per date via groupby('date').
import pandas as pd

panel = pd.read_csv("panel.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
panel["ret"] = panel.groupby("ticker")["close"].pct_change()
panel["mom"] = panel.groupby("ticker")["close"].pct_change(126)
panel["mom_rank"] = panel.groupby("date")["mom"].rank(pct=True)

panel["long"] = (panel["mom_rank"] > 0.9).astype(float)
panel["pos"] = panel.groupby("ticker")["long"].shift(1).fillna(0)
turn = panel.groupby("ticker")["pos"].diff().abs().fillna(0)
panel["net"] = panel["pos"] * panel["ret"] - 0.001 * turn
daily = panel[panel["pos"] > 0].groupby("date")["net"].mean()
print("Sharpe:", daily.mean() / daily.std() * 252 ** 0.5)
