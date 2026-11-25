# expect: LA09
# Per-ticker z-score via groupby('ticker').transform uses each stock's full-history mean and std.
import pandas as pd

panel = pd.read_csv("panel.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
panel["ret"] = panel.groupby("ticker")["close"].pct_change()
panel["turnover"] = panel["volume"] * panel["close"]
panel["turn_z"] = panel.groupby("ticker")["turnover"].transform(lambda x: (x - x.mean()) / x.std())

panel["signal"] = (panel["turn_z"] > 1.5).astype(float)
panel["pos"] = panel.groupby("ticker")["signal"].shift(1).fillna(0)
turn = panel.groupby("ticker")["pos"].diff().abs().fillna(0)
panel["net"] = panel["pos"] * panel["ret"] - 0.001 * turn
daily = panel.groupby("date")["net"].mean()
print("Sharpe:", daily.mean() / daily.std() * 252 ** 0.5)
