# expect: none
# Looks like a groupby('ticker').transform full-history z-score, but uses trailing rolling stats per ticker.
import pandas as pd

panel = pd.read_csv("panel.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
panel["ret"] = panel.groupby("ticker")["close"].pct_change()
panel["turnover"] = panel["volume"] * panel["close"]
g = panel.groupby("ticker")["turnover"]
mu = g.transform(lambda x: x.rolling(60).mean().shift(1))
sd = g.transform(lambda x: x.rolling(60).std().shift(1))
panel["turn_z"] = (panel["turnover"] - mu) / sd

panel["signal"] = (panel["turn_z"] > 1.5).astype(float)
panel["pos"] = panel.groupby("ticker")["signal"].shift(1).fillna(0)
turn = panel.groupby("ticker")["pos"].diff().abs().fillna(0)
panel["net"] = panel["pos"] * panel["ret"] - 0.001 * turn
daily = panel.groupby("date")["net"].mean()
print("Sharpe:", daily.mean() / daily.std() * 252 ** 0.5)
