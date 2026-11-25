# expect: none
import pandas as pd

bars = pd.read_csv("nvda_daily.csv", index_col="date", parse_dates=True)
bars["range_pct"] = (bars["high"] - bars["low"]) / bars["close"]
bars["atr"] = bars["range_pct"].rolling(14).mean()
bars["stop_px"] = bars["close"] - 2 * bars["atr"] * bars["close"]
print(bars[["atr", "stop_px"]].tail())
