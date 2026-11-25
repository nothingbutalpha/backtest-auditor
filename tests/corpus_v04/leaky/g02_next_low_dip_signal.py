# expect: LA01|LA10:high
import pandas as pd

df = pd.read_csv("spy_daily.csv", index_col="date", parse_dates=True)
dip = df["low"].shift(-1) / df["close"] - 1
signal = (dip < -0.02).astype(int)
strat = signal * df["close"].pct_change().shift(-1) - 0.0005 * signal.diff().abs()
print(strat.mean() * 252)
