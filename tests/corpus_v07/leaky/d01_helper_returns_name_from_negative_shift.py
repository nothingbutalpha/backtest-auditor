# expect: LA01|LA10@5
import pandas as pd

def smooth_momentum(px, n=10):
    lead = px.shift(-2)
    sm = lead.rolling(n).mean()
    return sm / px - 1

px = pd.read_csv("eurjpy.csv", index_col=0, parse_dates=True)["close"]
signal = (smooth_momentum(px) > 0).astype(int)
pnl = signal.shift(1) * px.pct_change()
print(pnl.sum())
