# expect: LA07@11
import pandas as pd
import vectorbt as vbt

bars = pd.read_csv("qqq_daily.csv", parse_dates=["date"], index_col="date")
ema_fast = bars["close"].ewm(span=12).mean()
ema_slow = bars["close"].ewm(span=48).mean()
cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
cross_dn = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
pf = vbt.Portfolio.from_signals(
    bars["close"],
    entries=cross_up,
    exits=cross_dn,
    price=bars["open"],
    fees=0.0005,
)
print(pf.total_return())
