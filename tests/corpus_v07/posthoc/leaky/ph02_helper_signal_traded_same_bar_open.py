# expect: LA07@15
import pandas as pd

COST = 0.0005


def trend_on(frame):
    return (frame["close"].ewm(span=30).mean() > frame["close"].ewm(span=120).mean()).astype(float)


px = pd.read_csv("gc_daily.csv", parse_dates=["date"], index_col="date")
px["hold"] = trend_on(px)
px["open_ret"] = px["open"].shift(-1) / px["open"] - 1
px["turn"] = px["hold"].diff().abs().fillna(0)
px["gross"] = px["hold"] * px["open_ret"]
px["net"] = px["gross"] - px["turn"] * COST
print((1 + px["net"].dropna()).prod())
