# expect: LA07@9
import pandas as pd

bars = pd.read_csv("sol_15m.csv", index_col=0, parse_dates=True)
bars["band"] = bars["close"].rolling(20).std()
entries = []
for ts, row in bars.dropna().iterrows():
    bid = row["close"] - row["band"]
    if row["low"] <= bid:
        entries.append((ts, bid))
fills = pd.DataFrame(entries, columns=["ts", "px"]).set_index("ts")
fills["out"] = bars["close"].shift(-4).reindex(fills.index)
print((fills["out"] / fills["px"] - 1).mean())
