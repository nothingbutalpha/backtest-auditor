# expect: none
import pandas as pd


def strategy_returns(signal, returns, cost=0.0005):
    pos = signal.shift(1)
    return pos * returns - cost * pos.diff().abs()


df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["ret"] = df["close"].pct_change()
df["signal"] = (df["close"] > df["close"].rolling(100).mean()).astype(int)
df["strategy"] = strategy_returns(df["signal"], df["ret"])
