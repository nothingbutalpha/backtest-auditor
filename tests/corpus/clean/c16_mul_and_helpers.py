# expect: none
import pandas as pd


def run_strategy(signal: pd.Series, returns: pd.Series, cost: float = 0.0005) -> pd.Series:
    position = signal.shift(1).fillna(0)
    return position * returns - position.diff().abs() * cost


df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["ret"] = df["close"].pct_change()
df["signal"] = (df["close"] > df["close"].rolling(100).mean()).astype(int)
df["position"] = df["signal"].shift(1).fillna(0)
commission = 0.0005
df["strat"] = df["position"].mul(df["ret"]) - df["position"].diff().abs() * commission
alt = run_strategy(df["signal"], df["ret"])
