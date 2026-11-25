# expect: none
# Looks like a zip(c[:-1], c[1:]) leak, but each pair's signal is only applied to the following move.
import pandas as pd

prices = pd.read_csv("btc_hourly.csv")["close"].tolist()

moves = [b / a - 1 for a, b in zip(prices[:-1], prices[1:])]
signals = [1 if b > a else -1 for a, b in zip(prices[:-1], prices[1:])]

fee = 0.001
equity = 1.0
prev = 0
for sig, mv in zip(signals[:-1], moves[1:]):
    equity *= 1 + sig * mv - fee * abs(sig - prev)
    prev = sig

print("final equity", round(equity, 4))
