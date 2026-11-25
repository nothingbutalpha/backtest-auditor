# expect: none
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Cfg:
    offset_bps: float = 50.0
    hold_bars: int = 5


def simulate(bars, signal_idx, cfg):
    o, lo, c = bars["open"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy()
    trades = []
    for i in signal_idx:
        if i + 1 + cfg.hold_bars >= len(c):
            break
        limit = c[i] * (1 - cfg.offset_bps / 1e4)
        if lo[i + 1] <= limit:
            fill = min(limit, o[i + 1])
            exit_px = c[i + 1 + cfg.hold_bars]
            trades.append((bars.index[i + 1], fill, exit_px / fill - 1))
    return pd.DataFrame(trades, columns=["date", "fill", "ret"])


bars = pd.read_parquet("spy.parquet")
sma = bars["close"].rolling(20).mean()
idx = np.flatnonzero((bars["close"] < sma).to_numpy())
res = simulate(bars, idx, Cfg())
print(res["ret"].mean(), len(res))
