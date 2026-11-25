# expect: LA02:high@14
from dataclasses import dataclass

import pandas as pd


@dataclass
class Cfg:
    window: int = 21
    center: bool = True


def detrend(s, cfg=Cfg()):
    return s / s.rolling(cfg.window, center=cfg.center).mean() - 1


px = pd.read_parquet("copper_1h.parquet")["close"]
signal = -detrend(px).clip(-0.01, 0.01) * 100
pnl = signal.shift(1) * px.pct_change()
print(pnl.cumsum().iloc[-1])
