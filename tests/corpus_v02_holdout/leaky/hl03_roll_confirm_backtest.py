# expect: LA01
# Entry confirmation uses np.roll(rets, -1), the next bar's return, inside a backtest function.
import numpy as np


def backtest(close, lookback=20, fee=0.0005):
    close = np.asarray(close, dtype=float)
    rets = np.empty_like(close)
    rets[0] = 0.0
    rets[1:] = np.diff(close) / close[:-1]
    mom = np.array([close[i] / close[max(i - lookback, 0)] - 1 for i in range(len(close))])
    confirm = np.roll(rets, -1) > 0
    pos = np.where((mom > 0) & confirm, 1.0, 0.0)
    pos = np.concatenate([[0.0], pos[:-1]])
    pnl = pos * rets - fee * np.abs(np.diff(pos, prepend=0.0))
    return pnl.sum(), pnl.mean() / pnl.std() * np.sqrt(252)
