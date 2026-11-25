# expect: CO01:possible
# Cross-sectional momentum backtest returns total return and Sharpe without any trading costs.
import numpy as np
import pandas as pd


def backtest(prices: pd.DataFrame, lookback=252, skip=21, top=10):
    rets = prices.pct_change()
    mom = prices.shift(skip) / prices.shift(lookback) - 1
    rank = mom.rank(axis=1, ascending=False)
    w = (rank <= top).astype(float)
    w = w.div(w.sum(axis=1), axis=0).shift(1).fillna(0)
    port = (w * rets).sum(axis=1)
    total = (1 + port).prod() - 1
    sharpe = port.mean() / port.std() * np.sqrt(252)
    return {"total_return": total, "sharpe": sharpe}


if __name__ == "__main__":
    px = pd.read_csv("sp500_closes.csv", index_col=0, parse_dates=True)
    print(backtest(px))
