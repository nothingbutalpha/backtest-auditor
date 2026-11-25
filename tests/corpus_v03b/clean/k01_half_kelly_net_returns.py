# expect: none
# CO01 class: sizing report on a return series that already includes costs.
import numpy as np
import pandas as pd

net = pd.read_csv("strategy_net_returns.csv", index_col=0, parse_dates=True)["net_ret"]
mu, var = net.mean(), net.var()
kelly = mu / var
half_kelly = 0.5 * kelly
sized = half_kelly * net
sharpe = sized.mean() / sized.std() * np.sqrt(252)
cagr = (1 + sized).prod() ** (252 / len(sized)) - 1
print(f"half-Kelly {half_kelly:.2f}, Sharpe {sharpe:.2f}, CAGR {cagr:.1%}")
