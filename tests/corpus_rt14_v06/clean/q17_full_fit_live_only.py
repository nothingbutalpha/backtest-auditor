# expect: none
import pandas as pd
import statsmodels.api as sm
a = pd.read_csv("ko.csv", index_col=0)["close"]
b = pd.read_csv("pep.csv", index_col=0)["close"]
beta = sm.OLS(a, sm.add_constant(b)).fit().params.iloc[1]
spread_today = a.iloc[-1] - beta * b.iloc[-1]
print("hedge ratio", round(beta, 3), "spread now", round(spread_today, 2))
