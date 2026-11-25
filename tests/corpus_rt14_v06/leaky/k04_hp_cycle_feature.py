# leak: 7
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
macro = pd.read_csv("macro.csv", index_col=0, parse_dates=True)
y = macro["spx"].pct_change().shift(-1)
cycle, trend = sm.tsa.filters.hpfilter(macro["indpro"], lamb=129600)
X = pd.DataFrame({"gap": cycle, "mom": macro["spx"].pct_change(12)}).dropna()
train = X.index < "2015"
model = Ridge().fit(X[train], y.reindex(X.index)[train].fillna(0))
print(model.score(X[~train], y.reindex(X.index)[~train].fillna(0)))
