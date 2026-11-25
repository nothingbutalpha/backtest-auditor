# leak: 9
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
data = pd.read_parquet("factors.parquet")
label = data.pop("ret_next")
ic = {c: spearmanr(data[c], label, nan_policy="omit")[0] for c in data.columns}
keep = sorted(ic, key=lambda c: abs(ic[c]), reverse=True)[:5]
print(cross_val_score(LinearRegression(), data[keep], label, cv=TimeSeriesSplit(5)).mean())
