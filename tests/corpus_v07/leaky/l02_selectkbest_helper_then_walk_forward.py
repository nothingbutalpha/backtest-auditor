# expect: LA05|LA11@7
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge

def top_k(X, y, k=15):
    sel = SelectKBest(f_regression, k=k).fit(X, y)
    return X.columns[sel.get_support()]

df = pd.read_parquet("signals_400.parquet").sort_index()
y = df.pop("fwd_ret_10")
keep = top_k(df, y)
preds = []
for t in range(1000, len(df), 50):
    m = Ridge().fit(df[keep].iloc[:t - 10], y.iloc[:t - 10])
    preds.append(pd.Series(m.predict(df[keep].iloc[t:t + 50]), index=df.index[t:t + 50]))
print(pd.concat(preds).corr(y))
