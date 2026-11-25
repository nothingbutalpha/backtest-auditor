# expect: LA01|LA10:high@12
import pandas as pd
from sklearn.linear_model import Ridge


def lag(s, k):
    return s.shift(k)


df = pd.read_csv("hyg.csv", index_col=0, parse_dates=True)
r = df["close"].pct_change()
X = pd.DataFrame({"r1": lag(r, 1), "r_next": lag(r, -2)})
y = r.shift(-1)
data = X.join(y.rename("y")).dropna()
print(Ridge().fit(data[["r1", "r_next"]], data["y"]).coef_)
