# expect: LA01:high@7
import pandas as pd
from sklearn.linear_model import LogisticRegression


def ahead(s, k):
    return s.shift(-k)


df = pd.read_parquet("btc_1h.parquet")
r = df["close"].pct_change()
X = pd.DataFrame({"r1": ahead(r, 1), "r3": ahead(r, 3)})
y = (r.shift(-1) > 0).astype(int)
data = X.join(y.rename("y")).dropna()
print(LogisticRegression().fit(data[["r1", "r3"]], data["y"]).coef_)
