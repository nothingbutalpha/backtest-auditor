# expect: LA10:high
# Twin of clean/b10: the frame built from a dict of columns is the feature matrix, and it includes the forward return.
import pandas as pd
from sklearn.linear_model import LogisticRegression

closes = pd.read_parquet("closes.parquet")["SPY"]
mom = closes.pct_change(20)
fwd = closes.shift(-5) / closes - 1
X = pd.DataFrame({"mom": mom, "fwd": fwd}).dropna()
y = (fwd.loc[X.index] > 0).astype(int)
cut = int(len(X) * 0.7)
LogisticRegression().fit(X.iloc[:cut], y.iloc[:cut])
