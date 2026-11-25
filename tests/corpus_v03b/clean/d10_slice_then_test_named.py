# expect: none
# LA05 class (added after the first measurement): fit on a date slice, then a variable named `test` for the later rows.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("spy_features.csv", index_col="date", parse_dates=True)
X = df[["mom_20", "vol_20"]]
y = df["up_next"]
cut = "2019-01-14"
scaler = StandardScaler().fit(X.loc[:cut])
model = LogisticRegression().fit(scaler.transform(X.loc[:cut]), y.loc[:cut])
X_test = scaler.transform(X.loc["2019-01-15":])
y_test = y.loc["2019-01-15":]
print(model.score(X_test, y_test))
