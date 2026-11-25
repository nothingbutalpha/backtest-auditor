# expect: LA05:high
# Twin of clean/d10: the scaler is fitted on all rows before the test rows are split off.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("spy_features.csv", index_col="date", parse_dates=True)
X = df[["mom_20", "vol_20"]]
y = df["up_next"]
scaler = StandardScaler().fit(X)
Xs = pd.DataFrame(scaler.transform(X), index=X.index)
X_train, X_test = Xs.loc[:"2019-01-14"], Xs.loc["2019-01-15":]
model = LogisticRegression().fit(X_train, y.loc[:"2019-01-14"])
print(model.score(X_test, y.loc["2019-01-15":]))
