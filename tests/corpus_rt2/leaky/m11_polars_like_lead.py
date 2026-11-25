import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["lead_ret"] = df["Close"].pct_change().shift(-1)
X = df.drop(columns=["Close"])
from sklearn.tree import DecisionTreeClassifier
X_train, X_test = X.iloc[:2000], X.iloc[2000:]
