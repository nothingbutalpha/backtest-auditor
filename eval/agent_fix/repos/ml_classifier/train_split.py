import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("daily.csv", parse_dates=["date"]).sort_values("date")
df["ret_1"] = df["close"].pct_change()
df["ret_5"] = df["close"].pct_change(5)
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
df = df.dropna()

X_train, X_test, y_train, y_test = train_test_split(df[["ret_1", "ret_5"]], df["target"], test_size=0.25)
clf = GradientBoostingClassifier().fit(X_train, y_train)
print("accuracy", clf.score(X_test, y_test))
