# expect: LA06:high
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("btc_1h.csv", parse_dates=["date"], index_col="date")
df["ret_1"] = df["close"].pct_change()
df["ret_24"] = df["close"].pct_change(24)
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
df = df.dropna()
X_train, X_test, y_train, y_test = train_test_split(df[["ret_1", "ret_24"]], df["target"], test_size=0.2)
clf = RandomForestClassifier(n_estimators=300).fit(X_train, y_train)
print("accuracy", clf.score(X_test, y_test))
