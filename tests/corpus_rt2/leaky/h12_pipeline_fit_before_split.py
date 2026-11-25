import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["ret"] = df["Close"].pct_change()
df["target"] = (df["ret"].shift(-1) > 0).astype(int)
X = df[["ret"]].fillna(0)
y = df["target"]
pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
pipe.fit(X, y)
split = int(len(df) * 0.7)
X_test = X.iloc[split:]
df.loc[X_test.index, "pred"] = pipe.predict(X_test)
