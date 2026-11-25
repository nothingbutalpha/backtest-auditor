# expect: none
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("features.csv", parse_dates=["date"], index_col="date")
df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
df = df.dropna()
X_train, X_test, y_train, y_test = train_test_split(df[["mom", "vol"]], df["label"], test_size=0.3, shuffle=False)
scaler = StandardScaler().fit(X_train)
model = LogisticRegression().fit(scaler.transform(X_train), y_train)
acc = model.score(scaler.transform(X_test), y_test)
