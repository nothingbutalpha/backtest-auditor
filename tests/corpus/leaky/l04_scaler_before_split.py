# expect: LA05:high
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("features.csv", parse_dates=["date"], index_col="date")
X = df[["mom", "vol", "rsi"]]
y = (df["close"].shift(-1) > df["close"]).astype(int)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, shuffle=False)
model = LogisticRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))
