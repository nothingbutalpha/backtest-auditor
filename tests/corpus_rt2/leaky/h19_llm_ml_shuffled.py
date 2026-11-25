import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = yf.download("SPY", period="5y")
df["Return"] = df["Close"].pct_change()
df["Lag1"] = df["Return"].shift(1)
df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
df = df.dropna()
X = df[["Lag1", "Return"]]
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier().fit(X_train, y_train)
print(clf.score(X_test, y_test))
