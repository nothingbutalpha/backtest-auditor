# expect: LA10:medium
# From the red-team review of v0.1 (regression case).
import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_parquet("f.parquet")
df["y_next"] = df["close"].pct_change().shift(-1)
df["trend"] = df["y_next"].rolling(5).mean()
X = df[["trend"]].dropna(); y = df["y_next"].loc[X.index]
split = int(len(X) * 0.7)
LinearRegression().fit(X[:split], y[:split])
