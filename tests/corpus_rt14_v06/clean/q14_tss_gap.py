# expect: none
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("daily.csv", index_col=0, parse_dates=True)
df["target"] = df["close"].shift(-10) / df["close"] - 1
df = df.dropna()
cv = TimeSeriesSplit(n_splits=5, gap=10)
print(cross_val_score(RandomForestRegressor(), df[["rsi", "vol"]], df["target"], cv=cv).mean())
