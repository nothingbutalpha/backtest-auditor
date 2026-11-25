import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("daily.csv", parse_dates=["date"]).sort_values("date")
df["eps"] = df["eps"].interpolate()
df["ret_1"] = df["close"].pct_change()
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
df = df.dropna()
X_train, X_test, y_train, y_test = train_test_split(df[["ret_1", "eps"]], df["target"], test_size=0.2,
                                                    stratify=df["target"])
