# expect: none
import pandas as pd
from sklearn.linear_model import Ridge
df = pd.read_parquet("x.parquet").sort_index()
cut = int(len(df) * 0.7)
fit_rows = df.iloc[:cut]
lo, hi = fit_rows["f1"].quantile(0.01), fit_rows["f1"].quantile(0.99)
df["f1"] = df["f1"].clip(lo, hi)
train, test = df.iloc[:cut], df.iloc[cut:]
print(Ridge().fit(train[["f1"]], train["y"]).score(test[["f1"]], test["y"]))
