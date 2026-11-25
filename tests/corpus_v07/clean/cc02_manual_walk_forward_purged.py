# expect: none
import pandas as pd
from sklearn.linear_model import Ridge

H = 15
df = pd.read_parquet("oil_features.parquet").sort_index()
df["target"] = df["close"].shift(-H) / df["close"] - 1
df = df.dropna()
cols = ["mom_10", "mom_60", "inv_chg"]
preds = []
for start in range(800, len(df), 40):
    fit_rows = df.iloc[:start - H]
    new_rows = df.iloc[start:start + 40]
    m = Ridge().fit(fit_rows[cols], fit_rows["target"])
    preds.append(pd.Series(m.predict(new_rows[cols]), index=new_rows.index))
print(pd.concat(preds).corr(df["target"]))
