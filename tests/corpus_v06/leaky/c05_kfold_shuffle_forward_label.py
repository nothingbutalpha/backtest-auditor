# expect: LA06:medium@8
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_parquet("rates_features.parquet")
X, y = df.drop(columns="fwd_chg_5d"), df["fwd_chg_5d"]
cv = KFold(n_splits=5, shuffle=True, random_state=0)
print(cross_val_score(GradientBoostingRegressor(), X, y, cv=cv).mean())
