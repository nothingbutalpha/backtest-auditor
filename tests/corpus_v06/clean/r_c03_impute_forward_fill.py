# expect: none
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("country_macro.csv", index_col="month", parse_dates=True)
feats = ["cpi_yoy", "ip_yoy", "m2_yoy", "cab_gdp"]
df[feats] = df[feats].ffill()
train, test = df.loc[:"2014"].dropna(), df.loc["2015":].dropna()
rf = RandomForestRegressor(n_estimators=300).fit(train[feats], train["fx_fwd_3m"])
print(rf.score(test[feats], test["fx_fwd_3m"]))
