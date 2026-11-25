# expect: LA09:medium@7
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("country_macro.csv", index_col="month", parse_dates=True)
feats = ["cpi_yoy", "ip_yoy", "m2_yoy", "cab_gdp"]
df[feats] = df[feats].fillna(df[feats].median())
train, test = df.loc[:"2014"], df.loc["2015":]
rf = RandomForestRegressor(n_estimators=300).fit(train[feats], train["fx_fwd_3m"])
print(rf.score(test[feats], test["fx_fwd_3m"]))
