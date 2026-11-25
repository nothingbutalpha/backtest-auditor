# leak: 7
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
def horizon_change(p, h):
    return p.pct_change(h).shift(-h)
df = pd.read_csv("cl_daily.csv", index_col=0, parse_dates=True)
df["chg5"] = horizon_change(df["close"], 5)
df["vol"] = df["close"].pct_change().rolling(20).std()
df["y"] = df["close"].pct_change().shift(-1)
df = df.dropna()
tr, te = df.loc[:"2019"], df.loc["2020":]
rf = RandomForestRegressor().fit(tr[["chg5", "vol"]], tr["y"])
print(rf.score(te[["chg5", "vol"]], te["y"]))
