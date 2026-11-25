# expect: LA01|LA10:high@7
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

panel = pd.read_parquet("eu_panel.parquet").sort_values(["isin", "date"])
panel["ret"] = panel.groupby("isin")["close"].pct_change()
panel["ret_avg"] = panel.groupby("isin")["ret"].transform(lambda s: s.rolling(3).mean().shift(-1))
panel["y"] = panel.groupby("isin")["ret"].shift(-1)
data = panel.dropna()
rf = RandomForestRegressor(n_estimators=100).fit(data[["ret_avg"]].iloc[:20000], data["y"].iloc[:20000])
print(rf.score(data[["ret_avg"]].iloc[20000:], data["y"].iloc[20000:]))
