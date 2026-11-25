# expect: LA05|LA09@7
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

def fill_gaps(panel, cols):
    vals = KNNImputer(n_neighbors=5).fit_transform(panel[cols])
    return pd.DataFrame(vals, index=panel.index, columns=cols)

panel = pd.read_parquet("muni_panel.parquet").sort_values("date")
cols = ["dur", "oas", "rating", "liq"]
X = fill_gaps(panel, cols)
is_train = panel["date"] < "2019-01-01"
rf = RandomForestRegressor(n_estimators=200).fit(X[is_train], panel.loc[is_train, "fwd_spread_chg"])
print(rf.score(X[~is_train], panel.loc[~is_train, "fwd_spread_chg"]))
