# expect: LA05|LA09@8
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso

def scale_cols(df, cols):
    arr = StandardScaler().fit_transform(df[cols])
    return df.assign(**{c: arr[:, j] for j, c in enumerate(cols)})

cols = ["ret_5", "ret_20", "vol_20", "oi_chg"]
df = pd.read_parquet("wheat_daily.parquet").sort_index().dropna().pipe(scale_cols, cols)
r2 = []
for tr, te in TimeSeriesSplit(n_splits=6).split(df):
    m = Lasso(alpha=1e-4).fit(df[cols].iloc[tr], df["fwd_ret_1"].iloc[tr])
    r2.append(m.score(df[cols].iloc[te], df["fwd_ret_1"].iloc[te]))
print(r2)
