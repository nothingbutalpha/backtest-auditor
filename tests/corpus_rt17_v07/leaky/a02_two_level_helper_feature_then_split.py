# leak: 5
import pandas as pd
from sklearn.linear_model import Ridge
def _standardise(col):
    centre = col.mean()
    spread = col.std()
    return (col - centre) / spread
def build_features(frame):
    return pd.DataFrame({"vol": _standardise(frame["vol20"]), "mom": _standardise(frame["mom60"])})
d = pd.read_parquet("panel_one_name.parquet")
X = build_features(d)
y = d["close"].pct_change().shift(-1)
cut = int(len(d) * 0.7)
mdl = Ridge().fit(X.iloc[:cut], y.iloc[:cut])
print(mdl.score(X.iloc[cut:].dropna(), y.iloc[cut:].dropna()))
