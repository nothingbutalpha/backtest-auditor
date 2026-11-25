# expect: none
import pandas as pd
from sklearn.linear_model import Ridge
def expz(col):
    return (col - col.expanding(60).mean()) / col.expanding(60).std()
d = pd.read_parquet("panel_one_name.parquet")
X = pd.DataFrame({"vol": expz(d["vol20"]), "mom": expz(d["mom60"])})
y = d["close"].pct_change().shift(-1)
data = pd.concat([X, y.rename("y")], axis=1).dropna()
cut = int(len(data) * 0.7)
mdl = Ridge().fit(data.iloc[:cut][["vol", "mom"]], data.iloc[:cut]["y"])
print(mdl.score(data.iloc[cut:][["vol", "mom"]], data.iloc[cut:]["y"]))
