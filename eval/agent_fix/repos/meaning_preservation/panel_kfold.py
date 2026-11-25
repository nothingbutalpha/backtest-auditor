import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

panel = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
X, y = panel[["mom", "bm"]], panel["fwd_ret"]
print(cross_val_score(Ridge(), X, y, cv=KFold(5, shuffle=True)).mean())
