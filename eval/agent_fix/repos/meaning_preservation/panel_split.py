import pandas as pd
from sklearn.model_selection import train_test_split

panel = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
X, y = panel[["mom", "bm"]], panel["fwd_ret"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
