# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
rows = []
for name, model in {"ridge": Ridge(), "lasso": Lasso(), "rf": RandomForestRegressor()}.items():
    model.fit(X_train, y_train)
    rows.append((name, model.score(X_test, y_test)))
import pandas as pd
print(pd.DataFrame(rows, columns=["model", "test_r2"]).sort_values("test_r2"))
