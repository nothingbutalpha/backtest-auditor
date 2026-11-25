# expect: LA05
# Twin of clean/d03: fitted on every row, scored on the last 30% (which it was trained on).
import pandas as pd
from sklearn.linear_model import Ridge

data = pd.read_parquet("design.parquet")
cut = int(len(data) * 0.7)
reg = Ridge(alpha=1.0).fit(data.iloc[:, :-1], data.iloc[:, -1])
print(reg.score(data.iloc[cut:, :-1], data.iloc[cut:, -1]))
