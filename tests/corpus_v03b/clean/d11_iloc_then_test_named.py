# expect: none
# LA05 class (added after the first measurement): iloc slice fit, then a `test` frame.
import pandas as pd
from sklearn.linear_model import Ridge

data = pd.read_parquet("design.parquet")
cut = int(len(data) * 0.7)
reg = Ridge(alpha=1.0).fit(data.iloc[:cut, :-1], data.iloc[:cut, -1])
test = data.iloc[cut:]
print(reg.score(test.iloc[:, :-1], test.iloc[:, -1]))
