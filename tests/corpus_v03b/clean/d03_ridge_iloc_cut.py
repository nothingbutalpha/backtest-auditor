# expect: none
# LA05 class: positional slice for training, the later rows for the score.
import pandas as pd
from sklearn.linear_model import Ridge

data = pd.read_parquet("design.parquet")  # last column is the forward return
cut = int(len(data) * 0.7)
reg = Ridge(alpha=1.0).fit(data.iloc[:cut, :-1], data.iloc[:cut, -1])
print(reg.score(data.iloc[cut:, :-1], data.iloc[cut:, -1]))
