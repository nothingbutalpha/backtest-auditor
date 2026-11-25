# expect: LA09:medium@7
import pandas as pd
from sklearn.linear_model import Ridge

df = pd.read_parquet("quality_signals.parquet").sort_index()
cols = ["gross_prof", "roa_chg", "accruals"]
q_lo, q_hi = df[cols].quantile(0.02), df[cols].quantile(0.98)
df[cols] = df[cols].clip(q_lo, q_hi, axis=1)
train, test = df.loc[:"2017-12-31"], df.loc["2018-02-01":]
m = Ridge().fit(train[cols], train["fwd_ret_1m"])
print(m.score(test[cols], test["fwd_ret_1m"]))
