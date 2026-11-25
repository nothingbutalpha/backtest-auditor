# expect: none
import pandas as pd
from sklearn.linear_model import Ridge

df = pd.read_parquet("quality_signals.parquet").sort_index()
cols = ["gross_prof", "roa_chg", "accruals"]
train, test = df.loc[:"2017-12-31"].copy(), df.loc["2018-02-01":].copy()
q_lo, q_hi = train[cols].quantile(0.02), train[cols].quantile(0.98)
train[cols] = train[cols].clip(q_lo, q_hi, axis=1)
test[cols] = test[cols].clip(q_lo, q_hi, axis=1)
m = Ridge().fit(train[cols], train["fwd_ret_1m"])
print(m.score(test[cols], test["fwd_ret_1m"]))
