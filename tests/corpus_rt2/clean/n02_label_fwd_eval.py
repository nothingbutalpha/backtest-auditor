import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["mom"] = df["Close"].pct_change(20)
df["fwd_ret_21"] = df["Close"].pct_change(21).shift(-21)
ic = df["mom"].corr(df["fwd_ret_21"], method="spearman")
print(ic)
