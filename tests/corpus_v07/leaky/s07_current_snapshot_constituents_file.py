# expect: SB01@4
import pandas as pd

syms = pd.read_csv("nasdaq100_members_latest.csv")["symbol"]
hist = pd.read_parquet("us_close_1999_2024.parquet")
hist = hist[hist.columns.intersection(syms)]
ret = hist.pct_change()
mom = hist.pct_change(126).shift(1)
pick = mom.rank(axis=1, ascending=False) <= 10
print(ret[pick].mean(axis=1).mean() * 252)
