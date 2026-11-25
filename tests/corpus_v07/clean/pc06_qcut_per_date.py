# expect: none
import pandas as pd

df = pd.read_parquet("accruals_panel.parquet")
df["q"] = df.groupby("date")["accruals"].transform(lambda s: pd.qcut(s, 5, labels=False))
table = df.groupby(["date", "q"])["ret_next_month"].mean().unstack()
spread = table[0] - table[4]
print(spread.mean())
