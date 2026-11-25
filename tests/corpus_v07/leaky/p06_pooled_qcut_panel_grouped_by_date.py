# expect: LA09@5
import pandas as pd

df = pd.read_parquet("accruals_panel.parquet")   # date, gvkey, accruals, ret_next_month
df["q"] = pd.qcut(df["accruals"], 5, labels=False)
table = df.groupby(["date", "q"])["ret_next_month"].mean().unstack()
spread = table[0] - table[4]
print(spread.mean(), spread.mean() / spread.std() * len(spread) ** 0.5)
