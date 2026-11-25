# expect: LA04:medium@12
import pandas as pd

px = pd.read_parquet("px.parquet").sort_values("date")
guid = pd.read_parquet("guidance.parquet").sort_values("issued")
df = pd.merge_asof(
    px,
    guid,
    left_on="date",
    right_on="issued",
    by="ticker",
    direction="nearest",
)
print(df.groupby("date")["guide_chg"].mean().tail())
