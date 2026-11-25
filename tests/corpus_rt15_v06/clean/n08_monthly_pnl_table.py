# expect: none
import pandas as pd
d = pd.read_parquet("strategy_daily.parquet")
d["pnl"] = d["position"].shift(1) * d["ret"]
d["month_pnl"] = d.groupby(d.index.to_period("M"))["pnl"].transform("sum")
d["month_dd"] = d.groupby(d.index.to_period("M"))["pnl"].transform("min")
print(d[["month_pnl", "month_dd"]].drop_duplicates().to_string())
