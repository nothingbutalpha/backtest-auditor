# expect: none
import pandas as pd

data = pd.read_parquet("credit_macro.parquet")
ic = data.drop(columns="hy_spread_fwd_3m").corrwith(data["hy_spread_fwd_3m"])
print(ic.sort_values().to_string())
