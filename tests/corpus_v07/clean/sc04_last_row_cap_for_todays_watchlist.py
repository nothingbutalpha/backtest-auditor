# expect: none
import pandas as pd

mcap = pd.read_parquet("eu_mcap.parquet")
final_cap = mcap.ffill().iloc[-1]
names = final_cap[final_cap >= 5e9].index
print(len(names), "large caps today:", sorted(names)[:20])
