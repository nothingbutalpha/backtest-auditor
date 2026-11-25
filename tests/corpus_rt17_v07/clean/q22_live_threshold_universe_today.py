# expect: none
import pandas as pd
cap = pd.read_parquet("mktcap.parquet")
last = cap.iloc[-1]
today = last[last > 2e9].index
px = pd.read_parquet("all_closes.parquet")[today]
mom = px.iloc[-1] / px.iloc[-127] - 1
print(mom.nlargest(25))
