# expect: SB01:medium@6
import pandas as pd

px = pd.read_parquet("semis_close.parquet").loc["2014":"2024"]
growth = px.iloc[-1] / px.iloc[0] - 1
top = growth.nlargest(8).index
basket = px[top].pct_change().mean(axis=1)
print("CAGR", (1 + basket.fillna(0)).prod() ** (252 / len(basket)) - 1)
