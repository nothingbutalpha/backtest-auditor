# expect: none
import pandas as pd
px = pd.read_parquet("russell_closes.parquet")
alive = px.columns[px.tail(1).notna().all()]
print(f"{len(alive)} of {px.shape[1]} names still trade today")
print(px.notna().sum(axis=1).describe())
