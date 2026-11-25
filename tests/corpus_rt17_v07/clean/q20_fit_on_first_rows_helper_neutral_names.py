# expect: none
import pandas as pd
from sklearn.linear_model import LinearRegression
def centre(x, ref):
    return (x - ref.mean()) / ref.std()
d = pd.read_parquet("features.parquet")
first = d.iloc[:2000]
rest = d.iloc[2000:]
cols = ["a", "b", "c"]
m = LinearRegression().fit(centre(first[cols], first[cols]), first["y"])
print(m.score(centre(rest[cols], first[cols]), rest["y"]))
