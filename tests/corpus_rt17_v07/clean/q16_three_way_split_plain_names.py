# expect: none
import pandas as pd
from sklearn.linear_model import Ridge
d = pd.read_parquet("features.parquet")
early, mid, late = d.loc[:"2015"], d.loc["2016":"2018"], d.loc["2019":]
cols = ["a", "b", "c"]
results = {}
for alpha in (0.1, 1.0, 10.0):
    m = Ridge(alpha=alpha).fit(early[cols], early["y"])
    results[alpha] = m.score(mid[cols], mid["y"])
pick = max(results, key=results.get)
final = Ridge(alpha=pick).fit(pd.concat([early, mid])[cols], pd.concat([early, mid])["y"])
print(pick, final.score(late[cols], late["y"]))
