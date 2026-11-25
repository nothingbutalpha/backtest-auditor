# expect: none
import pandas as pd
from sklearn.linear_model import LinearRegression
data = pd.read_parquet("factors.parquet")
label = data.pop("ret_next")
early, late = data.index < "2016-01-01", data.index >= "2016-01-01"
ic = data[early].corrwith(label[early], method="spearman")
keep = ic.abs().nlargest(5).index
lr = LinearRegression().fit(data.loc[early, keep], label[early])
print(lr.score(data.loc[late, keep], label[late]))
