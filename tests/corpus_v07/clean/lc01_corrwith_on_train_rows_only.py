# expect: none
import pandas as pd
from sklearn.linear_model import Ridge

data = pd.read_parquet("credit_macro.parquet").sort_index()
train, test = data.loc[:"2015"], data.loc["2016":]
ic = train.drop(columns="y").corrwith(train["y"]).abs()
chosen = ic.nlargest(10).index
m = Ridge().fit(train[chosen], train["y"])
print(m.score(test[chosen], test["y"]))
