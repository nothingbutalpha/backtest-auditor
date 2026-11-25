# expect: none
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

df = pd.read_parquet("signals_400.parquet").sort_index()
y = df.pop("ret_next")
pipe = make_pipeline(SelectKBest(f_regression, k=15), Ridge())
print(cross_val_score(pipe, df, y, cv=TimeSeriesSplit(5, gap=1)).mean())
