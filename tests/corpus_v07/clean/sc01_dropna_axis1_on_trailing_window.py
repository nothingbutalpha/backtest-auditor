# expect: none
import pandas as pd

wide = pd.read_parquet("asx_close.parquet")
month = wide.resample("ME").last()
picks = {}
for i in range(13, len(month)):
    window = month.iloc[i - 13:i].dropna(axis=1)
    mom = window.iloc[-1] / window.iloc[0] - 1
    picks[month.index[i]] = mom.nlargest(20).index
print(len(picks))
