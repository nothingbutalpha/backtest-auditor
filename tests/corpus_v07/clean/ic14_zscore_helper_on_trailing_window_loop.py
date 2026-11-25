# expect: none
import pandas as pd

def zscore(s):
    return (s.iloc[-1] - s.mean()) / s.std()

basis = pd.read_parquet("es_basis.parquet")["basis"]
z = pd.Series({basis.index[t]: zscore(basis.iloc[t - 60:t + 1]) for t in range(60, len(basis))})
position = (-z).clip(-1, 1).shift(1)
pnl = position * basis.diff().reindex(position.index)
print(pnl.sum())
