# expect: LA09|LA05@6
import numpy as np
import pandas as pd

def standardize(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std()
    return (s - mu) / sigma

basis = pd.read_parquet("es_basis.parquet")["basis"]
z = basis.pipe(standardize)
position = (-z).clip(-1, 1).shift(1)
pnl = position * basis.diff()
print("pnl:", pnl.sum())
