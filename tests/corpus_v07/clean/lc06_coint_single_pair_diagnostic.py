# expect: none
import pandas as pd
from statsmodels.tsa.stattools import coint

px = pd.read_parquet("reit_close.parquet")
stat, p, crit = coint(px["PLD"], px["DRE"])
print(f"Engle-Granger p-value over the sample: {p:.3f}")
