# expect: SB01@6
import pandas as pd

closes = (
    pd.read_parquet("kospi_close.parquet")
    .pipe(lambda f: f.dropna(axis=1, thresh=len(f)))
)
r = closes.pct_change()
w = (r.rolling(20).sum().rank(axis=1, pct=True) < 0.2).astype(float)
print((w.div(w.sum(axis=1), axis=0).shift(1) * r).sum(axis=1).mean())
