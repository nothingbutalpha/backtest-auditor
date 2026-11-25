# expect: SB01@6
import pandas as pd

def load(path):
    px = pd.read_parquet(path).sort_index()
    keep = px.columns[px.iloc[-1].notna()]
    return px[keep]

px = load("hk_close.parquet")
ret = px.pct_change(fill_method=None)
cheap = (px.rank(axis=1, pct=True) < 0.2).astype(float).shift(1)
print((cheap.div(cheap.sum(axis=1), axis=0) * ret).sum(axis=1).mean())
