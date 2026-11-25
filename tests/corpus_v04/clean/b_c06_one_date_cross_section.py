# expect: none
import pandas as pd

panel = pd.read_parquet("fundamentals_panel.parquet")
asof = panel["date"].max()
snap = panel[panel["date"] == asof].set_index("ticker")
cols = ["earnings_yield", "book_to_price", "roe"]
z = snap[cols].apply(lambda c: (c - c.mean()) / c.std())
picks = z.sum(axis=1).nlargest(30).index.tolist()
print(asof, picks)
