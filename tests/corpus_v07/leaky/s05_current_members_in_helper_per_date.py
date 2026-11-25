# expect: SB01@8
import pandas as pd

roster = pd.read_parquet("index_roster.parquet")   # symbol, added, removed
close = pd.read_parquet("close_wide.parquet")

def members_on(day):
    active = roster[roster["removed"].isna()]
    return list(active.loc[active["added"] <= day, "symbol"])

books = {}
for day in close.resample("BME").last().index:
    cols = [c for c in members_on(day) if c in close.columns]
    books[day] = pd.Series(1.0 / len(cols), index=cols)
w = pd.DataFrame(books).T.reindex(close.index).ffill().shift(1)
print((w * close.pct_change()).sum(axis=1).mean())
