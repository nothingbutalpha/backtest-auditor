# expect: none
import pandas as pd
mem = pd.read_csv("sp500_membership.csv", parse_dates=["start", "end"])
px = pd.read_parquet("all_closes.parquet")
def members_on(day):
    live = mem[(mem["start"] <= day) & (mem["end"].isna() | (mem["end"] > day))]
    return [t for t in live["ticker"] if t in px.columns]
rets = px.pct_change()
book = []
for day in rets.index[252::21]:
    names = members_on(day)
    look = rets.loc[:day, names].iloc[-252:-21].sum()
    top = look.nlargest(50).index
    nxt = rets.loc[rets.index > day, top].iloc[:21]
    book.append(nxt.mean(axis=1))
print(pd.concat(book).add(1).prod())
