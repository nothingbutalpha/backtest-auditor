# expect: none
import pandas as pd
mcap = pd.read_parquet("mcap_wide.parquet")
rets = pd.read_parquet("ret_wide.parquet")
book = []
for d in mcap.index[:-1]:
    top = mcap.loc[:d].iloc[-1].nlargest(100).index
    nxt = rets.index[rets.index > d][0]
    book.append(rets.loc[nxt, top].mean())
print(sum(book))
