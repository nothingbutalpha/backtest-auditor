# expect: SB01:medium
import pandas as pd
df = pd.read_parquet("daily.parquet")
end = df["date"].max()
df = df[df.groupby("ticker")["date"].transform("max") == end]
df["ret"] = df.groupby("ticker")["close"].pct_change()
