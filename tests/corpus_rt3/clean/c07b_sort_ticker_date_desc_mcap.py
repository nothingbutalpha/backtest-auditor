# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("panel.parquet")
df = df.sort_values(["ticker", "date", "volume"], ascending=[True, True, False]).drop_duplicates(["ticker", "date"])
df["ret"] = df.groupby("ticker")["close"].pct_change()
df["sma"] = df.groupby("ticker")["close"].transform(lambda s: s.rolling(50).mean())
df["signal"] = (df["close"] > df["sma"]).astype(int)
df["position"] = df.groupby("ticker")["signal"].shift(1)
