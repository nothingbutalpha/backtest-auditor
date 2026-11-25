# expect: LA07:high
import pandas as pd

df = pd.read_parquet("iwm.parquet")
df["ma"] = df["close"].rolling(30).mean()
exit_sig = df["close"] < df["ma"]
entry_px = df["open"].shift(-1)
exit_px = df["open"].mask(~exit_sig)
print(exit_px.dropna().head(), entry_px.dropna().head())
