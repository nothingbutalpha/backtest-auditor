# expect: none
import pandas as pd
es = pd.read_csv("es_1min.csv", parse_dates=["ts"], index_col="ts")
nq = pd.read_csv("nq_1min.csv", parse_dates=["ts"], index_col="ts")
es5 = es["close"].resample("5min").last().to_frame("es")
nq5 = nq["close"].resample("5min").last().to_frame("nq")
both = es5.join(nq5, how="inner")
both["spread_z"] = (both["es"] - both["nq"]).rolling(48).mean()
both["signal"] = (both["spread_z"] > 0).astype(int)
both["pnl"] = both["signal"].shift(1) * both["es"].pct_change() - 0.0001 * both["signal"].diff().abs()
