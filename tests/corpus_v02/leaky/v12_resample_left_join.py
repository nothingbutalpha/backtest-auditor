# expect: LA12:medium
import pandas as pd
ticks = pd.read_csv("es_1min.csv", parse_dates=["ts"], index_col="ts")
bars15 = ticks["close"].resample("15min").last()
ticks["close_15"] = bars15.reindex(ticks.index, method="ffill")
ticks["signal"] = (ticks["close"] > ticks["close_15"]).astype(int)
ticks["pnl"] = ticks["signal"].shift(1) * ticks["close"].pct_change() - 0.0001 * ticks["signal"].diff().abs()
