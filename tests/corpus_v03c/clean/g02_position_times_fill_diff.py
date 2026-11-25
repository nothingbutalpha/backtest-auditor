# expect: none
# Same execution model with a price difference and neutral names.
import pandas as pd

bars = pd.read_parquet("bars.parquet")
position = (bars["close"].pct_change(10) > 0).astype(float)
px_in = bars["open"].shift(-1)
px_out = bars["open"].shift(-2)
trade_ret = position * (px_out - px_in) / px_in - 0.0004 * position.diff().abs()
print(trade_ret.mean() * 252)
