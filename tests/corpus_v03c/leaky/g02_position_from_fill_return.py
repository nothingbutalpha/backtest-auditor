# expect: LA01|LA10:high
# Twin of clean/g02: the position is built from the fill-to-fill return itself.
import pandas as pd

bars = pd.read_parquet("bars.parquet")
px_in = bars["open"].shift(-1)
px_out = bars["open"].shift(-2)
oo = (px_out - px_in) / px_in
position = (oo.rolling(3).mean() > 0).astype(float)
trade_ret = position * oo - 0.0004 * position.diff().abs()
