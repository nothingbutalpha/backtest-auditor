# leak: 6
import numpy as np
import pandas as pd
m = pd.read_parquet("es_1m.parquet")
m["entry"] = m["open"]
m["tp"] = m.groupby(m.index.date)["high"].transform("max") * 0.999
exit_px = np.where(m["high"] >= m["tp"], m["tp"], m["close"])
m["pnl"] = exit_px / m["entry"] - 1
print(m["pnl"].mean())
