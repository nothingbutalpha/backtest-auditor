# expect: LA09:medium
import numpy as np
import pandas as pd


def run(prices, cost_bps=5):
    out = pd.DataFrame(index=prices.index)
    out["ret"] = prices["close"].pct_change()
    out["mom"] = prices["close"].pct_change(126)
    out["signal"] = np.clip(out["mom"] / out["mom"].std(), -1, 1)
    out["pos"] = out["signal"].shift(1)
    out["net"] = out["pos"] * out["ret"] - cost_bps / 1e4 * out["pos"].diff().abs()
    return out
