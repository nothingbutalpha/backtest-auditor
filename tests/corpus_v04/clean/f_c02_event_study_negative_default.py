# expect: none
"""Diagnostic: pre-event volume relative to the trailing median, per event."""
import numpy as np
import pandas as pd


def pre_event_ratio(events, vol, med, lo=-5, hi=-1):
    out = []
    for s in events:
        if s + lo < 0:
            out.append(np.nan)
            continue
        base = med[s + lo]
        seg = vol[s + lo: s + hi + 1]
        if not np.isfinite(base) or base <= 0 or np.isnan(seg).all():
            out.append(np.nan)
            continue
        out.append(np.nanmax(seg) / base)
    return pd.Series(out, name="pre_event_ratio")


v = pd.read_csv("volume.csv")["volume"]
m = v.rolling(20, min_periods=10).median().shift(1).to_numpy()
ev = np.flatnonzero(v.to_numpy() > 3 * np.nan_to_num(m, nan=np.inf))
print(pre_event_ratio(ev, v.to_numpy(), m).describe())
