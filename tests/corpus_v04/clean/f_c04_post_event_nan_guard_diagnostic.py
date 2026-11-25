# expect: none
"""Event study: how often the post-event window has missing data (a data-quality diagnostic)."""
import numpy as np
import pandas as pd

px = pd.read_parquet("closes.parquet").to_numpy()
events = [(10, 3), (250, 7), (999, 1)]
missing = 0
for s, j in events:
    window = px[s + 1: s + 21, j]
    if np.isnan(window).any():
        missing += 1
print(missing / len(events))
