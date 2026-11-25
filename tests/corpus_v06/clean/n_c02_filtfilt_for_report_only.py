# expect: none
"""Chart helper: a zero-phase smooth of the realized equity curve for the report (no trading decision)."""
import pandas as pd
from scipy.signal import butter, filtfilt

equity = pd.read_csv("equity_curve.csv", index_col=0, parse_dates=True)["equity"]
b, a = butter(2, 0.05)
display = pd.Series(filtfilt(b, a, equity.to_numpy()), index=equity.index)
print(display.describe())
display.to_csv("equity_smooth_for_chart.csv")
