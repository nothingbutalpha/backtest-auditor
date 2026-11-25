"""Red-team round 15 files (lint v0.6 after the F1-F7 fixes). Written fresh; not derived from dev sets or rt14."""
import sys
from pathlib import Path

D = Path(sys.argv[1])
(D / "leaky").mkdir(parents=True, exist_ok=True)
(D / "clean").mkdir(parents=True, exist_ok=True)
L, C = {}, {}

# ---------------- LA13 backtrader ----------------
L["m01_bt_btind_alias_ahead.py"] = '''import backtrader as bt
import backtrader.indicators as btind
class MeanRev(bt.Strategy):
    def __init__(self):
        self.rsi = btind.RSI(period=14)
    def next(self):
        if self.rsi[1] < 30 and not self.position:
            self.buy()
        elif self.rsi[0] > 70 and self.position:
            self.close()
'''
L["m02_bt_local_line_alias.py"] = '''import backtrader as bt
class Gap(bt.Strategy):
    def next(self):
        o = self.data.open
        c = self.data.close
        if o[1] > c[0] * 1.01:
            self.buy()
'''
L["m03_bt_datas_feed_line_ahead.py"] = '''import backtrader as bt
class Pair(bt.Strategy):
    def next(self):
        a = self.datas[0]
        if self.datas[1].close[1] > self.datas[1].close[0]:
            self.buy(data=a)
'''
C["n01_bt_params_tuple_in_next.py"] = '''import backtrader as bt
class Bands(bt.Strategy):
    params = (("bounds", (0.97, 1.03)), ("sizes", [10, 20]))
    def __init__(self):
        self.ma = bt.ind.SMA(self.data.close, period=20)
    def next(self):
        if self.data.close[0] < self.ma[0] * self.p.bounds[0]:
            self.buy(size=self.p.sizes[1])
        elif self.data.close[0] > self.ma[0] * self.params.bounds[1]:
            self.sell(size=self.p.sizes[0])
'''
C["n02_bt_order_list_and_feeds.py"] = '''import backtrader as bt
class Rotate(bt.Strategy):
    def __init__(self):
        self.orders = {}
        self.ranked = sorted(self.datas, key=lambda d: d._name)
        self.moms = {d: bt.ind.ROC(d.close, period=60) for d in self.datas}
    def next(self):
        best = max(self.datas, key=lambda d: self.moms[d][0])
        second = sorted(self.datas, key=lambda d: self.moms[d][0])[-2]
        if self.ranked[1] is best or second is best:
            self.order_target_percent(best, 0.5)
'''

# ---------------- LA07 limit checks ----------------
L["m04_limit_vs_same_bar_low_attr.py"] = '''import pandas as pd
df = pd.read_parquet("nq_1m.parquet")
df["limit_price"] = df.close - 2.0
df["got_filled"] = df.low <= df.limit_price
df["pnl"] = (df.close.shift(-5) - df.limit_price).where(df.got_filled, 0.0)
print(df.pnl.sum())
'''
L["m05_limit_same_bar_low_numpy.py"] = '''import numpy as np
import pandas as pd
k = pd.read_csv("cl_1h.csv")
lo = k["low"].to_numpy()
order_px = k["close"].to_numpy() * 0.997
hit = lo <= order_px
exit_px = np.roll(k["close"].to_numpy(), -3)
print(np.where(hit, exit_px / order_px - 1, 0.0).sum())
'''
L["m06_limit_alias_of_low_not_shifted.py"] = '''import pandas as pd
bars = pd.read_csv("es_5m.csv")
bar_low = bars["low"].copy()
limit_px = bars["close"] * 0.998
hit = bar_low <= limit_px
pnl = (bars["close"].shift(-1) / limit_px - 1).where(hit, 0.0)
print(pnl.sum())
'''
C["n03_limit_prev_close_vs_low.py"] = '''import pandas as pd
bars = pd.read_csv("es_5m.csv")
limit_px = bars["close"].shift(1) * 0.998
hit = bars["low"] <= limit_px
pnl = (bars["close"] / limit_px - 1).where(hit, 0.0)
print(pnl.sum())
'''
C["n04_limit_next_bar_column.py"] = '''import pandas as pd
df = pd.read_parquet("nq_1m.parquet")
df["limit_price"] = df["close"] - 2.0
df["low_next"] = df["low"].shift(-1)
df["got_filled"] = df["low_next"] <= df["limit_price"]
df["pnl"] = (df["close"].shift(-5) - df["limit_price"]).where(df["got_filled"], 0.0)
print(df["pnl"].sum())
'''

# ---------------- LA02 causal and non-causal filters ----------------
L["m07_gradient_lag1_times_same_bar_ret.py"] = '''import numpy as np
import pandas as pd
px = pd.read_csv("gc.csv")["close"]
vel = pd.Series(np.gradient(px.ewm(span=8).mean().to_numpy()), index=px.index)
pos = np.sign(vel.shift(1))
strat = pos * px.pct_change()
print(strat.sum())
'''
L["m08_filtfilt_where_pnl.py"] = '''import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
px = pd.read_csv("zn.csv")["close"]
b, a = butter(2, 0.1)
trend = filtfilt(b, a, px.to_numpy())
up = trend > np.roll(trend, 1)
ret = px.pct_change().to_numpy()
pnl = np.where(up, ret, -ret)
print(np.nansum(pnl))
'''
L["m09_argrel_shift_short_of_order.py"] = '''import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
df = pd.read_csv("eurusd_1h.csv")
idx = argrelextrema(df["close"].to_numpy(), np.greater, order=8)[0]
df["top"] = 0
df.loc[idx, "top"] = 1
df["short"] = -df["top"].shift(3)
df["pnl"] = df["short"].shift(1) * df["close"].pct_change()
print(df["pnl"].sum())
'''
L["m10_savgol_centered_rolling_apply.py"] = '''import pandas as pd
from scipy.signal import savgol_filter
px = pd.read_csv("btc.csv", index_col=0)["close"]
smooth = px.rolling(21, center=True).apply(lambda w: savgol_filter(w, 21, 2)[-1], raw=True)
signal = (px > smooth).astype(int)
pnl = signal.shift(1) * px.pct_change()
print(pnl.sum())
'''
C["n05_lfilter_ewm_signal.py"] = '''import numpy as np
import pandas as pd
from scipy.signal import lfilter
px = pd.read_csv("zn.csv")["close"]
alpha = 0.1
trend = lfilter([alpha], [1, alpha - 1], px.to_numpy())
up = pd.Series(trend).diff() > 0
pnl = up.shift(1).fillna(False).astype(int) * px.pct_change()
print(pnl.sum())
'''
C["n06_gradient_two_lags.py"] = '''import numpy as np
import pandas as pd
px = pd.read_csv("gc.csv")["close"]
vel = pd.Series(np.gradient(px.ewm(span=8).mean().to_numpy()), index=px.index)
pos = np.sign(vel.shift(1))
strat = pos.shift(1) * px.pct_change()
print(strat.sum())
'''
C["n07_savgol_helper_window_apply.py"] = '''import pandas as pd
from scipy.signal import savgol_filter
def last_smoothed(w):
    return savgol_filter(w, len(w), 3)[-1]
px = pd.read_csv("btc.csv", index_col=0)["close"]
smooth = px.rolling(41).apply(last_smoothed, raw=True)
signal = (px > smooth).astype(int)
pnl = signal.shift(1) * px.pct_change()
print(pnl.sum())
'''

# ---------------- LA12 report-only vs decision ----------------
L["m11_day_high_take_profit_fill.py"] = '''import numpy as np
import pandas as pd
m = pd.read_parquet("es_1m.parquet")
m["entry"] = m["open"]
m["tp"] = m.groupby(m.index.date)["high"].transform("max") * 0.999
exit_px = np.where(m["high"] >= m["tp"], m["tp"], m["close"])
m["pnl"] = exit_px / m["entry"] - 1
print(m["pnl"].mean())
'''
L["m12_week_vol_scaled_size.py"] = '''import pandas as pd
h = pd.read_parquet("btc_1h.parquet")
h["ret"] = h["close"].pct_change()
h["wk_vol"] = h.groupby(pd.Grouper(freq="W"))["ret"].transform("std")
h["size"] = 0.01 / h["wk_vol"]
h["strategy_ret"] = h["size"].shift(1) * h["ret"]
print(h["strategy_ret"].sum())
'''
C["n08_monthly_pnl_table.py"] = '''import pandas as pd
d = pd.read_parquet("strategy_daily.parquet")
d["pnl"] = d["position"].shift(1) * d["ret"]
d["month_pnl"] = d.groupby(d.index.to_period("M"))["pnl"].transform("sum")
d["month_dd"] = d.groupby(d.index.to_period("M"))["pnl"].transform("min")
print(d[["month_pnl", "month_dd"]].drop_duplicates().to_string())
'''
C["n09_day_volume_share_report.py"] = '''import pandas as pd
m = pd.read_parquet("es_1m.parquet")
m["signal"] = (m["close"] > m["close"].rolling(20).mean()).astype(int).shift(1)
m["pnl"] = m["signal"] * m["close"].pct_change()
m["day_vol"] = m.groupby(m.index.date)["volume"].transform("sum")
m["vol_share"] = m["volume"] / m["day_vol"]
print(m["vol_share"].describe(), m["pnl"].sum())
'''

LEAK_LINES = {
    "m01_bt_btind_alias_ahead.py": 7, "m02_bt_local_line_alias.py": 6, "m03_bt_datas_feed_line_ahead.py": 5,
    "m04_limit_vs_same_bar_low_attr.py": 4, "m05_limit_same_bar_low_numpy.py": 6,
    "m06_limit_alias_of_low_not_shifted.py": 5, "m07_gradient_lag1_times_same_bar_ret.py": 4,
    "m08_filtfilt_where_pnl.py": 6, "m09_argrel_shift_short_of_order.py": 5,
    "m10_savgol_centered_rolling_apply.py": 4, "m11_day_high_take_profit_fill.py": 5,
    "m12_week_vol_scaled_size.py": 4,
}
for k, v in L.items():
    (D / "leaky" / k).write_text(f"# leak: {LEAK_LINES[k] + 1}\n" + v, encoding="utf-8", newline="\n")
for k, v in C.items():
    (D / "clean" / k).write_text("# expect: none\n" + v, encoding="utf-8", newline="\n")
print(len(L), len(C))
