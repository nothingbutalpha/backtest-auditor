"""Red-team round 16 files (lint v0.6 B1/B2 re-check). Written fresh; not derived from dev sets, rt14 or rt15.
Leaky files: first line `# leak: N` (1-based, header included). Clean files: `# expect: none`."""
import sys
from pathlib import Path

D = Path(sys.argv[1])
(D / "leaky").mkdir(parents=True, exist_ok=True)
(D / "clean").mkdir(parents=True, exist_ok=True)
L, C = {}, {}

# ---------- B1: aggregates of PnL vs aggregates of prices/signals ----------
L["p01_day_loss_kill_switch.py"] = (6, '''import pandas as pd
m = pd.read_parquet("es_1m.parquet")
m["ret"] = m["close"].pct_change()
m["signal"] = (m["close"] > m["close"].rolling(30).mean()).astype(int).shift(1)
m["pnl"] = m["signal"] * m["ret"]
m["day_total"] = m.groupby(m.index.date)["pnl"].transform("sum")
m["signal"] = m["signal"].where(m["day_total"] > -0.02, 0)
m["pnl"] = m["signal"] * m["ret"]
print(m["pnl"].sum())
''')
L["p02_week_low_stop_loop.py"] = (5, '''import pandas as pd
h = pd.read_parquet("eth_1h.parquet")
h["entry_px"] = h["open"]
wk = h.index.to_period("W")
h["stop_px"] = h.groupby(wk)["low"].transform("min") * 1.001
pnl = []
for _, row in h.iterrows():
    exit_px = row["stop_px"] if row["low"] <= row["stop_px"] else row["close"]
    pnl.append(exit_px / row["entry_px"] - 1)
print(sum(pnl))
''')
L["p03_month_mean_reversion_exit.py"] = (5, '''import numpy as np
import pandas as pd
d = pd.read_csv("gold_daily.csv", index_col=0, parse_dates=True)
mon = pd.Grouper(freq="M")
d["fair"] = d.groupby(mon)["close"].transform("median")
d["exit_px"] = np.minimum(d["high"], d["fair"])
d["trade_ret"] = d["exit_px"] / d["open"] - 1
print(d["trade_ret"].mean())
''')
L["p04_day_vwap_target_weights.py"] = (5, '''import pandas as pd
b = pd.read_parquet("spy_5m.parquet")
b["pv"] = b["close"] * b["volume"]
day = b.index.floor("D")
b["day_px"] = b.groupby(day)["close"].transform("mean")
b["weight"] = (b["day_px"] / b["close"] - 1).clip(-1, 1)
b["strat"] = b["weight"].shift(1) * b["close"].pct_change()
print(b["strat"].sum())
''')
C["r01_equity_monthly_max_report.py"] = '''import pandas as pd
e = pd.read_csv("equity_curve.csv", index_col=0, parse_dates=True)
e["equity_month_peak"] = e.groupby(e.index.to_period("M"))["equity"].transform("max")
e["month_dd"] = e["equity"] / e["equity_month_peak"] - 1
print(e["month_dd"].groupby(e.index.to_period("M")).min())
'''
C["r02_strategy_ret_daily_vol_table.py"] = '''import pandas as pd
m = pd.read_parquet("es_1m.parquet")
m["position"] = (m["close"] > m["close"].rolling(60).mean()).astype(int).shift(1)
m["strategy_ret"] = m["position"] * m["close"].pct_change()
m["day_strategy_vol"] = m.groupby(m.index.date)["strategy_ret"].transform("std")
print(m["day_strategy_vol"].resample("D").last().describe())
'''
C["r03_expanding_sharpe_report.py"] = '''import numpy as np
import pandas as pd
d = pd.read_csv("strat.csv", index_col=0, parse_dates=True)
d["pnl"] = d["position"].shift(1) * d["ret"]
d["running_sharpe"] = d["pnl"].expanding(60).mean() / d["pnl"].expanding(60).std() * np.sqrt(252)
d["roll_dd"] = d["pnl"].cumsum() - d["pnl"].cumsum().rolling(126, min_periods=1).max()
print(d[["running_sharpe", "roll_dd"]].tail())
'''
C["r04_prev_day_range_stop.py"] = '''import pandas as pd
m = pd.read_parquet("nq_1m.parquet")
day = m.index.normalize()
rng = (m.groupby(day)["high"].max() - m.groupby(day)["low"].min()).shift(1)
m["stop_dist"] = rng.reindex(day).to_numpy() * 0.5
m["signal"] = (m["close"] > m["open"]).astype(int).shift(1)
m["pnl"] = m["signal"] * m["close"].pct_change()
print(m["pnl"].sum(), m["stop_dist"].mean())
'''

# ---------- B2: backtrader indicator aliases, feed lists and dicts ----------
L["p05_bt_macd_line_attr_ahead.py"] = (8, '''import backtrader as bt
class Macd(bt.Strategy):
    def __init__(self):
        m = bt.ind.MACD(self.data.close)
        self.macd = m.macd
        self.sig = m.signal
    def next(self):
        if self.macd[1] > self.sig[0]:
            self.buy()
''')
L["p06_bt_indicator_dict_ahead.py"] = (7, '''import backtrader as bt
class Multi(bt.Strategy):
    def __init__(self):
        self.smas = {d: bt.ind.SMA(d.close, period=50) for d in self.datas}
    def next(self):
        for d in self.datas:
            if d.close[0] > self.smas[d][1]:
                self.order_target_percent(d, 0.1)
''')
L["p07_bt_from_import_alias_ahead.py"] = (8, '''import backtrader as bt
from backtrader.indicators import BollingerBands as BB
class Band(bt.Strategy):
    def __init__(self):
        self.bb = BB(self.data, period=20)
    def next(self):
        top = self.bb.lines.top
        if self.data.close[0] < top[1]:
            self.sell()
''')
L["p08_bt_line_arith_ahead.py"] = (7, '''import backtrader as bt
class Spread(bt.Strategy):
    def __init__(self):
        self.spread = self.data0.close - 1.5 * self.data1.close
    def next(self):
        z = self.spread
        if z[2] > 0:
            self.sell(data=self.data0)
''')
C["r05_bt_feed_dict_and_names.py"] = '''import backtrader as bt
class Rank(bt.Strategy):
    def __init__(self):
        self.by_name = {d._name: d for d in self.datas}
        self.names = sorted(self.by_name)
        self.rocs = {d: bt.ind.ROC(d.close, period=20) for d in self.datas}
    def next(self):
        second = self.by_name[self.names[1]]
        if self.rocs[second][0] > self.rocs[self.datas[0]][0]:
            self.order_target_percent(second, 0.5)
'''
C["r06_bt_get_history_and_levels.py"] = '''import backtrader as bt
class Breakout(bt.Strategy):
    def next(self):
        hist = self.data.close.get(size=5)
        levels = (self.data.close[0] * 0.98, self.data.close[0] * 1.02)
        if len(hist) == 5 and hist[1] < levels[1] and self.data.close[0] > max(hist[:-1]):
            self.buy()
'''
C["r07_bt_closes_list_previous.py"] = '''import backtrader as bt
import backtrader.indicators as btind
class Pairs(bt.Strategy):
    def __init__(self):
        self.closes = [d.close for d in self.datas]
        self.atr = btind.ATR(self.datas[1], period=14)
    def next(self):
        if self.closes[1][0] > self.closes[1][-1] + self.atr[-1]:
            self.sell(data=self.datas[1])
'''
C["r08_bt_custom_indicator_lines.py"] = '''import backtrader as bt
class Mid(bt.Indicator):
    lines = ("mid",)
    params = (("windows", (10, 20)),)
    def __init__(self):
        self.lines.mid = (bt.ind.Highest(self.data.high, period=self.p.windows[1]) + bt.ind.Lowest(self.data.low, period=self.p.windows[1])) / 2
class UseMid(bt.Strategy):
    def __init__(self):
        self.mid = Mid(self.data)
    def next(self):
        if self.data.close[0] > self.mid[-1]:
            self.buy()
'''

for k, (ln, v) in L.items():
    (D / "leaky" / k).write_text(f"# leak: {ln + 1}\n" + v, encoding="utf-8", newline="\n")
for k, v in C.items():
    (D / "clean" / k).write_text("# expect: none\n" + v, encoding="utf-8", newline="\n")
print(len(L), len(C))
