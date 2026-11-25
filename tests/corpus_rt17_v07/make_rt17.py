"""Red-team round 17 (lint v0.7) fresh probe files, written by the red team from leak mechanics.

Run: python make_rt17.py  (writes leaky/ and clean/ next to this file; score with ../corpus_rt14_v06/score_rt14.py).
A line ending in `#@` is a leak line: the marker is stripped and the line number goes into `# leak: N` on line 1.
Clean files get `# expect: none`. Plain-style correct files are prefixed `q`, near-miss look-alikes `n`.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

LEAKY = {
    # --- interprocedural flow -------------------------------------------------------------------------------
    "a01_lambda_normalizer_signal": '''
import numpy as np
import pandas as pd
px = pd.read_csv("etf_closes.csv", index_col=0, parse_dates=True)["XLE"]
norm = lambda s: (s - s.mean()) / s.std()  #@
mom = px.pct_change(60)
score = norm(mom)
pos = np.sign(score).shift(1)
pnl = pos * px.pct_change()
print(pnl.sum())
''',
    "a02_two_level_helper_feature_then_split": '''
import pandas as pd
from sklearn.linear_model import Ridge
def _standardise(col):
    centre = col.mean()  #@
    spread = col.std()
    return (col - centre) / spread
def build_features(frame):
    return pd.DataFrame({"vol": _standardise(frame["vol20"]), "mom": _standardise(frame["mom60"])})
d = pd.read_parquet("panel_one_name.parquet")
X = build_features(d)
y = d["close"].pct_change().shift(-1)
cut = int(len(d) * 0.7)
mdl = Ridge().fit(X.iloc[:cut], y.iloc[:cut])
print(mdl.score(X.iloc[cut:].dropna(), y.iloc[cut:].dropna()))
''',
    "a03_percentile_bands_tuple_from_helper": '''
import numpy as np
import pandas as pd
def entry_bands(x):
    lo, hi = np.nanpercentile(x, [10, 90])  #@
    return lo, hi
a = pd.read_csv("ko.csv", index_col=0, parse_dates=True)["close"]
b = pd.read_csv("pep.csv", index_col=0, parse_dates=True)["close"]
spread = np.log(a) - np.log(b)
lower, upper = entry_bands(spread.values)
side = pd.Series(np.where(spread < lower, 1, np.where(spread > upper, -1, 0)), index=spread.index)
pnl = side.shift(1) * spread.diff()
print(pnl.cumsum().iloc[-1])
''',
    "a04_staticmethod_minmax_before_split": '''
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
class Prep:
    @staticmethod
    def squash(frame):
        return (frame - frame.min()) / (frame.max() - frame.min())  #@
raw = pd.read_parquet("btc_features.parquet")
target = (raw["close"].shift(-1) > raw["close"]).astype(int)
X = Prep.squash(raw[["rsi", "atr", "obv"]])
n = int(len(X) * 0.8)
clf = GradientBoostingClassifier().fit(X.iloc[:n], target.iloc[:n])
print((clf.predict(X.iloc[n:]) == target.iloc[n:]).mean())
''',
    "a05_horizon_helper_used_as_feature": '''
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
def horizon_change(p, h):
    return p.pct_change(h).shift(-h)
df = pd.read_csv("cl_daily.csv", index_col=0, parse_dates=True)
df["chg5"] = horizon_change(df["close"], 5)  #@
df["vol"] = df["close"].pct_change().rolling(20).std()
df["y"] = df["close"].pct_change().shift(-1)
df = df.dropna()
tr, te = df.loc[:"2019"], df.loc["2020":]
rf = RandomForestRegressor().fit(tr[["chg5", "vol"]], tr["y"])
print(rf.score(te[["chg5", "vol"]], te["y"]))
''',
    # --- selection on the full sample -----------------------------------------------------------------------
    "b01_spearman_dict_pick_then_tss": '''
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
data = pd.read_parquet("factors.parquet")
label = data.pop("ret_next")
ic = {c: spearmanr(data[c], label, nan_policy="omit")[0] for c in data.columns}
keep = sorted(ic, key=lambda c: abs(ic[c]), reverse=True)[:5]  #@
print(cross_val_score(LinearRegression(), data[keep], label, cv=TimeSeriesSplit(5)).mean())
''',
    "b02_lasso_coefs_pick_then_walk_forward": '''
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
tbl = pd.read_parquet("macro_panel.parquet")
y = tbl["spx_ret_1m"].shift(-1).dropna()
X = tbl.loc[y.index].drop(columns="spx_ret_1m")
lasso = LassoCV(cv=5).fit(X, y)
chosen = X.columns[np.abs(lasso.coef_) > 1e-6]  #@
preds = []
for k in range(120, len(X)):
    lr = LinearRegression().fit(X[chosen].iloc[:k], y.iloc[:k])
    preds.append(lr.predict(X[chosen].iloc[[k]])[0])
print(np.corrcoef(preds, y.iloc[120:])[0, 1])
''',
    "b03_aliased_coint_listcomp_pairs": '''
import itertools
import pandas as pd
from statsmodels.tsa.stattools import coint as eg_test
closes = pd.read_parquet("banks_close.parquet")
pairs = [(u, v) for u, v in itertools.combinations(closes.columns, 2) if eg_test(closes[u], closes[v])[1] < 0.05]  #@
book = []
for u, v in pairs:
    s = closes[u] - closes[v]
    z = (s - s.rolling(60).mean()) / s.rolling(60).std()
    book.append((-z.clip(-2, 2)).shift(1) * s.diff())
print(pd.concat(book, axis=1).sum(axis=1).cumsum().iloc[-1])
''',
    "b04_best_lookback_by_full_sharpe_reported": '''
import numpy as np
import pandas as pd
px = pd.read_csv("gld.csv", index_col=0, parse_dates=True)["close"]
r = px.pct_change()
def strat(n):
    return np.sign(px - px.rolling(n).mean()).shift(1) * r
def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252)
best_n = max(range(10, 210, 10), key=lambda n: sharpe(strat(n)))  #@
print("Sharpe", sharpe(strat(best_n)))
''',
    # --- survivorship ---------------------------------------------------------------------------------------
    "c01_tail_notna_universe": '''
import pandas as pd
px = pd.read_parquet("russell_closes.parquet")
alive = px.columns[px.tail(1).notna().all()]  #@
rets = px[alive].pct_change()
w = rets.rolling(126).mean().rank(axis=1, pct=True).gt(0.9).astype(float)
w = w.div(w.sum(axis=1), axis=0).shift(1)
print((w * rets).sum(axis=1).add(1).prod())
''',
    "c02_no_missing_count_mask": '''
import pandas as pd
closes = pd.read_csv("smallcaps.csv", index_col=0, parse_dates=True)
closes = closes.loc[:, closes.isna().sum() == 0]  #@
r = closes.pct_change()
pos = (r.rolling(5).sum().rank(axis=1) <= 20).astype(float).shift(1)
print((pos * r).mean(axis=1).cumsum().iloc[-1])
''',
    "c03_latest_snapshot_by_max_date": '''
import pandas as pd
hist = pd.read_parquet("index_membership_snapshots.parquet")
newest = hist[hist["as_of"] == hist["as_of"].max()]  #@
names = newest["ticker"].unique()
px = pd.read_parquet("all_closes.parquet")[names]
rets = px.pct_change()
sig = rets.rolling(252).sum().shift(21)
long = sig.rank(axis=1, ascending=False) <= 30
print(rets[long.shift(1).fillna(False)].mean(axis=1).cumsum().iloc[-1])
''',
    "c04_transpose_dropna": '''
import pandas as pd
wide = pd.read_parquet("eu_stocks.parquet")
wide = wide.T.dropna().T  #@
ret = wide.pct_change()
pos = ret.rolling(20).std().rank(axis=1, pct=True).lt(0.2).astype(float).shift(1)
print((pos * ret).sum(axis=1).cumsum().iloc[-1])
''',
    # --- same-bar open fills --------------------------------------------------------------------------------
    "d01_vbt_close_kw_is_open": '''
import pandas as pd
import vectorbt as vbt
bars = pd.read_parquet("qqq.parquet")
fast = bars["close"].ewm(span=12).mean()
slow = bars["close"].ewm(span=26).mean()
buy = fast > slow
sell = fast < slow
pf = vbt.Portfolio.from_signals(close=bars["open"], entries=buy, exits=sell)  #@
print(pf.total_return())
''',
    "d02_open_to_close_on_close_signal": '''
import pandas as pd
k = pd.read_csv("nq_daily.csv", index_col=0, parse_dates=True)
up = k["close"] > k["close"].rolling(10).mean()
intraday = k["close"] / k["open"] - 1
gain = intraday.where(up, 0.0)  #@
print(gain.add(1).prod())
''',
    "d03_index_loop_fill_at_same_open": '''
import pandas as pd
q = pd.read_csv("msft.csv", parse_dates=["date"])
ma = q["close"].rolling(50).mean()
cash, held, cost = 1.0, False, 0.0
for i in range(50, len(q)):
    if not held and q["close"].iat[i] > ma.iat[i]:
        cost, held = q["open"].iat[i], True  #@
    elif held and q["close"].iat[i] < ma.iat[i]:
        cash *= q["open"].iat[i] / cost
        held = False
print(cash)
''',
    # --- CV / walk-forward ----------------------------------------------------------------------------------
    "e01_walk_forward_no_gap_purge_zero_const": '''
import pandas as pd
from sklearn.linear_model import LogisticRegression
H = 10
PURGE = 0
f = pd.read_parquet("es_feats.parquet")
f["lbl"] = (f["close"].shift(-H) > f["close"]).astype(int)
f = f.dropna()
hits = []
for k in range(500, len(f) - 50, 50):
    a = f.iloc[: k - PURGE]  #@
    b = f.iloc[k : k + 50]
    m = LogisticRegression().fit(a[["x1", "x2"]], a["lbl"])
    hits.append((m.predict(b[["x1", "x2"]]) == b["lbl"]).mean())
print(sum(hits) / len(hits))
''',
}

CLEAN = {
    # --- plain-style correct files (must get no high/medium) -------------------------------------------------
    "q01_rolling_zscore_helper_signal": '''
import numpy as np
import pandas as pd
def zscore(s, n=60):
    m = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return (s - m) / sd
px = pd.read_csv("tlt.csv", index_col=0, parse_dates=True)["close"]
z = zscore(px)
pos = -np.sign(z).shift(1)
pnl = pos * px.pct_change()
print(pnl.sum())
''',
    "q02_expanding_helper_feature_then_split": '''
import pandas as pd
from sklearn.linear_model import Ridge
def expz(col):
    return (col - col.expanding(60).mean()) / col.expanding(60).std()
d = pd.read_parquet("panel_one_name.parquet")
X = pd.DataFrame({"vol": expz(d["vol20"]), "mom": expz(d["mom60"])})
y = d["close"].pct_change().shift(-1)
data = pd.concat([X, y.rename("y")], axis=1).dropna()
cut = int(len(data) * 0.7)
mdl = Ridge().fit(data.iloc[:cut][["vol", "mom"]], data.iloc[:cut]["y"])
print(mdl.score(data.iloc[cut:][["vol", "mom"]], data.iloc[cut:]["y"]))
''',
    "q03_scaler_helper_on_early_years": '''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
def fit_scaler(frame):
    return StandardScaler().fit(frame)
raw = pd.read_parquet("btc_features.parquet")
target = (raw["close"].shift(-1) > raw["close"]).astype(int)
cols = ["rsi", "atr", "obv"]
early = raw.index < "2021-01-01"
sc = fit_scaler(raw.loc[early, cols])
clf = LogisticRegression().fit(sc.transform(raw.loc[early, cols]), target[early])
print((clf.predict(sc.transform(raw.loc[~early, cols])) == target[~early]).mean())
''',
    "q04_sma_cross_next_open_vbt": '''
import pandas as pd
import vectorbt as vbt
bars = pd.read_parquet("qqq.parquet")
fast = bars["close"].ewm(span=12).mean()
slow = bars["close"].ewm(span=26).mean()
buy = (fast > slow).shift(1, fill_value=False)
sell = (fast < slow).shift(1, fill_value=False)
pf = vbt.Portfolio.from_signals(bars["close"], entries=buy, exits=sell, price=bars["open"])
print(pf.total_return())
''',
    "q05_open_to_close_on_yesterday_signal": '''
import pandas as pd
k = pd.read_csv("nq_daily.csv", index_col=0, parse_dates=True)
up = (k["close"] > k["close"].rolling(10).mean()).shift(1, fill_value=False)
intraday = k["close"] / k["open"] - 1
gain = intraday.where(up, 0.0)
print(gain.add(1).prod())
''',
    "q06_index_loop_next_open_fill": '''
import pandas as pd
q = pd.read_csv("msft.csv", parse_dates=["date"])
ma = q["close"].rolling(50).mean()
cash, held, cost = 1.0, False, 0.0
for i in range(51, len(q)):
    if not held and q["close"].iat[i - 1] > ma.iat[i - 1]:
        cost, held = q["open"].iat[i], True
    elif held and q["close"].iat[i - 1] < ma.iat[i - 1]:
        cash *= q["open"].iat[i] / cost
        held = False
print(cash)
''',
    "q07_ledoit_wolf_per_rebalance": '''
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
rets = pd.read_parquet("sector_etf_rets.parquet")
out = []
for t in range(250, len(rets), 21):
    cov = LedoitWolf().fit(rets.iloc[t - 250:t]).covariance_
    inv = np.linalg.pinv(cov)
    w = inv.sum(axis=1) / inv.sum()
    out.append(rets.iloc[t:t + 21] @ w)
print(pd.concat(out).add(1).prod())
''',
    "q08_coverage_report_only": '''
import pandas as pd
px = pd.read_parquet("russell_closes.parquet")
alive = px.columns[px.tail(1).notna().all()]
print(f"{len(alive)} of {px.shape[1]} names still trade today")
print(px.notna().sum(axis=1).describe())
''',
    "q09_drawdown_running_peak_numpy": '''
import numpy as np
import pandas as pd
px = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
peak = np.maximum.accumulate(px.values)
dd = pd.Series(px.values / peak - 1, index=px.index)
pos = (dd > -0.1).astype(float).shift(1)
print((pos * px.pct_change()).sum())
''',
    "q10_tss_with_horizon_gap": '''
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
HORIZON = 21
f = pd.read_parquet("es_feats.parquet")
f["y"] = f["close"].pct_change(HORIZON).shift(-HORIZON)
f = f.dropna()
cv = TimeSeriesSplit(n_splits=5, gap=HORIZON)
print(cross_val_score(Ridge(), f[["x1", "x2"]], f["y"], cv=cv).mean())
''',
    # --- near-miss look-alikes ------------------------------------------------------------------------------
    "n01_point_in_time_members_helper": '''
import pandas as pd
mem = pd.read_csv("sp500_membership.csv", parse_dates=["start", "end"])
px = pd.read_parquet("all_closes.parquet")
def members_on(day):
    live = mem[(mem["start"] <= day) & (mem["end"].isna() | (mem["end"] > day))]
    return [t for t in live["ticker"] if t in px.columns]
rets = px.pct_change()
book = []
for day in rets.index[252::21]:
    names = members_on(day)
    look = rets.loc[:day, names].iloc[-252:-21].sum()
    top = look.nlargest(50).index
    nxt = rets.loc[rets.index > day, top].iloc[:21]
    book.append(nxt.mean(axis=1))
print(pd.concat(book).add(1).prod())
''',
    "n02_sector_reference_for_exposure_report": '''
import pandas as pd
ref = pd.read_csv("security_master.csv")
sector_of = ref.groupby("ticker")["sector"].last()
px = pd.read_parquet("all_closes.parquet")
rets = px.pct_change()
w = rets.rolling(126).mean().rank(axis=1, pct=True).gt(0.9).astype(float).shift(1)
pnl = (w * rets).sum(axis=1)
exposure = w.T.groupby(sector_of).sum().T.mean()
print(pnl.cumsum().iloc[-1])
print(exposure.sort_values())
''',
    "n03_correlation_pick_on_early_rows": '''
import pandas as pd
from sklearn.linear_model import LinearRegression
data = pd.read_parquet("factors.parquet")
label = data.pop("ret_next")
early, late = data.index < "2016-01-01", data.index >= "2016-01-01"
ic = data[early].corrwith(label[early], method="spearman")
keep = ic.abs().nlargest(5).index
lr = LinearRegression().fit(data.loc[early, keep], label[early])
print(lr.score(data.loc[late, keep], label[late]))
''',
    "n04_rolling_formation_coint": '''
import itertools
import pandas as pd
from statsmodels.tsa.stattools import coint
closes = pd.read_parquet("banks_close.parquet")
legs = []
for t in range(252, len(closes) - 63, 63):
    look = closes.iloc[t - 252:t]
    trade = closes.iloc[t:t + 63]
    for u, v in itertools.combinations(closes.columns, 2):
        if coint(look[u], look[v])[1] < 0.05:
            s = trade[u] - trade[v]
            mu, sd = (look[u] - look[v]).mean(), (look[u] - look[v]).std()
            legs.append((-((s - mu) / sd).clip(-2, 2)).shift(1) * s.diff())
print(pd.concat(legs).sum())
''',
    "n05_recent_window_dropna_in_loop": '''
import pandas as pd
wide = pd.read_parquet("eu_stocks.parquet")
ret = wide.pct_change()
pnl = []
for i in range(126, len(ret) - 21, 21):
    recent = ret.iloc[i - 126:i].dropna(axis=1)
    pick = recent.std().nsmallest(30).index
    pnl.append(ret.iloc[i:i + 21][pick].mean(axis=1))
print(pd.concat(pnl).cumsum().iloc[-1])
''',
    "n06_walk_forward_gap_without_name": '''
import pandas as pd
from sklearn.linear_model import LogisticRegression
H = 10
f = pd.read_parquet("es_feats.parquet")
f["lbl"] = (f["close"].shift(-H) > f["close"]).astype(int)
f = f.dropna()
hits = []
for k in range(500, len(f) - 50, 50):
    a = f.iloc[: k - H]
    b = f.iloc[k : k + 50]
    m = LogisticRegression().fit(a[["x1", "x2"]], a["lbl"])
    hits.append((m.predict(b[["x1", "x2"]]) == b["lbl"]).mean())
print(sum(hits) / len(hits))
''',
    "n07_lookback_picked_on_first_half": '''
import numpy as np
import pandas as pd
px = pd.read_csv("gld.csv", index_col=0, parse_dates=True)["close"]
r = px.pct_change()
half = len(px) // 2
def strat(n):
    return np.sign(px - px.rolling(n).mean()).shift(1) * r
def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252)
best_n = max(range(10, 210, 10), key=lambda n: sharpe(strat(n).iloc[:half]))
print("Out-of-sample Sharpe", sharpe(strat(best_n).iloc[half:]))
''',
    "n08_liquidity_filter_lagged_per_date": '''
import pandas as pd
px = pd.read_parquet("all_closes.parquet")
vol = pd.read_parquet("all_volumes.parquet")
adv = (px * vol).rolling(20).mean().shift(1)
rets = px.pct_change()
ok = adv > 5e6
score = rets.rolling(60).sum().shift(1).where(ok)
w = score.rank(axis=1, pct=True).gt(0.8).astype(float)
w = w.div(w.sum(axis=1), axis=0)
print((w * rets).sum(axis=1).cumsum().iloc[-1])
''',
    "n09_full_history_minmax_for_chart_only": '''
import pandas as pd
import matplotlib.pyplot as plt
px = pd.read_csv("ko.csv", index_col=0, parse_dates=True)["close"]
def rescale(s):
    return (s - s.min()) / (s.max() - s.min())
sig = (px > px.rolling(100).mean()).astype(float).shift(1)
pnl = sig * px.pct_change()
plt.plot(rescale(px), label="price (rescaled)")
plt.plot(rescale(pnl.cumsum()), label="strategy (rescaled)")
plt.legend()
plt.savefig("chart.png")
''',
    "n10_shuffled_kfold_cross_section_one_date": '''
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
snap = pd.read_parquet("fundamentals_2019q4.parquet")
snap["fwd_beat"] = (snap["next_q_eps"] > snap["consensus"]).astype(int)
X = snap[["accruals", "roe", "pe"]]
cv = KFold(n_splits=5, shuffle=True, random_state=0)
print(cross_val_score(RandomForestClassifier(), X, snap["fwd_beat"], cv=cv).mean())
''',
    # --- second wave: traps for the v0.7 rules -------------------------------------------------------------
    "q11_tearsheet_helper_on_pnl": '''
import numpy as np
import pandas as pd
def tearsheet(r):
    mu, sd = r.mean(), r.std()
    return {"ann_ret": mu * 252, "ann_vol": sd * np.sqrt(252), "worst_z": ((r - mu) / sd).min()}
px = pd.read_csv("iwm.csv", index_col=0, parse_dates=True)["close"]
pos = (px > px.rolling(200).mean()).astype(float).shift(1)
pnl = pos * px.pct_change()
print(tearsheet(pnl.dropna()))
''',
    "q12_cross_sectional_demean_helper": '''
import pandas as pd
def demean_rows(frame):
    return frame.sub(frame.mean(axis=1), axis=0)
px = pd.read_parquet("sector_etfs.parquet")
r = px.pct_change()
w = -demean_rows(r.rolling(5).sum()).shift(1)
w = w.div(w.abs().sum(axis=1), axis=0)
print((w * r).sum(axis=1).cumsum().iloc[-1])
''',
    "q13_strategy_class_with_constants": '''
import numpy as np
import pandas as pd
class Breakout:
    def __init__(self, lookback):
        self.lookback = lookback
        self.ann = np.sqrt(252)
    def signal(self, px):
        hi = px.rolling(self.lookback).max().shift(1)
        return (px > hi).astype(float)
px = pd.read_csv("hg.csv", index_col=0, parse_dates=True)["close"]
s = Breakout(55)
pnl = s.signal(px).shift(1) * px.pct_change()
print(pnl.mean() / pnl.std() * s.ann)
''',
    "q14_helper_lags_its_own_signal_open_fill": '''
import pandas as pd
import vectorbt as vbt
def entries_for(c):
    return (c > c.rolling(50).mean()).shift(1, fill_value=False)
def exits_for(c):
    return (c < c.rolling(50).mean()).shift(1, fill_value=False)
df = pd.read_parquet("iwm.parquet")
pf = vbt.Portfolio.from_signals(df["close"], entries=entries_for(df["close"]), exits=exits_for(df["close"]),
                                price=df["open"])
print(pf.sharpe_ratio())
''',
    "q15_helper_signal_next_day_intraday": '''
import pandas as pd
def rule(d):
    return (d["close"] > d["close"].rolling(20).mean()).astype(float)
d = pd.read_csv("dia.csv", index_col=0, parse_dates=True)
pos = rule(d).shift(1)
pnl = pos * (d["close"] / d["open"] - 1)
print(pnl.sum())
''',
    "q16_three_way_split_plain_names": '''
import pandas as pd
from sklearn.linear_model import Ridge
d = pd.read_parquet("features.parquet")
early, mid, late = d.loc[:"2015"], d.loc["2016":"2018"], d.loc["2019":]
cols = ["a", "b", "c"]
results = {}
for alpha in (0.1, 1.0, 10.0):
    m = Ridge(alpha=alpha).fit(early[cols], early["y"])
    results[alpha] = m.score(mid[cols], mid["y"])
pick = max(results, key=results.get)
final = Ridge(alpha=pick).fit(pd.concat([early, mid])[cols], pd.concat([early, mid])["y"])
print(pick, final.score(late[cols], late["y"]))
''',
    "q17_cv_int_folds_one_day_label": '''
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
d = pd.read_parquet("spy_feats.parquet")
d["next_up"] = (d["close"].shift(-1) > d["close"]).astype(int)
d = d.dropna()
print(cross_val_score(LogisticRegression(), d[["rsi", "mom"]], d["next_up"], cv=5).mean())
''',
    "n11_live_watchlist_current_members": '''
import pandas as pd
mem = pd.read_csv("index_membership.csv", parse_dates=["added", "removed"])
current = mem.query("removed.isnull()", engine="python")["ticker"].tolist()
px = pd.read_parquet("all_closes.parquet")[current]
last = px.iloc[-1]
mom = px.iloc[-1] / px.iloc[-253] - 1
orders = mom.nlargest(20).index
print("buy at tomorrow's open:", list(orders), last[orders].round(2).to_dict())
''',
    "n12_attribute_table_dropna_columns": '''
import pandas as pd
meta = pd.read_csv("security_attributes.csv", index_col="ticker").dropna(axis=1)
px = pd.read_parquet("all_closes.parquet")
rets = px.pct_change()
big = meta.index[meta["float_shares"] > 1e8]
w = rets[big.intersection(rets.columns)].rolling(20).mean().shift(1).gt(0).astype(float)
print((w * rets[w.columns]).mean(axis=1).sum())
''',
    "n13_mutual_info_inside_tss_folds": '''
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
d = pd.read_parquet("factors.parquet")
y = d.pop("ret_next")
scores = []
for a, b in TimeSeriesSplit(5, gap=1).split(d):
    mi = pd.Series(mutual_info_regression(d.iloc[a], y.iloc[a]), index=d.columns)
    cols = mi.nlargest(5).index
    lr = LinearRegression().fit(d.iloc[a][cols], y.iloc[a])
    scores.append(lr.score(d.iloc[b][cols], y.iloc[b]))
print(np.mean(scores))
''',
    "n14_rolling_pca_residual_reversion": '''
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
r = pd.read_parquet("sector_etf_rets.parquet")
pnl = []
for t in range(252, len(r) - 1):
    past = r.iloc[t - 252:t]
    pca = PCA(n_components=3).fit(past)
    today = r.iloc[[t]]
    resid = today - pca.inverse_transform(pca.transform(today))
    w = -np.sign(resid.iloc[0]) / r.shape[1]
    pnl.append(float(r.iloc[t + 1] @ w))
print(sum(pnl))
''',
    "n15_point_in_time_cap_rank_loop": '''
import pandas as pd
cap = pd.read_parquet("mktcap.parquet")
px = pd.read_parquet("all_closes.parquet").dropna(axis=1, how="all")
r = px.pct_change()
pnl = []
for d in r.index[21::21]:
    known = cap.loc[:d].iloc[-2]
    top = known.dropna().nlargest(100).index
    hold = r.loc[r.index > d, top].iloc[:21]
    pnl.append(hold.mean(axis=1))
print(pd.concat(pnl).sum())
''',
    # --- third wave -------------------------------------------------------------------------------------------
    "q18_helper_signal_lagged_on_next_line_open": '''
import pandas as pd
import vectorbt as vbt
def trend(c):
    return c.rolling(20).mean() > c.rolling(100).mean()
df = pd.read_parquet("xlk.parquet")
on = trend(df["close"])
on = on.shift(1, fill_value=False)
pf = vbt.Portfolio.from_signals(df["close"], entries=on, exits=~on, price=df["open"])
print(pf.total_return())
''',
    "q19_helper_signal_filled_at_close": '''
import pandas as pd
import vectorbt as vbt
def trend(c):
    return c.rolling(20).mean() > c.rolling(100).mean()
df = pd.read_parquet("xlk.parquet")
on = trend(df["close"])
pf = vbt.Portfolio.from_signals(df["close"], entries=on.shift(1, fill_value=False), exits=(~on).shift(1, fill_value=False))
print(pf.total_return())
''',
    "q20_fit_on_first_rows_helper_neutral_names": '''
import pandas as pd
from sklearn.linear_model import LinearRegression
def centre(x, ref):
    return (x - ref.mean()) / ref.std()
d = pd.read_parquet("features.parquet")
first = d.iloc[:2000]
rest = d.iloc[2000:]
cols = ["a", "b", "c"]
m = LinearRegression().fit(centre(first[cols], first[cols]), first["y"])
print(m.score(centre(rest[cols], first[cols]), rest["y"]))
''',
    "q21_percentile_rank_trailing_apply": '''
import pandas as pd
from scipy.stats import percentileofscore
vix = pd.read_csv("vix.csv", index_col=0, parse_dates=True)["close"]
pct = vix.rolling(252).apply(lambda w: percentileofscore(w[:-1], w[-1]), raw=True)
spy = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
pos = (pct > 80).astype(float).shift(1)
print((pos * spy.pct_change()).sum())
''',
    "q22_live_threshold_universe_today": '''
import pandas as pd
cap = pd.read_parquet("mktcap.parquet")
last = cap.iloc[-1]
today = last[last > 2e9].index
px = pd.read_parquet("all_closes.parquet")[today]
mom = px.iloc[-1] / px.iloc[-127] - 1
print(mom.nlargest(25))
''',
    "q23_month_end_rebalance_vol_scaled": '''
import numpy as np
import pandas as pd
px = pd.read_csv("agg.csv", index_col=0, parse_dates=True)["close"]
r = px.pct_change()
vol = r.rolling(60).std() * np.sqrt(252)
lev = (0.10 / vol).clip(upper=2.0)
lev_m = lev.resample("M").last().reindex(r.index, method="ffill").shift(1)
print((lev_m * r).add(1).prod())
''',
    "q24_groupby_ticker_trailing_z": '''
import pandas as pd
panel = pd.read_parquet("panel_long.parquet")
g = panel.groupby("ticker")["ret"]
panel["z"] = (panel["ret"] - g.transform(lambda s: s.rolling(60).mean())) / g.transform(lambda s: s.rolling(60).std())
panel["w"] = -panel.groupby("ticker")["z"].shift(1)
print((panel["w"] * panel["ret"]).groupby(panel["date"]).sum().cumsum().iloc[-1])
''',
}


def write(kind: str, name: str, body: str) -> None:
    lines = body.strip("\n").splitlines()
    leak = [i + 2 for i, ln in enumerate(lines) if ln.rstrip().endswith("#@")]
    lines = [ln.rstrip()[:-2].rstrip() if ln.rstrip().endswith("#@") else ln for ln in lines]
    head = f"# leak: {', '.join(map(str, leak))}" if kind == "leaky" else "# expect: none"
    if kind == "leaky":
        assert leak, name
    (HERE / kind).mkdir(exist_ok=True)
    (HERE / kind / f"{name}.py").write_text(head + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


for n, b in LEAKY.items():
    write("leaky", n, b)
for n, b in CLEAN.items():
    write("clean", n, b)
