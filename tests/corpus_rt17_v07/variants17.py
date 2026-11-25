import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); C = Path(sys.argv[2])
from nba_auditor.lint import check_code
def show(tag, src):
    fs = [f"{f.rule_id}:{f.severity}@{f.line}" for f in check_code(src).findings if f.severity in ("high","medium")]
    print(f"{tag:<44} {fs}")
def rd(n): return (C/"leaky"/f"{n}.py").read_text()
s = rd("e01_walk_forward_no_gap_purge_zero_const")
show("e01 as-is", s)
show("e01 PURGE->OFFSET", s.replace("PURGE", "OFFSET"))
show("e01 iloc[:k]", s.replace("PURGE = 0\n", "").replace(": k - PURGE]", ":k]"))
s = rd("b03_aliased_coint_listcomp_pairs")
show("b03 as-is", s)
show("b03 unaliased", s.replace("coint as eg_test", "coint").replace("eg_test(", "coint("))
loop = s.replace("pairs = [(u, v) for u, v in itertools.combinations(closes.columns, 2) if eg_test(closes[u], closes[v])[1] < 0.05]",
 "pairs = []\nfor u, v in itertools.combinations(closes.columns, 2):\n    if eg_test(closes[u], closes[v])[1] < 0.05:\n        pairs.append((u, v))")
show("b03 for-loop aliased", loop)
show("b03 for-loop unaliased", loop.replace("coint as eg_test", "coint").replace("eg_test(", "coint("))
s = rd("c01_tail_notna_universe")
show("c01 as-is", s)
show("c01 iloc[-1].notna()", s.replace("px.tail(1).notna().all()", "px.iloc[-1].notna()"))
s = rd("c02_no_missing_count_mask"); show("c02 as-is", s)
show("c02 dropna(axis=1)", s.replace("closes.loc[:, closes.isna().sum() == 0]", "closes.dropna(axis=1)"))
show("c02 notna().all()", s.replace("closes.isna().sum() == 0", "closes.notna().all()"))
s = rd("c04_transpose_dropna"); show("c04 dropna(axis=1)", s.replace("wide.T.dropna().T", "wide.dropna(axis=1)"))
s = rd("a01_lambda_normalizer_signal"); show("a01 as-is", s)
show("a01 def", s.replace("norm = lambda s: (s - s.mean()) / s.std()", "def norm(s):\n    return (s - s.mean()) / s.std()"))
show("a01 inline", s.replace("score = norm(mom)", "score = (mom - mom.mean()) / mom.std()"))
s = rd("a03_percentile_bands_tuple_from_helper"); show("a03 as-is", s)
show("a03 inline", s.replace("lower, upper = entry_bands(spread.values)", "lower, upper = np.nanpercentile(spread.values, [10, 90])"))
show("a03 quantile helper", s.replace("np.nanpercentile(x, [10, 90])", "x.quantile(0.1), x.quantile(0.9)").replace("spread.values", "spread"))
s = rd("b01_spearman_dict_pick_then_tss"); show("b01 as-is", s)
show("b01 corrwith", s.replace('ic = {c: spearmanr(data[c], label, nan_policy="omit")[0] for c in data.columns}\nkeep = sorted(ic, key=lambda c: abs(ic[c]), reverse=True)[:5]',
 'ic = data.corrwith(label)\nkeep = ic.abs().nlargest(5).index'))
s = rd("d01_vbt_close_kw_is_open"); show("d01 as-is", s)
show("d01 price=open", s.replace('close=bars["open"], entries=buy, exits=sell', 'bars["close"], entries=buy, exits=sell, price=bars["open"]'))
s = rd("b04_best_lookback_by_full_sharpe_reported"); show("b04 as-is", s)
