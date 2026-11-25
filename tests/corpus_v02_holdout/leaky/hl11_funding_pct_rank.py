# expect: LA09
# Funding-rate percentile is computed with rank(pct=True) over the whole series.
import pandas as pd

perp = pd.read_csv("btc_perp_8h.csv", index_col="ts", parse_dates=True)
perp["ret"] = perp["mark"].pct_change()
perp["fr_pct"] = perp["funding"].rank(pct=True)

perp["side"] = 0
perp.loc[perp["fr_pct"] > 0.95, "side"] = -1
perp.loc[perp["fr_pct"] < 0.05, "side"] = 1
perp["pos"] = perp["side"].shift(1).fillna(0)
perp["pnl"] = perp["pos"] * perp["ret"] - 0.0004 * perp["pos"].diff().abs().fillna(0)
print("total", perp["pnl"].sum(), "trades", (perp["pos"].diff() != 0).sum())
