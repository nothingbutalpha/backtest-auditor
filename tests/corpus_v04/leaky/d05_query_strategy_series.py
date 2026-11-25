# expect: LA10
import pandas as pd

panel = pd.read_parquet("panel.parquet")
panel["fwd_ret"] = panel.groupby("ticker")["close"].pct_change().shift(-1)
daily = pd.DataFrame({
    "mkt_fwd": panel.groupby("date")["fwd_ret"].mean(),
    "spy_next": pd.read_csv("spy.csv", index_col="date", parse_dates=True)["close"].pct_change().shift(-1),
})
strat = daily.query("mkt_fwd > 0")["spy_next"]
print(strat.mean() * 252, strat.std() * 252 ** 0.5)
