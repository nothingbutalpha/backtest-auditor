# expect: none
import pandas as pd

panel = pd.read_parquet("panel.parquet")
panel["fwd_ret"] = panel.groupby("ticker")["close"].pct_change().shift(-1)
frame = pd.DataFrame({
    "m": panel.groupby("date")["fwd_ret"].mean(),
    "b": pd.read_csv("fund.csv", index_col="date", parse_dates=True)["nav"].pct_change().shift(-1),
})
up = frame.query("m > 0")
up_capture = up["b"].mean() / up["m"].mean()
print(up_capture)
