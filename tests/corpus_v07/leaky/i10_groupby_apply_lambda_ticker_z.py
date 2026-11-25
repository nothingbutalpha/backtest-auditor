# expect: LA09@6
import pandas as pd

def add_z(panel, col):
    return panel.groupby("symbol", group_keys=False).apply(
        lambda g: g.assign(z=(g[col] - g[col].mean()) / g[col].std())
    )

panel = pd.read_parquet("borrow_fee_panel.parquet").pipe(add_z, "fee")
panel["crowded"] = panel["z"] > 2
panel["fwd_5"] = panel.groupby("symbol")["close"].shift(-5) / panel["close"] - 1
print(panel.groupby("crowded")["fwd_5"].mean())
