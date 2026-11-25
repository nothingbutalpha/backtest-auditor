# expect: none
import pandas as pd

def xs_z(s):
    return (s - s.mean()) / s.std()

panel = pd.read_parquet("eq_panel.parquet")
panel["value_z"] = panel.groupby("date")["book_to_price"].transform(xs_z)
panel["w"] = panel["value_z"].clip(-3, 3)
pnl = (panel["w"] * panel["ret_next"]).groupby(panel["date"]).sum()
print(pnl.mean())
