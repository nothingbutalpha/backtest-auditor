# expect: LA09@5
import pandas as pd

panel = pd.read_parquet("crypto_panel.parquet").sort_index()   # index: (date, coin)
panel["funding_pct"] = panel.groupby(level="coin")["funding"].rank(pct=True)
panel["ret_next"] = panel.groupby(level="coin")["close"].pct_change().shift(-1)
shorts = panel[panel["funding_pct"] > 0.9]
print(shorts.groupby(level="date")["ret_next"].mean().mean())
