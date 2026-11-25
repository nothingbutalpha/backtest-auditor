# expect: LA09:medium
import pandas as pd
df = pd.read_csv("es.csv", index_col=0, parse_dates=True)
df["trend"] = df["close"].pct_change(100)
df["weight"] = df["trend"] / df["trend"].abs().max()
df["strategy"] = df["weight"].shift(1) * df["close"].pct_change() - 0.0002 * df["weight"].diff().abs()
