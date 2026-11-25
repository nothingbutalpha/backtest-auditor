# expect: LA10|LA01:high
import pandas as pd

df = pd.read_csv("spy.csv", index_col="date", parse_dates=True)
df["fwd ret"] = df["close"].pct_change().shift(-1)
df["signal"] = df.eval("`fwd ret` > 0")
print((df["signal"] * df["fwd ret"]).mean())
