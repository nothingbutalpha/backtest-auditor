import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["r"] = df["Close"].pct_change()
df["target"] = df["r"].shift(-1)
df = df.dropna()
out = []
for start in range(0, len(df) - 300, 50):
    tr = df.iloc[start:start + 250]
    te = df.iloc[start + 251:start + 300]
    sc = StandardScaler().fit(tr[["r"]])
    m = Ridge().fit(sc.transform(tr[["r"]]), tr["target"])
    out.append(m.predict(sc.transform(te[["r"]])))
