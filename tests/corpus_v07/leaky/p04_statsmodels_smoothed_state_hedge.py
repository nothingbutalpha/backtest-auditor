# expect: LA02@9
import numpy as np
import pandas as pd
import statsmodels.api as sm

d = pd.read_parquet("gld_gdx.parquet")
y, x = np.log(d["GDX"]), np.log(d["GLD"])
fit = sm.tsa.UnobservedComponents(y, level="llevel", exog=x).fit(disp=False)
hedge = pd.Series(fit.smoothed_state[1], index=y.index)
spread = y - hedge * x
pos = -np.sign(spread - spread.rolling(20).mean())
print((pos.shift(1) * spread.diff()).sum())
