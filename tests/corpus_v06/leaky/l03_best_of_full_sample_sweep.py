# expect: LA11:medium@12
import numpy as np
import pandas as pd

px = pd.read_parquet("ief.parquet")["close"]
ret = px.pct_change()
scores = {}
for lb in range(20, 260, 20):
    pos = np.sign(px.pct_change(lb)).shift(1)
    pnl = pos * ret
    scores[lb] = pnl.mean() / pnl.std() * np.sqrt(252)
best = max(scores, key=scores.get)
print("lookback", best, "Sharpe", round(scores[best], 2))
