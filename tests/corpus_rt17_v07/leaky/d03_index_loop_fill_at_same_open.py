# leak: 8
import pandas as pd
q = pd.read_csv("msft.csv", parse_dates=["date"])
ma = q["close"].rolling(50).mean()
cash, held, cost = 1.0, False, 0.0
for i in range(50, len(q)):
    if not held and q["close"].iat[i] > ma.iat[i]:
        cost, held = q["open"].iat[i], True
    elif held and q["close"].iat[i] < ma.iat[i]:
        cash *= q["open"].iat[i] / cost
        held = False
print(cash)
