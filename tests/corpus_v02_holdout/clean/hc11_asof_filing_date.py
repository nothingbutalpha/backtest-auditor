# expect: none
# Looks like a period-end fundamentals merge, but merge_asof keys on the filing date.
import pandas as pd

prices = pd.read_csv("prices.csv", parse_dates=["date"]).sort_values("date")
fund = pd.read_csv("fundamentals.csv", parse_dates=["period_end", "filed"])
fund["available"] = fund["filed"] + pd.Timedelta(days=1)
fund = fund.sort_values("available")

df = pd.merge_asof(
    prices,
    fund[["ticker", "available", "eps_ttm"]],
    left_on="date",
    right_on="available",
    by="ticker",
)
df["ey"] = df["eps_ttm"] / df["close"]
df["long"] = (df.groupby("date")["ey"].rank(pct=True) > 0.8).astype(float)
df["pos"] = df.groupby("ticker")["long"].shift(1).fillna(0)
df["ret"] = df.groupby("ticker")["close"].pct_change()
df["net"] = df["pos"] * df["ret"] - 0.001 * df.groupby("ticker")["pos"].diff().abs().fillna(0)
print(df.groupby("date")["net"].mean().describe())
