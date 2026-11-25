# expect: LA04
# Quarterly fundamentals are merge_asof'd onto daily prices by period_end, before they were filed.
import pandas as pd

prices = pd.read_csv("prices.csv", parse_dates=["date"]).sort_values("date")
fund = pd.read_csv("fundamentals.csv", parse_dates=["period_end", "filed"])
fund = fund.sort_values("period_end")

df = pd.merge_asof(
    prices,
    fund[["ticker", "period_end", "eps_ttm", "bvps"]],
    left_on="date",
    right_on="period_end",
    by="ticker",
)
df["ey"] = df["eps_ttm"] / df["close"]
df["bm"] = df["bvps"] / df["close"]
df["value"] = df.groupby("date")["ey"].rank(pct=True) + df.groupby("date")["bm"].rank(pct=True)
df["long"] = (df.groupby("date")["value"].rank(pct=True) > 0.8).astype(float)
df["pos"] = df.groupby("ticker")["long"].shift(1).fillna(0)
df["ret"] = df.groupby("ticker")["close"].pct_change()
df["net"] = df["pos"] * df["ret"] - 0.001 * df.groupby("ticker")["pos"].diff().abs().fillna(0)
print(df.groupby("date")["net"].mean().describe())
