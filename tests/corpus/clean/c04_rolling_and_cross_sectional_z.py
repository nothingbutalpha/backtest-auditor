# expect: none
import pandas as pd

panel = pd.read_parquet("panel.parquet")  # columns: date, ticker, close, book_to_market
panel["value_z"] = panel.groupby("date")["book_to_market"].transform(lambda s: (s - s.mean()) / s.std())
panel["rank"] = panel.groupby("date")["value_z"].rank(pct=True)
spread = pd.read_csv("spread.csv", parse_dates=["date"], index_col="date")["spread"]
z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
z_expanding = (spread - spread.expanding(60).mean()) / spread.expanding(60).std()
