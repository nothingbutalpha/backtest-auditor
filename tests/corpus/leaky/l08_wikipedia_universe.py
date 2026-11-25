# expect: SB01:high, DQ01:info
import pandas as pd
import yfinance as yf

table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
tickers = table["Symbol"].str.replace(".", "-").tolist()
data = yf.download(tickers, start="2005-01-01", end="2024-12-31")["Close"]
momentum = data.pct_change(252).shift(21)
print(momentum.tail())
