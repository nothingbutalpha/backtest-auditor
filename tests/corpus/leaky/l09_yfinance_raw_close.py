# expect: DQ02:medium, DQ01:info
import yfinance as yf

data = yf.download("NVDA", start="2015-01-01", auto_adjust=False)
daily = data["Close"].pct_change()
print(daily.min())  # shows a fake -75% day on the 2021 4:1 split
