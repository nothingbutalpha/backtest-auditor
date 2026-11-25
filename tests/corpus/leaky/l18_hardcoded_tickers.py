# expect: SB01:possible
import pandas as pd

UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "LLY", "JPM", "V", "COST"]
closes = pd.read_parquet("closes.parquet")[UNIVERSE]
momentum = closes.pct_change(252).shift(21)
