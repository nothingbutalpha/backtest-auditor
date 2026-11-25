"""Backtest Auditor template: vectorbt (one asset, daily bars, long or flat).

Timing conventions (keep them when you change the rule):
  1. Entry and exit signals for bar t use only data up to and including bar t's close.
  2. Orders fill at bar t+1's open: signals are shifted one bar and from_signals gets price=open.
  3. Fees and slippage are parameters. Never set them to zero by default.
  4. Train and test are split by date. Choose parameters on the train period only, and count every
     variant you tried (rules, parameters, assets) as n_trials when you audit the results.

Input: a CSV you supply with columns date, open, high, low, close (split- and dividend-adjusted, all the
same way). Output: net returns per bar in returns.csv, ready for the auditor's audit_returns or
audit_backtest. Results are hypothetical, simulated past performance, not a prediction or guarantee.
"""
import pandas as pd
import vectorbt as vbt

DATA_CSV = "prices.csv"      # your data: date, open, high, low, close
TRAIN_END = "2018-12-31"     # last date of the train period; everything after it is the test period
FAST, SLOW = 20, 100         # example parameters: replace with your own rule's
FEES = 0.0005                # commission per trade as a fraction of the trade value (0.0005 = 5 bps)
SLIPPAGE = 0.0005            # slippage per trade as a fraction of the price (0.0005 = 5 bps)


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df.sort_index()  # oldest first, so every rolling window looks back


def backtest(df: pd.DataFrame) -> vbt.Portfolio:
    # --- Your rule goes here. Use data up to bar t's close only: trailing windows, no shift(-k),
    # no center=True, no backward fill, no statistics over all rows. Keep it inside this function,
    # where the auditor's check can follow the signals to the fill.
    # Example rule, to be replaced by yours: enter when the fast moving average crosses above the slow
    # one, exit when it crosses below.
    fast = df["close"].rolling(FAST).mean()
    slow = df["close"].rolling(SLOW).mean()
    above = fast > slow
    entries = above & ~above.shift(1, fill_value=False)    # known at bar t's close
    exits = ~above & above.shift(1, fill_value=False)

    # --- Timing and costs: keep these lines as they are.
    return vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entries.shift(1, fill_value=False),         # act one bar later...
        exits=exits.shift(1, fill_value=False),
        price=df["open"],                                   # ...at that bar's open
        fees=FEES,
        slippage=SLIPPAGE,
        freq="1D",
    )


def main() -> None:
    df = load_prices(DATA_CSV)
    print("HYPOTHETICAL simulated results on past data. Not a prediction or guarantee of future results.")
    # Each period is simulated on its own, so the test period starts flat and uses nothing from the train run.
    # The slow average needs SLOW bars to warm up in each period.
    for label, part in (("train", df.loc[:TRAIN_END]), ("test (look at this once, after the rule is final)",
                                                         df.loc[df.index > pd.Timestamp(TRAIN_END)])):
        pf = backtest(part)
        print(f"{label}: total return {pf.total_return():.2%}, max drawdown {pf.max_drawdown():.2%}, "
              f"trades {pf.trades.count()}")
    backtest(df).returns().rename("return").rename_axis("date").to_csv("returns.csv")


if __name__ == "__main__":
    main()
