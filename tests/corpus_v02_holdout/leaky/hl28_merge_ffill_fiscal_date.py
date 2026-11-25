# expect: LA04
# Income statements are joined on fiscalDateEnding and forward-filled onto daily bars.
import json

import pandas as pd

with open("av_income_ibm.json") as fh:
    inc = pd.DataFrame(json.load(fh)["quarterlyReports"])
inc["fiscalDateEnding"] = pd.to_datetime(inc["fiscalDateEnding"])
inc["netIncome"] = inc["netIncome"].astype(float)
inc["ni_growth"] = inc.sort_values("fiscalDateEnding")["netIncome"].pct_change(4)

daily = pd.read_csv("ibm_daily.csv", parse_dates=["date"]).sort_values("date")
daily = daily.merge(inc[["fiscalDateEnding", "ni_growth"]], left_on="date", right_on="fiscalDateEnding", how="left")
daily["ni_growth"] = daily["ni_growth"].ffill()

daily["pos"] = (daily["ni_growth"] > 0.1).astype(float).shift(1).fillna(0)
daily["ret"] = daily["close"].pct_change()
daily["net"] = daily["pos"] * daily["ret"] - 0.0005 * daily["pos"].diff().abs().fillna(0)
print("Total:", (1 + daily["net"].fillna(0)).prod() - 1)
