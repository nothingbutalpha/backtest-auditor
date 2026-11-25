# expect: LA15:medium
import pandas as pd
sep = pd.read_csv("sep.csv", parse_dates=["date"], index_col="date")
sf1 = pd.read_csv("sf1_arq.csv", parse_dates=["calendardate"], index_col="calendardate")
daily = sep.join(sf1[["eps", "bvps"]], how="left").ffill()
daily["pe"] = daily["close"] / daily["eps"]
