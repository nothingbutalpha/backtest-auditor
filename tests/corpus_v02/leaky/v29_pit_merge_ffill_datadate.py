# expect: LA15:medium
import pandas as pd
crsp = pd.read_csv("crsp_daily.csv", parse_dates=["date"])
comp = pd.read_csv("compustat_q.csv", parse_dates=["datadate"])
m = crsp.merge(comp, left_on=["permno", "date"], right_on=["permno", "datadate"], how="left")
m[["atq", "ltq"]] = m.groupby("permno")[["atq", "ltq"]].ffill()
m["lev"] = m["ltq"] / m["atq"]
