# -*- coding: latin-1 -*-
# Bénéfice par action (BPA) trimestriel
import pandas as pd

bpa = pd.read_csv("bpa.csv", parse_dates=["date"], index_col="date").sort_index()
bpa["bpa"] = bpa["bpa"].bfill()
bpa.to_csv("bpa_daily.csv")
