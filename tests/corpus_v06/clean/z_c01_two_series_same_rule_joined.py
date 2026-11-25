# expect: none
import pandas as pd

q = pd.read_parquet("eth_quotes_250ms.parquet")
q["imb"] = (q["bid_sz"] - q["ask_sz"]) / (q["bid_sz"] + q["ask_sz"])
imb_1s = q["imb"].resample("1s").mean()
mid_1s = ((q["bid"] + q["ask"]) / 2).resample("1s").last()
feat = pd.DataFrame({"imb_5s": imb_1s.rolling(5).mean(), "imb_30s": imb_1s.rolling(30).mean()})
nxt = mid_1s.pct_change().shift(-1).rename("next_ret")
print(feat.join(nxt).corr()["next_ret"])
