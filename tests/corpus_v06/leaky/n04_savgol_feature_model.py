# expect: LA02:high@7
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression

df = pd.read_parquet("eurusd_1h.parquet")
df["smooth"] = savgol_filter(df["close"].to_numpy(), window_length=31, polyorder=2)
df["slope"] = df["smooth"].diff()
df["y"] = (df["close"].shift(-1) > df["close"]).astype(int)
data = df.dropna()
model = LogisticRegression().fit(data[["slope"]].iloc[:5000], data["y"].iloc[:5000])
print(model.score(data[["slope"]].iloc[5000:], data["y"].iloc[5000:]))
