# expect: none
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("events.csv", parse_dates=["date"]).sort_values("date")
cut = df["date"] < "2020-01-01"
X_fit, y_fit = ADASYN().fit_resample(df.loc[cut, ["f1", "f2"]], df.loc[cut, "crash"])
print(LogisticRegression().fit(X_fit, y_fit).score(df.loc[~cut, ["f1", "f2"]], df.loc[~cut, "crash"]))
