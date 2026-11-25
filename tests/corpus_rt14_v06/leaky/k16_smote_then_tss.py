# leak: 7
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("events.csv")
Xb, yb = ADASYN().fit_resample(df[["f1", "f2"]], df["crash"])
Xa, Xb2, ya, yb2 = train_test_split(Xb, yb, shuffle=False)
print(LogisticRegression().fit(Xa, ya).score(Xb2, yb2))
