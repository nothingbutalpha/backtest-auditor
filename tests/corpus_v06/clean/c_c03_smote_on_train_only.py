# expect: none
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

df = pd.read_parquet("default_events.parquet").sort_index()
X, y = df.drop(columns="defaulted_12m"), df["defaulted_12m"]
n = int(len(X) * 0.7)
X_tr, y_tr = SMOTE(random_state=1).fit_resample(X.iloc[:n], y.iloc[:n])
clf = RandomForestClassifier(n_estimators=300).fit(X_tr, y_tr)
print("test acc", clf.score(X.iloc[n + 12:], y.iloc[n + 12:]))
