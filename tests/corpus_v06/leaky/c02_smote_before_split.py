# expect: LA06:high@8
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

df = pd.read_parquet("default_events.parquet").sort_index()
X, y = df.drop(columns="defaulted_12m"), df["defaulted_12m"]
X_bal, y_bal = SMOTE(random_state=1).fit_resample(X, y)
n = int(len(X_bal) * 0.7)
clf = RandomForestClassifier(n_estimators=300).fit(X_bal[:n], y_bal[:n])
print("test acc", clf.score(X_bal[n:], y_bal[n:]))
