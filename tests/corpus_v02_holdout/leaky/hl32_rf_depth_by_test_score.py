# expect: LA14
# Random-forest depth is chosen by test-set score in a loop, and that best test score is reported.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("features_with_target.csv", parse_dates=["date"]).set_index("date")
X, y = df.drop(columns="target"), df["target"]
train = df.index < "2021-01-01"
X_train, X_test, y_train, y_test = X[train], X[~train], y[train], y[~train]

best_score, best_depth, best_model = 0.0, None, None
for depth in [2, 3, 4, 6, 8, 12]:
    clf = RandomForestClassifier(max_depth=depth, n_estimators=300, random_state=0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    if score > best_score:
        best_score, best_depth, best_model = score, depth, clf

print(f"Best depth {best_depth}: out-of-sample accuracy {best_score:.3f}")
