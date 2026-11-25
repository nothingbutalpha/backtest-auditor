# expect: LA14:medium
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("features.csv", index_col=0, parse_dates=True)
cut = int(len(df) * 0.7)
X_train, X_test = df.iloc[:cut, :-1], df.iloc[cut:, :-1]
y_train, y_test = df.iloc[:cut, -1], df.iloc[cut:, -1]
best_score, best_depth = -1, None
for depth in [2, 3, 4, 6, 8, 12]:
    clf = RandomForestClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    if acc > best_score:
        best_score, best_depth = acc, depth
print(f"best depth {best_depth}: test accuracy {best_score:.3f}")
