# expect: none
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("features.csv", index_col=0, parse_dates=True)
a, b = int(len(df) * 0.6), int(len(df) * 0.8)
X_train, X_val, X_test = df.iloc[:a, :-1], df.iloc[a:b, :-1], df.iloc[b:, :-1]
y_train, y_val, y_test = df.iloc[:a, -1], df.iloc[a:b, -1], df.iloc[b:, -1]
best_score, best_depth = -1, None
for depth in [2, 3, 4, 6, 8, 12]:
    clf = RandomForestClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
    acc = clf.score(X_val, y_val)
    if acc > best_score:
        best_score, best_depth = acc, depth
final = RandomForestClassifier(max_depth=best_depth, random_state=0).fit(X_train, y_train)
print(f"best depth {best_depth}: test accuracy {final.score(X_test, y_test):.3f}")
