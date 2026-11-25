# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler().fit(X_train)
X_train_s, X_test_s = sc.transform(X_train), sc.transform(X_test)
