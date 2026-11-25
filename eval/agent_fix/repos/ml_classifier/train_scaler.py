import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X = np.load("features.npy")
y = np.load("labels.npy")
split = int(len(X) * 0.8)

X = StandardScaler().fit_transform(X)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print("out-of-sample accuracy", model.score(X_test, y_test))
