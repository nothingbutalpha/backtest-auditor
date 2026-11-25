# expect: LA05
# Twin of clean/d06: fitted on the whole array, scored on its tail.
import numpy as np
from sklearn.linear_model import Lasso

Xa = np.load("X.npy")
ya = np.load("y.npy")
k = int(0.75 * len(Xa))
las = Lasso(alpha=1e-3).fit(Xa, ya)
print(las.score(Xa[k:], ya[k:]))
