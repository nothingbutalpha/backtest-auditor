# expect: none
# LA05 class: numpy arrays with a positional split.
import numpy as np
from sklearn.linear_model import Lasso

Xa = np.load("X.npy")
ya = np.load("y.npy")
k = int(0.75 * len(Xa))
las = Lasso(alpha=1e-3).fit(Xa[:k], ya[:k])
print(las.score(Xa[k:], ya[k:]))
