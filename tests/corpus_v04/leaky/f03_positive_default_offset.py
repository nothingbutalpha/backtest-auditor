# expect: LA13:high
import numpy as np


def breakout_flags(c, k=3):
    flags = np.zeros(len(c))
    for i in range(len(c) - k):
        if c[i + k] > c[i] * 1.02:
            flags[i] = 1
    return flags


c = np.load("closes.npy")
print(breakout_flags(c).sum())
