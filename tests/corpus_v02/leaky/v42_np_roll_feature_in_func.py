# expect: LA10:medium
import numpy as np


def features(close):
    nxt = np.roll(close, shift=-1)
    up = (nxt > close).astype(int)
    return np.column_stack([close, up])
