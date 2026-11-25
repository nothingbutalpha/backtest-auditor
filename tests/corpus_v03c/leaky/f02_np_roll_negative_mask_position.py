# expect: LA01|LA10:high
# A 3-bar-ahead reference price decides a position written through a mask.
import numpy as np

px = np.load("px.npy")
ret = np.r_[0.0, px[1:] / px[:-1] - 1]
ref = np.roll(px, -3)
cond = px < ref
pos = np.zeros_like(px)
pos[cond] = 1.0
pnl = pos[:-1] * ret[1:] - 0.0005
