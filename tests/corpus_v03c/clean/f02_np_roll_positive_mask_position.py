# expect: none
# Twin of leaky/f02: a 3-bar-old reference price, first elements masked.
import numpy as np

px = np.load("px.npy")
ret = np.r_[0.0, px[1:] / px[:-1] - 1]
ref = np.roll(px, 3)
ref[:3] = np.nan
cond = px < ref
pos = np.zeros_like(px)
pos[cond] = 1.0
pnl = pos[:-1] * ret[1:] - 0.0005
