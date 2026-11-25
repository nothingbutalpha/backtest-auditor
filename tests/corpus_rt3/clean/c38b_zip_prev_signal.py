# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
closes = [float(x) for x in open("close.txt")]
signal = [0.0] + [1.0 if b > a else -1.0 for a, b in zip(closes[:-1], closes[1:])]
rets = [0.0] + [b / a - 1 for a, b in zip(closes[:-1], closes[1:])]
pnl = [s * r - 0.0005 for s, r in zip(signal[:-1], rets[1:])]
print(sum(pnl) / len(pnl))
