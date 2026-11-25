# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
closes = [float(x) for x in open("close.txt")]
signals = [1 if closes[i] > closes[i - 1] else 0 for i in range(1, len(closes))]
pnl = []
for sig, p0, p1 in zip(signals[:-1], closes[1:-1], closes[2:]):
    pnl.append(sig * (p1 / p0 - 1) - 0.0005 * sig)
print(sum(pnl))
