"""Red-team rt17 adversarial 200 KB shapes. Usage: stress17.py <auditor dir> [shape]. Prints seconds per shape."""
import sys, time
sys.path.insert(0, sys.argv[1])
from nba_auditor.lint import check_code
HDR = ("import numpy as np\nimport pandas as pd\nimport vectorbt as vbt\nfrom sklearn.linear_model import Ridge\n"
       "from sklearn.model_selection import cross_val_score\npx = pd.read_parquet('close.parquet')\nret = px.pct_change()\n"
       "df = pd.read_parquet('d.parquet')\n")
SHAPES = {
  "helper chain": ("def h0(s):\n    return (s - s.mean()) / s.std()\n", "def h{k1}(s):\n    return h{k}(s) + 1\n", "x = h{k}(px)\npnl = np.sign(x).shift(1) * ret\n"),
  "mutual recursion": ("", "def f{k}(s):\n    return g{k}(s - s.mean())\ndef g{k}(s):\n    return f{k}(s) if len(s) else s\nx{k} = f{k}(px)\npnl{k} = np.sign(x{k}).shift(1) * ret\n", ""),
  "fan-in helper tuple": ("def bands(x):\n    lo, hi = x.quantile(0.1), x.quantile(0.9)\n    return lo, hi, (x - x.mean()) / x.std()\n",
                          "a{k}, b{k}, z{k} = bands(px)\nsig{k} = np.where(px < a{k}, 1, 0)\npnl{k} = pd.DataFrame(sig{k}).shift(1) * ret\n", ""),
  "helpers calling all previous": ("def h0(s):\n    return s - s.mean()\n", "def h{k1}(s):\n    return h{k}(s) + h0(s)\n", "x = h{k}(px)\npnl = np.sign(x).shift(1) * ret\n"),
  "walk-forward loops": ("", "for k{k} in range(500, len(df), 50):\n    a{k} = df.iloc[:k{k}]\n    b{k} = df.iloc[k{k}:k{k} + 50]\n    m{k} = Ridge().fit(a{k}[['x']], a{k}['y_fwd_5'])\n    print(m{k}.score(b{k}[['x']], b{k}['y_fwd_5']))\n", ""),
  "close funcs + vbt": ("", "def r{k}(c):\n    return (c.rolling(20).mean() > c.rolling(50).mean()).astype(float)\ns{k} = r{k}(df['close'])\npf{k} = vbt.Portfolio.from_signals(df['close'], entries=s{k} > 0, exits=s{k} < 1, price=df['open'])\n", ""),
  "one big dict of frames": ("feats = {}\n", "feats['f{k}'] = px['c{k}'].rolling(5).mean()\n", "X = pd.concat(feats, axis=1)\nm = Ridge().fit(X, px['y'])\n"),
  "class attr web": ("", "class P{k}:\n    def __init__(self, d):\n        self.mu = d.mean()\n        self.sd = d.std()\n    def z(self, d):\n        return (d - self.mu) / self.sd\n    def sig(self, d):\n        return np.sign(self.z(d)).shift(1)\np{k} = P{k}(px)\npnl{k} = p{k}.sig(px) * ret\n", ""),
  "selection comps": ("", "ic{k} = {{c: px[c].corr(ret['y']) for c in px.columns}}\nkeep{k} = sorted(ic{k}, key=ic{k}.get)[:5]\ns{k} = cross_val_score(Ridge(), px[keep{k}], ret['y'], cv=5)\n", ""),
  "survivor masks": ("", "u{k} = px.columns[px.iloc[-1].notna()]\nl{k} = px.iloc[-1]\nv{k} = l{k}[l{k} > {k}].index\nr{k} = px[u{k}.intersection(v{k})].pct_change()\nw{k} = r{k}.rolling(5).mean().shift(1)\nprint((w{k} * r{k}).sum().sum())\n", ""),
}
which = sys.argv[2:] or list(SHAPES)
for name in which:
    head, unit, tail = SHAPES[name]
    parts, k = [HDR, head], 0
    while sum(map(len, parts)) < 200_000 - 200:
        parts.append(unit.format(k=k, k1=k + 1)); k += 1
    parts.append(tail.format(k=k))
    src = "".join(parts)
    t = time.perf_counter()
    try:
        a = check_code(src); res = f"{len(a.findings)} findings"
    except Exception as e:
        res = f"EXC {type(e).__name__}: {str(e)[:80]}"
    print(f"{name:<30} {len(src)//1000:>4} KB {time.perf_counter() - t:7.2f} s  {res}", flush=True)
