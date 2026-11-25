"""Static lookahead and bias lint for Python/pandas strategy code (`check_code`).

The lint parses the source with `ast` and runs three kinds of analysis:

1. **Call patterns**: e.g. `rolling(center=True)`, `bfill()`, `train_test_split` without
   `shuffle=False`, `pd.read_html(<wikipedia S&P 500 URL>)`.
2. **A light data-flow pass** over assignments in source order. Every assigned name or column
   (`x = ...`, `df['x'] = ...`, `df.x = ...`) gets a record: is it forward-looking (built with a
   negative shift or from a label), lagged (a positive `.shift`), derived from prices, a return, a
   signal? Rules such as "signal times same-bar return" (LA07) and "feature built from the label"
   (LA10) read those records.
3. **Name heuristics**: a column called `target` / `fwd_ret` / `y` is treated as a label, `signal` /
   `position` as a signal, `ret` / `returns` as a return, `strategy` / `pnl` as strategy PnL.

Precision matters more than recall: when a pattern is ambiguous the finding is reported with
severity "possible", and patterns that are usually legitimate (a negative shift that builds a
label, `signal * returns.shift(-1)`) are not reported.

Severities: "high" (very likely inflates results), "medium" (likely), "possible" (worth a look,
may be fine), "info" (context, no score penalty).
"""

from __future__ import annotations

import ast
import bisect
import functools
import heapq
import re
import threading
from dataclasses import asdict, dataclass, field

from .catalog import CHECKS, DISCLAIMER, SCORE_CAVEAT

SEVERITIES = ("high", "medium", "possible", "info")
PENALTY = {"high": 25, "medium": 10, "possible": 3, "info": 0}
MAX_PENALIZED_PER_RULE = 2  # one noisy rule can't zero the score on its own

# ---------------------------------------------------------------------------------------------
# Name heuristics
# ---------------------------------------------------------------------------------------------
LABEL_T = {"target", "targets", "label", "labels", "y", "fwd", "forward", "future", "fut", "next",
           "outcome", "ahead", "lead", "tomorrow"}
PRED_T = {"pred", "preds", "prediction", "predictions", "predicted", "hat", "yhat", "proba", "forecast"}
EVAL_T = {"hit", "hits", "correct", "err", "error", "errors", "acc", "accuracy", "ic", "corr", "resid", "auc", "roc",
          "brier", "logloss", "f1", "r2", "sharpe", "sortino", "calmar", "precision", "recall",
          "residual", "residuals", "mse", "mae", "rmse", "loss", "eval", "metric", "metrics", "summary", "stats",
          "report", "result", "results", "table", "describe", "analysis", "agg", "quintile", "quintiles", "decile",
          "deciles", "bucket", "buckets"}
PNL_T = {"strat", "strategy", "pnl", "net", "gross", "equity", "portfolio", "port", "perf", "performance", "backtest",
         "bt", "cum", "cumulative", "wealth", "nav"}
SIGNAL_T = {"signal", "signals", "sig", "position", "positions", "pos", "weight", "weights", "holding",
            "holdings", "entry", "entries", "exit", "exits", "long", "short", "buy", "sell", "trade",
            "trades", "regime", "score", "direction", "side", "allocation", "alloc", "exposure"} | PRED_T
RETURN_T = {"ret", "rets", "return", "returns", "pct", "chg", "change", "logret", "logrets", "rtn"}
FEATURE_T = {"feat", "feature", "features", "sma", "ema", "ma", "rsi", "macd", "mom", "momentum", "vol",
             "volatility", "zscore", "z", "x", "bb", "band", "atr", "indicator", "lag", "rank", "spread",
             "ratio", "avg", "mean", "std", "upper", "lower", "boll", "stoch", "adx", "cci", "obv"}
PRICE_T = {"close", "adj", "open", "high", "low", "price", "prices", "px", "vwap", "mid", "bid", "ask",
           "last", "ohlc", "c"}
SPLIT_T = {"train", "test", "valid", "val", "validation", "oos", "insample", "is", "tr", "te", "holdout", "dev", "hold"}
COST_T = {"cost", "costs", "fee", "fees", "commission", "commissions", "slippage", "bps", "tcost",
          "tcosts", "spread", "transaction", "transactions", "tc"}
MODULE_ALIASES = {"np", "pd", "numpy", "pandas", "math", "ta", "talib", "vbt", "yf", "sm", "scipy",
                  "stats", "plt", "sns", "bt", "self", "sklearn", "lgb", "xgb", "torch", "tf", "os",
                  "sys", "json", "datetime", "dt", "itertools", "random", "print", "len", "range", "abs",
                  "min", "max", "sum", "list", "dict", "float", "int", "str", "zip", "enumerate", "round"}
PANDAS_METHODS = {"shift", "rolling", "mean", "std", "sum", "pct_change", "diff", "fillna", "dropna",
                  "values", "index", "columns", "loc", "iloc", "apply", "where", "mask", "clip", "abs",
                  "cumsum", "cumprod", "ewm", "expanding", "rank", "copy", "astype", "to_numpy", "max",
                  "min", "median", "quantile", "groupby", "transform", "resample", "bfill", "ffill", "T",
                  "shape", "dtypes", "head", "tail", "iat", "at", "str", "dt"}

PREPROCESSORS = {"StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer",
                 "QuantileTransformer", "PowerTransformer", "PCA", "KernelPCA", "TruncatedSVD",
                 "KBinsDiscretizer", "SimpleImputer", "KNNImputer", "IterativeImputer", "SelectKBest",
                 "VarianceThreshold", "FactorAnalysis", "FastICA"}
TODAY_LIST_FUNCS = {"tickers_sp500", "tickers_nasdaq", "tickers_dow", "tickers_ftse100", "tickers_ftse250",
                    "tickers_other", "get_sp500_tickers", "sp500_tickers", "get_sp500", "get_nasdaq100",
                    "nasdaq100_tickers", "get_sp500_symbols"}
CURRENT_LIST_URL_HINTS = ("wikipedia.org", "slickcharts.com", "stockanalysis.com/list",
                          "datahub.io/core/s-and-p-500", "s-and-p-500-companies")
INTRADAY_RULE = re.compile(r"^\s*\d*\s*(min|T|H|h|S|s|ms|L|us|U|N|ns|BH)\s*$")
TICKER_RE = re.compile(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$")
TS_HINTS = ("shift(", "rolling(", "pct_change", "datetime", "to_datetime", "DatetimeIndex", "resample",
            "yfinance", "Close", "close", "Date", "date", "returns", "ohlc", "ewm(")


_DATE_RE = re.compile(r"(date|^dt|dt$|month|year|yyyy|yymm|quarter|qtr|week|period|time|day$|^d$|fyear|fqtr)")


def _date_like(name: str | None) -> bool:
    """A date or period key: date, yyyymm, jdate, fyear, month, quarter, rebalance_dt..."""
    return bool(name) and (bool(_tokens(name) & DATE_KEY_T) or bool(_DATE_RE.search(name.lower())))


@functools.lru_cache(maxsize=65536)
def _tokens(name: str) -> frozenset[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return frozenset(t for t in re.split(r"[^A-Za-z]+", s.lower()) if t)


EXEC_T = {"price", "px", "fill", "fills", "exec", "execution", "vwap"}
# Price fields known only at the end of bar t (close, high, low...), as opposed to the open.
POST_OPEN_T = {"close", "closes", "adj", "high", "highs", "low", "lows", "price", "prices", "px", "vwap", "last", "c",
               "hi", "lo", "mid", "volume", "volumes"}  # v0.4: a bar's volume is complete only at its close
VOLUME_T = {"volume", "volumes"}
CLOSE_T = {"close", "closes", "adj", "price", "prices", "px", "last", "c"}  # the closing print of bar t
OPEN_T = {"open", "opens", "o", "opn", "opening"}
ENTRY_T = {"entry", "entries", "buy", "bought", "fill", "basis", "enter"}
# LA10 (v0.3): aggregations that turn a forward column into an outcome table, calls that put a series back onto
# rows, bucket names, and model-input calls.
AGG_T = {"mean", "sum", "median", "std", "var", "count", "agg", "aggregate", "prod", "nunique", "sem", "min", "max",
         "first", "last", "size", "describe", "quantile", "skew", "kurt"}
MAPBACK_T = {"map", "merge", "join", "reindex", "replace", "transform", "combine_first", "update", "align",
             "reindex_like", "assign", "apply", "where", "mask"}
BUCKET_T = {"decile", "deciles", "quintile", "quintiles", "bucket", "buckets", "tercile", "terciles", "quartile",
            "quartiles", "quantile", "quantiles", "q", "bin", "bins", "group", "portfolio", "port"}
FIT_ATTRS = {"fit", "fit_transform", "fit_predict", "partial_fit", "predict", "predict_proba", "decision_function"}
# Transform calls that, written inside a product factor around forward data, make the factor a position (v0.3 rt7).
# np.where counts only its condition. clip/abs are left out: a winsorized forward return is still a return.
INLINE_TRANSFORM_T = {"sign", "rolling", "ewm", "expanding", "rank", "qcut", "cut", "tanh", "argsort", "cumsum",
                      "cumprod", "nlargest", "nsmallest"}
# Signal tokens that also name return legs or fills (long_leg, short_leg, trades, entry): too ambiguous on their own
AMBIG_SIGNAL_T = {"long", "short", "trade", "trades", "buy", "sell", "entry", "entries", "exit", "exits", "side"}
# Calls that turn a return into something else (a sizing rule, a sign, a rank, a statistic): not a return any more
TRANSFORM_FUNCS = {"rolling", "ewm", "expanding", "sign", "rank", "clip", "where", "qcut", "cut", "std", "var", "abs",
                   "argsort", "tanh", "sqrt", "log", "log1p", "exp", "cumsum", "cumprod", "nlargest", "nsmallest",
                   "idxmax", "idxmin", "quantile", "sem", "skew", "corr", "cov", "zscore", "apply", "transform",
                   "max", "min", "minimum", "maximum", "clip_lower", "clip_upper", "sharpe"}
TEST_SET_T = {"test", "te", "oos", "holdout", "tst"}
# Names that say "this series is the strategy's return" (not perf/port/net, which also name benchmark analyses)
STRAT_T = {"strat", "strategy", "strategies", "pnl", "equity", "backtest", "bt", "wealth"}
VAL_SET_T = {"val", "valid", "validation", "dev", "va", "eval"}


@functools.lru_cache(maxsize=65536)
def classify(name: str | None) -> str:
    """Role of a variable/column judged from its name."""
    if not name:
        return "unknown"
    t = _tokens(name)
    if t & EXEC_T or (t & {"open"} and t & {"next", "entry", "exit"}):
        return "exec"  # an execution price such as entry_price = open.shift(-1): a fill, not a feature
    if t & PRED_T:
        return "signal"
    if t & EVAL_T:
        return "eval"
    if t & {"feat", "feature", "features"}:
        return "feature"
    if t & LABEL_T:
        return "label"
    if t & PNL_T:
        return "pnl"
    if t & RETURN_T and t & {"trade", "trades", "pos", "position", "long", "short"}:
        return "return"
    if t & SIGNAL_T:
        return "signal"
    if t & RETURN_T:
        return "return"
    if t & FEATURE_T:
        return "feature"
    return "unknown"


def _is_price_name(name: str) -> bool:
    return bool(_tokens(name) & PRICE_T)


_TIME_WORDS = {"next", "prev", "previous", "prior", "yesterday", "tomorrow", "lag", "lagged", "lead", "fwd", "future"}


def _is_open_name(name: str | None) -> bool:
    """`open`, `Open`, `opens`, `open_`: bar t's opening price (not `next_open`, not `open_close_ret`)."""
    if not name:
        return False
    t = _tokens(name)
    return bool(t & OPEN_T) and not (t & (POST_OPEN_T | _TIME_WORDS | RETURN_T | {"o2c", "oc"}))


def _is_post_open_name(name: str | None) -> bool:
    """A raw price field known only after bar t's open: close, high, low, adj close..."""
    if not name:
        return False
    t = _tokens(name)
    return bool(t & POST_OPEN_T) and not (t & (OPEN_T | _TIME_WORDS))


def _is_close_name(name: str | None) -> bool:
    if not name:
        return False
    t = _tokens(name)
    return bool(t & CLOSE_T) and not (t & (OPEN_T | _TIME_WORDS | {"high", "low", "highs", "lows"}))


def _is_np(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id in ("np", "numpy")


def _np_call(node: ast.AST, *names: str) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _is_np(node.func.value)
            and (not names or node.func.attr in names))


# ---------------------------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------------------------
@dataclass
class Finding:
    rule_id: str
    title: str
    severity: str
    line: int
    snippet: str
    message: str
    why: str
    fix: str
    same_as: int = 0  # v0.6: a repeat of the finding on this line, shown where the leak is written; not scored

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["same_as"]:
            d.pop("same_as")
        return d


@dataclass
class KeyInfo:
    """What the data-flow pass knows about an assigned name/column."""

    name: str
    frame: str | None
    line: int
    role: str
    forward: bool = False  # contains future data (negative shift, or built from a forward column)
    lagged: bool = False  # explicitly lagged with a positive shift
    price_derived: bool = False
    is_return: bool = False
    is_signal_expr: bool = False  # built from a comparison / np.where
    post_open: bool = False  # uses unlagged data known only after bar t's open (close, high, low)
    entry_open: bool = False  # a return measured from bar t's (unshifted) open
    ts: bool = False  # built with time-series operations (rolling, shift, pct_change, ...): a series over time
    outcome: bool = False  # forward data used only as an outcome: aggregated, selected or booked as PnL (LA10)
    bucket_fwd: bool = False  # a bucket/decile key formed from forward data (LA10)
    lead: int = 0  # how many bars ahead a forward key reads (0 = unknown, treated as 1)
    retlike: bool = False  # a return, price or PnL (or plain arithmetic/averages of them), not a transform of one

    def lead_bars(self) -> int:
        return self.lead or 1

    def ahead_after(self, lag: int) -> bool:
        """Still forward-looking when read with a lag of `lag` bars (a positive shift of the reference)."""
        return self.forward and _lead_after_lag(self.lead_bars(), lag) > 0


@dataclass
class CodeAudit:
    findings: list[Finding]
    score: int
    score_breakdown: list[dict]
    parsed: bool
    notes: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {s: sum(f.severity == s and not f.same_as for f in self.findings) for s in SEVERITIES}

    @property
    def summary(self) -> str:
        c = self.counts
        if not any(c[s] for s in ("high", "medium", "possible")):
            return ("No known lookahead or bias patterns found. A clean lint does not prove the backtest "
                    "is valid: static checks miss many leaks, and this one only checks the patterns listed by explain().")
        return (f"{c['high']} high, {c['medium']} medium and {c['possible']} possible issue(s) found. "
                "High and medium findings usually inflate backtest results; fix them before trusting any number.")

    def to_dict(self) -> dict:
        return {
            "tool": "check_code",
            "summary": self.summary,
            "audit_score": self.score,
            "audit_score_note": SCORE_CAVEAT,
            "score_breakdown": self.score_breakdown,
            "counts": self.counts,
            "parsed_with_ast": self.parsed,
            "notes": self.notes,
            "findings": [f.to_dict() for f in self.findings],
            "disclaimer": DISCLAIMER,
        }

    def to_markdown(self) -> str:
        from .report import code_audit_markdown

        return code_audit_markdown(self)


# ---------------------------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------------------------
def _num(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _num(node.operand)
        return -v if v is not None else None
    return None


_NUM_CONSTS: dict[str, float] = {}  # module-level numeric constants of the file being linted


def _numc(node: ast.AST | None, depth: int = 0) -> float | None:
    """A number, a module-level numeric constant (LAG = 1), a class-level one (cfg.horizon), or simple arithmetic of
    those (-H, H + 1, -1 * 5)."""
    if node is None or depth > 8:
        return None
    if isinstance(node, ast.Name):
        return _NUM_CONSTS.get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return _NUM_CONSTS.get(f"{node.value.id}.{node.attr}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _numc(node.operand, depth + 1)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        a, b = _numc(node.left, depth + 1), _numc(node.right, depth + 1)
        if a is None or b is None:
            return None
        return a + b if isinstance(node.op, ast.Add) else (a - b if isinstance(node.op, ast.Sub) else a * b)
    return _num(node)


_SYMS: dict[str, int] = {}  # symbolic shift amounts of the file being linted (shift(-horizon) ... shift(horizon))
_UNSTABLE: set[str] = set()  # names bound more than once (incl. parameters in several functions) or loop targets
SYM_BASE = 10 ** 4


def _sym_amount(node: ast.AST) -> int | None:
    """A positive pseudo-amount for an unresolved shift expression: the same expression always maps to the same
    band, so shift(-h) is undone by shift(h) (or shift(h + 1)) but not by shift(1) or shift(k)."""
    off = 0
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)) and _numc(node.right) is not None:
        off = int(_numc(node.right)) * (1 if isinstance(node.op, ast.Add) else -1)
        node = node.left
    if any(isinstance(x, ast.Name) and x.id in _UNSTABLE for x in ast.walk(node)):
        return None  # h reassigned, a loop variable, or bound in several scopes: the same text may differ in value
    key = ast.dump(node)
    if key not in _SYMS:
        if len(_SYMS) >= 90:
            return None
        _SYMS[key] = (len(_SYMS) + 1) * SYM_BASE + SYM_BASE // 2
    return _SYMS[key] + max(-SYM_BASE // 4, min(SYM_BASE // 4, off))


def _band(v: int) -> int | None:
    return v // SYM_BASE if SYM_BASE <= v < UNKNOWN_LEAD else None


def _lead_after_lag(lead: int, lag: int) -> int:
    """Bars a lead still looks ahead after a lag (<= 0: none). Symbolic amounts only cancel the same expression."""
    if lag <= 0 or lead >= UNKNOWN_LEAD:
        return lead
    bl, bg = _band(lead), _band(lag)
    if bg is not None and bl != bg:
        lag = 1  # a lag by some other unknown amount: assume the smallest
    elif bg is None and bl is not None:
        return lead - lag if lead - lag >= bl * SYM_BASE else lead
    return lead - lag
_FWD_FUNCS: set[str] = set()  # helper functions in the linted file whose return value uses a negative shift
_LAG_FUNCS: set[str] = set()  # helper functions that return a parameter shifted forward in time (a lag)
# v0.6: helpers whose shift amount is one of their parameters (`def lag(s, k): return s.shift(k)`): name -> (index, name)
_PARAM_SHIFT: dict[str, tuple[int, str]] = {}
MAX_SOURCE_CHARS = 1_000_000
_LOCK = threading.Lock()  # the two globals above are per-run state; MCP tool calls may run in threads
DATE_KEY_T = {"date", "dates", "dt", "time", "datetime", "day", "days", "timestamp", "period", "month", "year",
              "week", "quarter", "asof", "d", "rebalance", "rebal", "years", "months", "weeks", "quarters", "periods"}
TS_OPS = {"rolling", "expanding", "ewm", "shift", "pct_change", "diff", "cumsum", "cumprod", "cummax", "cummin",
          "resample"}
NP_STATS = {"mean", "nanmean", "median", "nanmedian", "std", "nanstd", "var", "nanvar", "min", "nanmin", "max",
            "nanmax", "amin", "amax", "quantile", "nanquantile", "percentile", "nanpercentile"}


def _sign(node: ast.AST | None) -> int:
    """+1 / -1 / 0 (unknown) for a shift amount. A bare name is assumed positive, `-name` negative,
    unless it is a module-level numeric constant (h = -1)."""
    if node is None:
        return 1  # default periods=1
    if isinstance(node, ast.Name) and node.id in _NUM_CONSTS:
        v = _NUM_CONSTS[node.id]
        return 1 if v > 0 else (-1 if v < 0 else 0)
    v = _numc(node)
    if v is not None:
        return 1 if v > 0 else (-1 if v < 0 else 0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -1
    if isinstance(node, (ast.Name, ast.Attribute)):
        return 1
    return 0


def _kw(call: ast.Call, name: str) -> ast.AST | None:
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _attr(call: ast.AST) -> str | None:
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _func_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


SHIFT_LIKE = {"shift": 0, "pct_change": 0, "diff": 0}  # attr -> positional index of `periods`


def _shift_sign(call: ast.Call) -> int | None:
    """Sign of a shift/pct_change/diff call (or `np.roll(x, k)`), or None if the call isn't one."""
    if _np_call(call, "roll"):
        amt = _kw(call, "shift") or (call.args[1] if len(call.args) > 1 else None)
        return _sign(amt) if amt is not None else 0
    a = _attr(call)
    if a not in SHIFT_LIKE:
        return None
    amt = _kw(call, "periods")
    if amt is None and call.args:
        amt = call.args[0]
    return _sign(amt)


UNKNOWN_LEAD = 10 ** 6  # a negative shift by an unknown amount: never undone by a later lag
TRAIN_DATA_CTORS = {"Dataset", "DMatrix", "Pool", "DeviceQuantileDMatrix", "QuantileDMatrix"}


def _shift_amount(call: ast.Call) -> int:
    """Signed number of bars of a shift/pct_change/diff/np.roll: +k lags, -k leads. Unknown positive amounts count
    as 1 (a small lag), unknown negative amounts as a lead that no lag undoes."""
    s = _shift_sign(call)
    if not s:
        return 0
    if _np_call(call, "roll"):
        node = _kw(call, "shift") or (call.args[1] if len(call.args) > 1 else None)
    else:
        node = _kw(call, "periods") or (call.args[0] if call.args else None)
    v = _numc(node) if node is not None else 1.0
    if v is not None and v == int(v):
        return int(v)
    neg = isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
    sym = _sym_amount(node.operand if neg else node) if node is not None else None
    if sym is None:
        return 1 if s > 0 else -UNKNOWN_LEAD
    return -sym if s < 0 else sym


def _chain(node: ast.AST):
    """Yield the method calls along a receiver chain: df['x'].shift(1).fillna(0) -> fillna, shift."""
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            yield node
            node = node.func.value
        elif isinstance(node, ast.Attribute) and isinstance(node.value, (ast.Call, ast.Attribute)) and node.attr in PANDAS_METHODS:
            node = node.value
        else:
            return


def _chain_base(node: ast.AST) -> ast.AST:
    for c in _chain(node):
        node = c.func.value
    while isinstance(node, ast.Attribute) and node.attr in ("values", "loc", "iloc") :
        node = node.value
    return node


def _chain_shift(node: ast.AST) -> int:
    """+1 if the receiver chain contains a positive shift, -1 if a negative one, else 0."""
    for c in _chain(node):
        s = _shift_sign(c)
        if s and (_attr(c) in ("shift", "roll") or s < 0):  # pct_change()/diff() are returns, not lags
            return s
        if _attr(c) in ("transform", "apply", "pipe") and c.args:
            # v0.5: groupby(...).transform(lambda s: s.pct_change(7).shift(1)) / .apply(lagged_helper): the function
            # ends in a positive shift, so the result is lagged
            f = c.args[0]
            if isinstance(f, ast.Lambda) and _chain_shift(f.body) > 0:
                return 1
            if isinstance(f, ast.Name) and f.id in _LAG_FUNCS:
                return 1
    b = _chain_base(node)
    if isinstance(b, ast.Call) and isinstance(b.func, ast.Name) and b.func.id in _LAG_FUNCS:
        return 1
    return 0


def _key(node: ast.AST) -> tuple[str | None, str | None]:
    """(column-or-variable name, frame name) for df['x'], df.x, x. Otherwise (None, None)."""
    if isinstance(node, ast.Subscript):
        s = _const_str(node.slice)
        if s is not None:
            base = node.value
            while isinstance(base, ast.Attribute) and base.attr in ("loc", "iloc"):
                base = base.value
            return s, base.id if isinstance(base, ast.Name) else None
        if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:  # df.loc[:, 'x']
            s = _const_str(node.slice.elts[1])
            if s is not None:
                base = node.value.value if isinstance(node.value, ast.Attribute) else node.value
                return s, base.id if isinstance(base, ast.Name) else None
        return None, None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id in MODULE_ALIASES or node.attr in PANDAS_METHODS:
            return None, None
        return node.attr, node.value.id
    if isinstance(node, ast.Name) and node.id not in MODULE_ALIASES:
        return node.id, None
    return None, None


_SHIFTED_FRAMES: dict[str, int] = {}  # per run (rt11): frame name -> k for `best = df[["low", "high"]].shift(-k)`


def _shifted_frame_read(n: ast.AST) -> int:
    """k if `n` reads one column of a frame shifted by -k (`best["high"]`, `best.loc[:, "high"]`, `best.high`)."""
    base, col = None, None
    if isinstance(n, ast.Subscript):
        sl, b = n.slice, n.value
        if isinstance(b, ast.Attribute) and b.attr == "loc":
            if isinstance(sl, ast.Tuple) and len(sl.elts) == 2 and isinstance(sl.elts[0], ast.Slice):
                col, base = _const_str(sl.elts[1]), b.value
        else:
            col, base = _const_str(sl), b
    elif isinstance(n, ast.Attribute) and n.attr not in PANDAS_METHODS and n.attr not in ("loc", "iloc", "at", "iat"):
        col, base = n.attr, n.value
    if col and isinstance(base, ast.Name):
        return _SHIFTED_FRAMES.get(base.id, 0)
    return 0


@dataclass
class _ExprScan:
    refs: list[tuple[str, int]] = field(default_factory=list)  # (key, lag sign context)
    neg_shift_calls: list[ast.Call] = field(default_factory=list)
    has_compare: bool = False
    has_return_op: bool = False


def _scan_expr(expr: ast.AST, skip: set[int] | None = None) -> _ExprScan:
    out = _ExprScan()

    def visit(n: ast.AST, lag: int) -> None:
        if skip and id(n) in skip:
            return
        if isinstance(n, ast.Call):
            s = _shift_sign(n)
            if s is not None:
                if s < 0:
                    out.neg_shift_calls.append(n)
                if _attr(n) in ("pct_change", "diff") and s >= 0:
                    out.has_return_op = True
                amt = _shift_amount(n)  # v0.3: signed number of bars (a lead of 21 isn't undone by a lag of 1)
                if _np_call(n, "roll"):  # np.roll(x, -1): the data argument is what moves in time
                    if n.args:
                        visit(n.args[0], lag if lag else amt)
                    return
                inner = lag if lag else (amt if _attr(n) == "shift" or s < 0 else 0)
                visit(n.func.value, inner)
                for a in n.args:
                    visit(a, lag)
                for k in n.keywords:
                    visit(k.value, lag)
                return
            if isinstance(n.func, ast.Name) and n.func.id in _FWD_FUNCS:
                out.neg_shift_calls.append(n)
            if isinstance(n.func, ast.Name) and n.func.id in _PARAM_SHIFT:
                # v0.6: lag(r, -2) with `def lag(s, k): return s.shift(k)` is a negative shift
                idx, pname = _PARAM_SHIFT[n.func.id]
                amt = _kw(n, pname) if _kw(n, pname) is not None else (n.args[idx] if len(n.args) > idx else None)
                v = _numc(amt) if amt is not None else None
                if v is not None and v < 0:
                    out.neg_shift_calls.append(n)
                    for a in n.args:
                        visit(a, lag)
                    for k in n.keywords:
                        visit(k.value, lag)
                    return
            if isinstance(n.func, ast.Name) and n.func.id in _LAG_FUNCS:
                for a in n.args:
                    visit(a, lag if lag else 1)
                for k in n.keywords:
                    visit(k.value, lag)
                return
            if _attr(n) in ("fit", "fit_transform", "fit_predict", "partial_fit"):
                visit(n.func.value, lag)  # the estimator, not the training data, is what gets assigned
                return
            if isinstance(n.func, ast.Name) and n.func.id == "len":
                return  # len(data) is a row count, not data
            if _func_name(n) in TRAIN_DATA_CTORS:
                # lgb.Dataset(X, label=y) / xgb.DMatrix(X, y): the label is what the model learns, not an input.
                # Only the feature data counts; train/test overlap is LA05/LA14's job.
                if n.args:
                    visit(n.args[0], lag)
                for k in n.keywords:
                    if k.arg in ("data", "X", "x"):
                        visit(k.value, lag)
                return
            fn = _func_name(n)
            if fn in ("pivot", "pivot_table"):
                # v0.4: df.pivot(index=..., columns=..., values="fwd_ret") reads the column named in `values`
                # (pivot_table also takes it positionally: df.pivot_table("fwd_ret", ...), pd.pivot_table(df, "fwd_ret"))
                vals = _kw(n, "values")
                if vals is None and fn == "pivot_table":
                    mod = isinstance(n.func, ast.Name) or (isinstance(n.func, ast.Attribute) and isinstance(
                        n.func.value, ast.Name) and n.func.value.id in ("pd", "pandas"))
                    pos = 1 if mod else 0
                    vals = n.args[pos] if len(n.args) > pos else None
                for v in (vals.elts if isinstance(vals, (ast.List, ast.Tuple)) else [vals]):
                    if _const_str(v):
                        out.refs.append((_const_str(v), lag))
            if fn == "where" and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id in ("np", "numpy"):
                out.has_compare = True
            if fn in ("log", "log1p") and n.args and isinstance(n.args[0], ast.BinOp) and isinstance(n.args[0].op, ast.Div):
                out.has_return_op = True
            # don't treat the function being called as a data reference
            if isinstance(n.func, ast.Attribute):
                visit(n.func.value, lag)
            for a in n.args:
                visit(a, lag)
            for k in n.keywords:
                visit(k.value, lag)
            return
        if isinstance(n, ast.Compare):
            out.has_compare = True
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            # price / price.shift(1) - 1 style returns
            for side in (n.left, n.right):
                if isinstance(side, ast.Call) and _shift_sign(side) == 1:
                    out.has_return_op = True
            # close / open - 1 style returns between two different price fields of the same bar
            kl, _fl = _key(n.left)
            kr, _fr = _key(n.right)
            if (kl and kr and kl != kr and _is_price_name(kl) and _is_price_name(kr)
                    and classify(kl) != "exec" and classify(kr) != "exec"):
                out.has_return_op = True
        if isinstance(n, ast.Attribute) and n.attr in ("shape", "columns", "dtypes", "ndim", "size"):
            return  # metadata of a frame, not its values
        if _SHIFTED_FRAMES and isinstance(n, (ast.Subscript, ast.Attribute)):
            k_sf = _shifted_frame_read(n)
            if k_sf:
                # rt11 G9: best = df[["low", "high"]].shift(-1); best["high"] is df["high"].shift(-1): report the read
                # as that column shifted by -k, so every negative-shift rule sees it the same way
                loc = {a: getattr(n, a, None) for a in ("lineno", "col_offset", "end_lineno", "end_col_offset")}
                fake = ast.Call(func=ast.Attribute(value=n, attr="shift", ctx=ast.Load(), **loc),
                                args=[ast.Constant(value=-k_sf, **loc)], keywords=[], **loc)
                out.neg_shift_calls.append(fake)
        if isinstance(n, (ast.Subscript, ast.Attribute, ast.Name)):
            k, _ = _key(n)
            if k is not None:
                out.refs.append((k, lag))
                if isinstance(n, ast.Subscript):
                    visit(n.value, lag) if not isinstance(n.value, ast.Name) else None
                return
        for c in ast.iter_child_nodes(n):
            visit(c, lag)

    visit(expr, 0)
    return out


def _and3(vals) -> bool | None:
    """Three-valued AND: False if any is False, True if all are True, else None (unknown)."""
    vals = list(vals)
    if any(v is False for v in vals):
        return False
    return True if all(v is True for v in vals) else None


def _or3(vals) -> bool | None:
    """Three-valued OR: True if any is True, False if all are False, else None (unknown)."""
    vals = list(vals)
    if any(v is True for v in vals):
        return True
    return False if all(v is False for v in vals) else None


def _merge_keep(keep, p):
    return p if keep is None else (lambda c, a=keep, b=p: _and3([a(c), b(c)]))


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[c] = p
    return parents


# ---------------------------------------------------------------------------------------------
# The linter
# ---------------------------------------------------------------------------------------------
class _Linter:
    def __init__(self, source: str, tree: ast.Module):
        self.src = source
        self.lines = source.splitlines()
        self.tree = tree
        self.parents = _parent_map(tree)
        self._all: list[ast.AST] = list(ast.walk(tree))  # every node, walked once (the tree is never modified)
        self._name_sets: dict[int, frozenset[str]] = {}
        self._funcs: dict[str, ast.AST] = {}
        for _n in self._all:
            if isinstance(_n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._funcs.setdefault(_n.name, _n)
        self._stmts_sorted = sorted((x for x in self._all if isinstance(x, ast.stmt)), key=lambda x: (x.lineno, x.col_offset))
        self.findings: list[Finding] = []
        self.info: dict[str, KeyInfo] = {}
        self._la10_pending: list[tuple[str, ast.AST, str, int]] = []
        self._la01_pending: list[tuple[str, ast.AST, str]] = []
        self._edges: dict[str, set[str]] = {}  # key -> forward-looking keys it is computed from (as inputs)
        self._signal_roots: set[str] = set()  # signal/position keys, including partial writes (pos[1:] = ...)
        self._containers: dict[str, dict] = {}  # frames built from columns, some forward-looking (LA10)
        self._cont_elems: list[tuple[str | None, str]] = []
        self._fill_masks: set[str] = set()  # v0.4: want & (low.shift(-1) <= limit): a fill model, judged by its uses
        self._fill_pending: list[tuple] = []
        self.assigned_names: set[str] = set()
        self.str_consts: dict[str, str] = {}
        lowered_src = source
        self.ts_context = any(h in lowered_src for h in TS_HINTS)
        self.imports: set[str] = set()
        self._lead_seen: dict[str, int] = {}

    # -- utilities --------------------------------------------------------------------------
    _ALL_PICKERS = ("max", "min", "idxmax", "idxmin", "argmax", "argmin", "nlargest", "nsmallest", "sorted", "sort_values")

    def _sel_index(self):
        """Indexes for selection checks, built once: single-name assignments by line, picker calls, top-row picks."""
        if not hasattr(self, "_seli"):
            assigns = sorted((a for a in self._all if isinstance(a, ast.Assign) and len(a.targets) == 1
                              and isinstance(a.targets[0], ast.Name)), key=lambda a: (a.lineno, a.col_offset))
            pickers = [c for c in self._all if isinstance(c, ast.Call) and (_func_name(c) or "") in self._ALL_PICKERS]
            tops = [x for x in self._all if (isinstance(x, ast.Call) and _attr(x) in ("head", "tail") and x.args
                                             and _num(x.args[0]) == 1)
                    or (isinstance(x, ast.Subscript) and _num(x.slice) in (0, -1)
                        and (isinstance(x.value, ast.Call) or (isinstance(x.value, ast.Attribute)
                                                               and x.value.attr in ("iloc", "index"))))]
            self._seli = (assigns, pickers, tops)
        return self._seli

    def _picked(self, stored: set[str], after: int, allow_sort: bool = True) -> bool:
        """Something stored in a loop is later maximized (max/idxmax/...), or ranked and its top row or a best-named
        result kept."""
        _assigns, pickers, tops = self._sel_index()
        if not hasattr(self, "_pick_by_name"):
            self._pick_by_name: dict[str, list[ast.Call]] = {}
            self._top_by_name: dict[str, list[ast.AST]] = {}
            for c in pickers:
                for nm in self.names_in(c):
                    self._pick_by_name.setdefault(nm, []).append(c)
            for t in tops:
                for nm in self.names_in(t):
                    self._top_by_name.setdefault(nm, []).append(t)
        derived, best = set(stored), False
        for st in self._iter_refs(derived, after):
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                if _tokens(st.targets[0].id) & {"best", "top", "optimal", "opt", "winner"}:
                    best = True
                    break
                derived.add(st.targets[0].id)
        cands = {id(c): c for nm in derived for c in self._pick_by_name.get(nm, []) if c.lineno > after}
        sorts = [c for c in cands.values() if _func_name(c) in ("sorted", "sort_values")]
        ranked_top = bool(sorts) and any(t.lineno > after for nm in derived for t in self._top_by_name.get(nm, []))
        for c in cands.values():
            if not (self.names_in(c) & stored):
                continue
            if _func_name(c) not in ("sorted", "sort_values"):
                return True
            if allow_sort and (best or ranked_top):
                return True
        return False

    def _ref_index(self) -> dict[str, list[int]]:
        """name -> positions (in line order) of the Assign/AugAssign/Expr statements whose value reads it."""
        if not hasattr(self, "_refidx"):
            self._ref_stmts = [st for st in self._stmts_sorted if isinstance(st, (ast.Assign, ast.AugAssign, ast.Expr))]
            self._ref_lines = [st.lineno for st in self._ref_stmts]
            self._refidx: dict[str, list[int]] = {}
            for i, st in enumerate(self._ref_stmts):
                for nm in self.names_in(st.value):
                    self._refidx.setdefault(nm, []).append(i)
        return self._refidx

    def _iter_refs(self, watch: set[str], after_line: int, before_line: int = 10 ** 9):
        """Statements after `after_line` that read any name in `watch`, in line order. `watch` may grow while
        iterating (the caller adds derived names)."""
        idx = self._ref_index()
        start = bisect.bisect_right(self._ref_lines, after_line)
        heap: list[tuple[int, str, int]] = []
        added: set[str] = set()
        done: set[int] = set()

        def add(nm: str) -> None:
            if nm in added:
                return
            added.add(nm)
            lst = idx.get(nm)
            if lst:
                i = bisect.bisect_left(lst, start)
                if i < len(lst):
                    heapq.heappush(heap, (lst[i], nm, i))

        for nm in list(watch):
            add(nm)
        while heap:
            pos, nm, i = heapq.heappop(heap)
            lst = idx[nm]
            if i + 1 < len(lst):
                heapq.heappush(heap, (lst[i + 1], nm, i + 1))
            if pos in done:
                continue
            done.add(pos)
            st = self._ref_stmts[pos]
            if st.lineno >= before_line:
                return
            yield st
            for w in list(watch):
                add(w)

    def names_in(self, node: ast.AST) -> frozenset[str]:
        """Names read or written anywhere under `node` (cached)."""
        got = self._name_sets.get(id(node))
        if got is None:
            got = frozenset(x.id for x in ast.walk(node) if isinstance(x, ast.Name))
            self._name_sets[id(node)] = got
        return got

    def add(self, rule: str, severity: str, node: ast.AST, message: str, same_as: int = 0) -> None:
        line = getattr(node, "lineno", 0) or 0
        snippet = self.lines[line - 1].strip()[:160] if 0 < line <= len(self.lines) else ""
        chk = CHECKS[rule]
        self.findings.append(Finding(rule, chk.title, severity, line, snippet, message,
                                     chk.why_it_inflates, chk.how_to_fix, same_as))

    def inside(self, node: ast.AST, types: tuple) -> ast.AST | None:
        p = self.parents.get(node)
        while p is not None:
            if isinstance(p, types):
                return p
            p = self.parents.get(p)
        return None

    def enclosing_assign_target(self, node: ast.AST) -> tuple[str | None, str | None, ast.AST | None]:
        p = self.parents.get(node)
        while p is not None and not isinstance(p, (ast.stmt,)):
            p = self.parents.get(p)
        if isinstance(p, ast.Assign) and len(p.targets) == 1:
            k, f = _key(p.targets[0])
            return k, f, p
        if isinstance(p, (ast.AugAssign, ast.AnnAssign)):
            k, f = _key(p.target)
            return k, f, p
        # df = df.assign(name=...)
        q = self.parents.get(node)
        while q is not None and not isinstance(q, ast.stmt):
            if isinstance(q, ast.keyword) and isinstance(self.parents.get(q), ast.Call) and _attr(self.parents[q]) == "assign":
                return q.arg, None, p
            q = self.parents.get(q)
        return None, None, p

    # -- driver -----------------------------------------------------------------------------
    def run(self) -> list[Finding]:
        _NUM_CONSTS.clear()
        counts: dict[str, int] = {}
        for n in self.tree.body:
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        counts[t.id] = counts.get(t.id, 0) + 1
                        if _num(n.value) is not None:
                            _NUM_CONSTS[t.id] = _num(n.value)
        for k, c in counts.items():
            if c > 1:
                _NUM_CONSTS.pop(k, None)  # reassigned: not a constant
        _SYMS.clear()
        # Names whose value may differ between two places with the same text: bound more than once anywhere
        # (assignments, parameters, loop targets, with/except/comprehension targets), or any loop target.
        binds: dict[str, int] = {}
        local: set[str] = set()  # bound inside a function (a parameter or a local): shadows a module constant
        _UNSTABLE.clear()
        for n in self._all:
            names: list[str] = []
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    names += [x.id for x in ast.walk(t) if isinstance(x, ast.Name)]
                if isinstance(n, ast.AugAssign):
                    names *= 2  # x += 1 rebinds
            elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
                loop = [x.id for x in ast.walk(n.target) if isinstance(x, ast.Name)]
                _UNSTABLE.update(loop)
                names += loop
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                a = n.args
                params = [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs] + (
                    [a.vararg.arg] if a.vararg else []) + ([a.kwarg.arg] if a.kwarg else [])
                names += params
                local.update(params)
                for x in ast.walk(n):
                    if isinstance(x, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For)) and x is not n:
                        for t in (x.targets if isinstance(x, ast.Assign) else [x.target]):
                            local.update(y.id for y in ast.walk(t) if isinstance(y, ast.Name))
            for nm in names:
                binds[nm] = binds.get(nm, 0) + 1
        _UNSTABLE.update(nm for nm, c in binds.items() if c > 1)
        for nm in local:
            _NUM_CONSTS.pop(nm, None)  # def f(df, H=1): ... shift(H) is the parameter, not the module's H
        for cd in self.tree.body:  # class cfg: horizon = 5  ->  cfg.horizon
            if isinstance(cd, ast.ClassDef):
                seen: dict[str, int] = {}
                for n in cd.body:
                    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                        seen[n.targets[0].id] = seen.get(n.targets[0].id, 0) + 1
                        if _num(n.value) is not None:
                            _NUM_CONSTS[f"{cd.name}.{n.targets[0].id}"] = _num(n.value)
                for k, c in seen.items():
                    if c > 1:
                        _NUM_CONSTS.pop(f"{cd.name}.{k}", None)
        _FWD_FUNCS.clear()
        _LAG_FUNCS.clear()
        _PARAM_SHIFT.clear()
        _SHIFTED_FRAMES.clear()
        for fd in self._all:
            if isinstance(fd, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = {a.arg for a in fd.args.posonlyargs + fd.args.args}
                rets = [r.value for r in ast.walk(fd) if isinstance(r, ast.Return) and r.value is not None]
                plist = [a.arg for a in fd.args.posonlyargs + fd.args.args]
                amts = {(_kw(r, "periods") if _kw(r, "periods") is not None else (r.args[0] if r.args else None))
                        for r in rets if isinstance(r, ast.Call) and _attr(r) == "shift"}
                amt_names = {a.id for a in amts if isinstance(a, ast.Name)}
                if rets and len(rets) == len([r for r in rets if isinstance(r, ast.Call) and _attr(r) == "shift"]) and len(
                        amts) == 1 and len(amt_names) == 1 and next(iter(amt_names)) in plist:
                    pn = next(iter(amt_names))
                    _PARAM_SHIFT[fd.name] = (plist.index(pn), pn)
                if rets and all(isinstance(r, ast.Call) and _attr(r) == "shift" and _shift_sign(r) == 1
                                and (isinstance(_chain_base(r), ast.Name) and _chain_base(r).id in params
                                     # rt12 K07: (s.ewm(...).mean() - s.ewm(...).mean()).shift(1) of a parameter
                                     or (bool({x.id for x in ast.walk(r.func.value) if isinstance(x, ast.Name)} & params)
                                         and not _scan_expr(r.func.value).neg_shift_calls))
                                for r in rets):
                    _LAG_FUNCS.add(fd.name)
        for _ in range(3):  # helpers calling helpers
            for fd in self._all:
                if isinstance(fd, (ast.FunctionDef, ast.AsyncFunctionDef)) and fd.name not in _FWD_FUNCS:
                    for ret in ast.walk(fd):
                        if isinstance(ret, ast.Return) and ret.value is not None and _scan_expr(ret.value).neg_shift_calls:
                            _FWD_FUNCS.add(fd.name)
                            break
        for n in self._all:
            if isinstance(n, ast.Import):
                self.imports |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                self.imports.add(n.module.split(".")[0])
            elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                s = _const_str(n.value)
                if s is not None:
                    self.str_consts[n.targets[0].id] = s
        self._close_funcs = self._v7_close_funcs()
        self.dataflow()
        self._flow_done = True
        self._resolve_la10_pending()
        self._collect_helper_mults()
        self.rule_calls()
        self.rule_array_alignment()
        self.rule_same_bar()
        self.rule_open_fill()
        self.rule_loop_same_bar()
        self.rule_fit_before_split()
        self.rule_feature_matrix()
        self.rule_grid_search()
        self.rule_test_set_selection()
        self.rule_pit_fundamentals()
        self.rule_future_index()
        self.rule_future_taint()
        self.rule_reversed_windows()
        self.rule_reversed_frames()
        self.rule_global_norm()
        self.rule_universe_lists()
        self.rule_survivor_filter()
        self.rule_v06()
        self.rule_v07()
        self.rule_costs()
        self.relocate_v06()
        self._v7_guarded(self.relocate_v07)
        self._v7_guarded(self._relocate_posthoc)
        return self.dedupe()

    def _v7_close_funcs(self) -> dict[str, bool]:
        """v0.7: local helpers whose every return is a condition on this bar's close/high/low, unlagged
        (`return (df["close"].rolling(20).mean() > df["close"].rolling(100).mean()).astype(float)`). The caller's
        result is a same-bar signal, exactly like the inline expression (PR #40 gap)."""
        out: dict[str, bool] = {}
        rets_of: dict[int, list] = {}
        for r in self._all:
            if isinstance(r, ast.Return):
                fd = self.inside(r, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                if isinstance(fd, ast.FunctionDef):
                    rets_of.setdefault(id(fd), []).append(r.value)
        for fd in self._funcs.values():
            if not isinstance(fd, ast.FunctionDef) or fd.name in _LAG_FUNCS or fd.name in _FWD_FUNCS:
                continue
            rets = rets_of.get(id(fd), [])
            if not rets or any(r is None for r in rets):
                continue
            if not all(any(isinstance(x, ast.Compare) or _np_call(x, "where", "sign") for x in ast.walk(r)) for r in rets):
                continue  # not a condition: cheap test before anything else
            if any(isinstance(c, ast.Call) and (_func_name(c) or "").startswith("read_") for c in ast.walk(fd)):
                continue
            ok = True
            for r in rets:
                sc = _scan_expr(r)
                cond = sc.has_compare or any(_np_call(c, "where", "sign") for c in ast.walk(r))
                if (not cond or sc.neg_shift_calls or _chain_shift(r) != 0
                        or not any(lag <= 0 and _is_post_open_name(k) for k, lag in sc.refs)):
                    ok = False
                    break
            if ok:
                out[fd.name] = True
        return out

    def dedupe(self) -> list[Finding]:
        seen: set[tuple[str, int]] = set()
        out = []
        for f in sorted(self.findings, key=lambda f: (f.line, SEVERITIES.index(f.severity), f.rule_id)):
            if (f.rule_id, f.line) in seen:
                continue
            seen.add((f.rule_id, f.line))
            out.append(f)
        return out

    # -- data flow over assignments ------------------------------------------------------------
    def _assignments(self):
        items = []
        for n in self._all:
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if (isinstance(t, (ast.Tuple, ast.List)) and isinstance(n.value, (ast.Tuple, ast.List))
                            and len(t.elts) == len(n.value.elts) and not any(isinstance(e, ast.Starred) for e in t.elts)):
                        # X, y = data.iloc[:, :-1], data.iloc[:, -1]: element by element
                        for i, (te, ve) in enumerate(zip(t.elts, n.value.elts)):
                            items.append((n.lineno, n.col_offset + i, te, ve, n))
                    else:
                        items.append((n.lineno, n.col_offset, t, n.value, n))
            elif (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                  and isinstance(n.value.func.value, ast.Name) and any(
                      k.arg == "inplace" and isinstance(k.value, ast.Constant) and k.value.value is True
                      for k in n.value.keywords)):
                # data.rename(columns=..., inplace=True): the same as data = data.rename(columns=...)
                items.append((n.lineno, n.col_offset, ast.Name(id=n.value.func.value.id, lineno=n.lineno), n.value, n))
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and n.value is not None:
                items.append((n.lineno, n.col_offset, n.target, n.value, n))
            elif isinstance(n, ast.Call) and _attr(n) == "assign":
                for k in n.keywords:
                    if k.arg:
                        items.append((n.lineno, n.col_offset, ast.Name(id=k.arg, lineno=n.lineno), k.value, n))
        items.sort(key=lambda x: (x[0], x[1]))
        return items

    def dataflow(self) -> None:
        for line, _, target, value, stmt in self._assignments():
            if isinstance(target, (ast.Tuple, ast.List)):
                for e in target.elts:
                    k, _f = _key(e)
                    if k:
                        self.assigned_names.add(k)
                continue
            key, frame = _key(target)
            if key is None:
                if not self._container_columns_assigned(target, value, stmt):
                    self._partial_signal_write(target, value, stmt)
                continue
            if frame in self._containers and self._containers[frame]["order"] is not None and not any(
                    nm == key for nm, _f in self._containers[frame]["order"]):
                self._containers[frame]["order"] = None  # a new column: positions from the end shift
            self.assigned_names.add(key)
            scan = _scan_expr(value)
            role = classify(key)
            ref_infos = [(self.info.get(k), lag) for k, lag in scan.refs if k != key or True]
            forward = bool(scan.neg_shift_calls) or any(
                ki is not None and ki.ahead_after(lag) and ki.name != key for ki, lag in ref_infos)
            leads = [(-_shift_amount(c) if _shift_sign(c) else UNKNOWN_LEAD) for c in scan.neg_shift_calls]
            leads += [_lead_after_lag(ki.lead_bars(), lag) for ki, lag in ref_infos
                      if ki is not None and ki.name != key and ki.ahead_after(lag)]
            lead = max(leads) if forward and leads else 0
            chain_s = _chain_shift(value)
            data_refs = [(ki, lag, k) for (ki, lag), (k, _) in zip(ref_infos, scan.refs)]
            # v0.5: in panel.groupby("date")["mom_7"].rank(), `panel` only groups the rows; the data is mom_7
            gb_frames = {x.func.value.id for x in ast.walk(value) if isinstance(x, ast.Call) and _attr(x) == "groupby"
                         and isinstance(x.func.value, ast.Name) and isinstance(self.parents.get(x), ast.Subscript)}
            lag_refs = [r for r in data_refs if r[2] not in gb_frames] if gb_frames else data_refs
            lagged = chain_s > 0 or (
                bool(lag_refs) and all(lag > 0 or (ki is not None and ki.lagged) for ki, lag, _k in lag_refs))
            price = any(_is_price_name(k) or (ki is not None and ki.price_derived) for ki, _l, k in data_refs)
            is_ret = scan.has_return_op or role == "return" or (
                isinstance(_chain_base(value), (ast.Name, ast.Subscript, ast.Attribute)) and len(scan.refs) == 1
                and data_refs[0][0] is not None and data_refs[0][0].is_return)
            is_sig = scan.has_compare or any(ki is not None and ki.is_signal_expr for ki, _l, _k in data_refs)
            post_open = not lagged and any(
                lag <= 0 and ((_is_post_open_name(k) and not (ki is not None and ki.lagged))
                              or (ki is not None and ki.post_open and not ki.lagged))
                for ki, lag, k in data_refs)
            entry_open = self._open_start(value)
            nodes = list(ast.walk(value))  # walked once, reused below
            if self._close_funcs and not lagged and not forward:
                # v0.7 (PR #40 gap): sig = rule(df) with `def rule(df): return (df["close"].rolling(..) > ..)`: the
                # helper's result is known only at this bar's close, like the same expression written inline
                hc = [self._close_funcs[c.func.id] for c in nodes if isinstance(c, ast.Call)
                      and isinstance(c.func, ast.Name) and c.func.id in self._close_funcs]
                if hc:
                    price = post_open = True
                    is_sig = is_sig or any(hc)
            ts = any(isinstance(c, ast.Call) and (_attr(c) in TS_OPS or _np_call(c, "diff", "roll")) for c in nodes) or any(
                ki is not None and ki.ts for ki, _l, _k in data_refs)

            def _pricey(x: ast.AST) -> bool:  # every data reference is a price (a ratio of prices is a return)
                rs = _scan_expr(x).refs
                return bool(rs) and all(_is_price_name(k) or classify(k) == "exec" or (
                    self.info.get(k) is not None and (self.info[k].price_derived or self.info[k].retlike)) for k, _l in rs)

            retlike = role in ("label", "return", "pnl") or classify(key) == "exec" or _is_price_name(key)
            if not retlike and (is_ret or any((ki is not None and ki.retlike) or classify(k) in ("label", "return", "pnl")
                                              or _is_price_name(k) for ki, _l, k in data_refs)):
                # a return, or arithmetic of returns, unless a transform (sign, rolling, rank, ...) is wrapped around it
                transformed = scan.has_compare or any(
                    isinstance(c, ast.Call) and _func_name(c) in TRANSFORM_FUNCS for c in nodes) or (
                    not scan.has_return_op and any(isinstance(b, ast.BinOp) and isinstance(b.op, ast.Div)
                                                   and _scan_expr(b.right).refs and not (_pricey(b.left) and _pricey(b.right))
                                                   for b in nodes))
                retlike = not transformed
            if forward and self._fill_mask_value(value):
                self._fill_masks.add(key)
            else:
                self._fill_masks.discard(key)
            if frame is None and isinstance(target, ast.Name):
                k_sf = self._whole_frame_lead(value)
                if k_sf:
                    _SHIFTED_FRAMES[key] = k_sf
                else:
                    _SHIFTED_FRAMES.pop(key, None)
            self._rule_negative_shift(key, role, scan, value, stmt)
            self._rule_label_same_bar(key, role, scan, forward, is_ret, stmt)
            outcome, bucket_fwd = self._rule_la10(key, role, frame, value, scan, data_refs, stmt)
            if (frame is None and key in self._containers and not forward
                    and self._cont_ahead(self._containers[key], self._containers[key]["src"])):
                # data = data.add_prefix('f_'): still holds the forward column (a self-reference doesn't count)
                forward = True
                src = self.info.get(self._containers[key]["src"])
                lead = _lead_after_lag(src.lead_bars(), self._containers[key].get("lag", 0)) if src is not None else lead

            if forward and lead and lead < 10 ** 5:  # v0.7: a label's horizon survives `y = df["y"]` rebinding
                self._lead_seen[key] = max(self._lead_seen.get(key, 0), lead)
            self.info[key] = KeyInfo(key, frame, line, role, forward=forward and role != "label" or forward,
                                     lagged=lagged, price_derived=price, is_return=is_ret, is_signal_expr=is_sig,
                                     post_open=post_open, entry_open=entry_open, ts=ts, outcome=outcome,
                                     bucket_fwd=bucket_fwd, lead=lead, retlike=retlike)
            if role == "label" and not scan.neg_shift_calls and not forward:
                # a label that is not forward-looking is just a column; don't mark it forward
                self.info[key].forward = False
            elif role == "label":
                self.info[key].forward = True

    # -- v0.4: fill-condition models ----------------------------------------------------------------------
    _EXTREME_T = {"low", "lows", "high", "highs", "lo", "hi"}

    def _ahead(self, x: ast.AST) -> bool:
        sc = _scan_expr(x)
        return bool(sc.neg_shift_calls) or any(
            self.info.get(k) is not None and self.info[k].ahead_after(lag) for k, lag in sc.refs)

    def _next_extreme(self, node: ast.AST) -> bool:
        """`low.shift(-1)`, `df["high"].shift(-1)`, or `best["low"]` with best = df[["low", "high"]].shift(-1): the
        next bar's low or high, exactly one bar ahead."""
        if _shifted_frame_read(node) == 1:
            k = _key(node)[0]
            t = _tokens(k) if k else frozenset()
            return bool(t & self._EXTREME_T) and not t & (RETURN_T | OPEN_T | CLOSE_T | _TIME_WORDS)
        calls = list(_chain(node))
        shifts = [c for c in calls if _attr(c) == "shift"]
        if len(shifts) != 1 or _shift_amount(shifts[0]) != -1 or any(
                _attr(c) not in ({"shift"} | self._HARMLESS) for c in calls):
            return False
        k = _key(_chain_base(node))[0]
        t = _tokens(k) if k else frozenset()
        return bool(t & self._EXTREME_T) and not t & (RETURN_T | OPEN_T | CLOSE_T | _TIME_WORDS)

    def _whole_frame_lead(self, value: ast.AST) -> int:
        """k for a frame shifted as a whole by -k: `df[["low", "high"]].shift(-1)`, `df.loc[:, cols].shift(-k)`,
        `df.shift(-1).dropna()` (other calls in the chain must keep the column names). 0 otherwise."""
        calls = list(_chain(value))
        sh = [c for c in calls if _attr(c) == "shift"]
        if len(sh) != 1 or _kw(sh[0], "axis") is not None or _kw(sh[0], "freq") is not None:
            return 0
        amt = _shift_amount(sh[0])
        if amt >= 0 or any(_attr(c) not in (self._NAME_KEEPING_OPS | self._HARMLESS) for c in calls):
            return 0
        recv = sh[0].func.value
        if isinstance(recv, ast.Subscript):
            sl = recv.slice
            if isinstance(recv.value, ast.Attribute) and recv.value.attr == "loc" and isinstance(sl, ast.Tuple):
                sl = sl.elts[1] if len(sl.elts) == 2 else None
            if not isinstance(sl, (ast.List, ast.Name)):
                return 0  # df["low"].shift(-1) is a series: its own rules apply
        elif not isinstance(recv, ast.Name):
            return 0
        return min(-amt, UNKNOWN_LEAD)

    def _extreme_lead(self, call: ast.Call) -> bool:
        """`low.shift(-k)` / `df["high"].shift(-k)`: a later bar's low or high."""
        k = _key(_chain_base(call.func.value))[0] if isinstance(call.func, ast.Attribute) else None
        t = _tokens(k) if k else frozenset()
        return bool(t & self._EXTREME_T) and not t & (RETURN_T | OPEN_T | CLOSE_T | _TIME_WORDS)

    def _fill_mask_value(self, value: ast.AST) -> bool:
        """`want_buy & (low.shift(-1) <= limit_px)`: an order decided without future data (at least one term) that
        fills if the next bar's low (high) reaches the limit. The only future data is the next bar's low/high,
        compared with a price known at t."""
        if not (isinstance(value, (ast.BoolOp, ast.BinOp)) and any(isinstance(x, ast.Compare) for x in ast.walk(value))):
            return False  # perf guard: a fill mask is an `&` (or `and`) with a comparison in it
        terms, stack = [], [value]
        while stack:
            x = stack.pop()
            if isinstance(x, ast.BoolOp) and isinstance(x.op, ast.And):
                stack += x.values
            elif isinstance(x, ast.BinOp) and isinstance(x.op, ast.BitAnd):
                stack += [x.left, x.right]
            else:
                terms.append(x)
        fills = other = 0
        for t in terms:
            if not self._ahead(t):
                other += 1
                continue
            if (isinstance(t, ast.Compare) and len(t.ops) == 1
                    and isinstance(t.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE))):
                a, b = t.left, t.comparators[0]
                if (self._next_extreme(a) and not self._ahead(b)) or (self._next_extreme(b) and not self._ahead(a)):
                    fills += 1
                    continue
            return False
        return fills >= 1 and other >= 1

    def _fill_mask_ok(self, key: str) -> bool:
        """A fill mask is used only to select or zero trade returns (`r[filled]`, `r.where(filled)`,
        `np.where(filled, r, 0)`) or to count fills, never in a signal, position, feature or model input."""
        if not hasattr(self, "_fmo"):
            self._fmo: dict[str, bool] = {}
        if key in self._fmo:
            return self._fmo[key]
        self._fmo[key] = False  # guards recursion
        ok = key in self._fill_masks and key not in _UNSTABLE
        if not hasattr(self, "_name_loads"):  # indexed once (red-team round 10: 1,600 fill models took 30 s)
            self._name_loads: dict[str, list[ast.Name]] = {}
            for x in self._all:
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                    self._name_loads.setdefault(x.id, []).append(x)
        for x in (self._name_loads.get(key, []) if ok else []):
            st = self._stmt_of(x)
            tg = st.targets if isinstance(st, ast.Assign) else [st.target] if isinstance(st, (ast.AugAssign, ast.AnnAssign)) else []
            for t in tg:
                tk = _key(t)[0]
                if tk and (classify(tk) == "feature" or (classify(tk) == "signal" and _tokens(tk) & (SIGNAL_T - AMBIG_SIGNAL_T))
                           or _tokens(tk) & {"x", "features", "feature", "feats"}):
                    ok = False
            a = self.parents.get(x)
            while a is not None and a is not st:
                if isinstance(a, ast.Call) and (_attr(a) in FIT_ATTRS or _func_name(a) in TRAIN_DATA_CTORS):
                    ok = False
                    break
                a = self.parents.get(a)
            c, p = x, self.parents.get(x)
            while isinstance(p, ast.UnaryOp) and isinstance(p.op, (ast.Invert, ast.Not)):
                c, p = p, self.parents.get(p)
            sel = (isinstance(p, ast.Subscript) and (p.slice is c or (isinstance(p.slice, ast.Tuple) and p.slice.elts
                                                                       and p.slice.elts[0] is c)))
            cond = isinstance(p, ast.Call) and ((_attr(p) in ("where", "mask") and p.args and p.args[0] is c)
                                                or any(k.arg == "cond" and k.value is c for k in p.keywords))
            counted = (isinstance(p, ast.Attribute) and p.attr in AGG_T | {"any", "all"}) or (
                isinstance(p, ast.Call) and _func_name(p) in ("print", "len", "sum", "count_nonzero", "mean"))
            if not (sel or cond or counted):
                ok = False
            if not ok:
                break
        self._fmo[key] = ok
        return ok

    def _rule_negative_shift(self, key, role, scan: _ExprScan, value, stmt, force: bool = False) -> None:
        if role == "exec" and scan.has_compare and not _np_call(value, "where"):
            role = classify(None)  # v0.4: will_fill = low.shift(-1) <= limit is a flag, not a fill price
        if (scan.neg_shift_calls and not scan.has_compare and role not in ("label", "eval")
                and (classify(key) == "exec" or _tokens(key) & (ENTRY_T | EXEC_T | {"exit", "exits", "sell", "stop"}))
                and any(self._extreme_lead(c) for c in scan.neg_shift_calls)):
            # red-team round 10 (G7): entry = low.shift(-1). A limit order fills at the limit, not at the next bar's
            # low or high, which is known only when that bar closes.
            self.add("LA01", "high", scan.neg_shift_calls[0],
                     f"`{key}` is priced at the next bar's low or high, which is known only when that bar has closed. "
                     "A resting limit order fills at its limit price (or the open, if it gaps through).")
            return
        if not scan.neg_shift_calls or role in ("label", "pnl", "eval", "exec"):
            return
        if key in self._fill_masks and not force:
            self._fill_pending.append((key, role, scan, value, stmt))  # judged once every use is known
            return
        if (_tokens(key) & {"entry", "exit", "fill", "buy", "sell"} and _chain_shift(value) < 0
                and _is_price_name(_key(_chain_base(value))[0] or "")):
            return  # entry = df["open"].shift(-1): the next bar's fill price, which is how a fill should be booked
        call = scan.neg_shift_calls[0]
        in_compare = self.inside(call, (ast.Compare,)) is not None or (
            isinstance(value, ast.Call) and _func_name(value) == "where")
        if isinstance(call.func, ast.Name) and call.func.id in _PARAM_SHIFT and call.func.id not in _FWD_FUNCS:
            what = (f"`{key}` is built with `{call.func.id}(...)` called with a negative lag, which shifts the series "
                    "backwards: it reads future bars")
        elif isinstance(call.func, ast.Name):
            what = f"`{key}` is built with `{call.func.id}()`, a helper whose return value uses a negative shift (future bars)"
        else:
            what = f"`{key}` is built with `.{_attr(call)}(<negative>)`, which reads future bars"
        if role in ("signal", "feature"):
            self.add("LA01", "high", call, what + (" inside a comparison used as a signal." if in_compare else "."))
        else:
            # v0.3: a neutral name; high if the value reaches a signal, position or model input (resolved later).
            # A comparison is high unless the value is only ever passed as a label (fit's y, Dataset(label=...)).
            self._la01_pending.append((key, call, what, in_compare))

    def _rule_label_same_bar(self, key, role, scan, forward, is_ret, stmt) -> None:
        if role != "label" or forward or scan.neg_shift_calls:
            return
        if not is_ret:
            return
        if self.inside(stmt, (ast.FunctionDef, ast.For)) is not None:
            return
        self.add("LA08", "possible", stmt,
                 f"Label `{key}` is a same-bar return (no negative shift). If the features are computed at the "
                 "same close, the model is explaining today's return with today's data.")

    # -- LA10 (v0.3): forward data as an input vs. forward data as the outcome ----------------
    def _la10_payload(self, value: ast.AST, containers: bool = True) -> tuple[set[int], list[ast.AST], list[ast.Call], list[str]]:
        """Places in `value` where a forward column is used as the *outcome*, not as an input:

        * `X.groupby(...)[fwd].mean()` (and other aggregations): the average outcome per date/bucket;
        * `X.loc[mask, fwd]` / `X[mask][fwd]`: the outcome of the selected rows (the mask is scanned as an input);
        * `position * fwd_ret`: a decision made without forward data, booked against the forward return.

        None of these count inside a comparison or a `where` condition (that is a decision). Returns the ids of the
        outcome nodes, the masks to scan as inputs, the groupby calls of aggregated outcomes, and the outcome keys.
        """
        par: dict[int, ast.AST] = {}
        for p in ast.walk(value):
            for c in ast.iter_child_nodes(p):
                par[id(c)] = p

        def guarded(n: ast.AST) -> bool:
            c, p = n, par.get(id(n))
            while p is not None:
                if isinstance(p, ast.Compare):
                    return True
                if isinstance(p, ast.Call) and _func_name(p) == "where" and p.args and p.args[0] is c:
                    return True
                c, p = p, par.get(id(p))
            return False

        def fwd_key(n: ast.AST) -> str | None:
            k, _f = _key(n)
            ki = self.info.get(k) if k else None
            return k if ki is not None and ki.forward and not ki.lagged else None

        payload: set[int] = set()
        extra: list[ast.AST] = []
        groupbys: list[ast.Call] = []
        keys: list[str] = []
        self._booked: set[str] = set()  # keys booked as PnL against an earlier position (no finding at all)
        self._cont_elems: list[tuple[str | None, str]] = []  # (column name or None, forward key) stored in a new frame
        self._cont_order: list[tuple[str | None, str | None]] | None = None  # every column in order, None = unknown
        for n in ast.walk(value):
            if isinstance(n, ast.Subscript):
                k = fwd_key(n)
                p = par.get(id(n))
                if (k is None and isinstance(n.value, ast.Name) and not guarded(n)
                        and isinstance(n.slice, (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp))
                        and isinstance(p, ast.Attribute) and p.attr in AGG_T
                        and isinstance(par.get(id(p)), ast.Call) and par[id(p)].func is p):
                    wk = fwd_key(n.value)  # v0.4: fw[q > 0.9].mean(axis=1): the average outcome of the selected cells
                    if wk is not None and not self._ahead(n.slice):
                        payload.add(id(n))
                        keys.append(wk)
                        extra.append(n.slice)
                    continue
                if k is None or guarded(n):
                    continue
                gbs = [c for c in _chain(n.value) if _attr(c) in ("groupby", "resample")]
                if (gbs and isinstance(p, ast.Attribute) and p.attr in AGG_T
                        and isinstance(par.get(id(p)), ast.Call) and par[id(p)].func is p):
                    payload.add(id(n))
                    groupbys += gbs
                    keys.append(k)
                    extra.append(n.value)  # the frame and the groupby keys are inputs
                elif (isinstance(n.slice, ast.Tuple) and len(n.slice.elts) == 2 and isinstance(n.value, ast.Attribute)
                      and n.value.attr == "loc" and not isinstance(n.slice.elts[0], ast.Slice)):
                    payload.add(id(n))
                    keys.append(k)
                    extra.append(n.slice.elts[0])
                elif (isinstance(n.value, ast.Subscript) and _key(n.value)[0] is None
                      and not isinstance(n.value.slice, (ast.Slice, ast.List, ast.Tuple))):
                    payload.add(id(n))
                    keys.append(k)
                    extra.append(n.value.slice)
            elif (isinstance(n, ast.Call) and _func_name(n) in ("pivot_table", "pivot") and not guarded(n)
                  and any(_tokens(s) & BUCKET_T for kw in n.keywords if kw.arg in ("index", "columns")
                          for e in (kw.value.elts if isinstance(kw.value, (ast.List, ast.Tuple)) else [kw.value])
                          if (s := _const_str(e)))):
                # v0.4: df.pivot_table(index="date", columns="q", values="fwd_ret", aggfunc="mean"): the average
                # outcome per bucket, like groupby(["date", "q"])["fwd_ret"].mean()
                vals = _kw(n, "values")
                if vals is None and _func_name(n) == "pivot_table" and n.args:
                    vals = n.args[1] if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and \
                        n.func.value.id in ("pd", "pandas") and len(n.args) > 1 else n.args[0]
                vnames = {_const_str(e) for e in (vals.elts if isinstance(vals, (ast.List, ast.Tuple)) else [vals])}
                vk = [k for k in vnames if k and self.info.get(k) is not None and self.info[k].forward
                      and not self.info[k].lagged]
                if vk:
                    payload.add(id(n))
                    keys += vk
                    extra.append(n.func.value if isinstance(n.func, ast.Attribute) else n)
            elif (containers and isinstance(n, ast.Call) and _func_name(n) in ("DataFrame", "concat")
                  and not guarded(n)):
                # pd.DataFrame({"score": s, "fwd_ret": fwd}) / pd.concat([feats, fwd.rename("y")]): the forward data
                # is stored as its own column of a new frame. Not an input *here*; the frame is tracked as a container
                # and is reported where a forward column of it reaches a model or a signal (see _rule_la10).
                elems: list[tuple[str | None, ast.AST]] = []
                names = _kw(n, "keys")
                names = [_const_str(e) for e in names.elts] if isinstance(names, (ast.List, ast.Tuple)) else []
                for a in n.args:
                    if isinstance(a, ast.Dict):
                        elems += [(_const_str(kk), v) for kk, v in zip(a.keys, a.values) if v is not None]
                    elif isinstance(a, (ast.List, ast.Tuple)):
                        for i, e in enumerate(a.elts):
                            nm = names[i] if i < len(names) else None
                            if nm is None and isinstance(e, ast.Call) and _attr(e) == "rename" and e.args:
                                nm = _const_str(e.args[0])
                            elems.append((nm, e))
                order = []
                unpacked = any(kk is None for a in n.args if isinstance(a, ast.Dict) for kk in a.keys)  # {**base}
                for nm, e in elems:
                    k = fwd_key(_chain_base(e)) if _chain_shift(e) == 0 else None
                    order.append((nm, k))
                    if k is not None:
                        payload.add(id(e))
                        self._cont_elems.append((nm, k))
                if self._cont_elems:
                    axis = _kw(n, "axis")
                    rows = _func_name(n) == "concat" and not (axis is not None and _num(axis) == 1) and _const_str(axis) != "columns"
                    self._cont_order = None if (unpacked or rows) else order
            elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) and not guarded(n):
                for side, other in ((n.left, n.right), (n.right, n.left)):
                    base = side
                    if isinstance(side, ast.Call) and _attr(side) in ("fillna", "clip", "astype") and _chain_shift(side) == 0:
                        base = _chain_base(side)
                    k = fwd_key(base)
                    side_keys: list[str] = []
                    if k is not None:
                        ki = self.info[k]
                        if ki.is_signal_expr or not (classify(k) in ("label", "return") or ki.is_return):
                            continue
                        side_keys = [k]
                    elif isinstance(side, ast.BinOp) and not any(isinstance(x, ast.Compare) for x in ast.walk(side)):
                        # sig * (exit_px / entry_px - 1): a return between next-bar fill prices
                        srefs = _scan_expr(side).refs
                        sinfo = [(sk, self.info.get(sk), lag) for sk, lag in srefs]
                        if not srefs or not all(si is not None and si.ahead_after(lag) and not si.is_signal_expr and (
                                classify(sk) == "exec" or _is_price_name(sk)) for sk, si, lag in sinfo):
                            continue
                        side_keys = [sk for sk, _si, _l in sinfo]
                    else:
                        continue
                    orefs = _scan_expr(other).refs
                    oinfo = [(self.info.get(ok), lag) for ok, lag in orefs]
                    if (any(classify(ok) == "signal" for ok, _l in orefs)
                            and not any(oi is not None and oi.ahead_after(lag) for oi, lag in oinfo)):
                        payload.add(id(side))
                        keys += side_keys
                        self._booked.update(side_keys)
        return payload, extra, groupbys, keys

    def _rule_la10(self, key, role, frame, value, scan: _ExprScan, data_refs, stmt) -> tuple[bool, bool]:
        """LA10 in the data-flow pass. Returns (outcome, bucket_fwd) for the assigned key."""
        if role == "signal" and not _tokens(key) & (SIGNAL_T - AMBIG_SIGNAL_T):
            role = "unknown"  # long, long_short, trades: a leg, a long-format frame or a fill as often as a position
        if role == "signal":
            self._signal_roots.add(key)
        if frame is None and key in self._containers:
            # reassigned: rebuilt below if it still holds the forward column, otherwise no longer a container
            old = self._containers.pop(key)
            if any(isinstance(x, ast.Name) and x.id == key for x in ast.walk(value)):
                self._containers[key] = old
                keeps = self._container_survives(value, key)
                del self._containers[key]
                if keeps:
                    self._containers[key] = old
        if frame is None and self._containers and any(k in self._containers for _ki, _l, k in data_refs):
            self._derive_container(key, value, getattr(stmt, "lineno", 0))  # data = data.rename(...).dropna()
        if not any(ki is not None and ki.ahead_after(lag) and k != key for ki, lag, k in data_refs):
            return False, False
        payload, extra, groupbys, pkeys = self._la10_payload(
            value, containers=role != "feature" and not (_tokens(key) & {"x", "features", "feature", "feats"}))
        booked_only = bool(pkeys) and all(k in self._booked for k in pkeys)
        if self._cont_elems and frame is None:
            self._register_container(key, stmt, self._cont_elems, self._cont_order)
        refs = list(_scan_expr(value, skip=payload).refs) if payload else list(scan.refs)
        for e in extra:
            refs += _scan_expr(e).refs
        mapped: set[str] | None = None
        decision = []
        outcome_refs = list(pkeys)
        for k, lag in refs:
            ki = self.info.get(k)
            if ki is None or not ki.ahead_after(lag) or k == key or classify(k) == "pnl":
                continue
            if k in self._containers and not self._container_survives(value, k):
                continue  # data.drop(columns=['target']) / data[['mom']]: the container's forward columns are gone
            if ki.outcome:
                if mapped is None:
                    mapped = self._mapped_back(value)
                # an aggregate of outcomes stays an outcome unless it is put back onto rows or compared
                if not (k in mapped or scan.has_compare or (frame is not None and ki.frame is None)):
                    outcome_refs.append(k)
                    continue
            decision.append((ki, lag, k))
        # v0.3 rt5: an outcome that decides a position is lookahead whatever the name
        self._la10_key = key
        # v0.4: r[filled] / np.where(filled, r, 0): the fill model selects the trades it books. Only the fill mask is
        # exempt; any other forward factor in the same product (size from the next bar's low, rt10 G2) still counts.
        fill_ok = (lambda k: k in self._fill_masks and self._fill_mask_ok(k)) if self._fill_masks else (lambda k: False)
        leak = self._pnl_position_leak(value, skip=fill_ok)
        if leak is None and self._fill_masks and any(fill_ok(k) for k, _l in scan.refs):
            ext = [c for c in scan.neg_shift_calls if self._extreme_lead(c)]
            if ext:
                # rt11 G9: (best["high"] / limit_px - 1)[filled]: the trade is priced at the next bar's high or low.
                # Only the comparison inside the fill mask is exempt, not a price taken from that bar.
                self.add("LA01", "high", ext[0],
                         f"`{key}` prices the trade at the next bar's low or high, which is known only when that bar has "
                         "closed. A resting limit order fills at its limit price (or the open, if it gaps through).")
            return True, False
        if leak is not None:
            lead_k, sev = leak
            what = ("sizes or signs the position in this product with a return" if sev == "high" else
                    "selects the returns in this line (if it is a trading rule, the position knows the outcome; "
                    "an up/down-capture or conditional statistic on the realised market is fine)")
            self.add("LA10", sev, stmt,
                     f"`{lead_k}` is built from future data (line {self.info[lead_k].line}) and {what}.")
            decision.append((self.info[lead_k], 0, lead_k))
            self._edges.setdefault(key, set()).add(lead_k)
            return False, False
        if (outcome_refs and not booked_only and role == "signal" and _tokens(key) & (SIGNAL_T - AMBIG_SIGNAL_T)):
            k = outcome_refs[0]
            self.add("LA10", "high", stmt,
                     f"`{key}` is built from `{k}`, an outcome that contains future data (line {self.info[k].line}). "
                     "A signal, position or weight decided from it knows the result.")
            self._edges.setdefault(key, set()).update(outcome_refs)
            return False, False
        # buckets formed from forward data, then used to group the outcome: the spread is built in
        for gb in groupbys:
            for a in list(gb.args) + [kw.value for kw in gb.keywords]:
                for x in ast.walk(a):
                    nm = _const_str(x) or (x.id if isinstance(x, ast.Name) else None) or (
                        _key(x)[0] if isinstance(x, ast.Subscript) else None)
                    bi = self.info.get(nm) if nm else None
                    if bi is not None and bi.bucket_fwd:
                        self.add("LA10", "medium", stmt,
                                 f"`{key}` averages the outcome within `{nm}` buckets that were formed from future data "
                                 f"(line {bi.line}). Sorting on the outcome builds the spread in.")
                        break
        if decision:
            self._edges.setdefault(key, set()).update(k for _ki, _l, k in decision)
            self._rule_feature_from_label(key, role, decision, stmt)
            return False, role in ("eval", "unknown") and bool(_tokens(key) & BUCKET_T)
        if booked_only and len(outcome_refs) == len(pkeys):
            return True, False  # position * return between fill prices: the execution model, booked as PnL
        if outcome_refs and role in ("signal", "feature"):
            k = outcome_refs[0]
            self.add("LA10", "possible", stmt,
                     f"`{key}` is built from the outcome `{k}` (future data, line {self.info[k].line}): an average, a "
                     "selection or a PnL. That is fine for evaluating a strategy; if `{key}` is then used as a signal "
                     "or feature, it is target leakage.".replace("{key}", key))
        return bool(outcome_refs), False

    # -- containers: frames built from a dict/list of columns, some of them forward-looking ---------------------
    def _register_container(self, key: str, stmt: ast.AST, elems: list[tuple[str | None, str]],
                            order: list[tuple[str | None, str | None]] | None = None) -> None:
        self._set_container(key, getattr(stmt, "lineno", 0), elems[0][1], order if order is not None else None,
                            [nm for nm, _k in elems])

    def _set_container(self, key: str, line: int, src: str, order, fwd_names: list[str | None], lag: int = 0) -> None:
        """order: every column as (name or None, forward key or None), or None if positions are unknown.
        lag: bars the whole frame has been shifted since the forward columns were stored."""
        self._containers[key] = {"cols": {nm for nm in fwd_names if nm}, "unnamed": any(nm is None for nm in fwd_names),
                                 "order": order, "line": line, "src": src, "lag": lag}
        for nm in fwd_names:  # named forward columns of the new frame, for column selections and drop lists
            if nm and nm not in self.info and src in self.info:
                s = self.info[src]
                self.info[nm] = KeyInfo(nm, key, line, classify(nm), forward=True, lead=s.lead, retlike=s.retlike)

    # Frame methods that keep every column name (they filter, reorder or fill rows, or change values).
    _NAME_KEEPING_OPS = {"dropna", "fillna", "copy", "reset_index", "set_index", "sort_index", "sort_values", "astype",
                         "to_numpy", "ffill", "bfill", "head", "tail", "assign", "replace", "clip", "round",
                         "infer_objects", "interpolate", "shift", "abs", "loc", "iloc"}

    def _derive_container(self, key: str, value: ast.AST, line: int) -> None:
        """data2 = data.dropna().rename(columns={'fwd': 'mom2'}): data2 holds the same forward columns, renamed.

        The default is inverted: any frame-level call on a container other than one that keeps the names
        (`_NAME_KEEPING_OPS`) or a `rename(columns={...})` with literal names makes the column names UNKNOWN
        (`add_prefix`, `set_axis`, `rename(mapper, axis=1)`, `rename(columns=lambda ...)`, ...). Unknown names
        survive any later explicit column list, so the forward column keeps counting as present."""
        node, steps = value, []
        while True:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                steps.append(("call", node))
                node = node.func.value
            elif isinstance(node, ast.Attribute) and node.attr == "values":
                node = node.value
            elif isinstance(node, ast.Attribute) and node.attr == "T":
                steps.append(("unknown", node))  # a transpose: positions can't be followed
                node = node.value
            elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr in ("iloc", "loc"):
                sl = node.slice
                if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:  # [rows, cols]
                    steps.append(("pos" if node.value.attr == "iloc" else "cols", sl.elts[1]))
                node = node.value.value
            elif isinstance(node, ast.Subscript):
                sl = node.slice
                if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:  # numpy arr[rows, cols]
                    steps.append(("pos", sl.elts[1]))
                elif _const_str(sl) is not None:
                    return  # data['zfwd']: a single column, not a frame
                elif not self._selects_rows(sl):
                    # a column list: literal, or anything else (sorted(data.columns), data.columns[::-1], a name)
                    steps.append(("cols", sl))
                # a slice, a comparison or a row-mask name selects rows: names and order unchanged
                node = node.value
            else:
                break
        if not (isinstance(node, ast.Name) and node.id in self._containers) or node.id == key and not steps:
            return
        c = self._containers[node.id]
        order = list(c["order"]) if c["order"] is not None else None
        names = sorted(c["cols"]) + ([None] if c["unnamed"] else [])
        lag = c.get("lag", 0)

        def select(lst_node):  # a column list: literal names set the order, anything else makes it unknown
            nonlocal order, names
            lst = [_const_str(e) for e in lst_node.elts] if isinstance(lst_node, (ast.List, ast.Tuple)) else None
            if lst is None or not all(lst):
                order = None  # a column list we can't read: keep every name, forget positions
                return
            if order is not None and all(any(n == x for n, _f in order) for x in lst) and not any(
                    n is None and f for n, f in order):
                order = [next((n, f) for n, f in order if n == x) for x in lst]
                names = [n for n, f in order if f]
            else:
                names = [n for n in names if n is None or n in lst]
                order = None

        for kind, s in reversed(steps):  # innermost first
            if kind == "cols":
                select(s)
                continue
            if kind == "unknown":
                order = None
                continue
            if kind == "pos":
                pl = self._pos_list(s, len(order)) if order is not None else None
                if pl is not None:
                    order = [order[i] for i in pl]
                    names = [n for n, f in order if f]
                else:
                    order = None  # a positional selection we can't resolve: positions unknown, names kept
                continue
            call = s
            op = call.func.attr
            axis = _kw(call, "axis")
            col_axis = axis is not None and (_num(axis) == 1 or _const_str(axis) == "columns")
            spec = _kw(call, "columns")
            if op == "shift" and not col_axis:  # data.shift(5): every column is lagged 5 bars
                amt = _shift_amount(call)
                lag = lag + amt if amt > 0 else lag
                continue
            if op in ("sort_index", "sort_values") and col_axis:
                order = None  # columns reordered
                continue
            if op in ("reindex", "reindex_like", "filter", "T", "transpose", "swapaxes", "reorder_levels"):
                if op == "reindex" and spec is not None:
                    select(spec)
                else:
                    order = None
                    names = names if op != "filter" else names
                continue
            if op == "assign":
                order = None
                continue
            if op in self._NAME_KEEPING_OPS:
                continue
            if op == "drop":
                dspec = spec or (call.args[0] if call.args else _kw(call, "labels"))
                d = self._colset(dspec) if dspec is not None else None
                if d is not None:  # data.drop(columns=['target']): the named forward column is gone
                    names = [n for n in names if not (n and d(n) is True)]
                    if order is not None:
                        order = [(n, f) for n, f in order if not (n and d(n) is True)]
                continue
            if (op == "rename" and not call.args and isinstance(spec, ast.Dict)
                    and all(k is not None and _const_str(k) and _const_str(v) for k, v in zip(spec.keys, spec.values))):
                m = {_const_str(k): _const_str(v) for k, v in zip(spec.keys, spec.values)}
                names = [m.get(n, n) if n else n for n in names]
                if order is not None:
                    order = [(m.get(n, n) if n else n, f) for n, f in order]
            elif op in ("rename", "add_prefix", "add_suffix", "set_axis"):
                names = [None]  # names changed in a way we can't read; positions stay
                if order is not None:
                    order = [(None, f) for _n, f in order]
            else:
                names = [None]  # any other frame operation: names and positions unknown
                order = None
        self._set_container(key, line, c["src"], order, names, lag)

    def _selects_rows(self, sl: ast.AST) -> bool:
        """`frame[sl]` selects rows (a slice, a comparison or boolean combination, a row-mask name or call),
        as opposed to columns (a list, a name bound to a list, `sorted(df.columns)`, `df.columns[::-1]`...)."""
        if isinstance(sl, (ast.Slice, ast.Compare, ast.BoolOp)):
            return True
        if isinstance(sl, ast.UnaryOp) and isinstance(sl.op, (ast.Invert, ast.Not)):
            return True
        if isinstance(sl, ast.BinOp) and isinstance(sl.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
            return True
        if isinstance(sl, ast.Call) and _attr(sl) in ("isin", "between", "notna", "isna", "notnull", "isnull", "duplicated"):
            return not any(isinstance(x, ast.Attribute) and x.attr == "columns" for x in ast.walk(sl))
        if isinstance(sl, ast.Name):
            return self._is_row_mask(sl) or bool(self._date_bound(sl)) or (
                sl.id in self.info and self.info[sl.id].is_signal_expr)
        return False

    def _pos_list(self, sel: ast.AST, n: int) -> list[int] | None:
        """Positions picked by a positional column selector, in the order picked (`::-1` reverses)."""
        idx = list(range(n))
        if isinstance(sel, ast.Slice):
            vals = [_num(x) if x is not None else None for x in (sel.lower, sel.upper, sel.step)]
            if any(x is None and y is not None for x, y in zip(vals, (sel.lower, sel.upper, sel.step))):
                return None
            return idx[slice(*[int(v) if v is not None else None for v in vals])]
        v = _num(sel)
        if v is not None and v == int(v) and -n <= int(v) < n:
            return [idx[int(v)]]
        if isinstance(sel, (ast.List, ast.Tuple)) and sel.elts and all(_num(e) is not None for e in sel.elts):
            return [idx[int(_num(e))] for e in sel.elts if -n <= int(_num(e)) < n]
        return None

    def _cont_ahead(self, c: dict, fkey: str | None) -> bool:
        """A forward column of a container still looks ahead after the container's own lag (data.shift(5))."""
        if not fkey or fkey not in self.info:
            return True
        return _lead_after_lag(self.info[fkey].lead_bars(), c.get("lag", 0)) > 0

    def _container_columns_assigned(self, target: ast.AST, value: ast.AST, stmt: ast.AST) -> bool:
        """data.columns = ['mom', 'target']: rename by position (or forget the names if we can't)."""
        if not (isinstance(target, ast.Attribute) and target.attr == "columns" and isinstance(target.value, ast.Name)
                and target.value.id in self._containers):
            return False
        name = target.value.id
        c = self._containers[name]
        new = [_const_str(e) for e in value.elts] if isinstance(value, (ast.List, ast.Tuple)) else None
        if c["order"] is not None and new is not None and len(new) == len(c["order"]) and all(new):
            order = [(n, f) for n, (_o, f) in zip(new, c["order"])]
            self._set_container(name, c["line"], c["src"], order, [n for n, f in order if f], c.get("lag", 0))
        else:
            self._set_container(name, c["line"], c["src"],
                                [(None, f) for _n, f in c["order"]] if c["order"] is not None else None, [None],
                                c.get("lag", 0))
        return True

    def _positions(self, sel: ast.AST, n: int) -> set[int] | None:
        """Column positions picked by a positional selector (`:-1`, `-1`, `[0, 2]`, `1:3`), or None if unknown."""
        idx = list(range(n))
        if isinstance(sel, ast.Slice):
            vals = [_num(x) if x is not None else None for x in (sel.lower, sel.upper, sel.step)]
            if any(x is None and y is not None for x, y in zip(vals, (sel.lower, sel.upper, sel.step))):
                return None
            return set(idx[slice(*[int(v) if v is not None else None for v in vals])])
        v = _num(sel)
        if v is not None and v == int(v) and -n <= int(v) < n:
            return {idx[int(v)]}
        if isinstance(sel, (ast.List, ast.Tuple)) and sel.elts and all(_num(e) is not None for e in sel.elts):
            return {idx[int(_num(e))] for e in sel.elts if -n <= int(_num(e)) < n}
        return None

    def _container_survives(self, value: ast.AST, cont: str) -> bool:
        """Does any forward column of container `cont` survive the selections and drops applied to it in `value`?"""
        c = self._containers[cont]
        par: dict[int, ast.AST] = {}
        for p in ast.walk(value):
            for ch in ast.iter_child_nodes(p):
                par[id(ch)] = p
        found = False
        order = c["order"]
        for n in ast.walk(value):
            if not (isinstance(n, ast.Name) and n.id == cont):
                continue
            found = True
            keep, dropped, pos, top = None, None, None, n
            while True:
                p = par.get(id(top))
                if isinstance(p, ast.Attribute) and p.value is top:
                    top = p
                elif isinstance(p, ast.Call) and p.func is top:
                    if _attr(p) == "drop":
                        spec = _kw(p, "columns") or (p.args[0] if p.args else _kw(p, "labels"))
                        d = self._colset(spec) if spec is not None else None
                        dropped = d if dropped is None or d is None else (lambda x, a=dropped, b=d: _or3([a(x), b(x)]))
                    top = p
                elif isinstance(p, ast.Subscript) and p.value is top:
                    sl = p.slice
                    attr = top.attr if isinstance(top, ast.Attribute) else None
                    two = isinstance(sl, ast.Tuple) and len(sl.elts) == 2
                    if attr == "loc" and two:
                        cols = sl.elts[1]
                        if not isinstance(cols, (ast.Slice, ast.Compare)):
                            kp = self._colset(cols)
                            if kp is not None:
                                keep = _merge_keep(keep, kp)
                    elif two and attr in ("iloc", "values", None) and order is not None:
                        ps = self._positions(sl.elts[1], len(order))
                        if ps is not None:
                            pos = ps if pos is None else {order_i for order_i in pos if order_i in ps}
                    elif attr is None and not two and not isinstance(sl, (ast.Slice, ast.Compare)):
                        kp = self._colset(sl)
                        if kp is not None:
                            keep = _merge_keep(keep, kp)
                    top = p
                else:
                    break
            if order is not None:
                for i, (nm, fk) in enumerate(order):
                    if fk is None or (pos is not None and i not in pos) or not self._cont_ahead(c, fk):
                        continue
                    if nm is None:
                        return True  # an unknown name survives any explicit column list or drop
                    if (keep(nm) if keep else True) is not False and (dropped(nm) if dropped else False) is not True:
                        return True
                continue
            if not self._cont_ahead(c, c["src"]):
                continue  # data.shift(5) with a 5-bar lead: no longer forward-looking
            for col in c["cols"]:
                if (keep(col) if keep else True) is not False and (dropped(col) if dropped else False) is not True:
                    return True
            if c["unnamed"]:
                return True  # an unknown name survives any explicit column list or drop
        return not found

    def _container_model_uses(self) -> None:
        """A container whose forward columns survive into the input of fit/predict or a training dataset."""
        if not self._containers:
            return
        for n in self._all:
            if not (isinstance(n, ast.Call) and (_attr(n) in FIT_ATTRS or _func_name(n) in TRAIN_DATA_CTORS)):
                continue
            args = ([n.args[0]] if n.args else []) + [kw.value for kw in n.keywords if kw.arg in ("X", "data")]
            for a in args:
                for nm in self.names_in(a) & set(self._containers):
                    if self._container_survives(a, nm):
                        c = self._containers[nm]
                        self.add("LA10", "high", n,
                                 f"`{nm}` (line {c['line']}) stores the forward-looking `{c['src']}` as a column, and it is "
                                 f"passed to `{_func_name(n)}(...)` with that column still in it.")
                        break

    def _pnl_position_leak(self, value: ast.AST, skip=lambda k: False) -> str | None:
        """In a product of a position and a return, a factor built from future data that is *not* itself a return
        (a rolling Sharpe of forward PnL, sign of a forward average, rank of forward returns): the position is decided
        by the outcome. Returns that key, or None."""
        par: dict[int, ast.AST] = {}
        for p in ast.walk(value):
            for ch in ast.iter_child_nodes(p):
                par[id(ch)] = p

        def is_prod(x):
            return isinstance(x, ast.BinOp) and isinstance(x.op, (ast.Mult, ast.MatMult))

        def guarded(x):
            p = par.get(id(x))
            while p is not None:
                if isinstance(p, ast.Compare):
                    return True
                p = par.get(id(p))
            return False

        def flat(x):
            return flat(x.left) + flat(x.right) if is_prod(x) else [x]

        def retish(f, refs):
            if any((self.info.get(k) is not None and self.info[k].retlike) or classify(k) in ("label", "return", "pnl")
                   or _is_price_name(k) for k, _l in refs):
                return True
            for c in ast.walk(f):  # df.pivot(..., values='fwd_ret'), close.pct_change()
                if isinstance(c, ast.Call) and _attr(c) in ("pct_change", "diff"):
                    return True
                if isinstance(c, ast.Call):
                    for kw in c.keywords:
                        s = _const_str(kw.value)
                        if kw.arg == "values" and s and (classify(s) in ("label", "return") or (
                                self.info.get(s) is not None and self.info[s].retlike)):
                            return True
            return False

        def neg_of(a, b):  # b is -a
            return isinstance(b, ast.UnaryOp) and isinstance(b.op, ast.USub) and ast.dump(b.operand) == ast.dump(a)

        def call_factors(c):
            """Products written as calls: x.mul(w), np.multiply(x, w), np.dot(w, r), np.where(cond, r, -r / 0),
            r.where(cond, -r / 0). For the where forms, the condition is the position."""
            if not isinstance(c, ast.Call):
                return None
            a = _attr(c)
            if a in ("mul", "multiply") and isinstance(c.func, ast.Attribute) and c.args and not _is_np(c.func.value):
                return [c.func.value, c.args[0]]
            if _np_call(c, "multiply", "dot", "matmul", "inner") and len(c.args) >= 2:
                return list(c.args[:2])
            if _np_call(c, "where") and len(c.args) == 3:
                r, other = c.args[1], c.args[2]
                if neg_of(r, other) or _num(other) == 0:
                    return [c.args[0], r]
                if neg_of(other, r) or _num(r) == 0:
                    return [c.args[0], other]
            if a == "where" and isinstance(c.func, ast.Attribute) and not _is_np(c.func.value):
                cond = c.args[0] if c.args else _kw(c, "cond")
                other = c.args[1] if len(c.args) > 1 else _kw(c, "other")
                if cond is not None and (other is None or _num(other) == 0 or neg_of(c.func.value, other)):
                    return [cond, c.func.value]
            return None

        def inline_pos(f):
            """np.sign(mkt), fwd.rolling(5).mean(), fwd.rank(), np.where(fwd > 0, ...) written inside the factor:
            a transform of forward data, so the factor is a position, not a return."""
            out = []
            for c in ast.walk(f):
                if not isinstance(c, ast.Call):
                    continue
                if _np_call(c, "where"):
                    src = c.args[0] if c.args else None
                elif _func_name(c) in INLINE_TRANSFORM_T:
                    src = c
                else:
                    continue
                if src is None:
                    continue
                out += [k for k, lag in _scan_expr(src).refs
                        if self.info.get(k) is not None and self.info[k].ahead_after(lag)]
            return out

        tkey = getattr(self, "_la10_key", None)
        strat_target = bool(tkey) and bool(_tokens(tkey) & STRAT_T) and classify(tkey) == "pnl"

        def aggregated(x):  # the selection is reduced to a statistic (bench[mkt > 0].mean()): a capture ratio
            p = par.get(id(x))
            while p is not None:
                if isinstance(p, ast.Attribute) and p.attr in AGG_T and isinstance(par.get(id(p)), ast.Call):
                    return True
                if isinstance(p, ast.Call) and _func_name(p) in AGG_T | {"mean", "nanmean", "average"}:
                    return True
                p = par.get(id(p))
            return False

        for m in ast.walk(value):
            if (strat_target and isinstance(m, ast.Subscript) and not guarded(m)
                    and isinstance(m.slice, (ast.Compare, ast.BoolOp)) and self._ahead(m.slice) and not aggregated(m)):
                # v0.4: strat = bench[mkt_fwd > 0] (also frame.query("mkt_fwd > 0")["b"]): a strategy series kept
                # only on rows chosen by future data. A capture statistic (aggregated, or under an evaluation name)
                # is not counted, so this is medium.
                brefs = _scan_expr(m.value).refs
                p = par.get(id(m))
                outer = p if isinstance(p, ast.Subscript) and p.value is m and _const_str(p.slice) else None
                orefs = _scan_expr(outer).refs if outer is not None else []
                if (brefs and retish(m.value, brefs)) or (orefs and retish(outer, orefs[:1])):
                    k = next((k for k, lag in _scan_expr(m.slice).refs
                              if self.info.get(k) is not None and self.info[k].ahead_after(lag)), None)
                    if k is not None and not skip(k):
                        return k, "medium"
            if (_np_call(m, "copysign") and len(m.args) >= 2 and not guarded(m)
                    and classify(tkey) not in ("eval", "label")):
                # v0.4: np.copysign(r, s) is |r| * sign(s): the sign argument is the position. Future data there is a
                # position decided by the outcome, whatever its name or form. A sign that is the very forward return
                # also in the magnitude (copysign(|pred - fwd|, fwd), a signed error) is not a position.
                mag, sgn = m.args[0], m.args[1]
                mrefs = _scan_expr(mag).refs
                plain = _key(_chain_base(sgn))[0] if not _scan_expr(sgn).neg_shift_calls and all(
                    _attr(c) in ("fillna", "astype", "to_numpy", "values") for c in _chain(sgn)) else None
                pos = [k for k, lag in _scan_expr(sgn).refs if self.info.get(k) is not None and self.info[k].ahead_after(lag)]
                pos = [k for k in pos if not skip(k)]
                if pos and mrefs and retish(mag, mrefs) and not (plain and plain in {k for k, _l in mrefs}):
                    return pos[0], "high"
            if isinstance(m, ast.Subscript) and not guarded(m):
                # bench[long] / bench.loc[short] / bench[~long]: a return selected by a position-named mask that was
                # built from future data (rt8). An inline comparison (bench[mkt > 0], up-capture) is not counted.
                base = m.value.value if isinstance(m.value, ast.Attribute) and m.value.attr in ("loc", "iloc") else m.value
                sl = m.slice
                mask = sl.elts[0] if isinstance(sl, ast.Tuple) and sl.elts else sl
                for _ in range(6):  # ~long, long.values, long.to_numpy(), long[long].index: the same mask
                    if isinstance(mask, ast.UnaryOp) and isinstance(mask.op, (ast.Invert, ast.Not)):
                        mask = mask.operand
                    elif isinstance(mask, ast.Attribute) and mask.attr in ("values", "index"):
                        mask = mask.value
                    elif isinstance(mask, ast.Call) and _attr(mask) in ("to_numpy", "astype", "fillna", "to_list", "tolist"):
                        mask = mask.func.value
                    elif (isinstance(mask, ast.Subscript) and isinstance(mask.value, ast.Name)
                          and isinstance(mask.slice, ast.Name) and mask.slice.id == mask.value.id):
                        mask = mask.value  # long[long]
                    else:
                        break
                mk = _key(mask)[0] if isinstance(mask, (ast.Name, ast.Subscript, ast.Attribute)) else None
                mi = self.info.get(mk) if mk else None
                if mi is not None and mi.ahead_after(0) and not isinstance(mask, ast.Slice) and (
                        mi.is_signal_expr or _tokens(mk) & SIGNAL_T):
                    brefs = _scan_expr(base).refs
                    if brefs and mk not in {k for k, _l in brefs} and retish(base, brefs):
                        # a position name (long, short, regime) is high; a neutral name (ok, up) can be an
                        # up/down-capture filter, so medium. An inline comparison is not counted at all.
                        if not skip(mk):
                            return mk, ("high" if _tokens(mk) & SIGNAL_T else "medium")
            cf = call_factors(m)
            if cf is not None and not guarded(m):
                facts = [(f, _scan_expr(f).refs) for f in cf]
                facts = [(f, r) for f, r in facts if r]
                if len(facts) == 2:
                    rets = [f for f, r in facts if retish(f, r)]
                    for f, r in facts:
                        pos = [k for k, lag in r if self.info.get(k) is not None and self.info[k].ahead_after(lag)
                               and not self.info[k].retlike and classify(k) not in ("label", "return", "pnl", "exec")] + inline_pos(f)
                        pos = [k for k in pos if not skip(k)]
                        if pos and any(g is not f for g in rets):
                            return pos[0], "high"
                continue
            if not is_prod(m) or is_prod(par.get(id(m))) or guarded(m):
                continue
            facts = [(f, _scan_expr(f).refs) for f in flat(m)]
            facts = [(f, r) for f, r in facts if r]
            if len(facts) < 2:
                continue
            rets = [f for f, r in facts if retish(f, r)]
            for f, r in facts:
                pos = [k for k, lag in r if self.info.get(k) is not None and self.info[k].ahead_after(lag)
                       and not self.info[k].retlike and classify(k) not in ("label", "return", "pnl", "exec")] + inline_pos(f)
                pos = [k for k in pos if not skip(k)]
                if pos and any(g is not f for g in rets):
                    return pos[0], "high"
        return None

    def _mapped_back(self, value: ast.AST) -> set[str]:
        """Keys passed to calls that put a series back onto rows (`map`, `merge`, `reindex`, `transform`...)."""
        out: set[str] = set()
        for c in ast.walk(value):
            if isinstance(c, ast.Call) and _attr(c) in MAPBACK_T:
                for a in list(c.args) + [kw.value for kw in c.keywords]:
                    for x in ast.walk(a):
                        k = _key(x)[0] if isinstance(x, (ast.Name, ast.Subscript, ast.Attribute)) else None
                        if k:
                            out.add(k)
        return out

    def _rule_feature_from_label(self, key, role, data_refs, stmt) -> None:
        if role not in ("signal", "feature", "unknown"):
            return
        val = getattr(stmt, "value", None)
        if isinstance(val, ast.Call) and re.search(r"(_score|_error|_loss|^corr|corrcoef|spearmanr|pearsonr)$",
                                                   _func_name(val) or ""):
            return  # roc_auc_score(test["target"], ...): an evaluation of the label, not a feature
        for ki, lag, k in data_refs:
            if ki is None or not ki.ahead_after(lag) or k == key:
                continue
            if classify(k) == "pnl":
                continue
            if role == "unknown":
                # v0.3: the name says nothing, so the severity depends on use (resolved in _resolve_la10_pending):
                # medium if it reaches a fit/predict input, a feature matrix or a feature builder's return value
                self._la10_pending.append((key, stmt, k, ki.line))
            else:
                self.add("LA10", "high", stmt,
                         f"`{key}` is computed from `{k}`, which contains future data (defined on line {ki.line}).")
            return

    def _feature_uses(self) -> set[str]:
        """Names and column strings that reach a model input: the first argument of fit/predict, the right-hand side
        of a feature-named assignment (minus drop lists and exclusion conditions), a feature builder's return value,
        or a multiplication inside a PnL (a comparison used as a position)."""
        if hasattr(self, "_feat_uses"):
            return self._feat_uses
        out: set[str] = set()
        self._lit_uses: dict[str, set[str]] = {}  # v0.4: frames passed to fit only through a literal column list
        self._feat_value_uses: set[str] = set()  # v0.4: names read for their values (not just .index/.shape/len)
        idx_names: set[str] = set()
        val_names: set[str] = set()

        def lit_cols(x: ast.AST):
            """(frame name, listed columns) for data[["a", "b"]] / data.loc[rows, ["a", "b"]]; else None."""
            sl, base = x.slice, x.value
            if isinstance(base, ast.Attribute) and base.attr == "loc":
                if not (isinstance(sl, ast.Tuple) and len(sl.elts) == 2):
                    return None
                sl, base = sl.elts[1], base.value
            if not (isinstance(sl, ast.List) and sl.elts and all(_const_str(e) for e in sl.elts)):
                return None
            b = _chain_base(base)
            return (b.id, {_const_str(e) for e in sl.elts}) if isinstance(b, ast.Name) else None

        def collect(node: ast.AST, skip: set[int] = frozenset(), fit: bool = False) -> None:
            stack = [node]
            while stack:
                x = stack.pop()
                if isinstance(x, (ast.Compare, ast.comprehension)) or id(x) in skip:
                    continue
                if fit and isinstance(x, ast.Subscript):
                    lc = lit_cols(x)
                    if lc is not None:
                        self._lit_uses.setdefault(lc[0], set()).update(lc[1])
                        out.update(lc[1])
                        self._feat_value_uses.update(lc[1])
                        if isinstance(x.slice, ast.Tuple):
                            stack.append(x.slice.elts[0])  # the row selector is still read
                        continue
                if isinstance(x, ast.Call) and _attr(x) == "drop":
                    stack.append(x.func.value)
                    continue
                if isinstance(x, ast.Name):
                    out.add(x.id)
                    (idx_names if self._index_only_name(x) else val_names).add(x.id)
                s = _const_str(x)
                if s:
                    out.add(s)
                    self._feat_value_uses.add(s)
                stack.extend(ast.iter_child_nodes(x))

        for n in self._all:
            if isinstance(n, ast.Call) and _attr(n) in ("nlargest", "nsmallest", "idxmax", "idxmin", "argmax", "argmin"):
                collect(n.func.value)  # corrs.nlargest(5).index: features or names picked by a forward-looking score
            elif isinstance(n, ast.Call) and _attr(n) in FIT_ATTRS:
                if n.args:
                    collect(n.args[0], fit=True)
                if _kw(n, "X") is not None:
                    collect(_kw(n, "X"), fit=True)
            elif isinstance(n, ast.Assign) and len(n.targets) == 1:
                tk = _key(n.targets[0])[0]
                if tk and (classify(tk) == "feature" or _tokens(tk) & {"x", "features", "feature", "feats"}):
                    saved = (getattr(self, "_booked", set()), self._cont_elems, getattr(self, "_cont_order", None))
                    skip = self._la10_payload(n.value, containers=False)[0]  # outcome nodes: not model inputs
                    self._booked, self._cont_elems, self._cont_order = saved
                    collect(n.value, skip)
                elif tk and classify(tk) == "pnl":
                    for m in ast.walk(n.value):
                        if isinstance(m, ast.BinOp) and isinstance(m.op, ast.Mult):
                            for side in (m.left, m.right):
                                k = _key(_chain_base(side))[0]
                                if k and self.info.get(k) is not None and self.info[k].is_signal_expr:
                                    out.add(k)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _tokens(n.name) & {"feat", "feature", "features"}:
                for r in ast.walk(n):
                    if isinstance(r, ast.Return) and r.value is not None:
                        collect(r.value)
        self._feat_uses = out
        self._feat_value_uses = out - (idx_names - val_names)
        return out

    def _partial_signal_write(self, target: ast.AST, value: ast.AST, stmt: ast.AST) -> None:
        """`pos[1:] = np.where(up[:-1], 1, -1)` / `pos[cond] = 1`: a position written through a slice or mask.
        The written name gets edges to the forward-looking keys read in the value or the mask."""
        if not isinstance(target, ast.Subscript):
            return
        base = target.value
        while isinstance(base, (ast.Subscript, ast.Attribute)) and not isinstance(base, ast.Name):
            base = base.value
        if not isinstance(base, ast.Name):
            return
        name = base.id
        # strat[tilt < 0] *= -1 / strat.loc[tilt < 0] = -bench: a return series flipped or overwritten on rows chosen
        # by forward data is a position decided by the outcome (v0.3 rt7)
        bi = self.info.get(name)
        if bi is not None and (bi.retlike or classify(name) in ("pnl", "return")):
            sl = target.slice
            mask = sl.elts[0] if isinstance(sl, ast.Tuple) and sl.elts else sl
            ahead = [k for k, lag in _scan_expr(mask).refs
                     if k != name and self.info.get(k) is not None and self.info[k].ahead_after(lag)]
            if ahead and not isinstance(mask, ast.Slice):
                k = ahead[0]
                self.add("LA10", "high", stmt,
                         f"`{name}` is flipped or overwritten on rows chosen by `{k}`, which is built from future data "
                         f"(line {self.info[k].line}): the position knows the outcome it is paid on.")
                return
        refs = _scan_expr(value).refs + _scan_expr(target.slice).refs
        fwd = [(self.info[k], lag, k) for k, lag in refs
               if k != name and k in self.info and self.info[k].ahead_after(lag) and classify(k) != "pnl"]
        if not fwd:
            return
        self._edges.setdefault(name, set()).update(k for _ki, _l, k in fwd)
        if classify(name) == "signal":
            self._signal_roots.add(name)
            self._rule_feature_from_label(name, "signal", fwd, stmt)

    def _signal_reach(self) -> dict[str, str]:
        """Keys whose (forward-looking) value reaches a signal or position, mapped to that signal's name. Computed once
        the data-flow pass is done (edges no longer change)."""
        if getattr(self, "_reach_cache", None) is not None:
            return self._reach_cache
        reach: dict[str, str] = {}
        stack = [(r, r) for r in self._signal_roots]
        while stack:
            k, root = stack.pop()
            for src in self._edges.get(k, ()):
                if src not in reach:
                    reach[src] = root
                    stack.append((src, root))
        if getattr(self, "_flow_done", False):
            self._reach_cache = reach
        return reach

    def _eval_only(self, key: str) -> bool:
        """Every statement that reads `key` is an evaluation: a bare expression (`print((pred == up).mean())`) or an
        assignment to evaluation/PnL names (`hit = ...`, `acc = ...`). Not used at all also counts."""
        idx = self._ref_index()
        positions = list(idx.get(key, []))
        for i, st in enumerate(self._ref_stmts):  # df['up'] used through its column name
            if any(_const_str(x) == key for x in ast.walk(st.value)):
                positions.append(i)
        for i in set(positions):
            st = self._ref_stmts[i]
            if isinstance(st, ast.Expr):
                continue
            targets = st.targets if isinstance(st, ast.Assign) else [st.target]
            keys = [_key(t)[0] for t in targets]
            if all(k is not None and k != key and classify(k) in ("eval", "pnl") for k in keys):
                continue
            return False
        return True

    def _label_uses(self) -> set[str]:
        """Names and columns passed as the label: fit(X, y) / Dataset(X, y) second argument, `label=` / `y=`."""
        if hasattr(self, "_label_uses_cache"):
            return self._label_uses_cache
        out: set[str] = set()
        for n in self._all:
            if isinstance(n, ast.Call) and (_attr(n) in FIT_ATTRS or _func_name(n) in TRAIN_DATA_CTORS):
                nodes = (n.args[1:2] if len(n.args) > 1 else []) + [kw.value for kw in n.keywords if kw.arg in ("y", "label")]
                for a in nodes:
                    out |= {x.id for x in ast.walk(a) if isinstance(x, ast.Name)}
                    out |= {s for x in ast.walk(a) if (s := _const_str(x))}
        self._label_uses_cache = out  # the AST never changes: computed once
        return out

    def _adds_columns_only(self, stmt: ast.AST, listed: set[str]) -> bool:
        """data = X.join(y.rename("y")).dropna(), later fitted only as data[["mom", "vol"]]: the forward data is a
        column of its own, and the explicit list doesn't name it (nor any forward or label-named column)."""
        v = getattr(stmt, "value", None)
        if v is None or not any(_attr(c) in ("join", "merge", "assign") for c in _chain(v)) and not (
                isinstance(v, ast.Call) and _func_name(v) == "concat"):
            return False
        if isinstance(v, ast.BinOp):
            return False
        return not any((self.info.get(c) is not None and self.info[c].forward) or classify(c) == "label" for c in listed)

    def _resolve_la10_pending(self) -> None:
        for key, role, scan, value, stmt in self._fill_pending:
            if not self._fill_mask_ok(key):
                self._rule_negative_shift(key, role, scan, value, stmt, force=True)
        self._fill_pending = []
        self._container_model_uses()
        pending = self._la10_pending or self._la01_pending
        uses = self._feature_uses() if pending else set()
        reach = self._signal_reach() if pending else {}
        labels = self._label_uses() if any(p[3] for p in self._la01_pending) else set()
        lit = getattr(self, "_lit_uses", {})
        for key, call, what, in_compare in self._la01_pending:
            if key in reach or key in uses or key in lit:
                into = f"the signal or position `{reach[key]}`" if key in reach else "a model input"
                self.add("LA01", "high", call, f"{what}, and it feeds {into}.")
            elif in_compare and key not in labels and not self._eval_only(key):
                self.add("LA01", "high", call, what + " inside a comparison used as a signal.")
            else:
                self.add("LA01", "possible", call,
                         what + ". If this is your label, rename it (e.g. `fwd_ret`) and keep it out of the features; "
                         "if it feeds a signal or feature, it is lookahead.")
        self._la01_pending = []
        for key, stmt, k, line in self._la10_pending:
            if key in reach:
                self.add("LA10", "high", stmt,
                         f"`{key}` is computed from `{k}`, which contains future data (defined on line {line}), and "
                         f"`{key}` feeds the signal or position `{reach[key]}`.")
            elif key in uses or (key in lit and not self._adds_columns_only(stmt, lit[key])):
                self.add("LA10", "medium", stmt,
                         f"`{key}` is computed from `{k}`, which contains future data (defined on line {line}), and "
                         f"`{key}` is used as a model input or feature.")
            else:
                self.add("LA10", "possible", stmt,
                         f"`{key}` is computed from `{k}`, which contains future data (defined on line {line}). "
                         "That is fine for a label or an evaluation; if `{key}` is used as a feature or signal, this "
                         "is target leakage.".replace("{key}", key))
        self._la10_pending = []

    # -- simple call patterns -----------------------------------------------------------------
    def rule_calls(self) -> None:
        for n in self._all:
            if not isinstance(n, ast.Call):
                continue
            a = _attr(n)
            fn = _func_name(n)
            tkey, _tf, _st = self.enclosing_assign_target(n)
            trole = classify(tkey)

            # LA01 outside assignments: negative shift inside an if/while condition
            if _shift_sign(n) == -1 and tkey is None:
                test_parent = self.inside(n, (ast.If, ast.While, ast.IfExp))
                if test_parent is not None and self._in_test(n, test_parent):
                    self.add("LA01", "high", n, "A condition reads a negatively shifted (future) value.")

            # LA02 centered windows
            if a == "rolling" and isinstance(_kw(n, "center"), ast.Constant) and _kw(n, "center").value is True:
                if trole != "label":
                    self.add("LA02", "high", n, "`rolling(..., center=True)` uses the next half-window of bars.")

            # LA03 backward fill / interpolation
            if trole != "label":
                if a in ("bfill", "backfill"):
                    self.add("LA03", "medium", n, f"`.{a}()` fills gaps with later values.")
                elif a in ("fillna", "reindex", "asfreq", "resample") and _const_str(_kw(n, "method")) in ("bfill", "backfill", "nearest"):
                    self.add("LA03", "medium", n, f"`{a}(method='{_const_str(_kw(n, 'method'))}')` fills with later values.")
                elif a == "interpolate":
                    m = _const_str(_kw(n, "method")) or (_const_str(n.args[0]) if n.args else None)
                    if m not in ("pad", "ffill", "zero", "previous"):
                        self.add("LA03", "possible", n,
                                 "`interpolate()` draws a line to the *next* known value, so filled points use future data.")

            # LA04 merge_asof direction
            if fn == "merge_asof":
                d = _const_str(_kw(n, "direction"))
                if d == "forward":
                    self.add("LA04", "high", n, "`merge_asof(direction='forward')` attaches the next (future) record.")
                elif d == "nearest":
                    self.add("LA04", "medium", n, "`merge_asof(direction='nearest')` can attach a future record.")

            # LA06 shuffled splits / non-temporal CV
            if fn == "train_test_split":
                sh = _kw(n, "shuffle")
                if not (isinstance(sh, ast.Constant) and sh.value is False):
                    sev = "high" if self.ts_context else "possible"
                    self.add("LA06", sev, n, "`train_test_split` shuffles rows by default, mixing future rows into training.")
            if fn in ("KFold", "StratifiedKFold", "RepeatedKFold") and not self.ts_context and self._fwd_label_hint():
                sh = _kw(n, "shuffle")
                if isinstance(sh, ast.Constant) and sh.value is True:
                    # v0.6: no time-series hint in the text, but a forward-return label (fwd_chg_5d) says the rows are
                    # dated observations
                    self.add("LA06", "medium", n, f"`{fn}(shuffle=True)` on rows labelled with a forward return: the "
                                                  "folds mix earlier and later dates, so training sees the future.")
            if fn in ("KFold", "StratifiedKFold", "RepeatedKFold") and self.ts_context:
                sh = _kw(n, "shuffle")
                if isinstance(sh, ast.Constant) and sh.value is True:
                    self.add("LA06", "medium", n, f"`{fn}(shuffle=True)` on time-ordered data.")
                else:
                    self.add("LA06", "possible", n, f"`{fn}` trains on folds that come after the test fold.")
            if fn in ("ShuffleSplit", "StratifiedShuffleSplit") and self.ts_context:
                self.add("LA06", "medium", n, f"`{fn}` shuffles time-ordered rows.")
            if fn in ("cross_val_score", "cross_validate", "cross_val_predict", "GridSearchCV", "RandomizedSearchCV") and self.ts_context:
                cv = _kw(n, "cv")
                if cv is None or _num(cv) is not None:
                    self.add("LA06", "possible", n,
                             f"`{fn}` with {'default' if cv is None else 'integer'} `cv` uses k-fold, which trains on the future. Pass `cv=TimeSeriesSplit(...)`.")

            # LA07 engine settings that fill on the signal bar
            for k in n.keywords:
                if k.arg == "cheat_on_close" and isinstance(k.value, ast.Constant) and k.value.value is True:
                    self.add("LA07", "high", n, "`cheat_on_close=True` fills orders at the close of the bar that generated them.")
                if k.arg == "trade_on_close" and isinstance(k.value, ast.Constant) and k.value.value is True:
                    self.add("LA07", "medium", n, "`trade_on_close=True` fills at the signal bar's close; make sure the signal doesn't use that close.")
                if k.arg == "coc" and isinstance(k.value, ast.Constant) and k.value.value is True:
                    self.add("LA07", "high", n, "`coc=True` (cheat-on-close) fills at the signal bar's close.")
            if a == "set_coc" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value is True:
                self.add("LA07", "high", n, "`set_coc(True)` (cheat-on-close) fills at the signal bar's close.")
            if a == "from_signals":
                entries = _kw(n, "entries") or (n.args[1] if len(n.args) > 1 else None)
                pkw = _kw(n, "price")
                close_arg = _kw(n, "close") or (n.args[0] if n.args else None)
                same_px = pkw is not None and _key(pkw)[0] is not None and _chain_shift(pkw) == 0 and (
                    (close_arg is not None and _key(pkw) == _key(close_arg)) or _is_close_name(_key(pkw)[0]))
                if entries is not None and (pkw is None or same_px) and _chain_shift(entries) <= 0:
                    k, _ = _key(_chain_base(entries))
                    ki = (self.info.get(k) if k else None) or self._inline_signal(_chain_base(entries))
                    if ki is not None and ki.price_derived and not ki.lagged:
                        if same_px:
                            self.add("LA07", "medium", n,
                                     "`from_signals(..., price=close)` explicitly fills at the close the entries were computed "
                                     "from. Shift entries/exits by one bar, or fill at the next bar's open.")
                        else:
                            self.add("LA07", "possible", n,
                                     "`from_signals` fills at `close` by default, and these entries are computed from the same close. "
                                     "Shift entries/exits by one bar or pass the next bar's price.")
                open_px = (pkw is not None and _chain_shift(pkw) == 0 and not self._ahead(pkw)
                           and _is_open_name(_key(_chain_base(pkw))[0]))
                if entries is not None and open_px and _chain_shift(entries) <= 0:
                    # v0.7 (PR #40 gap): entries from bar t's close filled at bar t's own open
                    k, _ = _key(_chain_base(entries))
                    ki = (self.info.get(k) if k else None) or self._inline_signal(_chain_base(entries))
                    if ki is not None and ki.price_derived and ki.post_open and not ki.lagged and not ki.forward:
                        self.add("LA07", "high", n,
                                 f"`from_signals(..., price={_key(_chain_base(pkw))[0]})` fills each entry at the open of the "
                                 "bar whose close produced the signal, before that close was known. Shift entries/exits "
                                 "by one bar (`entries.shift(1, fill_value=False)`) so they fill at the next bar's open.")

            # LA12 intraday resample without label/closed
            if a == "resample":
                self._rule_resample(n)

            # SB01 current-constituent lists
            if fn == "read_html":
                url = _const_str(n.args[0]) if n.args else _const_str(_kw(n, "io"))
                if url is None and n.args and isinstance(n.args[0], ast.Name):
                    url = self.str_consts.get(n.args[0].id)
                if url and any(h in url for h in CURRENT_LIST_URL_HINTS):
                    self.add("SB01", "high", n,
                             "The universe is scraped from a page that lists *today's* members, so the backtest "
                             "only contains survivors.")
            if fn in TODAY_LIST_FUNCS:
                self.add("SB01", "high", n, f"`{fn}()` returns today's index members, not the members on each past date.")
            if fn == "read_csv" and n.args:
                path = _const_str(n.args[0]) or ""
                low = path.lower()
                if re.search(r"(sp500|s&p|snp|constituents|tickers|universe|nasdaq100|russell)", low) and not re.search(
                        r"(hist|pit|point|membership|changes|asof|as_of|daily)", low):
                    self.add("SB01", "possible", n,
                             f"The universe is read from `{path}`. Make sure it records membership by date "
                             "(not a snapshot of today's list) and includes delisted names.")

            # DQ: yfinance usage
            if fn == "download" and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id in ("yf", "yfinance"):
                aa = _kw(n, "auto_adjust")
                if isinstance(aa, ast.Constant) and aa.value is False and "pct_change" in self.src and re.search(r"[\"']Close[\"']", self.src):
                    self.add("DQ02", "medium", n,
                             "`auto_adjust=False` returns raw `Close`; returns computed from it show fake drops on "
                             "split days and miss dividends. Use `Adj Close` or `auto_adjust=True`.")

    def _fwd_label_hint(self) -> bool:
        """A string or name in the file names a forward-looking quantity (fwd_ret_5d, forward_return, future_ret)."""
        if not hasattr(self, "_fwdh"):
            words = {x.id for x in self._all if isinstance(x, ast.Name)} | {
                _const_str(x) for x in self._all if _const_str(x) and len(_const_str(x)) < 40}
            self._fwdh = any(_tokens(w) & {"fwd", "forward", "future"} for w in words if w)
        return self._fwdh

    # -- LA12: intraday bars realigned onto other timestamps -------------------------------------
    _REALIGN = {"reindex", "join", "merge", "asof", "combine_first", "update", "align", "reindex_like"}

    @staticmethod
    def _dict_consts(d: ast.AST | None) -> dict[str, str | None] | None:
        """{'label': 'left'} or dict(label='left') -> its string items; None if it isn't a visible literal."""
        if isinstance(d, ast.Dict) and all(_const_str(k) is not None for k in d.keys):
            return {_const_str(k): _const_str(v) for k, v in zip(d.keys, d.values)}
        if isinstance(d, ast.Call) and _func_name(d) == "dict" and not d.args and all(k.arg for k in d.keywords):
            return {k.arg: _const_str(k.value) for k in d.keywords}
        return None

    def _realigned(self, n: ast.AST, rule: str) -> str | None:
        """How the resampled bars are put back on another time index, if they are."""
        p = self.parents.get(n)
        while isinstance(p, (ast.Call, ast.Attribute, ast.Subscript, ast.keyword, ast.List, ast.Tuple)):
            if isinstance(p, ast.Call) and _attr(p) in ("shift", "tshift") and self._positive_shift(p):
                return None  # bars moved to their end stamp (or later) before any join
            if isinstance(p, ast.Call) and (_attr(p) in self._REALIGN or _func_name(p) in ("merge_asof", "concat", "merge")):
                return None if self._joins_only_bars(p, set(), rule) else f"`{_attr(p) or _func_name(p)}(...)`"
            p = self.parents.get(p)
        stmt = p if isinstance(p, ast.stmt) else None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Subscript):
            fk = _key(stmt.targets[0])[1]
            if not (fk and fk in self._bar_names(rule)):
                return "assigning them into a column of the finer frame"
            first = fk  # v0.4: bars["volume"] = ticks["size"].resample("1min").sum(): a column of the same bars
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            first = stmt.targets[0].id
        else:
            return None
        watch, line = {first}, stmt.lineno
        for st in self._iter_refs(watch, line):
            tg = st.targets[0] if isinstance(st, ast.Assign) else getattr(st, "target", None)
            if (isinstance(tg, ast.Attribute) and tg.attr == "index" and isinstance(tg.value, ast.Name)
                    and tg.value.id in watch and (self._has_lag_expr(st.value) if isinstance(st, ast.Assign)
                                                  else self._is_lag_term(st.value))):
                return None  # h.index = h.index + pd.Timedelta(hours=1): re-stamped at the bar end
            if not self.names_in(st.value) & watch:
                continue
            if (isinstance(st, ast.Assign) and isinstance(tg, ast.Name) and tg.id in watch
                    and any(isinstance(c, ast.Call) and _attr(c) in ("shift", "tshift") and self._positive_shift(c)
                            for c in ast.walk(st.value))):
                return None  # h = h.shift(1): each bar now sits on the next stamp
            for c in ast.walk(st.value):
                if isinstance(c, ast.Call) and (_attr(c) in self._REALIGN or _func_name(c) in ("merge_asof", "concat", "merge")):
                    if self.names_in(c) & watch and not self._joins_only_bars(c, watch, rule):
                        return f"`{_attr(c) or _func_name(c)}(...)` on line {st.lineno}"
            if isinstance(st, ast.Assign) and len(st.targets) == 1:
                t = st.targets[0]
                if isinstance(t, ast.Name):
                    watch.add(t.id)
                elif isinstance(t, ast.Subscript):
                    fk = _key(t)[1]
                    if fk and fk not in watch:
                        return f"assigning them into `{fk}` on line {st.lineno}"
        return None

    @staticmethod
    def _positive_shift(c: ast.Call) -> bool:
        amt = _kw(c, "periods") or (c.args[0] if c.args else None)
        n = 1 if amt is None else _numc(amt)
        return n is not None and n > 0

    def _bar_names(self, rule: str | None) -> set[str]:
        """Names assigned (as a whole) from a resample with this rule: frames of bars with the same stamps."""
        if not hasattr(self, "_bars"):
            self._bars: dict[str | None, set[str]] = {}
            whole: dict[str, list[ast.Assign]] = {}
            for st in self._all:
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    whole.setdefault(st.targets[0].id, []).append(st)
                    for c in ast.walk(st.value):
                        if isinstance(c, ast.Call) and _attr(c) == "resample":
                            r = _const_str(_kw(c, "rule")) or (_const_str(c.args[0]) if c.args else None)
                            self._bars.setdefault(r, set()).add(st.targets[0].id)
            # v0.6 (v8 clean_16): a name computed only from bars of one rule (rolling stats, returns, a DataFrame of
            # them) keeps those bars' stamps; every assignment of it must qualify, and nothing else may be read.
            funcs = set(self._funcs) | MODULE_ALIASES | {"pd", "np", "len", "abs", "min", "max", "float", "int"}
            for r, bars in self._bars.items():
                grew = True
                while grew:
                    grew = False
                    for nm, sts in whole.items():
                        if nm in bars:
                            continue
                        if all((self.names_in(s.value) - funcs) and (self.names_in(s.value) - funcs) <= bars
                               and not any(isinstance(c, ast.Call) and (_attr(c) in self._REALIGN or _attr(c) == "resample"
                                                                        or _func_name(c) in ("merge_asof", "concat", "merge"))
                                           for c in ast.walk(s.value))
                               for s in sts):
                            bars.add(nm)
                            grew = True
        return self._bars.get(rule, set())

    def _joins_only_bars(self, call: ast.Call, watch: set[str], rule: str) -> bool:
        """Every other frame in the join was resampled with the same rule (same stamps): the join is consistent."""
        bars = self._bar_names(rule)
        others = set(self.names_in(call)) - watch - MODULE_ALIASES
        others = {o for o in others if o not in ("left", "right", "inner", "outer")}
        frames = {o for o in others if o in bars or o in self.assigned_names}
        return bool(frames) and frames <= bars

    def _rule_resample(self, n: ast.Call) -> None:
        rule = _const_str(_kw(n, "rule")) or (_const_str(n.args[0]) if n.args else None)
        if not (rule and INTRADAY_RULE.match(rule)):
            return
        label, closed = _const_str(_kw(n, "label")), _const_str(_kw(n, "closed"))
        for k in n.keywords:
            if k.arg is None:  # **kwargs: read a visible dict, otherwise give up (it may set label='right')
                d = k.value
                if isinstance(d, ast.Name):
                    first = next(iter(self._assigns_of(d.id)), None)
                    d = first.value if first is not None else None
                opts = self._dict_consts(d)
                if opts is None:
                    return
                label = opts.get("label", label)
                closed = opts.get("closed", closed)
        if label == "right":
            return
        agg = self.parents.get(n)
        agg_call = self.parents.get(agg) if isinstance(agg, ast.Attribute) else None
        if isinstance(agg_call, ast.Call) and _attr(agg_call) in ("first", "count", "size", "nunique"):
            return  # the first print of a bar is known at its start time
        how = self._realigned(n, rule)
        stamp = "`label='left'`" if label == "left" else "the default `label='left'`"
        if how is not None:
            self.add("LA12", "medium", n,
                     f"`resample('{rule}')` stamps each bar with its *start* time ({stamp}), but the bar holds data until "
                     f"its end; {how} then lines those values up with earlier timestamps, up to one bar of future data. "
                     "Pass `label='right'`.")
        elif label is None and closed is None:
            self.add("LA12", "possible", n,
                     f"`resample('{rule}')` stamps each bar with its *start* time by default, while the bar holds data until its end.")
        else:
            self.add("LA12", "possible", n,
                     f"`resample('{rule}')` with {stamp} stamps each bar with its start time while it holds data until its end.")

    def _in_test(self, node: ast.AST, stmt: ast.AST) -> bool:
        test = getattr(stmt, "test", None)
        if test is None:
            return False
        return any(n is node for n in ast.walk(test))

    # -- LA07: signal x same-bar return ---------------------------------------------------------
    _HARMLESS = {"to_numpy", "values", "astype", "fillna", "ffill", "copy", "squeeze", "to_list", "tolist"}

    def _open_key(self, node: ast.AST) -> bool:
        """`node` is bar t's own (unshifted) open price."""
        if any(_attr(c) not in self._HARMLESS for c in _chain(node)):
            return False
        k, _f = _key(_chain_base(node))
        if not (_is_open_name(k) or (k and _tokens(k) & {"vwap"} and not _tokens(k) & _TIME_WORDS)):
            return False  # v0.6: a fill at bar t's VWAP is, like its open, mostly before bar t's close
        ki = self.info.get(k)
        return not (ki is not None and (ki.lagged or ki.forward))

    def _later_price(self, node: ast.AST) -> bool:
        """A price at or after bar t's close: close/high/low of bar t, or any forward-shifted price."""
        if _chain_shift(node) < 0:
            return True
        if _chain_shift(node) > 0 or any(_attr(c) not in self._HARMLESS for c in _chain(node)):
            return False
        k, _f = _key(_chain_base(node))
        if not k:
            return False
        ki = self.info.get(k)
        if ki is not None and ki.lagged:
            return False
        return (_is_post_open_name(k) and not _tokens(k) & VOLUME_T) or (
            ki is not None and ki.forward and classify(k) in ("exec", "unknown"))

    def _open_start(self, expr: ast.AST) -> bool:
        """A return that starts at bar t's open: close/open - 1, open.shift(-1)/open - 1, close - open, log(c/o)."""
        e = expr
        if isinstance(e, ast.Call) and _func_name(e) in ("log", "log1p") and e.args:
            e = e.args[0]
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Sub) and _num(e.right) is not None:
            e = e.left
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Div):
            a, b = e.left, e.right
            if self._open_key(b) and self._later_price(a):
                return True
            if isinstance(a, ast.BinOp) and isinstance(a.op, ast.Sub) and self._open_key(a.right) and self._later_price(a.left):
                return True
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Sub):
            return self._open_key(e.right) and self._later_price(e.left)
        return False

    def _inline_signal(self, base: ast.AST) -> KeyInfo | None:
        """KeyInfo for an inline condition used as a position: (close > sma).astype(int), np.where(...), np.sign(...)."""
        if not (isinstance(base, ast.Compare) or _np_call(base, "where", "sign") or (
                isinstance(base, ast.BoolOp) and all(isinstance(v, (ast.Compare, ast.BoolOp)) for v in base.values))):
            return None
        sc = _scan_expr(base)
        if not sc.refs or sc.neg_shift_calls:
            return None
        infos = [(self.info.get(k), lag, k) for k, lag in sc.refs]
        if any(ki is not None and ki.forward and lag <= 0 for ki, lag, _k in infos):
            return None
        lagged = all(lag > 0 or (ki is not None and ki.lagged) for ki, lag, _k in infos)
        price = any(_is_price_name(k) or (ki is not None and ki.price_derived) for ki, _l, k in infos)
        post = not lagged and any(lag <= 0 and (_is_post_open_name(k) or (ki is not None and ki.post_open and not ki.lagged))
                                  for ki, lag, k in infos)
        return KeyInfo("inline condition", None, getattr(base, "lineno", 0), "signal", lagged=lagged,
                       price_derived=price, is_signal_expr=True, post_open=post)

    def _side_role(self, node: ast.AST) -> tuple[str | None, KeyInfo | None, str | None]:
        if any(_attr(c) in ("mean", "sum", "std", "var", "median", "max", "min", "prod", "count", "item", "corr", "cov")
               for c in _chain(node)):
            return None, None, None  # a scalar summary, not a per-bar series
        s = _chain_shift(node)
        if s > 0:
            return "lagged", None, None
        if s < 0 and not self._open_start(node):
            return "forward", None, None
        if self._open_start(node):
            return "open_return", None, "inline open-to-later return"
        if any(_attr(c) in ("pct_change", "diff") for c in _chain(node)):
            k, _ = _key(_chain_base(node))
            return "return", self.info.get(k) if k else None, (k or "?") + f".{_attr(next(_chain(node)))}()"
        if isinstance(node, ast.BinOp) and _scan_expr(node).has_return_op:
            sc = _scan_expr(node)
            if sc.neg_shift_calls or any(self.info.get(k) is not None and self.info[k].forward for k, _l in sc.refs):
                return "forward", None, None
            return "return", None, "inline return"
        base = _chain_base(node)
        inline = self._inline_signal(base)
        if inline is not None:
            return ("lagged" if inline.lagged else "signal"), inline, "inline condition"
        k, _ = _key(base)
        if k is None:
            return None, None, None
        ki = self.info.get(k)
        role = classify(k)
        if ki is not None and ki.lagged:
            return "lagged", ki, k
        if ki is not None and ki.entry_open:
            return "open_return", ki, k
        if ki is not None and ki.forward:
            return "forward", ki, k
        if role == "signal" or (ki is not None and ki.is_signal_expr and role not in ("return", "pnl", "label")):
            return "signal", ki, k
        if role == "return" or (ki is not None and ki.is_return):
            return "return", ki, k
        return None, ki, k

    def _cond_same_bar_price(self, cond: ast.AST, post_open: bool = False) -> bool:
        """True if a condition uses unlagged price-derived data and nothing lagged or forward.
        With post_open=True it must use data known only after the bar's open (close, high, low)."""
        if _chain_shift(cond) != 0:
            return False
        sc = _scan_expr(cond)
        if not sc.refs or sc.neg_shift_calls or any(lag > 0 for _k, lag in sc.refs):
            return False
        infos = [self.info.get(k) for k, _l in sc.refs]
        if any(ki is not None and (ki.lagged or ki.forward) for ki in infos):
            return False
        if post_open:
            return any(_is_post_open_name(k) or (ki is not None and ki.post_open) for (k, _l), ki in zip(sc.refs, infos))
        return any(_is_price_name(k) or (ki is not None and ki.price_derived) for (k, _l), ki in zip(sc.refs, infos))

    def rule_same_bar(self) -> None:
        for n in self._all:
            if (_np_call(n, "where") and len(n.args) == 3 and self.inside(n, (ast.FunctionDef,)) is None
                    and self._array_where(n) is None):
                roles = [self._side_role(a)[0] for a in n.args[1:]]
                if "return" in roles and self._cond_same_bar_price(n.args[0]):
                    self.add("LA07", "high", n,
                             "`np.where(<condition on this bar's prices>, <this bar's return>, ...)` books the return of "
                             "the bar whose close created the condition. Lag the condition with `.shift(1)`.")
                elif "open_return" in roles and self._cond_same_bar_price(n.args[0], post_open=True):
                    self.add("LA07", "high", n,
                             "`np.where(<condition on this bar's close>, <return from this bar's open>, ...)` enters at "
                             "the open of the bar whose close created the condition. Enter at the next bar's open.")
            pairs = []
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
                pairs.append((n.left, n.right))
            elif isinstance(n, ast.Call) and _attr(n) in ("mul", "multiply") and n.args:
                pairs.append((n.func.value, n.args[0]))
            for left, right in pairs:
                if id(n) in self._array_checked:
                    continue  # positional (numpy/list) alignment was already judged by rule_array_alignment
                self._check_pair(n, left, right)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.helper_mults:
                for ia, ib, pa, pb in self.helper_mults[n.func.id]:
                    a, b = self._call_arg(n, ia, pa), self._call_arg(n, ib, pb)
                    if a is not None and b is not None:
                        self._check_pair(n, a, b, via=n.func.id)

    # -- LA07 (v0.4): an open price selected by a signal from the same bar's close ------------------------
    def _close_cond(self, cond: ast.AST) -> str | None:
        """The name of what makes `cond` known only at bar t's close (close, high, low, volume, or something built
        from them, unlagged); None if it is lagged, forward or known at the open."""
        while isinstance(cond, ast.UnaryOp) and isinstance(cond.op, (ast.Invert, ast.Not)):
            cond = cond.operand
        while isinstance(cond, ast.Call) and _attr(cond) in ("astype", "fillna", "to_numpy", "copy") and isinstance(cond.func, ast.Attribute):
            cond = cond.func.value
        if isinstance(cond, ast.Attribute) and cond.attr == "values":
            cond = cond.value
        if _chain_shift(cond) != 0 or self._ahead(cond):
            return None
        sc = _scan_expr(cond)
        hit = None
        for k, lag in sc.refs:
            ki = self.info.get(k)
            if lag > 0:
                continue
            if ki is not None and ki.lagged:
                continue
            if (_is_post_open_name(k) and not (ki is not None and ki.lagged)) or (ki is not None and ki.post_open):
                hit = hit or k
        return hit

    def _open_fill_target(self, stmt: ast.AST) -> str | None:
        tg = stmt.targets[0] if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 else None
        k = _key(tg)[0] if tg is not None else None
        return k

    def rule_open_fill(self) -> None:
        """entry_px = open.where(sig) / open[sig] / np.where(sig, open, nan) / df.loc[sig, "entry_px"] = df["open"],
        with `sig` from bar t's close: the fill is at bar t's open, before the signal existed."""
        if not re.search(r"open|opn|\bo\b", self.src, re.I):
            return  # perf guard: no open price anywhere in the file
        for st in self._stmts_sorted:
            if not isinstance(st, ast.Assign) or len(st.targets) != 1:
                continue
            if self.inside(st, (ast.For, ast.While)) is not None:
                continue  # bar loops: rule_loop_same_bar
            found = None  # (node, cond, open node)
            tgt = st.targets[0]
            if (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Attribute) and tgt.value.attr == "loc"
                    and isinstance(tgt.slice, ast.Tuple) and len(tgt.slice.elts) == 2 and self._open_key(st.value)):
                found = (st, tgt.slice.elts[0], st.value)
            for n in ast.walk(st.value):
                if found:
                    break
                if isinstance(n, ast.Call) and _attr(n) in ("where", "mask") and isinstance(n.func, ast.Attribute) \
                        and not _is_np(n.func.value) and self._open_key(n.func.value):
                    cond = n.args[0] if n.args else _kw(n, "cond")
                    if cond is not None:
                        found = (n, cond, n.func.value)
                elif _np_call(n, "where") and len(n.args) == 3:
                    for a in n.args[1:]:
                        if self._open_key(a):
                            found = (n, n.args[0], a)
                elif isinstance(n, ast.Subscript) and not isinstance(n.slice, (ast.Slice, ast.Constant)):
                    base, sl = n.value, n.slice
                    if isinstance(base, ast.Attribute) and base.attr == "loc":
                        if isinstance(sl, ast.Tuple) and len(sl.elts) == 2 and _is_open_name(_const_str(sl.elts[1])):
                            k = _const_str(sl.elts[1])
                            ki = self.info.get(k)
                            if not (ki is not None and (ki.lagged or ki.forward)):
                                found = (n, sl.elts[0], n)
                            continue
                        base = base.value
                    if isinstance(sl, ast.Tuple) or _const_str(sl) is not None:
                        continue
                    if self._open_key(base) and (isinstance(sl, (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.Name,
                                                                 ast.Attribute, ast.Call, ast.BinOp))):
                        found = (n, sl, base)
            if not found:
                continue
            node, cond, _o = found
            why = self._close_cond(cond)
            if why is None:
                continue
            tk = self._open_fill_target(st)
            fill_name = bool(tk) and (classify(tk) == "exec" or bool(_tokens(tk) & (ENTRY_T | EXEC_T | {
                "exit", "exits", "sell", "sold", "cover", "cost"})))
            if fill_name:
                self.add("LA07", "high", node,
                         f"`{tk}` is priced at bar t's open, but the condition that selects it uses `{why}`, known only "
                         "at bar t's close: the trade is filled before its signal exists. Fill at the next bar's open "
                         "(`open.shift(-1)`), or lag the signal by one bar.")
            else:
                self.add("LA07", "possible", node,
                         f"Bar t's open is selected by a condition on `{why}`, known only at bar t's close. If this is "
                         "a fill price, the trade is entered before its signal exists; use the next bar's open.")

    def _slice_written(self) -> set[str]:
        """Names written through a positional slice or index (pos[1:] = ..., pos[i + 1] = ...)."""
        if not hasattr(self, "_sw"):
            self._sw = {t.value.id for st in self._all if isinstance(st, (ast.Assign, ast.AugAssign))
                        for t in (st.targets if isinstance(st, ast.Assign) else [st.target])
                        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and _const_str(t.slice) is None
                        and not (isinstance(t.slice, ast.Tuple) and any(_const_str(e) for e in t.slice.elts))}
        return self._sw

    def _carried(self, loop: ast.AST, sig: ast.AST, at: ast.AST) -> bool:
        """`pos` in `pnl += pos * r` is (re)assigned later in the loop body: it holds the previous bar's position."""
        b = _chain_base(sig)
        if not isinstance(b, ast.Name):
            return False
        for st in ast.walk(loop):
            if isinstance(st, (ast.Assign, ast.AugAssign)) and st.lineno > at.lineno:
                tg = st.targets if isinstance(st, ast.Assign) else [st.target]
                for t in tg:
                    if any(isinstance(e, ast.Name) and e.id == b.id for e in ast.walk(t)):
                        return True
        return False

    @staticmethod
    def _call_arg(call: ast.Call, idx: int, name: str) -> ast.AST | None:
        if idx < len(call.args) and not any(isinstance(a, ast.Starred) for a in call.args[: idx + 1]):
            return call.args[idx]
        return _kw(call, name)

    def _check_pair(self, n: ast.AST, left: ast.AST, right: ast.AST, via: str | None = None) -> bool:
        rl, kl, nl = self._side_role(left)
        rr, kr, nr = self._side_role(right)
        sig_info, sig_name = (kl, nl) if rl == "signal" else (kr, nr)
        other_name = nr if rl == "signal" else nl
        through = f" (inside `{via}()`)" if via else ""
        if {rl, rr} == {"signal", "open_return"}:
            if sig_info is not None and sig_info.post_open and not sig_info.lagged:
                self.add("LA07", "high", n,
                         f"`{sig_name}` uses bar t's close (or high/low), but `{other_name}` is a return that starts at "
                         f"bar t's open{through}: the trade is entered before the signal exists. Enter at the next "
                         "bar's open (`signal.shift(1)`) instead.")
                return True
            return False
        if {rl, rr} != {"signal", "return"}:
            return False
        if self.inside(n, (ast.FunctionDef,)) is not None and sig_info is None:
            return False
        loop = self.inside(n, (ast.For, ast.While))
        if loop is not None and self._carried(loop, left if rl == "signal" else right, n):
            return False
        if {x.id for x in ast.walk(n) if isinstance(x, ast.Name)} & (self._arr_ambiguous | self._slice_written()):
            self.add("LA07", "possible", n,
                     f"`{sig_name}` is multiplied by `{other_name}`, and one of them is written through slices or "
                     "element by element. Check that the position at bar t only uses data up to bar t-1.")
            return True
        if sig_info is not None and sig_info.price_derived:
            self.add("LA07", "high", n,
                     f"`{sig_name}` is computed from the same bar's prices and multiplied by the same bar's "
                     f"return `{other_name}` without a lag{through}: the strategy earns the move that created its signal.")
        else:
            self.add("LA07", "medium", n,
                     f"`{sig_name}` is multiplied by the same-bar return `{other_name}` without `.shift(1)`{through}. "
                     "Unless the signal was formed before the bar opened, this books returns it couldn't capture.")
        return True

    # -- LA07: helpers that multiply two of their parameters ---------------------------------------
    def _collect_helper_mults(self) -> None:
        """def strat(sig, ret): return sig * ret  ->  helper_mults['strat'] = [(0, 1, 'sig', 'ret')]."""
        self.helper_mults: dict[str, list[tuple[int, int, str, str]]] = {}
        for fd in self._all:
            if not isinstance(fd, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = [a.arg for a in fd.args.posonlyargs + fd.args.args]
            if params and params[0] in ("self", "cls"):
                params = params[1:]
            reassigned = {t.id for st in ast.walk(fd) if isinstance(st, (ast.Assign, ast.AugAssign))
                          for t in (st.targets if isinstance(st, ast.Assign) else [st.target]) if isinstance(t, ast.Name)}
            found = []
            for m in ast.walk(fd):
                if not (isinstance(m, ast.BinOp) and isinstance(m.op, ast.Mult)):
                    continue
                sides = []
                for side in (m.left, m.right):
                    if _chain_shift(side) != 0:
                        break
                    b = _chain_base(side)
                    if isinstance(b, ast.Name) and b.id in params and b.id not in reassigned:
                        sides.append(b.id)
                if len(sides) == 2 and sides[0] != sides[1]:
                    found.append((params.index(sides[0]), params.index(sides[1]), sides[0], sides[1]))
            if found:
                self.helper_mults[fd.name] = found

    # -- LA07: positional alignment of numpy arrays and lists ----------------------------------------
    _REDUCE = {"mean", "sum", "std", "var", "max", "min", "median", "prod", "nanmean", "nansum", "nanstd", "nanmax",
               "nanmin", "nanmedian", "count", "size", "len", "item", "any", "all", "argmax", "argmin", "quantile",
               "percentile", "nanpercentile", "corrcoef", "dot", "cumsum", "cumprod"}

    def _aoff(self, node: ast.AST, env: dict) -> tuple[int, bool, str | None] | None:
        """(information offset, uses positional alignment, kind) of a numpy/list expression, or None.

        Element j of `c[1:]` is bar j+1 (offset 1); of `c[:-1]` bar j (offset 0); `np.diff(c)` element j uses bar
        j+1; `np.roll(c, -1)` element j is bar j+1. kind is 'ret' (a return), 'sig' (a condition) or 'diff'."""
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            return self.arrays.get(node.id)
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr in ("r_", "c_")
                and _is_np(node.value.value)):
            if node.value.attr == "c_":
                return None
            parts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            return self._prefixed(parts, env)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            base = self._aoff(node.value, env)
            if base is None and self._array_source(node.value):
                lag = self._pandas_lag(node.value)
                base = None if lag is None else (-lag, lag != 0, self._name_kind(node.value))
            if base is None or node.slice.step is not None:
                return None
            lo = 0 if node.slice.lower is None else _numc(node.slice.lower)
            if lo is None or lo < 0:
                return None
            return base[0] + int(lo), True, base[2]
        if self._array_source(node):
            lag = self._pandas_lag(node)
            return None if lag is None else (-lag, lag != 0, self._name_kind(node))
        if isinstance(node, ast.Call):
            fn = _func_name(node)
            if fn in self._REDUCE:
                return None
            if _np_call(node, "pad") and node.args:
                b = self._aoff(node.args[0], env)
                w = node.args[1] if len(node.args) > 1 else _kw(node, "pad_width")
                before = _numc(w.elts[0]) if isinstance(w, (ast.Tuple, ast.List)) and w.elts else _numc(w)
                if b is None or before is None:
                    return None
                return b[0] - int(before), True, b[2]
            if _np_call(node, "concatenate", "hstack", "append", "r_") and node.args:
                parts = (list(node.args[0].elts) if _np_call(node, "concatenate", "hstack")
                         and isinstance(node.args[0], (ast.List, ast.Tuple)) else list(node.args[:2]))
                return self._prefixed(parts, env)
            if _np_call(node, "insert") and len(node.args) >= 3:
                b = self._aoff(node.args[0], env)
                if b is None or _numc(node.args[1]) != 0 or not self._scalarish(node.args[2]):
                    return None
                return b[0] - 1, True, b[2]
            if _np_call(node, "diff") and node.args:
                b = self._aoff(node.args[0], env)
                if b is None:
                    return None
                pre = _kw(node, "prepend") is not None
                is_log = _np_call(node.args[0], "log", "log1p", "log2", "log10")
                return b[0] + (0 if pre else 1), True, ("ret" if is_log else "diff")
            if _np_call(node, "roll") and node.args:
                b = self._aoff(node.args[0], env)
                amt_node = _kw(node, "shift") or (node.args[1] if len(node.args) > 1 else None)
                amt = _num(amt_node) if amt_node is not None else None
                if b is None or amt is None:
                    return None
                return b[0] - int(amt), True, b[2]
            if _np_call(node, "zeros", "zeros_like", "ones", "ones_like", "empty", "empty_like", "full", "full_like"):
                return 0, False, None  # a fresh container: its content comes from later writes
            if _np_call(node):
                parts = [self._aoff(a, env) for a in node.args]
                parts = [x for x in parts if x is not None]
                if not parts:
                    return (0, False, None) if fn in ("loadtxt", "genfromtxt", "load", "fromfile", "array", "asarray") else None
                kind = "sig" if fn in ("sign", "where") else parts[0][2]
                return max(x[0] for x in parts), any(x[1] for x in parts), kind
            if isinstance(node.func, ast.Attribute) and fn in ("astype", "copy", "clip", "flatten", "ravel", "round"):
                return self._aoff(node.func.value, env)
            return None
        if isinstance(node, (ast.BinOp, ast.Compare, ast.BoolOp, ast.IfExp, ast.UnaryOp)):
            if isinstance(node, ast.BinOp):
                kids = [node.left, node.right]
            elif isinstance(node, ast.Compare):
                kids = [node.left, *node.comparators]
            elif isinstance(node, ast.BoolOp):
                kids = list(node.values)
            elif isinstance(node, ast.IfExp):
                kids = [node.test, node.body, node.orelse]
            else:
                kids = [node.operand]
            parts = [(k, self._aoff(k, env)) for k in kids]
            known = [x for _k, x in parts if x is not None]
            if not known:
                return None
            off, tracked = max(x[0] for x in known), any(x[1] for x in known)
            kind = None
            if isinstance(node, (ast.Compare, ast.BoolOp)):
                kind = "sig"
            elif isinstance(node, ast.IfExp):
                consts = all(_num(x) is not None for x in (node.body, node.orelse))
                if consts and isinstance(node.test, (ast.Compare, ast.BoolOp)):
                    kind = "sig"
                else:
                    kind = next((x[2] for x in (parts[1][1], parts[2][1]) if x is not None), None)
            elif isinstance(node, ast.UnaryOp):
                kind = known[0][2]
            elif isinstance(node.op, ast.Div):
                lft, rgt = parts[0][1], parts[1][1]
                if lft is not None and (lft[2] == "diff" or (rgt is not None and lft[0] != rgt[0])):
                    kind = "ret"
                elif lft is not None:
                    kind = lft[2]
            elif isinstance(node.op, ast.Sub):
                lft, rgt = parts[0][1], parts[1][1]
                if lft is not None and rgt is not None and lft[0] != rgt[0] and lft[2] is None and rgt[2] is None:
                    kind = "diff"
                elif (lft is not None and lft[2] == "ret") or (rgt is not None and rgt[2] == "ret"):
                    kind = "ret"
                else:
                    kind = (lft or rgt)[2]
            elif isinstance(node.op, ast.Mult):
                kinds = {x[2] for x in known} - {None}
                kind = "pnl" if {"sig", "ret"} <= kinds else (next(iter(kinds)) if len(kinds) == 1 else None)
            else:
                kind = known[0][2]
            return off, tracked, kind
        return None

    @staticmethod
    def _scalarish(node: ast.AST) -> bool:
        if _num(node) is not None or isinstance(node, ast.Constant):
            return True
        return isinstance(node, ast.Attribute) and node.attr in ("nan", "inf", "NaN", "NINF")

    def _prefixed(self, parts: list, env: dict):
        """[0, x[:-1]] style concatenation: n leading scalars shift x's elements n bars later (a lag)."""
        n, arr = 0, None
        for part in parts:
            if isinstance(part, ast.Starred):
                return None
            scal = self._scalarish(part)
            lst = isinstance(part, (ast.List, ast.Tuple)) and all(self._scalarish(e) for e in part.elts)
            if arr is None:
                if scal:
                    n += 1
                    continue
                if lst:
                    n += len(part.elts)
                    continue
                arr = self._aoff(part, env)
                if arr is None:
                    return None
            elif not (scal or lst):
                return None  # two arrays glued together: no single alignment
        if arr is None:
            return None
        return arr[0] - n, True, arr[2]

    def _pandas_lag(self, node: ast.AST) -> int | None:
        """Bars of `.shift(k)` applied before `.to_numpy()` / `.values` (0 if none, None if not a plain number)."""
        inner = node.value if isinstance(node, ast.Attribute) else (node.func.value if isinstance(node, ast.Call)
                                                                    and isinstance(node.func, ast.Attribute) else None)
        if inner is None:
            return 0
        total = 0
        for c in _chain(inner):
            if _attr(c) == "shift":
                amt = _kw(c, "periods") or (c.args[0] if c.args else None)
                v = 1 if amt is None else _numc(amt)
                if v is None or _kw(c, "freq") is not None:
                    return None
                total += int(v)
        return total

    def _array_source(self, node: ast.AST) -> bool:
        """x.to_numpy(), x.values, x.tolist(), np.array(x), list(x): a positional (not index-aligned) container."""
        if isinstance(node, ast.Attribute) and node.attr == "values":
            return True
        if isinstance(node, ast.Call):
            if _attr(node) in ("to_numpy", "tolist", "to_list"):
                return True
            if _np_call(node, "array", "asarray", "asfarray"):
                return True
            if isinstance(node.func, ast.Name) and node.func.id == "list" and node.args:
                return True
        return False

    def _name_kind(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute) and node.attr == "values":
            node = node.value
        k, _f = _key(_chain_base(node))
        role = classify(k)
        ki = self.info.get(k) if k else None
        if role == "return" or (ki is not None and ki.is_return and role not in ("signal", "pnl")):
            return "ret"
        if role == "signal":
            return "sig"
        return None

    def _bind_zip(self, target: ast.AST, it: ast.AST, env: dict) -> dict | None:
        if not (isinstance(it, ast.Call) and _func_name(it) == "zip" and isinstance(target, ast.Tuple)
                and len(target.elts) == len(it.args)):
            return None
        out = dict(env)
        for t, a in zip(target.elts, it.args):
            if isinstance(t, ast.Name):
                v = self._aoff(a, env)
                if v is not None:
                    out[t.id] = v
        return out

    def _judge_array_mult(self, m: ast.BinOp, env: dict) -> None:
        lft, rgt = self._aoff(m.left, env), self._aoff(m.right, env)
        if lft is None or rgt is None or not (lft[1] or rgt[1]) or {lft[2], rgt[2]} != {"sig", "ret"}:
            return
        self._array_checked.add(id(m))
        sig, ret = (lft, rgt) if lft[2] == "sig" else (rgt, lft)
        names = {x.id for x in ast.walk(m) if isinstance(x, ast.Name)}
        if names & self._arr_ambiguous:
            if sig[0] >= ret[0]:
                self.add("LA07", "possible", m,
                         "A position array written element by element (or through slices this lint can't line up) is "
                         "multiplied by a return array. Check that the position at bar t only uses data up to bar t-1.")
            return
        if sig[0] >= ret[0]:
            self.add("LA07", "high", m,
                     "Positional slices line up a signal that already uses bar t's close with the return that ends at "
                     "bar t (for example a signal from `c[1:]` times `c[1:] / c[:-1] - 1`): the strategy earns the "
                     "move that created the signal. Make the position at bar t use data up to bar t-1 only: "
                     "`sig[:-1] * ret[1:]` when `sig` is not lagged yet (don't lag twice).")

    def _slice_write(self, st: ast.Assign) -> None:
        """pos[k:] = f(x): element m of pos holds f's element m - k, so pos's offset is off(f) - k."""
        tgt = st.targets[0]
        name = tgt.value.id
        sl = tgt.slice
        if name not in self.arrays:
            return  # not an array this pass knows (a DataFrame column write, a dict...)
        val = self._aoff(st.value, {})
        if not isinstance(sl, ast.Slice) or sl.step is not None:
            self._arr_ambiguous.add(name)  # pos[mask] = ..., pos[0] = ...: content no longer lines up predictably
            return
        k = 0 if sl.lower is None else _numc(sl.lower)
        if val is None and not any(isinstance(x, ast.Name) and (x.id in self.arrays) for x in ast.walk(st.value)):
            return  # a constant fill (pos[:5] = 0)
        if val is None or k is None or k < 0:
            self._arr_ambiguous.add(name)
            return
        kind = val[2] if val[2] in ("sig", "ret") else (self._name_kind(tgt.value) or val[2])
        old = self.arrays[name]
        if old[1] and old[2] is not None and (old[0], old[2]) != (val[0] - int(k), kind):
            self._arr_ambiguous.add(name)  # two different alignments written into one array
        self.arrays[name] = (val[0] - int(k), True, kind)

    def _loop_element_writes(self, loop: ast.AST) -> None:
        """for i in ...: pos[i + 1] = f(sig[i]) -> pos's offset is off(sig) - 1 (a lag)."""
        var, _row = self._loop_vars(loop)
        for st in ast.walk(loop):
            if not (isinstance(st, (ast.Assign, ast.AugAssign))):
                continue
            tgt = st.targets[0] if isinstance(st, ast.Assign) else st.target
            if not (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and tgt.value.id in self.arrays):
                continue
            name = tgt.value.id
            t_off = self._index_offset(tgt.slice, var)
            reads = [(self._index_offset(x.slice, var), x.value.id) for x in ast.walk(st.value)
                     if isinstance(x, ast.Subscript) and isinstance(x.value, ast.Name) and x.value.id in self.arrays]
            if t_off is None or not reads or any(r is None for r, _n in reads) or isinstance(st, ast.AugAssign):
                self._arr_ambiguous.add(name)
                continue
            info = max(r + self.arrays[n][0] for r, n in reads)
            kinds = {self.arrays[n][2] for _r, n in reads} - {None}
            v = st.value
            if isinstance(v, ast.IfExp) and isinstance(v.test, (ast.Compare, ast.BoolOp)) or isinstance(v, ast.Compare):
                kind = "sig"
            elif "sig" in kinds:
                kind = "sig"
            else:
                kind = self._name_kind(tgt.value) or next(iter(kinds), None)
            new = (info - t_off, True, kind)
            old = self.arrays[name]
            if old[1] and old[2] is not None and (old[0], old[2]) != (new[0], new[2]):
                self._arr_ambiguous.add(name)
            self.arrays[name] = new

    @staticmethod
    def _index_offset(sl: ast.AST, var: str | None) -> int | None:
        if var is None:
            return None
        if isinstance(sl, ast.Name) and sl.id == var:
            return 0
        if (isinstance(sl, ast.BinOp) and isinstance(sl.op, (ast.Add, ast.Sub)) and isinstance(sl.left, ast.Name)
                and sl.left.id == var and _num(sl.right) is not None):
            return int(_num(sl.right)) * (1 if isinstance(sl.op, ast.Add) else -1)
        return None

    def _array_where(self, call: ast.Call) -> bool | None:
        """np.where(cond, ret, 0) over positional arrays: True = leak, False = aligned, None = not array code."""
        c, a = self._aoff(call.args[0], {}), self._aoff(call.args[1], {})
        if c is None or a is None or not (c[1] or a[1]) or a[2] != "ret" or c[2] != "sig":
            return None
        return c[0] >= a[0]

    def rule_array_alignment(self) -> None:
        """Track numpy/list arrays and judge `sig * ret` products by positional offset (LA07)."""
        self.arrays: dict[str, tuple[int, bool, str | None]] = {}
        self._array_checked: set[int] = set()
        self._arr_ambiguous: set[str] = set()
        items = []
        for n in self._all:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                items.append((n.lineno, n.col_offset, n))
            elif (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript)
                  and isinstance(n.targets[0].value, ast.Name)):
                items.append((n.lineno, n.col_offset, n))
            elif isinstance(n, (ast.For, ast.While)) and self.inside(n, (ast.For, ast.While)) is None:
                items.append((n.lineno, n.col_offset, n))
        items.sort(key=lambda x: (x[0], x[1]))
        for _l, _c, st in items:
            if isinstance(st, (ast.For, ast.While)):
                self._loop_element_writes(st)
                continue
            if self.inside(st, (ast.For, ast.While)) is not None:
                continue
            if isinstance(st.targets[0], ast.Subscript):
                self._slice_write(st)
                continue
            name, v = st.targets[0].id, st.value
            self._arr_ambiguous.discard(name)
            if isinstance(v, (ast.ListComp, ast.GeneratorExp)) and len(v.generators) == 1:
                g = v.generators[0]
                env = self._bind_zip(g.target, g.iter, {})
                if env is None:
                    continue
                for m in ast.walk(v.elt):
                    if isinstance(m, ast.BinOp) and isinstance(m.op, ast.Mult):
                        self._judge_array_mult(m, env)
                val = self._aoff(v.elt, env)
                if val is not None and val[1]:
                    kind = val[2] or self._name_kind(st.targets[0])
                    self.arrays[name] = (val[0], True, kind)
                continue
            if _np_call(v, "where") and len(v.args) == 3 and classify(name) == "pnl" and self._array_where(v):
                self.add("LA07", "high", v, "`np.where(<condition on bar t>, <return ending at bar t>, ...)` over "
                                            "positional slices books the move that created the condition.")
            val = self._aoff(v, {})
            if val is None:
                continue
            kind = val[2]
            role_kind = self._name_kind(st.targets[0])
            if kind is None or (role_kind in ("sig", "ret") and kind == "diff"):
                kind = role_kind or kind
            self.arrays[name] = (val[0], val[1], kind)
            for m in ast.walk(v):
                if isinstance(m, ast.BinOp) and isinstance(m.op, ast.Mult):
                    self._judge_array_mult(m, {})
        # explicit loops over zip(c[:-1], c[1:])
        for loop in self._all:
            if not isinstance(loop, ast.For):
                continue
            env = self._bind_zip(loop.target, loop.iter, {})
            if env is None:
                continue
            for st in sorted((x for x in ast.walk(loop) if isinstance(x, (ast.Assign, ast.AugAssign, ast.Expr))),
                             key=lambda x: (x.lineno, x.col_offset)):
                val = st.value
                for m in ast.walk(val):
                    if isinstance(m, ast.BinOp) and isinstance(m.op, ast.Mult):
                        self._judge_array_mult(m, env)
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    v = self._aoff(val, env)
                    if v is not None:
                        env[st.targets[0].id] = (v[0], v[1], v[2] or self._name_kind(st.targets[0]))

    # -- LA07: same-bar decisions and fills inside bar loops ----------------------------------------
    def _bar_ref(self, sub: ast.AST, var: str | None, row: str | None) -> tuple[str, int] | None:
        """(key, bar offset) for x[i], x.iloc[i - 1], df.loc[i, 'close'], row['close'], row.close; else None."""
        if row is not None:
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id == row:
                k = _const_str(sub.slice)
                return (k, 0) if k else None
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == row:
                return sub.attr, 0
        if var is None or not isinstance(sub, ast.Subscript):
            return None
        sl, col = sub.slice, None
        if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
            sl, col = sl.elts[0], _const_str(sl.elts[1])
        if isinstance(sl, ast.Name) and sl.id == var:
            off = 0
        elif (isinstance(sl, ast.BinOp) and isinstance(sl.op, (ast.Add, ast.Sub)) and isinstance(sl.left, ast.Name)
              and sl.left.id == var and _num(sl.right) is not None):
            off = int(_num(sl.right)) * (1 if isinstance(sl.op, ast.Add) else -1)
        else:
            return None
        base = sub.value
        while isinstance(base, ast.Attribute) and base.attr in ("iloc", "loc", "iat", "at", "values"):
            base = base.value
        if isinstance(base, ast.Call) and _attr(base) in ("to_numpy", "tolist"):
            base = base.func.value
        k = col or _key(base)[0]
        return (k, off) if k else None

    def _close_info(self, k: str) -> bool:
        """`k` at bar i is known only at bar i's close (a close price, or something computed from it)."""
        ki = self.info.get(k)
        if ki is not None and (ki.lagged or ki.forward):
            return False
        if _is_close_name(k):
            return True
        return ki is not None and ki.post_open and not _is_post_open_name(k)

    def _loop_vars(self, loop: ast.AST) -> tuple[str | None, str | None]:
        if isinstance(loop, ast.For):
            t, it = loop.target, loop.iter
            if isinstance(it, ast.Call) and _attr(it) in ("iterrows", "itertuples"):
                if isinstance(t, ast.Tuple) and len(t.elts) == 2 and isinstance(t.elts[1], ast.Name):
                    return None, t.elts[1].id
                if isinstance(t, ast.Name):
                    return None, t.id
                return None, None
            if isinstance(t, ast.Name):
                return t.id, None
            if isinstance(t, ast.Tuple) and t.elts and isinstance(t.elts[0], ast.Name):
                return t.elts[0].id, None
            return None, None
        incs = [x.target.id for x in ast.walk(loop) if isinstance(x, ast.AugAssign) and isinstance(x.op, ast.Add)
                and isinstance(x.target, ast.Name) and _num(x.value) is not None]
        return (incs[0] if incs else None), None

    def rule_loop_same_bar(self) -> None:
        for loop in self._all:
            if not isinstance(loop, (ast.For, ast.While)):
                continue
            var, row = self._loop_vars(loop)
            if var is None and row is None:
                continue
            flagged = False
            fresh = None  # computed on first need (v0.5 perf)
            for st in ast.walk(loop):
                if flagged:
                    break
                if not isinstance(st, ast.If) or self.inside(st, (ast.For, ast.While)) is not loop:
                    continue
                refs = [self._bar_ref(x, var, row) for x in ast.walk(st.test)]
                close_now = next((r[0] for r in refs if r is not None and r[1] == 0 and self._close_info(r[0])), None)
                if close_now is None:  # v0.4: go_long = c[t] > hi[t] earlier in this iteration, then `if go_long:`
                    if fresh is None:
                        fresh = self._loop_close_vars(loop, var, row)
                    close_now = next((f"{x.id} = {fresh[x.id][0]}" for x in ast.walk(st.test) if isinstance(x, ast.Name)
                                      and x.id in fresh and fresh[x.id][1] < st.lineno), None)
                if close_now is None or any(r is not None and r[1] > 0 for r in refs):
                    continue
                flagged = self._loop_branch(st, var, row, close_now)
            if not flagged and var is not None:
                self._loop_products(loop, var)

    def _loop_close_vars(self, loop: ast.AST, var, row) -> dict[str, tuple[str, int]]:
        """Names assigned directly in the loop body (not in a branch) from bar `var`'s close and nothing later:
        name -> (the close key, line). A name assigned anywhere else in the loop is left out."""
        out: dict[str, tuple[str, int]] = {}
        other: set[str] = set()
        body = getattr(loop, "body", [])
        for st in ast.walk(loop):
            if isinstance(st, (ast.Assign, ast.AugAssign)) and st not in body:
                for t in (st.targets if isinstance(st, ast.Assign) else [st.target]):
                    other |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
        for st in body:
            if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
                continue
            refs = [self._bar_ref(x, var, row) for x in ast.walk(st.value)]
            now = next((r[0] for r in refs if r is not None and r[1] == 0 and self._close_info(r[0])), None)
            name = st.targets[0].id
            if now is not None and not any(r is not None and r[1] > 0 for r in refs) and name not in other:
                out[name] = (now, st.lineno)
        return out

    def _loop_branch(self, branch: ast.If, var, row, cond_key: str) -> bool:
        body_nodes = [x for b in branch.body for x in ast.walk(b)]
        for x in body_nodes:
            pairs: list[tuple[str, ast.AST, str]] = []  # (target name, value, op)
            if isinstance(x, ast.Assign):
                for t in x.targets:
                    if isinstance(t, ast.Tuple) and isinstance(x.value, ast.Tuple) and len(t.elts) == len(x.value.elts):
                        pairs += [(_key(a)[0] or "", b, "=") for a, b in zip(t.elts, x.value.elts)]
                    else:
                        pairs.append((_key(t)[0] or "", x.value, "="))
            elif isinstance(x, ast.AugAssign):
                pairs.append((_key(x.target)[0] or "", x.value, "aug"))
            elif isinstance(x, ast.Call) and _attr(x) == "append" and x.args:
                k = _key(x.func.value)[0] or ""
                pairs.append((k, x.args[0], "aug"))
            for tname, val, op in pairs:
                for sub in ast.walk(val):
                    r = self._bar_ref(sub, var, row)
                    if r is None or r[1] != 0:
                        continue
                    k = r[0]
                    if _is_open_name(k):
                        self.add("LA07", "high", x,
                                 f"The decision uses bar {var or 'row'}'s close (`{cond_key}`), and this line fills at the same "
                                 f"bar's open (`{k}`), which happened before that close. Fill at the next bar's open.")
                        return True
                    if _is_close_name(k) and _tokens(tname) & ENTRY_T:
                        self.add("LA07", "medium", x,
                                 f"The entry is filled at bar {var or 'row'}'s close (`{k}`), the same close the decision "
                                 f"used (`{cond_key}`). Fill at the next bar's open, or compute the signal before the close.")
                        return True
                    ki = self.info.get(k)
                    if (op == "aug" and (classify(k) == "return" or (ki is not None and ki.is_return))
                            and not (ki is not None and (ki.lagged or ki.forward))):
                        self.add("LA07", "high", x,
                                 f"The decision uses bar {var or 'row'}'s close (`{cond_key}`) and books the same bar's "
                                 f"return (`{k}`), which ended at that close. Book the next bar's return instead.")
                        return True
        return False

    def _bar_return_now(self, node: ast.AST, var: str) -> str | None:
        """`ret[i]` (a return column read at bar i) or `c[i] / c[i - 1] - 1`: a return that ends at bar i."""
        r = self._bar_ref(node, var, None)
        if r is not None and r[1] == 0:
            ki = self.info.get(r[0])
            if (classify(r[0]) == "return" or (ki is not None and ki.is_return)) and not (ki is not None and (ki.lagged or ki.forward)):
                return r[0]
        for d in ast.walk(node):
            if isinstance(d, ast.BinOp) and isinstance(d.op, (ast.Div, ast.Sub)):
                a, b = self._bar_ref(d.left, var, None), self._bar_ref(d.right, var, None)
                if a and b and a[0] == b[0] and a[1] == 0 and b[1] < 0 and _is_price_name(a[0]):
                    return f"{a[0]}[{var}] / {a[0]}[{var} - 1]"
        return None

    def _loop_fresh_positions(self, loop: ast.AST, var: str) -> None:
        """pos = 1 if close[i] > ma[i] else 0, then pnl += pos * ret[i] later in the same iteration."""
        mine = sorted((x for x in ast.walk(loop) if isinstance(x, ast.stmt) and x is not loop
                       and self.inside(x, (ast.For, ast.While)) is loop), key=lambda x: (x.lineno, x.col_offset))
        fresh: dict[str, str] = {}
        for st in mine:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                refs = [self._bar_ref(x, var, None) for x in ast.walk(st.value)]
                now = next((r[0] for r in refs if r and r[1] == 0 and self._close_info(r[0])), None)
                br = self.inside(st, (ast.If,))
                if now is None and br is not None and self.inside(br, (ast.For, ast.While)) is loop:
                    trefs = [self._bar_ref(x, var, None) for x in ast.walk(br.test)]
                    now = next((r[0] for r in trefs if r and r[1] == 0 and self._close_info(r[0])), None)
                    if any(r and r[1] > 0 for r in trefs):
                        now = None
                if now is not None and not any(r and r[1] > 0 for r in refs):
                    fresh[st.targets[0].id] = now
                elif st.targets[0].id in fresh:
                    fresh.pop(st.targets[0].id)
                continue
            for m in ast.walk(st):
                if not (isinstance(m, ast.BinOp) and isinstance(m.op, ast.Mult)):
                    continue
                for a, b in ((m.left, m.right), (m.right, m.left)):
                    if isinstance(a, ast.Name) and a.id in fresh:
                        ret = self._bar_return_now(b, var)
                        if ret is not None:
                            self.add("LA07", "high", m,
                                     f"`{a.id}` is set from bar {var}'s close (`{fresh[a.id]}`) earlier in this iteration and "
                                     f"then multiplied by the return that ends at bar {var} (`{ret}`). Book the PnL with "
                                     "the previous bar's position before updating it.")
                            return

    def _loop_products(self, loop: ast.AST, var: str) -> None:
        """pnl += sig[i] * ret[i] inside a bar loop."""
        self._loop_fresh_positions(loop, var)
        for m in ast.walk(loop):
            if not (isinstance(m, ast.BinOp) and isinstance(m.op, ast.Mult)):
                continue
            a, b = self._bar_ref(m.left, var, None), self._bar_ref(m.right, var, None)
            if a is None or b is None:
                continue
            ra, rb = self._side_role(ast.Name(id=a[0])), self._side_role(ast.Name(id=b[0]))
            pair = {ra[0]: (a, ra), rb[0]: (b, rb)}
            if set(pair) != {"signal", "return"}:
                continue
            (sk, soff), (_sr, sinfo, _sn) = pair["signal"]
            (rk, roff), _r = pair["return"]
            if soff < roff:
                continue
            sev = "high" if sinfo is not None and sinfo.price_derived else "medium"
            self.add("LA07", sev, m, f"`{sk}[{var}]` is multiplied by the same bar's return `{rk}[{var}]`: the position "
                                     f"earns the move that created it. Use `{sk}[{var} - 1]`.")
            return

    # -- LA05: fit before split -------------------------------------------------------------------
    def rule_fit_before_split(self) -> None:
        split_lines: list[int] = []
        constructors: dict[str, str] = {}
        for n in self._all:
            if isinstance(n, ast.Call):
                fn = _func_name(n)
                if fn in ("train_test_split", "TimeSeriesSplit", "KFold", "PurgedKFold"):
                    split_lines.append(n.lineno)
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    names = [e for e in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t])]
                    for e in names:
                        k, _ = _key(e)
                        if k and _tokens(k) & SPLIT_T:
                            split_lines.append(n.lineno)
                    if isinstance(t, ast.Name) and isinstance(n.value, ast.Call):
                        cn = _func_name(n.value)
                        if cn:
                            constructors[t.id] = cn
        first_split = min(split_lines) if split_lines else None
        has_predict = any(isinstance(n, ast.Call) and _attr(n) in ("predict", "predict_proba", "score") for n in self._all)

        for n in self._all:
            if not (isinstance(n, ast.Call) and _attr(n) in ("fit", "fit_transform")):
                continue
            if self.inside(n, (ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ListComp)) is not None:
                continue  # probably a walk-forward loop or a helper: too uncertain to flag
            if not n.args:
                continue
            arg = n.args[0]
            if self._fit_subset(arg):
                continue  # v0.3: X.loc[:cut], data.iloc[:k, :-1], Xa[:k], X[date_mask], a frame cut by date: a training subset
            k, _ = _key(_chain_base(arg))
            if k and _tokens(k) & SPLIT_T:
                continue
            recv = n.func.value
            est = _func_name(recv) if isinstance(recv, ast.Call) else constructors.get(getattr(recv, "id", ""), None)
            is_pre = est in PREPROCESSORS or _attr(n) == "fit_transform"
            label = f"`{est}`" if est else "the estimator"
            base = self._fit_base(arg)
            if is_pre and base is not None and self._subset_fit_on(base, n):
                self.add("LA05", "medium", n,
                         f"{label} is fitted on every row of `{base}`, while the model is trained on a subset of "
                         f"`{base}`: the statistics of the held-out rows are inside the training features. Fit it on "
                         "the training subset only.")
                continue
            if first_split is not None and n.lineno < first_split:
                self.add("LA05", "high", n,
                         f"{label} is fitted on all rows (line {n.lineno}) before the train/test split on line {first_split}, "
                         "so test-period information leaks into training.")
            elif first_split is not None and k and any(nm.startswith(k + "_") and _tokens(nm) & SPLIT_T for nm in self.assigned_names):
                self.add("LA05", "medium", n,
                         f"{label} is fitted on `{k}` although a training subset of it exists; fit on the training rows only.")
            elif first_split is None and not is_pre and has_predict:
                self.add("LA05", "medium", n,
                         f"{label} is fitted and then used to predict without any train/test split in this code: "
                         "results are in-sample.")
            elif first_split is None and is_pre and self._output_sliced_later(n):
                self.add("LA05", "high", n,
                         f"{label} is fitted on every row, and its output is then sliced into training and test rows: "
                         "the test rows' statistics are inside the training features.")
            elif first_split is None and is_pre and self.ts_context:
                self.add("LA05", "possible", n,
                         f"{label} is fitted on every row. If these rows include the period you evaluate, its "
                         "statistics include the future.")

    _PASS_THROUGH = ("to_numpy", "dropna", "fillna", "astype", "copy", "reshape", "ffill", "sort_index", "values")

    def _is_date_operand(self, x: ast.AST) -> bool:
        return any(isinstance(y, (ast.Subscript, ast.Attribute)) and (
            self._date_col(y) or (isinstance(y, ast.Attribute) and y.attr in ("year", "month", "date")))
            for y in ast.walk(x))

    def _date_bound(self, m: ast.AST, depth: int = 0) -> str | None:
        """Direction of a date mask: "upper" (rows up to a date: df.date < cut, between, isin), "lower" (rows from a
        date on: df.index >= '2000', ~train), or None if it isn't a date mask."""
        if depth > 4:
            return None
        if isinstance(m, ast.UnaryOp) and isinstance(m.op, (ast.Invert, ast.Not)):
            b = self._date_bound(m.operand, depth + 1)
            return {"upper": "lower", "lower": "upper"}.get(b) if b else None
        if isinstance(m, (ast.BoolOp, ast.BinOp)):
            parts = m.values if isinstance(m, ast.BoolOp) else [m.left, m.right]
            bs = [self._date_bound(p, depth + 1) for p in parts]
            both = isinstance(m, ast.BoolOp) and isinstance(m.op, ast.And) or (
                isinstance(m, ast.BinOp) and isinstance(m.op, ast.BitAnd))
            if both:
                return "upper" if "upper" in bs else ("lower" if "lower" in bs else None)
            return "upper" if bs and all(b == "upper" for b in bs) else ("lower" if any(bs) else None)
        if isinstance(m, ast.Name):
            assigns = self._assigns_of(m.id)
            return self._date_bound(assigns[0].value, depth + 1) if len(assigns) == 1 else None
        if isinstance(m, ast.Call) and _attr(m) in ("isin", "between") and self._is_date_operand(m.func.value):
            return "upper"
        if isinstance(m, ast.Compare):
            if len(m.ops) > 1:
                return "upper" if self._is_date_operand(m) else None
            op, left, right = m.ops[0], m.left, m.comparators[0]
            if self._is_date_operand(left):
                return "upper" if isinstance(op, (ast.Lt, ast.LtE, ast.Eq)) else (
                    "lower" if isinstance(op, (ast.Gt, ast.GtE)) else None)
            if self._is_date_operand(right):
                return "upper" if isinstance(op, (ast.Gt, ast.GtE, ast.Eq)) else (
                    "lower" if isinstance(op, (ast.Lt, ast.LtE)) else None)
        return None

    def _date_mask(self, m: ast.AST, depth: int = 0) -> bool:
        """A training-subset mask: a date mask bounded above (`df.date < cut`, `between`). A start-date filter
        (`df.index >= '2000'`) or a test-window mask keeps the later rows, so it is not a training subset."""
        return self._date_bound(m, depth) == "upper"

    def _fit_subset(self, node: ast.AST, depth: int = 0) -> bool:
        """The data passed to fit is a training subset: a bounded slice (`.loc[:cut]`, `.iloc[:k, :-1]`, `[:k]`,
        `.loc[a:b]`), a date mask (`X[mask]`, `df[df.date < cut]`), a split-named frame, or a name bound to one."""
        if depth > 6:
            return False
        if isinstance(node, ast.Call):
            if _attr(node) == "transform" and node.args:
                return self._fit_subset(node.args[0], depth + 1)
            if _attr(node) in self._PASS_THROUGH:
                return self._fit_subset(node.func.value, depth + 1)
            if _func_name(node) in ("asarray", "array", "nan_to_num", "DataFrame") and node.args:
                return self._fit_subset(node.args[0], depth + 1)
            return False
        if isinstance(node, ast.Attribute) and node.attr in ("values", "loc", "iloc"):
            return self._fit_subset(node.value, depth + 1)
        if isinstance(node, ast.Subscript):
            sl = node.slice
            first = sl.elts[0] if isinstance(sl, ast.Tuple) and sl.elts else sl
            if isinstance(first, ast.Slice):
                if first.upper is not None:
                    return True  # .loc[:cut], .iloc[:k], [:k], .loc[a:b]
                if first.lower is not None:
                    return False  # .loc['2000':], [20:]: a start filter keeps every later row, test rows included
                return self._fit_subset(node.value, depth + 1)  # [:, cols]: every row
            bound = self._date_bound(first)
            if bound == "upper":
                return True
            if bound == "lower":
                return False
            if isinstance(first, ast.Name) and _tokens(first.id) & SPLIT_T:
                return True
            return self._fit_subset(node.value, depth + 1)  # a column selection
        if isinstance(node, ast.Name):
            if _tokens(node.id) & SPLIT_T:
                return True
            assigns = self._assigns_of(node.id)
            return len(assigns) == 1 and len(assigns[0].targets) == 1 and self._fit_subset(assigns[0].value, depth + 1)
        return False

    def _fit_base(self, node: ast.AST) -> str | None:
        for _ in range(10):
            if isinstance(node, ast.Call):
                if _attr(node) == "transform" and node.args:
                    node = node.args[0]
                elif isinstance(node.func, ast.Attribute):
                    node = node.func.value
                else:
                    return None
            elif isinstance(node, (ast.Subscript, ast.Attribute)):
                node = node.value
            elif isinstance(node, ast.Name):
                return node.id
            else:
                return None
        return None

    def _subset_fit_on(self, base: str, pre: ast.Call) -> bool:
        """Another estimator is fitted on a subset of `base` (directly or through `scaler.transform(...)`)."""
        for c in self._all:
            if (c is not pre and isinstance(c, ast.Call) and _attr(c) in ("fit", "fit_transform", "partial_fit")
                    and c.args and self._fit_base(c.args[0]) == base and self._fit_subset(c.args[0])):
                return True
        return False

    def _output_sliced_later(self, call: ast.Call) -> bool:
        tkey, _f, stmt = self.enclosing_assign_target(call)
        if not tkey or not isinstance(stmt, ast.Assign) or not isinstance(stmt.targets[0], ast.Name):
            return False
        for c in self._all:
            if (isinstance(c, ast.Call) and _attr(c) in ("fit", "score", "predict", "predict_proba")
                    and getattr(c, "lineno", 0) > stmt.lineno):
                for a in c.args:
                    if (isinstance(a, ast.Subscript) and isinstance(a.value, ast.Name) and a.value.id == tkey
                            and isinstance(a.slice, (ast.Slice, ast.Name))):
                        return True
        return False

    # -- LA10: feature matrix that keeps forward columns --------------------------------------
    def _forward_cols(self, frame: str | None) -> set[str]:
        out = {k for k, ki in self.info.items()
               if ki.forward and ki.frame is not None and (frame is None or ki.frame == frame)
               and classify(k) not in ("pnl", "eval")}
        for name, c in self._containers.items():  # named forward columns stored in a new frame
            if (frame is None or frame == name) and self._cont_ahead(c, c["src"]):  # unless lagged by the full lead
                out |= {col for col in c["cols"] if classify(col) not in ("pnl", "eval")}
        return out

    # Column sets are evaluated per column name with three values: True (in the set), False (not in it), None (the
    # lint can't tell). v0.3 resolves constants, `+` of lists, comprehensions over `df.columns` with startswith /
    # endswith / in / == / regex conditions, and `df.columns.str.startswith(...)` / `isin(...)` masks. Anything it
    # can't resolve is None, which is never reported as high.
    def _colset(self, node: ast.AST | None, depth: int = 0):
        """A predicate name -> True/False/None for a column-list expression, or None if it isn't one we can read."""
        if node is None or depth > 6:
            return None
        s = _const_str(node)
        if s is not None:
            return lambda c, s=s: c == s
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            consts = {_const_str(e) for e in node.elts if _const_str(e) is not None}
            subs = [self._colset(e, depth + 1) for e in node.elts if _const_str(e) is None]
            if any(p is None for p in subs):
                return lambda c, consts=consts: True if c in consts else None
            return lambda c, consts=consts, subs=subs: _or3([c in consts] + [p(c) for p in subs])
        if isinstance(node, ast.Starred):
            return self._colset(node.value, depth + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.BitOr)):
            a, b = self._colset(node.left, depth + 1), self._colset(node.right, depth + 1)
            a = a or (lambda c: None)
            b = b or (lambda c: None)
            return lambda c, a=a, b=b: _or3([a(c), b(c)])
        if isinstance(node, ast.Name):
            assigns = self._assigns_of(node.id)
            if len(assigns) == 1 and len(assigns[0].targets) == 1:
                return self._colset(assigns[0].value, depth + 1)
            return None
        if isinstance(node, ast.Call) and _func_name(node) in ("list", "sorted", "set", "tuple") and len(node.args) == 1:
            return self._colset(node.args[0], depth + 1)
        if isinstance(node, ast.Call) and _attr(node) in ("tolist", "to_list") and not node.args:
            return self._colset(node.func.value, depth + 1)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)) and len(node.generators) == 1:
            g = node.generators[0]
            if not (isinstance(g.target, ast.Name) and isinstance(node.elt, ast.Name) and node.elt.id == g.target.id):
                return None
            if not self._is_columns(g.iter):
                inner = self._colset(g.iter, depth + 1)
                if inner is None:
                    return None
            else:
                inner = lambda c: True  # noqa: E731
            var, ifs = g.target.id, list(g.ifs)
            return lambda c, var=var, ifs=ifs, inner=inner: _and3([inner(c)] + [self._cond(t, var, c) for t in ifs])
        if isinstance(node, ast.Subscript) and self._is_columns(node.value):  # df.columns[mask]
            mask = node.slice
            return lambda c, mask=mask: self._colmask(mask, c)
        return None

    def _is_columns(self, node: ast.AST) -> bool:
        while isinstance(node, ast.Call) and _attr(node) in ("tolist", "to_list") or (
                isinstance(node, ast.Call) and _func_name(node) in ("list", "sorted") and len(node.args) == 1):
            node = node.args[0] if _func_name(node) in ("list", "sorted") and not isinstance(node.func, ast.Attribute) else node.func.value
        return isinstance(node, ast.Attribute) and node.attr == "columns"

    def _strs(self, node: ast.AST | None) -> tuple[str, ...] | None:
        s = _const_str(node)
        if s is not None:
            return (s,)
        if isinstance(node, ast.Name) and node.id in self.str_consts:
            return (self.str_consts[node.id],)
        if isinstance(node, (ast.Tuple, ast.List)):
            vals = [_const_str(e) for e in node.elts]
            return tuple(vals) if all(v is not None for v in vals) else None
        return None

    def _cond(self, t: ast.AST, var: str, c: str):
        """Evaluate a comprehension condition on column name `c`: True/False/None."""
        if isinstance(t, ast.BoolOp):
            vals = [self._cond(v, var, c) for v in t.values]
            return _and3(vals) if isinstance(t.op, ast.And) else _or3(vals)
        if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
            v = self._cond(t.operand, var, c)
            return None if v is None else not v
        is_var = lambda x: isinstance(x, ast.Name) and x.id == var  # noqa: E731
        if isinstance(t, ast.Call) and isinstance(t.func, ast.Attribute) and is_var(t.func.value) and t.args:
            pre = self._strs(t.args[0])
            if pre is not None and t.func.attr == "startswith":
                return c.startswith(pre)
            if pre is not None and t.func.attr == "endswith":
                return c.endswith(pre)
            return None
        if isinstance(t, ast.Call) and _func_name(t) in ("match", "search", "fullmatch") and len(t.args) >= 2 and is_var(t.args[1]):
            pat = _const_str(t.args[0])
            if pat is None:
                return None
            try:
                return bool(getattr(re, _func_name(t))(pat, c))
            except re.error:
                return None
        if isinstance(t, ast.Compare) and len(t.ops) == 1:
            left, op, right = t.left, t.ops[0], t.comparators[0]
            if is_var(left) and isinstance(op, (ast.Eq, ast.NotEq)) and _const_str(right) is not None:
                return (c == _const_str(right)) == isinstance(op, ast.Eq)
            if is_var(left) and isinstance(op, (ast.In, ast.NotIn)):
                p = self._colset(right)
                v = p(c) if p is not None else None
                return None if v is None else (v if isinstance(op, ast.In) else not v)
            if is_var(right) and isinstance(op, (ast.In, ast.NotIn)) and _const_str(left) is not None:
                v = _const_str(left) in c
                return v if isinstance(op, ast.In) else not v
        return None

    def _colmask(self, m: ast.AST, c: str):
        """Evaluate a boolean column mask (`~df.columns.str.startswith('fwd_')`, `df.columns.isin([...])`)."""
        if isinstance(m, ast.UnaryOp) and isinstance(m.op, (ast.Invert, ast.Not)):
            v = self._colmask(m.operand, c)
            return None if v is None else not v
        if isinstance(m, ast.BinOp) and isinstance(m.op, (ast.BitAnd, ast.BitOr)):
            a, b = self._colmask(m.left, c), self._colmask(m.right, c)
            return _and3([a, b]) if isinstance(m.op, ast.BitAnd) else _or3([a, b])
        if isinstance(m, ast.Call) and isinstance(m.func, ast.Attribute) and m.args:
            recv = m.func.value
            if (m.func.attr in ("startswith", "endswith", "contains") and isinstance(recv, ast.Attribute)
                    and recv.attr == "str" and self._is_columns(recv.value)):
                pre = self._strs(m.args[0])
                if pre is None:
                    return None
                if m.func.attr == "startswith":
                    return c.startswith(pre)
                if m.func.attr == "endswith":
                    return c.endswith(pre)
                if _kw(m, "regex") is not None and not (isinstance(_kw(m, "regex"), ast.Constant) and _kw(m, "regex").value):
                    return any(p in c for p in pre)
                try:
                    return bool(re.search(pre[0], c)) if len(pre) == 1 else None
                except re.error:
                    return None
            if m.func.attr == "isin" and self._is_columns(recv):
                p = self._colset(m.args[0])
                return p(c) if p is not None else None
        return None

    def _frame_cols(self, node: ast.AST):
        """For a frame expression, (base frame node, kept-columns predicate or None if all columns are kept, ok).
        Handles df, df[cols], df.loc[:, cols], df.loc[rows, cols], df[cols][rows], chained calls (dropna...)."""
        keep = None
        while True:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _attr(node) in (
                    "dropna", "fillna", "copy", "astype", "select_dtypes", "sort_index", "sort_values", "reset_index",
                    "set_index", "head", "tail", "to_numpy", "values", "replace", "clip", "ffill"):
                node = node.func.value
            elif isinstance(node, ast.Attribute) and node.attr in ("values",):
                node = node.value
            elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr in ("loc", "iloc"):
                sl = node.slice
                if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
                    if node.value.attr == "loc" and not isinstance(sl.elts[1], ast.Slice):
                        p = self._colset(sl.elts[1]) or (lambda c, m=sl.elts[1]: self._colmask(m, c))
                        keep = _merge_keep(keep, p)
                node = node.value.value
            elif isinstance(node, ast.Subscript):
                sl = node.slice
                p = self._colset(sl) if not isinstance(sl, (ast.Slice, ast.Compare, ast.Constant)) or _const_str(sl) else None
                if p is not None and not (isinstance(sl, ast.Name) and self._is_row_mask(sl)):
                    keep = _merge_keep(keep, p)
                elif isinstance(sl, ast.Name) and not self._is_row_mask(sl) and self._colset(sl) is None:
                    return node.value, keep, False  # df[cols] with cols we can't read
                node = node.value
            else:
                return node, keep, True

    def _is_row_mask(self, nm: ast.Name) -> bool:
        assigns = self._assigns_of(nm.id)
        return bool(assigns) and all(isinstance(a.value, (ast.Compare, ast.BoolOp, ast.UnaryOp)) or (
            isinstance(a.value, ast.Call) and _attr(a.value) in ("isin", "between", "notna", "isna")) or (
            isinstance(a.value, ast.BinOp) and isinstance(a.value.op, (ast.BitAnd, ast.BitOr))) for a in assigns)

    def _report_cols(self, n: ast.AST, frame_node: ast.AST, keep, dropped, what: str) -> None:
        """Report forward columns that survive `keep` minus `dropped`: high if certain, possible if unknown."""
        frame = frame_node.id if isinstance(frame_node, ast.Name) and frame_node.id != "__any__" else None
        sure, maybe = [], []
        for col in sorted(self._forward_cols(frame)):
            k = keep(col) if keep is not None else True
            d = dropped(col) if dropped is not None else False
            if k is False or d is True:
                continue
            (sure if (k is True and d is False) else maybe).append(col)
        if sure:
            self.add("LA10", "high", n, f"{what} keeps forward-looking column(s) {sure}.")
        elif maybe:
            self.add("LA10", "possible", n,
                     f"{what} may keep forward-looking column(s) {maybe}; the lint could not resolve the column list.")

    def rule_feature_matrix(self) -> None:
        drop_names: set[str] = set()  # names used as drop lists: comprehensions that select columns to remove
        for n in self._all:
            if isinstance(n, ast.Call) and _attr(n) == "drop":
                for a in list(n.args) + [kw.value for kw in n.keywords if kw.arg in ("columns", "labels")]:
                    drop_names |= {x.id for x in ast.walk(a) if isinstance(x, ast.Name)}
        for n in self._all:
            if not isinstance(n, ast.Assign) or len(n.targets) != 1:
                continue
            tk, _ = _key(n.targets[0]) if not isinstance(n.targets[0], ast.Tuple) else (None, None)
            if tk is None or classify(tk) not in ("feature",) and not (_tokens(tk) & {"x", "features", "feature", "feats"}):
                continue
            v = n.value
            # X = df.drop(columns=[...]) / df.drop('target', axis=1) / df.loc[:, mask].drop(...)
            call = v if isinstance(v, ast.Call) else None
            while call is not None and _attr(call) != "drop":
                call = call.func.value if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Call) else None
            if call is not None and _attr(call) == "drop":
                spec = _kw(call, "columns") or (call.args[0] if call.args else _kw(call, "labels"))
                if spec is None:
                    continue
                dropped = self._colset(spec) or (lambda c: None)
                base, keep, ok = self._frame_cols(call.func.value)
                if not ok:
                    continue
                names = sorted(c for c in self._forward_cols(None) if dropped(c) is True)
                self._report_cols(n, base, keep, dropped, f"`{tk}` (dropping {names or 'other columns'})")
                continue
            # X = df[['a', 'fwd_ret']] / df[feature_cols] / df.loc[:, cols]
            base, keep, ok = self._frame_cols(v)
            if ok and keep is not None and isinstance(base, ast.Name):
                self._report_cols(n, base, keep, None, f"The feature matrix `{tk}`")
        # model.fit(df[['mom', 'fwd_ret']], y) / model.fit(train[features], ...) / model.predict(df.loc[m, cols])
        for n in self._all:
            if not (isinstance(n, ast.Call) and (_attr(n) in FIT_ATTRS or _func_name(n) in TRAIN_DATA_CTORS) and n.args):
                continue
            base, keep, ok = self._frame_cols(n.args[0])
            if ok and keep is not None and isinstance(base, ast.Name):
                frame_any = ast.Name(id="__any__")  # the fitted frame is often a slice (train) of the panel
                self._report_cols(n, frame_any, keep, None, f"The input to `{_func_name(n)}(...)`")
        # features = [c for c in df.columns if c != 'target']
        for n in self._all:
            if not isinstance(n, ast.ListComp) or len(n.generators) != 1 or not n.generators[0].ifs:
                continue
            g = n.generators[0]
            if not self._is_columns(g.iter):
                continue
            tkey, _f, st = self.enclosing_assign_target(n)
            parent = self.parents.get(n)
            if tkey is None and isinstance(parent, ast.keyword) and parent.arg in ("columns", "labels"):
                continue  # df.drop(columns=[c for c ...]): handled above
            if tkey is not None and (tkey in drop_names or classify(tkey) in ("label", "eval", "pnl")
                                     or _tokens(tkey) & {"drop", "exclude", "excluded", "leak", "leaky", "leaks"}):
                continue  # fwd_cols = [c for c in df.columns if c.startswith('fwd_')]: the columns to remove
            if tkey is not None and isinstance(st, ast.Assign) and st.value is not n:
                continue  # part of a larger expression (drop(...), a + b): handled where it is used
            pred = self._colset(n)
            frame = g.iter.value if isinstance(g.iter, ast.Attribute) else None
            self._report_cols(n, frame if isinstance(frame, ast.Name) else ast.Name(id="__any__"), pred, None,
                              "The column list" + (f" `{tkey}`" if tkey else ""))

    # -- LA11: grid search on the full sample -------------------------------------------------
    def _grid_len(self, it: ast.AST, depth: int = 0) -> int | None:
        if depth > 6:
            return None
        if isinstance(it, (ast.List, ast.Tuple, ast.Set)):
            return len(it.elts)
        if isinstance(it, ast.Call) and _func_name(it) in ("list", "tuple", "sorted") and len(it.args) == 1:
            return self._grid_len(it.args[0], depth + 1)
        if isinstance(it, ast.Name):
            first = next(iter(self._assigns_of(it.id)), None)
            return self._grid_len(first.value, depth + 1) if first is not None else None
        if isinstance(it, ast.Call):
            fn = _func_name(it)
            nums = [_num(a) for a in it.args]
            if fn == "range" and nums and all(x is not None for x in nums):
                start, stop, step = (0, nums[0], 1) if len(nums) == 1 else (nums[0], nums[1], nums[2] if len(nums) > 2 else 1)
                return max(0, int((stop - start + step - (1 if step > 0 else -1)) // step))
            if fn == "arange" and nums and all(x is not None for x in nums):
                start, stop, step = (0, nums[0], 1) if len(nums) == 1 else (nums[0], nums[1], nums[2] if len(nums) > 2 else 1)
                return max(0, int(-(-(stop - start) // step)))
            if fn == "linspace":
                num = _kw(it, "num") or (it.args[2] if len(it.args) > 2 else None)
                return int(_num(num)) if num is not None and _num(num) is not None else 50
            if fn in ("product",):
                lens = [self._grid_len(a, depth + 1) for a in it.args]
                if lens and all(x for x in lens):
                    out = 1
                    for x in lens:
                        out *= x
                    return out
            if fn in ("ParameterGrid",):
                return None
        return None

    def _is_grid_iter(self, it: ast.AST, depth: int = 0) -> bool:
        if depth > 6:
            return False  # x = [f(v) for v in x] and other self-references
        if isinstance(it, (ast.List, ast.Tuple)) and len(it.elts) >= 2 and all(_num(e) is not None for e in it.elts):
            return True
        if isinstance(it, ast.Call) and _func_name(it) in ("list", "tuple", "sorted") and len(it.args) == 1:
            return self._is_grid_iter(it.args[0], depth + 1)
        if isinstance(it, (ast.ListComp, ast.GeneratorExp)):
            return any(self._is_grid_iter(g.iter, depth + 1) for g in it.generators)
        if isinstance(it, ast.Call) and _func_name(it) in ("range", "arange", "linspace", "product", "ParameterGrid", "logspace"):
            return True
        if isinstance(it, ast.Name):
            first = next(iter(self._assigns_of(it.id)), None)
            return self._is_grid_iter(first.value, depth + 1) if first is not None else False
        return False

    def _toks_of(self, node: ast.AST) -> set[str]:
        names = {x.id for x in ast.walk(node) if isinstance(x, ast.Name)} | {
            x.attr for x in ast.walk(node) if isinstance(x, ast.Attribute)} | {
            _const_str(x) for x in ast.walk(node) if _const_str(x)}
        return set().union(*(_tokens(s) for s in names if s)) if names else set()

    def rule_grid_search(self) -> None:
        metric_t = {"sharpe", "sortino", "calmar", "cagr", "profit", "pnl", "perf", "performance", "returns", "ret",
                    "equity", "total", "score"}
        for n in self._all:
            comp = isinstance(n, (ast.DictComp, ast.ListComp)) and isinstance(self.parents.get(n), ast.Assign)
            if not (isinstance(n, ast.For) or comp) or self.inside(n, (ast.For,)) is not None:
                continue
            if comp:
                grid = [g for g in n.generators if self._is_grid_iter(g.iter)]
            else:
                grid = [lp for lp in ast.walk(n) if isinstance(lp, ast.For) and self._is_grid_iter(lp.iter)]
            if not grid:
                continue
            # skip loops over bar indices (`for i in range(len(df))`) - that's a loop backtest, not a grid
            if all(isinstance(lp.iter, ast.Call) and _func_name(lp.iter) == "range" and any(
                    isinstance(a, ast.Call) and _func_name(a) == "len" for a in ast.walk(lp.iter)) for lp in grid):
                continue
            toks = self._toks_of(n)
            if toks & SPLIT_T or toks & {"walk", "wf", "fold", "folds"}:
                continue
            tgts = [g.target for g in grid] if comp else [lp.target for lp in grid]
            tgts += [g.iter for g in grid] if comp else [lp.iter for lp in grid]
            if any(isinstance(e, ast.Name) and _tokens(e.id) & (DATE_KEY_T | {"attempt", "retry", "retries", "tries", "trial"})
                   for t in tgts for e in ast.walk(t)):
                continue  # a loop over years / dates / retries, not a parameter grid
            # helpers called in the loop (sharpe_for(w), run_backtest(df, w)) count toward the metric
            helper_toks: set[str] = set()
            for c in ast.walk(n):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and self._helper_fn(c.func.id) is not None:
                    helper_toks |= (_tokens(c.func.id) | self._toks_of(self._helper_fn(c.func.id))) & {
                        "sharpe", "sortino", "calmar", "cagr", "walk", "wf", "fold", "folds", "oos"}
            if helper_toks & {"walk", "wf", "fold", "folds", "oos"}:
                continue
            metric = bool((toks | helper_toks) & metric_t)
            metric = metric or any(isinstance(x, ast.BinOp) and isinstance(x.op, ast.Div) and _attr(x.left) == "mean"
                                   and _attr(x.right) == "std" for x in ast.walk(n))
            selection = not comp and bool(toks & {"best", "max", "argmax", "idxmax", "top"})
            stored: set[str] = set()
            if not selection:  # results[(a, b)] = sharpe in the loop, best = max(results, key=...) after it
                stored = {x.value.id for x in ast.walk(n) if isinstance(x, ast.Subscript)
                          and isinstance(x.ctx, ast.Store) and isinstance(x.value, ast.Name)}
                stored |= {x.func.value.id for x in ast.walk(n) if isinstance(x, ast.Call) and _attr(x) == "append"
                           and isinstance(x.func.value, ast.Name)}
                if comp and isinstance(self.parents[n].targets[0], ast.Name):
                    stored.add(self.parents[n].targets[0].id)
                selection = self._picked(stored, n.lineno)
            if not (metric and selection):
                continue
            # results appended in the loop and maximized later also count
            size = 1
            known = True
            for lp in grid:
                ln = self._grid_len(lp.iter)
                if ln is None:
                    known = False
                else:
                    size *= ln
            msg = "This loop scores parameter values on the whole sample and keeps the best."
            if known and size > 1:
                msg += f" It tries about {size} combinations: pass n_trials>={size} to audit_returns."
            else:
                msg += " Count every combination tried and pass it as n_trials to audit_returns."
            self.add("LA11", "medium", n, msg)
            if not comp and stored:
                # v0.6: also where the best one is kept (best = max(results, key=results.get))
                end = getattr(n, "end_lineno", n.lineno)
                for c in {id(c): c for nm in stored for c in getattr(self, "_pick_by_name", {}).get(nm, [])}.values():
                    if c.lineno > end and _func_name(c) not in ("sorted", "sort_values") and self.names_in(c) & stored:
                        self.add("LA11", "medium", c, "The best parameter value is picked here from the full-sample sweep "
                                                      f"on line {n.lineno}.", same_as=n.lineno)
                        break

    # -- LA14: early stopping or model selection on the test set ---------------------------------
    def _test_names(self) -> set[str]:
        """Names that hold test-set data: X_test, test, dtest = lgb.Dataset(test[...]), X_te, ..."""
        if hasattr(self, "_tn"):
            return self._tn

        def is_test(name: str) -> bool:
            t = _tokens(name)
            if t & VAL_SET_T or t & {"train", "tr"}:
                return False
            return bool(t & TEST_SET_T) or name.startswith(("dtest", "test"))

        out = {x.id for x in self._all if isinstance(x, ast.Name) and is_test(x.id)}
        for _ in range(2):
            for st in self._all:
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    v = st.value
                    if isinstance(v, ast.Call) and _func_name(v) in ("Dataset", "DMatrix", "Pool", "TensorDataset", "DataLoader"):
                        if {x.id for x in ast.walk(v) if isinstance(x, ast.Name)} & out:
                            out.add(st.targets[0].id)
        self._tn = out
        return out

    def _refs_test(self, node: ast.AST | None) -> bool:
        if node is None:
            return False
        names = {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}
        strs = {_const_str(x) for x in ast.walk(node) if _const_str(x)}
        return bool(names & self._test_names()) and not any(_tokens(s) & VAL_SET_T for s in names | strs)

    def _assigned_call(self, name: str) -> ast.Call | None:
        for st in self._assigns_of(name):
            if isinstance(st.value, ast.Call):
                return st.value
        return None

    def _early_stopping(self, call: ast.Call) -> bool:
        off = (None, False, 0)
        if any(k.arg in ("early_stopping_rounds", "early_stopping")
               and not (isinstance(k.value, ast.Constant) and k.value.value in off) for k in call.keywords):
            return True
        if any(k.arg == "use_best_model" and isinstance(k.value, ast.Constant) and k.value.value is True for k in call.keywords):
            return True
        stoppers = {"early_stopping", "EarlyStopping", "ModelCheckpoint", "EarlyStoppingCallback"}
        cbs = _kw(call, "callbacks")
        if cbs is not None:
            for x in ast.walk(cbs):
                if isinstance(x, ast.Call) and _func_name(x) in stoppers:
                    return True
                if isinstance(x, ast.Name):
                    c = self._assigned_call(x.id)
                    if c is not None and _func_name(c) in stoppers:
                        return True
        recv = call.func.value if isinstance(call.func, ast.Attribute) else None
        est = recv if isinstance(recv, ast.Call) else (self._assigned_call(recv.id) if isinstance(recv, ast.Name) else None)
        if est is not None:
            if any(k.arg == "early_stopping_rounds" and not (isinstance(k.value, ast.Constant) and k.value.value in off)
                   for k in est.keywords):
                return True
            if (_func_name(est) or "").startswith("CatBoost") and _attr(call) == "fit":
                return True  # CatBoost keeps the best iteration on eval_set by default (use_best_model=True)
        return False

    def rule_test_set_selection(self) -> None:
        for n in self._all:
            if not isinstance(n, ast.Call):
                continue
            evals = [k.value for k in n.keywords if k.arg in ("eval_set", "valid_sets", "evals", "validation_data", "eval_dataset")]
            # xgboost early-stops on the LAST eval_set entry, lightgbm on any valid set: judge the entries one by one
            parts = []
            lgbm = self._is_lightgbm(n)
            for k in n.keywords:
                if k.arg in ("eval_set", "evals") and isinstance(k.value, (ast.List, ast.Tuple)) and len(k.value.elts) > 1 \
                        and all(isinstance(e, (ast.Tuple, ast.List)) for e in k.value.elts):
                    parts.extend(k.value.elts if lgbm else [k.value.elts[-1]])
                elif k.arg == "valid_sets" and isinstance(k.value, (ast.List, ast.Tuple)):
                    parts.extend(k.value.elts)
            if parts and any(self._refs_test(e) for e in parts) and self._early_stopping(n) and not self._defer_holdout(parts):
                self.add("LA14", "high", n,
                         "Early stopping watches the *test* set (xgboost stops on the last `eval_set` entry; lightgbm "
                         "on any `valid_sets` entry), so the number of trees is chosen on the data you then report on. "
                         "Early-stop on a validation fold only, and touch the test set once.")
                continue
            if evals and any(self._refs_test(e) for e in evals) and self._early_stopping(n) and not self._defer_holdout(evals):
                self.add("LA14", "high", n,
                         "Early stopping (or best-model selection) watches the *test* set: the number of trees or epochs "
                         "is chosen to fit the data you then report on. Carve a validation fold out of the training "
                         "period for early stopping and touch the test set once.")
        for loop in self._all:
            if isinstance(loop, ast.For):
                if self.inside(loop, (ast.For,)) is not None:
                    continue
            elif not isinstance(loop, (ast.DictComp, ast.ListComp)):
                continue
            tgt = loop.target if isinstance(loop, ast.For) else loop.generators[0].target
            it = loop.iter if isinstance(loop, ast.For) else loop.generators[0].iter
            if isinstance(it, ast.Call) and _attr(it) == "split":
                continue  # for tr, te in TimeSeriesSplit(...).split(X): folds, each with its own test rows
            if not any(isinstance(c, ast.Call) and self._scores_on_test(c) for c in ast.walk(loop)):
                continue  # cheapest filter first (v0.5 perf; these checks are independent filters)
            if isinstance(loop, ast.For) and self._test_follows_loop_var(loop):
                continue  # the test window is built from the loop variable: walk-forward evaluation
            if self._selects_on_cv(loop):
                continue  # candidates are compared by cross-validation; the test score is only reported
            if self._picks_in_sample(loop):
                continue  # the best candidate is picked on the in-sample part (e.g. a PBO / CSCV computation)
            if not any(isinstance(c, ast.Call) and self._scores_on_test(c) for c in ast.walk(loop)):
                continue
            if any(isinstance(c, ast.Call) and self._scores_on_val(c) for c in ast.walk(loop)):
                continue
            if not self._selects_best(loop):
                continue
            self.add("LA14", "medium", loop,
                     "This loop scores each model, parameter or feature on the *test* set and keeps the best one, so the "
                     "reported test score is selected, not out-of-sample. Select on a validation window, then evaluate "
                     "the winner once on the test set.")

    def _is_lightgbm(self, call: ast.Call) -> bool:
        """The fit belongs to LightGBM: lgb.train(...), an LGBM* estimator, or an lgb.early_stopping callback."""
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id in ("lgb", "lightgbm"):
            return True
        recv = call.func.value if isinstance(call.func, ast.Attribute) else None
        est = recv if isinstance(recv, ast.Call) else (self._assigned_call(recv.id) if isinstance(recv, ast.Name) else None)
        if est is not None and (_func_name(est) or "").startswith("LGBM"):
            return True
        cbs = _kw(call, "callbacks")
        return cbs is not None and any(isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)
                                       and x.value.id in ("lgb", "lightgbm") for x in ast.walk(cbs))

    def _defer_holdout(self, evals: list) -> bool:
        """Early stopping watches a set named holdout/oos while a distinct *test* set is scored separately: the
        holdout is acting as the validation set."""
        names = set().union(*(self.names_in(e) for e in evals)) & self._test_names()
        if any(_tokens(nm) & {"test", "te", "tst"} or nm.startswith(("dtest", "test")) for nm in names):
            return False
        if not hasattr(self, "_scored_test"):
            scored: set[str] = set()
            for c in self._all:
                if not isinstance(c, ast.Call):
                    continue
                fn = _attr(c) or _func_name(c) or ""
                is_metric = fn in ("score", "evaluate") or fn in self._SCORE_FUNCS or bool(
                    re.search(r"(_score|_error|_loss)$", fn))
                if is_metric:
                    scored |= {nm for nm in self.names_in(c) if _tokens(nm) & {"test", "te", "tst"}}
            self._scored_test = scored
        other = self._scored_test - names
        # the separate test set must be scored with its own labels (score(X_test, y_test)); a bare predict on a
        # live/unlabeled set doesn't make the holdout a validation set
        return any(o.startswith("y") or _tokens(o) & {"y", "target", "label", "labels"} for o in other)

    def _picks_in_sample(self, loop: ast.AST) -> bool:
        """argmax(is_sr) inside the loop, where is_sr is a container of in-sample scores filled in the loop from data
        that is not test-named. Scalar clamps (max(1, k)) and statistics (max(y_train.abs())) don't count."""
        is_t = {"is", "train", "tr", "insample", "dev"}
        tn = self._test_names()
        for c in ast.walk(loop):
            if not (isinstance(c, ast.Call) and _func_name(c) in ("argmax", "idxmax", "max", "nlargest", "argmin", "idxmin")
                    and len(c.args) == 1 and isinstance(c.args[0], ast.Name)):
                continue
            name = c.args[0].id
            if not _tokens(name) & is_t:
                continue
            fills = [st for st in ast.walk(loop) if isinstance(st, ast.Assign)
                     and any(isinstance(e, ast.Name) and e.id == name for t in st.targets for e in ast.walk(t))]
            if fills and not any(self.names_in(st.value) & tn for st in fills):
                return True
        return False

    def _test_follows_loop_var(self, loop: ast.For) -> bool:
        """The rows of a test-named variable are selected with the loop variable (directly or through index
        arithmetic): X_test = X[start + 504: start + 567]; te = slice(start, start + 63); X_test = X[te]. A test set
        that is only transformed with the loop variable (X_test.rolling(window)) is not a new window."""
        def index_like(v: ast.AST) -> bool:  # arithmetic, slice(), range(), int(), min(), max() of names/numbers
            return all(isinstance(x, (ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Load, ast.operator,
                                      ast.unaryop, ast.Tuple, ast.Slice))
                       or (isinstance(x, ast.Call) and _func_name(x) in ("slice", "range", "int", "min", "max", "len"))
                       for x in ast.walk(v))
        derived = {e.id for e in ast.walk(loop.target) if isinstance(e, ast.Name)}
        stmts = sorted((st for st in ast.walk(loop) if isinstance(st, ast.Assign)), key=lambda x: (x.lineno, x.col_offset))
        for _ in range(2):
            for st in stmts:
                if index_like(st.value) and self.names_in(st.value) & derived:
                    for t in st.targets:
                        derived |= {e.id for e in ast.walk(t) if isinstance(e, ast.Name)}
        tn = self._test_names()
        for st in stmts:
            if not any(isinstance(e, ast.Name) and e.id in tn for t in st.targets for e in ast.walk(t)):
                continue
            for sub in ast.walk(st.value):
                if isinstance(sub, ast.Subscript) and self.names_in(sub.slice) & derived:
                    return True
        return False

    def _selects_on_cv(self, loop: ast.AST) -> bool:
        """The loop stores a cross-validation result (cv = cross_val_score(...)); results[k] = (cv, test_score))."""
        cvs = {"cross_val_score", "cross_validate", "cross_val_predict"}
        cv_names = set()
        for st in ast.walk(loop):
            if isinstance(st, ast.Assign) and any(isinstance(c, ast.Call) and _func_name(c) in cvs for c in ast.walk(st.value)):
                cv_names |= {e.id for t in st.targets for e in ast.walk(t) if isinstance(e, ast.Name)}
        if not cv_names:
            return any(isinstance(c, ast.Call) and _func_name(c) in ("GridSearchCV", "RandomizedSearchCV") for c in ast.walk(loop))
        for st in ast.walk(loop):  # the stored value carries the CV score
            if isinstance(st, ast.Assign) and any(isinstance(t, ast.Subscript) for t in st.targets) and self.names_in(st.value) & cv_names:
                return True
            if isinstance(st, ast.Call) and _attr(st) == "append" and st.args and self.names_in(st.args[0]) & cv_names:
                return True
        return False

    _SCORE_FUNCS = {"score", "evaluate", "accuracy_score", "r2_score", "f1_score", "roc_auc_score", "mean_squared_error",
                    "mean_absolute_error", "log_loss", "precision_score", "recall_score", "corr", "spearmanr",
                    "pearsonr", "balanced_accuracy_score", "sharpe", "information_coefficient"}

    def _stmt_of(self, n: ast.AST) -> ast.AST:
        p = n
        while p is not None and not isinstance(p, ast.stmt):
            p = self.parents.get(p)
        return p if p is not None else n

    def _scores_on_test(self, c: ast.Call) -> bool:
        fn = _attr(c) or _func_name(c)
        if fn in ("predict", "predict_proba"):
            return self._refs_test(c) and any(
                isinstance(x, ast.Call) and (_attr(x) or _func_name(x)) in self._SCORE_FUNCS for x in ast.walk(self._stmt_of(c)))
        if fn in self._SCORE_FUNCS or (fn and _tokens(fn) & {"score", "sharpe", "accuracy"}):
            return self._refs_test(c)
        return False

    def _scores_on_val(self, c: ast.Call) -> bool:
        fn = _attr(c) or _func_name(c)
        if not (fn in self._SCORE_FUNCS or fn in ("predict", "predict_proba")):
            return False
        names = {x.id for x in ast.walk(c) if isinstance(x, ast.Name)} | {_const_str(x) for x in ast.walk(c) if _const_str(x)}
        return any(_tokens(s) & VAL_SET_T for s in names if s)

    def _selects_best(self, loop: ast.AST) -> bool:
        names = {x.id for x in ast.walk(loop) if isinstance(x, ast.Name)}
        if any(_tokens(nm) & {"best", "top", "winner"} for nm in names):
            return True
        # only containers filled with test scores matter: picking from is_scores while also recording oos_scores
        # is the textbook correct pattern
        def test_filled(value: ast.AST) -> bool:
            return bool(self.names_in(value) & self._test_names()) or any(
                isinstance(c, ast.Call) and self._scores_on_test(c) for c in ast.walk(value))
        stored: set[str] = set()
        for x in ast.walk(loop):
            if (isinstance(x, ast.Assign) and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                                                   for t in x.targets) and test_filled(x.value)):
                stored |= {t.value.id for t in x.targets if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)}
            if (isinstance(x, ast.Call) and _attr(x) == "append" and isinstance(x.func.value, ast.Name) and x.args
                    and test_filled(x.args[0])):
                stored.add(x.func.value.id)
        p = self.parents.get(loop)
        if isinstance(p, ast.Assign) and len(p.targets) == 1 and isinstance(p.targets[0], ast.Name):
            stored.add(p.targets[0].id)
        return self._picked(stored, loop.lineno - 1)

    # -- LA15: fundamentals aligned by fiscal period end -------------------------------------------
    @staticmethod
    def _period_end_name(name: str | None) -> bool:
        if not name:
            return False
        flat = re.sub(r"[^a-z]", "", name.lower())
        if flat in ("datadate", "calendardate", "reportperiod", "fiscaldateending", "periodenddate", "periodend",
                    "fiscalperiodend", "fiscalquarterend", "quarterend", "quarterenddate", "fiscalyearend", "fyend",
                    "periodending", "perioddate", "fiscaldate", "statementdate", "periodendingdate", "fpe"):
            return True
        t = _tokens(name)
        if t & {"filing", "filed", "file", "release", "released", "announce", "announcement", "available", "availability",
                "published", "publish", "acceptance", "accepted", "knowledge", "rdq", "datekey", "pit", "known", "asof"}:
            return False
        return bool((t & {"period", "fiscal", "quarter", "fy"}) and (t & {"end", "ending", "ended"}))

    _LAG_WORDS = {"Timedelta", "DateOffset", "BDay", "BusinessDay", "MonthEnd", "QuarterEnd", "YearEnd", "MonthBegin",
                  "QuarterBegin", "YearBegin", "offsets", "timedelta", "to_timedelta", "Day", "Week", "tshift", "CDay",
                  "BMonthEnd", "BQuarterEnd", "relativedelta"}

    def _is_lag_term(self, x: ast.AST) -> bool:
        """A positive time offset: pd.Timedelta(days=90), pd.DateOffset(months=3), pd.offsets.QuarterEnd(1),
        BDay(45), to_timedelta('60D'), or a name bound to one. Zero offsets such as MonthEnd(0) only snap a date to
        its own period end, so they are not a lag."""
        if isinstance(x, ast.Name):
            return x.id in self._lag_constants()
        if not (isinstance(x, ast.Call) and (_func_name(x) or "") in self._LAG_WORDS):
            return False
        vals = [a for a in x.args] + [k.value for k in x.keywords if k.arg not in ("unit", "normalize", "weekday")]
        if not vals:
            return _func_name(x) not in ("Timedelta", "timedelta", "to_timedelta", "DateOffset", "relativedelta")
        for v in vals:
            n = _numc(v)
            if n is not None:
                if n > 0:
                    return True
                continue
            sv = _const_str(v)
            if sv is not None:
                m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", sv)
                if m and float(m.group(1)) > 0:
                    return True
                continue
            return True  # a variable amount: assume it is a lag (precision first)
        return False

    @staticmethod
    def _freq_shift(x: ast.AST) -> int:
        """+1 for shift(n > 0, freq=...) / tshift(n > 0), -1 for n <= 0, 0 if not a frequency shift."""
        if not (isinstance(x, ast.Call) and _attr(x) in ("shift", "tshift")):
            return 0
        if _attr(x) == "shift" and _kw(x, "freq") is None and len(x.args) < 2:
            return 0
        amt = _kw(x, "periods") or (x.args[0] if x.args else None)
        n = 1 if amt is None else _numc(amt)
        return 1 if n is None or n > 0 else -1

    def _has_lag_expr(self, node: ast.AST) -> bool:
        """The expression itself moves dates later: <date> + <positive offset>, <date>.add(<offset>), or a positive
        shift(n, freq=...). Keyword arguments (tolerance=..., freq of other calls) are not lags."""
        stack = [node]
        while stack:
            x = stack.pop()
            if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Add) and (self._is_lag_term(x.left) or self._is_lag_term(x.right)):
                return True
            if isinstance(x, ast.Call):
                if _attr(x) == "add" and x.args and self._is_lag_term(x.args[0]):
                    return True
                if self._freq_shift(x) > 0:
                    return True
                stack.append(x.func)
                stack.extend(x.args)  # keyword values are skipped on purpose
                continue
            stack.extend(ast.iter_child_nodes(x))
        return False

    def _lag_constants(self) -> set[str]:
        if not hasattr(self, "_lagc"):
            self._lagc = set()
            for st in self._all:
                if (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                        and isinstance(st.value, ast.Call) and (_func_name(st.value) or "") in self._LAG_WORDS):
                    if self._is_lag_term(st.value):
                        self._lagc.add(st.targets[0].id)
        return self._lagc

    def _lagged_cols(self) -> set[str]:
        """Period-end columns given a reporting lag anywhere: f['period_end'] = f['period_end'] + pd.Timedelta(days=60),
        .add(LAG), assign(period_end=... + Timedelta), ..."""
        out = set()
        for st in self._all:
            if isinstance(st, (ast.Assign, ast.AugAssign)):
                t = st.targets[0] if isinstance(st, ast.Assign) else st.target
                k, _f = _key(t)
                if k and self._period_end_name(k) and (
                        (isinstance(st, ast.AugAssign) and isinstance(st.op, ast.Add) and self._is_lag_term(st.value))
                        or (isinstance(st, ast.Assign) and self._has_lag_expr(st.value))):
                    out.add(k)
            if isinstance(st, ast.Call) and _attr(st) == "assign":
                for kw in st.keywords:
                    if kw.arg and self._period_end_name(kw.arg) and self._has_lag_expr(kw.value):
                        out.add(kw.arg)
        return out

    def _later_ffill(self, call: ast.Call) -> bool:
        """The merge result is forward-filled afterwards (each period's numbers spread over later dates)."""
        def is_ffill(x):
            return isinstance(x, ast.Call) and (_attr(x) in ("ffill", "pad") or _const_str(_kw(x, "method")) in ("ffill", "pad")
                                                or (_attr(x) in ("transform", "apply", "agg") and x.args
                                                    and _const_str(x.args[0]) in ("ffill", "pad")))
        if any(is_ffill(x) for x in [self.parents.get(self.parents.get(call))] if x is not None):
            return True
        _k, _f, stmt = self.enclosing_assign_target(call)
        if not (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)):
            return False
        nm = stmt.targets[0].id
        for st in self._iter_refs({nm}, stmt.lineno):
            if any(is_ffill(x) for x in ast.walk(st)):
                return True
        return False

    def _merge_keys(self, call: ast.Call) -> tuple[list[str], list[str]]:
        def strs(node):
            if node is None:
                return []
            if isinstance(node, (ast.List, ast.Tuple)):
                return [s for s in (_const_str(e) for e in node.elts) if s]
            s = _const_str(node)
            return [s] if s else []
        on = strs(_kw(call, "on"))
        return on + strs(_kw(call, "left_on")), on + strs(_kw(call, "right_on"))

    def rule_pit_fundamentals(self) -> None:
        lagged = self._lagged_cols()
        # frames indexed by a period-end column: f = f.set_index('period_end') / read_csv(index_col='fiscalDateEnding')
        pe_frames: dict[str, tuple[str, int]] = {}
        for st in sorted((x for x in self._all if isinstance(x, ast.Assign)), key=lambda x: x.lineno):
            if len(st.targets) != 1 or not isinstance(st.targets[0], ast.Name):
                continue
            name = st.targets[0].id
            col = None
            for c in ast.walk(st.value):
                if isinstance(c, ast.Call) and _attr(c) == "set_index" and c.args:
                    col = _const_str(c.args[0]) or col
                if isinstance(c, ast.Call) and _func_name(c) in ("read_csv", "read_parquet", "read_excel"):
                    col = _const_str(_kw(c, "index_col")) or col
            if col and self._period_end_name(col) and col not in lagged:
                pe_frames[name] = (col, st.lineno)
            elif name in pe_frames and (col is not None or self._has_lag_expr(st.value)):
                pe_frames.pop(name)  # re-indexed, or shifted by a reporting lag (fund.shift(90, freq='D'))
        for st in self._all:  # fund.index = fund.index + pd.Timedelta(days=90) (tz_localize is not a lag)
            if (isinstance(st, (ast.Assign, ast.AugAssign))
                    and isinstance(st.targets[0] if isinstance(st, ast.Assign) else st.target, ast.Attribute)):
                tg = st.targets[0] if isinstance(st, ast.Assign) else st.target
                if (tg.attr == "index" and isinstance(tg.value, ast.Name) and tg.value.id in pe_frames
                        and (self._has_lag_expr(st.value) if isinstance(st, ast.Assign)
                             else isinstance(st.op, ast.Add) and self._is_lag_term(st.value))):
                    pe_frames.pop(tg.value.id)
        for n in self._all:
            if not isinstance(n, ast.Call):
                continue
            fn = _attr(n) or _func_name(n)
            if fn in ("merge_asof", "merge"):
                left, right = self._merge_keys(n)
                if _kw(n, "on") is not None:
                    continue  # the same column on both sides: two tables keyed by period (fine), or unknown
                pe = [k for k in left + right if self._period_end_name(k) and k not in lagged]
                other = [k for k in (right if any(k in left for k in pe) else left) if not self._period_end_name(k)]
                event = {"public", "announce", "announcement", "announced", "release", "released", "filing", "filed",
                         "report", "reported", "accepted", "acceptance", "earnings", "event", "call", "guidance"}
                if pe and other and any(_tokens(k) & DATE_KEY_T and not _tokens(k) & event for k in other):
                    if self._has_lag_expr(n):
                        continue
                    sev = "high" if fn == "merge_asof" else ("medium" if self._later_ffill(n) else "possible")
                    self.add("LA15", sev, n,
                             f"Fundamentals are matched to price dates by `{pe[0]}`, the fiscal period end. The numbers "
                             "were published weeks later, so the backtest uses them before they existed. Merge on the "
                             "filing/report date, or lag the period end (e.g. +90 days) first.")
                    continue
            if fn in ("reindex", "join", "merge", "asfreq", "resample", "align", "combine_first", "concat", "merge_asof"):
                names = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
                hit = next((f for f in pe_frames if f in names and pe_frames[f][1] < n.lineno), None)
                if hit is None or any(_chain_shift(a) > 0 for a in [n.func.value] if isinstance(n.func, ast.Attribute)):
                    continue
                if self._has_lag_expr(n):
                    continue  # fund['eps'].shift(1, freq='QE') inside the join
                if fn in ("concat",) and not (isinstance(_kw(n, "axis"), ast.Constant) and _kw(n, "axis").value in (1, "columns")):
                    continue
                self.add("LA15", "medium", n,
                         f"`{hit}` is indexed by `{pe_frames[hit][0]}`, the fiscal period end, and `.{fn}()` lines it up "
                         "with price dates, so each figure appears on the period-end date instead of when it was "
                         "published. Index by the filing date, or lag the period end first.")
                pe_frames.pop(hit)

    # -- LA13: x[i+1] in a condition --------------------------------------------------------------
    def rule_future_index(self) -> None:
        for loop in self._all:
            var = None
            if isinstance(loop, ast.For):
                if isinstance(loop.target, ast.Name):
                    var = loop.target.id
                elif isinstance(loop.target, ast.Tuple) and loop.target.elts and isinstance(loop.target.elts[0], ast.Name):
                    var = loop.target.elts[0].id
            elif isinstance(loop, ast.While):
                incs = [x.target.id for x in ast.walk(loop) if isinstance(x, ast.AugAssign)
                        and isinstance(x.op, ast.Add) and isinstance(x.target, ast.Name)]
                var = incs[0] if incs else None
            if var is None:
                continue
            scope = self._scope_of(loop)
            label_only = None
            possible_done = False
            for stmt in ast.walk(loop):
                test = getattr(stmt, "test", None) if isinstance(stmt, (ast.If, ast.While, ast.IfExp)) else None
                if test is None:
                    continue
                subs = []
                for sub in ast.walk(test):
                    if isinstance(sub, ast.Subscript):
                        offs = [self._offset_of(c, var, scope) for c in self._index_parts(sub)]
                        k = max((o for o in offs if isinstance(o, int)), default=0)
                        if k > 0 and not self._str_base(sub.value, scope):
                            subs.append((sub, k))
                if not subs:
                    continue
                if label_only is None:
                    label_only = self._label_only_loop(loop)
                if label_only:
                    break  # v0.4: a label builder (the later bars only reach a label or target), not a decision
                sim = self._order_sim(stmt, loop, var, scope) if isinstance(stmt, ast.If) else None
                self._report_sim_decision(var)
                if sim == "exempt":
                    continue  # v0.5: a resting order placed with past data, filled if bar i+1 reaches its level
                if sim == "possible":
                    self.add("LA13", "possible", stmt.test,
                             f"This looks like a resting order placed at bar `{var}` and filled if bar `{var}+1` reaches "
                             "its level, but the fill isn't priced from that level. Price a limit or stop fill at its "
                             "level (or the open, if the bar gaps through it).")
                    continue
                bad = [(s, k) for s, k in subs if not self._in_validity_check(s, test)]
                sub, k = (bad or subs)[0]
                if bad:
                    self.add("LA13", "high", sub, f"The condition at bar `{var}` reads bar `{var}+{k}`.")
                    break
                if not possible_done:  # keep looking: a later condition may use the value (rt10 F3)
                    possible_done = True
                    self.add("LA13", "possible", sub,
                             f"The condition at bar `{var}` checks bar `{var}+{k}` for missing data. That is fine for a "
                             "data-quality report; in a backtest it can skip trades before a delisting or a data gap.")

    # -- LA13 (v0.5): loop order simulation ----------------------------------------------------------------------
    _ORDER_EXEC_T = ENTRY_T | EXEC_T | {"exit", "exits", "sell", "sold", "cover", "stop", "tp", "sl", "target_px"}
    _ORDER_BOOK_T = {"trades", "trade", "rets", "returns", "ret", "pnl", "pnls", "out", "results", "profits", "gains",
                     "res", "result", "log", "logs", "fills", "records", "ledger", "book"}

    def _bar_base_key(self, sub: ast.Subscript) -> str | None:
        """The series read by x[i + k], x.iloc[i + k], df["low"].iloc[i + k], df.low.values[i + k]."""
        b = sub.value
        while isinstance(b, ast.Attribute) and b.attr in ("iloc", "loc", "iat", "at", "values"):
            b = b.value
        if isinstance(b, ast.Call) and _attr(b) in ("to_numpy", "tolist") and isinstance(b.func, ast.Attribute):
            b = b.func.value
        return _key(b)[0]

    def _pricey_key(self, k: str | None) -> bool:
        """A price series name, including short array names (o, h, lo, hi, c)."""
        if not k:
            return False
        return _is_price_name(k) or _is_open_name(k) or bool(_tokens(k) & (self._EXTREME_T | {"h", "l"}))

    def _chain_offset(self, c: ast.AST | None, var: str, scope) -> int | str | None:
        """Like _offset_of, also for chains `var + 1 + cfg.hold_bars` / `i + 1 + HOLD`: the sum of the terms after
        `var` ('sym' when a positive term can't be resolved)."""
        terms: list[tuple[int, ast.AST]] = []
        x = c
        while isinstance(x, ast.BinOp) and isinstance(x.op, (ast.Add, ast.Sub)):
            terms.append((1 if isinstance(x.op, ast.Add) else -1, x.right))
            x = x.left
        if not (isinstance(x, ast.Name) and x.id == var):
            return None
        total, sym = 0, False
        for sign, t in terms:
            v = _numc(t)
            if v is None and isinstance(t, ast.Name):
                v = self._param_default(t.id, scope)
            if v is None:
                if sign > 0 and not (isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.USub)):
                    sym = True
                continue
            total += sign * int(v)
        return "sym" if sym and total >= 0 else total

    def _sub_offset(self, sub: ast.Subscript, var: str, scope) -> int | str:
        offs = [self._chain_offset(c, var, scope) for c in self._index_parts(sub)]
        if "sym" in offs:
            return "sym"
        return max((o for o in offs if isinstance(o, int)), default=0)

    def _row_aliases(self, var: str) -> set[str]:
        """Names bound to bar var+1's whole row inside a loop over var: nxt = bars.iloc[t + 1] (also .loc, x[t + 1])."""
        if not hasattr(self, "_aliases"):
            self._aliases: dict[str, set[str]] = {}
            for st in self._all:
                if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                        and isinstance(st.value, ast.Subscript)):
                    continue
                sub = st.value
                sl = sub.slice
                if isinstance(sl, (ast.Tuple, ast.Slice)) or _const_str(sl) is not None:
                    continue
                if not (isinstance(sl, ast.BinOp) and isinstance(sl.op, ast.Add) and isinstance(sl.left, ast.Name)
                        and _num(sl.right) == 1):
                    continue
                if not isinstance(sub.value, (ast.Attribute, ast.Name)) or (
                        isinstance(sub.value, ast.Attribute) and sub.value.attr not in ("iloc", "loc")):
                    continue
                if isinstance(sub.value, ast.Name) and self._pricey_key(sub.value.id):
                    continue  # entry = o[i + 1] is one price, not a row (rows[k + 1] of itertuples is a row)
                if self.inside(st, (ast.For, ast.While)) is not None:
                    self._aliases.setdefault(sl.left.id, set()).add(st.targets[0].id)
        return self._aliases.get(var, set())

    def _alias_col(self, x: ast.AST, var: str) -> str | None:
        """The column of a read of a next-bar row alias: nxt["high"], nxt.high (None if `x` isn't one)."""
        aliases = self._row_aliases(var)
        if not aliases:
            return None
        if isinstance(x, ast.Subscript) and isinstance(x.value, ast.Name) and x.value.id in aliases:
            return _const_str(x.slice)
        if (isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id in aliases
                and x.attr not in PANDAS_METHODS):
            return x.attr
        return None

    def _next_read(self, x: ast.AST, var: str, scope) -> tuple[int | str, str | None] | None:
        """(offset, series or column name) if `x` reads a later bar: x[var + k] or a next-bar row alias column."""
        col = self._alias_col(x, var)
        if col is not None:
            return 1, col
        if isinstance(x, ast.Subscript):
            off = self._sub_offset(x, var, scope)
            if (off == "sym" or (isinstance(off, int) and off > 0)) and not self._str_base(x.value, scope):
                return off, self._bar_base_key(x)
        return None

    def _has_future(self, node: ast.AST, var: str, scope, allow_open1: bool = False) -> bool:
        """`node` reads a later bar (x[var + k], k > 0 or unknown, or a next-bar row alias); with allow_open1, bar
        var+1's open is allowed."""
        for x in ast.walk(node):
            r = self._next_read(x, var, scope) if isinstance(x, (ast.Subscript, ast.Attribute)) else None
            if r is None:
                continue
            if allow_open1 and r[0] == 1 and _is_open_name(r[1]):
                continue
            return True
        return False

    def _fill_terms(self, test: ast.AST, var: str, scope):
        """(levels, past-only terms that read bar data, wrong-direction flag) if `test` is `<past terms> and <bar
        var+1's low/high> <op> <level>`, the level known before that bar's high and low (past data, or its open);
        None otherwise. A fill is the low at or below the level, or the high at or above it; the other direction
        (`nxt["low"] > level`) means the order did not fill, and is flagged so the caller can tell a skip from an
        entry decision."""
        terms = list(test.values) if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else [test]
        levels, past, rev, lows = [], 0, False, []
        is_low = False
        bad_names = getattr(self, "_taint_nonopen", set())  # level names tainted by a later bar's non-open data
        for t in terms:
            if any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in self._funcs
                   for x in ast.walk(t)):
                return None  # rt12 R09: a helper defined in the file (good(i)) may read later bars: can't verify
            if not self._has_future(t, var, scope) and not (self.names_in(t) & bad_names):
                # a past-only term counts as the order decision only if it reads bar data (not `not in_pos`)
                past += any(isinstance(x, ast.Subscript) for x in ast.walk(t))
                continue
            if not (isinstance(t, ast.Compare) and len(t.ops) == 1
                    and isinstance(t.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE))):
                return None
            got = None
            for side, (x, lvl) in enumerate(((t.left, t.comparators[0]), (t.comparators[0], t.left))):
                r = self._next_read(x, var, scope) if isinstance(x, (ast.Subscript, ast.Attribute)) else None
                if (r is not None and r[0] == 1 and not self._has_future(lvl, var, scope, allow_open1=True)
                        and not (self.names_in(lvl) & bad_names)):
                    kt = _tokens(r[1]) if r[1] else frozenset()
                    if kt & (self._EXTREME_T | {"h", "l"}) and not kt & (RETURN_T | OPEN_T | CLOSE_T | _TIME_WORDS):
                        below = isinstance(t.ops[0], (ast.Lt, ast.LtE)) == (side == 0)  # x below the level
                        is_low = bool(kt & {"low", "lows", "lo", "l"})
                        rev = rev or (below != is_low)
                        got = lvl
                        break
            if got is None:
                return None
            levels.append(got)
            lows.append(is_low)
        return (levels, past, rev, lows) if levels else None

    def _index_source_forward(self, it: ast.AST, loop: ast.AST, depth: int = 0, seen: set | None = None) -> bool:
        """Anything forward-looking feeds the loop's index source: followed through names, function parameters (to
        their call-site arguments), np.flatnonzero / np.where(...)[0] / .nonzero() / .index / comprehensions."""
        seen = set() if seen is None else seen
        if depth > 6:
            return True  # too deep to tell: no exemption
        if self._ahead(it) or self._slice_lead(it):
            return True
        for sub in ast.walk(it):
            if isinstance(sub, (ast.BinOp, ast.Subscript, ast.Compare, ast.Call)):
                a = self._aoff(sub, {})
                if a is not None and a[1] and a[0] > 0:
                    return True  # c[5:] / c[:-5]: element j reads bar j + 5
        for x in ast.walk(it):  # rt13 X02: s.idx / self.idx: the values assigned to that attribute anywhere
            if not (isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)) or x.value.id in MODULE_ALIASES - {
                    "self"} or x.attr in PANDAS_METHODS or x.attr in ("values", "index", "columns"):
                continue
            if isinstance(self.parents.get(x), ast.Call) and self.parents[x].func is x:
                continue  # a method call: judged by its receiver
            ak = ("attr", x.attr)
            if ak in seen:
                continue
            seen.add(ak)
            vals = [a.value for a in self._all if isinstance(a, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr == x.attr for t in a.targets)]
            if not vals:
                return True  # unresolvable attribute: no exemption
            if any(self._index_source_forward(v, self._stmt_of(v), depth + 1, seen) for v in vals):
                return True
        fd = self._scope_of(loop)
        params = {p.arg: k for k, p in enumerate((fd.args.posonlyargs + fd.args.args) if fd else [])}
        for x in ast.walk(it):
            if not isinstance(x, ast.Name) or x.id in MODULE_ALIASES:
                continue
            pk = (x.id, id(fd)) if fd is not None and x.id in params else x.id  # a parameter isn't the global name
            if pk in seen:
                continue
            seen.add(pk)
            if fd is not None and x.id in params:
                if not hasattr(self, "_calls_by_name"):
                    self._calls_by_name = {}
                    for c in self._all:
                        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name):
                            self._calls_by_name.setdefault(c.func.id, []).append(c)
                for c in self._calls_by_name.get(fd.name, []):
                    arg = self._call_arg(c, params[x.id], x.id)
                    if arg is not None and self._index_source_forward(arg, c, depth + 1, seen):
                        return True
                continue
            if not hasattr(self, "_name_assigns"):
                self._str_name("\0", None)  # builds the name -> assignments index used by _str_name
            for a in self._name_assigns.get(x.id, []):
                if a.value is not None and a is not self._stmt_of(it) and self._index_source_forward(
                        a.value, loop, depth + 1, seen):
                    return True
        return False

    def _index_loop(self, loop: ast.AST) -> bool:
        """for i in signal_idx: a loop over bars chosen beforehand (not range/enumerate/zip/arange/iterrows)."""
        if not isinstance(loop, ast.For):
            return False
        it = loop.iter
        if isinstance(it, ast.Call) and (_func_name(it) in ("range", "enumerate", "zip", "arange", "iterrows",
                                                           "itertuples", "items", "iteritems") or _np_call(it, "arange")):
            return False
        return bool(_scan_expr(it).refs)

    def _signal_index_loop(self, loop: ast.AST) -> bool:
        """An index loop where nothing forward-looking feeds the indices."""
        return self._index_loop(loop) and not self._index_source_forward(loop.iter, loop)

    @staticmethod
    def _slice_lead(node: ast.AST) -> bool:
        """c[5:] / c[:-5] - 1: two positional slices of the same array that start at different bars, so element j
        reads a later bar than j (the same expression without np.r_ padding is how forward returns are written)."""
        for b in ast.walk(node):
            if not isinstance(b, (ast.BinOp, ast.Compare)):
                continue
            sides = [b.left, b.right] if isinstance(b, ast.BinOp) else [b.left, *b.comparators]
            starts: dict[str, set[int]] = {}
            for s in sides:
                for x in ast.walk(s):
                    if isinstance(x, ast.Subscript) and isinstance(x.value, ast.Name) and isinstance(x.slice, ast.Slice):
                        lo = 0 if x.slice.lower is None else _num(x.slice.lower)
                        if lo is not None and lo >= 0:
                            starts.setdefault(x.value.id, set()).add(int(lo))
            if any(len(v) > 1 for v in starts.values()):
                return True
        return False

    _SINK_CALLS = {"append", "extend", "insert", "print", "write", "writerow", "info", "debug", "warning", "log",
                   "save", "savez", "savetxt", "to_csv", "to_parquet", "to_json", "dump"}

    def _flows_to_book_only(self, name: str, st: ast.AST, depth: int = 0) -> bool:
        """A fill or exit value assigned in a fill branch reaches only a trade log or return (append/extend, a
        return value, print/log/file), never a condition, a comparison or a decision variable (rt12 R03, R04)."""
        if depth > 3:
            return False
        if not hasattr(self, "_name_loads"):
            self._name_loads = {}
            for x in self._all:
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                    self._name_loads.setdefault(x.id, []).append(x)
        scope = self._scope_of(st)
        for x in self._name_loads.get(name, []):
            if self._scope_of(x) is not scope:
                continue
            c, p, ok = x, self.parents.get(x), False
            while p is not None and not isinstance(p, ast.stmt):
                if isinstance(p, (ast.Compare, ast.BoolOp)) or (isinstance(p, ast.IfExp) and p.test is c):
                    return False
                if isinstance(p, ast.Call) and (_attr(p) or _func_name(p)) in self._SINK_CALLS and p.func is not c:
                    ok = True
                    if _attr(p) in ("append", "extend", "insert") and isinstance(p.func.value, ast.Name):
                        getattr(self, "_flow_containers", set()).add(p.func.value.id)  # rt13: follow the list too
                    break
                c, p = p, self.parents.get(p)
            if ok:
                continue
            if isinstance(p, ast.Return):
                continue
            if isinstance(p, (ast.If, ast.While)) and any(y is x for y in ast.walk(p.test)):
                return False
            if isinstance(p, (ast.Assign, ast.AugAssign)):
                tg = p.targets if isinstance(p, ast.Assign) else [p.target]
                for t in tg:
                    tk = _key(t)[0]
                    if tk == name:
                        continue
                    if not (isinstance(t, ast.Name) and tk and (classify(tk) in ("exec", "pnl", "return") or _tokens(
                            tk) & (self._ORDER_EXEC_T | self._ORDER_BOOK_T))):
                        return False
                    if not self._flows_to_book_only(tk, st, depth + 1):
                        return False
                continue
            return False
        return True

    def _report_sim_decision(self, var: str) -> None:
        """The order-simulation exemption was lost because a list holding fill/exit values is read in a decision:
        report that decision line (rt13)."""
        node = getattr(self, "_sim_decision", None)
        self._sim_decision = None
        if node is None:
            return
        at = getattr(node, "test", None) or getattr(node, "iter", None) or node
        self.add("LA13", "high", at,
                 f"This reads a list that holds fill or exit prices from bars after bar `{var}` (simulated trade returns) "
                 "to make a decision. Those returns aren't known until the trades close; book them only into the "
                 "results (a log, a DataFrame, summary statistics printed after the run).")

    def _inside_minmax(self, node: ast.AST, top: ast.AST) -> bool:
        p = self.parents.get(node)
        while p is not None and p is not self.parents.get(top):
            if isinstance(p, ast.Call) and _func_name(p) in ("min", "max", "minimum", "maximum", "fmin", "fmax", "clip"):
                return True
            p = self.parents.get(p)
        return False

    def _loop_assigns(self, loop: ast.AST, name: str, before: ast.AST | None = None) -> list[ast.AST]:
        """Values assigned to `name` in the loop; with `before`, only assignments that dominate it (earlier in a block
        that contains it, not in a sibling branch or an earlier iteration's other branch)."""
        out = []
        for a in ast.walk(loop):
            if not (isinstance(a, ast.Assign) and len(a.targets) == 1 and isinstance(a.targets[0], ast.Name)
                    and a.targets[0].id == name):
                continue
            if before is not None:
                par = self.parents.get(a)
                block = next((getattr(par, f) for f in ("body", "orelse", "finalbody")
                              if any(s is a for s in getattr(par, f, []) or [])), None)
                if block is None:
                    continue
                k = next(j for j, s in enumerate(block) if s is a)
                if not any(s is before or any(y is before for y in ast.walk(s)) for s in block[k + 1:]):
                    continue
            out.append(a.value)
        return out

    def _open_derived(self, lv: ast.AST, var: str, scope, loop: ast.AST, st: ast.AST, depth: int = 0) -> bool:
        """The level reads the fill bar's open (bar var+1), directly or through names assigned on the way to the fill
        check (tp = entry * 1.02 with entry = o[i + 1], both earlier in the same branch)."""
        for x in ast.walk(lv):
            if isinstance(x, (ast.Subscript, ast.Attribute)):
                r = self._next_read(x, var, scope)
                if r is not None and r[0] == 1 and _is_open_name(r[1]):
                    return True
        if depth >= 2:
            return False
        return any(self._open_derived(v, var, scope, loop, st, depth + 1)
                   for x in ast.walk(lv) if isinstance(x, ast.Name) for v in self._loop_assigns(loop, x.id, st))

    def _far_side(self, lv: ast.AST, is_low: bool, loop: ast.AST, depth: int = 0, st: ast.AST | None = None) -> bool:
        """A buy level below the open it was set from (`o * 0.99`, `o - 0.5`), a sell level above it (`o * 1.01`)."""
        e = lv
        if isinstance(e, ast.Name):
            vals = self._loop_assigns(loop, e.id, st)
            if len(vals) != 1 or depth > 2:
                return False
            return self._far_side(vals[0], is_low, loop, depth + 1, st)
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mult):
            k = _numc(e.right) if _numc(e.right) is not None else _numc(e.left)
            return k is not None and ((k < 1) if is_low else (k > 1))
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Sub)):
            k = _numc(e.right)
            if k is None or k <= 0:
                return False
            return isinstance(e.op, ast.Sub) if is_low else isinstance(e.op, ast.Add)
        return False

    _CONTAINER_SINKS = {"mean", "nanmean", "sum", "nansum", "len", "std", "nanstd", "median", "nanmedian", "print",
                        "DataFrame", "Series", "array", "asarray", "describe", "to_csv", "to_parquet", "to_json", "save",
                        "savez", "savetxt", "dump", "write", "writerow", "info", "debug", "warning", "log", "round",
                        "count", "plot", "hist", "prod", "var", "cumprod", "cumsum", "concat", "from_records"}

    def _count_only(self, x: ast.Name) -> bool:
        """`x` is read only for its size: the single argument of `len(...)`, or a truthiness test (`if x`, `not x`, an
        operand of and/or whose result is itself only a test)."""
        p = self.parents.get(x)
        if isinstance(p, ast.Call) and _func_name(p) == "len" and len(p.args) == 1 and p.args[0] is x and not p.keywords:
            return True
        node = x
        while True:
            p = self.parents.get(node)
            if isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.Not):
                return True  # `not x` is a bool whatever surrounds it
            if isinstance(p, ast.BoolOp):
                node = p
                continue
            return isinstance(p, (ast.If, ast.While, ast.IfExp)) and p.test is node

    def _container_sink_only(self, name: str, st: ast.AST, depth: int = 0) -> bool:
        """A list or dict that received exempt fill/exit values reaches only summaries and sinks (mean, len, sum,
        print, a DataFrame/Series, a return value, a file); never a condition, a comprehension filter, a loop it
        drives, or a decision input (rt13 X05, X07, X08). On failure, the decision node is kept for the report."""
        if depth > 3:
            return False
        if not hasattr(self, "_name_loads"):
            self._name_loads = {}
            for x in self._all:
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                    self._name_loads.setdefault(x.id, []).append(x)
        scope = self._scope_of(st)
        for x in self._name_loads.get(name, []):
            if self._scope_of(x) is not scope:
                continue
            p = self.parents.get(x)
            if (isinstance(p, ast.Attribute) and p.value is x and p.attr in self._WRITE_CALLS
                    and isinstance(self.parents.get(p), ast.Call) and self.parents[p].func is p):
                continue  # trades.append(...): a write into the container
            c, a, sunk = x, p, False
            if (isinstance(p, ast.Attribute) and p.value is x and p.attr in self._CONTAINER_SINKS
                    and isinstance(self.parents.get(p), ast.Call) and self.parents[p].func is p):
                sunk = True  # res.to_csv(...), res.describe()
            if self._count_only(x):
                # v0.6 (Y04): `len(trades) >= 50`, `if trades`, `not trades`: how many fills happened so far, which
                # is known in real time; the fill and exit values themselves are never read here.
                continue
            while a is not None and not isinstance(a, ast.stmt):
                if isinstance(a, (ast.Compare, ast.BoolOp)) or (isinstance(a, ast.IfExp) and a.test is c) or (
                        isinstance(a, ast.comprehension) and a.ifs):
                    self._sim_decision = self._stmt_of(x)
                    return False
                if isinstance(a, ast.Call) and (_attr(a) or _func_name(a)) in self._CONTAINER_SINKS and a.func is not c:
                    sunk = True
                c, a = a, self.parents.get(a)
            if isinstance(a, (ast.If, ast.While)) and any(y is x for y in ast.walk(a.test)) or (
                    isinstance(a, ast.For) and any(y is x for y in ast.walk(a.iter))):
                self._sim_decision = a
                return False
            if isinstance(a, ast.Return) or (isinstance(a, ast.Expr) and sunk):
                continue
            if isinstance(a, ast.Assign):
                for t in a.targets:
                    b = t
                    while isinstance(b, (ast.Subscript, ast.Attribute)):
                        b = b.value
                    if isinstance(b, ast.Name) and b.id != name and not self._container_sink_only(b.id, st, depth + 1):
                        return False
                continue
            self._sim_decision = a
            return False
        return True

    def _order_sim(self, st: ast.If, loop: ast.AST, var: str, scope) -> str | None:
        """'exempt' when `st` checks whether a resting order, placed at bar var from past data, fills during bar var+1
        (the next bar's low reaches a buy level, or its high a sell or stop level), the fill or exit is priced from
        that level, and fill/exit values reach only the trade's log or return. 'possible' when an explicitly decided
        order's fill isn't priced from the level. None otherwise (judged as a normal condition)."""
        ft = self._fill_terms(st.test, var, scope)
        if ft is None:
            return None
        levels, past, rev, lows = ft
        self._sim_decision = None
        if rev:
            # `if lo[i + 1] > limit: continue`: skipping an order that didn't fill is fine; booking on it is a decision
            return "exempt" if all(isinstance(b, (ast.Continue, ast.Break, ast.Pass)) for b in st.body) else None
        decided, child, p = past > 0, st, self.parents.get(st)
        while not decided and p is not None and p is not loop:
            if isinstance(p, ast.If):
                if any(child is x for x in p.orelse) and self._fill_terms(p.test, var, scope) is not None:
                    pass  # elif of another fill check (take-profit, then stop): the same order
                elif any(child is x for x in p.body) and not self._has_future(p.test, var, scope):
                    decided = True
                    break
                else:
                    return None
            elif isinstance(p, (ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef)):
                return None
            child, p = p, self.parents.get(p)
        if not decided and p is loop and self._index_loop(loop):
            # for i in signal_idx: the order was decided when the indices were chosen. Only if nothing forward-looking
            # feeds them (rt12 R01, R02); otherwise no exemption at all
            if not self._signal_index_loop(loop):
                return None
            decided = True
        lvl_dumps = {ast.dump(lv) for lv in levels}
        lvl_names = {x.id for lv in levels for x in ast.walk(lv) if isinstance(x, ast.Name)  # scalar level names only
                     and not isinstance(self.parents.get(x), (ast.Subscript, ast.Attribute)) and x.id != var}
        # rt13 (R10): a level set from bar var+1's open must sit on the far side of that open (below it for a buy, above
        # it for a sell) and be booked at the level itself, not min/max(level, open) decided before the open
        open_lvl = False
        for lv, is_low in zip(levels, lows):
            if self._open_derived(lv, var, scope, loop, st):
                if not self._far_side(lv, is_low, loop, st=st):
                    return None
                open_lvl = True
        priced, assigned = False, []
        containers: set[str] = set()
        for x in (n for b in st.body for n in ast.walk(b)):
            if isinstance(x, (ast.Assign, ast.AugAssign)):
                tgts = x.targets if isinstance(x, ast.Assign) else [x.target]
                tks = [_key(t)[0] for t in tgts]
                ok_t = all(k and (classify(k) in ("exec", "pnl", "return") or _tokens(k) & (
                    self._ORDER_EXEC_T | self._ORDER_BOOK_T)) for k in tks)
                val = x.value
            elif isinstance(x, ast.Call) and _attr(x) in ("append", "extend") and x.args:
                rk = _key(x.func.value)[0]
                ok_t = bool(rk) and (classify(rk) in ("pnl", "return") or bool(_tokens(rk) & self._ORDER_BOOK_T))
                val, tgts = x.args[0], []
                if isinstance(x.func.value, ast.Name):
                    containers.add(x.func.value.id)
            else:
                continue
            fut = self._has_future(val, var, scope)
            if fut and not ok_t:
                return None  # size, weight or position from the fill bar: an input (normal finding)
            for s in (ast.walk(val) if fut else ()):
                if not isinstance(s, (ast.Subscript, ast.Attribute)):
                    continue
                if isinstance(s, ast.Subscript) and isinstance(s.value, ast.Attribute) and s.value.attr == "index":
                    continue  # bars.index[i + 1]: the fill's timestamp in the trade log, not data
                r = self._next_read(s, var, scope)
                if r is None:
                    continue
                if not self._pricey_key(r[1]):
                    return None  # a later bar's non-price value
                kt = _tokens(r[1]) if r[1] else frozenset()
                if kt & (self._EXTREME_T | {"h", "l"}) and not kt & (OPEN_T | CLOSE_T):
                    return None  # rt12 R11: priced at a later bar's low or high itself, not at the order's level
            hits = [s for s in ast.walk(val) if ast.dump(s) in lvl_dumps
                    or (isinstance(s, ast.Name) and s.id in lvl_names)]
            if open_lvl:  # booked at the level itself: not inside min(...)/max(...)
                hits = [s for s in hits if not self._inside_minmax(s, val)]
            if ok_t and hits:
                priced = True
            if fut or self.names_in(val) & set(assigned):  # fill/exit values (busy_until = i + 1 + HOLD is not one)
                assigned += [t.id for t in tgts if isinstance(t, ast.Name)]
        self._flow_containers = containers
        for nm in dict.fromkeys(assigned):  # rt12 R03/R04: fill and exit values may only reach the trade's log
            if not self._flows_to_book_only(nm, st):
                return None
        for cn in sorted(self._flow_containers):  # rt13 X05/X07/X08: and so may the lists they are appended to
            if not self._container_sink_only(cn, st):
                return None
        if open_lvl and not priced:
            return None  # rt13: a level set from the next open must be booked at the level itself
        if priced:
            return "exempt"
        return "possible" if decided else None

    # -- LA13 helpers (v0.4): offsets, string loops, missing-data checks, label-only loops ------------------
    _STR_METHODS = {"join", "strip", "lstrip", "rstrip", "lower", "upper", "replace", "format", "casefold", "title",
                    "capitalize", "zfill", "ljust", "rjust", "center", "removeprefix", "removesuffix", "expandtabs",
                    "swapcase", "translate"}
    _VALIDITY_FUNCS = {"isnan", "isfinite", "isinf", "isnat", "isna", "isnull", "notna", "notnull"}

    @staticmethod
    def _index_parts(sub: ast.Subscript) -> list[ast.AST]:
        """The index expressions of x[...] that say which bar is read: each tuple element, a slice's start."""
        parts = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
        return [p.lower if isinstance(p, ast.Slice) else p for p in parts if not (isinstance(p, ast.Slice) and p.lower is None)]

    def _param_default(self, name: str, fd: ast.AST | None) -> float | None:
        """The numeric default of parameter `name` of `fd`, if the function never rebinds it."""
        if not isinstance(fd, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        if not hasattr(self, "_pdef"):
            self._pdef: dict[tuple[str, int], float | None] = {}
        mk = (name, id(fd))
        if mk not in self._pdef:
            self._pdef[mk] = self._param_default_uncached(name, fd)
        return self._pdef[mk]

    def _param_default_uncached(self, name: str, fd: ast.AST) -> float | None:
        a = fd.args
        pos = a.posonlyargs + a.args
        defs = {p.arg: d for p, d in zip(pos[len(pos) - len(a.defaults):], a.defaults)}
        defs.update({p.arg: d for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None})
        if name not in defs or _num(defs[name]) is None:
            return None
        if not hasattr(self, "_rebound"):  # v0.5 perf: (function, name) pairs rebound in the body, indexed once
            self._rebound: set[tuple[int, str]] = set()
            for x in self._all:
                tg = (x.targets if isinstance(x, ast.Assign) else [x.target] if isinstance(
                    x, (ast.AugAssign, ast.AnnAssign, ast.For)) else [])
                if tg:
                    sc = id(self._scope_of(x))
                    self._rebound.update((sc, y.id) for t in tg for y in ast.walk(t) if isinstance(y, ast.Name))
        if (id(fd), name) in self._rebound:
            return None
        return _num(defs[name])

    def _offset_of(self, c: ast.AST | None, var: str, scope: ast.AST | None) -> int | str | None:
        """Bars past `var` in an index expression: an int when known (`i + 1`, `i + LAG` with a module constant,
        `s + lo` with a parameter default), 'sym' for an unknown positive offset, None if it isn't `var + ...`."""
        if isinstance(c, ast.Name) and c.id == var:
            return 0
        if not (isinstance(c, ast.BinOp) and isinstance(c.op, (ast.Add, ast.Sub)) and isinstance(c.left, ast.Name)
                and c.left.id == var):
            return None
        v = _numc(c.right)
        if v is None and isinstance(c.right, ast.Name):
            v = self._param_default(c.right.id, scope)
        if v is None:
            if isinstance(c.right, ast.UnaryOp) and isinstance(c.right.op, ast.USub):
                return -1
            return "sym" if isinstance(c.op, ast.Add) else -1
        return int(v) if isinstance(c.op, ast.Add) else -int(v)

    def _str_expr(self, e: ast.AST, scope: ast.AST | None, depth: int = 0) -> bool:
        if depth > 4:
            return False
        if isinstance(e, ast.Constant):
            return isinstance(e.value, str)
        if isinstance(e, ast.JoinedStr):
            return True
        if isinstance(e, ast.Call):
            if isinstance(e.func, ast.Name) and e.func.id in ("str", "repr", "chr", "format"):
                return True
            if isinstance(e.func, ast.Attribute) and e.func.attr in self._STR_METHODS:
                return self._str_expr(e.func.value, scope, depth + 1)
            return False
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Mod)):
            return self._str_expr(e.left, scope, depth + 1) or self._str_expr(e.right, scope, depth + 1)
        if isinstance(e, ast.Name):
            return self._str_name(e.id, scope, depth + 1)
        return False

    def _str_name(self, name: str, scope: ast.AST | None, depth: int = 0) -> bool:
        """`name` is visibly a str: a parameter annotated `str`, or every assignment in the scope is a str expression."""
        if not hasattr(self, "_strc"):
            self._strc: dict[tuple[int, str], bool] = {}
            self._name_assigns: dict[str, list[ast.AST]] = {}
            for x in self._all:
                if isinstance(x, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    for t in (x.targets if isinstance(x, ast.Assign) else [x.target]):
                        if isinstance(t, ast.Name):
                            self._name_assigns.setdefault(t.id, []).append(x)
        ck = (id(scope), name)
        if ck in self._strc:
            return self._strc[ck]
        self._strc[ck] = False  # guards cycles
        out = None
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = scope.args
            for p in a.posonlyargs + a.args + a.kwonlyargs:
                if p.arg == name:
                    ann = p.annotation
                    out = (isinstance(ann, ast.Name) and ann.id == "str") or _const_str(ann) == "str"
        if out is None:
            assigns = [x for x in self._name_assigns.get(name, [])
                       if scope is None or self.inside(x, (ast.FunctionDef, ast.AsyncFunctionDef)) is scope]
            out = bool(assigns) and all(x.value is not None and self._str_expr(x.value, scope, depth) for x in assigns)
        self._strc[ck] = out
        return out

    def _str_base(self, base: ast.AST, scope: ast.AST | None) -> bool:
        """t[i + 1] with `t` a string: a character scan, not a bar loop."""
        return isinstance(base, ast.Name) and self._str_name(base.id, scope)

    def _in_validity_check(self, node: ast.AST, test: ast.AST) -> bool:
        """`node` is only checked for missing data: np.isnan(x), pd.isna(x), x.isna(), x is None, np.isfinite(x)..."""
        p, stop = self.parents.get(node), self.parents.get(test)
        while p is not None and p is not stop:
            if isinstance(p, ast.Call) and _func_name(p) in self._VALIDITY_FUNCS:
                return True
            if (isinstance(p, ast.Compare) and len(p.ops) == 1 and isinstance(p.ops[0], (ast.Is, ast.IsNot))
                    and isinstance(p.comparators[0], ast.Constant) and p.comparators[0].value is None):
                return True
            if p is test:
                return False
            p = self.parents.get(p)
        return False

    @staticmethod
    def _key_occ(node: ast.AST, key: str, is_col: bool) -> list[ast.AST]:
        """Reads of `key` under `node`: the name `key`, or (is_col) a column subscript frame["key"] / .loc[:, "key"]."""
        if not is_col:
            return [x for x in ast.walk(node) if isinstance(x, ast.Name) and x.id == key]
        out = []
        for x in ast.walk(node):
            if isinstance(x, ast.Subscript):
                sl = x.slice
                if _const_str(sl) == key or (isinstance(sl, ast.Tuple) and len(sl.elts) == 2 and _const_str(sl.elts[1]) == key):
                    out.append(x)
        return out

    def _only_label_arg(self, value: ast.AST, key: str, is_col: bool = False, occ: list | None = None) -> bool:
        """Every read of `key` in `value` is the label argument of a fit, a training dataset or a score."""
        occ = self._key_occ(value, key, is_col) if occ is None else occ
        if not occ:
            return False
        for x in occ:  # walk up from each read: it must sit inside the y / label= argument of a fit-like call
            c, p, ok = x, self.parents.get(x), False
            while p is not None and not isinstance(p, ast.stmt):
                if isinstance(p, ast.Call) and (_attr(p) in FIT_ATTRS | {"score"} or _func_name(p) in TRAIN_DATA_CTORS):
                    ok = (len(p.args) > 1 and p.args[1] is c) or any(
                        (kw is c or kw.value is c) and kw.arg in ("y", "label", "y_true") for kw in p.keywords)
                    break
                c, p = p, self.parents.get(p)
            if not ok:
                return False
        return True

    _SUMMARY_T = {"value_counts", "describe", "mean", "median", "std", "var", "sum", "count", "nunique", "unique",
                  "min", "max", "quantile", "plot", "hist", "to_csv", "to_parquet", "to_json", "to_pickle", "to_excel",
                  "head", "tail", "info", "agg", "aggregate", "corr", "cumsum", "round", "tofile"}
    _SUMMARY_FUNCS = {"print", "len", "mean", "nanmean", "average", "bincount", "unique", "histogram", "hist", "plot",
                      "display", "Counter", "histplot", "countplot", "barplot", "lineplot", "boxplot", "bar", "scatter",
                      "savetxt", "save", "dump", "info", "debug", "warning",
                      # v0.5: numpy summaries and files (np.unique(..., return_counts=True), np.savez, nan-aware stats)
                      "savez", "savez_compressed", "nansum", "nanmedian", "nanstd", "nanmin", "nanmax", "count_nonzero",
                      "percentile", "nanpercentile", "median", "std", "histogram_bin_edges"}
    _WRITE_CALLS = {"extend", "append", "insert", "update", "add", "put", "setdefault"}

    def _summary_only(self, value: ast.AST, key: str, is_col: bool = False, occ: list | None = None) -> bool:
        """Every read of `key` in `value` ends in a summary or a file (y.value_counts(), y.loc[idx].describe(),
        print(y), y.to_csv(...)), with no product, comparison or model call on the way."""
        occ = self._key_occ(value, key, is_col) if occ is None else occ
        if not occ:
            return False
        for x in occ:
            c, p, ok = x, self.parents.get(x), False
            while p is not None and not isinstance(p, ast.stmt):
                if (isinstance(p, ast.Compare) and len(p.ops) == 1 and all(
                        isinstance(s, ast.Constant) for s in (p.left, p.comparators[0]) if s is not c)):
                    c, p = p, self.parents.get(p)  # v0.5: np.count_nonzero(labels == 1): a count, still a summary
                    continue
                if isinstance(p, (ast.Compare, ast.BoolOp)) or (isinstance(p, ast.BinOp) and isinstance(
                        p.op, (ast.Mult, ast.MatMult, ast.Div))):
                    break
                if isinstance(p, ast.Call) and (_attr(p) in FIT_ATTRS or _func_name(p) in TRAIN_DATA_CTORS
                                                or _attr(p) in self._WRITE_CALLS or _attr(p) in MAPBACK_T):
                    break
                if isinstance(p, ast.Attribute) and p.attr in self._SUMMARY_T and isinstance(
                        self.parents.get(p), ast.Call) and self.parents[p].func is p:
                    ok = True
                    break
                if isinstance(p, ast.Call) and _func_name(p) in self._SUMMARY_FUNCS and p.func is not c:
                    ok = True
                    break
                c, p = p, self.parents.get(p)
            if not ok:
                return False
        return True

    _INDEX_ATTRS = {"index", "shape", "columns", "dtypes", "ndim", "size", "dtype"}

    def _index_only_name(self, x: ast.AST) -> bool:
        """`y.index`, `y.shape`, `len(y)`: the name's alignment or size, not its values (X.loc[y.index])."""
        p = self.parents.get(x)
        if isinstance(p, ast.Attribute) and p.value is x and p.attr in self._INDEX_ATTRS:
            return True
        if isinstance(p, ast.Attribute) and p.value is x and p.attr in self._VALIDITY_FUNCS and isinstance(
                self.parents.get(p), ast.Call):
            return True  # v0.5: y.isna(): which values are missing, not the values
        if isinstance(p, ast.Call) and _func_name(p) in self._VALIDITY_FUNCS and x in p.args:
            return True  # v0.5: valid = ~np.isnan(labels): a missing-value mask
        return isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id == "len" and x in p.args

    _LOADERS = {"load", "loadtxt", "genfromtxt", "fromfile", "read_csv", "read_parquet", "read_pickle", "read_json",
                "read_excel", "read_feather", "read_hdf", "read_table"}

    def _reloaded(self, call: ast.AST) -> bool:
        """A file write whose literal path is read back somewhere in the file (the data comes back as values)."""
        paths = {s for x in ast.walk(call) if (s := _const_str(x)) and ("." in s or "/" in s or "\\" in s)}
        if not paths:
            return False
        if not hasattr(self, "_loaded_paths"):
            self._loaded_paths = {s for c in self._all if isinstance(c, ast.Call) and _func_name(c) in self._LOADERS
                                  for a in c.args[:1] + [k.value for k in c.keywords if k.arg in ("file", "path",
                                                                                                    "filepath_or_buffer")]
                                  if (s := _const_str(a))}
        return bool(paths & self._loaded_paths)

    def _sink_stmts(self, key: str, line: int, is_col: bool = False) -> list[ast.AST]:
        """Assign/AugAssign/Expr statements after `line` that mention `key`: as a name read in the value or the
        targets, or (is_col) as a string constant. Each index is built once, the string one only when needed."""
        refidx = self._ref_index()
        if not is_col:
            if not hasattr(self, "_tgtidx"):
                self._tgtidx: dict[str, list[int]] = {}
                for i, st in enumerate(self._ref_stmts):
                    for t in (st.targets if isinstance(st, ast.Assign) else [st.target] if isinstance(st, ast.AugAssign) else []):
                        if not isinstance(t, ast.Name):  # a plain rebinding target carries no read
                            for x in ast.walk(t):
                                if isinstance(x, ast.Name):
                                    self._tgtidx.setdefault(x.id, []).append(i)
            pos = sorted(set(refidx.get(key, [])) | set(self._tgtidx.get(key, [])))
        else:
            if not hasattr(self, "_stridx"):
                self._stridx: dict[str, list[int]] = {}
                for i, st in enumerate(self._ref_stmts):
                    for s in {s for x in ast.walk(st) if isinstance(x, ast.Constant) and (s := _const_str(x))}:
                        self._stridx.setdefault(s, []).append(i)
            pos = self._stridx.get(key, [])
        start = bisect.bisect_left(pos, bisect.bisect_right(self._ref_lines, line))  # positions follow line order
        return [self._ref_stmts[i] for i in pos[start:]]

    def _label_sink_ok(self, key: str, is_col: bool, line: int, depth: int = 0) -> bool:
        """`key` (a name, or a column when is_col) only ever ends up as a label (y of a fit, label=...), an
        evaluation, or a summary (value_counts, describe, mean, print, plot, to_csv): never a feature, signal,
        position or model input. Index-only reads (`X.loc[y.index]`, `reindex(y.index)`, `len(y)`, `y.shape`) are
        alignment and don't count (addendum 3). Writes into another container (`positions.extend(labels)`,
        `df.insert(1, "lab", labels)`, `out[k] = labels`) are followed into that container (red-team round 10)."""
        self._feature_uses()
        if depth > 4 or key in self._feat_value_uses or key in self._lit_uses or key in self._signal_reach():
            return False
        for st in self._sink_stmts(key, line, is_col):
            occ = self._key_occ(st.value, key, is_col)
            tgts = st.targets if isinstance(st, ast.Assign) else [st.target] if isinstance(st, ast.AugAssign) else []
            tocc = [x for t in tgts for x in self._key_occ(t, key, is_col)]
            if not occ and not tocc:
                continue  # the string or name appears for another reason (a different frame's column list...)
            if not occ:
                # key only in a target: rebinding it (y = ...), a write into it (labels[0] = 1), or a mask (bad)
                if all(any(x is t for t in tgts) or (isinstance(self.parents.get(x), ast.Subscript)
                                                     and self.parents[x].value is x) for x in tocc):
                    continue
                return False
            if all(self._index_only_name(x) for x in occ):
                continue  # X = feats.loc[y.index].dropna(): aligned to the labels' dates, no label values read
            if isinstance(st, ast.Expr):
                v = st.value
                if isinstance(v, ast.Call) and _attr(v) in self._WRITE_CALLS and isinstance(v.func, ast.Attribute):
                    recv = _chain_base(v.func.value)
                    rk = _key(recv)[0] if isinstance(recv, (ast.Name, ast.Subscript, ast.Attribute)) else None
                    if _attr(v) == "insert" and len(v.args) >= 3 and _const_str(v.args[1]):
                        if not self._label_sink_ok(_const_str(v.args[1]), True, st.lineno, depth + 1):
                            return False  # df.insert(1, "lab", labels): the labels are now column "lab"
                        continue
                    if not rk or rk == key or classify(rk) in ("signal", "feature", "pnl", "return"):
                        return False
                    if not self._label_sink_ok(rk, isinstance(recv, ast.Subscript), st.lineno, depth + 1):
                        return False  # positions.extend(labels): follow `positions`
                    continue
                if self._reloaded(v):
                    return False  # rt12 S02: np.save("labels.npy", labels), then np.load("labels.npy") as a signal
                if self._only_label_arg(v, key, is_col, occ) or self._summary_only(v, key, is_col, occ):
                    continue  # model.fit(X, y), print(y.mean()), y.to_csv(...)
                return False
            if isinstance(st, ast.AugAssign):
                return False
            for t in st.targets:
                for e in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                    tk = _key(e)[0]
                    e_col = isinstance(e, ast.Subscript)
                    if tk == key and e_col == is_col:
                        continue  # y = y.reindex(...), px["target"] = px["target"].fillna(0)
                    if tk and classify(tk) in ("label", "eval"):
                        if not self._label_sink_ok(tk, e_col, st.lineno, depth + 1):
                            return False
                        continue
                    if self._only_label_arg(st.value, key, is_col, occ):
                        continue
                    if (tk and classify(tk) not in ("signal", "feature", "pnl", "return")
                            and self._summary_only(st.value, key, is_col, occ)):
                        # v0.4: balance = y.value_counts(normalize=True): a summary; follow it like an evaluation
                        if not self._label_sink_ok(tk, e_col, st.lineno, depth + 1):
                            return False
                        continue
                    return False
        return True

    def _label_only_loop(self, loop: ast.AST) -> bool:
        """Everything the loop writes is a label or target that is only used as a label: a label builder."""
        if not hasattr(self, "_lol"):
            self._lol: dict[int, bool] = {}
        if id(loop) not in self._lol:
            self._lol[id(loop)] = self._label_only_loop_uncached(loop)
        return self._lol[id(loop)]

    def _label_only_loop_uncached(self, loop: ast.AST) -> bool:
        outs: set[str] = set()
        for st in ast.walk(loop):
            if isinstance(st, (ast.Assign, ast.AugAssign)):
                for t in (st.targets if isinstance(st, ast.Assign) else [st.target]):
                    if isinstance(t, ast.Subscript):
                        b = t.value
                        while isinstance(b, (ast.Subscript, ast.Attribute)):
                            b = b.value
                        if isinstance(b, ast.Name):
                            outs.add(b.id)
                    elif isinstance(st, ast.AugAssign) and isinstance(t, ast.Name):
                        outs.add(t.id)
            elif (isinstance(st, ast.Call) and _attr(st) in ("append", "extend", "insert")
                  and isinstance(st.func.value, ast.Name)):
                outs.add(st.func.value.id)
        if not outs:
            return False
        fd = self._scope_of(loop)
        rets = [r.value for r in ast.walk(fd) if isinstance(r, ast.Return) and r.value is not None] if fd else []
        for o in outs:
            if any(o in self.names_in(r) for r in rets):
                if not hasattr(self, "_calls_by_name"):  # indexed once (red-team round 10: 1,100 labelers took 13 s)
                    self._calls_by_name: dict[str, list[ast.Call]] = {}
                    for c in self._all:
                        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name):
                            self._calls_by_name.setdefault(c.func.id, []).append(c)
                sites = self._calls_by_name.get(fd.name, [])
                if not sites:
                    return False
                fn_label = bool(_tokens(fd.name) & {"label", "labels", "target", "targets", "labeling", "labelling"})
                for c in sites:
                    st = self._stmt_of(c)
                    v = st.value if isinstance(st, ast.Assign) else None
                    while (isinstance(v, ast.Call) and v is not c and _func_name(v) in (
                            "Series", "DataFrame", "array", "asarray") and v.args and v.args[0] is not None):
                        v = v.args[0]  # hits = pd.Series(forward_hit_labels(c), index=...)
                    if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and v is c):
                        return False
                    tk = _key(st.targets[0])[0]
                    if not tk or classify(tk) in ("signal", "feature", "pnl", "return") or not (
                            classify(tk) == "label" or tk in self._label_uses() or fn_label):
                        return False
                    if not self._label_sink_ok(tk, isinstance(st.targets[0], ast.Subscript), st.lineno):
                        return False
            elif classify(o) == "label":
                if not self._label_sink_ok(o, False, getattr(loop, "end_lineno", loop.lineno) or loop.lineno):
                    return False
            else:
                return False
        return True

    # -- LA13: values read from a future bar inside a loop, then used in a decision ------------------
    def _future_sub(self, node: ast.AST, var: str, scope: ast.AST | None = None) -> bool:
        """x[var + k], x.iloc[var + k], x.loc[var + offset:], ... with k > 0 or unknown (a string isn't bars)."""
        if not isinstance(node, ast.Subscript):
            return False
        for c in self._index_parts(node):
            off = self._offset_of(c, var, scope)
            if off == "sym" or (isinstance(off, int) and off > 0):
                if isinstance(c, ast.BinOp) and isinstance(c.op, ast.Sub):
                    continue
                return not self._str_base(node.value, scope)
        return False

    def _comp_and_loop_vars(self):
        for node in self._all:
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                g = node.generators[0]
                t = g.target
                var = t.id if isinstance(t, ast.Name) else (
                    t.elts[0].id if isinstance(t, ast.Tuple) and t.elts and isinstance(t.elts[0], ast.Name) else None)
                if var:
                    body = node.value if isinstance(node, ast.DictComp) else node.elt
                    tests = [x.test for x in ast.walk(body) if isinstance(x, ast.IfExp)] + list(g.ifs)
                    if isinstance(node, ast.DictComp):
                        tests.append(node.value)  # {d: int(x[k + 5] > x[k]) ...}: the value itself is the decision
                    else:
                        tests += [x for x in ast.walk(body) if isinstance(x, ast.Compare)]
                    yield "comp", node, var, tests
            elif isinstance(node, ast.For):
                t = node.target
                var = t.id if isinstance(t, ast.Name) else (
                    t.elts[0].id if isinstance(t, ast.Tuple) and t.elts and isinstance(t.elts[0], ast.Name) else None)
                if var:
                    yield "loop", node, var, None
            elif isinstance(node, ast.While):
                incs = [x.target.id for x in ast.walk(node) if isinstance(x, ast.AugAssign)
                        and isinstance(x.op, ast.Add) and isinstance(x.target, ast.Name) and _num(x.value) is not None]
                if incs:  # the counter (i += 1), not an accumulator (pnl += ...)
                    yield "loop", node, incs[0], None

    def rule_future_taint(self) -> None:
        for kind, node, var, tests in self._comp_and_loop_vars():
            scope = self._scope_of(node)
            if kind == "comp":
                for t in tests:
                    sub = next((x for x in ast.walk(t) if self._future_sub(x, var, scope)), None)
                    if sub is not None:
                        tk, _f, st = self.enclosing_assign_target(node)
                        if (tk and classify(tk) == "label" and st is not None
                                and self._label_sink_ok(tk, isinstance(st.targets[0], ast.Subscript)
                                                        if isinstance(st, ast.Assign) else False, st.lineno)):
                            break  # v0.4: labels = [int(c[i + 5] > c[i]) for i in ...] used only as a label
                        self.add("LA13", "high", sub,
                                 f"A comprehension decides for item `{var}` using item `{var}`+k, a later bar.")
                        break
                continue
            tainted: dict[str, int] = {}
            nonopen: set[str] = set()  # v0.5: tainted by a later bar's data other than bar var+1's open
            aliases = self._row_aliases(var)
            assigns = sorted((x for x in ast.walk(node) if isinstance(x, ast.Assign) and len(x.targets) == 1
                              and isinstance(x.targets[0], ast.Name)), key=lambda a: (a.lineno, a.col_offset))
            for a in assigns:
                tname = a.targets[0].id
                if classify(tname) == "exec":
                    continue  # entry_price = open[i + 1] books a fill; it is not a decision input
                if tname in aliases:
                    tainted.setdefault(tname, a.lineno)  # nxt = bars.iloc[t + 1]: judged by the columns read from it
                    continue
                reads = [r for x in ast.walk(a.value) if isinstance(x, (ast.Subscript, ast.Attribute))
                         and (r := self._next_read(x, var, scope)) is not None]
                names = [x for x in ast.walk(a.value) if isinstance(x, ast.Name) and x.id in tainted]
                if any(self._future_sub(x, var, scope) for x in ast.walk(a.value)) or reads or names:
                    tainted.setdefault(tname, a.lineno)
                    whole_rows = [x for x in names if x.id in aliases and self._alias_col(self.parents.get(x), var) is None]
                    if (any(not (r[0] == 1 and _is_open_name(r[1])) for r in reads) or whole_rows
                            or any(x.id in nonopen for x in names) or not (reads or names)):
                        nonopen.add(tname)
            if not tainted:
                continue
            for st in ast.walk(node):
                test = getattr(st, "test", None) if isinstance(st, (ast.If, ast.While, ast.IfExp)) else None
                if test is None:
                    continue
                occ = [x for x in ast.walk(test) if isinstance(x, ast.Name) and x.id in tainted]
                if not occ:
                    continue
                if self._label_only_loop(node):
                    break  # v0.4: a label builder (triple barrier, forward max): the later bars only reach a label
                if (all(x.id not in nonopen and x.id not in aliases for x in occ)
                        and self.inside(st, (ast.For, ast.While)) is not node):
                    # rt12 K02: entry = o[i + 1]; then an inner loop over later bars j checks a stop derived from that
                    # entry price. Bar i+1's open is known before bar j; the inner loop's own reads are judged there.
                    continue
                if isinstance(st, ast.If):
                    # v0.5: `if nxt["high"] >= target` for a resting order placed with past data (target from the next
                    # bar's open is fine; a level tainted by any other later-bar value is not)
                    self._taint_nonopen = nonopen
                    sim = self._order_sim(st, node, var, scope)
                    self._taint_nonopen = set()
                    self._report_sim_decision(var)
                    if sim == "exempt":
                        continue
                    if sim == "possible":
                        self.add("LA13", "possible", test,
                                 f"This looks like a resting order placed at bar `{var}` and filled if bar `{var}+1` "
                                 "reaches its level, but the fill isn't priced from that level.")
                        continue
                bad = [x for x in occ if not self._in_validity_check(x, test)]
                used = (bad or occ)[0].id
                if bad:
                    self.add("LA13", "high", test,
                             f"The decision at bar `{var}` uses `{used}`, which was read from a later bar "
                             f"(line {tainted[used]}).")
                else:
                    self.add("LA13", "possible", test,
                             f"The condition at bar `{var}` only checks `{used}` (read from a later bar, line "
                             f"{tainted[used]}) for missing data. Fine in a data-quality diagnostic; in a backtest it "
                             "can skip trades before a delisting or a data gap.")
                break

    # -- LA02: windows computed on a reversed series -------------------------------------------
    def rule_reversed_windows(self) -> None:
        win = ("rolling", "expanding", "ewm", "cummax", "cummin", "cumsum", "cumprod")
        for n in self._all:
            if not (isinstance(n, ast.Call) and _attr(n) in win):
                continue
            node, rev = n.func.value, False
            while isinstance(node, (ast.Call, ast.Attribute, ast.Subscript)):
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice) and _num(node.slice.step) == -1:
                    rev = True
                    break
                if isinstance(node, ast.Call) and _attr(node) in ("sort_index", "sort_values"):
                    rev = self._desc_sort(node) is True
                    break
                node = node.func if isinstance(node, ast.Call) else node.value
            if not rev:
                continue
            tkey, _f, _s = self.enclosing_assign_target(n)
            if classify(tkey) in ("label", "eval") and not self._feeds_signal(tkey):
                continue
            self.add("LA02", "high", n, f"`.{_attr(n)}()` runs over a reversed series, so each value summarizes the "
                                        "bars *after* it (a forward-looking window).")

    @staticmethod
    def _desc_sort(call: ast.Call) -> bool | None:
        """True: sorts time descending; False: sorts time ascending; None: not a time sort."""
        a = _attr(call)
        asc = _kw(call, "ascending")
        ax = _kw(call, "axis")
        if ax is not None and (_num(ax) == 1 or _const_str(ax) == "columns"):
            return None  # sorts columns, not time

        def desc(e):
            return isinstance(e, ast.Constant) and (e.value is False or (type(e.value) is int and e.value == 0))
        if a == "sort_index":
            if asc is None:
                return False
            if isinstance(asc, ast.Constant):
                return desc(asc)
            if isinstance(asc, (ast.List, ast.Tuple)):
                return any(desc(e) for e in asc.elts)
            return None
        if a != "sort_values":
            return None
        by = _kw(call, "by") or (call.args[0] if call.args else None)
        keys = [_const_str(e) for e in by.elts] if isinstance(by, (ast.List, ast.Tuple)) else [_const_str(by)]
        date_pos = [i for i, k in enumerate(keys) if k and _tokens(k) & DATE_KEY_T]
        if not date_pos:
            return None
        if asc is None:
            return False
        if isinstance(asc, ast.Constant):
            return desc(asc)
        if isinstance(asc, (ast.List, ast.Tuple)) and len(asc.elts) == len(keys):
            return any(desc(asc.elts[i]) for i in date_pos)
        return None

    def _order_of(self, value: ast.AST, frame: str) -> bool | None:
        """Final time order of `frame = <value>`: True descending, False ascending, None unchanged/unknown."""
        node = value
        while isinstance(node, (ast.Call, ast.Attribute, ast.Subscript)):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice) and _num(node.slice.step) == -1:
                return None  # often fixes a newest-first file; the order before it is unknown
            if isinstance(node, ast.Call) and _attr(node) in ("sort_index", "sort_values"):
                d = self._desc_sort(node)
                if d is not None:
                    return d
            if isinstance(node, ast.Call) and _attr(node) in ("merge", "join", "groupby", "pivot", "pivot_table", "concat"):
                return None
            node = node.func if isinstance(node, ast.Call) else node.value
        if isinstance(node, ast.Name) and node.id == frame:
            return None  # derived from itself without re-sorting: keeps its order
        return False  # a new object (read_csv, another frame): assume ascending

    def rule_reversed_frames(self) -> None:
        """df = df.sort_values('date', ascending=False) (or [::-1]); later df['x'].rolling(...) / pct_change()."""
        events: list[tuple[int, ast.AST | None, str, bool]] = []  # (line, scope, frame, descending)
        for st in self._all:
            scope = self.inside(st, (ast.FunctionDef, ast.AsyncFunctionDef))
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                d = self._order_of(st.value, st.targets[0].id)
                if d is not None:
                    events.append((st.lineno, scope, st.targets[0].id, d))
            elif (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and _attr(st.value) in ("sort_index", "sort_values")
                  and isinstance(st.value.func.value, ast.Name)):
                inplace = _kw(st.value, "inplace")
                d = self._desc_sort(st.value)
                if isinstance(inplace, ast.Constant) and inplace.value is True and d is not None:
                    events.append((st.lineno, scope, st.value.func.value.id, d))
        moving = ("rolling", "expanding", "ewm", "cummax", "cummin", "cumsum", "cumprod", "shift", "pct_change", "diff")
        for line, scope, frame, desc in sorted(events, key=lambda e: e[0]):
            if not desc:
                continue
            end = min((ln for ln, sc, fr, d in events if fr == frame and sc is scope and ln > line), default=10 ** 9)
            hits = 0
            for st in self._iter_refs({frame}, line, end):
                if not isinstance(st, ast.Assign) or hits >= 2:
                    continue
                if frame not in self.names_in(st.value):
                    continue
                if self.inside(st, (ast.FunctionDef, ast.AsyncFunctionDef)) is not scope:
                    continue
                calls = [c for c in ast.walk(st.value) if isinstance(c, ast.Call) and _attr(c) in moving]
                if not calls or (isinstance(st.targets[0], ast.Name) and st.targets[0].id == frame):
                    continue
                tkey, _f = _key(st.targets[0])
                if classify(tkey) in ("label", "eval") and not self._feeds_signal(tkey):
                    continue
                self.add("LA02", "high", st,
                         f"`{frame}` was sorted newest-first on line {line}, so `.{_attr(calls[0])}()` here looks at later "
                         "dates, not earlier ones: every value summarizes the future. Sort ascending by date first.")
                hits += 1

    def _assign_refs(self) -> list[tuple[str, set[str]]]:
        if not hasattr(self, "_arefs"):
            self._arefs: list[tuple[str, set[str]]] = []
            for a in self._all:
                if isinstance(a, ast.Assign) and len(a.targets) == 1:
                    t = a.targets[0]
                    tk, _f = _key(t)
                    if not tk:
                        continue
                    refs = {k for k, _l in _scan_expr(a.value).refs}
                    if isinstance(t, ast.Subscript):  # df.loc[df['x'] > thr, 'signal'] = 1
                        refs |= {k for k, _l in _scan_expr(t.slice).refs}
                    self._arefs.append((tk, refs))
        return self._arefs

    def _feeds_signal(self, key: str | None, depth: int = 0) -> bool:
        """True if `key` is read (directly or through intermediate columns) when building a signal or feature."""
        if not key or depth > 3:
            return False
        if not hasattr(self, "_readers"):  # v0.4 perf: key -> targets of assignments that read it (indexed once)
            self._readers: dict[str, list[str]] = {}
            self._feeds_memo: dict[tuple[str, int], bool] = {}
            for tk, refs in self._assign_refs():
                for r in refs:
                    self._readers.setdefault(r, []).append(tk)
        mk = (key, depth)
        if mk in self._feeds_memo:
            return self._feeds_memo[mk]
        out = False
        for tk in dict.fromkeys(self._readers.get(key, ())):
            if tk == key:
                continue
            role = classify(tk)
            if role in ("signal", "feature") or (role in ("unknown", "eval") and self._feeds_signal(tk, depth + 1)):
                out = True
                break
        self._feeds_memo[mk] = out
        return out

    def _sig_relevant(self, key: str | None) -> bool:
        return classify(key) in ("signal", "feature") or self._feeds_signal(key)

    # -- LA09: global normalization -----------------------------------------------------------
    def _global_stat(self, call: ast.AST, names: tuple[str, ...]) -> ast.AST | None:
        """If `call` is X.<name>() (or np.<name>(X)) with no rolling/expanding/groupby in its chain, return X."""
        if _np_call(call):
            if not call.args or _kw(call, "axis") is not None or call.func.attr not in NP_STATS:
                return None
            base = call.func.attr.removeprefix("nan")
            base = {"amin": "min", "amax": "max", "percentile": "quantile"}.get(base, base)
            return self._global_stat_recv(call.args[0]) if base in names else None
        if not (isinstance(call, ast.Call) and _attr(call) in names):
            return None
        ax = _kw(call, "axis")
        if ax is not None and (_num(ax) == 1 or _const_str(ax) == "columns"):
            return None
        return self._global_stat_recv(call.func.value)

    def _global_stat_recv(self, recv: ast.AST) -> ast.AST | None:
        n = recv
        while True:  # walk down the receiver through calls, attributes and subscripts
            if isinstance(n, ast.Call):
                if _attr(n) in ("rolling", "expanding", "ewm", "groupby", "resample", "transform", "tail", "head"):
                    return None
                n = n.func
            elif isinstance(n, ast.Subscript):
                sl = n.slice.elts[0] if isinstance(n.slice, ast.Tuple) and n.slice.elts else n.slice
                if isinstance(sl, ast.Slice) and (sl.lower is not None or sl.upper is not None):
                    return None  # a window or a training subset, not the whole column
                if isinstance(sl, (ast.Compare, ast.BoolOp)):
                    return None  # a filtered subset (df[df.date < cutoff]['x'])
                n = n.value
            elif isinstance(n, ast.Attribute):
                n = n.value
            else:
                break
        if isinstance(n, ast.Name) and (_tokens(n.id) & SPLIT_T or n.id in self._xs_frames()):
            return None  # statistic of the training set, or of a single date's cross-section
        return recv

    def _xs_frames(self) -> set[str]:
        """Frames holding one date: last = df[df['date'] == d], df.loc[d], df.xs(d), df.iloc[-1], groupby(..).last()."""
        if hasattr(self, "_xsf"):
            return self._xsf
        out: set[str] = set()
        for st in self._all:
            if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
                continue
            for x in ast.walk(st.value):
                if isinstance(x, ast.Subscript) and isinstance(x.slice, ast.Compare) and isinstance(x.slice.ops[0], ast.Eq):
                    k = _key(_chain_base(x.slice.left))[0]
                    if k and _tokens(k) & DATE_KEY_T:
                        out.add(st.targets[0].id)
                if isinstance(x, ast.Call) and (_attr(x) in ("xs", "get_group") or (_attr(x) == "last" and not x.args)):
                    out.add(st.targets[0].id)
                if (isinstance(x, ast.Subscript) and isinstance(x.value, ast.Attribute) and x.value.attr == "iloc"
                        and _num(x.slice) is not None):
                    out.add(st.targets[0].id)
        self._xsf = out
        return out

    @staticmethod
    def _strip_abs(node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Call) and _attr(node) == "abs" and not node.args:
            return node.func.value
        if _np_call(node, "abs", "absolute", "fabs") and node.args:
            return node.args[0]
        return node

    def _scope_of(self, n: ast.AST) -> ast.AST | None:
        return self.inside(n, (ast.FunctionDef, ast.AsyncFunctionDef))

    @property
    def global_scalars(self) -> dict[tuple[int, str], int]:
        """Names assigned a full-column statistic (thr = df['x'].quantile(0.8)), keyed by (scope id, name)."""
        if not hasattr(self, "_gs"):
            self._gs: dict[tuple[int, str], int] = {}
            for st in self._all:
                if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
                    continue
                if self.inside(st, (ast.For, ast.While, ast.Lambda)) is not None:
                    continue
                if classify(st.targets[0].id) in ("label", "eval", "pnl"):
                    continue
                v = st.value
                if self._global_stat(v, ("mean", "median", "quantile", "std", "max", "min")) is not None:
                    recv = self._global_stat(v, ("mean", "median", "quantile", "std", "max", "min"))
                    k, _f = _key(_chain_base(self._strip_abs(recv)))
                    if k and _tokens(k) & DATE_KEY_T:
                        continue
                    self._gs[(id(self._scope_of(st)), st.targets[0].id)] = st.lineno
        return self._gs

    def _scalar_line(self, name: str, at: ast.AST) -> int | None:
        gs = self.global_scalars
        return gs.get((id(self._scope_of(at)), name)) or gs.get((id(None), name))

    def _helper_fn(self, name: str) -> ast.FunctionDef | None:
        fd = self._funcs.get(name)
        return fd if isinstance(fd, ast.FunctionDef) else None

    def _stat_lambda(self, f: ast.AST) -> bool:
        """A function (lambda, local def, zscore) that computes a whole-group mean/std/rank/... with no window."""
        body = None
        if isinstance(f, ast.Lambda):
            body = f.body
        elif isinstance(f, ast.Name):
            if f.id in ("zscore",):
                return True
            fd = self._helper_fn(f.id)
            body = fd if fd is not None else None
        elif isinstance(f, ast.Attribute) and f.attr == "zscore":
            return True
        if body is None:
            return False
        calls = [_attr(c) or _func_name(c) for c in ast.walk(body) if isinstance(c, ast.Call)]
        if any(a in ("rolling", "expanding", "ewm", "shift", "cummax", "cummin", "cumsum", "diff", "pct_change") for a in calls):
            return False
        return any(a in ("mean", "std", "median", "min", "max", "rank", "quantile", "qcut", "zscore") for a in calls)

    def _full_group_stat(self, call: ast.Call) -> bool:
        """groupby(<non-date key>)[...].transform('mean' | lambda s: (s - s.mean()) / s.std())."""
        node = call.func.value
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            if self._date_col(node):
                return False  # each asset's first/last date is bookkeeping, not a scaled feature
            node = node.value
        if not (isinstance(node, ast.Call) and _attr(node) == "groupby"):
            return False
        key = node.args[0] if node.args else (_kw(node, "by") or _kw(node, "level"))
        names = [_const_str(e) for e in key.elts] if isinstance(key, (ast.List, ast.Tuple)) else [_const_str(key)]
        if not names or any(nm is None for nm in names) or any(_date_like(nm) for nm in names):
            return False  # unknown, index-level or date grouping: cross-sectional or too uncertain
        if not call.args:
            return False
        f = call.args[0]
        if _const_str(f) in ("mean", "std", "median", "min", "max", "var", "rank", "quantile", "zscore"):
            return True
        return self._stat_lambda(f)

    def _applied_funcs(self) -> set[str]:
        """Functions run per group, per row or per window (passed to apply/transform/agg/map, or called in a lambda),
        or called inside a loop over dates: their statistics are not full-sample."""
        if hasattr(self, "_applied"):
            return self._applied
        out: set[str] = set()
        for c in self._all:
            if not isinstance(c, ast.Call):
                continue
            if _attr(c) in ("apply", "transform", "agg", "aggregate", "map", "applymap") or _func_name(c) == "map":
                for a in list(c.args) + [k.value for k in c.keywords]:
                    if isinstance(a, ast.Name):
                        out.add(a.id)
            if isinstance(c.func, ast.Name) and (self.inside(c, (ast.Lambda,)) is not None
                                                 or any(self._xs_loop(lp) for lp in self._loops_of(c))):
                out.add(c.func.id)
        self._applied = out
        return out

    def _loops_of(self, n: ast.AST) -> list[ast.AST]:
        out, p = [], self.parents.get(n)
        while p is not None:
            if isinstance(p, (ast.For, ast.comprehension)):
                out.append(p)
            p = self.parents.get(p)
        return out

    @staticmethod
    def _xs_loop(loop: ast.AST) -> bool:
        """A loop over dates (one cross-section per iteration): for d in dates / for dt, g in df.groupby('date')."""
        t, it = loop.target, loop.iter
        names = [e for e in (t.elts if isinstance(t, ast.Tuple) else [t]) if isinstance(e, ast.Name)]
        if any(_tokens(e.id) & DATE_KEY_T for e in names[:1]):
            return True
        if isinstance(it, ast.Call) and _attr(it) == "groupby":
            key = it.args[0] if it.args else _kw(it, "by")
            k = _const_str(key)
            return _date_like(k) or _kw(it, "level") is not None
        base = _chain_base(it)
        k, _f = _key(base)
        return _date_like(k)

    def _ts_frames(self) -> set[tuple[int, str]]:
        """(scope, frame) pairs that time-series operations are applied to (df['close'].pct_change(), ...)."""
        if hasattr(self, "_tsf"):
            return self._tsf
        out: set[tuple[int, str]] = set()
        for c in self._all:
            if isinstance(c, ast.Call) and _attr(c) in TS_OPS:
                base = _chain_base(c.func.value)
                while isinstance(base, ast.Call) and _attr(base) == "groupby":
                    base = _chain_base(base.func.value)
                _k, f = _key(base)
                name = f or (base.id if isinstance(base, ast.Name) else None)
                if name:
                    out.add((id(self._scope_of(c)), name))
        for st in self._all:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and any(
                    isinstance(c, ast.Call) and _attr(c) in TS_OPS for c in ast.walk(st.value)):
                f = _key(st.targets[0])[1]
                if f:
                    out.add((id(self._scope_of(st)), f))
        self._tsf = out
        return out

    def _ts_info(self, k: str, at: ast.AST, depth: int = 0) -> bool:
        """`k` was last built with time-series operations in the same scope as `at` (directly or via its inputs)."""
        if depth > 4:
            return False
        if not hasattr(self, "_ts_memo"):
            self._ts_memo: dict[tuple, bool] = {}
            self._assign_by_key: dict[str, list[ast.Assign]] = {}
            for st in self._stmts_sorted:
                if isinstance(st, ast.Assign) and len(st.targets) == 1:
                    tk = _key(st.targets[0])[0]
                    if tk:
                        self._assign_by_key.setdefault(tk, []).append(st)
        scope = self._scope_of(at)
        memo_key = (k, id(scope), getattr(at, "lineno", 0))
        if memo_key in self._ts_memo:
            return self._ts_memo[memo_key]
        self._ts_memo[memo_key] = False  # guards cycles
        out = False
        for st in reversed(self._assign_by_key.get(k, [])):
            # v0.4: strictly earlier lines: `feats = feats.apply(...)` reads the previous `feats`, not itself
            if st.lineno < getattr(at, "lineno", 0) and self._scope_of(st) is scope:
                v = st.value
                if any(isinstance(c, ast.Call) and (_attr(c) in TS_OPS or _np_call(c, "diff", "roll")) for c in ast.walk(v)):
                    out = True
                else:
                    refs = [rk for rk, _l in _scan_expr(v).refs if rk != k]
                    out = len(refs) <= 6 and any(self._ts_info(rk, st, depth + 1) for rk in refs)
                break
        self._ts_memo[memo_key] = out
        return out

    def _ts_evidence(self, x: ast.AST | None) -> bool:
        """`x` is (part of) a series over time, not one date's cross-section."""
        if x is None:
            return False
        x = self._strip_abs(x)
        if any(isinstance(c, ast.Call) and _attr(c) in TS_OPS for c in ast.walk(x)):
            return True
        k, f = _key(_chain_base(x))
        if not k:
            return False
        sc = id(self._scope_of(x))
        frame = f or (k if isinstance(_chain_base(x), ast.Name) else None)
        return bool((frame and (sc, frame) in self._ts_frames()) or self._ts_info(k, x))

    def _ts_ok(self, n: ast.AST, x: ast.AST | None) -> bool:
        """Inside functions (often run on one date's cross-section) require evidence that x is a time series."""
        return self._scope_of(n) is None or self._ts_evidence(x)

    def _norm_scope_ok(self, n: ast.AST) -> bool:
        if self.inside(n, (ast.Lambda,)) is not None:
            return False
        fd = self._scope_of(n)
        if fd is not None and fd.name in self._applied_funcs():
            return False
        return not any(self._xs_loop(lp) for lp in self._loops_of(n))

    def _param_helper(self, n: ast.AST, x: ast.AST) -> ast.FunctionDef | None:
        """The normalization is `return (s - s.mean()) / s.std()` on a bare parameter of a helper function."""
        fd = self._scope_of(n)
        if fd is None or not isinstance(x, ast.Name):
            return None
        params = {a.arg for a in fd.args.posonlyargs + fd.args.args}
        return fd if x.id in params else None

    def _norm_expr(self, n: ast.AST) -> tuple[ast.AST, str] | None:
        """(X, description) if `n` scales X by statistics of the whole of X."""
        if not (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)):
            return None
        if isinstance(n.left, ast.BinOp) and isinstance(n.left.op, ast.Sub):
            x = n.left.left
            mean_of = self._global_stat(n.left.right, ("mean", "median", "min"))
            std_of = self._global_stat(n.right, ("std", "var", "mad"))
            if std_of is None and isinstance(n.right, ast.BinOp) and isinstance(n.right.op, ast.Sub):
                std_of = self._global_stat(n.right.left, ("max",))
            if mean_of is not None and std_of is not None and ast.dump(mean_of) == ast.dump(x) == ast.dump(std_of):
                return x, "This normalizes with the mean/std (or min/max) of the entire series, including rows after the one being scaled."
        scale_of = self._global_stat(n.right, ("std", "max", "var", "mad"))
        if scale_of is not None and ast.dump(self._strip_abs(scale_of)) == ast.dump(n.left) and not isinstance(n.left, ast.Constant):
            return n.left, ("This divides by the standard deviation or maximum of the entire series, so every row's "
                            "scale (and position size) depends on values that came later.")
        return None

    # -- LA09 (v0.4): full-column normalization passed to apply/transform/pipe -----------------------------
    def _self_norm_func(self, f: ast.AST) -> bool:
        """`f` scales its argument by that argument's own whole mean/std/min/max: a lambda or helper doing
        (c - c.mean()) / c.std(), (c - c.min()) / (c.max() - c.min()), c / c.abs().max(), or zscore."""
        if isinstance(f, ast.Name) and f.id == "zscore" and self._helper_fn("zscore") is None and (
                "scipy" in self.imports or "zscore" in self.src):
            return True
        if isinstance(f, ast.Attribute) and f.attr == "zscore":
            return True
        fd, param = None, None
        if isinstance(f, ast.Lambda):
            args = f.args.posonlyargs + f.args.args
            if not args:
                return False
            param, bodies = args[0].arg, [f.body]
        elif isinstance(f, ast.Name) and self._helper_fn(f.id) is not None:
            fd = self._helper_fn(f.id)
            args = fd.args.posonlyargs + fd.args.args
            if not args:
                return False
            param = args[0].arg
            bodies = [r.value for r in ast.walk(fd) if isinstance(r, ast.Return) and r.value is not None]
            if not bodies:
                return False
        else:
            return False
        for body in bodies:
            for b in ast.walk(body):
                got = self._norm_expr(b) if isinstance(b, ast.BinOp) else None
                if got is not None and isinstance(got[0], ast.Name) and got[0].id == param:
                    return True
                if (isinstance(b, ast.Call) and _func_name(b) == "zscore" and b.args and isinstance(b.args[0], ast.Name)
                        and b.args[0].id == param and "zscore" not in self._funcs):
                    return True
        return False

    def _apply_receiver_ok(self, n: ast.Call) -> ast.AST | None:
        """The frame whose columns are normalized one by one, or None when the call works per row (axis=1), per
        window (rolling/expanding/ewm), per group (groupby), per date (a transposed frame), or on a training slice or
        one date's cross-section."""
        if _attr(n) != "pipe":
            ax = _kw(n, "axis") if _kw(n, "axis") is not None else (n.args[1] if len(n.args) > 1 and _attr(n) == "apply" else None)
            if ax is not None and (_num(ax) == 1 or _const_str(ax) == "columns"):
                return None
        recv = n.func.value
        x = recv
        while True:
            if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute):
                if x.func.attr in ("groupby", "rolling", "expanding", "ewm", "resample", "transpose", "T", "stack",
                                   "unstack", "apply", "transform", "agg", "aggregate", "pivot", "pivot_table"):
                    return None
                x = x.func.value
            elif isinstance(x, ast.Attribute):
                if x.attr == "T":
                    return None
                x = x.value
            elif isinstance(x, ast.Subscript):
                x = x.value
            else:
                break
        if self._global_stat_recv(recv) is None:
            return None
        return recv

    def _ts_recv(self, recv: ast.AST, at: ast.AST) -> bool:
        """The receiver of apply/transform is a series over time: built with rolling/shift/pct_change (here or where
        its base frame was assigned)."""
        if any(isinstance(c, ast.Call) and _attr(c) in TS_OPS for c in ast.walk(recv)):
            return True
        b = recv
        while isinstance(b, (ast.Call, ast.Attribute, ast.Subscript)):
            b = b.func.value if isinstance(b, ast.Call) and isinstance(b.func, ast.Attribute) else (
                b.func if isinstance(b, ast.Call) else b.value)
        if not isinstance(b, ast.Name):
            return False
        scope = id(self._scope_of(at))
        return (scope, b.id) in self._ts_frames() or self._ts_info(b.id, at)

    def _feeds_model_or_signal(self, key: str | None) -> bool:
        """Like _sig_relevant, but a frame also counts when one of its columns is read (scaled["trend"])."""
        if key is None:
            return True
        if self._sig_relevant(key) or key in self._feature_uses() or key in self._lit_uses:
            return True
        for st in self._iter_refs({key}, 0):
            if isinstance(st, ast.Assign) and len(st.targets) == 1:
                tk = _key(st.targets[0])[0]
                if tk and tk != key and (classify(tk) in ("signal", "feature") or self._sig_relevant(tk)
                                         or tk in self._feature_uses()):
                    return True
        return False

    def _rule_apply_norm(self, n: ast.Call) -> None:
        if not (n.args and _attr(n) in ("apply", "transform", "pipe") and self._self_norm_func(n.args[0])):
            return
        recv = self._apply_receiver_ok(n)
        if recv is None or not self._norm_scope_ok(n):
            return
        tkey, _f, _s = self.enclosing_assign_target(n)
        if classify(tkey) in ("label", "eval", "pnl"):
            return
        ts = self.ts_context and self._ts_recv(recv, n)
        sev = "medium" if ts and self._feeds_model_or_signal(tkey) else "possible"
        self.add("LA09", sev, n,
                 f"`.{_attr(n)}(...)` scales each column by its own mean/std (or min/max) over every row, including "
                 "rows after the one being scaled. Use a rolling or expanding statistic, statistics of the training "
                 "window, or a per-date (cross-sectional) transform.")

    def rule_global_norm(self) -> None:
        norm_helpers: dict[str, ast.FunctionDef] = {}
        for n in self._all:
            if isinstance(n, ast.Call) and _attr(n) in ("apply", "transform", "pipe"):
                self._rule_apply_norm(n)
            got = self._norm_expr(n) if isinstance(n, ast.BinOp) else None
            if got is not None:
                x, why = got
                helper = self._param_helper(n, x)
                if helper is not None and self.inside(n, (ast.Return,)) is not None:
                    norm_helpers[helper.name] = helper  # judged at the call sites below
                    continue
                if not self._norm_scope_ok(n):
                    continue
                tkey, _f, _s = self.enclosing_assign_target(n)
                if classify(tkey) == "label":
                    continue
                if not self.ts_context or not self._ts_ok(n, x):
                    sev = "possible"
                elif tkey is None or self._sig_relevant(tkey):
                    sev = "medium"
                else:
                    sev = "possible"
                self.add("LA09", sev, n, why)
            if isinstance(n, ast.Compare) and self.ts_context and self._norm_scope_ok(n):
                self._threshold_compare(n)
            if isinstance(n, ast.Call) and _attr(n) in ("transform", "apply") and self.ts_context and self._full_group_stat(n):
                tkey, _f, _s = self.enclosing_assign_target(n)
                if classify(tkey) not in ("label", "eval", "pnl") and self._ts_ok(n, n.func.value):
                    self.add("LA09", "medium", n,
                             "`groupby(<asset>).transform(...)` computes each asset's mean/std/min/max over its whole "
                             "history, including rows after the one being scored. Use a rolling or expanding statistic "
                             "inside the group, or group by date for cross-sectional scores.")
            if (isinstance(n, ast.Call) and _func_name(n) in ("zscore", "scale", "minmax_scale", "robust_scale")
                    and self.ts_context and n.args and ({"scipy", "sklearn"} & self.imports or _func_name(n) == "zscore")
                    and self._helper_fn(_func_name(n)) is None):
                if not (isinstance(self.parents.get(n), ast.Call) and _attr(self.parents.get(n)) in ("transform", "apply")):
                    if self._norm_scope_ok(n) and self._global_stat_recv(n.args[0]) is not None:
                        tkey, _f, _s = self.enclosing_assign_target(n)
                        sev = "medium" if self._sig_relevant(tkey) and self._ts_evidence(n.args[0]) else "possible"
                        self.add("LA09", sev, n, f"`{_func_name(n)}(...)` over the full column uses the future mean and std.")
            if isinstance(n, ast.Call) and _attr(n) == "rank" and self.ts_context:
                pct = _kw(n, "pct")
                is_col = isinstance(n.func.value, ast.Subscript) and _const_str(n.func.value.slice) is not None
                if ((isinstance(pct, ast.Constant) and pct.value is True) or is_col) and self._global_stat(n, ("rank",)) is not None:
                    tkey, _f, _s = self.enclosing_assign_target(n)
                    if classify(tkey) != "label" and self._norm_scope_ok(n):
                        sev = "medium" if self._sig_relevant(tkey) and self._ts_evidence(n.func.value) else "possible"
                        self.add("LA09", sev, n,
                                 "`rank()` over the whole column ranks each row against future rows. "
                                 "Rank within each date (cross-section) or over a trailing window instead.")
            if isinstance(n, ast.Call) and _func_name(n) == "qcut" and self.ts_context and n.args:
                k, _ = _key(_chain_base(n.args[0]))
                tkey, _f, _s = self.enclosing_assign_target(n)
                if (k and classify(tkey) != "label" and self._norm_scope_ok(n)
                        and self._global_stat_recv(n.args[0]) is not None):
                    sev = "medium" if self._sig_relevant(tkey) and self._ts_evidence(n.args[0]) else "possible"
                    self.add("LA09", sev, n, "`pd.qcut` over the whole column sets bin edges with future data.")
        for c in self._all:
            if not (isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in norm_helpers and c.args):
                continue
            if not self.ts_context or not self._norm_scope_ok(c) or self._global_stat_recv(c.args[0]) is None:
                continue
            if self.inside(c, (ast.FunctionDef,)) is norm_helpers[c.func.id]:
                continue
            tkey, _f, _s = self.enclosing_assign_target(c)
            if classify(tkey) == "label":
                continue
            sev = "medium" if self._sig_relevant(tkey) and self._ts_ok(c, c.args[0]) else "possible"
            self.add("LA09", sev, c, f"`{c.func.id}()` normalizes with the mean/std (or min/max) of the whole series "
                                     "it is given, here a full column, so every row is scaled with future data.")

    def _threshold_compare(self, n: ast.Compare) -> None:
        tkey, _f, stmt = self.enclosing_assign_target(n)
        if classify(tkey) not in ("signal", "feature", "unknown") or tkey is None:
            return
        filt = self.inside(n, (ast.Subscript,))
        in_filter = filt is not None and any(x is n for x in ast.walk(filt.slice))
        target_is_row_filter = in_filter and not (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Subscript)
                                                  and any(x is n for x in ast.walk(stmt.targets[0])))
        sev = "possible" if target_is_row_filter or not self._sig_relevant(tkey) else "medium"
        if sev == "medium" and self._scope_of(n) is not None and not any(
                self._ts_evidence(sd) or (isinstance(sd, ast.Name) and self._scalar_line(sd.id, n) is None
                                          and self._ts_evidence(sd)) for sd in [n.left] + list(n.comparators)):
            sev = "possible"
        sides = [n.left] + list(n.comparators)

        def series_like(side: ast.AST) -> bool:
            names = [x for x in ast.walk(side) if isinstance(x, (ast.Name, ast.Subscript, ast.Attribute))]
            return any(not (isinstance(x, ast.Name) and (self._scalar_line(x.id, n) or x.id in MODULE_ALIASES))
                       for x in names if _key(x)[0] is not None)

        if not any(series_like(sd) for sd in sides):
            return
        for side in [n.left] + list(n.comparators):
            k, _f2 = _key(_chain_base(side))
            if k and _tokens(k) & DATE_KEY_T:
                return
            scalar = next((x.id for x in ast.walk(side) if isinstance(x, ast.Name) and self._scalar_line(x.id, n)), None)
            if scalar is not None:
                self.add("LA09", sev, n,
                         f"`{tkey}` compares against `{scalar}`, a statistic of the whole column (line "
                         f"{self._scalar_line(scalar, n)}) that includes future rows. Use a trailing "
                         "(rolling/expanding) statistic or compute it on the training window only.")
                return
            recv = self._global_stat(side, ("mean", "median", "quantile", "std", "max", "min"))
            if recv is not None:
                rk, _f3 = _key(_chain_base(self._strip_abs(recv)))
                if rk and _tokens(rk) & DATE_KEY_T:
                    return
                self.add("LA09", sev, n,
                         f"`{tkey}` compares against `.{_attr(side) or _func_name(side)}()` of the whole column, a threshold "
                         "that includes future rows. Use a trailing (rolling/expanding) statistic.")
                return

    # -- SB01: keeping only names that trade on the last date ------------------------------------
    def _assigns_of(self, name: str) -> list[ast.Assign]:
        """Assignments to the bare name `name`, in tree-walk order (indexed once per file)."""
        if not hasattr(self, "_by_name"):
            self._by_name: dict[str, list[ast.Assign]] = {}
            for st in self._all:
                if isinstance(st, ast.Assign):
                    for t in st.targets:
                        if isinstance(t, ast.Name):
                            self._by_name.setdefault(t.id, []).append(st)
            for v in self._by_name.values():
                v.sort(key=lambda x: (x.lineno, x.col_offset))
        return self._by_name.get(name, [])

    def _assigned_value(self, name: str, before: int) -> ast.AST | None:
        best = None
        for st in self._assigns_of(name):
            if st.lineno > before:
                break
            if len(st.targets) == 1:
                best = st.value
        return best

    def _date_col(self, node: ast.AST) -> bool:
        k, _f = _key(node) if isinstance(node, (ast.Subscript, ast.Attribute)) else (None, None)
        if isinstance(node, ast.Attribute) and node.attr == "index":
            return True
        return bool(k and _tokens(k) & DATE_KEY_T)

    def _last_date_kind(self, node: ast.AST, depth: int = 0) -> str | None:
        """'asset' for each asset's last date (groupby(t)['date'].max()/.transform('max')), 'global' for the
        panel's last date (df['date'].max(), df.index.max()), else None."""
        if isinstance(node, ast.Name) and depth < 2:
            v = self._assigned_value(node.id, getattr(node, "lineno", 10 ** 9))
            return self._last_date_kind(v, depth + 1) if v is not None else None
        if (isinstance(node, ast.Subscript) and _num(node.slice) == -1 and isinstance(node.value, ast.Attribute)
                and node.value.attr == "iloc" and self._date_col(node.value.value)):
            return "global"  # df['date'].iloc[-1]
        if not isinstance(node, ast.Call):
            return None
        a = _attr(node)
        is_max = a in ("max", "last") or (a in ("transform", "agg", "aggregate") and node.args and _const_str(node.args[0]) in ("max", "last"))
        if not is_max:
            return None
        recv = node.func.value
        chain = [recv]
        while isinstance(chain[-1], (ast.Call, ast.Attribute, ast.Subscript)):
            nxt = chain[-1].func if isinstance(chain[-1], ast.Call) else chain[-1].value
            chain.append(nxt)
        grouped = any(isinstance(c, ast.Call) and _attr(c) == "groupby" for c in chain)
        dated = any(self._date_col(c) for c in chain if isinstance(c, (ast.Subscript, ast.Attribute)))
        if not dated:
            return None
        return "asset" if grouped else "global"

    def _survivor_compare(self, n: ast.AST, param: str | None = None) -> bool:
        if not (isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Eq, ast.GtE))):
            return False
        sides = [n.left, n.comparators[0]]
        kinds = []
        for sd in sides:
            k = self._last_date_kind(sd)
            if param is not None and k == "global" and isinstance(_chain_base(sd), ast.Subscript):
                b = _chain_base(sd)
                while isinstance(b, (ast.Subscript, ast.Attribute)):
                    b = b.value
                if isinstance(b, ast.Name) and b.id == param:
                    k = "asset"  # g['date'].max() inside groupby(...).filter(lambda g: ...)
            if param is not None and k == "global" and isinstance(_chain_base(sd), ast.Attribute):
                b = _chain_base(sd)
                while isinstance(b, (ast.Subscript, ast.Attribute)):
                    b = b.value
                if isinstance(b, ast.Name) and b.id == param:
                    k = "asset"
            kinds.append(k)
        return set(kinds) == {"asset", "global"}

    def _row_date(self, node: ast.AST) -> bool:
        """A date column (or the last date) on either side of `df['date'] == df['date'].max()`."""
        if isinstance(node, (ast.Subscript, ast.Attribute)) and self._date_col(node):
            return True
        return self._last_date_kind(node) == "global"

    def _panel_cut(self, stmt: ast.AST, src_frame: str | None) -> bool:
        """`stmt` assigns a filtered panel that is then used as the backtest universe: it overwrites the panel
        (df = df[...]) or time-series operations run on it."""
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            return False
        f = stmt.targets[0].id
        if src_frame is not None and f == src_frame:
            return True
        return (id(self._scope_of(stmt)), f) in self._ts_frames()

    def rule_survivor_filter(self) -> None:
        msg = ("This keeps only the names whose last date equals the panel's last date, i.e. the ones still trading at "
               "the end. Every delisted, acquired or bankrupt company is dropped from the whole backtest. Keep them, "
               "with their final returns.")
        for n in self._all:
            # df.groupby('ticker').filter(lambda g: g['date'].max() == end)
            if (isinstance(n, ast.Call) and _attr(n) == "filter" and n.args and isinstance(n.args[0], ast.Lambda)
                    and isinstance(n.func.value, ast.Call) and _attr(n.func.value) == "groupby"):
                lam = n.args[0]
                param = lam.args.args[0].arg if lam.args.args else None
                cmp = next((c for c in ast.walk(lam.body) if self._survivor_compare(c, param)), None)
                src = _chain_base(n.func.value.func.value)
                _k, _f, stmt = self.enclosing_assign_target(n)
                if cmp is not None and self._panel_cut(stmt, src.id if isinstance(src, ast.Name) else None):
                    self.add("SB01", "medium", n, msg)
                continue
            last_day_rows = (isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq)
                             and {self._row_date(n.left), self._row_date(n.comparators[0])} == {True}
                             and {self._last_date_kind(n.left), self._last_date_kind(n.comparators[0])} == {None, "global"})
            if not (self._survivor_compare(n) or last_day_rows):
                continue
            _k, _f, stmt = self.enclosing_assign_target(n)
            filt = self.inside(n, (ast.Subscript,))
            if filt is not None and any(x is n for x in ast.walk(filt.slice)) and isinstance(stmt, ast.Assign):
                src = _chain_base(filt.value)
                base = src
                while isinstance(base, (ast.Subscript, ast.Attribute)):
                    base = base.value
                srcname = base.id if isinstance(base, ast.Name) else None
                if not last_day_rows and self._panel_cut(stmt, srcname):
                    self.add("SB01", "medium", n, msg)
                    continue  # (rows of the last day alone are fine: that is one date's cross-section)
                # alive = last[last == end].index ; df = df[df['ticker'].isin(alive)]
                if isinstance(stmt.targets[0], ast.Name):
                    keep = stmt.targets[0].id
                    if not hasattr(self, "_isin_calls"):
                        self._isin_calls = [c for c in self._all if isinstance(c, ast.Call) and _attr(c) == "isin"]
                    for c in self._isin_calls:
                        if c.lineno <= stmt.lineno or keep not in self.names_in(c):
                            continue
                        st = self._stmt_of(c)
                        if not isinstance(st, ast.Assign) or isinstance(self.parents.get(c), ast.UnaryOp):
                            continue  # ~isin(...) marks the delisted ones; it doesn't drop them
                        sub = self.inside(c, (ast.Subscript,))
                        if sub is None:
                            continue
                        b2 = sub.value
                        while isinstance(b2, (ast.Subscript, ast.Attribute)):
                            b2 = b2.value
                        if self._panel_cut(st, b2.id if isinstance(b2, ast.Name) else None):
                            self.add("SB01", "medium", n, msg)
                            break

    # -- SB01: hard-coded ticker lists ----------------------------------------------------------
    def rule_universe_lists(self) -> None:
        for n in self._all:
            if isinstance(n, (ast.List, ast.Tuple)) and len(n.elts) >= 10:
                vals = [_const_str(e) for e in n.elts]
                if all(v is not None and TICKER_RE.match(v) for v in vals):
                    self.add("SB01", "possible", n,
                             f"Hard-coded list of {len(vals)} tickers. If it is today's list (or names you know did "
                             "well), the backtest excludes companies that failed or were dropped.")
        if "yfinance" in self.imports:
            node = next((x for x in self._all if isinstance(x, (ast.Import, ast.ImportFrom))
                         and any(getattr(a, "name", "") == "yfinance" for a in x.names) or
                         (isinstance(x, ast.ImportFrom) and x.module == "yfinance")), self.tree.body[0])
            self.add("DQ01", "info", node, "Data comes from yfinance (no delisted tickers; adjusted history is restated).")
            if re.search(r"[\"']Adj Close[\"']", self.src) and re.search(r"[\"'](Open|High|Low)[\"']", self.src):
                n2 = next((x for x in self._all if _const_str(x) == "Adj Close"), node)
                self.add("DQ02", "possible", n2,
                         "`Adj Close` is combined with raw Open/High/Low; on split and dividend days they disagree.")

    # =========================================================================================================
    # v0.6 recall rules (auditor/docs/RECALL_ANALYSIS_v06.md; tests/corpus_v06/PREREGISTRATION.md). Each rule is a
    # narrow pattern; when it can't show that the value reaches a position, a model or PnL it says "possible".
    # =========================================================================================================
    def _v6_writes(self, st: ast.AST) -> tuple[set[str], set[str], set[str], set[str]]:
        """(names written, keys written, extra names read, extra keys read) of an assignment statement. A subscript
        target with a computed selector (`signal.iloc[lows] = 1`) writes its base name and reads the selector."""
        wn: set[str] = set()
        wk: set[str] = set()
        rn: set[str] = set()
        rk: set[str] = set()
        tg = st.targets if isinstance(st, ast.Assign) else [getattr(st, "target", None)]
        for t in tg:
            if t is None:
                continue
            for x in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                if isinstance(x, ast.Starred):
                    x = x.value
                if isinstance(x, ast.Name):
                    wn.add(x.id)
                    wk.add(x.id)
                    continue
                if isinstance(x, ast.Subscript):
                    rn |= self.names_in(x.slice)
                    rk |= {k for k, _l in _scan_expr(x.slice).refs}
                k = _key(x)[0]
                if k:
                    wk.add(k)
                else:
                    b = x
                    while isinstance(b, (ast.Subscript, ast.Attribute)):
                        b = b.value
                    if isinstance(b, ast.Name):
                        wn.add(b.id)
                        wk.add(b.id)
        return wn, wk, rn, rk

    def _v6_index(self):
        """Assignments in line order with what they write, indexed by the names and keys they read (built once)."""
        if hasattr(self, "_v6i"):
            return self._v6i
        rows: list[tuple[ast.AST, set[str], set[str]]] = []
        by_name: dict[str, list[int]] = {}
        by_key: dict[str, list[int]] = {}
        for st in self._stmts_sorted:
            if not isinstance(st, (ast.Assign, ast.AugAssign, ast.AnnAssign)) or getattr(st, "value", None) is None:
                continue
            wn, wk, rn, rk = self._v6_writes(st)
            rn |= self.names_in(st.value)
            rk |= self._refkeys(st.value)
            i = len(rows)
            rows.append((st, wn, wk))
            for nm in rn:
                by_name.setdefault(nm, []).append(i)
            for k in rk:
                by_key.setdefault(k, []).append(i)
        self._v6i = (rows, by_name, by_key)
        return self._v6i

    @staticmethod
    def _lag_covers(amt: ast.AST | None, reach: int | ast.AST | None) -> bool:
        """A positive shift by `amt` bars is at least `reach` bars (a number, or the same name/expression)."""
        if amt is None or reach is None:
            return False
        v = _numc(amt)
        if isinstance(reach, int):
            return v is not None and v >= reach
        rv = _numc(reach)
        if v is not None and rv is not None:
            return v >= rv
        return ast.dump(amt) == ast.dump(reach)

    def _lagged_read(self, value: ast.AST, x: str, reach) -> bool:
        """Every read of `x` in `value` sits under a positive shift of at least `reach` bars."""
        reads = [n for n in ast.walk(value) if (isinstance(n, ast.Name) and n.id == x)
                 or (isinstance(n, ast.Subscript) and _const_str(n.slice) == x)]
        if not reads:
            return False
        for r in reads:
            ok, p = False, self.parents.get(r)
            while p is not None and not isinstance(p, ast.stmt):
                if isinstance(p, ast.Call) and _attr(p) == "shift" and _shift_sign(p) == 1 and self._lag_covers(
                        _kw(p, "periods") or (p.args[0] if p.args else None), reach):
                    ok = True
                    break
                p = self.parents.get(p)
            if not ok:
                return False
        return True

    _SYM_REACH = 1000  # a look-ahead given by a name (order=k): a shift by the same name covers it

    def _lag_amount(self, c: ast.Call, reach) -> int:
        amt = _kw(c, "periods") or (c.args[0] if c.args else None)
        if amt is None:
            return 1
        v = _numc(amt)
        if v is not None:
            return int(v) if v > 0 else 0
        if reach is not None and not isinstance(reach, int) and ast.dump(amt) == ast.dump(reach):
            return self._SYM_REACH
        return 0

    def _lag_above(self, r: ast.AST, top: ast.AST | None, reach) -> int:
        """Bars of positive shift applied to `r` on the way up to `top` (or to its statement)."""
        lag, p = 0, self.parents.get(r)
        while p is not None and p is not top and not isinstance(p, ast.stmt):
            if isinstance(p, ast.Call) and _attr(p) == "shift" and _shift_sign(p) == 1 and p.func.value is not None:
                lag += self._lag_amount(p, reach)
            p = self.parents.get(p)
        if p is top and isinstance(p, ast.Call) and _attr(p) == "shift" and _shift_sign(p) == 1:
            lag += self._lag_amount(p, reach)
        return lag

    def _v6_mult_reads(self) -> dict[str, list[tuple[ast.AST, ast.AST]]]:
        """key -> (read node, product side) for reads inside a side of a product whose other side is a return."""
        if not hasattr(self, "_v6mr"):
            def is_ret(side: ast.AST) -> bool:
                return any(classify(k) == "return" or (self.info.get(k) is not None and self.info[k].is_return)
                           for k, _l in _scan_expr(side).refs) or any(
                    isinstance(c, ast.Call) and _attr(c) in ("pct_change", "diff") for c in ast.walk(side))
            out: dict[str, list[tuple[ast.AST, ast.AST]]] = {}
            for m in self._all:
                if not (isinstance(m, ast.BinOp) and isinstance(m.op, ast.Mult)):
                    continue
                for side, other in ((m.left, m.right), (m.right, m.left)):
                    if not is_ret(other):
                        continue
                    for r in ast.walk(side):
                        k = r.id if isinstance(r, ast.Name) else (_const_str(r.slice) if isinstance(r, ast.Subscript) else None)
                        if k:
                            out.setdefault(k, []).append((r, side))
            self._v6mr = out
        return self._v6mr

    def _v6_bounded_sev(self, n: ast.Call, reach) -> str | None:
        """Severity for a transform that reads a bounded number of bars ahead (np.gradient: 1, argrelextrema: order).
        Lags are added up along the flow: a signal is causal once lagged by the look-ahead, and a position booked
        against the same bar's return needs one bar more (rt14 F3; v3 leaky_12 lags only once)."""
        R = reach if isinstance(reach, int) else (int(_numc(reach)) if _numc(reach) is not None else self._SYM_REACH)
        names, keys, after = self._v6_start(n)
        if any(classify(k) in ("label", "eval") for k in keys) and not any(classify(k) in ("signal", "pnl", "feature")
                                                                             for k in keys):
            return None
        st0 = self._stmt_of(n)
        acc = {k: self._lag_above(n, None, reach) for k in names | keys}
        rows, by_name, by_key = self._v6_index()
        todo, steps = list(acc), 0
        while todo and steps < 2000:
            x = todo.pop()
            for i in set(by_name.get(x, ())) | set(by_key.get(x, ())):
                steps += 1
                st, wn, wk = rows[i]
                if st.lineno <= after or st is st0:
                    continue
                reads = [r for r in ast.walk(st.value) if (isinstance(r, ast.Name) and r.id == x)
                         or (isinstance(r, ast.Subscript) and _const_str(r.slice) == x)]
                lag = acc[x] + (min(self._lag_above(r, None, reach) for r in reads) if reads else 0)
                for y in wn | wk:
                    if y not in acc or lag < acc[y]:
                        acc[y] = lag
                        todo.append(y)
        mi = self._model_inputs()
        classes = ("signal",) if self._pnl_input(n.args[0] if n.args else None) else ("signal", "pnl", "exec")
        for k, lag in acc.items():
            r = classify(k)
            if lag < R and (r in classes or (k in mi and r not in ("label", "eval", "pnl"))):
                return "high"
        mr = self._v6_mult_reads()
        for k, lag in acc.items():
            for rd, side in mr.get(k, ()):
                if lag + self._lag_above(rd, side, reach) < R + 1:
                    return "high"
        return "possible"

    def _v6_reach(self, names: set[str], keys: set[str], after: int = 0, cap: int = 400, reach=None) -> set[str]:
        """Names and keys computed (directly or through other assignments) from `names`/`keys` after line `after`.
        With `reach` (bars of look-ahead), a statement that reads them only shifted by at least that many bars is a
        barrier: the value there is causal."""
        rows, by_name, by_key = self._v6_index()
        names, keys = set(names), set(keys)
        todo = [("n", x) for x in names] + [("k", x) for x in keys]
        seen: set[int] = set()
        while todo and len(seen) < cap:
            kind, x = todo.pop()
            for i in (by_name if kind == "n" else by_key).get(x, ()):
                if i in seen:
                    continue
                st, wn, wk = rows[i]
                if st.lineno <= after:
                    continue
                if reach is not None and self._lagged_read(st.value, x, reach):
                    continue
                seen.add(i)
                for y in wn - names:
                    names.add(y)
                    todo.append(("n", y))
                for y in wk - keys:
                    keys.add(y)
                    todo.append(("k", y))
        return names | keys

    def _model_inputs(self) -> set[str]:
        """Names and column strings passed to fit/predict (also through a list constant of column names)."""
        if not hasattr(self, "_mi"):
            out: set[str] = set()
            for c in self._all:
                if isinstance(c, ast.Call) and _attr(c) in FIT_ATTRS:
                    for a in list(c.args[:1]) + [k.value for k in c.keywords if k.arg in ("X", "x")]:
                        for x in ast.walk(a):
                            if isinstance(x, ast.Name):
                                out.add(x.id)
                                for st in self._assigns_of(x.id):
                                    if isinstance(st.value, (ast.List, ast.Tuple)):
                                        out |= {_const_str(e) for e in st.value.elts if _const_str(e)}
                            elif _const_str(x):
                                out.add(_const_str(x))
            self._mi = out
        return self._mi

    def _pnl_input(self, node: ast.AST | None) -> bool:
        """The series being smoothed or aggregated is itself a PnL, equity or evaluation series."""
        if node is None:
            return False
        ks = [k for k, _l in _scan_expr(node).refs]
        ks += [_const_str(x) for x in ast.walk(node) if _const_str(x)]
        return any(classify(k) in ("pnl", "eval") for k in ks if k)

    def _v6_decides(self, reached: set[str], report_src: bool = False) -> bool:
        """Something in `reached` is a position, signal, fill or exit price, PnL or model input, or is multiplied
        with a return as a position. When the input itself is PnL (`report_src`: a daily PnL total, a smoothed
        equity curve), a PnL-named result is a report, not a decision (rt14 F4, rt15 B1)."""
        mi = self._model_inputs()
        classes = ("signal",) if report_src else ("signal", "pnl", "exec")
        for k in reached:
            r = classify(k)
            if r in classes:
                return True
            if k in mi and r not in ("label", "eval", "pnl"):
                return True
        return self._v6_booked(reached)

    def _v6_mults(self) -> list[tuple[set[str], bool, set[str], bool]]:
        """Every product in the file as (refs of a side, that side is a return) pairs, built once (F5)."""
        if not hasattr(self, "_v6m"):
            def is_ret(side: ast.AST) -> bool:
                return any(classify(k) == "return" or (self.info.get(k) is not None and self.info[k].is_return)
                           for k, _l in _scan_expr(side).refs) or any(
                    isinstance(c, ast.Call) and _attr(c) in ("pct_change", "diff") for c in ast.walk(side))
            out = []
            for m in self._all:
                if isinstance(m, ast.BinOp) and isinstance(m.op, ast.Mult):
                    a = {k for k, _l in _scan_expr(m.left).refs}
                    b = {k for k, _l in _scan_expr(m.right).refs}
                    out.append((a, is_ret(m.left), b, is_ret(m.right)))
            self._v6m = out
        return self._v6m

    def _v6_booked(self, reached: set[str]) -> bool:
        """Something in `reached` is multiplied with a return (a position booked against it)."""
        return any((a & reached and rb) or (b & reached and ra) for a, ra, b, rb in self._v6_mults())

    def _v6_backtested(self, reached: set[str]) -> bool:
        """Something in `reached` is booked as a strategy return: a PnL-named key, or a product with a return."""
        return any(classify(k) == "pnl" for k in reached) or self._v6_booked(reached)

    def _v6_start(self, n: ast.AST, depth: int = 0) -> tuple[set[str], set[str], int]:
        """What holds the value of expression `n`: its assignment target, or, when it is returned by a helper, the
        targets at the helper's call sites."""
        st = self._stmt_of(n)
        names: set[str] = set()
        keys: set[str] = set()
        after = getattr(st, "lineno", 0)
        if isinstance(st, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            wn, wk, _rn, _rk = self._v6_writes(st)
            names |= wn
            keys |= wk
        fd = self.inside(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        if fd is not None and depth < 2:
            local = self._v6_reach(names, keys, after) if (names or keys) else set()
            returned = isinstance(st, ast.Return) or any(
                isinstance(r, ast.Return) and r.value is not None and self.names_in(r.value) & local
                for r in ast.walk(fd))
            if returned:
                after = 0
                if not hasattr(self, "_calls_by_fn"):
                    self._calls_by_fn: dict[str, list[ast.Call]] = {}
                    for c in self._all:
                        if isinstance(c, ast.Call) and _func_name(c):
                            self._calls_by_fn.setdefault(_func_name(c), []).append(c)
                sites = list(self._calls_by_fn.get(fd.name, ()))
                sites += [c for c, _a in self._v7_call_sites(fd.name) if _attr(c) in self._V7_APPLIERS
                          and self._v7_call_ok(c)]  # v0.7: closes.apply(helper), df.pipe(helper)
                for c in sites:
                    if c is not n:
                        if isinstance(c.func, ast.Attribute) and not (isinstance(c.func.value, ast.Name)
                                                                      and c.func.value.id in ("self", "cls")) \
                                 and _attr(c) not in self._V7_APPLIERS:
                            continue
                        n2, k2, _a = self._v6_start(c, depth + 1)
                        names |= n2
                        keys |= k2
        return names, keys, after

    def _v6_flow_sev(self, n: ast.AST, reach=None) -> str | None:
        """'high' when the value of `n` reaches a position, signal or model input (or is booked against a return);
        'possible' otherwise; None when it is a label or an evaluation (where future data belongs). With `reach`, a
        shift of at least that many bars on the way makes the value causal."""
        names, keys, after = self._v6_start(n)
        if any(classify(k) in ("label", "eval") for k in keys) and not any(classify(k) in ("signal", "pnl", "feature")
                                                                             for k in keys):
            return None
        reached = self._v6_reach(names, keys, after, reach=reach) | keys
        src = n.args[0] if isinstance(n, ast.Call) and n.args else (
            n.func.value if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) else None)
        return "high" if self._v6_decides(reached, report_src=self._pnl_input(src)) else "possible"

    # -- LA02 (v0.6): two-sided transforms ------------------------------------------------------------------
    _TWO_SIDED = {"filtfilt": "runs the filter forward and then backward over the whole series",
                  "sosfiltfilt": "runs the filter forward and then backward over the whole series",
                  "hpfilter": "solves for the trend using every observation, later ones included",
                  "savgol_filter": "fits each point with a window centred on it",
                  "gaussian_filter1d": "averages each point with a window centred on it",
                  "uniform_filter1d": "averages each point with a window centred on it",
                  "argrelextrema": "marks a turning point only once the bars after it are known",
                  "argrelmin": "marks a turning point only once the bars after it are known",
                  "argrelmax": "marks a turning point only once the bars after it are known",
                  "find_peaks": "marks a peak only once the bars after it are known",
                  "find_peaks_cwt": "marks a peak only once the bars after it are known",
                  "seasonal_decompose": "estimates the trend with a centred moving average",
                  "STL": "fits the trend with a loess smoother over bars on both sides",
                  "MSTL": "fits the trend with a loess smoother over bars on both sides",
                  "lowess": "fits each point with a local regression over bars on both sides",
                  "bkfilter": "is a symmetric band-pass filter (it uses leads as well as lags)",
                  "cffilter": "uses the whole sample, later observations included"}

    def _const_true(self, c: ast.AST) -> bool:
        """`cfg.center` / `center` whose every visible definition is the constant True (a class field, a module name
        or a parameter default)."""
        vals = []
        if isinstance(c, ast.Attribute):
            for cd in self._all:
                if isinstance(cd, ast.ClassDef):
                    for s in cd.body:
                        tg = s.targets if isinstance(s, ast.Assign) else ([s.target] if isinstance(s, ast.AnnAssign) else [])
                        if any(isinstance(t, ast.Name) and t.id == c.attr for t in tg) and getattr(s, "value", None) is not None:
                            vals.append(s.value)
        elif isinstance(c, ast.Name):
            vals = [st.value for st in self._assigns_of(c.id)]
            fd = self.inside(c, (ast.FunctionDef, ast.AsyncFunctionDef))
            if fd is not None:
                a = fd.args
                pos = a.posonlyargs + a.args
                for p, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
                    if p.arg == c.id:
                        vals.append(d)
                for p, d in zip(a.kwonlyargs, a.kw_defaults):
                    if p.arg == c.id and d is not None:
                        vals.append(d)
        return bool(vals) and all(isinstance(v, ast.Constant) and v.value is True for v in vals)

    def _two_sided(self, n: ast.Call) -> str | None:
        fn = _func_name(n)
        if fn in self._TWO_SIDED and fn not in self._funcs:
            if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id in ("self", "cls"):
                return None
            if fn == "uniform_filter1d" and _kw(n, "origin") is not None:
                return None
            return f"`{fn}(...)` {self._TWO_SIDED[fn]}"
        if _np_call(n, "gradient"):
            return "`np.gradient(...)` takes central differences, (x[t+1] - x[t-1]) / 2, so each value uses the next bar"
        if fn in ("convolve", "fftconvolve", "oaconvolve") and _const_str(
                _kw(n, "mode") or (n.args[2] if len(n.args) > 2 else None)) == "same":
            return f"`{fn}(..., mode='same')` centres the kernel on each point, so it averages bars on both sides"
        if _attr(n) == "smooth" and isinstance(n.func.value, ast.Name):
            r = n.func.value.id
            c = self._assigned_call(r)
            if (c is not None and "Kalman" in (_func_name(c) or "")) or _tokens(r) & {"kf", "kalman"}:
                return (f"`{r}.smooth(...)` is the Kalman *smoother* (forward-backward), so each state uses later "
                        "observations; `.filter(...)` is the causal version")
        if _attr(n) == "rolling":
            c = _kw(n, "center")
            if c is not None and not isinstance(c, ast.Constant) and self._const_true(c):
                return "`rolling(..., center=...)` is centred here (the setting resolves to True), so it uses the next half-window"
        return None

    def _two_sided_reach(self, n: ast.Call):
        """How many bars ahead the transform reads, when that is bounded: 1 for np.gradient, `order` for
        argrelextrema, `distance` (else 1) for find_peaks, half the window for savgol_filter. None = unbounded."""
        fn = _func_name(n)
        if _np_call(n, "gradient"):
            return 1
        if fn in ("argrelextrema", "argrelmin", "argrelmax"):
            o = _kw(n, "order")
            return 1 if o is None else (int(_numc(o)) if _numc(o) is not None else o)
        if fn == "find_peaks":
            if any(_kw(n, k) is not None for k in ("prominence", "width", "wlen", "plateau_size")):
                return None
            d = _kw(n, "distance")
            return 1 if d is None else (int(_numc(d)) if _numc(d) is not None else d)
        if fn == "savgol_filter":
            w = _kw(n, "window_length") or (n.args[1] if len(n.args) > 1 else None)
            return int(_numc(w)) // 2 if w is not None and _numc(w) is not None else None
        return None

    def _in_window_apply(self, n: ast.AST) -> bool:
        """`n` runs inside a function passed to `rolling(...).apply` / `expanding(...).apply`: it only sees the window
        ending at each bar."""
        f = self.inside(n, (ast.Lambda, ast.FunctionDef))
        if f is None:
            return False
        if isinstance(f, ast.Lambda):
            uses = [f]
        else:
            if not hasattr(self, "_loads_by_name"):
                self._loads_by_name: dict[str, list[ast.Name]] = {}
                for x in self._all:
                    if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                        self._loads_by_name.setdefault(x.id, []).append(x)
            uses = self._loads_by_name.get(f.name, [])
        for u in uses:
            p = self.parents.get(u)
            if (isinstance(p, ast.Call) and _attr(p) in ("apply", "agg", "aggregate")
                    and (any(a is u for a in p.args) or any(k.value is u for k in p.keywords))
                    and isinstance(p.func.value, ast.Call) and _attr(p.func.value) in ("rolling", "expanding")):
                continue
            return False
        return bool(uses)

    def _rule_two_sided(self, n: ast.Call) -> None:
        what = self._two_sided(n)
        if what is None:
            return
        loop = self.inside(n, (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.GeneratorExp))
        if loop is not None and n.args and any(isinstance(s, ast.Subscript) and isinstance(s.slice, ast.Slice)
                                               for s in ast.walk(n.args[0])):
            return  # re-run on data up to each date inside a loop: a one-sided (expanding) use
        if self._in_window_apply(n):
            return  # rt14 q02: inside rolling(...).apply the call only sees the trailing window
        reach = self._two_sided_reach(n)
        sev = self._v6_flow_sev(n) if reach is None else self._v6_bounded_sev(n, reach)
        if sev is None:
            return
        tail = ("Use a causal version (`lfilter`, a trailing window, a one-sided filter re-run on data up to each date)."
                if sev == "high" else
                "If this only feeds a chart or a description, ignore this; if it feeds a signal, use a causal version.")
        self.add("LA02", sev, n, f"{what}: each output value uses bars after it. {tail}")

    # -- LA13 (v0.6): backtrader positive line index ----------------------------------------------------------
    _BT_IND_MODS = {"ind", "indicators", "talib"}

    def _bt_line(self, b: ast.AST, cd: ast.ClassDef) -> bool:
        """`b` is a backtrader line object: rooted at self.data / self.dataN / self.datas[i] (then a field), or an
        attribute of `self` assigned in the class from bt.ind.* / bt.indicators.* / bt.talib.* or from data lines.
        Chains through `p`/`params`, and attributes assigned from literals, are not lines (rt14 F1)."""
        chain: list[str] = []
        x = b
        while isinstance(x, (ast.Attribute, ast.Subscript)):
            if isinstance(x, ast.Attribute):
                chain.append(x.attr)
            elif isinstance(x.value, ast.Attribute) and x.value.attr == "datas":
                pass  # self.datas[0].close: the feed index, then a line
            else:
                return False
            x = x.value
        if not (isinstance(x, ast.Name) and x.id == "self" and chain):
            return False
        chain.reverse()  # outermost first: ["data", "close"]
        if set(chain) & {"p", "params"}:
            return False
        first = chain[0]
        if first in ("datas",):
            return len(chain) >= 2
        if first == "data" or re.fullmatch(r"data\d+", first):
            return True
        if first == "lines" and len(chain) >= 2:
            return True
        return first in self._bt_line_attrs(cd)

    def _bt_aliases(self) -> tuple[set[str], set[str], set[str]]:
        """From the imports: names bound to backtrader itself, to its indicator modules, and indicator classes
        imported directly (`import backtrader.indicators as btind`, `from backtrader import indicators as ind`,
        `from backtrader.indicators import RSI`)."""
        if not hasattr(self, "_bta"):
            bt_mods, ind_mods, ind_names = set(), set(), set()
            for n in self._all:
                if isinstance(n, ast.Import):
                    for a in n.names:
                        if a.name == "backtrader":
                            bt_mods.add(a.asname or "backtrader")
                        elif a.name.startswith("backtrader.") and a.name.split(".")[1] in ("indicators", "ind", "talib"):
                            if a.asname:
                                ind_mods.add(a.asname)
                elif isinstance(n, ast.ImportFrom) and n.module:
                    if n.module == "backtrader":
                        for a in n.names:
                            if a.name in ("indicators", "ind", "talib"):
                                ind_mods.add(a.asname or a.name)
                    elif n.module.startswith("backtrader.") and n.module.split(".")[1] in ("indicators", "ind", "talib"):
                        ind_names |= {a.asname or a.name for a in n.names if a.name != "*"}
            self._bta = (bt_mods, ind_mods, ind_names)
        return self._bta

    def _bt_indicator_call(self, c: ast.AST) -> bool:
        if not isinstance(c, ast.Call):
            return False
        bt_mods, ind_mods, ind_names = self._bt_aliases()
        f = c.func
        if isinstance(f, ast.Name):
            return f.id in ind_names
        if isinstance(f, ast.Attribute):
            v = f.value
            if isinstance(v, ast.Name) and v.id in ind_mods:
                return True  # btind.RSI(...)
            if isinstance(v, ast.Attribute) and v.attr in self._BT_IND_MODS and isinstance(v.value, ast.Name) and (
                    v.value.id in bt_mods or v.value.id in ("bt", "backtrader")):
                return True  # bt.ind.SMA(...)
        return False

    def _bt_line_expr(self, v: ast.AST) -> bool:
        """`v` is a line: a data line (self.data.close, self.datas[0].close), arithmetic of lines, or an indicator."""
        if self._bt_indicator_call(v):
            return True
        if isinstance(v, (ast.BinOp, ast.UnaryOp, ast.Compare)):
            parts = [x for x in ast.iter_child_nodes(v) if isinstance(x, ast.expr)]
            return any(self._bt_line_expr(x) for x in parts) and all(
                self._bt_line_expr(x) or isinstance(x, ast.Constant) or (isinstance(x, ast.Name) and x.id not in ("self",))
                for x in parts)
        chain, x = [], v
        while isinstance(x, (ast.Attribute, ast.Subscript)):
            if isinstance(x, ast.Attribute):
                chain.append(x.attr)
            elif not (isinstance(x.value, ast.Attribute) and x.value.attr == "datas"):
                return False
            x = x.value
        if not (isinstance(x, ast.Name) and x.id == "self" and chain):
            return False
        chain.reverse()
        return (chain[0] == "datas" and len(chain) >= 2) or chain[0] == "data" or bool(re.fullmatch(r"data\d+", chain[0]))

    def _bt_line_attrs(self, cd: ast.ClassDef) -> set[str]:
        out: set[str] = set()
        for st in ast.walk(cd):
            if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Attribute)
                    and isinstance(st.targets[0].value, ast.Name) and st.targets[0].value.id == "self"):
                continue
            if self._bt_line_expr(st.value):  # not sorted(self.datas), a dict of indicators or a literal
                out.add(st.targets[0].attr)
        return out

    def _bt_local_line(self, name: ast.Name) -> bool:
        """A local name assigned (in the same method) from a line: o = self.data.open; o[1]."""
        fd = self.inside(name, (ast.FunctionDef, ast.AsyncFunctionDef))
        if fd is None:
            return False
        vals = [s.value for s in ast.walk(fd) if isinstance(s, ast.Assign) and len(s.targets) == 1
                and isinstance(s.targets[0], ast.Name) and s.targets[0].id == name.id]
        return bool(vals) and all(self._bt_line_expr(v) for v in vals)

    def _rule_bt_future(self) -> None:
        if "backtrader" not in self.imports:
            return
        for cd in self._all:
            if not (isinstance(cd, ast.ClassDef) and any(
                    (isinstance(b, ast.Attribute) and b.attr in ("Strategy", "Indicator", "SignalStrategy"))
                    or (isinstance(b, ast.Name) and b.id in ("Strategy", "Indicator", "SignalStrategy")) for b in cd.bases)):
                continue
            for sub in ast.walk(cd):
                if not (isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Load)):
                    continue
                v = _num(sub.slice)
                if v is None or v <= 0 or v != int(v):
                    continue
                if not (self._bt_line(sub.value, cd) or (isinstance(sub.value, ast.Name) and self._bt_local_line(sub.value))):
                    continue  # parameters (self.p.periods[1]), lists of levels, the list of feeds
                self.add("LA13", "high", sub,
                         f"In backtrader `[{int(v)}]` reads {int(v)} bar(s) *ahead* (`[0]` is the current bar, `[-1]` the "
                         "previous one), so this decision uses a bar that hasn't happened yet. Use `[0]` or negative indices.")

    # -- LA14 (v0.6): pickers keyed on a test-set metric ----------------------------------------------------
    def _metric_call(self, c: ast.AST) -> bool:
        if not isinstance(c, ast.Call):
            return False
        fn = _attr(c) or _func_name(c) or ""
        return fn in self._SCORE_FUNCS or bool(re.search(r"(_score|_error|_loss)$", fn)) or bool(
            _tokens(fn) & {"sharpe", "accuracy", "auc", "f1", "score"})

    def _test_rows(self, node: ast.AST) -> bool:
        """`node` reads test data: a test-named name, or rows selected by `~train_mask`."""
        if self._refs_test(node):
            return True
        return any(isinstance(u, ast.UnaryOp) and isinstance(u.op, ast.Invert) and isinstance(u.operand, ast.Name)
                   and _tokens(u.operand.id) & {"train", "tr", "insample", "is", "fit"} for u in ast.walk(node))

    def _rule_test_pickers(self) -> None:
        stored_test: dict[str, ast.AST] = {}
        for st in self._all:
            if (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Subscript)
                    and isinstance(st.targets[0].value, ast.Name) and self.inside(st, (ast.For,)) is not None):
                d = st.targets[0].value.id
                if _tokens(d) & VAL_SET_T or self._selects_on_cv(self.inside(st, (ast.For,))):
                    continue  # results[k] = (cv_score, test_score): compared on the CV score
                if isinstance(st.value, (ast.Tuple, ast.List, ast.Dict)):
                    continue  # several numbers per candidate: which one decides the pick is not visible here
                if any(self._metric_call(c) and self._test_rows(c) for c in ast.walk(st.value)) or (
                        _tokens(d) & TEST_SET_T and any(self._metric_call(c) for c in ast.walk(st.value))):
                    stored_test.setdefault(d, st)
        for c in self._all:
            if not (isinstance(c, ast.Call) and _func_name(c) in ("max", "min", "sorted") and c.args):
                continue
            key = _kw(c, "key")
            if isinstance(key, ast.Lambda):
                if any(self._metric_call(m) and self._test_rows(m) for m in ast.walk(key.body)) and not any(
                        _tokens(x.id) & VAL_SET_T for x in ast.walk(key.body) if isinstance(x, ast.Name)):
                    self.add("LA14", "high", c,
                             "This picks the candidate with the best score *on the test set* (the `key` scores test "
                             "data), so the reported test result is selected on the test data. Choose on a validation "
                             "window, then score the winner once on the test set.")
                    continue
            arg = c.args[0]
            d = arg.id if isinstance(arg, ast.Name) else (
                arg.func.value.id if isinstance(arg, ast.Call) and _attr(arg) in ("items", "keys")
                and isinstance(arg.func.value, ast.Name) else None)
            if d in stored_test and c.lineno > stored_test[d].lineno and key is not None:
                loop = stored_test[d]
                while self.inside(loop, (ast.For,)) is not None:
                    loop = self.inside(loop, (ast.For,))
                prior = next((f for f in self.findings if f.rule_id == "LA14" and f.line == loop.lineno and not f.same_as), None)
                if prior is not None:  # the selection loop is already reported: show the pick, scored once
                    self.add("LA14", prior.severity, c, f"The candidate with the best *test-set* score in `{d}` is "
                                                        f"kept here (the loop on line {loop.lineno}).", same_as=loop.lineno)
                    continue
                self.add("LA14", "high", c,
                         f"`{d}` holds each candidate's *test-set* score (line {stored_test[d].lineno}) and this keeps "
                         "the best one: the reported test result is selected on the test data. Choose on a "
                         "validation window, then score the winner once on the test set.")
        for c in self._all:  # study.optimize(objective, ...) with an objective that scores the test data
            if not (isinstance(c, ast.Call) and _attr(c) == "optimize" and c.args and isinstance(c.args[0], ast.Name)):
                continue
            fd = self._funcs.get(c.args[0].id)
            if fd is None:
                continue
            tn = self._test_names()
            for r in ast.walk(fd):
                if isinstance(r, ast.Return) and r.value is not None and self.names_in(r.value) & tn and not any(
                        _tokens(x) & VAL_SET_T for x in self.names_in(r.value)):
                    self.add("LA14", "high", r,
                             f"The objective that `{_attr(c)}` maximizes returns a score on test data, so the tuner picks "
                             "the parameters that do best on the test set. Optimize on training/validation data and "
                             "evaluate the best parameters once on the test set.")
                    break

    # -- LA06 (v0.6): time-series CV and sampling -------------------------------------------------------------
    def _label_lead(self) -> int:
        """The longest known horizon (in bars) of a forward-looking label in the file; 0 if none is known."""
        best = 0
        for k, ki in self.info.items():
            if ki.forward and ki.lead and ki.lead < 10 ** 5 and (classify(k) == "label" or _tokens(k) & {"y", "target"}):
                best = max(best, ki.lead)
        for k, ld in self._lead_seen.items():  # v0.7: also a label column later rebound (`y = df["y"]`)
            if classify(k) == "label" or _tokens(k) & {"y", "target"}:
                best = max(best, ld)
        return best

    def _rule_cv_sampling(self) -> None:
        toks_all = None
        for n in self._all:
            if not isinstance(n, ast.Call):
                continue
            fn = _func_name(n)
            if fn == "TimeSeriesSplit":
                gap = _kw(n, "gap")
                if gap is not None and _num(gap) != 0:
                    continue
                lead = self._label_lead()
                if lead < 2:
                    continue
                if toks_all is None:
                    toks_all = self._toks_of(self.tree)
                if toks_all & {"purge", "purged", "embargo", "gap"}:
                    continue
                loop = self.inside(n, (ast.For,))
                if loop is not None and any(isinstance(s, ast.Slice) and isinstance(s.upper, (ast.UnaryOp, ast.BinOp))
                                            for s in ast.walk(loop)):
                    continue  # the fold is trimmed by hand
                self.add("LA06", "medium", n,
                         f"`TimeSeriesSplit` without `gap` while the label looks {lead} bars ahead: the last {lead - 1} "
                         "training rows of each fold have labels that overlap the test fold. Pass "
                         f"`gap={lead}` (or purge those rows).")
            elif _attr(n) == "fit_resample" and n.args:
                if self._fit_subset(n.args[0]) or (_key(_chain_base(n.args[0]))[0] and _tokens(
                        _key(_chain_base(n.args[0]))[0]) & SPLIT_T):
                    continue
                st = self._stmt_of(n)
                if not isinstance(st, ast.Assign):
                    continue
                outs, _wk, _r1, _r2 = self._v6_writes(st)
                later = False
                for c in self._all:
                    if getattr(c, "lineno", 0) <= st.lineno:
                        continue
                    if isinstance(c, ast.Subscript) and isinstance(c.value, ast.Name) and c.value.id in outs and isinstance(
                            c.slice, ast.Slice):
                        later = True
                    elif isinstance(c, ast.Call) and (_func_name(c) in ("train_test_split", "cross_val_score", "cross_validate",
                                                                        "cross_val_predict")) and self.names_in(c) & outs:
                        later = True
                    if later:
                        break
                if later:
                    self.add("LA06", "high", n,
                             f"`{_attr(n)}` creates synthetic rows from neighbours across the whole sample before the "
                             "split, so test rows shape the training data (and synthetic copies of them land in both "
                             "sets). Resample the training rows only, after the split.")
            elif _attr(n) == "sample" and _num(_kw(n, "frac")) == 1:
                st = self._stmt_of(n)
                if not isinstance(st, ast.Assign):
                    continue
                outs, _wk, _r1, _r2 = self._v6_writes(st)
                used = any(isinstance(c, ast.Call) and _attr(c) == "split" and getattr(c, "lineno", 0) > st.lineno
                           and self.names_in(c) & outs and (self.ts_context or self._tss_receiver(c.func.value))
                           for c in self._all)
                if used:
                    self.add("LA06", "medium", n,
                             "`.sample(frac=1)` shuffles the time-ordered rows, and a split then cuts them by position, so "
                             "training folds contain rows from after the test rows. Keep the rows in time order.")
            elif fn == "random_split" and self.ts_context:
                self.add("LA06", "medium", n,
                         "`random_split` assigns time-ordered rows to train and test at random, so training contains "
                         "rows from after the test rows. Split by date.")

    def _tss_receiver(self, r: ast.AST) -> bool:
        """`r` is TimeSeriesSplit(...) or a name assigned from it."""
        if isinstance(r, ast.Name):
            r = self._assigned_call(r.id)
        return isinstance(r, ast.Call) and _func_name(r) == "TimeSeriesSplit"

    # -- SB01 (v0.6): universes fixed with hindsight --------------------------------------------------------
    @staticmethod
    def _wide_price_name(k: str | None) -> bool:
        return bool(k) and bool(_tokens(k) & (PRICE_T | RETURN_T | {"closes", "prices", "rets", "returns", "px", "adjclose"}))

    def _last_row(self, x: ast.AST) -> ast.AST | None:
        """`F.iloc[-1]` / `F.tail(1)` (the last row of F): return F."""
        if isinstance(x, ast.Subscript) and isinstance(x.value, ast.Attribute) and x.value.attr == "iloc" and _num(x.slice) == -1:
            return x.value.value
        if isinstance(x, ast.Call) and _attr(x) == "tail" and x.args and _num(x.args[0]) == 1:
            return x.func.value
        return None

    def _first_row(self, x: ast.AST) -> ast.AST | None:
        if isinstance(x, ast.Subscript) and isinstance(x.value, ast.Attribute) and x.value.attr == "iloc" and _num(x.slice) == 0:
            return x.value.value
        return None

    def _selects_backtest_columns(self, names: set[str], after: int) -> bool:
        """A name in `names` later selects the columns (or members) of a frame that is run through time-series
        operations: px[top], rets.loc[:, names], px.columns.intersection(live), df[df.ticker.isin(live)]."""
        rows, by_name, _bk = self._v6_index()
        for i in sorted({i for nm in names for i in by_name.get(nm, ())}):
            st = rows[i][0]
            if st.lineno <= after or not isinstance(st, ast.Assign):
                continue
            hit = False
            for x in ast.walk(st.value):
                if isinstance(x, ast.Subscript) and self.names_in(x.slice) & names:
                    hit = True
                elif isinstance(x, ast.Call) and _attr(x) in ("intersection", "isin", "reindex", "filter") and self.names_in(x) & names:
                    hit = True
            if not hit:
                continue
            if any(isinstance(c, ast.Call) and _attr(c) in TS_OPS for c in ast.walk(st.value)):
                return True  # basket = px[top].pct_change(): the selected names' history is the backtest
            wn, wk, _a, _b = self._v6_writes(st)
            reached = self._v6_reach(wn, wk, st.lineno) | wk
            later = {j for k in reached for j in by_name.get(k, ())}
            if any(rows[j][0].lineno > after and self._has_ts_op(rows[j][0].value) for j in later) or self._v6_decides(reached):
                return True
        return False

    def _has_ts_op(self, v: ast.AST, ops=TS_OPS) -> bool:
        if not hasattr(self, "_tsop_memo"):
            self._tsop_memo: dict[tuple[int, int], bool] = {}
        key = (id(v), id(ops))
        if key not in self._tsop_memo:
            self._tsop_memo[key] = any(isinstance(c, ast.Call) and _attr(c) in ops for c in ast.walk(v))
        return self._tsop_memo[key]

    def _reads_under_op(self, v: ast.AST, names: set[str]) -> bool:
        """`v` applies a return/window operation to one of `names` (rN = pxN.pct_change())."""
        return any(isinstance(c, ast.Call) and _attr(c) in self._RET_OPS and self.names_in(c.func.value) & names
                   for c in ast.walk(v))

    _RET_OPS = frozenset({"pct_change", "rolling", "shift", "diff", "rank"})

    def _rule_survivors_v6(self) -> None:
        sb = ("The universe is fixed with knowledge of the end of the sample, so names that were delisted, acquired or "
              "went bankrupt are left out of the whole backtest (survivorship bias). Decide membership on each date "
              "with data available then, and keep dead names with their final returns.")
        for n in self._all:
            if isinstance(n, ast.Call) and _attr(n) == "dropna":
                ax = _kw(n, "axis")
                if not (_num(ax) == 1 or _const_str(ax) == "columns"):
                    continue
                how = _const_str(_kw(n, "how"))
                thresh = _kw(n, "thresh")
                if how == "all":
                    continue
                if thresh is not None and not (isinstance(thresh, ast.Call) and _func_name(thresh) == "len"):
                    continue
                k = _key(_chain_base(n.func.value))[0]
                if not self._wide_price_name(k):
                    continue
                st = self._stmt_of(n)
                wn, wk, _a, _b = self._v6_writes(st) if isinstance(st, ast.Assign) else (set(), set(), 0, 0)
                if not wn:
                    continue
                rows, by_name, _bk = self._v6_index()
                if not any(rows[j][0].lineno > st.lineno and self._reads_under_op(rows[j][0].value, wn)
                           for nm in wn for j in by_name.get(nm, ())):
                    continue
                self.add("SB01", "medium", n,
                         f"`dropna(axis=1)` on `{k}` keeps only the columns with a value on *every* date, i.e. the names "
                         "that traded for the whole sample. " + sb)
                continue
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "loc" and isinstance(
                    n.slice, ast.Tuple) and len(n.slice.elts) == 2:
                sel = n.slice.elts[1]
                if isinstance(sel, ast.Call) and _attr(sel) in ("notna", "notnull") and self._last_row(sel.func.value) is not None:
                    self.add("SB01", "medium", n,
                             "The columns are kept by `iloc[-1].notna()`: only names that still have a price on the "
                             "last date survive. " + sb)
                continue
            if isinstance(n, ast.Call) and _attr(n) in ("nlargest", "nsmallest", "sort_values", "head"):
                recv = n.func.value
                if _attr(n) == "head" and isinstance(recv, ast.Call) and _attr(recv) == "sort_values":
                    recv = recv.func.value
                src = self._last_row(recv)
                why = None
                if src is not None:
                    sk = _key(_chain_base(src))[0]
                    if classify(sk) in ("signal", "feature", "eval", "label") or (sk and _tokens(sk) & PRED_T):
                        continue  # today's top scores for live picks: not a historical universe
                    why = f"The universe is the top names on the *last* row of `{sk or 'the frame'}`. "
                elif isinstance(recv, ast.Name):
                    v = self._assigned_value(recv.id, n.lineno)
                    if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Sub) and _num(v.right) is not None:
                        v = v.left  # px.iloc[-1] / px.iloc[0] - 1
                    if isinstance(v, ast.BinOp) and isinstance(v.op, (ast.Div, ast.Sub)):
                        a = v.left
                        if (self._last_row(a) is not None and self._first_row(v.right) is not None
                                and ast.dump(self._last_row(a)) == ast.dump(self._first_row(v.right))):
                            why = f"`{recv.id}` is each name's return over the *whole* sample, and the best of them are kept. "
                if why is None:
                    continue
                st = self._stmt_of(n)
                if not isinstance(st, ast.Assign):
                    continue
                wn, _wk, _a, _b = self._v6_writes(st)
                if wn and self._selects_backtest_columns(wn, st.lineno):
                    self.add("SB01", "medium", n, why + sb)
                continue
            if isinstance(n, ast.Call) and _attr(n) in ("isna", "isnull") and isinstance(n.func.value, ast.Subscript):
                col = _const_str(n.func.value.slice)
                if not (col and _tokens(col) & {"end", "removed", "delisted", "exit", "until", "to", "out", "drop", "dropped"}):
                    continue
                st = self._stmt_of(n)
                if not isinstance(st, ast.Assign) or self.inside(n, (ast.BoolOp,)) is not None or any(
                        isinstance(b, ast.BinOp) and isinstance(b.op, (ast.BitOr, ast.BitAnd)) for b in ast.walk(st.value)):
                    continue  # (start <= d) & (end.isna() | (end > d)): alive on date d, the point-in-time test
                wn, _wk, _a, _b = self._v6_writes(st)
                if wn and self._selects_backtest_columns(wn, st.lineno):
                    self.add("SB01", "medium", n,
                             f"`{col}` is empty only for *current* members, so this keeps today's members for the whole "
                             "history. " + sb)
                continue
            if (isinstance(n, ast.Call) and _attr(n) == "filter" and n.args and isinstance(n.args[0], ast.Lambda)
                    and isinstance(n.func.value, ast.Call) and _attr(n.func.value) == "groupby"):
                lam = n.args[0]
                param = lam.args.args[0].arg if lam.args.args else None
                for cmp in ast.walk(lam.body):
                    if not (isinstance(cmp, ast.Compare) and len(cmp.ops) == 1 and isinstance(cmp.ops[0], (ast.GtE, ast.Eq, ast.Gt))):
                        continue
                    left, right = cmp.left, cmp.comparators[0]
                    if not (isinstance(left, ast.Call) and _attr(left) == "max"):
                        continue
                    lb = _chain_base(left.func.value)
                    if not (isinstance(lb, (ast.Subscript, ast.Attribute)) and self._date_col(lb)):
                        continue
                    b = _chain_base(left.func.value)
                    while isinstance(b, (ast.Subscript, ast.Attribute)):
                        b = b.value
                    if not (isinstance(b, ast.Name) and b.id == param):
                        continue
                    fixed = isinstance(right, ast.Constant) or (isinstance(right, ast.Call) and _func_name(right) in (
                        "Timestamp", "to_datetime", "datetime", "date")) or (
                        isinstance(right, ast.Name) and isinstance(self._assigned_value(right.id, n.lineno), ast.Call)
                        and _func_name(self._assigned_value(right.id, n.lineno)) in ("Timestamp", "to_datetime", "datetime", "date"))
                    if fixed:
                        self.add("SB01", "medium", n,
                                 "This keeps only the names whose last date reaches the end date, i.e. the ones still "
                                 "trading at the end. " + sb)
                        break

    # -- LA12 (v0.6): period aggregates broadcast back to every bar of the period -------------------------------
    def _index_period_key(self, k: ast.AST, depth: int = 0) -> bool:
        """A groupby key derived from the time index: idx.date, idx.normalize(), idx.to_period(...), idx.floor('D'),
        pd.Grouper(freq=...), or a name assigned from one of these."""
        if isinstance(k, ast.Name) and depth < 2:
            v = self._assigned_value(k.id, getattr(k, "lineno", 10 ** 9))
            return v is not None and self._index_period_key(v, depth + 1)
        if (_const_str(k) or isinstance(k, ast.Subscript) and _const_str(k.slice)) and depth < 2:
            # v0.7: groupby("day") / groupby(df["day"]) with df["day"] = df.index.date (a day column of the index)
            col = _const_str(k) or _const_str(k.slice)
            sts = self._assign_by_key_v7().get(col, [])
            return bool(sts) and all(self._index_period_key(st.value, depth + 1) for st in sts)
        if isinstance(k, ast.Call) and _func_name(k) == "Grouper" and _kw(k, "freq") is not None and _kw(k, "key") is None:
            return True
        x = k
        if isinstance(x, ast.Call) and _attr(x) in ("normalize", "to_period", "floor", "date"):
            x = x.func.value
        elif isinstance(x, ast.Attribute) and x.attr in ("date", "week", "month", "year", "day", "dayofyear", "isocalendar"):
            x = x.value
        elif isinstance(x, (ast.List, ast.Tuple)) and x.elts:
            return any(self._index_period_key(e, depth + 1) for e in x.elts)
        else:
            return False
        if isinstance(x, ast.Attribute) and x.attr == "dt":
            x = x.value
        return isinstance(x, ast.Attribute) and x.attr == "index"

    def _rule_period_broadcast(self, n: ast.Call) -> None:
        if not (_attr(n) == "transform" and n.args):
            return
        f = _const_str(n.args[0])
        if f not in ("last", "max", "min", "sum", "mean", "std", "median", "var", "nunique", "prod", "idxmax", "idxmin"):
            return
        g = n.func.value
        if isinstance(g, ast.Subscript):
            g = g.value
        if isinstance(g, ast.Name):  # v0.7: grp = df.groupby("date"); grp["pv"].transform("sum")
            g = self._assigned_call(g.id) or g
        if not (isinstance(g, ast.Call) and _attr(g) == "groupby"):
            return
        key = _kw(g, "by") or (g.args[0] if g.args else None)
        if key is None or not self._index_period_key(key):
            return
        p = self.parents.get(n)
        if isinstance(p, ast.Attribute) and p.attr == "shift":
            return  # the period value moved to the next bar/period
        names, keys, after = self._v6_start(n)
        if any(classify(k) in ("label", "eval") for k in keys):
            return
        reached = self._v6_reach(names, keys, after) | keys
        report_src = self._pnl_input(n.func.value)
        sev = "medium" if self._v6_decides(reached, report_src=report_src) or (
            not report_src and self._v7_decides(reached)) else "possible"
        self.add("LA12", sev, n,
                 f"`groupby(<period of the index>).transform('{f}')` writes the period's final {f} onto *every* bar of "
                 "that period, including bars before it was known. Use a running statistic (`cummax`, `expanding`) "
                 "inside the period, or the previous period's value.")

    # -- LA07 (v0.6): fills at the bar's own low/high ---------------------------------------------------------
    @staticmethod
    def _extreme_name(k: str | None) -> str | None:
        t = _tokens(k) if k else frozenset()
        if t & {"low", "lows", "lo"}:
            return "low"
        if t & {"high", "highs", "hi"}:
            return "high"
        return None

    def _fill_target(self, k: str | None) -> bool:
        return bool(k) and bool(_tokens(k) & (ENTRY_T | {"fill", "fills", "exec", "execution", "fillpx"})) and not (
            _tokens(k) & {"stop", "sl", "tp", "target", "limit", "trail"})

    def _shifted_name(self, x: ast.AST, before: int) -> bool:
        """`x` (a name or a column) was defined as a shifted series (next_low = low.shift(-1)) or is known to the
        data flow as lagged or forward: it is not this bar's own value (rt14 F2)."""
        k = _key(_chain_base(x))[0]
        if not k:
            return False
        ki = self.info.get(k)
        if ki is not None and (ki.forward or ki.lagged):
            return True
        if isinstance(x, ast.Name):
            v = self._assigned_value(x.id, before)
            return v is not None and _chain_shift(v) != 0
        return False

    def _rule_extreme_fills(self) -> None:
        for st in self._stmts_sorted:
            if not (isinstance(st, ast.Assign) and len(st.targets) == 1):
                continue
            tk = _key(st.targets[0])[0]
            v = st.value
            if self._fill_target(tk):
                # entry_px = bar["low"] inside a bar loop, or df["fill_px"] = np.where(sig, df["low"], nan)
                cand = v.args[1] if _np_call(v, "where") and len(v.args) == 3 else v
                if isinstance(cand, (ast.Subscript, ast.Attribute, ast.Name)):
                    col = _const_str(cand.slice) if isinstance(cand, ast.Subscript) else _key(cand)[0]
                    side = self._extreme_name(col)
                    if side is None or _chain_shift(cand) != 0:
                        continue
                    in_loop = self.inside(st, (ast.For, ast.While)) is not None
                    if in_loop and isinstance(cand, ast.Subscript) and isinstance(cand.value, ast.Name) and _const_str(cand.slice):
                        self.add("LA07", "high", st,
                                 f"`{tk}` is the same bar's {side}: nobody can know, while the bar trades, which price will "
                                 f"be its {side}, so every fill is at the best price of the bar. Fill at the next bar's open, "
                                 "or at a limit price set beforehand.")
                    elif not in_loop and v is not cand:
                        self.add("LA07", "medium", st,
                                 f"`{tk}` fills at the bar's own {side}, the best price of the bar, which is known only "
                                 "when the bar closes. Fill at the open or at a limit set before the bar.")
                continue
            # bars["hit"] = bars["low"] <= bars["limit"], limit set from the same bar's close
            if isinstance(v, ast.Compare) and len(v.ops) == 1 and self.inside(st, (ast.For, ast.While)) is None:
                a, b = v.left, v.comparators[0]
                if isinstance(v.ops[0], (ast.GtE, ast.Gt)):
                    a, b = b, a
                    want = "high"
                else:
                    want = "low"
                lo, lim = (a, b)
                side = self._extreme_name(_key(_chain_base(lo))[0] if lo is not None else None)
                if side != want or _chain_shift(lo) != 0 or self._shifted_name(lo, st.lineno):
                    continue
                lk = _key(_chain_base(lim))[0]
                ki = self.info.get(lk) if lk else None
                if not (lk and _tokens(lk) & {"limit", "lmt", "bid", "entry", "order"} and ki is not None):
                    continue
                if ki.lagged or ki.forward or _chain_shift(lim) != 0:
                    continue
                lv = self._assigned_value(lk, st.lineno) if isinstance(lim, ast.Name) else None
                if lv is None:
                    if not hasattr(self, "_assign_by_key"):  # rt14 F5: indexed once, not rescanned per compare
                        self._assign_by_key: dict[str, list[ast.Assign]] = {}
                        for s2 in self._stmts_sorted:
                            if isinstance(s2, ast.Assign) and len(s2.targets) == 1 and _key(s2.targets[0])[0]:
                                self._assign_by_key.setdefault(_key(s2.targets[0])[0], []).append(s2)
                    for s2 in self._assign_by_key.get(lk, ()):
                        if s2.lineno >= st.lineno:
                            break
                        lv = s2.value
                if lv is None or not any(_is_close_name(k2) and lag <= 0 for k2, lag in _scan_expr(lv).refs) or any(
                        lag > 0 for _k2, lag in _scan_expr(lv).refs):
                    continue
                self.add("LA07", "high", v,
                         f"The limit `{lk}` is set from this bar's close, but the fill is checked against this same bar's "
                         f"{side}, which happened earlier in the bar. Set the limit at bar t and check it against bar t+1.")
        for n in self._all:  # vbt.Portfolio.from_signals(..., price=low)
            if isinstance(n, ast.Call) and _attr(n) in ("from_signals", "from_orders"):
                p = _kw(n, "price")
                if p is not None and _chain_shift(p) == 0:
                    side = self._extreme_name(_key(_chain_base(p))[0])
                    if side:
                        self.add("LA07", "medium", n,
                                 f"`price={_key(_chain_base(p))[0]}` fills every order at the bar's {side}, the best price of "
                                 "the bar, which no order can count on. Fill at the open or at a limit set beforehand.")

    # -- LA09 (v0.6): full-history per-asset statistics and pre-split winsorizing/imputing -----------------------
    def _later_split_of(self, frame: str, after: int) -> bool:
        """`frame` (or a name derived from it) is later cut into train/test rows."""
        for st in self._stmts_sorted:
            if st.lineno <= after or not isinstance(st, ast.Assign):
                continue
            if not (self.names_in(st.value) & {frame}):
                continue
            wn, _wk, _a, _b = self._v6_writes(st)
            if any(_tokens(x) & SPLIT_T for x in wn):
                return True
        return False

    def _rule_asset_stats(self, n: ast.Call) -> None:
        a = _attr(n)
        if a == "rank" and self.ts_context:
            pct = _kw(n, "pct")
            if not (isinstance(pct, ast.Constant) and pct.value is True):
                return
            g = n.func.value
            if isinstance(g, ast.Subscript):
                g = g.value
            if not (isinstance(g, ast.Call) and _attr(g) == "groupby"):
                return
            key = _kw(g, "by") or (g.args[0] if g.args else None) or _kw(g, "level")  # v0.7: level="ticker"
            ks = _const_str(key)
            if not ks or _date_like(ks) or _tokens(ks) & {"sector", "industry", "bucket", "group", "decile"}:
                return
            if any(_attr(c) in ("rolling", "expanding", "shift") for c in _chain(n)):
                return
            names, keys, after = self._v6_start(n)
            if any(classify(k) in ("label", "eval") for k in keys):
                return
            reached = self._v6_reach(names, keys, after) | keys
            sev = "medium" if self._v6_decides(reached) or self._v7_decides(reached) else "possible"
            self.add("LA09", sev, n,
                     f"`groupby('{ks}').rank(pct=True)` ranks each row against that asset's *whole* history, later rows "
                     "included. Use a trailing window (`rolling(...).rank(pct=True)`) or rank across assets on each date.")
            return
        if a == "quantile" and n.args and _num(n.args[0]) is not None and (_num(n.args[0]) <= 0.1 or _num(n.args[0]) >= 0.9):
            base = _chain_base(n.func.value)
            fb = base.value if isinstance(base, ast.Subscript) else base
            if not isinstance(fb, ast.Name) or _tokens(fb.id) & SPLIT_T:
                return
            if any(_attr(c) in ("rolling", "expanding", "groupby") for c in _chain(n)):
                return
            st = self._stmt_of(n)
            if not isinstance(st, ast.Assign) or self.inside(st, (ast.FunctionDef, ast.For)) is not None:
                return
            wn, _wk, _a, _b = self._v6_writes(st)
            clipped = any(isinstance(c, ast.Call) and _attr(c) == "clip" and self.names_in(c) & wn
                          and _chain_base(c.func.value) is not None and fb.id in self.names_in(c.func.value)
                          for c in self._all)
            if clipped and self._later_split_of(fb.id, st.lineno):
                self.add("LA09", "medium", n,
                         f"`{fb.id}` is winsorized at quantiles of *every* row before it is split into training and "
                         "test rows, so the cut-offs use the test period. Compute the quantiles on the training rows.")
            return
        if a == "fillna" and n.args and isinstance(n.args[0], ast.Call) and _attr(n.args[0]) in ("mean", "median"):
            stat = n.args[0]
            if any(_attr(c) in ("rolling", "expanding", "groupby", "shift") for c in _chain(stat)):
                return
            if _kw(stat, "axis") is not None and _num(_kw(stat, "axis")) == 1:
                return
            b1 = _chain_base(n.func.value)
            b2 = _chain_base(stat.func.value)
            f1 = b1.value if isinstance(b1, ast.Subscript) else b1
            f2 = b2.value if isinstance(b2, ast.Subscript) else b2
            if not (isinstance(f1, ast.Name) and isinstance(f2, ast.Name) and f1.id == f2.id) or _tokens(f1.id) & SPLIT_T:
                return
            st = self._stmt_of(n)
            if self.inside(st, (ast.FunctionDef, ast.For)) is not None:
                return
            if self._later_split_of(f1.id, st.lineno):
                self.add("LA09", "medium", n,
                         f"Missing values are filled with the {_attr(stat)} of *every* row of `{f1.id}`, before it is split "
                         "into training and test rows, so the fill uses the test period. Use the training rows' "
                         "statistic or a forward fill.")

    # -- LA05 (v0.6): full-sample estimates that drive the backtest ---------------------------------------------
    _EST_FUNCS = {"OLS", "WLS", "GLS", "RLM", "polyfit", "linregress", "coint", "KMeans", "GaussianMixture",
                  "GaussianHMM", "arch_model", "ARIMA", "SARIMAX", "PCA", "FactorAnalysis", "LinearRegression"}

    def _has_split(self) -> bool:
        if not hasattr(self, "_v6_split"):
            out = any(isinstance(n, ast.Call) and _func_name(n) in (
                "train_test_split", "TimeSeriesSplit", "KFold", "PurgedKFold", "walk_forward") for n in self._all)
            out = out or any(_tokens(nm) & (SPLIT_T | {"walk", "wf", "fold", "folds", "window", "lookback", "expanding",
                                                       "split", "cutoff", "cut", "insample", "outsample"})
                             for nm in self.assigned_names)
            out = out or any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr in ("loc", "iloc")
                             and isinstance(n.slice, ast.Slice) and (n.slice.upper is not None) for n in self._all
                             if self.inside(n, (ast.Call,)) is not None and _attr(self.inside(n, (ast.Call,))) in FIT_ATTRS)
            self._v6_split = out
        return self._v6_split

    def _rule_full_sample_estimate(self, n: ast.Call) -> None:
        a = _attr(n)
        if a not in ("fit", "fit_predict", "fit_transform"):
            if not (_func_name(n) in ("polyfit", "linregress", "coint") and n.args):
                return
            est = _func_name(n)
            data = n.args
        else:
            recv = n.func.value
            c = recv if isinstance(recv, ast.Call) else (self._assigned_call(recv.id) if isinstance(recv, ast.Name) else None)
            est = _func_name(c) if c is not None else None
            if est not in self._EST_FUNCS:
                return
            data = (c.args if a == "fit" and est in ("OLS", "WLS", "GLS", "RLM", "arch_model", "ARIMA", "SARIMAX") and not n.args
                    else n.args)
            if not data:
                return
        if self.inside(n, (ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ListComp,
                           ast.DictComp)) is not None:
            return  # a walk-forward loop or a helper: too uncertain
        if any(self._fit_subset(d) for d in data) or any(
                _key(_chain_base(d))[0] and _tokens(_key(_chain_base(d))[0]) & SPLIT_T for d in data):
            return
        if self._has_split() or any(k.arg and _tokens(k.arg) & {"last", "first", "obs", "train", "split", "end", "start"}
                                    for k in n.keywords):
            return  # fit(last_obs=...): the estimation window is cut
        names, keys, after = self._v6_start(n)
        if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and a == "fit" and not keys:
            names, keys = {n.func.value.id}, {n.func.value.id}  # pca.fit(R): the fitted estimator is used later
        reached = self._v6_reach(names, keys, after) | keys
        if not any(classify(k) in ("signal", "pnl") and not _tokens(k) & PRED_T for k in reached):
            return  # a forecast compared with outcomes is an evaluation, not a position
        if not self._v6_backtested(reached):
            return  # positions for live use (fit on all history, trade tomorrow) are fine: there is no backtest here
        self.add("LA05", "medium", n,
                 f"`{est}` is estimated on the whole sample (no train/test split or rolling window in this code) and "
                 "its output sets the positions, so every trade uses parameters fitted on later data. Re-estimate on a "
                 "rolling or expanding window, or fit on a training period and trade only after it.")

    # -- driver and location post-pass ----------------------------------------------------------------------------
    def rule_v06(self) -> None:
        if "backtrader" in self.imports:
            self._rule_bt_future()
        self._rule_test_pickers()
        self._rule_cv_sampling()
        self._rule_survivors_v6()
        self._rule_extreme_fills()
        for n in self._all:
            if not isinstance(n, ast.Call):
                continue
            self._rule_two_sided(n)
            self._rule_period_broadcast(n)
            self._rule_asset_stats(n)
            self._rule_full_sample_estimate(n)

    def relocate_v06(self) -> None:
        """Also report a finding where the leak is written, when that is not the line already reported: the
        negative shift inside a helper, the `direction=` keyword of merge_asof, the pick after a parameter sweep."""
        by_line: dict[int, list[Finding]] = {}
        for f in self.findings:
            by_line.setdefault(f.line, []).append(f)
        extra = []
        for n in self._all:
            if not isinstance(n, ast.Call):
                continue
            fs = by_line.get(getattr(n, "lineno", 0), [])
            if not fs:
                continue
            if isinstance(n.func, ast.Name) and n.func.id in _FWD_FUNCS:
                for f in fs:
                    if f.rule_id == "LA01" and f.severity in ("high", "medium"):
                        fd = self._funcs.get(n.func.id)
                        ret = next((r for r in ast.walk(fd) if isinstance(r, ast.Return) and r.value is not None
                                    and _scan_expr(r.value).neg_shift_calls), None) if fd is not None else None
                        if ret is not None and ret.lineno != n.lineno:
                            extra.append(("LA01", f.severity, ret, n.lineno,
                                          f"`{n.func.id}()` returns a negatively shifted (future) series here; it is used "
                                          f"in a {('signal or feature' if f.severity == 'high' else 'derived value')} on "
                                          f"line {n.lineno}."))
                        break
            if _func_name(n) == "merge_asof":
                for f in fs:
                    if f.rule_id == "LA04" and f.severity in ("high", "medium"):
                        d = next((k for k in n.keywords if k.arg == "direction"), None)
                        if d is not None and d.value.lineno != n.lineno:
                            extra.append(("LA04", f.severity, d.value, n.lineno, f.message))
                        break
        for kind, sev, node, first, msg in extra:
            self.add(kind, sev, node, msg, same_as=first)

    # =========================================================================================================
    # v0.7 recall rules (auditor/docs/RECALL_ANALYSIS_v07.md; tests/corpus_v07/PREREGISTRATION.md). An
    # interprocedural flow index (helper returns, method calls, .pipe/.apply targets, class attributes, list appends,
    # loop targets) lets a value be followed out of the helper or constructor where it is computed. As in v0.6, a
    # rule says "possible" (or nothing) when it can't show the value reaches a position, a model, a CV or PnL.
    # =========================================================================================================
    _V7_FN = "\x00fn:"
    _V7_BUDGET = 400_000  # row visits for all v0.7 flow searches in one file (keeps every input linear-time)
    _V7_MAX_SITES = 25  # call sites examined per helper
    _V7_APPLIERS = ("pipe", "apply", "transform", "agg", "aggregate", "map", "applymap")

    def _refkeys(self, v: ast.AST) -> frozenset[str]:
        """Keys read by expression `v` (`_scan_expr(v).refs`), cached per node."""
        if not hasattr(self, "_rkm"):
            self._rkm: dict[int, frozenset[str]] = {}
        got = self._rkm.get(id(v))
        if got is None:
            got = self._rkm[id(v)] = frozenset(k for k, _l in _scan_expr(v).refs)
        return got

    def _v7_scope(self, n: ast.AST) -> ast.AST | None:
        return self.inside(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))

    def _v7_locals(self, fd: ast.AST | None) -> frozenset[str]:
        """Parameters and names bound inside function `fd` (cached)."""
        if fd is None:
            return frozenset()
        if not hasattr(self, "_v7loc"):
            self._v7loc: dict[int, frozenset[str]] = {}
        got = self._v7loc.get(id(fd))
        if got is None:
            a = fd.args
            out = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
            out |= {a.vararg.arg} if a.vararg else set()
            out |= {a.kwarg.arg} if a.kwarg else set()
            for x in ast.walk(fd):
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                    out.add(x.id)
            got = self._v7loc[id(fd)] = frozenset(out)
        return got

    def _v7_methods(self) -> set[str]:
        if not hasattr(self, "_v7meth"):
            self._v7meth = {f.name for cd in self._all if isinstance(cd, ast.ClassDef) for f in cd.body
                            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and not f.name.startswith("__")}
        return self._v7meth

    def _v7_fn_reads(self, nodes: list[ast.AST]) -> set[str]:
        """Pseudo-names of local helpers whose results these expressions read: f(...), x.pipe(f), x.apply(f),
        minimize(f, ...), self.m(...) / obj.m(...) for a method m defined in this file."""
        out: set[str] = set()
        meths = self._v7_methods()
        for v in nodes:
            for x in ast.walk(v):
                if isinstance(x, ast.Name) and x.id in self._funcs and isinstance(x.ctx, ast.Load):
                    out.add(self._V7_FN + x.id)
                elif isinstance(x, ast.Attribute) and x.attr in meths and isinstance(self.parents.get(x), ast.Call) \
                        and self.parents[x].func is x:
                    out.add(self._V7_FN + x.attr)
        return out

    def _v7_index(self):
        """Rows (node, names written, keys written, line, scope) indexed by the names and keys they read."""
        if hasattr(self, "_v7i"):
            return self._v7i
        rows: list[tuple[ast.AST, set[str], set[str], int, ast.AST | None]] = []
        by_name: dict[str, list[int]] = {}
        by_key: dict[str, list[int]] = {}
        for st in self._stmts_sorted:
            vals: list[ast.AST] = []
            if isinstance(st, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and getattr(st, "value", None) is not None:
                wn, wk, rn, rk = self._v6_writes(st)
                wn, wk, rn, rk = set(wn), set(wk), set(rn), set(rk)
                vals = [st.value]
            elif isinstance(st, ast.Return) and st.value is not None:
                fd = self.inside(st, (ast.FunctionDef, ast.AsyncFunctionDef))
                if fd is None:
                    continue
                wn = {self._V7_FN + fd.name}
                wk, rn, rk = set(wn), set(), set()
                vals = [st.value]
            elif (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                  and _attr(st.value) in ("append", "extend", "update", "add", "insert", "setdefault")):
                b = st.value.func.value
                while isinstance(b, (ast.Subscript, ast.Attribute)):
                    b = b.value
                if not isinstance(b, ast.Name):
                    continue
                wn, wk, rn, rk = {b.id}, {b.id}, set(), set()
                vals = list(st.value.args) + [k.value for k in st.value.keywords]
            elif isinstance(st, (ast.For, ast.AsyncFor)):
                wn = {x.id for x in ast.walk(st.target) if isinstance(x, ast.Name)}
                wk, rn, rk = set(wn), set(), set()
                vals = [st.iter]
            else:
                continue
            meths = self._v7_methods()
            for v in vals:
                rn |= self.names_in(v)
                rk |= self._refkeys(v)
                for x in ast.walk(v):  # one walk: self.<attr> reads and helper results read
                    if isinstance(x, ast.Attribute):
                        if isinstance(x.value, ast.Name) and x.value.id == "self" and isinstance(x.ctx, ast.Load):
                            rk.add("@" + x.attr)
                        if x.attr in meths and isinstance(self.parents.get(x), ast.Call) and self.parents[x].func is x:
                            rn.add(self._V7_FN + x.attr)
            rn |= {self._V7_FN + nm for nm in rn if nm in self._funcs}
            sw = self._v7_self_targets(st)
            if sw:
                wk |= sw
                wn.discard("self")
                wk.discard("self")
            i = len(rows)
            rows.append((st, wn, wk, st.lineno, self._v7_scope(st)))
            for nm in rn:
                by_name.setdefault(nm, []).append(i)
            for k in rk:
                by_key.setdefault(k, []).append(i)
        self._v7i = (rows, by_name, by_key)
        return self._v7i

    @staticmethod
    def _v7_self_attrs(v: ast.AST) -> set[str]:
        return {"@" + x.attr for x in ast.walk(v) if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)
                and x.value.id == "self" and isinstance(x.ctx, ast.Load)}

    @staticmethod
    def _v7_self_targets(st: ast.AST) -> set[str]:
        tg = st.targets if isinstance(st, ast.Assign) else ([st.target] if isinstance(st, (ast.AugAssign, ast.AnnAssign)) else [])
        out = set()
        for t in tg:
            for e in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == "self":
                    out.add("@" + e.attr)
        return out

    def _v7_reach(self, names: set[str], keys: set[str], after: int = 0, scope: ast.AST | None = None,
                  cap: int = 600, stop=None) -> set[str]:
        """Names and keys computed from `names`/`keys` (written at line `after` in `scope`), following helper returns
        to their call sites and attributes into methods. A plain name is followed only in its own scope (or from
        module level into functions that don't rebind it)."""
        rows, by_name, by_key = self._v7_index()
        memo_key = (frozenset(names), frozenset(keys), after, id(scope), cap) if stop is None else None
        if memo_key is not None:
            if not hasattr(self, "_v7rm"):
                self._v7rm: dict[tuple, frozenset[str]] = {}
            got = self._v7rm.get(memo_key)
            if got is not None:
                return set(got)
        if not hasattr(self, "_v7budget"):
            self._v7budget = self._V7_BUDGET
        out_n: dict[str, ast.AST | None] = {x: scope for x in names}
        out_k = set(keys)
        todo = [("n", x, scope, after) for x in names] + [("k", x, None, after) for x in keys]
        seen: set[int] = set()
        while todo and len(seen) < cap:
            kind, x, sc, aft = todo.pop()
            for i in (by_name if kind == "n" else by_key).get(x, ()):
                self._v7budget -= 1
                if self._v7budget <= 0:
                    return set(names) | set(keys)  # out of budget (a huge file): say nothing rather than guess
                if i in seen:
                    continue
                st, wn, wk, line, rsc = rows[i]
                if kind == "n" and not x.startswith(self._V7_FN):
                    if rsc is not sc and not (sc is None and x not in self._v7_locals(rsc)):
                        continue
                if line <= aft and not x.startswith(self._V7_FN) and not (kind == "k" and rsc is not None):
                    continue
                if stop is not None and stop(st, x):
                    continue
                seen.add(i)
                for y in wn:
                    if y not in out_n:
                        out_n[y] = rsc
                        todo.append(("n", y, None if y.startswith(self._V7_FN) else rsc, line))
                for y in wk - out_k:
                    out_k.add(y)
                    todo.append(("k", y, None, line))
        out = set(out_n) | out_k
        if memo_key is not None:
            self._v7rm[memo_key] = frozenset(out)
        return out

    def _v7_cv_inputs(self) -> set[str]:
        """Names given to cross-validation: cross_val_* data, `.split(X)` data, and data indexed by fold positions."""
        if not hasattr(self, "_v7cvi"):
            out: set[str] = set()
            for c in self._all:
                if not isinstance(c, ast.Call):
                    continue
                fn = _func_name(c)
                if fn in ("cross_val_score", "cross_validate", "cross_val_predict", "permutation_test_score"):
                    for a in c.args[1:2] + [k.value for k in c.keywords if k.arg == "X"]:
                        out |= self.names_in(a) | {_const_str(x) for x in ast.walk(a) if _const_str(x)}
                elif fn == "split" and c.args:
                    out |= self.names_in(c.args[0])
            self._v7cvi = out
        return self._v7cvi

    def _v7_label_key(self, k: str | None) -> bool:
        if not k:
            return False
        ki = self.info.get(k)
        return classify(k) == "label" or (ki is not None and ki.forward) or k in self._lead_seen

    def _v7_eval_selectors(self) -> set[str]:
        """Keys used to pick or group rows of a forward return, a label, PnL or returns: fwd[sig], df.loc[sig, "fwd"],
        df[df.sig]["fwd"], df.groupby("sig")["fwd"].mean(), fwd.groupby(sig).mean()."""
        if hasattr(self, "_v7sel"):
            return self._v7sel
        out: set[str] = set()

        def target_ok(node: ast.AST) -> bool:
            ks = [k for k, _l in _scan_expr(node).refs] + [_const_str(x) for x in ast.walk(node) if _const_str(x)]
            return any(k and (self._v7_label_key(k) or classify(k) in ("pnl", "return")) for k in ks)

        for x in self._all:
            if isinstance(x, ast.Subscript) and not isinstance(x.slice, ast.Slice):
                sl = x.slice
                if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
                    sel, other = sl.elts[0], [sl.elts[1], x.value]
                else:
                    sel, other = sl, [x.value]
                if _const_str(sel) is not None:
                    p = self.parents.get(x)
                    if isinstance(p, ast.Subscript) and p.value is x:  # df[df.sig]["fwd"] read the other way round
                        continue
                    continue
                if any(target_ok(o) for o in other) or self._v7_filtered_then_read(x, target_ok):
                    out |= {k for k, _l in _scan_expr(sel).refs}
                p = self.parents.get(x)
                if isinstance(p, ast.Subscript) and p.value is x and _const_str(p.slice) and target_ok(p.slice):
                    out |= {k for k, _l in _scan_expr(sel).refs}
            elif isinstance(x, ast.Call) and _attr(x) == "groupby":
                key = x.args[0] if x.args else _kw(x, "by")
                if key is None:
                    continue
                p = self.parents.get(x)
                tgt = p.slice if isinstance(p, ast.Subscript) and p.value is x else x.func.value
                if target_ok(tgt):
                    out |= {k for k, _l in _scan_expr(key).refs} | {_const_str(e) for e in ast.walk(key) if _const_str(e)}
        out.discard(None)
        self._v7sel = out
        return out

    def _v7_filtered_then_read(self, x: ast.Subscript, target_ok) -> bool:
        """longs = panel[panel.sig > 0.9]; longs.groupby("date")["fwd_ret"].mean(): rows picked by a signal whose
        forward return is then read."""
        st = self._stmt_of(x)
        if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
            return False
        if self._v7_frame_of(st.value) is not self._v7_frame_of(x) and st.value is not x:
            return False
        f2 = st.targets[0].id
        if not hasattr(self, "_v7reads"):
            self._v7reads: dict[str, list[ast.AST]] = {}
            for y in self._all:
                if isinstance(y, ast.Subscript) and _const_str(y.slice):
                    b = self._v7_frame_of(y.value)
                    if isinstance(b, ast.Name):
                        self._v7reads.setdefault(b.id, []).append(y)
        return any(getattr(y, "lineno", 0) > st.lineno and target_ok(y.slice) for y in self._v7reads.get(f2, ()))

    def _v7_booked(self, reached: set[str]) -> bool:
        """`reached` weights applied to returns with @ / .dot / a product (a portfolio return)."""
        if self._v6_booked(reached):
            return True
        if not hasattr(self, "_v7mm"):
            def is_ret(side: ast.AST) -> bool:
                return any(classify(k) == "return" or (self.info.get(k) is not None and self.info[k].is_return)
                           for k, _l in _scan_expr(side).refs) or any(
                    isinstance(c, ast.Call) and _attr(c) in ("pct_change", "diff") for c in ast.walk(side))
            mm = []
            for m in self._all:
                pair = None
                if isinstance(m, ast.BinOp) and isinstance(m.op, ast.MatMult):
                    pair = (m.left, m.right)
                elif isinstance(m, ast.Call) and _attr(m) == "dot" and m.args:
                    pair = (m.func.value, m.args[0])
                if pair:
                    a = {k for k, _l in _scan_expr(pair[0]).refs}
                    b = {k for k, _l in _scan_expr(pair[1]).refs}
                    mm.append((a, is_ret(pair[0]), b, is_ret(pair[1])))
            self._v7mm = mm
        if any((a & reached and rb) or (b & reached and ra) for a, ra, b, rb in self._v7mm):
            return True
        # weights times a forward-return column: w * panel["ret_fwd_1m"]
        fr = lambda ks: any(_tokens(k) & RETURN_T and self._v7_label_key(k) for k in ks)
        return any((a & reached and fr(b)) or (b & reached and fr(a)) for a, _ra, b, _rb in self._v6_mults())

    def _v7_decides(self, reached: set[str], evals: bool = True) -> bool:
        """`reached` holds a position, signal, fill, PnL, a model or CV input, weights booked against returns, or (with
        `evals`) a selector of forward returns or PnL."""
        if self._v6_decides(reached) or self._v7_booked(reached):
            return True
        cvi = self._v7_cv_inputs()
        if any(k in cvi and classify(k) not in ("label", "eval", "pnl") for k in reached):
            return True
        return evals and bool(reached & self._v7_eval_selectors())

    def _v7_targets(self, st: ast.AST) -> tuple[set[str], set[str]]:
        if isinstance(st, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            wn, wk, _a, _b = self._v6_writes(st)
            sw = self._v7_self_targets(st)
            if sw:
                return set(wn) - {"self"}, (set(wk) - {"self"}) | sw
            return set(wn), set(wk)
        if isinstance(st, ast.Return):
            fd = self.inside(st, (ast.FunctionDef, ast.AsyncFunctionDef))
            if fd is not None:
                return {self._V7_FN + fd.name}, set()
        return set(), set()

    def _v7_flow(self, n: ast.AST) -> set[str]:
        """Everything computed from the statement holding `n`."""
        st = self._stmt_of(n)
        wn, wk = self._v7_targets(st)
        if not (wn or wk):
            return set()
        return self._v7_reach(wn, wk, getattr(st, "lineno", 0), self._v7_scope(st)) | wk

    # -- call sites of helpers -----------------------------------------------------------------------------------
    def _v7_call_sites(self, fname: str) -> list[tuple[ast.Call, ast.AST | None]]:
        """(call, the argument that is the helper's first parameter's data) for f(x), x.pipe(f), x.apply(f),
        obj.f(x) and Cls(x) (for __init__)."""
        if not hasattr(self, "_v7cs"):
            idx: dict[str, list[tuple[ast.Call, ast.AST | None]]] = {}
            classes = {cd.name for cd in self._all if isinstance(cd, ast.ClassDef)}
            for c in self._all:
                if not isinstance(c, ast.Call):
                    continue
                if isinstance(c.func, ast.Name):
                    nm = c.func.id
                    if nm in classes:
                        idx.setdefault(f"{nm}.__init__", []).append((c, c.args[0] if c.args else None))
                    else:
                        idx.setdefault(nm, []).append((c, c.args[0] if c.args else None))
                elif isinstance(c.func, ast.Attribute):
                    if _attr(c) in self._V7_APPLIERS and c.args and isinstance(c.args[0], ast.Name):
                        idx.setdefault(c.args[0].id, []).append((c, c.func.value))
                    else:
                        idx.setdefault("." + c.func.attr, []).append((c, c.args[0] if c.args else None))
            self._v7cs = idx
        return self._v7cs.get(fname, [])

    def _v7_full_arg(self, arg: ast.AST | None, at: ast.AST) -> bool:
        """`arg` is a whole frame or series: not a training/test slice, a window, one date, or a split-named frame."""
        if arg is None:
            return False
        if self._fit_subset(arg):
            return False
        x = arg
        while True:
            if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute):
                if _attr(x) in ("rolling", "expanding", "ewm", "groupby", "resample", "tail", "head", "xs",
                                "get_group", "sample"):
                    return False
                x = x.func.value
            elif isinstance(x, ast.Subscript):
                sl = x.slice.elts[0] if isinstance(x.slice, ast.Tuple) and x.slice.elts else x.slice
                if isinstance(sl, ast.Slice) and (sl.lower is not None or sl.upper is not None):
                    return False
                if isinstance(sl, (ast.Compare, ast.BoolOp)) or (isinstance(sl, ast.Name) and self._date_bound(sl)):
                    return False
                x = x.value
            elif isinstance(x, ast.Attribute):
                x = x.value
            else:
                break
        if isinstance(x, ast.Name):
            if _tokens(x.id) & (SPLIT_T | VAL_SET_T | {"window", "win", "block", "chunk", "fold", "hist", "past"}):
                return False
            if x.id in self._xs_frames():
                return False
            v = self._assigned_value(x.id, getattr(at, "lineno", 10 ** 9))
            if v is not None and v is not arg and not isinstance(v, ast.Call) and not self._v7_full_arg(v, at):
                return False
            if v is not None and isinstance(v, ast.Call) and any(
                    isinstance(s, ast.Subscript) and isinstance(s.slice, ast.Slice) and s.slice.upper is not None
                    for s in ast.walk(v)):
                return False
        return True

    def _v7_call_ok(self, c: ast.Call) -> bool:
        """A call site that runs once on the whole sample: not per window, per date, per row or per fold."""
        if self.inside(c, (ast.For, ast.While, ast.comprehension, ast.Lambda, ast.ListComp, ast.DictComp,
                           ast.GeneratorExp, ast.SetComp)) is not None:
            return False
        if _attr(c) in self._V7_APPLIERS and _attr(c) != "pipe":
            ax = _kw(c, "axis")
            if ax is not None and (_num(ax) == 1 or _const_str(ax) == "columns"):
                return False
            for r in _chain(c.func.value):
                if _attr(r) in ("rolling", "expanding", "ewm", "resample"):
                    return False
                if _attr(r) == "groupby":
                    key = r.args[0] if r.args else (_kw(r, "by") or _kw(r, "level"))
                    ks = _const_str(key)
                    if ks is None or _date_like(ks):
                        return False
            if isinstance(c.func.value, ast.Subscript) and isinstance(c.func.value.value, ast.Call) and _attr(
                    c.func.value.value) == "groupby":
                g = c.func.value.value
                key = g.args[0] if g.args else (_kw(g, "by") or _kw(g, "level"))
                ks = _const_str(key)
                if ks is None or _date_like(ks):
                    return False
        return True

    # -- A: full-sample statistics written inside helpers, pipe/apply functions and constructors -------------------
    _V7_NORM_STATS = ("mean", "std", "var", "median", "quantile", "mad", "cov", "min", "max")
    _V7_FITTERS = PREPROCESSORS | {"SelectKBest", "SelectPercentile", "SelectFromModel", "RFE", "RFECV",
                                    "GenericUnivariateSelect", "IterativeImputer", "SimpleImputer", "KNNImputer"}

    def _v7_param_of(self, x: ast.AST, fd: ast.AST) -> str | None:
        """The parameter of `fd` that `x` reads its data from (directly, through a column/attribute selection, or
        through `self.<attr>` set from a constructor parameter)."""
        params = [p.arg for p in fd.args.posonlyargs + fd.args.args if p.arg not in ("self", "cls")]
        while (_np_call(x) or (isinstance(x, ast.Call) and isinstance(x.func, ast.Name))) and x.args:
            x = x.args[0]  # np.log(px.values), np.asarray(s)
        lam = self.inside(x, (ast.Lambda,))
        if lam is not None and self._v7_group_lambda(x):
            gb = self._v7_frame_of(x)
            if isinstance(gb, ast.Name) and gb.id in {a.arg for a in lam.args.args}:
                x = self.parents[lam].func.value.func.value  # groupby receiver
        b = _chain_base(x)
        while isinstance(b, (ast.Subscript, ast.Attribute, ast.Call)):
            if isinstance(b, ast.Attribute) and isinstance(b.value, ast.Name) and b.value.id == "self":
                src = self._v7_self_attr_param(b.attr)
                return src
            b = b.func.value if isinstance(b, ast.Call) and isinstance(b.func, ast.Attribute) else (
                b.value if isinstance(b, (ast.Subscript, ast.Attribute)) else None)
            if b is None:
                return None
        if isinstance(b, ast.Name):
            if b.id in params:
                return b.id
            # a local bound once from a parameter without a window/slice: rev = x.copy(), d = df[cols]
            assigns = [a for a in self._assigns_of(b.id) if self.inside(a, (ast.FunctionDef, ast.AsyncFunctionDef)) is fd]
            v = None
            if len(assigns) == 1 and len(assigns[0].targets) == 1 and isinstance(assigns[0].targets[0], ast.Name):
                v = assigns[0].value
            if not assigns:
                for a in ast.walk(fd):
                    if isinstance(a, ast.Assign) and len(a.targets) == 1 and isinstance(a.targets[0], ast.Tuple) \
                             and isinstance(a.value, ast.Tuple) and len(a.value.elts) == len(a.targets[0].elts):
                        for te, ve in zip(a.targets[0].elts, a.value.elts):
                            if isinstance(te, ast.Name) and te.id == b.id:
                                v = ve
            if v is not None:
                if self._v7_full_arg(v, v) and not any(_attr(c) in ("rolling", "expanding", "ewm", "shift", "groupby")
                                                       for c in ast.walk(v) if isinstance(c, ast.Call)):
                    vb = _chain_base(v)
                    while isinstance(vb, (ast.Subscript, ast.Attribute)):
                        vb = vb.value
                    if isinstance(vb, ast.Name) and vb.id in params:
                        return vb.id
        return None

    def _v7_self_attr_param(self, attr: str) -> str | None:
        """`self.<attr> = <param>...` in a constructor: the constructor parameter it holds."""
        for cd in self._all:
            if not isinstance(cd, ast.ClassDef):
                continue
            init = next((f for f in cd.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
            if init is None:
                continue
            params = {p.arg for p in init.args.args} - {"self"}
            for st in ast.walk(init):
                if isinstance(st, ast.Assign):
                    for t in st.targets:
                        tl = t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]
                        vl = st.value.elts if isinstance(st.value, (ast.Tuple, ast.List)) and len(st.value.elts) == len(tl) else [st.value] * len(tl)
                        for te, ve in zip(tl, vl):
                            if isinstance(te, ast.Attribute) and isinstance(te.value, ast.Name) and te.value.id == "self" and te.attr == attr:
                                vb = _chain_base(ve)
                                while isinstance(vb, (ast.Subscript, ast.Attribute)):
                                    vb = vb.value
                                if isinstance(vb, ast.Name) and vb.id in params and self._v7_full_arg(ve, ve):
                                    return vb.id
        return None

    def _v7_helper_stat(self, n: ast.AST) -> tuple[ast.AST, str, str, str] | None:
        """(function, parameter, rule, description) when `n` is a whole-sample statistic or fit of a helper's
        parameter."""
        if not isinstance(n, ast.Call):
            return None
        a = _attr(n)
        if not (a in self._V7_NORM_STATS or a in ("fit", "fit_transform", "transform") or _np_call(n)):
            return None
        fd = self.inside(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        if fd is None:
            return None
        if a in self._V7_NORM_STATS and not _np_call(n):
            if self._global_stat(n, self._V7_NORM_STATS) is None:
                return None
            if self.inside(n, (ast.Lambda,)) is not None and not self._v7_group_lambda(n):
                return None
            p = self._v7_param_of(n.func.value, fd)
            if p is None:
                return None
            if a in ("min", "max") and not self._v7_scaling_use(n):
                return None
            rk = _key(_chain_base(n.func.value))[0]
            if rk and (_date_like(rk) or _tokens(rk) & DATE_KEY_T):
                return None  # df["date"].min(): a date bound, not a statistic of the data
            return fd, p, "LA09", f"`.{a}()` of the whole `{p}`"
        if _np_call(n) and n.args and n.func.attr in ("mean", "std", "nanmean", "nanstd", "median", "percentile",
                                                       "quantile", "cov", "polyfit"):
            if n.func.attr == "polyfit":
                if len(n.args) < 2:
                    return None
                p = self._v7_param_of(n.args[1], fd) or self._v7_param_of(n.args[0], fd)
                return (fd, p, "LA05", f"`np.polyfit` over the whole `{p}`") if p else None
            if _kw(n, "axis") is not None and _num(_kw(n, "axis")) == 1:
                return None
            p = self._v7_param_of(n.args[0], fd)
            return (fd, p, "LA09", f"`np.{n.func.attr}` of the whole `{p}`") if p else None
        if a == "transform" and n.args and _const_str(n.args[0]) in ("mean", "std", "median", "min", "max", "var"):
            g = n.func.value
            if isinstance(g, ast.Name):
                g = self._assigned_value(g.id, n.lineno)
            if isinstance(g, ast.Subscript):
                g = g.value
            if not (isinstance(g, ast.Call) and _attr(g) == "groupby"):
                return None
            key = g.args[0] if g.args else (_kw(g, "by") or _kw(g, "level"))
            ks = _const_str(key)
            if not ks or _date_like(ks) or _tokens(ks) & {"sector", "industry", "bucket", "group", "decile", "year", "month"}:
                return None
            p = self._v7_param_of(g.func.value, fd)
            return (fd, p, "LA09", f"`groupby('{ks}').transform('{_const_str(n.args[0])}')` over each asset's whole history") if p else None
        if a in ("fit", "fit_transform") and n.args:
            recv = n.func.value
            c = recv if isinstance(recv, ast.Call) else (self._assigned_call(recv.id) if isinstance(recv, ast.Name) else None)
            est = _func_name(c) if c is not None else None
            if est not in self._V7_FITTERS:
                return None
            p = self._v7_param_of(n.args[0], fd)
            return (fd, p, "LA05", f"`{est}` fitted on the whole `{p}`") if p else None
        return None

    def _v7_group_lambda(self, n: ast.AST) -> bool:
        """`n` sits in `groupby(<asset>).apply(lambda g: ...)`: the group's whole history."""
        lam = self.inside(n, (ast.Lambda,))
        p = self.parents.get(lam)
        if not (isinstance(p, ast.Call) and _attr(p) in ("apply", "transform") and isinstance(p.func.value, ast.Call)
                and _attr(p.func.value) == "groupby"):
            return False
        g = p.func.value
        key = g.args[0] if g.args else (_kw(g, "by") or _kw(g, "level"))
        ks = _const_str(key)
        if not ks or _date_like(ks) or _tokens(ks) & {"sector", "industry", "bucket", "group", "decile", "year", "month"}:
            return False
        if not isinstance(n, ast.Call):
            return True
        return isinstance(n.func.value, (ast.Subscript, ast.Name)) and isinstance(_chain_base(n.func.value), (ast.Subscript, ast.Name))

    def _v7_scaling_use(self, n: ast.Call) -> bool:
        """A min/max that scales or normalizes (x / x.max(), (x - x.min()) / (x.max() - x.min()))."""
        p = self.parents.get(n)
        while isinstance(p, (ast.Call, ast.Attribute)) and not isinstance(p, ast.BinOp):
            p = self.parents.get(p)
        return isinstance(p, ast.BinOp) and isinstance(p.op, (ast.Div, ast.Sub))

    def _rule_helper_stats(self) -> None:
        done: set[int] = set()
        for n in self._all:
            got = self._v7_helper_stat(n)
            if got is None:
                continue
            fd, p, rule, what = got
            st = self._stmt_of(n)
            if st.lineno in done:
                continue
            if (classify(p) in ("label", "eval", "pnl") and not _tokens(p) & RETURN_T) or _tokens(p) & {
                    "true", "pred", "actual", "preds", "predicted", "yhat"} or _tokens(
                    fd.name) & {"stats", "metrics", "metric", "report", "summary", "evaluate", "score", "sharpe", "perf"}:
                continue  # a metrics helper (r2 of y_true, a Sharpe of the returns): an evaluation, not a transform
            # 1) the statistic reaches the helper's return (or, in a constructor, an attribute a method returns)
            wn, wk = self._v7_targets(st)
            local = self._v7_reach(wn, wk, st.lineno, self._v7_scope(st)) | wk | wn
            fn_out = {x[len(self._V7_FN):] for x in local if x.startswith(self._V7_FN)}
            if not fn_out:
                continue
            combo = fd.name == "__init__" or self._v7_data_combo(fd, p, n, local)
            # 2) a call site passes the whole sample for that parameter, once
            pidx = [q.arg for q in fd.args.posonlyargs + fd.args.args if q.arg not in ("self", "cls")].index(p) \
                if p in [q.arg for q in fd.args.posonlyargs + fd.args.args] else 0
            is_init = fd.name == "__init__"
            cname = None
            if is_init:
                cd = self.inside(fd, (ast.ClassDef,))
                cname = f"{cd.name}.__init__" if cd is not None else None
            sites = self._v7_call_sites(cname) if is_init and cname else (
                self._v7_call_sites(fd.name) + self._v7_call_sites("." + fd.name))
            hit = None
            for c, first in sites[: self._V7_MAX_SITES]:
                if c is n or self.inside(c, (ast.FunctionDef, ast.AsyncFunctionDef)) is fd:
                    continue
                if not self._v7_call_ok(c):
                    continue
                if _attr(c) in self._V7_APPLIERS and isinstance(c.args[0], ast.Name) and c.args[0].id == fd.name:
                    arg = first if pidx == 0 else (c.args[pidx] if len(c.args) > pidx else None)
                else:
                    arg = c.args[pidx] if len(c.args) > pidx else _kw(c, p)
                if not self._v7_full_arg(arg, c):
                    continue
                cst = self._stmt_of(c)
                if is_init:
                    names, keys = self._v7_targets(cst)
                    reached = self._v7_reach(names, keys, cst.lineno, self._v7_scope(cst)) | keys
                else:
                    names, keys = self._v7_targets(cst)
                    reached = (self._v7_reach(names, keys, cst.lineno, self._v7_scope(cst)) | keys) if (names or keys) else set()
                    if isinstance(cst, ast.Expr):
                        reached = set()
                if not combo and not self._v7_applied_back(reached, arg, cst.lineno):
                    continue  # a scalar summary of the data (a volatility, a mean), not a transform of it
                if self._v7_decides(reached) or self._v7_split_later(reached, cst.lineno) or self._v7_compounded(reached):
                    hit = c
                    break
            if hit is None:
                continue
            done.add(st.lineno)
            fix = ("Compute it on the training rows (or a trailing window) and pass it in, or fit inside the "
                   "walk-forward loop.")
            self.add(rule, "medium", n,
                     f"{what} is computed inside `{fd.name}()`, and `{fd.name}` is called on the full sample on line "
                     f"{hit.lineno}, so every row is transformed with statistics that include later rows. " + fix)

    _V7_REDUCERS = {"sum", "mean", "std", "var", "median", "min", "max", "count", "nansum", "nanmean", "nanstd",
                    "len", "float", "int", "round", "prod", "item", "corr", "cov", "quantile", "describe",
                    "print", "format", "idxmax", "idxmin", "argmax", "argmin", "all", "any", "nunique", "size"}

    def _v7_data_combo(self, fd: ast.AST, p: str, stat: ast.AST, local: set[str]) -> bool:
        """Inside `fd`, the statistic (or a name computed from it) is combined with the parameter's data itself (not
        a reduction of it): (s - mu) / sd, frame.clip(lo, hi), r * (target / vol), log(px) - fit."""
        params = {p}
        lam = self.inside(stat, (ast.Lambda,))
        if lam is not None and self._v7_group_lambda(stat):
            params |= {a.arg for a in lam.args.args}  # groupby(asset).apply(lambda g: (g[c] - g[c].mean()) / ...)
        for a in ast.walk(fd):  # locals bound from the parameter without a reduction: x = s.values, rev = s[::-1]
            if isinstance(a, ast.Assign) and len(a.targets) == 1 and isinstance(a.targets[0], ast.Name) \
                    and p in self.names_in(a.value) and not any(
                        isinstance(c, ast.Call) and (_attr(c) or _func_name(c)) in self._V7_REDUCERS for c in ast.walk(a.value)):
                params.add(a.targets[0].id)
        stat_names = {x for x in local if not x.startswith(self._V7_FN)}

        def reduced(node: ast.AST, top: ast.AST) -> bool:
            q = self.parents.get(node)
            while q is not None and q is not top:
                if isinstance(q, ast.Call) and (_attr(q) or _func_name(q)) in self._V7_REDUCERS:
                    return True
                if isinstance(q, ast.Subscript) and _num(q.slice) is not None:
                    return True
                q = self.parents.get(q)
            return False

        for b in ast.walk(fd):
            if not (isinstance(b, ast.BinOp) or (isinstance(b, ast.Call) and _attr(b) in (
                    "clip", "where", "mask", "sub", "div", "mul", "subtract", "divide", "assign", "fit_transform",
                    "transform"))):
                continue
            if reduced(b, fd):
                continue
            has_stat = any(x is stat for x in ast.walk(b)) or bool(self.names_in(b) & stat_names) or bool(
                self._v7_self_attrs(b) & local)
            if not has_stat:
                continue
            for x in ast.walk(b):
                if isinstance(x, ast.Name) and x.id in params and isinstance(x.ctx, ast.Load) and not reduced(x, b) \
                        and not any(y is x for y in ast.walk(stat)):
                    return True
        if isinstance(stat, ast.Call) and _attr(stat) == "fit_transform":
            return True
        est = self._v7_est_call(stat) if isinstance(stat, ast.Call) and _attr(stat) == "fit" else None
        return est is not None and _func_name(est) in ("SelectKBest", "SelectPercentile", "SelectFromModel", "RFE",
                                                       "GenericUnivariateSelect")

    def _v7_est_call(self, n: ast.Call) -> ast.Call | None:
        recv = n.func.value
        return recv if isinstance(recv, ast.Call) else (self._assigned_call(recv.id) if isinstance(recv, ast.Name) else None)

    def _v7_applied_back(self, reached: set[str], arg: ast.AST | None, after: int) -> bool:
        """The helper's result is applied to the very frame it was estimated on: rets @ w, rets.dot(w), rets * w."""
        b = self._v7_frame_of(arg) if arg is not None else None
        if not isinstance(b, ast.Name):
            return False
        same = self._v7_reach({b.id}, set(), after, None) | {b.id}
        for line, la, ra in self._v7_products():
            if line > after and ((la & reached and ra & same) or (ra & reached and la & same)):
                return True
        return False

    def _v7_products(self) -> list[tuple[int, frozenset[str], frozenset[str]]]:
        """(line, names of one side, names of the other) for every product, matrix product, .dot and .mul."""
        if not hasattr(self, "_v7prod"):
            out = []
            for m in self._all:
                pair = None
                if isinstance(m, ast.BinOp) and isinstance(m.op, (ast.MatMult, ast.Mult)):
                    pair = (m.left, m.right)
                elif isinstance(m, ast.Call) and _attr(m) in ("dot", "mul") and m.args:
                    pair = (m.func.value, m.args[0])
                if pair:
                    out.append((m.lineno, self.names_in(pair[0]), self.names_in(pair[1])))
            self._v7prod = out
        return self._v7prod

    def _v7_compounded(self, reached: set[str]) -> bool:
        """A reached series is compounded into an equity curve ((1 + r).cumprod(), (1 + r).prod())."""
        if not hasattr(self, "_v7cmp"):
            self._v7cmp = [self.names_in(c.func.value) for c in self._all if isinstance(c, ast.Call)
                           and _attr(c) in ("cumprod", "prod") and isinstance(c.func.value, ast.BinOp)
                           and isinstance(c.func.value.op, ast.Add)]
        return any(ns & reached for ns in self._v7cmp)

    def _v7_split_later(self, reached: set[str], after: int) -> bool:
        """A reached frame is later cut into train/test rows by name (train = panel[panel.date < cut])."""
        if not hasattr(self, "_v7spl"):
            self._v7spl: dict[str, int] = {}  # name -> last line where it is read by a split-named assignment
            for st in self._stmts_sorted:
                if isinstance(st, ast.Assign):
                    wn, _wk, _a, _b = self._v6_writes(st)
                    if any(_tokens(x) & SPLIT_T for x in wn):
                        for nm in self.names_in(st.value):
                            self._v7spl[nm] = max(self._v7spl.get(nm, 0), st.lineno)
        return any(self._v7spl.get(nm, 0) > after for nm in reached)

    # -- B: survivorship ---------------------------------------------------------------------------------------
    def _v7_frame_of(self, x: ast.AST) -> ast.AST:
        b = x
        while True:
            if isinstance(b, ast.Call) and isinstance(b.func, ast.Attribute):
                b = b.func.value
            elif isinstance(b, (ast.Subscript, ast.Attribute)):
                b = b.value
            else:
                return b

    def _v7_pricey_frame(self, name: str | None, at: int) -> bool:
        if not name:
            return False
        if self._wide_price_name(name):
            return True
        v = self._assigned_value(name, at)
        for c in (ast.walk(v) if v is not None else ()):
            if isinstance(c, ast.Call) and (_func_name(c) or "").startswith("read_") and c.args:
                path = (_const_str(c.args[0]) or "").lower()
                if re.search(r"(close|price|px|adj|ret|ohlc|quote)", path):
                    return True
        return False

    def _v7_backtested_names(self, st: ast.AST, names: set[str], keys: set[str]) -> bool:
        """The kept names (or frame) reach a backtest: time-series operations on a selection, positions or PnL."""
        reached = self._v7_reach(names, keys, st.lineno, self._v7_scope(st)) | keys | names
        rows, by_name, by_key = self._v7_index()
        if not hasattr(self, "_v7tsrow"):
            self._v7tsrow: dict[int, bool] = {}
        for nm in reached:
            for i in by_name.get(nm, ()):
                node, _wn, _wk, line, _sc = rows[i]
                if line < st.lineno and not nm.startswith(self._V7_FN):
                    continue
                got = self._v7tsrow.get(i)
                if got is None:
                    v = getattr(node, "value", None) if not isinstance(node, ast.For) else node.iter
                    nodes = list(ast.walk(v)) if v is not None else []
                    got = self._v7tsrow[i] = any(
                        isinstance(c, ast.Call) and _attr(c) in self._RET_OPS | {"resample", "cumprod"} for c in nodes) \
                         and not any(self._last_row(x) is not None for x in nodes)
                if got:
                    return True  # (a value reduced to the last row is today's list, not a backtest)
        return self._v6_backtested(reached) or self._v7_booked(reached) or bool(reached & self._v7_eval_selectors())

    def _rule_survivors_v7(self) -> None:
        sb = ("The universe is fixed with knowledge of the end of the sample, so names that were delisted, acquired or "
              "went bankrupt are left out of the whole backtest (survivorship bias). Decide membership on each date "
              "with data available then, and keep dead names with their final returns.")
        flagged = {f.line for f in self.findings if f.rule_id == "SB01" and f.severity in ("high", "medium")}
        for n in self._all:
            if isinstance(n, ast.Call):
                if _attr(n) not in ("dropna", "isna", "isnull"):
                    continue
            elif isinstance(n, ast.Subscript):
                if not (isinstance(n.slice, ast.Compare) or (isinstance(n.slice, ast.Call) and _attr(n.slice) in (
                        "notna", "notnull"))):
                    continue
            else:
                continue
            if getattr(n, "lineno", 0) in flagged:
                continue
            st = self._stmt_of(n)
            if not isinstance(st, (ast.Assign, ast.Return)):
                continue
            why = None
            # dropna(axis=1) reached through .loc[...] / .pipe(lambda d: ...) / resample
            if isinstance(n, ast.Call) and _attr(n) == "dropna":
                ax = _kw(n, "axis")
                if not (_num(ax) == 1 or _const_str(ax) == "columns") or _const_str(_kw(n, "how")) == "all":
                    continue
                thresh = _kw(n, "thresh")
                if thresh is not None and not (isinstance(thresh, ast.Call) and _func_name(thresh) == "len"):
                    continue
                if self.inside(n, (ast.For, ast.While, ast.comprehension)) is not None:
                    continue
                if any(_attr(c) in ("tail", "head", "rolling", "expanding") for c in _chain(n.func.value)) or any(
                        isinstance(x, ast.Subscript) and isinstance(x.value, ast.Attribute) and x.value.attr == "iloc"
                        for x in ast.walk(n.func.value)):
                    continue  # a trailing window (.tail(252), .iloc[-252:]): the names alive in that window
                if any(isinstance(x, ast.Subscript) and any(
                        isinstance(sl, ast.Slice) and any(b is not None and not isinstance(b, ast.Constant)
                                                         for b in (sl.lower, sl.upper))
                        for sl in (x.slice.elts if isinstance(x.slice, ast.Tuple) else [x.slice]))
                       for x in ast.walk(n.func.value)):
                    continue  # prices.loc[t_lookback:t0]: a window that moves with the rebalance date
                base = self._v7_frame_of(n.func.value)
                lam = self.inside(n, (ast.Lambda,))
                if lam is not None:
                    if isinstance(base, ast.Name) and base.id in {a.arg for a in lam.args.args}:
                        base = None
                    else:
                        continue
                wn, wk = self._v7_targets(st)
                fname = base.id if isinstance(base, ast.Name) else next(iter(wn), None)
                if not (self._v7_pricey_frame(fname, st.lineno - 1) or any(self._v7_pricey_frame(w, st.lineno - 1) for w in wn)):
                    continue
                if wn and self._v7_windowed_later(wn, st.lineno):
                    continue  # formation = prices.loc[a:b]; trading = prices.loc[c:d]: windows of the kept frame
                if wn and self._v7_backtested_names(st, wn, wk):
                    why = "`dropna(axis=1)` keeps only the columns with a value on *every* date, i.e. the names that " \
                          "traded for the whole sample. "
            # cols[F.iloc[-1].notna()], F.iloc[-1].dropna().index
            elif isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Call) and _attr(n.slice) in ("notna", "notnull") \
                    and self._last_row(n.slice.func.value) is not None:
                why = "The names are kept by `iloc[-1].notna()`: only names that still have a price on the last date survive. "
            elif isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Compare) and len(n.slice.ops) == 1 \
                    and isinstance(n.value, ast.Name):
                # last[last > x].index / last_seen[last_seen == px.index[-1]].index
                lv = self._assigned_value(n.value.id, n.lineno)
                lrow = lv is not None and any(self._last_row(c) is not None for c in [lv, *_chain(lv)])
                lvi = lv is not None and isinstance(lv, ast.Call) and _attr(lv) == "apply" and lv.args and (
                    (isinstance(lv.args[0], ast.Attribute) and lv.args[0].attr == "last_valid_index"))
                left, right = n.slice.left, n.slice.comparators[0]
                if not (isinstance(left, ast.Name) and left.id == n.value.id):
                    continue
                p = self.parents.get(n)
                if not (isinstance(p, ast.Attribute) and p.attr == "index"):
                    continue
                if lvi and isinstance(n.slice.ops[0], (ast.Eq, ast.GtE)) and self._last_date_like(right):
                    why = "The universe is the names whose last valid price falls on the last date (the survivors). "
                elif lrow and isinstance(n.slice.ops[0], (ast.Gt, ast.GtE)) and not self._last_date_like(right):
                    sk = _key(_chain_base(self._v7_frame_of(lv)))[0]
                    if classify(sk) in ("signal", "feature", "eval", "label") or (sk and _tokens(sk) & PRED_T):
                        continue
                    why = f"The universe is cut from the *last* row of `{sk or 'the frame'}` (a value known only at the end). "
                if why is not None:
                    n = p
            # current members: an end/removed/delist column that is empty, also inside a helper or a query string
            elif isinstance(n, ast.Call) and _attr(n) in ("isna", "isnull") and (
                    isinstance(n.func.value, (ast.Subscript, ast.Attribute)) or (
                        isinstance(n.func.value, ast.Name) and not self._assigns_of(n.func.value.id))):
                col = _key(n.func.value)[0]
                if not (col and _tokens(col) & {"end", "removed", "delisted", "delist", "exit", "until", "out", "dropped", "removal"}):
                    continue
                if self.inside(n, (ast.BoolOp,)) is not None or any(
                        isinstance(b, ast.BinOp) and isinstance(b.op, (ast.BitOr, ast.BitAnd)) for b in ast.walk(st.value)):
                    continue
                if isinstance(self.parents.get(n), ast.UnaryOp):
                    continue
                why = f"`{col}` is empty only for *current* members, so this keeps today's members for the whole history. "
            if why is None:
                continue
            wn, wk = self._v7_targets(st)
            if not (wn or wk):
                continue
            if self._v7_backtested_names(st, wn, wk):
                flagged.add(n.lineno)
                self.add("SB01", "medium", n, why + sb)

    def _v7_windowed_later(self, names: set[str], after: int) -> bool:
        """A frame in `names` is later cut into a date window bounded on both sides."""
        if not hasattr(self, "_v7win"):
            self._v7win: dict[str, int] = {}
            for st in self._stmts_sorted:
                if not isinstance(st, ast.Assign):
                    continue
                for x in ast.walk(st.value):
                    if not (isinstance(x, ast.Subscript) and isinstance(x.slice, ast.Slice) and x.slice.lower is not None
                            and x.slice.upper is not None):
                        continue
                    fr = self._v7_frame_of(x)
                    if isinstance(fr, ast.Name):
                        self._v7win[fr.id] = max(self._v7win.get(fr.id, 0), st.lineno)
        return any(self._v7win.get(nm, 0) > after for nm in names)

    def _last_date_like(self, x: ast.AST) -> bool:
        """px.index[-1], px.index.max(), END-like constants."""
        if isinstance(x, ast.Subscript) and _num(x.slice) == -1 and isinstance(x.value, ast.Attribute) and x.value.attr == "index":
            return True
        if isinstance(x, ast.Call) and _attr(x) == "max" and isinstance(x.func.value, ast.Attribute) and x.func.value.attr == "index":
            return True
        return False

    def _rule_current_files(self) -> None:
        """A constituents file named as today's snapshot whose names pick the columns of a backtested frame."""
        for n in self._all:
            if not (isinstance(n, ast.Call) and _func_name(n) in ("read_csv", "read_parquet", "read_excel") and n.args):
                continue
            path = (_const_str(n.args[0]) or "").lower()
            if not re.search(r"(sp500|s&p|snp|constituent|member|tickers|universe|nasdaq|russell|index|components)", path):
                continue
            if not re.search(r"(current|latest|today|snapshot|live|active|now)", path) or re.search(
                    r"(hist|pit|point|asof|as_of|daily|changes)", path):
                continue
            st = self._stmt_of(n)
            if not isinstance(st, ast.Assign):
                continue
            wn, wk = self._v7_targets(st)
            if wn and self._v7_backtested_names(st, wn, wk) and self._v7_selects_history(st, wn):
                self.add("SB01", "medium", n,
                         f"The universe comes from `{_const_str(n.args[0])}`, a snapshot of *today's* members, and is "
                         "applied to the whole history: names that left before today are missing. Use point-in-time "
                         "membership and keep delisted names.")

    def _v7_selects_history(self, st: ast.AST, names: set[str]) -> bool:
        """The names select columns or rows of another frame (px[names], px.columns.intersection(names), isin)."""
        reached = self._v7_reach(names, set(), st.lineno, self._v7_scope(st)) | names
        for x in self._all:
            if getattr(x, "lineno", 0) <= st.lineno:
                continue
            if isinstance(x, ast.Subscript) and self.names_in(x.slice) & reached and not isinstance(x.ctx, ast.Store):
                return True
            if isinstance(x, ast.Call) and _attr(x) in ("intersection", "isin", "reindex", "filter") and self.names_in(x) & reached:
                return True
        return False

    # -- C: selection on the full sample -------------------------------------------------------------------------
    _V7_SCORERS = ("mutual_info_regression", "mutual_info_classif", "f_regression", "f_classif", "chi2", "r_regression")

    def _v7_used_in_validation(self, reached: set[str], after: int) -> bool:
        """The selection reaches a cross-validation, a fit inside a loop (walk-forward) or a fit on a split frame."""
        if reached & self._v7_cv_inputs():
            return True
        if not hasattr(self, "_v7vf"):
            self._v7vf = []  # (line, names read by X) of fits in a loop or on a training subset
            for c in self._all:
                if isinstance(c, ast.Call) and _attr(c) in ("fit", "fit_transform") and c.args and (
                        self.inside(c, (ast.For, ast.While)) is not None or self._fit_subset(c.args[0])):
                    ns = self.names_in(c.args[0]) | {_const_str(x) for x in ast.walk(c.args[0]) if _const_str(x)}
                    self._v7vf.append((c.lineno, ns))
        return any(line > after and ns & reached for line, ns in self._v7vf)

    def _rule_full_sample_selection(self) -> None:
        for n in self._all:
            if not isinstance(n, ast.Call):
                continue
            what, data = None, []
            if _attr(n) == "corrwith" and n.args:
                what, data = "`corrwith` against the target on every row", [n.func.value, n.args[0]]
            elif _func_name(n) in self._V7_SCORERS and len(n.args) >= 2:
                what, data = f"`{_func_name(n)}` on every row", n.args[:2]
            elif _attr(n) in ("fit", "fit_transform") and len(n.args) >= 2:
                recv = n.func.value
                c = recv if isinstance(recv, ast.Call) else (self._assigned_call(recv.id) if isinstance(recv, ast.Name) else None)
                est = _func_name(c) if c is not None else None
                if est not in ("SelectKBest", "SelectPercentile", "SelectFromModel", "RFE", "GenericUnivariateSelect",
                               "VarianceThreshold"):
                    continue
                what, data = f"`{est}` fitted on every row", n.args[:2]
            if what is None:
                continue
            if self.inside(n, (ast.For, ast.While, ast.comprehension, ast.Lambda)) is not None:
                continue
            fd = self.inside(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            if fd is not None:
                if any(self._v7_param_of(d, fd) is None for d in data):
                    continue
            elif not all(self._v7_full_arg(d, n) for d in data):
                continue
            # y is a target: a label, or the value later used as a fit/CV target
            ykey = _key(_chain_base(data[1]))[0]
            if fd is None and not (self._v7_label_key(ykey) or self._v7_is_fit_target(ykey)):
                continue
            st = self._stmt_of(n)
            wn, wk = self._v7_targets(st)
            reached = self._v7_reach(wn, wk, st.lineno, self._v7_scope(st)) | wk | wn
            if fd is not None:
                outs = {x for x in reached if x.startswith(self._V7_FN + fd.name)}
                if not outs:
                    continue
                ok = False
                for c, _first in self._v7_call_sites(fd.name)[: self._V7_MAX_SITES]:
                    if c is n or not self._v7_call_ok(c):
                        continue
                    if not all(self._v7_full_arg(a, c) for a in c.args[:2]):
                        continue
                    cst = self._stmt_of(c)
                    cn, ck = self._v7_targets(cst)
                    r2 = (self._v7_reach(cn, ck, cst.lineno, self._v7_scope(cst)) | cn | ck) if (cn or ck) else set()
                    if isinstance(cst, ast.Expr):
                        r2 = self.names_in(cst)
                    if self._v7_used_in_validation(r2 | {x for x in self.names_in(cst) if False}, cst.lineno - 1) or (
                            isinstance(cst, ast.Expr) and any(_func_name(x) in ("evaluate", "cross_val_score", "backtest", "walk_forward")
                                                              for x in ast.walk(cst) if isinstance(x, ast.Call))):
                        ok = True
                        break
                if not ok:
                    continue
            elif not self._v7_used_in_validation(reached, st.lineno):
                continue
            self.add("LA05", "medium", n,
                     f"Features are chosen by {what}, before the cross-validation or walk-forward that later scores "
                     "them, so the test folds helped pick the features. Select features inside each training fold "
                     "(e.g. in a Pipeline).")

    def _v7_is_fit_target(self, k: str | None) -> bool:
        if not k:
            return False
        if not hasattr(self, "_v7ft"):
            self._v7ft: set[str] = set()
            for c in self._all:
                if isinstance(c, ast.Call) and (_attr(c) in ("fit",) and len(c.args) >= 2 or _func_name(c) in (
                        "cross_val_score", "cross_validate", "cross_val_predict")):
                    ys = c.args[1:2] if _attr(c) == "fit" else c.args[2:3]
                    for y in ys + [kw.value for kw in c.keywords if kw.arg == "y"]:
                        self._v7ft |= self._refkeys(y)
        return k in self._v7ft

    def _rule_pair_selection(self) -> None:
        """Pairs or spreads kept by a full-sample cointegration/stationarity test, then backtested on the same rows."""
        for n in self._all:
            if not (isinstance(n, ast.Call) and _func_name(n) in ("coint", "adfuller", "kpss") and n.args):
                continue
            loop = self.inside(n, (ast.For,))
            if loop is None or self.inside(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) is not None:
                continue
            if not all(self._v7_full_arg(a, n) and not self._v7_window_name(a, loop) for a in n.args[: 2 if _func_name(n) == "coint" else 1]):
                continue
            st = self._stmt_of(n)
            # what the test decides: the names appended/stored under an `if` on its p-value, or a branch body
            kept: set[str] = set()
            body_nodes: list[ast.AST] = []
            wn, _wk = self._v7_targets(st)
            for ifs in ast.walk(loop):
                if not isinstance(ifs, ast.If):
                    continue
                tn = self.names_in(ifs.test)
                if not (tn & wn or any(x is n for x in ast.walk(ifs.test))):
                    continue
                body_nodes += ifs.body
                for b in ifs.body:
                    if isinstance(b, ast.Expr) and isinstance(b.value, ast.Call) and _attr(b.value) in ("append", "add"):
                        bb = self._v7_frame_of(b.value.func.value)
                        if isinstance(bb, ast.Name):
                            kept.add(bb.id)
                    elif isinstance(b, ast.Assign):
                        kept |= self._v7_targets(b)[0] | self._v7_targets(b)[1]
            if not (kept or body_nodes):
                continue
            reached = (self._v7_reach(kept, kept, loop.lineno, self._v7_scope(loop)) | kept) if kept else set()
            for b in body_nodes:
                for x in ast.walk(b):
                    if isinstance(x, (ast.Assign, ast.Expr)):
                        bn, bk = self._v7_targets(x) if isinstance(x, ast.Assign) else (self._v7_targets(x) or (set(), set()))
                        reached |= bn | bk
            if not self._v7_pnl_like(reached):
                continue
            self.add("LA05", "medium", n,
                     f"Pairs are kept by `{_func_name(n)}` over the *whole* sample and then traded on that same sample, "
                     "so the selection already knows which spreads mean-reverted. Test on a formation window and trade "
                     "only after it (re-select as you roll forward).")

    def _v7_window_name(self, a: ast.AST, loop: ast.AST) -> bool:
        b = self._v7_frame_of(a)
        return isinstance(b, ast.Name) and _tokens(b.id) & {"form", "formation", "insample", "train", "window"}

    def _v7_pnl_like(self, reached: set[str]) -> bool:
        if self._v6_backtested(reached) or self._v7_booked(reached):
            return True
        if not hasattr(self, "_v7pnlh"):
            pnl_fn: dict[str, bool] = {}
            self._v7pnlh = []
            for c in self._all:  # print({p: trade(*p) for p in pairs}): a PnL helper run over the kept names
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in self._funcs:
                    nm = c.func.id
                    if nm not in pnl_fn:
                        fd = self._funcs[nm]
                        pnl_fn[nm] = any(isinstance(r, ast.Return) and r.value is not None and self._v7_ret_pnl(r.value, fd)
                                         for r in ast.walk(fd))
                    if pnl_fn[nm]:
                        self._v7pnlh.append(self.names_in(self._stmt_of(c)))
        if any(ns & reached for ns in self._v7pnlh):
            return True
        rows, by_name, _bk = self._v7_index()
        for nm in reached:
            if nm.startswith(self._V7_FN):
                continue
            # backtest_pair(*p) for p in pairs / trade(a, b): a helper whose return is PnL
            for i in by_name.get(nm, ()):
                node = rows[i][0]
                for c in ast.walk(getattr(node, "value", None) or getattr(node, "iter", None) or node):
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in self._funcs:
                        fd = self._funcs[c.func.id]
                        if any(isinstance(r, ast.Return) and r.value is not None and self._v7_ret_pnl(r.value, fd) for r in ast.walk(fd)):
                            return True
        return False

    def _v7_ret_pnl(self, v: ast.AST, fd: ast.AST) -> bool:
        """A helper's return value is a strategy return: a lagged position times a return/diff, or a PnL-named value."""
        for m in ast.walk(v):
            if isinstance(m, ast.BinOp) and isinstance(m.op, ast.Mult):
                sides = (m.left, m.right)
                if any(_chain_shift(s) > 0 or any(_attr(c) == "shift" and _shift_sign(c) == 1 for c in ast.walk(s) if isinstance(c, ast.Call))
                       for s in sides) and any(any(isinstance(c, ast.Call) and _attr(c) in ("diff", "pct_change") for c in ast.walk(s)) or (
                            isinstance(s, ast.Name) and classify(s.id) == "return") for s in sides):
                    return True
        ks = {k for k, _l in _scan_expr(v).refs}
        if any(classify(k) == "pnl" for k in ks):
            return True
        # return pos * s.pct_change() where pos is a local lagged position
        for k in ks:
            for a in self._assigns_of(k):
                if self.inside(a, (ast.FunctionDef,)) is fd and classify(k) in ("signal", "pnl"):
                    return any(isinstance(m, ast.BinOp) and isinstance(m.op, ast.Mult) for m in ast.walk(v))
        return False

    def _rule_full_sample_weights(self) -> None:
        """Portfolio weights from the full-sample covariance (min-variance, mean-variance) applied to the same returns."""
        for n in self._all:
            if not (isinstance(n, ast.Call) and _attr(n) == "cov" and not n.args):
                continue
            if self.inside(n, (ast.For, ast.While, ast.comprehension, ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)) is not None:
                continue
            recv = n.func.value
            if self._global_stat_recv(recv) is None or not self._v7_full_arg(recv, n):
                continue
            rb = self._v7_frame_of(recv)
            if not (isinstance(rb, ast.Name) and (classify(rb.id) == "return" or (
                    self.info.get(rb.id) is not None and self.info[rb.id].is_return))):
                continue
            if self._has_split():
                continue
            st = self._stmt_of(n)
            wn, wk = self._v7_targets(st)
            reached = self._v7_reach(wn, wk, st.lineno, None) | wk | wn
            same = self._v7_reach({rb.id}, set(), st.lineno, None) | {rb.id}
            mm = any(line > st.lineno and ((la & reached and ra & (same - reached)) or (ra & reached and la & (same - reached)))
                     for line, la, ra in self._v7_products())
            if not (self._v7_booked(reached) or mm):
                continue
            self.add("LA05", "medium", n,
                     f"The weights come from the covariance of *all* of `{rb.id}` and are then applied to those same "
                     "returns, so every past rebalance uses risk estimates from later data. Estimate the covariance on "
                     "a trailing window at each rebalance.")

    # -- D: findings on the line where the leak is written --------------------------------------------------------
    def _rule_feature_lists(self) -> None:
        fits = [c for c in self._all if isinstance(c, ast.Call) and _attr(c) == "fit" and len(c.args) >= 2]
        if not fits:
            return
        label_src: dict[str, set[str]] = {}
        for k, sts in self._assign_by_key_v7().items():
            for st in sts:
                label_src.setdefault(k, set()).update(self._refkeys(st.value))
        la10_by_line: dict[int, list[Finding]] = {}
        for f in self.findings:
            if f.rule_id == "LA10":
                la10_by_line.setdefault(f.line, []).append(f)
        la10_lines = set(la10_by_line)
        for c in fits:
            X, Y = c.args[0], c.args[1]
            ys = {k for k, _l in _scan_expr(Y).refs} | {_const_str(x) for x in ast.walk(Y) if _const_str(x)}
            lists = []
            for x in ast.walk(X):
                if isinstance(x, ast.Subscript):
                    sl = x.slice.elts[1] if isinstance(x.slice, ast.Tuple) and len(x.slice.elts) == 2 else x.slice
                    if isinstance(sl, ast.Name):
                        v = self._assigned_value(sl.id, c.lineno)
                        if isinstance(v, (ast.List, ast.Tuple)):
                            lists.append(v)
                    elif isinstance(sl, (ast.List, ast.Tuple)):
                        lists.append(sl)
            ysrc: set[str] = set()
            for y in ys:
                if y:
                    ysrc |= label_src.get(y, set())
            fit_findings = [f for ln in range(c.lineno, getattr(c, "end_lineno", c.lineno) + 1)
                            for f in la10_by_line.get(ln, ()) if f.severity in ("high", "medium") and not f.same_as]
            for lst in lists:
                for e in lst.elts:
                    s = _const_str(e)
                    if not s or s in ys:
                        continue
                    ki = self.info.get(s)
                    fwd = ki is not None and ki.forward and classify(s) not in ("pnl", "eval") and not ki.lagged
                    src = s in ysrc and bool(_tokens(s) & {"fwd", "forward", "future", "fut", "next", "ahead", "lead"})
                    if not (fwd or src):
                        continue
                    if e.lineno in la10_lines:
                        continue
                    if fit_findings:
                        f0 = fit_findings[0]
                        self.add("LA10", f0.severity, e, f"`{s}` (forward-looking) is in the feature list used by the "
                                                         f"fit on line {f0.line}.", same_as=f0.line)
                    elif src:
                        self.add("LA10", "high", e,
                                 f"`{s}` is in the feature list, and the label is built from it: the model is given the "
                                 "answer. Remove it from the features.")

    def relocate_v07(self) -> None:
        """Also report LA15 at the `right_on=`/`left_on=`/`on=` keyword of the merge when it sits on another line."""
        for f in list(self.findings):
            if f.rule_id != "LA15" or f.same_as or f.severity not in ("high", "medium"):
                continue
            for c in self._v7_at_line(f.line):
                if not (isinstance(c, ast.Call) and _func_name(c) in ("merge_asof", "merge")):
                    continue
                for k in c.keywords:
                    if k.arg in ("right_on", "left_on", "on") and self._period_end_name(_const_str(k.value)) \
                            and k.value.lineno != c.lineno:
                        self.add("LA15", f.severity, k.value, f.message, same_as=c.lineno)
                break

    # -- E: CV and walk-forward ------------------------------------------------------------------------------------
    def _v7_y_lead(self, y: ast.AST | None) -> int:
        """Horizon of the label `y`: its computed lead, or the bars named in a forward column it is built from."""
        if y is None:
            return 0
        ks = {k for k, _l in _scan_expr(y).refs} | {_const_str(x) for x in ast.walk(y) if _const_str(x)}
        best = 0
        for k in [k for k in ks if k]:
            best = max(best, self._lead_seen.get(k, 0))
            ki = self.info.get(k)
            if ki is not None and ki.forward and ki.lead and ki.lead < 10 ** 5:
                best = max(best, ki.lead)
            srcs = {k}
            for st in self._assign_by_key_v7().get(k, ()):
                srcs |= {kk for kk, _l in _scan_expr(st.value).refs}
            for s in srcs:
                if s and _tokens(s) & {"fwd", "forward", "future", "fut", "ahead", "next"}:
                    m = re.search(r"(\d+)\s*([dmwh]?)", s)
                    if m:
                        v = int(m.group(1)) * {"": 1, "d": 1, "h": 1, "w": 5, "m": 21}[m.group(2)]
                        best = max(best, v)
                    else:
                        best = max(best, 1)
        return best

    def _assign_by_key_v7(self) -> dict[str, list[ast.Assign]]:
        if not hasattr(self, "_abk7"):
            self._abk7: dict[str, list[ast.Assign]] = {}
            for st in self._stmts_sorted:
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and _key(st.targets[0])[0]:
                    self._abk7.setdefault(_key(st.targets[0])[0], []).append(st)
        return self._abk7

    def _rule_cv_v7(self) -> None:
        for n in self._all:
            if not isinstance(n, ast.Call):
                continue
            fn = _func_name(n)
            if fn in ("cross_val_score", "cross_validate", "cross_val_predict"):
                cv = _kw(n, "cv")
                kfold = cv is None or _num(cv) is not None
                if isinstance(cv, ast.Name):
                    cv = self._assigned_call(cv.id)
                if isinstance(cv, ast.Call) and _func_name(cv) in ("KFold", "StratifiedKFold"):
                    sh = _kw(cv, "shuffle")
                    kfold = not (isinstance(sh, ast.Constant) and sh.value is True) or kfold
                if not kfold:
                    continue
                y = n.args[2] if len(n.args) > 2 else _kw(n, "y")
                if self._v7_y_lead(y) < 1 or self._v7_purged():
                    continue
                self.add("LA06", "medium", n,
                         f"`{fn}` with k-fold folds on time-ordered rows labelled with a forward return: each test fold "
                         "is scored by a model trained on the folds after it. Use `TimeSeriesSplit` with a gap of at "
                         "least the label horizon.")
            elif fn == "range" and isinstance(self.parents.get(n), ast.For) and self.parents[n].iter is n:
                if not hasattr(self, "_v7fitloops"):
                    self._v7fitloops = {id(lp) for c in self._all if isinstance(c, ast.Call) and _attr(c) == "fit"
                                        for lp in self._loops_of(c) if isinstance(lp, ast.For)}
                if id(self.parents[n]) in self._v7fitloops:
                    self._v7_manual_walk_forward(self.parents[n])
        self._v7_search_then_slice()

    def _v7_purged(self) -> bool:
        """The code mentions purging or an embargo (identifiers and strings, not comments); computed on demand."""
        if not hasattr(self, "_v7purge"):
            self._v7purge = bool(self._toks_of(self.tree) & {"purge", "purged", "embargo"})
        return self._v7purge

    def _v7_manual_walk_forward(self, loop: ast.For) -> None:
        if not isinstance(loop.target, ast.Name):
            return
        v = loop.target.id

        def is_v(x):
            return isinstance(x, ast.Name) and x.id == v

        trains, tests = {}, set()
        for st in ast.walk(loop):
            if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
                continue
            for s in ast.walk(st.value):
                if isinstance(s, ast.Subscript) and isinstance(s.slice, ast.Slice):
                    if is_v(s.slice.upper):
                        trains[st.targets[0].id] = st
                    if is_v(s.slice.lower):
                        tests.add(st.targets[0].id)
        if not trains or not tests:
            return
        for c in ast.walk(loop):
            if not (isinstance(c, ast.Call) and _attr(c) == "fit" and len(c.args) >= 2):
                continue
            used = [t for t in trains if t in self.names_in(c.args[0])]
            if not used:
                continue
            lead = self._v7_y_lead(c.args[1])
            if lead < 2 or self._v7_purged():
                continue
            st = trains[used[0]]
            self.add("LA06", "medium", st,
                     f"The training rows run right up to the first test row, but the label looks {lead} bars ahead: the "
                     f"last {lead - 1} training labels overlap the test window. End the training slice {lead} bars "
                     "earlier (purge).")
            return

    _V7_SEARCH = ("GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV", "BayesSearchCV")

    def _v7_search_then_slice(self) -> None:
        for n in self._all:
            if not (isinstance(n, ast.Call) and _attr(n) == "fit" and len(n.args) >= 1):
                continue
            recv = n.func.value
            c = recv if isinstance(recv, ast.Call) else (self._assigned_call(recv.id) if isinstance(recv, ast.Name) else None)
            if c is None or _func_name(c) not in self._V7_SEARCH:
                continue
            if self.inside(n, (ast.For, ast.While, ast.comprehension, ast.Lambda)) is not None:
                continue
            fd = self.inside(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            report, xbase = n, None
            if fd is not None:
                if self._v7_param_of(n.args[0], fd) is None:
                    continue
                sites = [s for s, _f in self._v7_call_sites(fd.name) if self._v7_call_ok(s) and s.args
                         and self._v7_full_arg(s.args[0], s)]
                if not sites:
                    continue
                report = sites[0]
                xbase = self._fit_base(report.args[0])
                st = self._stmt_of(report)
            else:
                if not self._v7_full_arg(n.args[0], n):
                    continue
                xbase = self._fit_base(n.args[0])
                st = self._stmt_of(n)
            if not xbase:
                continue
            holders = set(self._v7_targets(st)[0]) | ({recv.id} if isinstance(recv, ast.Name) and fd is None else set())
            reached = self._v7_reach(holders, set(), st.lineno, self._v7_scope(st)) | holders
            rows_of_x = self._v7_row_subsets(xbase, st.lineno)
            hit = False
            for u in self._all:
                if not (isinstance(u, ast.Call) and getattr(u, "lineno", 0) > st.lineno):
                    continue
                if _attr(u) in ("predict", "score", "predict_proba") and u.args and self.names_in(u.func.value) & reached:
                    a0 = u.args[0]
                    if self._fit_base(a0) in rows_of_x or (self._fit_base(a0) == xbase and any(
                            isinstance(s, ast.Subscript) and not _const_str(s.slice) for s in ast.walk(a0))):
                        hit = True
                        break
                if _attr(u) == "fit" and self.inside(u, (ast.For,)) is not None and any(
                        isinstance(x, ast.keyword) and x.arg is None and self.names_in(x.value) & reached for x in ast.walk(u.func.value)):
                    hit = True
                    break
            if hit:
                self.add("LA05", "medium", report,
                         "The search tunes and refits on *every* row, and the tuned model or its parameters are then "
                         "evaluated on a slice of those same rows. Run the search on the training rows only.")

    def _v7_row_subsets(self, xbase: str, after: int) -> set[str]:
        out: set[str] = set()
        for st in self._stmts_sorted:
            if not isinstance(st, ast.Assign) or st.lineno <= after - 50:
                continue
            tl = st.targets[0].elts if isinstance(st.targets[0], ast.Tuple) else [st.targets[0]]
            vl = st.value.elts if isinstance(st.value, ast.Tuple) and len(st.value.elts) == len(tl) else [st.value] * len(tl)
            for t, v in zip(tl, vl):
                if isinstance(t, ast.Name) and self._fit_base(v) == xbase and any(
                        isinstance(s, ast.Subscript) and isinstance(s.slice, ast.Slice) for s in ast.walk(v)):
                    out.add(t.id)
        return out

    # -- F: point patterns ------------------------------------------------------------------------------------------
    def _rule_points_v7(self) -> None:
        for n in self._all:
            if isinstance(n, ast.Attribute) and n.attr in ("smoothed_state", "smoothed_state_cov", "smoothed_forecasts",
                                                          "smoothed_measurement_disturbance") and isinstance(n.ctx, ast.Load):
                sev = self._v6_flow_sev(n)
                if sev == "high":
                    self.add("LA02", "high", n,
                             f"`.{n.attr}` is the Kalman *smoother's* output, which uses every observation (later ones "
                             "included) for each date. Use `.filtered_state` (and lag it) for anything you trade on.")
            elif isinstance(n, ast.Call) and _func_name(n) == "percentileofscore" and n.args:
                self._v7_percentileofscore(n)
            elif isinstance(n, ast.Call) and _func_name(n) == "qcut" and n.args:
                self._v7_pooled_qcut(n)
            elif isinstance(n, ast.Call) and _attr(n) == "quantile" and n.args and isinstance(n.args[0], (ast.List, ast.Tuple)):
                self._v7_pooled_clip(n)
            elif isinstance(n, ast.Call) and _attr(n) == "rank":
                self._v7_dated_rank(n)
            elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                self._v7_all_time_extreme(n)
            elif isinstance(n, ast.If):
                self._v7_loop_limit(n)

    def _v7_percentileofscore(self, n: ast.Call) -> None:
        lam = self.inside(n, (ast.Lambda,))
        if lam is None or self._in_window_apply(n):
            return
        p = self.parents.get(lam)
        if not (isinstance(p, ast.Call) and _attr(p) in ("apply", "map")):
            return
        h = n.args[0]
        hb = self._v7_frame_of(h)
        hv = self._assigned_value(hb.id, n.lineno) if isinstance(hb, ast.Name) else None
        src = self._v7_frame_of(hv) if hv is not None else hb
        rb = self._v7_frame_of(p.func.value)
        if not (isinstance(src, ast.Name) and isinstance(rb, ast.Name) and src.id == rb.id):
            return
        if hv is not None and not self._v7_full_arg(hv, n):
            return
        st = self._stmt_of(p)
        wn, wk = self._v7_targets(st)
        reached = self._v7_reach(wn, wk, st.lineno, self._v7_scope(st)) | wk | wn
        sev = "medium" if self._v7_decides(reached) else "possible"
        self.add("LA09", sev, n,
                 f"`percentileofscore` ranks each value against `{hb.id if isinstance(hb, ast.Name) else 'the series'}`, "
                 "the *whole* history including later values. Rank against the values before each date (a trailing "
                 "window).")

    def _v7_panel_by_date(self, frame: str) -> bool:
        """`frame` is grouped by a date column somewhere (a panel evaluated date by date)."""
        for c in self._all:
            if isinstance(c, ast.Call) and _attr(c) == "groupby":
                b = self._v7_frame_of(c.func.value)
                if not (isinstance(b, ast.Name) and b.id == frame):
                    continue
                key = c.args[0] if c.args else (_kw(c, "by") or _kw(c, "level"))
                if key is None:
                    continue
                ks = [_const_str(e) for e in (key.elts if isinstance(key, (ast.List, ast.Tuple)) else [key])]
                ks += [_key(e)[0] for e in (key.elts if isinstance(key, (ast.List, ast.Tuple)) else [key]) if isinstance(e, ast.Subscript)]
                if any(k and _tokens(k) & DATE_KEY_T for k in ks):
                    return True
        return False

    def _v7_pooled_qcut(self, n: ast.Call) -> None:
        if self.inside(n, (ast.Lambda, ast.FunctionDef, ast.For, ast.comprehension)) is not None:
            return
        arg = n.args[0]
        fb = self._v7_frame_of(arg)
        if not isinstance(fb, ast.Name) or self._global_stat_recv(arg) is None or not self._v7_full_arg(arg, n):
            return
        st = self._stmt_of(n)
        if not isinstance(st, ast.Assign) or len(st.targets) != 1:
            return
        bk = _key(st.targets[0])[0]
        if not bk:
            return
        for c in self._all:
            if isinstance(c, ast.Call) and _attr(c) == "groupby" and getattr(c, "lineno", 0) > st.lineno:
                key = c.args[0] if c.args else _kw(c, "by")
                if not isinstance(key, (ast.List, ast.Tuple)):
                    continue
                ks = [_const_str(e) or _key(e)[0] for e in key.elts]
                if bk in ks and any(k and _tokens(k) & DATE_KEY_T for k in ks):
                    self.add("LA09", "medium", n,
                             "`pd.qcut` cuts the buckets on every date of the panel at once, so each date's breakpoints "
                             "use later dates' values, while the result is then evaluated date by date. Cut the buckets "
                             "within each date (`groupby(date).transform(lambda s: pd.qcut(s, ...))`).")
                    return

    def _v7_pooled_clip(self, n: ast.Call) -> None:
        vals = [_num(e) for e in n.args[0].elts]
        if len(vals) != 2 or None in vals or not (min(vals) <= 0.1 and max(vals) >= 0.9):
            return
        if self.inside(n, (ast.Lambda, ast.FunctionDef, ast.For, ast.comprehension)) is not None:
            return
        recv = n.func.value
        fb = self._v7_frame_of(recv)
        if not isinstance(fb, ast.Name) or self._global_stat_recv(recv) is None or not self._v7_full_arg(recv, n):
            return
        if not self._v7_panel_by_date(fb.id):
            return
        st = self._stmt_of(n)
        wn, _wk = self._v7_targets(st)
        clips = [c for c in self._all if isinstance(c, ast.Call) and _attr(c) == "clip" and self.names_in(c) & wn
                 and getattr(c, "lineno", 0) > st.lineno]
        if not clips:
            return
        reached: set[str] = set()
        for c in clips:
            reached |= self._v7_flow(c)
        if self._v7_decides(reached) or self._v7_split_later(reached, st.lineno):
            self.add("LA09", "medium", n,
                     f"`{fb.id}` is winsorized at quantiles pooled over *every* date of the panel, so earlier dates are "
                     "clipped with bounds that include later data. Winsorize within each date, or with bounds from the "
                     "training period.")

    def _v7_dated_source(self, name: str, at: int) -> bool:
        v = self._assigned_value(name, at)
        for c in (ast.walk(v) if v is not None else ()):
            if isinstance(c, ast.Call) and _func_name(c) in ("read_csv", "read_parquet", "read_excel"):
                ic = _kw(c, "index_col")
                pdates = _kw(c, "parse_dates")
                if ic is not None and pdates is not None:
                    return True
        return False

    def _v7_dated_rank(self, n: ast.Call) -> None:
        pct = _kw(n, "pct")
        if not (isinstance(pct, ast.Constant) and pct.value is True) or self._global_stat(n, ("rank",)) is None:
            return
        if _kw(n, "axis") is not None or not self._norm_scope_ok(n) or self.inside(n, (ast.FunctionDef,)) is not None:
            return
        if any(_attr(c) in ("groupby", "rolling", "expanding") for c in _chain(n)):
            return
        fb = self._v7_frame_of(n.func.value)
        if not isinstance(fb, ast.Name) or not self._v7_dated_source(fb.id, n.lineno):
            return
        st = self._stmt_of(n)
        wn, wk = self._v7_targets(st)
        if any(classify(k) == "label" for k in wn | wk):
            return
        reached = self._v7_reach(wn, wk, st.lineno, None) | wk | wn
        if self._v7_decides(reached):
            self.add("LA09", "medium", n,
                     "`rank(pct=True)` over the whole dated series ranks each day against every other day, later days "
                     "included. Use an expanding or trailing rank.")

    def _v7_all_time_extreme(self, st: ast.Assign) -> None:
        v = st.value
        if not (isinstance(v, ast.Call) and _attr(v) in ("max", "min") and not v.args and not v.keywords):
            return
        if self.inside(st, (ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef)) is not None:
            return
        recv = self._global_stat(v, ("max", "min"))
        if recv is None or not self._v7_full_arg(recv, st):
            return
        rk = _key(_chain_base(recv))[0]
        if not rk or not (_is_price_name(rk) or (self.info.get(rk) is not None and self.info[rk].price_derived)):
            return
        if _tokens(rk) & DATE_KEY_T:
            return
        peak = st.targets[0].id
        for d in self._stmts_sorted:
            if not (isinstance(d, ast.Assign) and d.lineno > st.lineno and peak in self.names_in(d.value)):
                continue
            if not any(isinstance(b, ast.BinOp) and isinstance(b.op, (ast.Div, ast.Sub)) and peak in self.names_in(b)
                       and rk in {k for k, _l in _scan_expr(b).refs} for b in ast.walk(d.value)):
                continue
            wn, wk = self._v7_targets(d)
            reached = self._v7_reach(wn, wk, d.lineno, None) | wk | wn
            if self._v7_decides(reached, evals=False):
                self.add("LA09", "medium", st,
                         f"`{peak}` is the {_attr(v)} of the *whole* `{rk}` history, and `{d.targets[0].id if isinstance(d.targets[0], ast.Name) else 'a signal'}` "
                         "is measured against it, so early dates know the later peak. Use a running extreme "
                         f"(`{rk}.cum{_attr(v)}()`).")
                return

    def _v7_loop_limit(self, ifs: ast.If) -> None:
        loop = self.inside(ifs, (ast.For, ast.While))
        if loop is None or not isinstance(ifs.test, ast.Compare) or len(ifs.test.ops) != 1:
            return
        op = ifs.test.ops[0]
        a, b = ifs.test.left, ifs.test.comparators[0]
        if isinstance(op, (ast.GtE, ast.Gt)):
            want = "high"
        elif isinstance(op, (ast.LtE, ast.Lt)):
            want = "low"
        else:
            return
        if not (isinstance(a, ast.Subscript) and isinstance(a.value, ast.Name) and isinstance(b, ast.Name)):
            return
        col = _const_str(a.slice)
        if not col or self._extreme_name(col) != want or _tokens(col) & _TIME_WORDS:
            return
        row = a.value.id
        lim = None
        for st in ast.walk(loop):
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) \
                    and st.targets[0].id == b.id and st.lineno < ifs.lineno:
                lim = st
        if lim is None:
            return
        reads = [x for x in ast.walk(lim.value) if isinstance(x, ast.Subscript) and isinstance(x.value, ast.Name)
                 and x.value.id == row and _is_close_name(_const_str(x.slice))]
        if not reads or _scan_expr(lim.value).neg_shift_calls or any(_attr(c) == "shift" for c in ast.walk(lim.value) if isinstance(c, ast.Call)):
            return
        self.add("LA07", "high", ifs.test,
                 f"The limit `{b.id}` is set from this bar's close, but the fill is checked against this same bar's "
                 f"{want}, which happened earlier in the bar. Set the limit at bar t and check it against bar t+1.")

    # -- future-derived series through helpers, class attributes and dicts of frames ----------------------------------
    def _v7_future_feeds(self, reached: set[str], value: ast.AST) -> bool:
        """A future value feeds a decision: a signal-named key or a model input. A forward *return* that is only
        booked against positions (the outcome leg) does not count."""
        mi = self._model_inputs() | self._v7_cv_inputs()
        if any(k in mi and classify(k) not in ("label", "eval", "pnl") for k in reached):
            return True
        same_ratio = any(isinstance(b, ast.BinOp) and isinstance(b.op, ast.Div)
                         and _key(_chain_base(b.left))[0] is not None
                         and _key(_chain_base(b.left))[0] == _key(_chain_base(b.right))[0] for b in ast.walk(value))
        if same_ratio or _scan_expr(value).has_return_op or any(
                isinstance(c, ast.Call) and _attr(c) in ("pct_change", "diff") for c in ast.walk(value)):
            return False  # an outcome return: judged by the position it is multiplied with, not here
        return any(classify(k) == "signal" and not _tokens(k) & PRED_T for k in reached)

    def _rule_future_carriers(self) -> None:
        have = {(f.rule_id, f.line) for f in self.findings if f.severity in ("high", "medium")}
        for n in self._all:
            if not isinstance(n, ast.Call):
                continue
            neg = _shift_sign(n) == -1 and _attr(n) in ("shift", "pct_change", "diff")
            if not neg and isinstance(n.func, ast.Name) and n.func.id in _PARAM_SHIFT:
                idx, pname = _PARAM_SHIFT[n.func.id]
                amt = _kw(n, pname) if _kw(n, pname) is not None else (n.args[idx] if len(n.args) > idx else None)
                neg = amt is not None and _numc(amt) is not None and _numc(amt) < 0
            if not neg:
                continue
            st = self._stmt_of(n)
            d = self.parents.get(n)
            if isinstance(d, ast.Dict) and isinstance(st, ast.Assign) and len(st.targets) == 1 \
                     and isinstance(st.targets[0], ast.Name) and ("LA01", st.lineno) not in have \
                     and ("LA01", n.lineno) not in have:
                key = next((_const_str(k) for k, v in zip(d.keys, d.values) if v is n), None)
                if key and not (classify(key) in ("eval", "pnl", "exec") or _tokens(key) & {
                        "target", "targets", "label", "labels", "y", "outcome", "answer"}):
                    wn, wk = self._v7_targets(st)
                    reached = self._v7_reach(wn, wk, st.lineno, self._v7_scope(st)) | wk | wn
                    if self._v7_future_feeds(reached, n):
                        self.add("LA01", "high", n,
                                 f"The `{key}` entry is built with a negative shift (future data) and reaches a "
                                 "signal, position or model input. Shift by a positive amount.")
                        have.add(("LA01", n.lineno))
                continue
            if _attr(n) not in ("shift", "pct_change", "diff"):
                continue
            if not isinstance(st, ast.Assign) or len(st.targets) != 1 or ("LA01", st.lineno) in have or ("LA10", st.lineno) in have:
                continue
            t = st.targets[0]
            fd = self.inside(st, (ast.FunctionDef, ast.AsyncFunctionDef))
            carrier = None
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                carrier, key = "attribute", t.attr
            elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and _const_str(t.slice) and \
                    isinstance(self._assigned_value(t.value.id, st.lineno), ast.Dict) and fd is None:
                carrier, key = "dict", _const_str(t.slice)
            elif isinstance(t, ast.Name) and fd is not None:
                carrier, key = "helper", t.id
            if carrier is None or classify(key) in ("eval", "pnl", "exec") or _tokens(key) & {
                    "target", "targets", "label", "labels", "y", "outcome", "answer"}:
                continue
            rb = _key(_chain_base(n.func.value))[0]
            if rb and (_date_like(rb) or _tokens(rb) & DATE_KEY_T):
                continue  # dates.shift(-1): tomorrow's calendar date, not data
            wn, wk = self._v7_targets(st)
            if carrier == "dict":
                wn = {t.value.id}
            reached = self._v7_reach(wn, wk, st.lineno, self._v7_scope(st)) | wk | wn
            if carrier == "helper" and not any(x.startswith(self._V7_FN) for x in reached):
                continue
            if self._v7_future_feeds(reached, st.value):
                where = {"attribute": f"the attribute `self.{key}`", "dict": f"the `{key}` entry of `{t.value.id}`" if carrier == "dict" else "",
                         "helper": f"`{key}` inside `{fd.name if fd else ''}()`"}[carrier]
                self.add("LA01", "high", n,
                         f"A negative shift (future data) is stored in {where} and reaches a signal, position or model "
                         "input. Use only data known at the bar (shift by a positive amount).")

    @staticmethod
    def _v7_reversal(x: ast.AST) -> bool:
        return (isinstance(x, ast.Subscript) and isinstance(x.slice, ast.Slice) and _num(x.slice.step) == -1
                and x.slice.lower is None and x.slice.upper is None)

    def _v7_reversed_input(self, e: ast.AST) -> bool:
        """`e` reads a reversed series: s[::-1], s.iloc[::-1], or a local bound to one."""
        for x in ast.walk(e):
            if self._v7_reversal(x):
                return True
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                v = self._assigned_value(x.id, getattr(x, "lineno", 10 ** 9))
                if v is not None and self._v7_reversal(v):
                    return True
        return False

    def _rule_reverse_sandwich(self) -> None:
        """`x[::-1].expanding().mean()[::-1]`, `rev = x.iloc[::-1]; rev.rolling(n).std().iloc[::-1]`,
        `-log_ret(s[::-1], 20)[::-1]`: a window run backwards in time and flipped back, so each value summarizes later
        bars."""
        win = ("rolling", "expanding", "ewm", "cummax", "cummin", "cumsum", "cumprod", "diff", "pct_change", "shift")
        have = {f.line for f in self.findings if f.rule_id == "LA02" and f.severity in ("high", "medium")}
        for n in self._all:
            if not self._v7_reversal(n) or getattr(n, "lineno", 0) in have:
                continue
            inner = n.value.value if isinstance(n.value, ast.Attribute) and n.value.attr == "iloc" else n.value
            ops = [c for c in ast.walk(inner) if isinstance(c, ast.Call) and (
                _attr(c) in win or (isinstance(c.func, ast.Name) and c.func.id in self._funcs and any(
                    isinstance(d, ast.Call) and _attr(d) in win for d in ast.walk(self._funcs[c.func.id]))))]
            if not ops:
                continue
            src = [o for o in ops if self._v7_reversed_input(o.func.value if isinstance(o.func, ast.Attribute) else o)]
            if not src:
                continue
            sev = self._v6_flow_sev(n)
            if sev is None:
                continue
            have.add(n.lineno)
            self.add("LA02", sev, n,
                     "The series is reversed, run through a window, and flipped back, so each value summarizes the bars "
                     "*after* it (a forward-looking window). Apply the window to the series in time order.")

    def _rule_sample_in_helper(self) -> None:
        """`return df.sample(frac=1)` in a helper whose result is then split by position (TimeSeriesSplit)."""
        for n in self._all:
            if not (isinstance(n, ast.Call) and _attr(n) == "sample" and _num(_kw(n, "frac")) == 1):
                continue
            st = self._stmt_of(n)
            if not isinstance(st, ast.Return) or any(f.rule_id == "LA06" and f.line == n.lineno for f in self.findings):
                continue
            reached = self._v7_flow(n)
            if reached & self._v7_cv_inputs():
                self.add("LA06", "medium", n,
                         "This helper shuffles the time-ordered rows (`.sample(frac=1)`), and its result is then split by "
                         "position, so training folds contain rows from after the test rows. Keep the rows in time order.")

    # -- post-hoc extensions (not in the pre-registration; added after reading the dev-set misses, see
    #    RECALL_ANALYSIS_v07.md "Result") --------------------------------------------------------------------------
    def _v7_at_line(self, line: int) -> list[ast.AST]:
        """Calls and loops that start on `line` (indexed once)."""
        if not hasattr(self, "_v7bl"):
            self._v7bl: dict[int, list[ast.AST]] = {}
            for x in self._all:
                if isinstance(x, (ast.Call, ast.For)):
                    self._v7bl.setdefault(x.lineno, []).append(x)
        return self._v7bl.get(line, [])

    def _relocate_posthoc(self) -> None:
        """P1: LA14 on a selection loop is also reported at the test-scoring call inside it. P2: LA10 on a fit is also
        reported where a forward-looking feature it uses is defined."""
        hm_lines = {(g.line, "LA") for g in self.findings if g.rule_id in ("LA10", "LA01") and g.severity in ("high", "medium")}
        for f in list(self.findings):
            if f.same_as or f.severity not in ("high", "medium"):
                continue
            if f.rule_id == "LA14":
                loop = next((x for x in self._v7_at_line(f.line) if isinstance(x, ast.For)), None)
                if loop is None:
                    continue
                c = next((x for x in ast.walk(loop) if isinstance(x, ast.Call) and self._scores_on_test(x)
                          and abs(x.lineno - f.line) > 2), None)
                if c is not None:
                    self.add("LA14", f.severity, c, "The test-set score that the loop selects on is computed here.",
                             same_as=f.line)
            elif f.rule_id == "LA10":
                fit = next((x for x in self._v7_at_line(f.line) if isinstance(x, ast.Call) and _attr(x) == "fit"
                            and x.args), None)
                if fit is None:
                    continue
                cols = set()
                for x in ast.walk(fit.args[0]):
                    if _const_str(x):
                        cols.add(_const_str(x))
                    elif isinstance(x, ast.Name):
                        v = self._assigned_value(x.id, fit.lineno)
                        if isinstance(v, (ast.List, ast.Tuple)):
                            cols |= {_const_str(e) for e in v.elts if _const_str(e)}
                ys = {k for a in fit.args[1:2] for k, _l in _scan_expr(a).refs} | {
                    _const_str(x) for a in fit.args[1:2] for x in ast.walk(a) if _const_str(x)}
                for col in sorted(cols - ys):
                    ki = self.info.get(col)
                    if ki is None or not ki.forward or ki.lagged and not ki.lead or classify(col) in ("pnl", "eval") \
                             or ki.line >= fit.lineno or abs(ki.line - fit.lineno) <= 2:
                        continue
                    if (ki.line, "LA") in hm_lines:
                        continue
                    if not hasattr(self, "_v7stl"):
                        self._v7stl = {}
                        for x in self._stmts_sorted:
                            self._v7stl.setdefault(x.lineno, x)
                    node = self._v7stl.get(ki.line)
                    if node is not None:
                        self.add("LA10", f.severity, node, f"`{col}` is built here from future data and is a feature of "
                                                           f"the model fitted on line {fit.lineno}.", same_as=fit.lineno)
                        hm_lines.add((ki.line, "LA"))
                        break

    def _rule_posthoc_cv(self) -> None:
        """P3: a shuffled K-fold on rows whose label is a forward return, when the file has no time-series hint
        (v0.6 covers files with one). P5: a dict comprehension of test-set scores picked with max/min(d, key=d.get)."""
        for n in self._all:
            if not isinstance(n, ast.Call):
                continue
            fn = _func_name(n)
            if fn in ("KFold", "StratifiedKFold", "RepeatedKFold") and not self.ts_context:
                sh = _kw(n, "shuffle")
                if not (isinstance(sh, ast.Constant) and sh.value is True):
                    continue
                users = [c for c in self._all if isinstance(c, ast.Call) and _func_name(c) in (
                    "cross_val_score", "cross_validate", "cross_val_predict") and len(c.args) > 2]
                if any(self._v7_y_lead(c.args[2]) >= 1 for c in users):
                    self.add("LA06", "medium", n, f"`{fn}(shuffle=True)` mixes rows from every date into each fold, and the "
                                                  "label looks ahead, so training sees the future of the test rows.")
            elif fn in ("max", "min") and n.args and isinstance(n.args[0], ast.Name) and _kw(n, "key") is not None:
                d = n.args[0].id
                key = _kw(n, "key")
                if not (isinstance(key, ast.Attribute) and key.attr == "get" and isinstance(key.value, ast.Name)
                        and key.value.id == d):
                    continue
                v = self._assigned_value(d, n.lineno)
                if not isinstance(v, ast.DictComp):
                    continue
                tn = self._test_names()
                if any(isinstance(c, ast.Call) and self.names_in(c) & tn and not self.names_in(c) & {
                        x for x in self.assigned_names if _tokens(x) & VAL_SET_T} for c in ast.walk(v.value)) \
                         and not any(_tokens(x) & VAL_SET_T for x in self.names_in(v.value)):
                    self.add("LA14", "high", n, f"`{d}` holds each candidate's score *on the test data*, and the best one "
                                                "is kept, so the reported test result is selected on the test set.")

    def rule_v07(self) -> None:
        for rule in (
                self._rule_reverse_sandwich, self._rule_sample_in_helper, self._rule_survivors_v7,
                self._rule_current_files, self._rule_helper_stats, self._rule_full_sample_selection,
                self._rule_pair_selection, self._rule_full_sample_weights, self._rule_cv_v7, self._rule_points_v7,
                self._rule_future_carriers, self._rule_feature_lists, self._rule_posthoc_cv):
            self._v7_guarded(rule)

    def _v7_guarded(self, rule) -> None:
        """Run one v0.7 rule; an unexpected error skips that rule (its findings so far are kept) instead of failing
        the whole check. Recursion and memory errors still reach check_code's text-only fallback."""
        try:
            rule()
        except (RecursionError, MemoryError):
            raise
        except Exception:  # noqa: BLE001 - a rule bug must not take the other rules down
            pass

    # -- CO01: no transaction costs ------------------------------------------------------------
    def _charges_costs(self) -> bool:
        docstrings = {id(x.value) for x in self._all if isinstance(x, ast.Expr) and _const_str(x.value)}
        idents = {x.id for x in self._all if isinstance(x, ast.Name)} | {
            x.attr for x in self._all if isinstance(x, ast.Attribute)} | {
            k.arg for x in self._all if isinstance(x, ast.Call) for k in x.keywords if k.arg} | {
            _const_str(x) for x in self._all if _const_str(x) and id(x) not in docstrings}
        toks = set().union(*(_tokens(s) for s in idents if s)) if idents else set()
        if toks & ((COST_T - {"spread"}) | {"slip", "friction", "impact", "brokerage", "borrow", "haircut"}):
            return True
        # "spread" alone is usually a pairs-trading feature; it is a cost only next to bid/ask/half/cost words
        if "spread" in toks and toks & {"half", "bid", "ask", "quoted", "effective", "crossing"}:
            return True
        for x in self._all:
            right = x.right if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Sub) else (
                x.value if isinstance(x, ast.AugAssign) and isinstance(x.op, ast.Sub) else None)
            if right is None:
                continue
            # `gross - turnover_expr.abs() * 0.0005` charges costs even without a cost-like name
            if any(isinstance(y, ast.Call) and (_attr(y) == "abs" or _func_name(y) == "abs") for y in ast.walk(right)):
                return True
            consts = [_num(y) for y in ast.walk(right) if _num(y) is not None] + [
                self._small_consts().get(y.id) for y in ast.walk(right) if isinstance(y, ast.Name) and y.id in self._small_consts()]
            # `trade * 0.1 / 100`: a percentage cost
            if any(isinstance(y, ast.BinOp) and isinstance(y.op, ast.Div) and _num(y.right) in (100.0, 1e4, 10000.0)
                   for y in ast.walk(right)):
                return True
            # `ret - 0.0005`, `ret - 0.0005 * pos`, `ret - bps / 1e4 * pos`, `pnl - 2.5 * contracts`
            if any(0 < v <= 0.01 or v >= 1000 for v in consts):
                return True
            if isinstance(right, ast.BinOp) and isinstance(right.op, ast.Mult) and consts and any(
                    classify(_key(_chain_base(s))[0]) == "signal" for s in (right.left, right.right)):
                return True
        return False

    def _small_consts(self) -> dict[str, float]:
        """Names assigned a plain number anywhere in the file (k = 0.0005)."""
        if not hasattr(self, "_smc"):
            self._smc = {st.targets[0].id: _num(st.value) for st in self._all
                         if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                         and _num(st.value) is not None}
        return self._smc

    def _reports_performance(self) -> bool:
        idents = {x.id for x in self._all if isinstance(x, ast.Name)} | {
            x.attr for x in self._all if isinstance(x, ast.Attribute)} | {
            _const_str(x) for x in self._all if _const_str(x)}
        toks = set().union(*(_tokens(s) for s in idents if s)) if idents else set()
        if toks & {"sharpe", "sortino", "calmar", "cagr", "annualized", "annualised", "drawdown"}:
            return True
        for x in self._all:
            if isinstance(x, ast.Call) and _attr(x) in ("cumprod", "prod", "sharpe_ratio", "total_return", "stats",
                                                        "annualized_return", "max_drawdown", "run", "plot_cum_returns"):
                return True
            if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Div) and _attr(x.left) == "mean" and _attr(x.right) == "std":
                return True
            if isinstance(x, ast.Call) and _func_name(x) == "print":
                for y in ast.walk(x):
                    k = _key(y)[0] if isinstance(y, (ast.Name, ast.Subscript, ast.Attribute)) else None
                    if k and classify(k) == "pnl":
                        return True
        return False

    @staticmethod
    def _bar_ratio(node: ast.AST) -> bool:
        """closes[i + 1] / closes[i] - 1 style per-bar return inside a loop."""
        for d in ast.walk(node):
            if (isinstance(d, ast.BinOp) and isinstance(d.op, (ast.Div, ast.Sub)) and isinstance(d.left, ast.Subscript)
                    and isinstance(d.right, ast.Subscript) and ast.dump(d.left.value) == ast.dump(d.right.value)
                    and not isinstance(d.left.slice, ast.Slice)):
                return True
        return False

    def _loop_strategy_return(self) -> ast.AST | None:
        rets = {k for k, ki in self.info.items() if ki.is_return}
        for st in self._all:
            if isinstance(st, ast.Assign) and self._bar_ratio(st.value):
                rets |= {t.id for t in st.targets if isinstance(t, ast.Name)}
        for x in self._all:
            if self.inside(x, (ast.For, ast.While)) is None:
                continue
            if isinstance(x, ast.AugAssign) and isinstance(x.op, (ast.Add, ast.Mult)):
                tk, val = _key(x.target)[0], x.value
            elif isinstance(x, ast.Call) and _attr(x) == "append" and x.args:
                tk, val = _key(x.func.value)[0], x.args[0]
            else:
                continue
            if not tk or not (classify(tk) == "pnl" or _tokens(tk) & {"trades", "rets", "returns", "profits", "pnls", "cash"}):
                continue
            if self._bar_ratio(val) or any(isinstance(y, ast.Name) and y.id in rets for y in ast.walk(val)):
                return x
        return None

    def rule_costs(self) -> None:
        # v0.3: missing costs is a realism caveat, not a lookahead red flag, so CO01 is never more than "possible".
        # The message still says when a performance figure is computed from the cost-free returns.
        perf = self._reports_performance()
        sev = "possible"
        tail = (" A performance figure is computed from these cost-free returns; if they already include costs "
                "(e.g. a net return series), ignore this.") if perf else ""
        for n in self._all:
            if isinstance(n, ast.Call) and _attr(n) in ("from_signals", "from_orders", "Backtest") or (
                    isinstance(n, ast.Call) and _func_name(n) == "Backtest"):
                if any(k.arg is None for k in n.keywords):
                    if not self._charges_costs():
                        self.add("CO01", "possible", n, "The engine settings come from `**kwargs`; make sure they include "
                                                        "fees or commission.")
                    return  # **cfg may carry the costs: never more than possible
            if isinstance(n, ast.Call) and _attr(n) in ("from_signals", "from_orders") and not any(
                    k.arg in ("fees", "fixed_fees", "slippage") for k in n.keywords):
                self.add("CO01", sev, n, f"`{_attr(n)}` is called without `fees` or `slippage`, so trades are free.{tail}")
                return
            if isinstance(n, ast.Call) and _func_name(n) == "Backtest" and "backtesting" in self.imports and _kw(n, "commission") is None:
                self.add("CO01", sev, n, f"`Backtest(...)` is created without `commission`, so trades are free.{tail}")
                return
        if self._charges_costs():
            return
        msg = "Strategy returns are computed, but the code never charges commissions, fees, spread or slippage." + tail
        for n in self._all:
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
                roles = {self._side_role(n.left)[0], self._side_role(n.right)[0]}
                if roles & {"return", "forward", "open_return"}:
                    tkey, _f, _s = self.enclosing_assign_target(n)
                    if classify(tkey) in ("pnl", "return", "unknown") and ({"signal", "lagged"} & roles):
                        self.add("CO01", sev, n, msg)
                        return
        node = self._loop_strategy_return()
        if node is not None:
            self.add("CO01", sev, node, msg)


# ---------------------------------------------------------------------------------------------
# v0.4: query/eval strings. `frame.eval("t * b")`, `frame.query("m > 0")` and `pd.eval("a + b")` with a literal
# string are rewritten, before linting, into the code they stand for, so they are judged like the same code written
# out. Names in the string are the frame's columns (`frame["t"]`); `@name` is a local variable; backticks quote a
# column name; `index` is the frame's index. A frame built once from a dict literal (`pd.DataFrame({"t": tilt})`)
# maps its column names back to the expressions they were built from.
# ---------------------------------------------------------------------------------------------
_QE_MAX = 2000
_QE_BACKTICK = re.compile(r"`([^`]*)`")
_QE_AT = re.compile(r"@([A-Za-z_]\w*)")
_QE_KEEP = {"True", "False", "None", "abs", "sin", "cos", "tan", "exp", "log", "log1p", "expm1", "sqrt", "sinh", "cosh",
            "tanh", "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh", "arctan2"}


def _relocate(node: ast.AST, ref: ast.AST) -> ast.AST:
    for x in ast.walk(node):
        x.lineno, x.col_offset = ref.lineno, ref.col_offset
        x.end_lineno, x.end_col_offset = getattr(ref, "end_lineno", ref.lineno), getattr(ref, "end_col_offset", ref.col_offset)
    return node


class _QueryEvalRewriter(ast.NodeTransformer):
    def __init__(self, tree: ast.Module):
        counts: dict[str, int] = {}
        dict_frames: dict[str, dict[str, ast.AST]] = {}
        for n in ast.walk(tree):
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            counts[x.id] = counts.get(x.id, 0) + 1
                if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                        and isinstance(n.value, ast.Call) and _func_name(n.value) == "DataFrame" and n.value.args
                        and isinstance(n.value.args[0], ast.Dict)
                        and all(_const_str(k) is not None for k in n.value.args[0].keys)):
                    d = n.value.args[0]
                    dict_frames[n.targets[0].id] = {_const_str(k): v for k, v in zip(d.keys, d.values)}
        self.cols = {k: v for k, v in dict_frames.items() if counts.get(k) == 1}

    def _parse(self, s: str) -> tuple[str | None, ast.AST, dict] | None:
        if not s or len(s) > _QE_MAX:
            return None
        names: dict[str, tuple[str, str]] = {}

        def bt(m):
            k = f"__qe_col{len(names)}"
            names[k] = ("col", m.group(1))
            return k

        def at(m):
            k = f"__qe_var_{m.group(1)}"
            names[k] = ("var", m.group(1))
            return k

        s2 = _QE_AT.sub(at, _QE_BACKTICK.sub(bt, s.strip()))
        try:
            mod = ast.parse(s2, mode="exec")
        except (SyntaxError, ValueError, RecursionError):
            return None
        if len(mod.body) != 1:
            return None
        st = mod.body[0]
        if isinstance(st, ast.Expr):
            return None, st.value, names
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            return st.targets[0].id, st.value, names
        return None

    def _subst(self, expr: ast.AST, frame: ast.AST | None, names: dict, ref: ast.AST) -> ast.AST:
        fname = frame.id if isinstance(frame, ast.Name) else None
        cols = self.cols.get(fname, {}) if fname else {}
        rw = self

        class Sub(ast.NodeTransformer):
            def visit_Call(self, node):  # abs(x), log(x): keep the function name, rewrite the arguments
                node.args = [self.visit(a) for a in node.args]
                node.keywords = [self.visit(k) for k in node.keywords]
                return node

            def visit_Name(self, node):
                kind, val = names.get(node.id, (None, node.id))
                if kind == "var" or frame is None:
                    return ast.Name(id=val, ctx=ast.Load())
                if kind is None and node.id in _QE_KEEP:
                    return node
                if kind is None and node.id == "index":
                    return ast.Attribute(value=rw._copy(frame, ref), attr="index", ctx=ast.Load())
                if val in cols:
                    return rw._copy(cols[val], ref)
                return ast.Subscript(value=rw._copy(frame, ref), slice=ast.Constant(value=val), ctx=ast.Load())

        return _relocate(Sub().visit(expr), ref)

    @staticmethod
    def _copy(node: ast.AST, ref: ast.AST) -> ast.AST:
        import copy

        return _relocate(copy.deepcopy(node), ref)

    def _target(self, call: ast.Call):
        """(kind, frame, parsed) for X.query("..."), X.eval("..."), pd.eval("..."); None otherwise."""
        a = _attr(call)
        if a not in ("query", "eval") or not isinstance(call.func, ast.Attribute):
            return None
        s = _const_str(call.args[0]) if call.args else _const_str(_kw(call, "expr"))
        parsed = self._parse(s) if s is not None else None
        if parsed is None:
            return None
        recv = call.func.value
        frame = None if (isinstance(recv, ast.Name) and recv.id in ("pd", "pandas")) else recv
        if a == "query" and (frame is None or parsed[0] is not None):
            return None
        return a, frame, parsed

    def _inplace(self, call: ast.Call) -> bool:
        v = _kw(call, "inplace")
        return isinstance(v, ast.Constant) and v.value is True

    def visit_Expr(self, node):
        c = node.value
        if isinstance(c, ast.Call) and self._inplace(c):
            got = self._target(c)
            if got is not None and got[1] is not None:
                kind, frame, (tgt, expr, names) = got
                value = self._subst(expr, frame, names, c)
                if kind == "query":
                    value = _relocate(ast.Subscript(value=self._copy(frame, c), slice=value, ctx=ast.Load()), c)
                    target = self._copy(frame, c)
                    target.ctx = ast.Store()
                elif tgt is not None:
                    target = _relocate(ast.Subscript(value=self._copy(frame, c), slice=ast.Constant(value=tgt),
                                                     ctx=ast.Store()), c)
                else:
                    return self.generic_visit(node)
                return ast.copy_location(ast.Assign(targets=[target], value=value), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        got = self._target(node)
        if got is None or self._inplace(node):
            return node
        kind, frame, (tgt, expr, names) = got
        value = self._subst(expr, frame, names, node)
        if kind == "query":
            return _relocate(ast.Subscript(value=self._copy(frame, node), slice=value, ctx=ast.Load()), node)
        if tgt is not None:  # df.eval("c = a + b") returns df with a new column: df.assign(c=a + b)
            if frame is None:
                return node
            return _relocate(ast.Call(func=ast.Attribute(value=self._copy(frame, node), attr="assign", ctx=ast.Load()),
                                      args=[], keywords=[ast.keyword(arg=tgt, value=value)]), node)
        return value


def _rewrite_query_eval(tree: ast.Module, source: str | None = None) -> ast.Module:
    if source is not None and "query" not in source and "eval" not in source:
        return tree  # cheap text pre-check: no walk of the tree for files that never call query/eval
    if not any(isinstance(n, ast.Call) and _attr(n) in ("query", "eval") for n in ast.walk(tree)):
        return tree
    tree = _QueryEvalRewriter(tree).visit(tree)
    return ast.fix_missing_locations(tree)


def _strip_notebook_magics(source: str) -> str:
    return "\n".join("" if ln.lstrip().startswith(("%", "!")) else ln for ln in source.splitlines())


_FALLBACK_PATTERNS = [
    ("LA01", re.compile(r"\.(shift|pct_change|diff)\(\s*(periods\s*=\s*)?-\s*\w"), "Negative shift found by text search."),
    ("LA02", re.compile(r"center\s*=\s*True"), "`center=True` found by text search."),
    ("LA03", re.compile(r"\.bfill\(|method\s*=\s*[\"'](bfill|backfill)"), "Backward fill found by text search."),
    ("LA06", re.compile(r"train_test_split\((?![^)]*shuffle\s*=\s*False)"), "`train_test_split` without `shuffle=False`."),
    ("LA07", re.compile(r"cheat_on_close\s*=\s*True|set_coc\(\s*True"), "Cheat-on-close fill setting."),
    ("SB01", re.compile(r"wikipedia\.org/wiki/List_of_S%26P_500|tickers_sp500\("), "Universe from today's constituents."),
]


def _fallback(source: str, error: str) -> list[Finding]:
    out = []
    lines = source.splitlines()
    for i, ln in enumerate(lines, 1):
        for rule, rx, msg in _FALLBACK_PATTERNS:
            if rx.search(ln):
                chk = CHECKS[rule]
                out.append(Finding(rule, chk.title, "possible", i, ln.strip()[:160],
                                   msg + " (The code could not be parsed, so this is a text match only.)",
                                   chk.why_it_inflates, chk.how_to_fix))
    return out


def score_findings(findings: list[Finding]) -> tuple[int, list[dict]]:
    """100 minus weighted penalties: high 25, medium 10, possible 3, info 0; at most 2 findings per rule count."""
    by_rule: dict[str, list[Finding]] = {}
    for f in findings:
        if not f.same_as:  # a repeat of another finding, shown at a second line: counted once
            by_rule.setdefault(f.rule_id, []).append(f)
    breakdown = []
    total = 0
    for rule, fs in sorted(by_rule.items()):
        fs_sorted = sorted(fs, key=lambda f: SEVERITIES.index(f.severity))
        pen = sum(PENALTY[f.severity] for f in fs_sorted[:MAX_PENALIZED_PER_RULE])
        total += pen
        breakdown.append({"rule_id": rule, "title": CHECKS[rule].title, "findings": len(fs),
                          "worst_severity": fs_sorted[0].severity, "penalty": pen})
    return max(0, 100 - total), breakdown


def check_code(source: str) -> CodeAudit:
    """Lint Python/pandas strategy source for lookahead, survivorship, cost and overfitting patterns."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string of Python code")
    if len(source) > MAX_SOURCE_CHARS:
        raise ValueError(f"source is too large ({len(source):,} characters; the limit is {MAX_SOURCE_CHARS:,}). "
                         "Send the strategy and backtest code only.")
    notes: list[str] = []
    parsed = True
    too_deep = ("The code is too deeply nested to analyse; only a short list of text patterns was checked, "
                "all reported as 'possible'.")

    def _text_only(note: str) -> CodeAudit:
        notes.append(note)
        findings = _fallback(source, note)
        score, breakdown = score_findings(findings)
        return CodeAudit(findings, score, breakdown, False, notes)

    try:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            try:
                tree = ast.parse(_strip_notebook_magics(source))
                notes.append("Notebook magics (lines starting with % or !) were ignored.")
            except SyntaxError as e:
                return _text_only(f"The code could not be parsed as Python ({e.msg}, line {e.lineno}); "
                                  "only a short list of text patterns was checked, all reported as 'possible'.")
        tree = _rewrite_query_eval(tree, source)
        with _LOCK:
            # v0.5 perf: a lint run allocates many short-lived objects with no reference cycles; in a long-lived
            # process (the MCP server, a test session) cyclic GC passes over a large heap can double the run time.
            # gc.disable() is process-global: other threads keep running with reference counting only (memory is
            # still freed when the last reference goes) until the run ends. Runs are short and serialized by _LOCK;
            # a young-generation collection afterwards clears whatever cyclic garbage accumulated meanwhile.
            import gc

            was_enabled = gc.isenabled()
            gc.disable()
            try:
                findings = _Linter(source, tree).run()
            finally:
                if was_enabled:
                    gc.enable()
                    gc.collect(0)
    except (RecursionError, MemoryError):
        return _text_only(too_deep)
    score, breakdown = score_findings(findings)
    notes.append("Name-based heuristics: columns named like target/label/fwd_*/y are treated as labels, "
                 "signal/position/pred as signals, ret/returns as returns, strategy/pnl as strategy results.")
    return CodeAudit(findings, score, breakdown, parsed, notes)
