"""Suggested fixes: minimal edits, each re-checked by the linter, for a few unambiguous lint findings.

Agents act on what this module returns without asking, so it is deliberately narrow. A fix is proposed only for
these patterns, and only on high or medium findings:

| Rule | Pattern | Edit |
|---|---|---|
| LA01 | a literal negative `shift`/`pct_change`/`diff` in a feature or signal | the same literal, positive |
| LA02 | `rolling(..., center=True)` | `center=False` (a trailing window) |
| LA03 | `.bfill()`, `.backfill()`, `fillna(method="bfill")`, `reindex(..., method="bfill")` | the forward-fill version |
| LA05 | a scaler or model fitted on all rows, with a visible slice split `X[:k]` / `X.iloc[:k]` later | fit on the training slice |
| LA06 | `train_test_split` without `shuffle=False`; `KFold(shuffle=True)` | `shuffle=False`; `TimeSeriesSplit` |

Every fix is re-checked before it is returned: the edit is applied in memory, the result must parse, the code is
linted again, the finding must be gone (no finding of the same rule left in the edited lines) and no high or
medium finding may be new. Anything that fails is dropped. Nothing is suggested when the statement is already
lagged elsewhere, or when there is more than one candidate to change. Returning no fix is the normal outcome.

Edits are line-based: `start_line`..`end_line` (1-based, inclusive) are replaced by `replacement`. `original`
holds the current text of those lines, so a string-replace edit tool can apply the fix too. Edits never change the
number of lines. Code is parsed with `ast` only; it is never imported or run.
"""

from __future__ import annotations

import ast
import builtins
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass

from .lint import check_code

FIXABLE = ("LA01", "LA02", "LA03", "LA05", "LA06")
_ACTIONABLE = ("high", "medium")
_SHIFT_LIKE = ("shift", "pct_change", "diff")
_BUILTINS = set(dir(builtins))


@dataclass
class Edit:
    start_line: int
    end_line: int
    original: str
    replacement: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SuggestedFix:
    rule_id: str
    finding_line: int
    rationale: str
    edits: list[Edit]
    self_checked: bool = True
    file: str | None = None

    def to_dict(self) -> dict:
        d = {"rule_id": self.rule_id, "finding_line": self.finding_line, "rationale": self.rationale,
             "self_checked": self.self_checked, "edits": [e.to_dict() for e in self.edits]}
        if self.file is not None:
            d = {"file": self.file, **d}
        return d


# -- source positions ----------------------------------------------------------------------------------------
class _Src:
    """Source text with line access and AST (UTF-8 byte) column conversion."""

    def __init__(self, source: str):
        self.text = source
        self.lines = source.split("\n")

    def char_col(self, line: int, byte_col: int) -> int:
        return len(self.lines[line - 1].encode("utf-8")[:byte_col].decode("utf-8", errors="replace"))

    def segment(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.text, node) or ""


_Repl = tuple[int, int, int, int, str]  # (line, char_col, end_line, end_char_col, text)


def _repl(src: _Src, node: ast.AST, text: str) -> _Repl:
    return (node.lineno, src.char_col(node.lineno, node.col_offset),
            node.end_lineno, src.char_col(node.end_lineno, node.end_col_offset), text)


def _to_edits(src: _Src, repls: list[_Repl]) -> list[Edit] | None:
    """Turn segment replacements into whole-line edits; None if they overlap or would change the line count."""
    repls = sorted(repls)
    for a, b in zip(repls, repls[1:]):
        if (b[0], b[1]) < (a[2], a[3]):
            return None
    groups: list[list[_Repl]] = []
    for r in repls:
        if groups and r[0] <= max(g[2] for g in groups[-1]):
            groups[-1].append(r)
        else:
            groups.append([r])
    edits = []
    for g in groups:
        s, e = g[0][0], max(r[2] for r in g)
        block = src.lines[s - 1:e]
        text = "\n".join(block)
        offs = [0]
        for ln in block:
            offs.append(offs[-1] + len(ln) + 1)

        def pos(line: int, col: int) -> int:
            return offs[line - s] + col

        for (l1, c1, l2, c2, new) in sorted(g, reverse=True):
            text = text[:pos(l1, c1)] + new + text[pos(l2, c2):]
        if text.count("\n") != len(block) - 1:
            return None
        edits.append(Edit(s, e, "\n".join(block), text))
    return edits


def apply_edits(source: str, edits: list[Edit]) -> str:
    """Apply line edits (from one or several fixes) to `source`. Edits must not overlap."""
    lines = source.split("\n")
    for ed in sorted(edits, key=lambda x: x.start_line, reverse=True):
        old = lines[ed.start_line - 1:ed.end_line]
        cr = old and old[-1].endswith("\r")
        new = ed.replacement.split("\n")
        if cr:
            new = [ln if ln.endswith("\r") else ln + "\r" for ln in new]
        lines[ed.start_line - 1:ed.end_line] = new
    return "\n".join(lines)


# -- statement lookup ----------------------------------------------------------------------------------------
_BODY_FIELDS = {"body", "orelse", "finalbody", "handlers", "cases"}


_INDEX: dict[int, tuple[ast.AST, dict]] = {}  # per-tree indexes; a few recent trees (identity-checked)


def _index(tree: ast.AST) -> dict:
    hit = _INDEX.get(id(tree))
    if hit is not None and hit[0] is tree:
        return hit[1]
    stmts = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)]
    by_line: dict[int, ast.stmt] = {}
    for st in sorted(stmts, key=lambda s: -((s.end_lineno or s.lineno) - s.lineno)):  # widest first, innermost wins
        for ln in range(st.lineno, (st.end_lineno or st.lineno) + 1):
            by_line[ln] = st
    body_of: dict[int, list] = {}
    for n in ast.walk(tree):
        for f in ("body", "orelse", "finalbody"):
            body = getattr(n, f, None)
            if isinstance(body, list):
                for s in body:
                    body_of[id(s)] = body
    idx = {"stmts": stmts, "by_line": by_line, "body_of": body_of}
    if len(_INDEX) > 8:
        _INDEX.clear()
    _INDEX[id(tree)] = (tree, idx)
    return idx


def release_index() -> None:
    """Drop cached syntax-tree indexes (they keep large trees alive, which slows later garbage collection)."""
    _INDEX.clear()


def _stmts(tree: ast.AST) -> list[ast.stmt]:
    return _index(tree)["stmts"]


def stmt_for_line(tree: ast.AST, line: int) -> ast.stmt | None:
    """The innermost statement whose span contains `line`."""
    return _index(tree)["by_line"].get(line)


def header_nodes(st: ast.stmt) -> list[ast.AST]:
    """The statement itself if simple; for compound statements, only the header expressions (not the body)."""
    if not any(hasattr(st, f) and getattr(st, f) for f in _BODY_FIELDS):
        return [st]
    out = []
    for name, val in ast.iter_fields(st):
        if name in _BODY_FIELDS or name == "decorator_list":
            continue
        if isinstance(val, ast.AST):
            out.append(val)
        elif isinstance(val, list):
            out += [v for v in val if isinstance(v, ast.AST)]
    return out


def line_range(tree: ast.AST | None, line: int) -> tuple[int, int]:
    """Exact line range for a finding: its statement, or the header of a compound statement."""
    if tree is None or line <= 0:
        return line, line
    st = stmt_for_line(tree, line)
    if st is None:
        return line, line
    hdr = header_nodes(st)
    if hdr == [st]:
        return st.lineno, st.end_lineno or st.lineno
    end = max([getattr(n, "end_lineno", st.lineno) or st.lineno for n in hdr] + [st.lineno])
    return st.lineno, end


def _walk_header(st: ast.stmt):
    for n in header_nodes(st):
        yield from ast.walk(n)


# -- helpers -------------------------------------------------------------------------------------------------
def _int_value(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) \
            and type(node.operand.value) is int:
        return -node.operand.value
    return None


def _kw(call: ast.Call, name: str) -> ast.keyword | None:
    return next((k for k in call.keywords if k.arg == name), None)


def _periods_node(call: ast.Call) -> ast.AST | None:
    k = _kw(call, "periods")
    if k is not None:
        return k.value
    return call.args[0] if call.args else None


def _shift_calls(nodes) -> list[ast.Call]:
    return [n for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in _SHIFT_LIKE]


def _attr_name_repl(src: _Src, attr: ast.Attribute, new: str) -> _Repl | None:
    """Replace just the attribute name of `x.attr` (checked against the text)."""
    if attr.end_lineno is None:
        return None
    line, end = attr.end_lineno, src.char_col(attr.end_lineno, attr.end_col_offset)
    start = end - len(attr.attr)
    if start < 0 or src.lines[line - 1][start:end] != attr.attr:
        return None
    return (line, start, line, end, new)


def _base_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _free_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _assigned_before(tree: ast.AST, line: int) -> set[str]:
    out = set()
    for st in _stmts(tree):
        if st.lineno >= line:
            continue
        if isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For)):
            targets = st.targets if isinstance(st, ast.Assign) else [st.target]
            for t in targets:
                out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
        elif isinstance(st, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in st.names}
        elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(st.name)
    return out


_FIT_Y_ATTRS = {"fit", "fit_transform", "partial_fit", "score", "fit_predict"}
_CV_FUNCS = {"cross_val_score", "cross_validate", "cross_val_predict", "learning_curve", "validation_curve",
             "permutation_test_score"}
_LABEL_NAME_T = {"y", "target", "targets", "label", "labels", "ytrain", "ytest", "fwd", "forward", "future", "ahead",
                 "outcome", "outcomes"}
_DATE_KEY = re.compile(r"(date|time|^dt$|_dt$|^dt_|timestamp|period|month|year|day)", re.I)


def _assigned_keys(st: ast.stmt) -> set[str]:
    """Names and string column keys assigned by a statement (`x = ...`, `df['x'] = ...`, `df.x = ...`)."""
    targets = st.targets if isinstance(st, ast.Assign) else [st.target] if isinstance(st, (ast.AugAssign, ast.AnnAssign)) else []
    out = set()
    for t in targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
            elif isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store):
                sl = n.slice
                elts = sl.elts if isinstance(sl, (ast.List, ast.Tuple)) else [sl]
                out |= {e.value for e in elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store):
                out.add(n.attr)
    return out


def _mentions(node: ast.AST, keys: set[str]) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in keys:
            return True
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in keys:
            return True
        if isinstance(n, ast.Attribute) and n.attr in keys:
            return True
    return False


def _label_like(key: str) -> bool:
    k = key.lower()
    return k.replace("_", "") in _LABEL_NAME_T or bool(set(k.split("_")) & _LABEL_NAME_T)


def _derived(tree: ast.AST, keys: set[str], after: int, frames: set[str] = frozenset()) -> set[str]:
    """`keys` plus every name or column assigned after line `after` from an expression that mentions them."""
    out = set(keys)
    later = sorted((s for s in _stmts(tree) if isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                    and s.lineno > after and getattr(s, "value", None) is not None), key=lambda s: s.lineno)
    for _ in range(4):
        grew = False
        for s in later:
            new = _assigned_keys(s) - out - set(frames)
            if new and _mentions(s.value, out):
                out |= new
                grew = True
        if not grew:
            break
    return out


def _frames(tree: ast.AST) -> set[str]:
    """Names used as frames (`df[...] = ...`): assigning a column doesn't make the whole frame derived from it."""
    return {_base_name(t) for s in _stmts(tree) if isinstance(s, ast.Assign) for t in s.targets
            if isinstance(t, ast.Subscript)} - {None}


def _used_as_label(tree: ast.AST, keys: set[str], after: int) -> bool:
    """Is any of `keys`, or anything later derived from them, used as a model's target (y of fit/split/CV, a
    metric, or assigned to a label-like name such as `y`, `target`, `label`, `fwd_*`)?

    The lint may flag a forward-looking or centered column whose name it doesn't recognize as a label. Changing it
    would quietly change what the model predicts, so such columns are never auto-fixed."""
    if not keys:
        return False
    keys = _derived(tree, keys, after, _frames(tree))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else ""
            cands: list[ast.AST] = []
            if fn in _FIT_Y_ATTRS:
                cands = n.args[1:] + [k.value for k in n.keywords if k.arg in ("y", "Y")]
            elif fn == "train_test_split":
                cands = n.args[1:] + [k.value for k in n.keywords if k.arg == "stratify"]
            elif fn in _CV_FUNCS:
                cands = n.args[2:] + [k.value for k in n.keywords if k.arg == "y"]
            elif fn in ("OLS", "WLS", "GLS", "Logit", "Probit", "polyfit"):
                cands = list(n.args[:2]) if fn != "polyfit" else list(n.args[1:2])
            elif fn.endswith(("_score", "_error", "_loss")) or fn in ("confusion_matrix", "classification_report"):
                cands = list(n.args) + [k.value for k in n.keywords]
            if any(_mentions(c, keys) for c in cands):
                return True
        elif isinstance(n, (ast.Assign, ast.AnnAssign)) and n.lineno > after and n.value is not None:
            if any(_label_like(k) for k in _assigned_keys(n)) and _mentions(n.value, keys):
                return True
    return False


def _later_stmts(tree: ast.AST, st: ast.stmt) -> list[ast.stmt]:
    end = st.end_lineno or st.lineno
    return [s for s in _stmts(tree) if s.lineno > end and not any(getattr(s, f, None) for f in _BODY_FIELDS)]


def _series_keys(node: ast.AST) -> set[str]:
    """Column or variable names a series expression reads: `close`, `df['close']`, `df.close`, `df.groupby(k)[c]`."""
    if isinstance(node, ast.Name):
        return {node.id}
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript):
            sl = n.slice
            elts = sl.elts if isinstance(sl, (ast.List, ast.Tuple)) else [sl]
            out |= {e.value for e in elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.add(n.value)
    if isinstance(node, ast.Attribute):
        out.add(node.attr)
    return out


def _plain_series(node: ast.AST) -> bool:
    """A name, column or attribute access, optionally through `groupby(...)` — no computation."""
    while True:
        if isinstance(node, ast.Name):
            return True
        if isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "groupby":
            node = node.func.value
        else:
            return False


# -- per-rule proposers: each returns (segment replacements, rationale) or None ------------------------------
def _fix_la01(src: _Src, tree: ast.AST, st: ast.stmt):
    """Only `target = x.shift(-k)`, with `x` a plain column or variable, whose result is never combined with the
    unshifted `x` later and is never lagged again. Comparing, subtracting or dividing a lead by `x` (`x.shift(-1) >
    x`, `x.shift(-1) / x - 1`) has a meaning that a mirrored shift would flip, and `pct_change(-k)`/`diff(-k)` are such
    combinations by construction, so none of those get a fix."""
    if not (isinstance(st, ast.Assign) and len(st.targets) == 1):
        return None
    c = st.value
    if not (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "shift"):
        return None
    if any(k.arg != "periods" for k in c.keywords) or len(c.args) > 1:
        return None  # freq=, axis=, fill_value= change what the shift means
    p = _periods_node(c)
    v = _int_value(p)
    if v is None or v >= 0:
        return None
    recv = c.func.value
    if not _plain_series(recv):
        return None
    keys, base = _assigned_keys(st), _series_keys(recv)
    if not keys or keys & base or _used_as_label(tree, keys, st.lineno):
        return None
    later = _later_stmts(tree, st)
    for s in later:
        if _mentions(s, keys) and _mentions(s, base):
            return None  # the lead is combined with the unshifted series: mirroring would flip the meaning
    # Is the result (or anything built from it) lagged again later, e.g. `position.shift(1)`? Then flipping the
    # shift would lag twice; the minimal fix is to drop the lead and keep the later lag as the only one.
    derived = _derived(tree, keys, st.lineno, _frames(tree))
    lagged = False
    for s in later:
        for sc in _shift_calls(ast.walk(s)):
            if sc.func.attr != "shift" or not _mentions(sc.func.value, derived):
                continue
            pv = _int_value(_periods_node(sc)) if (_periods_node(sc) is not None or sc.keywords) else 1
            if pv is None or any(k.arg not in ("periods",) for k in sc.keywords):
                return None  # a lag we can't read: can't tell what the minimal fix is
            if pv > 0:
                lagged = True
    if lagged:
        seg = src.segment(recv)
        if not seg:
            return None
        return ([_repl(src, c, seg)],
                f"`shift({v})` reads {-v} bar(s) ahead, and the result is already lagged later in the code, so the "
                "lead isn't needed: dropping it leaves that later lag as the only one (flipping it would lag twice).")
    return ([_repl(src, p, str(-v))],
            f"`shift({v})` reads {-v} bar(s) ahead; `shift({-v})` uses the value from {-v} bar(s) ago.")


def _fix_la02(src: _Src, tree: ast.AST, st: ast.stmt):
    if _used_as_label(tree, _assigned_keys(st), st.lineno):
        return None  # centered windows are legitimate in labels
    cands = [n for n in _walk_header(st) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "rolling"]
    hits = [(c, k) for c in cands for k in c.keywords if k.arg == "center"
            and isinstance(k.value, ast.Constant) and k.value.value is True]
    if len(hits) != 1:
        return None
    _c, k = hits[0]
    return ([_repl(src, k.value, "False")],
            "A centered window averages future bars into today's value; a trailing window (`center=False`) uses only "
            "past bars.")


# Ordering: a positional fill (bfill/ffill) only reads "the future" when rows are in ascending time order. On
# newest-first data, bfill pulls past values and ffill is the leak, so LA03 is fixed only where ascending order
# is visible: a resample, an ascending sort_index(), or an ascending sort_values() whose last key is a date.
_ORDER_KEEP = {"groupby", "astype", "fillna", "ffill", "bfill", "pad", "backfill", "copy", "rename", "assign",
               "set_index", "reset_index", "drop", "dropna", "loc", "iloc", "last", "first", "mean", "sum", "ohlc",
               "agg", "max", "min", "median", "std", "rolling", "shift", "pct_change", "diff", "join", "where",
               "mask", "clip", "abs", "round", "tz_localize", "tz_convert", "to_frame", "squeeze", "filter",
               "add_prefix", "add_suffix", "replace", "apply", "transform"}


def _sort_direction(call: ast.Call, date_only: bool = False) -> str:
    """'asc', 'desc' or 'unknown' for a sort_index/sort_values call (sort_values must end on a date-like key;
    with `date_only`, the date must be the only key: a panel sorted by [ticker, date] is not in time order)."""
    asc = _kw(call, "ascending")
    if any(k.arg == "axis" for k in call.keywords):
        return "unknown"
    if asc is None:
        direction = "asc"
    elif isinstance(asc.value, ast.Constant) and isinstance(asc.value.value, bool):
        direction = "asc" if asc.value.value else "desc"
    elif isinstance(asc.value, (ast.List, ast.Tuple)) and asc.value.elts and \
            isinstance(asc.value.elts[-1], ast.Constant) and isinstance(asc.value.elts[-1].value, bool):
        direction = "asc" if asc.value.elts[-1].value else "desc"
    else:
        return "unknown"
    if call.func.attr == "sort_values":
        by = _kw(call, "by").value if _kw(call, "by") else (call.args[0] if call.args else None)
        if date_only and isinstance(by, (ast.List, ast.Tuple)) and len(by.elts) != 1:
            return "unknown"
        last = by.elts[-1] if isinstance(by, (ast.List, ast.Tuple)) and by.elts else by
        if not (isinstance(last, ast.Constant) and isinstance(last.value, str) and _DATE_KEY.search(last.value)):
            return "unknown"
    return direction


def _chain_order(node: ast.AST, start: str, date_only: bool = False) -> str:
    """Row order after evaluating a method chain, starting from `start` for its base."""
    ops: list[ast.AST] = []
    cur = node
    while True:
        if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
            ops.append(cur)
            cur = cur.func.value
        elif isinstance(cur, ast.Subscript):
            ops.append(cur)
            cur = cur.value
        elif isinstance(cur, ast.Attribute):
            cur = cur.value
        else:
            break
    state = start
    for op in reversed(ops):
        if isinstance(op, ast.Subscript):
            sl = op.slice
            if isinstance(sl, ast.Slice) and sl.step is not None:
                state = "unknown" if _int_value(sl.step) != 1 else state
            continue
        a = op.func.attr
        if a in ("sort_index", "sort_values"):
            state = _sort_direction(op, date_only)
        elif a == "groupby" and date_only:
            state = "unknown"
        elif a in ("resample", "asfreq"):
            state = "asc"
        elif a not in _ORDER_KEEP:
            state = "unknown"
    return state


def _scope_body(tree: ast.AST, st: ast.stmt) -> list[ast.stmt] | None:
    return _index(tree)["body_of"].get(id(st))


def _value_for(s: ast.stmt, name: str) -> ast.AST | None | bool:
    """The expression assigned to `name` by statement `s` (also `X, y = a, b`); False if `s` doesn't assign it;
    None if it does but the value can't be matched (e.g. tuple unpacking from a call)."""
    if not isinstance(s, ast.Assign):
        return False
    for t in s.targets:
        if isinstance(t, ast.Name) and t.id == name:
            return s.value
        if isinstance(t, (ast.Tuple, ast.List)) and any(isinstance(e, ast.Name) and e.id == name for e in t.elts):
            i = next(i for i, e in enumerate(t.elts) if isinstance(e, ast.Name) and e.id == name)
            if isinstance(s.value, (ast.Tuple, ast.List)) and len(s.value.elts) == len(t.elts):
                return s.value.elts[i]
            return None
    return False


def _name_order(tree: ast.AST, name: str, before: ast.stmt, depth: int = 0, date_only: bool = False) -> str:
    """Row order of variable `name` just before statement `before`, from the same block's earlier statements."""
    body = _scope_body(tree, before)
    if body is None or depth > 12:
        return "unknown"
    for s in reversed(body[:body.index(before)]):
        val = _value_for(s, name)
        if val is None:
            return "unknown"
        if val is not False:
            b = _base_name(_chain_root(val))
            start = _name_order(tree, b, s, depth + 1, date_only) if b else "unknown"  # `df = df.x()` too
            return _chain_order(val, start, date_only)
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute) \
                and _base_name(s.value.func.value) == name and _kw(s.value, "inplace") is not None:
            a = s.value.func.attr
            if a in ("sort_index", "sort_values"):
                return _sort_direction(s.value, date_only)
            if a not in _ORDER_KEEP:
                return "unknown"
        if not isinstance(s, (ast.Assign, ast.Expr, ast.AugAssign, ast.AnnAssign, ast.Import, ast.ImportFrom)):
            if _mentions(s, {name}):
                return "unknown"  # a loop, branch or function that may reorder it
    return "unknown"


def _chain_root(node: ast.AST) -> ast.AST:
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
        elif isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        else:
            return node


def _fill_axis_ok(call: ast.Call) -> bool:
    ax = _kw(call, "axis")
    return ax is None or (isinstance(ax.value, ast.Constant) and ax.value.value in (0, "index", "rows"))


def _fix_la03(src: _Src, tree: ast.AST, st: ast.stmt):
    repls = []
    for n in _walk_header(st):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
            continue
        a = n.func.attr
        if a in ("bfill", "backfill", "fillna") and (a != "fillna" or _kw(n, "method") is not None):
            if not _fill_axis_ok(n):
                return None  # a fill across columns isn't a time leak; leave it
            root = _base_name(_chain_root(n.func.value))
            start = _name_order(tree, root, st) if root else "unknown"
            if _chain_order(n.func.value, start) != "asc":
                return None  # can't show rows are in ascending time order here, so ffill might be the leak
        if a in ("bfill", "backfill"):
            r = _attr_name_repl(src, n.func, "ffill")
            if r is None:
                return None
            repls.append(r)
        elif a == "fillna":
            m = _kw(n, "method")
            if m is None:
                continue
            if not (isinstance(m.value, ast.Constant) and m.value.value in ("bfill", "backfill")) or n.args \
                    or _kw(n, "value") is not None:
                return None
            others = [src.segment(k) for k in n.keywords if k is not m]
            if any(not s for s in others):
                return None
            start = _attr_name_repl(src, n.func, "")
            if start is None:
                return None
            repls.append((start[0], start[1], n.end_lineno, src.char_col(n.end_lineno, n.end_col_offset),
                          f"ffill({', '.join(others)})"))
        elif a == "reindex":
            m = _kw(n, "method")
            if m is not None and isinstance(m.value, ast.Constant) and m.value.value in ("bfill", "backfill"):
                # reindex fills by index value (the index must be monotonic), not by row position, so the
                # direction is about time regardless of sort order. Only for the row index.
                if _kw(n, "columns") is not None or not _fill_axis_ok(n) or len(n.args) > 1:
                    return None
                q = src.segment(m.value)[:1]
                if q not in ("'", '"'):
                    return None
                repls.append(_repl(src, m.value, f"{q}ffill{q}"))
        elif a == "interpolate":
            return None  # interpolation also reads ahead, but there's no one-for-one replacement
    if len(repls) != 1:
        return None
    return (repls, "Backward fill copies a later value into earlier rows; forward fill only carries known values "
                   "forward (rows here are visibly in ascending time order). Leading rows stay missing.")



_SPLIT_FUNCS = ("train_test_split",)
_PANEL_KEYS = {"ticker", "tickers", "symbol", "symbols", "asset", "assets", "permno", "permco", "cusip", "isin",
               "sedol", "gvkey", "secid", "sec_id", "instrument", "instruments", "stock", "stocks", "firm", "firms",
               "company", "companies", "security", "securities", "coin", "coins", "pair", "pairs"}


def _is_panel(tree: ast.AST) -> bool:
    """Visible signs of panel data (many assets per date): an asset-like column or name, or a groupby on a key that
    could be a date."""
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.lower() in _PANEL_KEYS:
            return True
        if isinstance(n, (ast.Name, ast.Attribute)):
            nm = n.id if isinstance(n, ast.Name) else n.attr
            if nm.lower() in _PANEL_KEYS:
                return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "groupby":
            return True  # grouping by asset, or by date (many rows per date): either way, not one time series
    return False


def _rows_by_date_alone(tree: ast.AST, expr: ast.AST, at: ast.stmt, depth: int = 0) -> bool:
    """Is `expr` (the X of a split) visibly ordered by date alone, ascending, just before statement `at`?"""
    if depth > 6:
        return False
    if isinstance(expr, ast.Name):
        body = _scope_body(tree, at)
        if body is None:
            return False
        for s in reversed(body[:body.index(at)]):
            val = _value_for(s, expr.id)
            if val is None:
                return False
            if val is not False:
                return _rows_by_date_alone(tree, val, s, depth + 1)
        return False
    root = _base_name(_chain_root(expr))
    if root is None:
        return False
    start = _name_order(tree, root, at, date_only=True)
    return _chain_order(expr, start, date_only=True) == "asc"


_CV_USERS = {"cross_val_score", "cross_validate", "cross_val_predict"}


def _cv_data(tree: ast.AST, st: ast.stmt, kfold: ast.Call) -> list[tuple[ast.AST, ast.stmt]]:
    """The X each use of this KFold splits: `cross_val_*(model, X, y, cv=...)` or `cv.split(X)`. Empty if unknown."""
    out: list[tuple[ast.AST, ast.stmt]] = []
    names: set[str] = set()
    if isinstance(st, ast.Assign) and st.value is kfold and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
        names.add(st.targets[0].id)
    for s in _stmts(tree):
        if any(getattr(s, f, None) for f in _BODY_FIELDS):
            continue
        for n in ast.walk(s):
            if not isinstance(n, ast.Call):
                continue
            fn = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else ""
            cv = _kw(n, "cv")
            uses = cv is not None and (cv.value is kfold or (isinstance(cv.value, ast.Name) and cv.value.id in names))
            if fn in _CV_USERS and uses:
                if len(n.args) < 2:
                    return []
                out.append((n.args[1], s))
            elif uses:
                return []  # GridSearchCV and friends: the data is passed elsewhere
            elif fn == "split" and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                    and n.func.value.id in names:
                if not n.args:
                    return []
                out.append((n.args[0], s))
    return out


def _fix_la06(src: _Src, tree: ast.AST, st: ast.stmt):
    calls = [n for n in _walk_header(st) if isinstance(n, ast.Call)]
    tts = [c for c in calls if (isinstance(c.func, ast.Name) and c.func.id == "train_test_split")
           or (isinstance(c.func, ast.Attribute) and c.func.attr == "train_test_split")]
    kf = [c for c in calls if (isinstance(c.func, ast.Name) and c.func.id == "KFold")
          or (isinstance(c.func, ast.Attribute) and c.func.attr == "KFold")]
    if len(tts) + len(kf) != 1:
        return None
    if _is_panel(tree):
        return None  # panel rows (many assets): an unshuffled split by position isn't a split by time
    if tts:
        c = tts[0]
        if _kw(c, "stratify") is not None or any(k.arg is None for k in c.keywords):
            return None  # stratify needs shuffling; **kwargs may carry shuffle
        if not c.args or not _rows_by_date_alone(tree, c.args[0], st):
            return None  # only when the rows are visibly ordered by date alone
        sh = _kw(c, "shuffle")
        if sh is not None:
            if not (isinstance(sh.value, ast.Constant) and sh.value.value is True):
                return None
            return ([_repl(src, sh.value, "False")],
                    "A shuffled split puts later rows in the training set; `shuffle=False` keeps the test set after "
                    "the training set in time.")
        last = (c.keywords[-1].value if c.keywords else (c.args[-1] if c.args else None))
        if last is None:
            return None
        end_line, end_col = last.end_lineno, src.char_col(last.end_lineno, last.end_col_offset)
        return ([(end_line, end_col, end_line, end_col, ", shuffle=False")],
                "`train_test_split` shuffles by default, which puts later rows in the training set; `shuffle=False` "
                "keeps the test set after the training set in time.")
    c = kf[0]
    sh = _kw(c, "shuffle")
    if sh is None or not (isinstance(sh.value, ast.Constant) and sh.value.value is True):
        return None
    data = _cv_data(tree, st, c)
    if not data or not all(_rows_by_date_alone(tree, x, at) for x, at in data):
        return None  # TimeSeriesSplit is only right when the rows it splits are ordered by date alone
    if any(k.arg not in ("n_splits", "shuffle", "random_state") for k in c.keywords) or len(c.args) > 1:
        return None
    n = c.args[0] if c.args else (_kw(c, "n_splits").value if _kw(c, "n_splits") else None)
    args = f"n_splits={src.segment(n)}" if n is not None else ""
    if n is not None and not src.segment(n):
        return None
    repls: list[_Repl] = []
    if isinstance(c.func, ast.Attribute):  # ms.KFold(...) -> ms.TimeSeriesSplit(...): no import needed
        new_func = f"{src.segment(c.func.value)}.TimeSeriesSplit"
        if not src.segment(c.func.value):
            return None
    else:
        new_func = "TimeSeriesSplit"
        imp = [s for s in _stmts(tree) if isinstance(s, ast.ImportFrom) and s.module == "sklearn.model_selection"
               and any(a.name == "KFold" and a.asname is None for a in s.names)]
        already = any(isinstance(s, ast.ImportFrom) and s.module == "sklearn.model_selection"
                      and any(a.name == "TimeSeriesSplit" and a.asname is None for a in s.names) for s in _stmts(tree))
        if not already:
            if len(imp) != 1 or imp[0].lineno != imp[0].end_lineno or imp[0].lineno >= c.lineno:
                return None
            last = imp[0].names[-1]
            if getattr(last, "end_lineno", None) != imp[0].lineno:
                return None
            line, col = last.end_lineno, src.char_col(last.end_lineno, last.end_col_offset)
            repls.append((line, col, line, col, ", TimeSeriesSplit"))
    repls.append(_repl(src, c, f"{new_func}({args})"))
    return (repls, "Shuffled k-fold trains on folds that come after the test fold; `TimeSeriesSplit` always tests on "
                   "data after the training window.")


_FIT_ATTRS = ("fit", "fit_transform")


def _slices_after(tree: ast.AST, names: set[str], after: int) -> tuple[set[str], set[str], str | None]:
    """Upper bounds of `[:k]` train slices and lower bounds of `[k:]` test slices on `names` after a line.

    Returns (train_bounds, test_bounds, accessor) with the bounds as source text; accessor is 'iloc' or ''."""
    train, test, accessors = set(), set(), set()
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice) and getattr(n, "lineno", 0) > after):
            continue
        v, acc = n.value, ""
        if isinstance(v, ast.Attribute) and v.attr == "iloc":
            v, acc = v.value, "iloc"
        if not (isinstance(v, ast.Name) and v.id in names):
            continue
        sl = n.slice
        if sl.step is not None:
            continue
        if sl.lower is None and sl.upper is not None:
            train.add(ast.unparse(sl.upper))
            accessors.add(acc)
        elif sl.upper is None and sl.lower is not None:
            test.add(ast.unparse(sl.lower))
            accessors.add(acc)
    return train, test, (accessors.pop() if len(accessors) == 1 else None)


def _fix_la05(src: _Src, tree: ast.AST, st: ast.stmt):
    calls = [n for n in _walk_header(st) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in _FIT_ATTRS]
    if len(calls) != 1:
        return None
    c = calls[0]
    if c.keywords or not 1 <= len(c.args) <= 2 or any(isinstance(a, ast.Starred) for a in c.args):
        return None
    if c.func.attr == "fit_transform" and len(c.args) != 1:
        return None
    bases = [_base_name(a) for a in c.args]
    if any(b is None for b in bases):
        return None
    related = set(bases)
    if isinstance(st, ast.Assign) and len(st.targets) == 1:
        t = _base_name(st.targets[0])
        if t:
            related.add(t)
    # names later built from <est>.transform(<arg>)
    est = c.func.value
    est_src = ast.unparse(est)
    est_name = est.id if isinstance(est, ast.Name) else (
        st.targets[0].id if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
        and c.func.attr == "fit" else None)
    for s in _stmts(tree):
        if s.lineno > st.lineno and isinstance(s, ast.Assign) and len(s.targets) == 1 and \
                isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Call) and \
                isinstance(s.value.func, ast.Attribute) and s.value.func.attr == "transform" and \
                ast.unparse(s.value.func.value) in {est_src, est_name}:
            related.add(s.targets[0].id)
    train, test, acc = _slices_after(tree, related, st.lineno)
    bounds = train | test
    if len(bounds) != 1 or acc is None:
        return None
    k = bounds.pop()
    k_node = ast.parse(k, mode="eval").body
    known = _assigned_before(tree, st.lineno) | _BUILTINS
    if not _free_names(k_node) <= known:
        return None
    # with X and y, each must itself be split at the same k later
    if len(bases) > 1:
        for b in bases:
            tr, te, _ = _slices_after(tree, {b}, st.lineno)
            if (tr | te) != {k}:
                return None
    sub = ".iloc" if acc == "iloc" else ""
    new_args = [f"{src.segment(a)}{sub}[:{k}]" for a in c.args]
    if any(not src.segment(a) for a in c.args):
        return None
    if c.func.attr == "fit_transform":
        arg = src.segment(c.args[0])
        attr = _attr_name_repl(src, c.func, "")
        if attr is None:
            return None
        repl = (attr[0], attr[1], c.end_lineno, src.char_col(c.end_lineno, c.end_col_offset),
                f"fit({new_args[0]}).transform({arg})")
        return ([repl], "Fitting on all rows lets test-period statistics shape the training data; fitting on the "
                        f"training slice (`[:{k}]`, as in the split below) and transforming everything does not.")
    repls = [_repl(src, a, na) for a, na in zip(c.args, new_args)]
    return (repls, "Fitting on all rows lets test-period data shape the fit; fitting on the training slice "
                   f"(`[:{k}]`, as in the split below) does not.")


_PROPOSERS = {"LA01": _fix_la01, "LA02": _fix_la02, "LA03": _fix_la03, "LA05": _fix_la05, "LA06": _fix_la06}


# -- verification --------------------------------------------------------------------------------------------
def _actionable_counts(findings) -> Counter:
    return Counter((f.rule_id, f.severity) for f in findings if f.severity in _ACTIONABLE)


def verify_fix(source: str, rule_id: str, edits: list[Edit], before=None) -> bool:
    """Apply in memory, re-lint: the rule must be gone from the edited lines and no high/medium finding may be new."""
    new = apply_edits(source, edits)
    try:
        ast.parse(new)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    before = before if before is not None else check_code(source)
    after = check_code(new)
    if not after.parsed:
        return False
    lines = {ln for e in edits for ln in range(e.start_line, e.end_line + 1)}
    if any(f.rule_id == rule_id and f.line in lines for f in after.findings):
        return False
    b, a = _actionable_counts(before.findings), _actionable_counts(after.findings)
    if any(a[k] > b[k] for k in a):
        return False
    return True


class FixBudget:
    """Limits on fix self-checks. Each check re-lints the whole file, so a file with hundreds of fixable lines
    could otherwise take minutes. When a limit is reached, the remaining fixes are skipped and counted."""

    MAX_CHECKS_PER_CALL = 60
    MAX_CHECKS_PER_FILE = 12
    SECONDS_PER_CALL = 10.0

    def __init__(self, max_checks: int | None = None, seconds: float | None = None,
                 per_file: int | None = None):
        self.checks_left = self.MAX_CHECKS_PER_CALL if max_checks is None else max_checks
        self.per_file = self.MAX_CHECKS_PER_FILE if per_file is None else per_file
        self.deadline = time.monotonic() + (self.SECONDS_PER_CALL if seconds is None else seconds)
        self.skipped = 0

    def exhausted(self) -> bool:
        return self.checks_left <= 0 or time.monotonic() >= self.deadline

    def note(self) -> str | None:
        if not self.skipped:
            return None
        return (f"{self.skipped} possible fix(es) were not checked because the fix-checking budget ran out "
                f"(at most {self.per_file} per file, {self.MAX_CHECKS_PER_CALL} per call, "
                f"{self.SECONDS_PER_CALL:g} s per call). Re-run on fewer files to get them.")


def suggest_fixes(source: str, findings=None, file: str | None = None, *, tree: ast.AST | None = None,
                  audit=None, budget: FixBudget | None = None) -> list[SuggestedFix]:
    """Minimal edits for unambiguous findings in `source`, each re-checked by re-linting. May be empty.

    Pass a shared `budget` to bound the work across several files; the default allows one call's worth."""
    source = source.removeprefix("﻿").replace("\r\n", "\n")
    budget = budget if budget is not None else FixBudget()
    if audit is None:
        audit = check_code(source)
    if findings is None:
        findings = audit.findings
    if tree is None:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return []
    src = _Src(source)
    out: list[SuggestedFix] = []
    seen: set[tuple] = set()
    file_checks = attempts = 0
    for f in findings:
        if f.rule_id not in _PROPOSERS or f.severity not in _ACTIONABLE:
            continue
        if budget.exhausted() or file_checks >= budget.per_file or attempts >= 4 * budget.per_file:
            budget.skipped += 1
            continue
        attempts += 1
        st = stmt_for_line(tree, f.line)
        if st is None:
            continue
        try:
            prop = _PROPOSERS[f.rule_id](src, tree, st)
        except (ValueError, SyntaxError, AttributeError, IndexError, RecursionError, TypeError, KeyError):
            prop = None
        if not prop:
            continue
        repls, why = prop
        edits = _to_edits(src, repls)
        if not edits:
            continue
        key = (f.rule_id, tuple((e.start_line, e.replacement) for e in edits))
        if key in seen:
            continue
        seen.add(key)
        if any(e.original == e.replacement for e in edits):
            continue
        budget.checks_left -= 1
        file_checks += 1
        if not verify_fix(source, f.rule_id, edits, before=audit):
            continue
        out.append(SuggestedFix(f.rule_id, f.line, why, edits, True, file))
    _INDEX.clear()  # don't keep large syntax trees alive between calls
    return out
