"""`audit_backtest`: one call that runs every audit that applies and returns one combined verdict.

- `code_path` (a file or folder) -> `check_path`. Any high finding is a major red flag; any medium finding is a red
  flag. A clean lint adds no red flag, but it is weak evidence and the report says so. If any code file was skipped
  (too large, unreadable, undecodable) or could only be text-matched (it didn't parse), the code part is at most
  "Some red flags" and the reason names the files.
- `returns_csv` -> `audit_returns`; `trades_csv` -> `audit_trade_log` (with `prices_csv` if given).
- `benchmark_csv` goes to both return audits.

The combined verdict is the worst verdict of the parts, in the existing language ("No red flags found", "Some red
flags", "Major red flags"), always followed by the caveat line. The existing caps hold: without a benchmark the
best verdict is "Some red flags", and a code-only audit (no returns or trade log) is at most "Some red flags" too.

Data files: local paths only (no UNC or device paths), `.csv`, `.json` or `.txt` only (checked before reading),
20 MB each. Errors never quote a data file's contents (see `safety.py`). Nothing is executed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .catalog import DISCLAIMER, NO_BENCHMARK_REASON
from .parsing import InputError, parse_series, read_table
from .path_audit import check_path, path_markdown
from .returns_audit import (VERDICT_MAJOR_FLAGS, VERDICT_NO_FLAGS, VERDICT_NOTE, VERDICT_SOME_FLAGS, audit_returns)
from .safety import check_data_path, check_resolved_local, file_error, is_safe_message
from .trade_log import audit_trade_log, parse_prices, parse_trades

MAX_CSV_BYTES = 20_000_000
_RANK = {VERDICT_NO_FLAGS: 0, VERDICT_SOME_FLAGS: 1, VERDICT_MAJOR_FLAGS: 2}
NO_DATA_REASON = ("Only the code was checked: no return series or trade log was audited, so the result itself "
                  "was not reviewed.")
CLEAN_CODE_NOTE = ("check_code found none of its listed patterns in the code. Static checks miss many leaks, so this "
                   "is weak evidence.")
# skipped files that don't hide strategy code
_HARMLESS_SKIPS = ("empty file", "virtual environment", "notebook has no code cells")


def _read_csv(path: str, what: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise InputError(f"{what}: expected a file path")
    check_data_path(path, what)
    p = Path(os.path.expanduser(path.strip()))
    try:
        resolved = p.resolve(strict=True)
        check_resolved_local(str(resolved), what)
        size = resolved.stat().st_size
    except (OSError, RuntimeError):
        raise InputError(f"{what}: file not found or not accessible: {path}") from None
    if not resolved.is_file():
        raise InputError(f"{what}: not a file: {path}")
    if size > MAX_CSV_BYTES:
        raise InputError(f"{what}: file is too large ({size:,} bytes; the limit is {MAX_CSV_BYTES:,}). "
                         "Aggregate to daily or send fewer rows.")
    try:
        with open(resolved, "rb") as fh:
            data = fh.read(MAX_CSV_BYTES + 1)
    except OSError as e:
        raise InputError(f"{what}: unreadable ({e.strerror or type(e).__name__})") from None
    if b"\x00" in data[:8192]:
        raise InputError(f"{what}: looks like a binary file, expected CSV or JSON text")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise InputError(f"{what}: not UTF-8 text") from None
    if "\n" not in text.strip():
        raise InputError(f"{what}: the file has fewer than two lines; expected a header and data rows")
    return text


def _load(path: str, what: str, kind: str) -> Any:
    """Read a data file and parse it into a table or series. Parse errors never quote the file."""
    text = _read_csv(path, what)
    try:
        if kind == "series":
            return parse_series(text, what)
        table = read_table(text, what)
        (parse_trades if kind == "trades" else parse_prices)(table)  # validate now, so errors name this file
        return table
    except (InputError, ValueError, KeyError, TypeError, IndexError) as e:
        raise file_error(what, e) from None


def _run(what: str, fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except (InputError, ValueError, KeyError, TypeError, IndexError) as e:
        msg = str(e.args[0]) if e.args else str(e)
        if is_safe_message(msg):
            raise InputError(msg) from None
        raise InputError(f"{what}: the files could not be audited together (for example too few rows, or dates "
                         "that don't line up). Details are withheld because parser messages can quote the files' "
                         "contents.") from None


def _code_verdict(res: dict) -> tuple[str, list[str]]:
    c = res["counts"]
    files_with = [f for f in res["files"] if f["findings"]]

    def where(sev: str) -> str:
        hits = [f"{f['path']}:{x['start_line']} {x['rule_id']}" for f in files_with for x in f["findings"]
                if x["severity"] == sev]
        return ", ".join(hits[:5]) + (f" and {len(hits) - 5} more" if len(hits) > 5 else "")

    def names(items: list[str]) -> str:
        return ", ".join(items[:5]) + (f" and {len(items) - 5} more" if len(items) > 5 else "")

    if res["files_checked"] == 0:
        why = names([f"{s['path']}: {s['reason']}" for s in res["skipped"]])
        return VERDICT_SOME_FLAGS, ["No Python file could be read at code_path, so the code was not checked"
                                    + (f" ({why})." if why else ".")]
    if c["high"]:
        return VERDICT_MAJOR_FLAGS, [f"The code has {c['high']} high-severity lookahead or bias finding(s) "
                                     f"({where('high')}): backtest numbers from it are likely inflated."]
    gaps: list[str] = []
    unparsed = [f["path"] for f in res["files"] if not f.get("parsed_with_ast", True)]
    if unparsed:
        gaps.append(f"{len(unparsed)} code file(s) could not be parsed, so only a few text patterns were checked "
                    f"there ({names(unparsed)}).")
    skipped = [s for s in res["skipped"] if not s["reason"].startswith(_HARMLESS_SKIPS)]
    if skipped:
        gaps.append(f"{len(skipped)} file(s) or folder(s) were not checked ("
                    + names([f"{s['path']}: {s['reason']}" for s in skipped]) + ").")
    if c["medium"]:
        return VERDICT_SOME_FLAGS, [f"The code has {c['medium']} medium-severity finding(s) ({where('medium')})."] + gaps
    if gaps:
        return VERDICT_SOME_FLAGS, gaps + [CLEAN_CODE_NOTE]
    return VERDICT_NO_FLAGS, [CLEAN_CODE_NOTE]


def audit_backtest(code_path: str | None = None, returns_csv: str | None = None, trades_csv: str | None = None,
                   benchmark_csv: str | None = None, *, n_trials: int | None = None, turnover: float | None = None,
                   costs_bps: float | None = None, prices_csv: str | None = None, freq: str = "auto",
                   returns_are_net: bool = False) -> dict[str, Any]:
    """Run every audit that applies to the paths given and return one combined verdict (JSON-ready dict)."""
    if not any(x for x in (code_path, returns_csv, trades_csv)):
        raise InputError("give at least one of code_path, returns_csv or trades_csv")
    if benchmark_csv and not (returns_csv or trades_csv):
        raise InputError("benchmark_csv needs returns_csv or trades_csv")
    parts: list[tuple[str, str, list[str]]] = []
    out: dict[str, Any] = {"tool": "audit_backtest", "content_is_untrusted_data": True}
    md: list[str] = []

    # read and validate every data file before any work, so a bad file fails fast and names itself
    bench = _load(benchmark_csv, "benchmark_csv", "series") if benchmark_csv else None
    rets = _load(returns_csv, "returns_csv", "series") if returns_csv else None
    trades = _load(trades_csv, "trades_csv", "trades") if trades_csv else None
    prices = _load(prices_csv, "prices_csv", "prices") if (prices_csv and trades_csv) else None

    if code_path:
        res = check_path(code_path)
        v, rs = _code_verdict(res)
        parts.append(("code", v, rs))
        out["code"] = res
        md.append(path_markdown(res).rsplit("\n---\n", 1)[0])
    if rets is not None:
        rep = _run("returns_csv", audit_returns, rets, freq=freq, costs_bps=costs_bps, turnover=turnover,
                   benchmark=bench, n_trials=n_trials, returns_are_net=returns_are_net)
        parts.append(("returns", rep.verdict, rep.reasons))
        out["returns"] = rep.to_dict()
        md.append(rep.to_markdown().rsplit("\n---\n", 1)[0])
    if trades is not None:
        rep = _run("trades_csv", audit_trade_log, trades, prices=prices, costs_bps=costs_bps, benchmark=bench,
                   n_trials=n_trials)
        parts.append(("trades", rep.verdict, rep.reasons))
        out["trades"] = rep.to_dict()
        md.append(rep.to_markdown().rsplit("\n---\n", 1)[0])

    verdict = max((v for _, v, _ in parts), key=_RANK.__getitem__)
    reasons: list[str] = []
    for name, v, rs in sorted(parts, key=lambda x: -_RANK[x[1]]):
        reasons += [f"[{name}] {r}" for r in rs[:4]]
    data_audited = bool(returns_csv or trades_csv)
    if not data_audited and _RANK[verdict] < _RANK[VERDICT_SOME_FLAGS]:
        verdict = VERDICT_SOME_FLAGS
    if not data_audited:
        reasons.append(NO_DATA_REASON)
    if bench is None and NO_BENCHMARK_REASON not in " ".join(reasons):
        reasons.append(NO_BENCHMARK_REASON)
        if _RANK[verdict] < _RANK[VERDICT_SOME_FLAGS]:
            verdict = VERDICT_SOME_FLAGS
    if not code_path:
        reasons.append("The strategy code was not checked (pass code_path to lint it for lookahead).")

    out.update({"verdict": verdict, "verdict_note": VERDICT_NOTE, "reasons": reasons,
                "parts": [{"part": n, "verdict": v} for n, v, _ in parts], "disclaimer": DISCLAIMER})
    head = ["# Backtest Auditor: combined audit", "", f"## Verdict: {verdict}", "", VERDICT_NOTE, "",
            "*A statistical and code review of a hypothetical backtest, not a certification of the strategy.*", ""]
    head += [f"- {r}" for r in reasons] + [""]
    head += ["| Part | Verdict |", "|---|---|"] + [f"| {n} | {v} |" for n, v, _ in parts] + [""]
    out["markdown"] = "\n".join(head + ["\n\n".join(md), "", "---", DISCLAIMER])
    return out
