"""Input-safety helpers shared by every tool that reads from disk.

- Local paths only: UNC (`\\\\server\\share`, `//server/share`) and Windows device paths (`\\\\.\\`, `\\\\?\\`) are
  refused before anything touches the file system, because opening a UNC path can send the user's NTLM credentials
  to another machine. Alternate data streams (`file.py:stream`) are refused too.
- Data files: only `.csv`, `.json` and `.txt` are read as data, checked before reading.
- Error messages about data read from disk never quote the file: parser messages can contain column names, cell
  values or whole lines (a `.env` file's first line, for instance). Only messages from a fixed list of templates
  that carry counts, not content, are passed through; everything else becomes a generic message.
"""

from __future__ import annotations

import re

from .parsing import InputError

DATA_SUFFIXES = (".csv", ".json", ".txt")

_REMOTE = re.compile(r"^\s*([\\/]{2})")


def reject_remote_path(path: str, what: str = "path") -> None:
    """Raise InputError for UNC, device, URL-like and alternate-data-stream paths."""
    if not isinstance(path, str):
        return
    p = path.strip()
    if _REMOTE.match(p):
        raise InputError(f"{what}: network (UNC) and device paths are not accepted; use a local path")
    if ":" in re.sub(r"^[A-Za-z]:", "", p):
        raise InputError(f"{what}: paths with ':' after the drive letter (URLs, alternate data streams) are not "
                         "accepted; use a local path")


def check_resolved_local(resolved: str, what: str = "path") -> None:
    """After resolving (mapped drives, links), the path must still be local."""
    if resolved.startswith(("\\\\", "//")) and not resolved.startswith("\\\\?\\") or resolved.startswith("\\\\?\\UNC"):
        raise InputError(f"{what}: resolves to a network or device location; use a local path")


# Messages from parsing / returns_audit / trade_log that contain counts only, never file content.
_SAFE = [re.compile(p) for p in (
    r"[\w ]+: need at least \d+ periods to audit, got \d+",
    r"[\w ]+: no rows",
    r"[\w ]+: values must be finite \(found inf\); check for division by zero or bad prints",
    r"benchmark: needs dates that overlap the returns, or the same length",
    r"benchmark: only \d+ overlapping periods",
    r"turnover must be >= 0.*",
    r"turnover has \d+ values but returns have \d+; give dates or equal lengths",
    r"n_trials must be >= 1",
    r"equity values must be positive",
    r"freq: the dates are intraday; pass freq as periods per year \(e\.g\. 252\*6\.5 for hourly bars\)",
    r"trade log: no rows",
    r"trade log: \d+ row\(s\) have a non-numeric quantity or price",
    r"trade log: prices must be positive",
    r"trade log: could not infer capital \(no exposure\); pass initial_capital",
    r"trade log: equity hits zero or below with this capital; pass a larger initial_capital",
)]

EXPECTED = {
    "returns_csv": "a date column and one numeric column of per-period returns (CSV or JSON)",
    "benchmark_csv": "a date column and one numeric column of benchmark returns (CSV or JSON)",
    "trades_csv": "columns date, ticker, side, qty and price (optionally fee)",
    "prices_csv": "daily closes: a date column plus one column per ticker, or date,ticker,close",
    "returns": "a date column and one numeric column of per-period returns (CSV or JSON)",
    "benchmark": "a date column and one numeric column of benchmark returns (CSV or JSON)",
    "turnover": "a number, or a date column and one numeric column",
    "positions": "a date column plus one weight column per asset, or date,ticker,weight",
    "trades": "columns date, ticker, side, qty and price (optionally fee)",
    "prices": "daily closes: a date column plus one column per ticker, or date,ticker,close",
}


def is_safe_message(msg: str) -> bool:
    return any(rx.fullmatch(msg.strip()) for rx in _SAFE)


def file_error(what: str, err: Exception) -> InputError:
    """An error about a file read from disk, without quoting its contents."""
    msg = str(err.args[0]) if err.args else str(err)
    if is_safe_message(msg):
        return InputError(msg)
    exp = EXPECTED.get(what, "CSV or JSON data")
    return InputError(f"{what}: the file could not be read as expected. Expected {exp}. (Details are withheld "
                      "because parser messages can quote the file's contents.)")


def check_data_path(path: str, what: str) -> None:
    reject_remote_path(path, what)
    if not path.strip().lower().endswith(DATA_SUFFIXES):
        raise InputError(f"{what}: only .csv, .json or .txt files are read")
