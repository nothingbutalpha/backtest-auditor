"""Parse user-supplied series and tables (CSV text, JSON text, file paths, or Python objects)."""

from __future__ import annotations

import io
import json
import os
from typing import Any

import numpy as np
import pandas as pd

DATE_COLS = ("date", "datetime", "time", "timestamp", "dt", "day", "period")
VALUE_COLS = ("return", "returns", "ret", "rets", "strategy", "strategy_return", "strategy_returns", "r",
              "pnl_pct", "daily_return", "value", "values", "equity", "nav", "close", "benchmark", "turnover")
FREQ_PERIODS = {"D": 252, "B": 252, "DAILY": 252, "W": 52, "WEEKLY": 52, "M": 12, "ME": 12, "MS": 12,
                "MONTHLY": 12, "Q": 4, "QE": 4, "QUARTERLY": 4, "A": 1, "Y": 1, "YE": 1, "ANNUAL": 1,
                "YEARLY": 1, "H": 252 * 6.5, "HOURLY": 252 * 6.5}


class InputError(ValueError):
    """Raised when user input can't be parsed; the message says what was expected."""


def _looks_like_path(text: str) -> bool:
    t = text.strip()
    return "\n" not in t and len(t) <= 1000 and t.lower().endswith((".csv", ".json", ".txt", ".tsv"))


def _maybe_path(text: str) -> str | None:
    t = text.strip()
    if not _looks_like_path(t):
        return None
    # never touch a network location: even a stat on a UNC path can send the user's NTLM credentials
    if t.startswith(("\\\\", "//")) or ":" in (t[2:] if len(t) > 1 and t[1] == ":" else t):
        raise InputError("network (UNC), device and URL-like paths are not accepted; use a local path")
    if os.path.isfile(t):
        return t
    return None


def is_file_path(data: Any) -> bool:
    """True if `data` is a string that the parsers will read as a file from disk."""
    if not isinstance(data, str) or not _looks_like_path(data):
        return False
    try:
        return _maybe_path(data) is not None
    except InputError:
        return False


def load_text(data: str) -> str:
    p = _maybe_path(data)
    if p:
        with open(p, encoding="utf-8-sig") as fh:
            return fh.read()
    return data


def read_table(data: Any, what: str = "table") -> pd.DataFrame:
    """CSV/TSV text, JSON (list of records or dict of columns), a path, a DataFrame, or a list of dicts."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame(data)
    if not isinstance(data, str) or not data.strip():
        raise InputError(f"{what}: expected CSV or JSON text, a file path, or a table; got {type(data).__name__}")
    text = load_text(data).strip()
    if text[0] in "[{":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise InputError(f"{what}: invalid JSON ({e.msg} at position {e.pos})") from None
        if isinstance(obj, dict) and all(not isinstance(v, (list, dict)) for v in obj.values()):
            return pd.DataFrame({"date": list(obj.keys()), "value": list(obj.values())})
        return pd.DataFrame(obj)
    sep = "\t" if text.split("\n", 1)[0].count("\t") > text.split("\n", 1)[0].count(",") else ","
    try:
        return pd.read_csv(io.StringIO(text), sep=sep)
    except Exception as e:  # pandas raises many types
        raise InputError(f"{what}: could not parse CSV ({type(e).__name__})") from None


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _to_dates(col: pd.Series) -> pd.DatetimeIndex | None:
    try:
        idx = pd.to_datetime(col, errors="raise", format="mixed")
    except (ValueError, TypeError):
        return None
    if getattr(idx.dt, "tz", None) is not None:
        idx = idx.dt.tz_localize(None)
    return pd.DatetimeIndex(idx)


def parse_series(data: Any, what: str = "returns", value_col: str | None = None) -> pd.Series:
    """Parse a numeric series. Accepts:

    * a pandas Series, a list of numbers, or a dict {date: value};
    * JSON text of the same, or a list of records like [{"date": ..., "return": ...}];
    * CSV text (or a .csv path) with an optional date column and one value column.
    """
    if isinstance(data, pd.Series):
        s = data.copy()
    elif isinstance(data, (list, tuple, np.ndarray)) and (len(data) == 0 or not isinstance(data[0], dict)):
        s = pd.Series(np.asarray(data, dtype=float))
    elif isinstance(data, dict) and all(not isinstance(v, (list, dict)) for v in data.values()):
        s = pd.Series(data)
        d = _to_dates(pd.Series(list(s.index)))
        if d is not None:
            s.index = d
    else:
        if isinstance(data, str) and data.strip() and data.strip()[0] == "[":
            obj = json.loads(load_text(data))
            if obj and not isinstance(obj[0], dict):
                return parse_series(obj, what)
        df = read_table(data, what)
        if df.empty:
            raise InputError(f"{what}: no rows")
        dcol = _find_col(df, DATE_COLS)
        if dcol is None and df.shape[1] >= 2 and _to_dates(df.iloc[:, 0].astype(str)) is not None \
                and not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
            dcol = df.columns[0]
        vcol = value_col if value_col in df.columns else None
        if vcol is None:
            others = [c for c in df.columns if c != dcol]
            vcol = _find_col(df[others], VALUE_COLS) if others else None
            if vcol is None:
                numeric = [c for c in others if pd.api.types.is_numeric_dtype(df[c])]
                if len(numeric) == 1:
                    vcol = numeric[0]
                else:
                    raise InputError(f"{what}: can't tell which column holds the values; columns are "
                                     f"{[str(c) for c in df.columns]}. Name it 'return' (or pass one value column).")
        s = pd.to_numeric(df[vcol], errors="coerce")
        if dcol is not None:
            idx = _to_dates(df[dcol].astype(str))
            if idx is None:
                raise InputError(f"{what}: could not parse dates in column {dcol!r}")
            s.index = idx
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if np.isinf(s.to_numpy()).any():
        raise InputError(f"{what}: values must be finite (found inf); check for division by zero or bad prints")
    if isinstance(s.index, pd.DatetimeIndex):
        if s.index.has_duplicates:
            raise InputError(f"{what}: duplicate dates (e.g. {s.index[s.index.duplicated()][0].date()}); "
                             "aggregate to one value per period first")
        s = s.sort_index()
    s.name = what
    return s


def _infer_from_dates(index: pd.Index | None) -> tuple[float, str] | None:
    if index is None or not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None
    days = float(np.median(np.diff(index.values).astype("timedelta64[s]").astype(float)) / 86400)
    if days < 0.5:
        return None  # intraday: spacing alone can't tell trading hours per year
    for lim, label in ((1.6, "D"), (8, "W"), (35, "M"), (100, "Q")):
        if days <= lim:
            return float(FREQ_PERIODS[label]), label
    return 1.0, "Y"


def periods_per_year(freq: str | float | int, index: pd.Index | None = None) -> tuple[float, str]:
    """Map a frequency to periods per year.

    'auto' infers it from the median spacing of the dates (daily or slower). A stated frequency that
    contradicts the dates by more than 1.6x (e.g. freq='D' on month-end dates) raises an error, because a
    wrong frequency silently mis-scales every annualized number.
    """
    inferred = _infer_from_dates(index)
    if isinstance(freq, (int, float)) and not isinstance(freq, bool):
        if freq <= 0:
            raise InputError("freq: periods per year must be positive")
        ppy, label = float(freq), f"{freq:g} periods/year"
    else:
        f = str(freq).strip().upper()
        if f == "AUTO":
            if inferred is not None:
                return inferred[0], f"{inferred[1]} (inferred from dates)"
            if isinstance(index, pd.DatetimeIndex) and len(index) >= 3:
                raise InputError("freq: the dates are intraday; pass freq as periods per year (e.g. 252*6.5 for hourly bars)")
            return 252.0, "D (assumed: no dates to infer from)"
        if f in FREQ_PERIODS:
            ppy, label = float(FREQ_PERIODS[f]), f
        else:
            try:
                ppy, label = float(f), f"{float(f):g} periods/year"
            except ValueError:
                raise InputError(f"freq: unknown frequency {freq!r}; use auto, D, W, M, Q, Y, or periods per year") from None
            if ppy <= 0:
                raise InputError("freq: periods per year must be positive")
    if inferred is not None and not (1 / 1.6 <= ppy / inferred[0] <= 1.6):
        raise InputError(f"freq: you passed {label} ({ppy:g} periods/year) but the dates are spaced like {inferred[1]} "
                         f"({inferred[0]:g}/year). Pass freq='auto' or the matching frequency.")
    return ppy, label
