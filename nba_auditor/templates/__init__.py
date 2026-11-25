"""Backtest templates, one per format, served as MCP resources and by `backtest-auditor template FORMAT`.

The template files are read as text and never imported or run by the auditor. Each one keeps the same timing
conventions: signals use data up to the bar's close, trades fill on the next bar's open, costs are a parameter,
and the train/test split is by date. tests/test_draft_backtest.py checks that each lints clean and that a broken
variant of each is flagged.
"""

from __future__ import annotations

from pathlib import Path

FORMATS = ("pandas", "vectorbt", "backtrader")
_DIR = Path(__file__).resolve().parent


def resource_uri(fmt: str) -> str:
    return f"nba-auditor://templates/{fmt}"


def normalize_format(fmt: str | None) -> str:
    """'' or None means pandas; anything else must be one of FORMATS (case and surrounding spaces ignored)."""
    f = (fmt or "pandas").strip().lower()
    if f not in FORMATS:
        raise ValueError(f"Unknown format {fmt!r}. Choose one of: {', '.join(FORMATS)}.")
    return f


def template_source(fmt: str) -> str:
    """The template's Python source for `fmt` (pandas, vectorbt or backtrader)."""
    return (_DIR / f"{normalize_format(fmt)}_template.py").read_text(encoding="utf-8")
