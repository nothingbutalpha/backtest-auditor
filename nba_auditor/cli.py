"""Command line: `python -m nba_auditor {check,returns,trades,explain,template} ...` (Markdown by default, --json for JSON).

`nba-auditor mcp` starts the MCP server over stdio (same as `nba-auditor-mcp`). The `backtest-auditor` command
(the PyPI package's own name, so `uvx backtest-auditor` works) starts the server when given no arguments and
behaves like `nba-auditor` otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import audit_backtest, audit_returns, audit_trade_log, check_code, check_path, explain, explain_markdown
from .parsing import InputError
from .templates import FORMATS, template_source


def main(argv: list[str] | None = None, prog: str = "nba-auditor") -> int:
    ap = argparse.ArgumentParser(prog=prog, description="Backtest Auditor: red-team lint for trading strategies.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="lint a Python strategy file")
    c.add_argument("path")

    pth = sub.add_parser("path", help="lint a file or folder of Python files and notebooks (read-only)")
    pth.add_argument("path")
    pth.add_argument("--include-glob")
    pth.add_argument("--max-files", type=int)

    b = sub.add_parser("backtest", help="combined audit: code + returns or trades (+ benchmark)")
    b.add_argument("--code")
    b.add_argument("--returns")
    b.add_argument("--trades")
    b.add_argument("--benchmark")
    b.add_argument("--prices")
    b.add_argument("--n-trials", type=int)
    b.add_argument("--turnover", type=float)
    b.add_argument("--costs-bps", type=float)

    r = sub.add_parser("returns", help="audit a return series (CSV/JSON)")
    r.add_argument("path")
    r.add_argument("--freq", default="auto")
    r.add_argument("--costs-bps", type=float)
    r.add_argument("--turnover", help="a number or a CSV/JSON path")
    r.add_argument("--positions")
    r.add_argument("--benchmark")
    r.add_argument("--n-trials", type=int)
    r.add_argument("--input-kind", default="returns", choices=["returns", "equity"])
    r.add_argument("--returns-are-net", action="store_true")
    r.add_argument("--units", default="decimal", choices=["decimal", "percent"])

    t = sub.add_parser("trades", help="audit a trade log (CSV/JSON)")
    t.add_argument("path")
    t.add_argument("--prices")
    t.add_argument("--initial-capital", type=float)
    t.add_argument("--costs-bps", type=float)
    t.add_argument("--benchmark")
    t.add_argument("--n-trials", type=int)

    e = sub.add_parser("explain", help="explain a check (no id: list all)")
    e.add_argument("check_id", nargs="?")

    tpl = sub.add_parser("template", help="print a backtest template (pandas, vectorbt or backtrader)")
    tpl.add_argument("format", nargs="?", default="pandas", choices=FORMATS)

    for p in (c, r, t, e, pth, b):
        p.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    sub.add_parser("mcp", help="run the MCP server over stdio (same as nba-auditor-mcp)")
    a = ap.parse_args(argv)

    if a.cmd == "mcp":
        from .mcp import main as mcp_main  # imported lazily: nothing may be printed to stdout before the server starts

        mcp_main()
        return 0

    if a.cmd == "template":
        sys.stdout.write(template_source(a.format))
        return 0

    try:
        if a.cmd == "explain":
            print(json.dumps(explain(a.check_id), indent=2) if a.json else explain_markdown(a.check_id))
            return 0
        if a.cmd == "path":
            from .path_audit import path_markdown

            res = check_path(a.path, include_glob=a.include_glob, max_files=a.max_files)
            print(json.dumps(res, indent=2) if a.json else path_markdown(res))
            return 0
        if a.cmd == "backtest":
            res = audit_backtest(a.code, a.returns, a.trades, a.benchmark, n_trials=a.n_trials, turnover=a.turnover,
                                 costs_bps=a.costs_bps, prices_csv=a.prices)
            md = res.pop("markdown")
            print(json.dumps(res, indent=2) if a.json else md)
            return 0
        if a.cmd == "check":
            with open(a.path, encoding="utf-8-sig") as fh:
                rep = check_code(fh.read())
        elif a.cmd == "returns":
            to = a.turnover
            try:
                to = float(to) if to is not None else None
            except ValueError:
                pass
            rep = audit_returns(a.path, freq=a.freq, costs_bps=a.costs_bps, turnover=to, positions=a.positions,
                                benchmark=a.benchmark, n_trials=a.n_trials, input_kind=a.input_kind,
                                returns_are_net=a.returns_are_net, units=a.units)
        else:
            rep = audit_trade_log(a.path, prices=a.prices, initial_capital=a.initial_capital, costs_bps=a.costs_bps,
                                  benchmark=a.benchmark, n_trials=a.n_trials)
    except (InputError, ValueError, KeyError, OSError) as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    print(json.dumps(rep.to_dict(), indent=2) if a.json else rep.to_markdown())
    return 0


def server_or_cli() -> int:
    """`backtest-auditor`: no arguments starts the MCP server (what `uvx backtest-auditor` runs); otherwise the CLI."""
    argv = sys.argv[1:]
    if not argv:
        from .mcp import main as mcp_main

        mcp_main()
        return 0
    return main(argv, prog="backtest-auditor")


if __name__ == "__main__":
    raise SystemExit(main())
