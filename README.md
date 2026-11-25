# Backtest Auditor

Static checks for common lookahead and bias errors in Python backtest code, plus a return and trade-log audit. A
Python package and a local MCP server.

> **Limits. Read these first.**
>
> **In short.** The Auditor's static checks catch some common look-ahead leaks and miss many. In our latest
> full-size sealed test (2026-10-18), they flagged 34 of 60 planted leaks. They can also flag correct code: in that
> test, 1 of 105 plain correct files and 4 of 55 correct files written to look like leaks (5 of 160 overall). On
> leaks that open-source developers had found and fixed in their own code, they flagged 2 of 28 in a first trial,
> then 5 of 22 and 0 of 10 in two small pilots, which may repeat examples from the first trial (in the first pilot,
> possibly most of them); with so few examples these counts are very uncertain. The full-size test and both pilots
> did not meet our pass rule. Treat a clean result as weak evidence, never as proof that a backtest is free of
> leaks.
>
> - These results and our earlier sealed tests from test set v6 on, including the ones it failed, are on
>   https://nothingbutalpha.app/backtest-audit
> - The return-series verdict depends on you stating honestly how many variants you tried. In our synthetic
>   calibration (`scripts/calibration.py`), the best of 100 pure-noise strategies, reported with that number
>   understated as 1, got "No red flags found" in 67% of 200 runs.
> - It reads your code with Python's `ast` module. It never imports or runs it.

## Install

With [uv](https://docs.astral.sh/uv/) there is no separate install step. `uvx backtest-auditor` starts the MCP
server over stdio.

Claude Code:

```bash
claude mcp add backtest-auditor -- uvx backtest-auditor
```

Claude Desktop (`claude_desktop_config.json`) and Cursor (`~/.cursor/mcp.json` or `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "backtest-auditor": {
      "command": "uvx",
      "args": ["backtest-auditor"]
    }
  }
}
```

Or install it into a Python 3.12 environment:

```bash
pip install backtest-auditor
nba-auditor check strategy.py
```

Setup for VS Code and other clients, and a venv-based setup: [docs/quickstart.md](docs/quickstart.md). Every
snippet there is tested in CI or marked as checked by hand. No API key and no environment variables are needed.

## What it does

| Tool | What you get |
|---|---|
| `check_path` | Lints a `.py` file, notebook or folder on disk. Each finding has a rule id, severity, line range and a one-line reason. For a few unambiguous patterns it also returns a minimal edit, which it re-lints before returning |
| `check_code` | The same lint on code pasted into the chat |
| `audit_returns` | A return-series review: cost grid and break-even cost, autocorrelation-aware Sharpe t-statistics, the Deflated Sharpe Ratio for the number of variants you tried, stability by sub-period, concentration, drawdown, optional benchmark |
| `audit_trade_log` | Builds daily PnL and turnover from a trade log, then runs the same review, plus per-ticker concentration and survivorship hints |
| `audit_backtest` | Runs every check that applies to your code, returns or trades, and gives one verdict: No red flags found, Some red flags or Major red flags |
| `explain` | What a check looks for, why it matters, how to fix it |

All six tools are annotated read-only. The server also has three prompts: **Find look-ahead bias in this repo**,
**Audit this backtest before I rely on it** and **Draft a backtest from my idea** (your assistant writes the code
from a template and runs `check_code` on it; the server writes and runs nothing). The templates are resources:
`nba-auditor://templates/pandas`, `nba-auditor://templates/vectorbt` and `nba-auditor://templates/backtrader`.

Every verdict is followed by this line: *No red flags doesn't mean a strategy will make money; it means these checks
didn't find the common errors.*

An edit from `check_path` changes what the code does. It removes a pattern the lint sees; it doesn't make a backtest
correct. Review it like any other change.

## Rules

Patterns it looks for include negative shifts in features or signals, centered rolling windows, backward fills, a
scaler or model fitted before the train/test split, shuffled cross-validation on time series, a signal and its fill
on the same bar, labels on the wrong bar, today's index members used for past dates, full-sample normalization or
parameter sweeps, fundamentals joined by period end, and missing costs.

| Group | Rule ids |
|---|---|
| Lookahead and leakage | LA01 to LA15 |
| Survivorship | SB01 |
| Data quality | DQ01, DQ02 |
| Costs | CO01 |
| Return series | RA01 to RA09 |
| Trade logs | TL01, TL02 |

The full rule list, the verdict rules and the command line: [docs/RULES.md](docs/RULES.md). What each rule misses:
[docs/LIMITATIONS.md](docs/LIMITATIONS.md). A worked example, from a leaky strategy to its fix:
[examples/leaky_to_clean](examples/leaky_to_clean/README.md).

## Privacy

The server runs on your machine over stdio. It makes no network calls, calls no data vendor and needs no API key,
so we never receive your code. Your AI assistant is a different matter: anything the tools return to it (findings
quote file names and code lines, and suggested edits include the original text) is sent to that assistant's
provider like the rest of the chat.

## How we test it

We score the lint on sealed test sets: the test files and the answer key are fixed and hashed before the run, the
pass rule is written down before the test, the code is scored once, and a separate AI reviewer re-runs the result.
The sets and the results from test set v6 on, including the ones it failed, are described at
https://nothingbutalpha.app/backtest-audit. The tests in this repository (`tests/`) are development and regression
tests; their results are in-sample and are not an accuracy measure.

## Built with AI

Most of this code, its tests and its docs were written by Claude, an AI model, working as our engineering team,
and reviewed by separate AI reviewer passes. Issues and pull requests are triaged by AI agents working for the
company.

## Feedback

The most useful reports are leaks it misses and false alarms on correct code, each with a short code snippet. Use
the issue templates. See [CONTRIBUTING.md](CONTRIBUTING.md) and, for security issues, [SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE).

## Disclaimer

Reports are automated statistical and code reviews of hypothetical backtests. They are general information, not
investment advice, not tailored to anyone's financial situation or objectives, and do not predict future results.
The checks can miss problems and can flag correct code. Nothing But Alpha is not a registered investment adviser.

<!-- mcp-name: app.nothingbutalpha/backtest-auditor -->
