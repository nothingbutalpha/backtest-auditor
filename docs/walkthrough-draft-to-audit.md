# From idea to audited backtest

Describe a strategy idea in plain English, have your own AI assistant draft the backtest from our template, run it
yourself on your own data, then audit it and fix what the audit flags. The auditor never runs your code and never
supplies data. It provides the prompt, the templates and the checks. Your assistant writes the code, and you run
it. Anything you paste into the assistant, and anything the tools return to it, goes to that assistant's provider as
usual.

The example below is a moving-average rule on a public ETF. It is an illustration of the workflow, not a
suggestion to trade it, and it shows no results on purpose.

## Setup with uvx

Install [uv](https://docs.astral.sh/uv/), then connect the server (the [quickstart](quickstart.md) has absolute-path
setups and VS Code).

Claude Code:

<!-- test: claude-mcp-add -->
```bash
claude mcp add backtest-auditor -- uvx backtest-auditor
```

Claude Desktop (Settings > Developer > Edit Config, then restart) and Cursor (`~/.cursor/mcp.json`):

<!-- test: mcp-config -->
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

## Step 1: describe the idea

The server offers a prompt, **Draft a backtest from my idea** (`draft_backtest`, arguments `idea` and `format`:
`pandas`, `vectorbt` or `backtrader`). Most clients list it as a command; in Claude Code, type `/` and look for
`nba-auditor`. Or just ask for it by name. For example:

> Use the draft_backtest prompt. Idea: hold SPY when its 50-day simple moving average is above its 200-day,
> otherwise hold cash. Daily bars, 2010 to 2023, format pandas.

## Step 2: read what comes back

Your assistant should restate the rules and ask about anything unclear (dates, costs, sizing), then write the code
from the template, then run `check_code` on it and fix high and medium findings before it shows you anything.
Abbreviated, the code looks like this:

<!-- test: config-only (excerpt of the pandas template, not runnable alone) -->
```python
FAST, SLOW = 50, 200
COST_BPS = 10.0              # all-in cost per unit of one-way turnover; never zero by default
TRAIN_END = "2018-12-31"     # train and test are split by date

fast = bt["close"].rolling(FAST).mean()
slow = bt["close"].rolling(SLOW).mean()
bt["signal"] = (fast > slow).astype(float)          # known at bar t's close
bt["position"] = bt["signal"].shift(1).fillna(0.0)  # filled at bar t+1's open
```

The timing is the point: signals use data up to the close, and the trade fills at the next open. Read it yourself
too. The check misses many leaks, so a clean result is weak evidence. You can also print any template without an
assistant (`nba-auditor template pandas`, `vectorbt` or `backtrader`):

<!-- test: run -->
```bash
nba-auditor template pandas
```

## Step 3: run it on your own data

Save the code as `backtest.py`. Next to it, put a `prices.csv` you got yourself, with columns `date, open, high,
low, close`, adjusted for splits and dividends the same way. Then run it:

<!-- test: config-only (needs your own prices.csv) -->
```bash
python backtest.py
```

It prints a train and a test summary and writes net returns to `returns.csv`. Look at the test period once, after
the rule is final. Count every variant you try (windows, assets, rules) as a trial. Say you tried three window
pairs: that is `--n-trials 3`, even if only one remains in the file.

## Step 4: audit and fix

Check the code first, then the code and returns together. Use your own average turnover per bar; 0.02 is a
placeholder.

<!-- test: config-only (needs your own backtest.py and returns.csv) -->
```bash
nba-auditor path backtest.py
nba-auditor backtest --code backtest.py --returns returns.csv --n-trials 3 --turnover 0.02 --costs-bps 10
```

Suppose that, while tidying the file, you asked the assistant to simplify the position line and it dropped the
shift, so `position = signal`. The check reports:

> **[HIGH] LA07 Signal and fill on the same bar**: The position is credited with the very price move that
> created the signal.

`nba-auditor explain LA07` says what the rule looks for and how to fix it:

<!-- test: run -->
```bash
nba-auditor explain LA07
```

The fix is to restore the shift, `bt["signal"].shift(1).fillna(0.0)`, so the position is decided at one close and
earned from the next open. Run `path` again. With the shift back, it reports that none of the listed patterns
matched. That does not prove the code is free of leaks.

The combined audit then gives one verdict, the worst of its parts. With no benchmark file it cannot say
"No red flags found". It lists the reason: *No benchmark supplied: market exposure can't be separated from skill.*
Add `--benchmark` with a returns file for the same period to separate market exposure from the rule. If the
verdict lists more reasons, fix them or accept them knowingly, then re-run.

---
*This is an automated statistical and code review of a hypothetical backtest. It is general information, not
investment advice, not tailored to anyone, and does not predict future results. The checks can miss problems and
can flag correct code. Nothing But Alpha is not a registered investment adviser.*
