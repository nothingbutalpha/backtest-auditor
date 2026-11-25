# Backtest Auditor quickstart

Connect the `nba-auditor` MCP server to Claude Code, Claude Desktop, Cursor or VS Code, or use it as a Python
library. It runs locally over stdio, calls no data vendor and needs no API key. Code you pass to it is read, never
executed. Anything you paste into an AI assistant is still sent to that assistant's provider as usual.

Every snippet on this page is tested in CI or checked by hand (`tests/test_docs_snippets.py`). Each code block carries a marker
saying how: `run` blocks are executed, `mcp-config` and `claude-mcp-add` blocks are launched as real MCP servers
with a tool call, and `config-only` blocks (installs, client UI steps) are checked by hand.

## 0. One-line setup with uvx

The package is on PyPI as `backtest-auditor`. `uvx backtest-auditor` downloads it and runs the `backtest-auditor`
command, which starts the MCP server when given no arguments. CI tests these configs against that same entry point.

Claude Code:

<!-- test: claude-mcp-add -->
```bash
claude mcp add backtest-auditor -- uvx backtest-auditor
```

Claude Desktop (`claude_desktop_config.json`, see section 3 for where it is) and Cursor (`~/.cursor/mcp.json` or
`.cursor/mcp.json`):

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

VS Code (`.vscode/mcp.json`):

<!-- test: mcp-config -->
```json
{
  "servers": {
    "backtest-auditor": {
      "type": "stdio",
      "command": "uvx",
      "args": ["backtest-auditor"]
    }
  }
}
```

You need [uv](https://docs.astral.sh/uv/); it fetches Python 3.12 if you don't have it.

## 1. Install

You need **Python 3.12** (3.13 isn't supported yet). Install the auditor into its own virtual environment so the
MCP client can launch it by absolute path.

macOS / Linux:

<!-- test: config-only (network install) -->
```bash
python3.12 -m venv ~/.venvs/nba-auditor
~/.venvs/nba-auditor/bin/python -m pip install backtest-auditor
```

Windows (PowerShell):

<!-- test: config-only (network install) -->
```powershell
py -3.12 -m venv $HOME\.venvs\nba-auditor
& $HOME\.venvs\nba-auditor\Scripts\python.exe -m pip install backtest-auditor
```

This installs three commands into the environment:

| Command | What it is |
|---|---|
| `nba-auditor-mcp` | The MCP server (stdio). This is what your MCP client launches. |
| `nba-auditor` | A command-line version of the same tools. `nba-auditor mcp` also starts the server. |
| `backtest-auditor` | With no arguments, the MCP server (what `uvx backtest-auditor` runs); otherwise the same CLI. |

The server's absolute path is what you'll paste into client configs:

- macOS / Linux: `/Users/<you>/.venvs/nba-auditor/bin/nba-auditor-mcp` (or `/home/<you>/...`)
- Windows: `C:/Users/<you>/.venvs/nba-auditor/Scripts/nba-auditor-mcp.exe`

Check the install with the CLI (with the venv activated, or by its full path):

<!-- test: run -->
```bash
nba-auditor explain LA07
```

Running `nba-auditor-mcp` by itself prints nothing: it waits for an MCP client on stdin. Press Ctrl+C to quit.

## 2. Claude Code

Add the server with the CLI. Replace the path with yours, and keep the `--` before the command.

<!-- test: claude-mcp-add -->
```bash
claude mcp add nba-auditor -- /Users/you/.venvs/nba-auditor/bin/nba-auditor-mcp
```

On Windows:

<!-- test: claude-mcp-add -->
```powershell
claude mcp add nba-auditor -- C:/Users/you/.venvs/nba-auditor/Scripts/nba-auditor-mcp.exe
```

Then confirm that it connects:

<!-- test: config-only (needs the Claude Code CLI) -->
```bash
claude mcp list
# nba-auditor: /Users/you/.venvs/nba-auditor/bin/nba-auditor-mcp - ✔ Connected
```

**Share it with a project.** `claude mcp add --scope project ...` writes a `.mcp.json` file at the project root
that you can commit. It looks like this:

<!-- test: mcp-config -->
```json
{
  "mcpServers": {
    "nba-auditor": {
      "type": "stdio",
      "command": "/Users/you/.venvs/nba-auditor/bin/nba-auditor-mcp",
      "args": []
    }
  }
}
```

Claude Code asks each person to approve a project's servers the first time they run `claude` in it. Until then,
`claude mcp list` shows the server as "Pending approval". Because the path is absolute, everyone who uses the
file needs the venv in the same place, so for a team, per-user setup (`claude mcp add` without `--scope`) is often simpler.

**Non-interactive check.** This runs one tool call through the server, headless:

<!-- test: config-only (needs the Claude Code CLI and a Claude login) -->
```bash
claude -p --mcp-config .mcp.json --strict-mcp-config --allowedTools "mcp__nba-auditor__explain" \
  "Call the nba-auditor explain tool with check_id LA07. Reply with only the first line of its markdown output, verbatim."
# # LA07: Signal and fill on the same bar
```

## 3. Claude Desktop

Open **Settings > Developer > Edit Config**. That opens `claude_desktop_config.json`:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Add the server under `mcpServers` (keep any servers already there), save, then quit and restart Claude Desktop.

macOS:

<!-- test: mcp-config -->
```json
{
  "mcpServers": {
    "nba-auditor": {
      "command": "/Users/you/.venvs/nba-auditor/bin/nba-auditor-mcp",
      "args": []
    }
  }
}
```

Windows (use forward slashes, or double every backslash):

<!-- test: mcp-config -->
```json
{
  "mcpServers": {
    "nba-auditor": {
      "command": "C:/Users/you/.venvs/nba-auditor/Scripts/nba-auditor-mcp.exe",
      "args": []
    }
  }
}
```

The tools then appear under the tools (plug) menu in a new chat.

## 4. Cursor

Put the same block in `~/.cursor/mcp.json` (all projects) or `.cursor/mcp.json` (one project), then check that
the server shows as enabled on Cursor's MCP settings page.

<!-- test: mcp-config -->
```json
{
  "mcpServers": {
    "nba-auditor": {
      "command": "/Users/you/.venvs/nba-auditor/bin/nba-auditor-mcp",
      "args": []
    }
  }
}
```

## 5. VS Code and other clients

VS Code (GitHub Copilot agent mode) reads `.vscode/mcp.json`, which uses a `servers` key:

<!-- test: mcp-config -->
```json
{
  "servers": {
    "nba-auditor": {
      "type": "stdio",
      "command": "/Users/you/.venvs/nba-auditor/bin/nba-auditor-mcp",
      "args": []
    }
  }
}
```

Any other client that supports local stdio servers works the same way: command = the absolute path of
`nba-auditor-mcp`, no arguments, no environment variables. If a client can only launch Python, use the venv's
Python with the arguments `["-m", "nba_auditor.mcp"]`:

<!-- test: mcp-config -->
```json
{
  "mcpServers": {
    "nba-auditor": {
      "command": "/Users/you/.venvs/nba-auditor/bin/python",
      "args": ["-m", "nba_auditor.mcp"]
    }
  }
}
```

## 6. Try it

The server offers two prompts, which most clients show as one-click commands (in Claude Code, type `/` and look
for `nba-auditor`):

- **Find look-ahead bias in this repo** (`find_lookahead_bias`): runs `check_path` on a folder, summarizes the
  findings by file and line, and applies only the suggested fixes (it asks first unless you pass
  `apply_fixes=yes`).
- **Audit this backtest before I rely on it** (`audit_backtest_before_trusting`): finds or asks for the strategy
  code, the returns file, a benchmark and your number of trials, then calls `audit_backtest` for one verdict.

Or ask your assistant things like:

- *"Run check_path on this project and explain each finding."*
- *"Audit this backtest with audit_backtest: strategy.py, backtest.csv, and spy.csv as the benchmark. I tried
  about 40 parameter combinations and turnover is about 0.3 a day."*
- *"What does check LA07 look for? Use explain."*

To start from an idea instead of existing code, see [From idea to audited backtest](walkthrough-draft-to-audit.md).

Tell it how many variants you tried (`n_trials`). The verdict depends on that number being honest.

**Use absolute paths** with `check_path` and `audit_backtest`: the server resolves relative paths against its own
working directory, which depends on the client. Both tools only read files: they never write, import or run your
code.

**Suggested fixes.** For a few unambiguous patterns (a negative shift in a feature, `rolling(center=True)`,
backward fill, a scaler fitted before a visible slice split, a shuffled time-series split), `check_path` and
`check_code` return a minimal edit: the line range, the current text and the replacement. The server applies
each edit in memory and lints again before returning it, and drops it if the finding remains or a new high or
medium finding appears. Everything else comes with an explanation, not an edit. Review edits like any other
change: a fix removes a pattern, it doesn't make a backtest trustworthy.

## 7. Python library

The same tools are plain functions. This example is self-contained (synthetic data):

<!-- test: run -->
```python
import numpy as np
import pandas as pd
from nba_auditor import audit_returns, check_code

code = '''
import pandas as pd
df = pd.read_csv("prices.csv")
df["signal"] = (df["close"] > df["close"].rolling(50).mean()).astype(int)
df["strategy"] = df["signal"] * df["close"].pct_change()
'''
rep = check_code(code)
for f in rep.findings:
    print(f.rule_id, f.severity, f"line {f.line}", f.title)

rng = np.random.default_rng(0)
dates = pd.bdate_range("2020-01-01", periods=1000)
returns = pd.Series(rng.normal(0.0002, 0.01, 1000), index=dates)    # a strategy's daily returns
benchmark = pd.Series(rng.normal(0.0003, 0.011, 1000), index=dates)  # e.g. an index
rep = audit_returns(returns, turnover=0.2, n_trials=20, benchmark=benchmark)
print(rep.verdict)
for reason in rep.reasons:
    print("-", reason)

markdown = rep.to_markdown()  # the full report, ending with the disclaimer
data = rep.to_dict()          # the same report as JSON-ready data
```

Files on disk, with suggested fixes (self-contained; it writes a small file into the current folder):

<!-- test: run -->
```python
from pathlib import Path
from nba_auditor import apply_edits, check_path
from nba_auditor.fixes import Edit

Path("strategy.py").write_text(
    "import pandas as pd\n"
    "df = pd.read_csv('prices.csv')\n"
    "df['trend'] = df['close'].rolling(20, center=True).mean()\n"
    "df['signal'] = (df['close'] > df['trend']).astype(int)\n",
    encoding="utf-8",
)
res = check_path("strategy.py")
for f in res["files"][0]["findings"]:
    print(f["rule_id"], f["severity"], f"lines {f['start_line']}-{f['end_line']}", f["why_it_matters"])

fix = res["files"][0]["suggested_fixes"][0]
print(fix["rationale"])
source = Path("strategy.py").read_text(encoding="utf-8")
fixed = apply_edits(source, [Edit(**e) for e in fix["edits"]])
assert "center=False" in fixed
```

`audit_returns` also takes a CSV path or CSV/JSON text, and `audit_trade_log` takes a trade log.
`audit_backtest(code_path, returns_csv, trades_csv, benchmark_csv, n_trials=...)` runs all of them on files and
returns one combined verdict. See the
[reference](RULES.md) for every parameter and the input conventions, and
[examples/leaky_to_clean](../examples/leaky_to_clean/README.md) for a full walkthrough.

## 8. Troubleshooting

- **Server shows as failed or "spawn ENOENT".** The command must be an absolute path; clients don't expand `~`
  or activate virtual environments. Check the path by running it in a terminal (it should wait silently).
- **Windows JSON errors.** Backslashes must be doubled in JSON. Forward slashes work everywhere.
- **`requires a different Python`.** The package needs Python 3.12.
- **Tools don't appear in Claude Desktop.** Restart it fully (quit from the tray or menu bar, not just the window).

---
*The auditor's reports are automated statistical and code reviews of hypothetical backtests. They are general
information, not investment advice, not tailored to anyone's financial situation or objectives, and do not predict future results. The checks can miss
problems and can flag correct code. "No red flags found" doesn't mean a strategy will make money. Nothing But Alpha
is not a registered investment adviser.*
