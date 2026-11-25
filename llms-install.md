# Installing Backtest Auditor (for AI agents)

Backtest Auditor is a local MCP server over stdio. It needs no API key and no environment variables.

1. Check that `uvx` is available (`uvx --version`). If it isn't, ask the user to install uv:
   https://docs.astral.sh/uv/getting-started/installation/ . uv fetches Python 3.12 if needed.
2. Add this server to the client's MCP settings:

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

3. Restart or reload the MCP servers, then confirm it connects: the server is named `nba-auditor` and lists six
   tools (`check_path`, `check_code`, `audit_returns`, `audit_trade_log`, `audit_backtest`, `explain`).
4. Test it: call `explain` with `check_id` set to `LA07`. It returns a description of the "Signal and fill on the
   same bar" check.

Alternative without uv: in a Python 3.12 environment run `pip install backtest-auditor`, then use the absolute path
of the installed `nba-auditor-mcp` executable as `command`, with no `args`.

When you use it: the tools read code without running it. A clean result is weak evidence, not proof that a backtest
is correct: the checks catch some common errors, miss many, and can flag correct files, even plain ones. Show the user each
suggested edit before applying it.
