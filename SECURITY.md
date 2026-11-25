# Security

## Reporting a problem

Please report security problems privately through GitHub's private vulnerability reporting: the "Report a
vulnerability" button on this repository's Security tab
(https://github.com/nothingbutalpha/backtest-auditor/security/advisories/new). Don't open a public issue for them.

Include the version, what you did, what happened, and a minimal reproduction if you have one.

## What the server is designed to do

These are the properties we treat as security bugs if they break:

- **It never runs your code.** Python files and notebooks are parsed with `ast`. Nothing in `nba_auditor/` imports,
  executes or evaluates user code, and nothing starts a process (`tests/test_static_only.py`).
- **It makes no network calls.** The server runs locally over stdio and needs no API key.
- **It doesn't write files.** All six MCP tools are read-only. Suggested edits are returned as text; your client
  decides whether to apply them.
- **It bounds what it reads.** `check_path` reads only `.py`, `.pyw` and `.ipynb` files, skips virtual environments,
  hidden folders and caches, doesn't follow folder symlinks, refuses network (UNC) and device paths, and has file
  count and size limits. Data tools accept local `.csv`, `.json` or `.txt` files, and their error messages never
  quote file contents.
- **Tool output is marked as untrusted.** Results quote file names and code lines, so they are flagged as data,
  not instructions, for the AI client that reads them.

Anything the tools return goes to your AI assistant's provider like the rest of the chat. That is how MCP clients
work, not a vulnerability in this server.

## Supported versions

Only the latest release on PyPI (`backtest-auditor`) gets fixes.
