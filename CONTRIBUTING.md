# Contributing

Thanks for looking. The most useful contributions are reports, not code.

## Reports

- **A leak it misses.** Use the "Missed leak" issue template. Include the smallest Python snippet that still has the
  leak, the line that uses future information, and the version.
- **A false alarm on correct code.** Use the "False alarm on correct code" template. Include the snippet, the rule
  id and line it reported, and why the code is correct.
- **A docs or install problem.** A plain issue is fine. Say which client (Claude Code, Claude Desktop, Cursor,
  VS Code or another) and which operating system.

Issues are public. Remove anything private (paths, data, credentials) before posting. For security problems, see
[SECURITY.md](SECURITY.md) instead.

We review code for the patterns the checks look for. We can't tell you whether a strategy is any good or whether
to trade it.

Snippets from issues may become regression tests in `tests/`. They are never added to our sealed test sets,
because a test the rules were tuned on can't measure them fairly.

## Pull requests

Pull requests are welcome, but this repository is a mirror. The code is maintained in our main repository and
exported here at each release. We don't merge pull requests here directly: we port the change, run it through our
own review and test suite, and it comes back with the next release. Rule
changes are normally scored once on a fresh sealed test set before a release, so a change can wait for the next test.

Before opening one:

1. Use Python 3.12. Install with `pip install -r requirements.txt` and `pip install -e . --no-deps`.
2. Run `python -m pytest -q`. Everything should pass. The timing tests use a tighter budget locally (3 s) than on
   CI (5 s), so on a busy machine one of them can fail without a real problem.
3. Add a test for the change: a leaky snippet that should be flagged, and a correct look-alike that shouldn't.
4. Keep the server read-only. Nothing in `nba_auditor/` may import or run user code, start processes or open
   network connections (`tests/test_static_only.py` checks this).

By contributing you agree that your contribution is licensed under Apache-2.0 (see [LICENSE](LICENSE), section 5).

## Who reads these

Issues and pull requests are triaged by AI agents working for the company.
