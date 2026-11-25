# Agent-fix eval

Agents may act on the auditor's suggested fixes without asking, so a fix that changes correct code is worse than no fix.
This eval applies the suggested fixes to small leaky and clean repos and measures what changed.

**In-sample.** We wrote these repos and the fix rules, so the numbers below say the harness and the
fixes work as designed on known patterns. They are not a measure of how often fixes help on code we haven't seen.
They are not a performance claim and must not be quoted as one.

## Repos (`repos/`, ground truth in `manifest.json`)

| Repo | What's in it |
|---|---|
| `momentum_pandas` | A negative `pct_change` feature (no fix: the mirror flips its sign), a centered rolling mean, a signal shifted forward with `shift(-1)`; a clean, lagged backtest with costs |
| `ml_classifier` | A scaler fitted before a slice split, `train_test_split` without `shuffle=False`, shuffled `KFold`; a clean train/test script |
| `fundamentals` | `bfill()`, `fillna(method="backfill")`, `reindex(method="bfill")` on rows sorted ascending; a clean forward-filled version |
| `meaning_preservation` | Code where the obvious mechanical fix changes what the code means (from the red-team review of the fix rules): a centered window that builds a label, `bfill(axis=1)`, `bfill` on newest-first data, `close.shift(-1) / close - 1`, a lead lagged again later, a lead compared with the unshifted series, `pct_change(-k)` and `diff(-k)`, and from the re-review: a `train_test_split` and a shuffled `KFold` on a panel sorted by `[ticker, date]` (z01, y12). None of these lines may change. Plus z04: a lead whose result is lagged later, where the only allowed edit is dropping the lead |
| `near_misses` | Clean code written to look leaky: labels built with `shift(-1)`, `center=False`, `shuffle=False`, `TimeSeriesSplit`, forward fills |
| `unfixable` | Leaks that get no suggested fix (the edit would be ambiguous): same-bar signal and return, a label the lint misreads as a feature (`up`), `interpolate()`, a stratified split |
| `encodings` | A UTF-8 file with a byte-order mark and CRLF line endings, a latin-1 file with a coding line, a notebook |

## Deterministic run (no LLM, no network)

```bash
python eval/agent_fix/run_eval.py          # Markdown table; exit 1 if a clean or protected line changed
python eval/agent_fix/run_eval.py --json   # per-leak details
```

A scripted agent copies each repo to a temporary folder and loops: `check_path` -> apply the first suggested fix
-> `check_path` again, until no fix is left. Then the copy is compared with the original.

Result on 2026-10-16, after the red-team fixes (in-sample; `tests/test_agent_tools.py` checks these numbers):

| repo | files | planted_leaks | detected | fixed_by_suggestion | leaks_removed | new_findings | clean_lines_changed | semantic_changes | unparseable_after |
|---|---|---|---|---|---|---|---|---|---|
| encodings | 3 | 3 | 3 | 2 | 2 | 0 | 0 | 0 | 0 |
| fundamentals | 2 | 3 | 3 | 3 | 3 | 0 | 0 | 0 | 0 |
| meaning_preservation | 10 | 7 | 7 | 1 | 1 | 0 | 0 | 0 | 0 |
| ml_classifier | 4 | 3 | 3 | 3 | 3 | 0 | 0 | 0 | 0 |
| momentum_pandas | 2 | 3 | 3 | 2 | 2 | 0 | 0 | 0 | 0 |
| near_misses | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unfixable | 3 | 4 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| **total** | 27 | 23 | 22 | 11 | 11 | 0 | 0 | 0 | 0 |

For comparison, the fix rules as first submitted for review, run on this same eval, made
**10 semantic changes** (and changed 10 clean lines) across `meaning_preservation` and `momentum_pandas`. That
is why the eval now fails on any of them.

- **detected**: a high or medium finding of the planted rule on the planted lines, before any fix. The one miss is
  `interpolate()`, which the lint flags as "possible" only. (With the v0.2 lint, `diff(-5)` was a second miss; lint
  v0.3 flags it, which moved `detected` from 21 to 22. The fix results did not change.)
- **leaks_removed**: no finding of that rule on those lines afterwards. The 12 not removed get no suggested fix by
  design: 4 in `unfixable`, 7 in `meaning_preservation` and `momentum_pandas` where the mechanical fix would change
  the meaning (including the two panel splits), and the notebook (no fixes for notebooks yet, and its
  `pct_change(-10)` has no safe mirror).
- **semantic_changes**: changed lines the manifest marks `protected` or `no_safe_fix`, or lines edited to anything
  other than their `expect` text (z04: only dropping the lead is allowed). Must be 0.
- **new_findings** and **clean_lines_changed** must be 0. In `unfixable/label_named_up.py` the lint flags line 7
  (LA01) but that line builds the model's label; the fix rules refuse to touch it, which is why it stays clean.

## LLM in the loop (documented, not run: it makes paid API calls)

The same scorer grades edits made by a real agent. Nothing below has been run.

1. Copy the repos somewhere disposable: `cp -r eval/agent_fix/repos /tmp/agent_fix_run`
2. Give the agent the auditor's MCP server, using the private install from `docs/quickstart.md`:

   ```json
   {"mcpServers": {"nba-auditor": {"command": "/abs/path/.venv/bin/nba-auditor-mcp", "args": []}}}
   ```

3. For each repo, run Claude Code headless with the `find_lookahead_bias` prompt's wording, allowing only the
   auditor's tools and file edits:

   ```bash
   cd /tmp/agent_fix_run/momentum_pandas
   claude -p --mcp-config /abs/path/mcp.json --strict-mcp-config \
     --allowedTools "mcp__nba-auditor__check_path,Edit,Read" \
     "Find look-ahead bias in $(pwd). Call check_path on it, apply every suggested fix as given, \
      don't edit anything else, then call check_path again and report what changed."
   ```

4. Score the edited copies: `python eval/agent_fix/run_eval.py --score-dir /tmp/agent_fix_run --json`

Read the result the same way: `clean_lines_changed` and `semantic_changes` must be 0. Two variants are worth comparing when this is run:
the agent told to apply only `suggested_fixes` (the prompt above), and the agent free to fix every finding itself.
The second shows what an agent does with findings that have no suggested edit. Record the model, the date, the
prompt and the cost next to any number, and treat a run on these repos as in-sample too.
