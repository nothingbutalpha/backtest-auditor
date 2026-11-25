"""Agent-fix eval: apply the auditor's suggested fixes to small leaky and clean repos, then score the result.

Deterministic mode (default, no LLM): each repo is copied to a temporary folder, and a scripted "agent" loops
check_path -> apply the first suggested fix -> check_path again, until no fix is left. That is what an agent
following the MCP instructions does ("apply suggested_fixes as given, re-run").

Score mode (`--score-dir DIR`): DIR holds copies of the repos that something else edited (for example Claude Code
running the `find_lookahead_bias` prompt; see README.md). Only the scoring runs.

Metrics (per repo and in total):
- planted_leaks:        leaks listed in manifest.json.
- detected:             planted leaks with a high or medium finding of their rule on their lines, before any fix.
- fixed_by_suggestion:  planted leaks whose lines a suggested fix edited (deterministic mode only).
- leaks_removed:        planted leaks with no finding of their rule on their lines after the fixes.
- new_findings:         high or medium findings after the fixes that weren't there before (rule, line).
- clean_lines_changed:  changed lines outside the fixable planted leaks' lines (must be 0). Every line of a clean
                        file is a clean line.
- semantic_changes:     changed lines that the manifest marks `protected` or `no_safe_fix` (places where the obvious
                        mechanical fix changes what the code means), or lines edited to anything but their
                        `expect` text (must be 0).
- unparseable_after:    edited Python files that no longer parse.

Code is parsed, never imported or run. Usage:

    python eval/agent_fix/run_eval.py              # deterministic run, Markdown table
    python eval/agent_fix/run_eval.py --json
    python eval/agent_fix/run_eval.py --score-dir /path/to/edited/repos
"""

from __future__ import annotations

import argparse
import ast
import codecs
import difflib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # the auditor folder, so the eval runs from a plain checkout

from nba_auditor import check_code, check_path  # noqa: E402
from nba_auditor.fixes import Edit, apply_edits  # noqa: E402
from nba_auditor.path_audit import NB_SUFFIX, decode_source, notebook_source  # noqa: E402

REPOS = HERE / "repos"
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))["repos"]
ACTIONABLE = ("high", "medium")
MAX_STEPS = 100


def _read_code(p: Path) -> tuple[str, str, bytes]:
    """(text used for linting, encoding, raw bytes). Notebooks give their joined code cells."""
    raw = p.read_bytes()
    text, enc = decode_source(raw)
    if p.name.endswith(NB_SUFFIX):
        text, _ = notebook_source(text)
    return text, enc, raw


def _write_code(p: Path, text: str, enc: str) -> None:
    data = text.encode("utf-8" if enc == "utf-8-sig" else enc)
    if enc == "utf-8-sig":
        data = codecs.BOM_UTF8 + data
    p.write_bytes(data)


def _findings(text: str) -> list:
    return check_code(text).findings if text.strip() else []


def run_agent(folder: Path) -> tuple[int, dict[str, set[int]]]:
    """Apply suggested fixes one at a time, re-checking after each. Returns (fixes applied, edited lines per file)."""
    applied, touched = 0, {}
    for _ in range(MAX_STEPS):
        res = check_path(str(folder), max_files=500)
        fixes = [fx for fr in res["files"] for fx in fr.get("suggested_fixes", [])]
        if not fixes:
            break
        fx = sorted(fixes, key=lambda f: (f["file"], f["finding_line"]))[0]
        path = folder / fx["file"]
        text, enc, _ = _read_code(path)
        edits = [Edit(**e) for e in fx["edits"]]
        for e in edits:  # the edit must still match the file (it was computed from it a moment ago)
            cur = "\n".join(text.split("\n")[e.start_line - 1:e.end_line]).replace("\r", "")
            assert cur == e.original, (fx["file"], e.start_line)
        _write_code(path, apply_edits(text, edits), enc)
        touched.setdefault(fx["file"], set()).update(ln for e in edits for ln in range(e.start_line, e.end_line + 1))
        applied += 1
    return applied, touched


def _changed_lines(before: str, after: str) -> set[int]:
    a, b = before.replace("\r", "").split("\n"), after.replace("\r", "").split("\n")
    out = set()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            out |= set(range(i1 + 1, i2 + 1))
        elif tag == "insert":
            out.add(max(i1, 1))  # an inserted line counts against the line it follows
    return out


def score_repo(name: str, original: Path, edited: Path, touched: dict[str, set[int]] | None = None) -> dict:
    spec = MANIFEST[name]
    on_disk = sorted(p.relative_to(original).as_posix() for p in original.rglob("*")
                     if p.is_file() and p.suffix in (".py", NB_SUFFIX))
    assert sorted(spec) == on_disk, f"manifest and files differ in {name}: {sorted(spec)} vs {on_disk}"
    row = {"repo": name, "files": len(spec), "planted_leaks": 0, "detected": 0, "fixed_by_suggestion": 0,
           "leaks_removed": 0, "new_findings": 0, "clean_lines_changed": 0, "semantic_changes": 0,
           "unparseable_after": 0, "details": []}
    for fname, fspec in spec.items():
        before, _, raw_before = _read_code(original / fname)
        after, _, raw_after = _read_code(edited / fname)
        fb, fa = _findings(before), _findings(after)
        allowed, protected = set(), set()
        for a, b in fspec.get("protected", []):
            protected |= set(range(a, b + 1))
        for leak in fspec["leaks"]:
            lines = set(range(leak["lines"][0], leak["lines"][1] + 1))
            if leak.get("no_safe_fix"):
                protected |= lines
            else:
                allowed |= lines | set(leak.get("also_allowed", []))
            row["planted_leaks"] += 1
            det = any(f.rule_id == leak["rule"] and f.line in lines and f.severity in ACTIONABLE for f in fb)
            gone = not any(f.rule_id == leak["rule"] and f.line in lines for f in fa)
            fixed = bool(touched and lines & touched.get(fname, set()))
            row["detected"] += det
            row["leaks_removed"] += gone
            row["fixed_by_suggestion"] += fixed
            row["details"].append({"file": fname, "rule": leak["rule"], "lines": leak["lines"], "detected": det,
                                   "removed": gone, "fixed_by_suggestion": fixed})
        old = {(f.rule_id, f.line) for f in fb if f.severity in ACTIONABLE}
        new = [(f.rule_id, f.severity, f.line) for f in fa if f.severity in ACTIONABLE and (f.rule_id, f.line) not in old]
        row["new_findings"] += len(new)
        if new:
            row["details"].append({"file": fname, "new_findings": new})
        changed = _changed_lines(before, after) if raw_before != raw_after else set()
        bad = sorted(changed - allowed)
        row["clean_lines_changed"] += len(bad)
        if bad:
            row["details"].append({"file": fname, "clean_lines_changed": bad})
        after_lines = after.replace("\r", "").split("\n")
        for leak in fspec["leaks"]:
            for ln, want in leak.get("expect", {}).items():
                ln = int(ln)
                if ln in changed and (ln > len(after_lines) or after_lines[ln - 1] != want):
                    protected.add(ln)  # edited, but not to the one meaning-preserving text
        sem = sorted(changed & protected)
        row["semantic_changes"] += len(sem)
        if sem:
            row["details"].append({"file": fname, "semantic_changes": sem})
        if raw_before != raw_after and not fname.endswith(NB_SUFFIX):
            try:
                ast.parse(after)
            except SyntaxError:
                row["unparseable_after"] += 1
    return row


def evaluate(score_dir: Path | None = None) -> dict:
    rows = []
    applied_total = 0
    for name in sorted(MANIFEST):
        if score_dir is not None:
            rows.append(score_repo(name, REPOS / name, score_dir / name))
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / name
            shutil.copytree(REPOS / name, work)
            applied, touched = run_agent(work)
            applied_total += applied
            row = score_repo(name, REPOS / name, work, touched)
            row["fixes_applied"] = applied
            rows.append(row)
    keys = ("files", "planted_leaks", "detected", "fixed_by_suggestion", "leaks_removed", "new_findings",
            "clean_lines_changed", "semantic_changes", "unparseable_after")
    total = {k: sum(r[k] for r in rows) for k in keys}
    if score_dir is None:
        total["fixes_applied"] = applied_total
    return {"mode": "score-only" if score_dir else "deterministic", "repos": rows, "total": total}


def markdown(res: dict) -> str:
    cols = ["repo", "files", "planted_leaks", "detected", "fixed_by_suggestion", "leaks_removed", "new_findings",
            "clean_lines_changed", "semantic_changes", "unparseable_after"]
    out = [f"Mode: {res['mode']}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in res["repos"] + [dict(res["total"], repo="**total**")]:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--score-dir", type=Path, help="score repos edited elsewhere (e.g. by an LLM agent)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = evaluate(a.score_dir)
    print(json.dumps(res, indent=2) if a.json else markdown(res))
    t = res["total"]
    return 1 if (t["clean_lines_changed"] or t["semantic_changes"] or t["new_findings"] or t["unparseable_after"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
