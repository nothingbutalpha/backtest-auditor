"""Red-team round 14 (lint v0.6) fresh files: recall and false alarms.

Usage: python score_rt14.py <this folder> <auditor dir to import nba_auditor from>
Leaky files carry `# leak: N` on line 1 (caught = high/medium within +/-2 lines); clean files `# expect: none`.
"""
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
sys.path.insert(0, sys.argv[2])
from nba_auditor.lint import check_code  # noqa: E402

caught = fa = nl = nc = 0
for kind in ("leaky", "clean"):
    for p in sorted((D / kind).glob("*.py")):
        src = p.read_text(encoding="utf-8")
        fs = check_code(src).findings
        hm = [f for f in fs if f.severity in ("high", "medium")]
        allf = ", ".join(f"{f.rule_id}:{f.severity}@{f.line}" + (f"(same_as {f.same_as})" if getattr(f, "same_as", 0) else "")
                         for f in fs if f.severity != "info") or "-"
        if kind == "leaky":
            nl += 1
            lines = [int(x) for x in re.findall(r"\d+", src.splitlines()[0])]
            ok = any(abs(f.line - ln) <= 2 for f in hm for ln in lines)
            caught += ok
            print(f"{'CAUGHT' if ok else 'MISSED'} {p.name} [{allf}]")
        else:
            nc += 1
            bad = bool(hm)
            fa += bad
            print(f"{'FALSE-ALARM' if bad else 'ok'} {p.name} [{allf}]")
print(f"recall {caught}/{nl}; false alarms {fa}/{nc}")
