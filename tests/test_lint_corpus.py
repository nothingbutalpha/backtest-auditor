"""check_code against a corpus of leaky and clean strategy snippets.

Each corpus file starts with `# expect: RULE[:severity], ...` (or `# expect: none`).

* Recall: every expected rule fires on its leaky file, at the expected severity.
* Precision on leaky files: no *high/medium* finding from a rule the file doesn't expect.
* Precision on clean files: no finding above "info" at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nba_auditor import check_code

CORPUS = Path(__file__).parent / "corpus"
LEAKY = sorted((CORPUS / "leaky").glob("*.py"))
CLEAN = sorted((CORPUS / "clean").glob("*.py"))


def _expect(path: Path) -> dict[str, str | None]:
    line = re.search(r"# expect: (.*)", path.read_text(encoding="utf-8")).group(1).strip()
    if line == "none":
        return {}
    out = {}
    for item in line.split(","):
        rule, _, sev = item.strip().partition(":")
        out[rule] = sev or None
    return out


def test_corpus_is_big_enough():
    assert len(LEAKY) >= 20 and len(CLEAN) >= 15


@pytest.mark.parametrize("path", LEAKY, ids=lambda p: p.stem)
def test_leaky_snippet_is_flagged(path):
    exp = _expect(path)
    audit = check_code(path.read_text(encoding="utf-8"))
    by_rule = {}
    for f in audit.findings:
        by_rule.setdefault(f.rule_id, set()).add(f.severity)
    for rule, sev in exp.items():
        assert rule in by_rule, f"{path.name}: expected {rule}, got {sorted(by_rule)}"
        if sev:
            assert sev in by_rule[rule], f"{path.name}: expected {rule} at {sev}, got {by_rule[rule]}"
    unexpected = [f for f in audit.findings if f.rule_id not in exp and f.severity in ("high", "medium")]
    assert not unexpected, f"{path.name}: unexpected findings {[(f.rule_id, f.line) for f in unexpected]}"
    assert audit.score < 100


@pytest.mark.parametrize("path", CLEAN, ids=lambda p: p.stem)
def test_clean_snippet_has_no_findings(path):
    audit = check_code(path.read_text(encoding="utf-8"))
    noisy = [(f.rule_id, f.severity, f.line, f.message) for f in audit.findings if f.severity != "info"]
    assert not noisy, f"{path.name}: false positives {noisy}"
    assert audit.score == 100


def test_corpus_precision_and_recall_summary():
    """Aggregate numbers quoted in the report: precision over all non-info findings, recall over expectations."""
    tp = fp = expected = hit = 0
    for p in LEAKY + CLEAN:
        exp = _expect(p)
        found = check_code(p.read_text(encoding="utf-8")).findings
        rules = {f.rule_id for f in found}
        expected += len(exp)
        hit += len(set(exp) & rules)
        for f in found:
            if f.severity == "info":
                continue
            if f.rule_id in exp:
                tp += 1
            else:
                fp += 1
    precision = tp / (tp + fp)
    recall = hit / expected
    assert recall == 1.0
    assert precision >= 0.95, (tp, fp)


def test_phase0_same_close_leak_is_high():
    """The leak we know from our own research: rank on today's return, credit today's return."""
    src = (CORPUS / "leaky" / "l06_phase0_same_close.py").read_text(encoding="utf-8")
    f = [x for x in check_code(src).findings if x.rule_id == "LA07"]
    assert f and f[0].severity == "high" and f[0].line == 10
    clean = (CORPUS / "clean" / "c14_cross_sectional_next_day.py").read_text(encoding="utf-8")
    assert not [x for x in check_code(clean).findings if x.rule_id == "LA07"]
