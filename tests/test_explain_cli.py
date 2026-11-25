"""explain() catalog and the command-line entry point."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_auditor import DISCLAIMER, explain, explain_markdown
from nba_auditor.catalog import CHECKS
from nba_auditor.cli import main


def test_explain_all_and_one():
    allc = explain()
    ids = [c["id"] for c in allc["checks"]]
    assert len(ids) == len(set(ids)) == len(CHECKS)
    one = explain("la07")
    assert one["id"] == "LA07" and one["references"] and one["disclaimer"] == DISCLAIMER
    assert explain_markdown("RA03").rstrip().endswith(DISCLAIMER)
    with pytest.raises(KeyError):
        explain("XX99")


def test_dsr_reference_is_cited():
    refs = " ".join(explain("RA03")["references"])
    assert "Bailey" in refs and "Deflated Sharpe" in refs and "40(5)" in refs


def test_cli_check_and_returns(tmp_path: Path, capsys):
    src = Path(__file__).parent / "corpus" / "leaky" / "l01_negative_shift_signal.py"
    assert main(["check", str(src), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["findings"][0]["rule_id"] == "LA01"
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300), index=pd.bdate_range("2020-01-01", periods=300))
    p = tmp_path / "r.csv"
    r.rename("return").rename_axis("date").reset_index().to_csv(p, index=False)
    assert main(["returns", str(p), "--n-trials", "20", "--turnover", "0.1"]) == 0
    md = capsys.readouterr().out
    assert "## Verdict" in md and md.rstrip().endswith(DISCLAIMER)
    assert main(["returns", str(tmp_path / "missing.csv")]) == 2
