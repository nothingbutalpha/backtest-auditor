"""Agent-facing tools: check_path (path safety), suggested fixes (self-check), audit_backtest, MCP prompts, eval."""

from __future__ import annotations

import codecs
import json
import os
import sys
from pathlib import Path

import anyio
import numpy as np
import pandas as pd
import pytest
from mcp import Client

from nba_auditor import DISCLAIMER, apply_edits, audit_backtest, check_code, check_path, suggest_fixes
from nba_auditor import path_audit
from nba_auditor.fixes import Edit, verify_fix
from nba_auditor.mcp import server
from nba_auditor.parsing import InputError
from nba_auditor.returns_audit import VERDICT_NOTE

AUDITOR = Path(__file__).resolve().parents[1]

LEAKY_CENTER = ("import pandas as pd\n"
                "df = pd.read_csv('p.csv')\n"
                "df['trend'] = df['close'].rolling(20, center=True).mean()\n"
                "df['signal'] = (df['close'] > df['trend']).astype(int)\n")


def _fix_map(src: str) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    for fx in suggest_fixes(src):
        out.setdefault(fx.rule_id, []).extend((e.original, e.replacement) for e in fx.edits)
    return out


# -- suggested fixes ---------------------------------------------------------------------------------------
FIX_CASES = {
    "LA01 bare signal lead": ("import pandas as pd\ndf = pd.read_csv('p.csv', index_col=0, parse_dates=True)\n"
                              "df['raw'] = (df['close'] > df['close'].rolling(20).mean()).astype(int)\n"
                              "df['signal'] = df['raw'].shift(-1)\n",
                              "LA01", "df['signal'] = df['raw'].shift(1)"),
    "LA01 lead lagged later: drop the lead (z04)": (
        "import pandas as pd\ndf = pd.read_csv('spy.csv', index_col=0, parse_dates=True).sort_index()\n"
        "df['vix_signal'] = df['vix'].shift(-1)\ndf['position'] = (df['vix_signal'] < 20).astype(int)\n"
        "df['strat'] = df['position'].shift(1) * df['close'].pct_change() - 0.0005\n",
        "LA01", "df['vix_signal'] = df['vix']"),
    "LA02": (LEAKY_CENTER, "LA02", "df['trend'] = df['close'].rolling(20, center=False).mean()"),
    "LA03 bfill": ("import pandas as pd\ndf = pd.read_csv('p.csv', index_col=0, parse_dates=True).sort_index()\n"
                   "df['eps'] = df['eps'].bfill()\n",
                   "LA03", "df['eps'] = df['eps'].ffill()"),
    "LA03 fillna": ("import pandas as pd\ndf = pd.read_csv('p.csv')\ndf = df.sort_values(['ticker', 'date'])\n"
                    "df['eps'] = df.groupby('ticker')['eps'].fillna(method='bfill', limit=3)\n",
                    "LA03", "df['eps'] = df.groupby('ticker')['eps'].ffill(limit=3)"),
    "LA03 inplace sort": ("import pandas as pd\ndf = pd.read_csv('p.csv', index_col=0, parse_dates=True)\n"
                          "df.sort_index(inplace=True)\ndf['eps'] = df['eps'].bfill()\n",
                          "LA03", "df['eps'] = df['eps'].ffill()"),
    "LA03 reindex": ("import pandas as pd\ndf = pd.read_csv('p.csv', index_col=0, parse_dates=True)\n"
                     "wk = df['close'].resample('W').last()\ndf['wk'] = wk.reindex(df.index, method='bfill')\n",
                     "LA03", "df['wk'] = wk.reindex(df.index, method='ffill')"),
    "LA05 fit_transform": ("import numpy as np\nfrom sklearn.preprocessing import StandardScaler\nX = np.load('X.npy')\n"
                           "split = int(len(X) * 0.8)\nX = StandardScaler().fit_transform(X)\n"
                           "X_train, X_test = X[:split], X[split:]\n",
                           "LA05", "X = StandardScaler().fit(X[:split]).transform(X)"),
    "LA05 iloc": ("import pandas as pd\nfrom sklearn.preprocessing import StandardScaler\ndf = pd.read_csv('f.csv')\n"
                  "n = int(len(df) * 0.7)\nscaler = StandardScaler()\ndf[['a']] = scaler.fit_transform(df[['a']])\n"
                  "train, test = df.iloc[:n], df.iloc[n:]\n",
                  "LA05", "df[['a']] = scaler.fit(df[['a']].iloc[:n]).transform(df[['a']])"),
    "LA06 default shuffle": ("import pandas as pd\nfrom sklearn.model_selection import train_test_split\n"
                             "df = pd.read_csv('p.csv', parse_dates=['date']).sort_values('date')\n"
                             "df['ret'] = df['close'].pct_change()\ndf['target'] = (df['ret'].shift(-1) > 0).astype(int)\n"
                             "X_train, X_test, y_train, y_test = train_test_split(df[['ret']], df['target'], test_size=0.2)\n",
                             "LA06", "X_train, X_test, y_train, y_test = train_test_split(df[['ret']], df['target'], "
                                     "test_size=0.2, shuffle=False)"),
}


@pytest.mark.parametrize("name", list(FIX_CASES))
def test_fix_is_proposed_and_removes_the_finding(name):
    src, rule, expected = FIX_CASES[name]
    assert any(f.rule_id == rule and f.severity in ("high", "medium") for f in check_code(src).findings)
    fixes = [fx for fx in suggest_fixes(src) if fx.rule_id == rule]
    assert len(fixes) == 1, fixes
    fx = fixes[0]
    assert [e.replacement for e in fx.edits] == [expected]
    assert fx.self_checked and fx.rationale and "\n" not in fx.rationale
    new = apply_edits(src, fx.edits)
    assert new.count("\n") == src.count("\n")  # edits never change the line count
    after = check_code(new)
    assert not [f for f in after.findings if f.rule_id == rule]
    before_hm = {(f.rule_id, f.severity) for f in check_code(src).findings if f.severity in ("high", "medium")}
    assert {(f.rule_id, f.severity) for f in after.findings if f.severity in ("high", "medium")} <= before_hm


def test_kfold_shuffle_becomes_time_series_split_with_import():
    src = ("import pandas as pd\nfrom sklearn.linear_model import Ridge\n"
           "from sklearn.model_selection import KFold, cross_val_score  # KFold here\n"
           "df = pd.read_csv('d.csv', parse_dates=['date']).sort_values('date')\n"
           "df['ret_1'] = df['close'].pct_change()\ndf['target'] = df['close'].pct_change().shift(-1)\n"
           "cv = KFold(n_splits=5, shuffle=True, random_state=7)\n"
           "print(cross_val_score(Ridge(), df[['ret_1']], df['target'], cv=cv).mean())\n")
    (fx,) = suggest_fixes(src)
    assert [(e.start_line, e.replacement) for e in fx.edits] == [
        (3, "from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit  # KFold here"),
        (7, "cv = TimeSeriesSplit(n_splits=5)")]


NO_FIX = {
    "label used as y (LA01 on a column the lint can't tell is a label)":
        ("import pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n"
         "df = pd.read_csv('spy.csv', index_col=0, parse_dates=True)\ndf['r1'] = df['Close'].pct_change()\n"
         "df['up'] = (df['Close'].shift(-1) > df['Close']).astype(int)\n"
         "clf = LogisticRegression().fit(df[['r1']], df['up'])\n"),
    "already lagged elsewhere in the statement":
        ("import pandas as pd\ndf = pd.read_csv('p.csv')\n"
         "df['signal'] = (df['close'].shift(-1) > df['close'].shift(1)).astype(int)\n"),
    "two negative shifts (ambiguous)":
        ("import pandas as pd\ndf = pd.read_csv('p.csv')\n"
         "df['signal'] = (df['close'].shift(-1) > df['open'].shift(-2)).astype(int)\n"),
    "shift by a variable":
        ("import pandas as pd\nH = 5\ndf = pd.read_csv('p.csv')\n"
         "df['signal'] = (df['close'].pct_change(-H) > 0).astype(int)\n"),
    "stratified split needs shuffling":
        ("import pandas as pd\nfrom sklearn.model_selection import train_test_split\n"
         "df = pd.read_csv('p.csv', parse_dates=['date']).sort_values('date')\ndf['ret'] = df['close'].pct_change()\n"
         "df['target'] = (df['ret'].shift(-1) > 0).astype(int)\n"
         "a, b, c, d = train_test_split(df[['ret']], df['target'], test_size=0.2, stratify=df['target'])\n"),
    "interpolate has no one-for-one replacement":
        "import pandas as pd\ndf = pd.read_csv('p.csv')\ndf['eps'] = df['eps'].interpolate()\n",
    "fillna with a value and a method":
        "import pandas as pd\ndf = pd.read_csv('p.csv')\ndf['eps'] = df['eps'].fillna(0, method='bfill')\n",
    "fit with no visible split":
        ("import numpy as np\nfrom sklearn.preprocessing import StandardScaler\nX = np.load('X.npy')\n"
         "X = StandardScaler().fit_transform(X)\nnp.save('Xs.npy', X)\n"),
    "fit with two different split points":
        ("import numpy as np\nfrom sklearn.preprocessing import StandardScaler\nX = np.load('X.npy')\na, b = 100, 200\n"
         "X = StandardScaler().fit_transform(X)\nX_train, X_test = X[:a], X[b:]\n"),
    "split point defined after the fit":
        ("import numpy as np\nfrom sklearn.preprocessing import StandardScaler\nX = np.load('X.npy')\n"
         "X = StandardScaler().fit_transform(X)\nsplit = int(len(X) * 0.8)\nX_train, X_test = X[:split], X[split:]\n"),
    "KFold imported over several lines":
        ("import pandas as pd\nfrom sklearn.linear_model import Ridge\nfrom sklearn.model_selection import (\n"
         "    KFold, cross_val_score)\ndf = pd.read_csv('d.csv', parse_dates=['date']).sort_values('date')\n"
         "df['ret_1'] = df['close'].pct_change()\ndf['target'] = df['close'].pct_change().shift(-1)\n"
         "cv = KFold(n_splits=5, shuffle=True, random_state=7)\n"
         "print(cross_val_score(Ridge(), df[['ret_1']], df['target'], cv=cv).mean())\n"),
}

_SPY = "import pandas as pd\ndf = pd.read_csv('spy.csv', index_col=0, parse_dates=True).sort_index()\n"
# PR #11 red-team review: the lint flags these (high/medium), and the mechanical fix would change their meaning.
MEANING = {
    "x04 centered window feeding a label": _SPY + "df['trend'] = df['close'].rolling(21, center=True).mean()\n"
                                                  "df['label'] = (df['trend'].diff() > 0).astype(int)\n",
    "x05 bfill across columns": _SPY + "df[['open', 'close']] = df[['open', 'close']].bfill(axis=1)\n"
                                       "df['signal'] = (df['close'] > df['close'].rolling(20).mean()).astype(int)\n",
    "x07 lead divided by the series": _SPY + "df['signal'] = (df['close'].shift(-1) / df['close'] - 1 > 0).astype(int)\n",
    "x10 lead lagged again later": _SPY + "df['fast'] = df['close'].rolling(10).mean()\n"
                                          "df['slow'] = df['close'].rolling(50).mean()\n"
                                          "df['signal'] = (df['fast'].shift(-1) > df['slow']).astype(int)\n"
                                          "df['position'] = df['signal'].shift(1)\n",
    "x12 bfill on newest-first data": ("import pandas as pd\ndf = pd.read_csv('newest_first.csv', index_col=0, parse_dates=True)\n"
                                       "df['close'] = df['close'].bfill()\ndf = df.sort_index()\n"
                                       "df['signal'] = df['close'] > df['close'].rolling(20).mean()\n"),
    "bfill after a descending sort": ("import pandas as pd\ndf = pd.read_csv('p.csv', index_col=0, parse_dates=True)"
                                      ".sort_index(ascending=False)\ndf['eps'] = df['eps'].bfill()\n"),
    "bfill after sorting by a non-date key": ("import pandas as pd\ndf = pd.read_csv('p.csv').sort_values('ticker')\n"
                                              "df['eps'] = df['eps'].bfill()\n"),
    "bfill after a reversing slice": ("import pandas as pd\ndf = pd.read_csv('p.csv', index_col=0, parse_dates=True)"
                                      ".sort_index().iloc[::-1]\ndf['eps'] = df['eps'].bfill()\n"),
    "pct_change(-k) (mirror flips the sign)": _SPY + "df['mom'] = df['close'].pct_change(-5)\n"
                                                     "df['signal'] = (df['mom'] > 0).astype(int)\n",
    "lead compared with the series": _SPY + "df['signal'] = (df['close'].shift(-1) > df['close']).astype(int)\n",
    "lead column compared later": _SPY + "df['nxt'] = df['close'].shift(-1)\n"
                                         "df['signal'] = (df['nxt'] > df['close']).astype(int)\n",
    "z01 train_test_split on a ticker-sorted panel": ("import pandas as pd\nfrom sklearn.model_selection import train_test_split\n"
                                                     "panel = pd.read_parquet('p.parquet').sort_values(['ticker', 'date'])\n"
                                                     "X, y = panel[['mom', 'bm']], panel['fwd_ret']\n"
                                                     "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"),
    "y12 shuffled KFold on a ticker-sorted panel": ("import pandas as pd\nfrom sklearn.linear_model import Ridge\n"
                                                    "from sklearn.model_selection import KFold, cross_val_score\n"
                                                    "panel = pd.read_parquet('p.parquet').sort_values(['ticker', 'date'])\n"
                                                    "X, y = panel[['mom', 'bm']], panel['fwd_ret']\n"
                                                    "print(cross_val_score(Ridge(), X, y, cv=KFold(5, shuffle=True)).mean())\n"),
    "split on a panel sorted by date (many rows per date)": (
        "import pandas as pd\nfrom sklearn.model_selection import train_test_split\n"
        "df = pd.read_parquet('p.parquet').sort_values('date')\ndf['rank'] = df.groupby('date')['mom'].rank()\n"
        "a, b, c, d = train_test_split(df[['rank']], df['fwd_ret'], test_size=0.2)\n"),
    "split with no visible ordering": ("import pandas as pd\nfrom sklearn.model_selection import train_test_split\n"
                                       "df = pd.read_csv('d.csv')\n"
                                       "a, b, c, d = train_test_split(df[['x']], df['target'], test_size=0.2)\n"),
    "rolled then shifted (not a plain series)": _SPY + "df['vol'] = df['close'].pct_change().rolling(21).std().shift(-21)\n"
                                                       "df['weight'] = 0.1 / df['vol']\n",
}


@pytest.mark.parametrize("name", list(NO_FIX))
def test_no_fix_for_ambiguous_cases(name):
    assert suggest_fixes(NO_FIX[name]) == []


@pytest.mark.parametrize("name", list(MEANING))
def test_no_fix_where_the_mechanical_fix_changes_meaning(name):
    src = MEANING[name]
    assert suggest_fixes(src) == [], [(f.rule_id, [e.replacement for e in f.edits]) for f in suggest_fixes(src)]


def test_clean_code_gets_no_fix():
    for p in sorted((AUDITOR / "tests" / "corpus_v02" / "clean").glob("*.py")) + \
             sorted((AUDITOR / "tests" / "corpus" / "clean").glob("*.py")):
        assert suggest_fixes(p.read_text(encoding="utf-8")) == [], p.name


def test_self_check_rejects_bad_edits():
    src = LEAKY_CENTER
    # an edit that leaves the finding in place
    assert not verify_fix(src, "LA02", [Edit(3, 3, "", "df['trend'] = df['close'].rolling(21, center=True).mean()")])
    # an edit that removes it but introduces a new high finding
    assert not verify_fix(src, "LA02", [Edit(3, 3, "", "df['trend'] = df['close'].shift(-1).rolling(20).mean()")])
    # an edit that doesn't parse
    assert not verify_fix(src, "LA02", [Edit(3, 3, "", "df['trend'] = df['close'].rolling(20,")])
    # the real fix passes
    assert verify_fix(src, "LA02", [Edit(3, 3, "", "df['trend'] = df['close'].rolling(20, center=False).mean()")])


def test_fixes_are_dropped_when_the_self_check_fails(monkeypatch):
    from nba_auditor import fixes

    monkeypatch.setattr(fixes, "verify_fix", lambda *a, **k: False)
    assert fixes.suggest_fixes(LEAKY_CENTER) == []


def test_fixes_handle_crlf_bom_and_non_ascii():
    src = "﻿# Bénéfice — ünïcode\r\n" + LEAKY_CENTER.replace("\n", "\r\n")
    (fx,) = suggest_fixes(src)
    (e,) = fx.edits
    assert e.start_line == 4 and "\r" not in e.original + e.replacement
    new = apply_edits(src.removeprefix("﻿"), fx.edits)
    assert new.count("\r\n") == src.count("\r\n") and "center=False" in new


def test_check_code_tool_strips_bom_and_returns_fixes():
    async def main():
        async with Client(server) as c:
            res = await c.call_tool("check_code", {"source": "﻿" + LEAKY_CENTER})
            return res.structured_content or json.loads(res.content[0].text)

    out = anyio.run(main)
    assert out["report"]["parsed_with_ast"] is True
    assert [(f["rule_id"], f["severity"]) for f in out["report"]["findings"]] == [("LA02", "high")]
    assert out["suggested_fixes"][0]["edits"][0]["replacement"].endswith("center=False).mean()")


# -- check_path --------------------------------------------------------------------------------------------
def _w(p: Path, text: str | bytes) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    return p


def test_check_path_file_and_folder(tmp_path):
    _w(tmp_path / "strat" / "a.py", LEAKY_CENTER)
    _w(tmp_path / "strat" / "clean.py", "x = 1\n")
    _w(tmp_path / "notes.txt", "rolling(5, center=True)\n")
    res = check_path(str(tmp_path))
    assert res["files_checked"] == 2 and res["clean_files"] == ["strat/clean.py"]
    (fr,) = res["files"]
    assert fr["path"] == "strat/a.py"
    (f,) = fr["findings"]
    assert (f["rule_id"], f["severity"], f["start_line"], f["end_line"]) == ("LA02", "high", 3, 3)
    assert f["title"] and f["why_it_matters"].endswith(".") and len(f["why_it_matters"]) < 300
    assert fr["suggested_fixes"][0]["file"] == "strat/a.py"
    assert res["disclaimer"] == DISCLAIMER
    one = check_path(str(tmp_path / "strat" / "a.py"))
    assert one["files"][0]["path"] == "a.py"
    with pytest.raises(ValueError, match="Python files"):
        check_path(str(tmp_path / "notes.txt"))
    with pytest.raises(ValueError, match="not found"):
        check_path(str(tmp_path / "missing"))


def test_multiline_statement_gets_its_full_range(tmp_path):
    _w(tmp_path / "m.py", "import pandas as pd\ndf = pd.read_csv('p.csv')\n"
                          "df['trend'] = (\n    df['close']\n    .rolling(20, center=True)\n    .mean()\n)\n")
    (f,) = check_path(str(tmp_path))["files"][0]["findings"]
    assert (f["start_line"], f["end_line"]) == (3, 7)


def test_include_glob_and_only_python(tmp_path):
    _w(tmp_path / "strategies" / "s.py", LEAKY_CENTER)
    _w(tmp_path / "lib" / "l.py", LEAKY_CENTER)
    _w(tmp_path / "strategies" / "s.txt", LEAKY_CENTER)
    res = check_path(str(tmp_path), include_glob="strategies/*")
    assert [f["path"] for f in res["files"]] == ["strategies/s.py"]


def test_skips_venvs_node_modules_git_and_hidden(tmp_path):
    for d in ("node_modules/pkg", ".git/hooks", ".venv/lib", "venv/lib", "site-packages/x", ".claude/worktrees/w",
              "__pycache__", "build/lib", "proj.egg-info"):
        _w(tmp_path / d / "leak.py", LEAKY_CENTER)
    _w(tmp_path / "myenv" / "pyvenv.cfg", "home = x\n")
    _w(tmp_path / "myenv" / "Lib" / "leak.py", LEAKY_CENTER)
    _w(tmp_path / "src" / "ok.py", LEAKY_CENTER)
    res = check_path(str(tmp_path))
    assert [f["path"] for f in res["files"]] == ["src/ok.py"]
    assert {"path": "myenv", "reason": "virtual environment"} in res["skipped"]


def test_huge_binary_undecodable_empty_files(tmp_path, monkeypatch):
    monkeypatch.setattr(path_audit, "MAX_FILE_BYTES", 2000)
    _w(tmp_path / "huge.py", "x = 1\n" * 1000)
    _w(tmp_path / "bin.py", b"\x00\x01\x02import os\n")
    _w(tmp_path / "cp1252.py", "# caf\xe9\nx = 1\n".encode("cp1252"))
    _w(tmp_path / "latin.py", "# -*- coding: latin-1 -*-\n# caf\xe9\nimport pandas as pd\n"
                              "df = pd.read_csv('p.csv')\ndf['e'] = df['e'].bfill()\n".encode("latin-1"))
    _w(tmp_path / "empty.py", b"")
    _w(tmp_path / "bad.ipynb", "{not json")
    res = check_path(str(tmp_path))
    reasons = {s["path"]: s["reason"] for s in res["skipped"]}
    assert reasons["huge.py"].startswith("too large")
    assert reasons["bin.py"] == "binary file (NUL bytes)"
    assert reasons["cp1252.py"] == "not valid UTF-8 and no usable coding declaration"
    assert reasons["empty.py"] == "empty file"
    assert reasons["bad.ipynb"] == "notebook is not valid JSON"
    (fr,) = res["files"]
    assert fr["path"] == "latin.py" and fr["encoding"] == "iso-8859-1" and fr["findings"][0]["rule_id"] == "LA03"


def test_bom_is_stripped_before_parsing(tmp_path):
    _w(tmp_path / "bom.py", codecs.BOM_UTF8 + LEAKY_CENTER.replace("\n", "\r\n").encode("utf-8"))
    (fr,) = check_path(str(tmp_path))["files"]
    assert fr["parsed_with_ast"] is True and fr["encoding"] == "utf-8-sig"
    assert [(f["rule_id"], f["severity"]) for f in fr["findings"]] == [("LA02", "high")]  # not the text fallback
    assert fr["suggested_fixes"]


def test_total_and_file_count_caps(tmp_path, monkeypatch):
    for i in range(5):
        _w(tmp_path / f"f{i}.py", "x = 1\n" * 50)
    res = check_path(str(tmp_path), max_files=3)
    assert res["files_checked"] == 3 and res["truncated"]
    assert any("file limit" in s["reason"] for s in res["skipped"])
    monkeypatch.setattr(path_audit, "MAX_TOTAL_BYTES", 700)
    res = check_path(str(tmp_path))
    assert res["files_checked"] == 2 and res["truncated"]
    assert any("total size limit" in s["reason"] for s in res["skipped"])
    with pytest.raises(ValueError, match="max_files"):
        check_path(str(tmp_path), max_files=0)


def test_permission_errors_are_reported_not_raised(tmp_path, monkeypatch):
    _w(tmp_path / "locked.py", LEAKY_CENTER)
    _w(tmp_path / "ok.py", LEAKY_CENTER)
    _w(tmp_path / "secret" / "s.py", LEAKY_CENTER)
    real_open, real_scandir = open, os.scandir

    def fake_open(p, *a, **k):
        if Path(p).name == "locked.py":
            raise PermissionError(13, "Permission denied")
        return real_open(p, *a, **k)

    def fake_scandir(p):
        if Path(p).name == "secret":
            raise PermissionError(13, "Permission denied")
        return real_scandir(p)

    monkeypatch.setattr(path_audit, "open", fake_open, raising=False)
    monkeypatch.setattr(path_audit.os, "scandir", fake_scandir)
    res = check_path(str(tmp_path))
    reasons = {s["path"]: s["reason"] for s in res["skipped"]}
    assert reasons["locked.py"] == "unreadable (Permission denied)"
    assert reasons["secret"] == "unreadable folder (Permission denied)"
    assert [f["path"] for f in res["files"]] == ["ok.py"]


def _make_dir_link(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            pytest.skip("cannot create directory symlinks here")
        import _winapi

        _winapi.CreateJunction(str(target), str(link))  # no admin rights needed on Windows


def test_link_loops_are_not_followed(tmp_path):
    root = tmp_path / "repo"
    _w(root / "a" / "s.py", LEAKY_CENTER)
    _make_dir_link(root / "a" / "loop", root)  # a -> loop -> repo -> a -> ...
    _make_dir_link(root / "outside", tmp_path)
    res = check_path(str(root))
    assert [f["path"] for f in res["files"]] == ["a/s.py"]
    assert {s["path"] for s in res["skipped"]} == {"a/loop", "outside"}
    assert all("not followed" in s["reason"] for s in res["skipped"])


def test_file_symlink_outside_root_is_not_read(tmp_path):
    _w(tmp_path / "secret.py", LEAKY_CENTER)
    root = tmp_path / "repo"
    _w(root / "s.py", "x = 1\n")
    try:
        os.symlink(tmp_path / "secret.py", root / "link.py")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create file symlinks here (Windows without developer mode)")
    res = check_path(str(root))
    assert res["files"] == [] and res["skipped"][0]["path"] == "link.py"


def test_code_is_never_executed(tmp_path):
    marker = tmp_path / "EXECUTED"
    _w(tmp_path / "evil.py", f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\n"
                             "import pandas as pd\ndf = pd.read_csv('p.csv')\ndf['e'] = df['e'].bfill()\n")
    _w(tmp_path / "nb.ipynb", json.dumps({"cells": [{"cell_type": "code", "source": [
        f"open({str(marker)!r}, 'w').write('x')\n", "df['m'] = df['close'].pct_change(-3)\n",
        "df['signal'] = (df['m'] > 0).astype(int)"]}]}))
    before = sorted(p.name for p in tmp_path.iterdir())
    res = check_path(str(tmp_path))
    assert not marker.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # read-only: nothing written
    nb = next(f for f in res["files"] if f["path"] == "nb.ipynb")
    f = next(x for x in nb["findings"] if x["rule_id"] == "LA01")
    assert (f["cell"], f["cell_line"]) == (0, 2) and "suggested_fixes" not in nb


def test_response_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(path_audit, "MAX_FINDINGS_RETURNED", 3)
    for i in range(3):
        _w(tmp_path / f"f{i}.py", LEAKY_CENTER + "df['t2'] = df['close'].rolling(5, center=True).mean()\n")
    res = check_path(str(tmp_path))
    assert sum(len(f["findings"]) for f in res["files"]) == 3 and res["truncated"]
    assert res["counts"]["high"] == 6  # totals still count everything


# -- audit_backtest ----------------------------------------------------------------------------------------
def _csv(path: Path, s: pd.Series) -> str:
    s.rename("return").rename_axis("date").reset_index().to_csv(path, index=False)
    return str(path)


@pytest.fixture
def series(tmp_path):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2019-01-01", periods=1260)
    strong = pd.Series(rng.normal(0.0012, 0.01, len(idx)), index=idx)
    noise = pd.Series(rng.normal(0.0, 0.01, len(idx)), index=idx)
    bench = pd.Series(rng.normal(0.0003, 0.011, len(idx)), index=idx)
    return {"strong": _csv(tmp_path / "strong.csv", strong), "noise": _csv(tmp_path / "noise.csv", noise),
            "bench": _csv(tmp_path / "bench.csv", bench)}


def test_audit_backtest_combines_and_caps(tmp_path, series):
    clean = _w(tmp_path / "code" / "clean.py", "x = 1\n")
    leaky = _w(tmp_path / "leaky" / "s.py", LEAKY_CENTER)

    best = audit_backtest(str(clean), series["strong"], benchmark_csv=series["bench"], n_trials=1, turnover=0.05)
    assert best["verdict"] == "No red flags found", best["reasons"]
    assert best["verdict_note"] == VERDICT_NOTE and VERDICT_NOTE in best["markdown"]
    assert best["markdown"].rstrip().endswith(DISCLAIMER)
    assert any("weak evidence" in r for r in best["reasons"])

    nobench = audit_backtest(str(clean), series["strong"], n_trials=1, turnover=0.05)
    assert nobench["verdict"] == "Some red flags"
    assert any("No benchmark supplied" in r for r in nobench["reasons"])

    code_only = audit_backtest(str(clean))
    assert code_only["verdict"] == "Some red flags" and VERDICT_NOTE in code_only["markdown"]
    assert any("Only the code was checked" in r for r in code_only["reasons"])

    bad_code = audit_backtest(str(leaky), series["strong"], benchmark_csv=series["bench"], n_trials=1, turnover=0.05)
    assert bad_code["verdict"] == "Major red flags"
    assert bad_code["parts"] == [{"part": "code", "verdict": "Major red flags"},
                                 {"part": "returns", "verdict": "No red flags found"}]
    assert any("s.py:3 LA02" in r for r in bad_code["reasons"])

    noise = audit_backtest(None, series["noise"], benchmark_csv=series["bench"], n_trials=20, turnover=0.05)
    assert noise["verdict"] == "Major red flags"
    assert any("code was not checked" in r for r in noise["reasons"])


def test_audit_backtest_trades_and_errors(tmp_path, series):
    trades = tmp_path / "trades.csv"
    trades.write_text("date,ticker,side,qty,price\n2024-01-02,ABC,buy,10,100\n" +
                      "".join(f"2024-{m:02d}-15,ABC,{'sell' if m % 2 else 'buy'},10,{100 + m}\n" for m in range(2, 12)),
                      encoding="utf-8")
    try:
        res = audit_backtest(trades_csv=str(trades))
        assert res["verdict"] in ("Some red flags", "Major red flags") and "trades" in res
    except InputError:
        pass  # too short a log is a clean input error, also fine
    with pytest.raises(InputError, match="at least one"):
        audit_backtest()
    with pytest.raises(InputError, match="needs returns_csv"):
        audit_backtest(code_path=str(tmp_path), benchmark_csv=series["bench"])
    with pytest.raises(InputError, match="not found"):
        audit_backtest(returns_csv=str(tmp_path / "nope.csv"))
    binf = _w(tmp_path / "r.csv", b"\x00\x01\x02")
    with pytest.raises(InputError, match="binary"):
        audit_backtest(returns_csv=str(binf))


def test_audit_backtest_size_cap(tmp_path, monkeypatch):
    from nba_auditor import backtest_audit

    monkeypatch.setattr(backtest_audit, "MAX_CSV_BYTES", 10)
    p = _w(tmp_path / "r.csv", "date,return\n2020-01-01,0.1\n")
    with pytest.raises(InputError, match="too large"):
        audit_backtest(returns_csv=str(p))


# -- MCP: new tools and prompts ----------------------------------------------------------------------------
def test_mcp_check_path_audit_backtest_and_prompts(tmp_path, series):
    _w(tmp_path / "s.py", LEAKY_CENTER)

    async def main():
        async with Client(server) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            for name in ("check_path", "audit_backtest"):
                assert tools[name].annotations.read_only_hint is True
            r = await c.call_tool("check_path", {"path": str(tmp_path)})
            out = r.structured_content or json.loads(r.content[0].text)
            assert out["report"]["files"][0]["suggested_fixes"] and "Suggested fix" in out["markdown"]
            r = await c.call_tool("check_path", {"path": str(tmp_path / "missing")})
            assert "not found" in (r.structured_content or json.loads(r.content[0].text))["error"]
            r = await c.call_tool("audit_backtest", {"code_path": str(tmp_path), "returns_csv": series["noise"]})
            out = r.structured_content or json.loads(r.content[0].text)
            assert out["report"]["verdict"] == "Major red flags" and VERDICT_NOTE in out["markdown"]
            r = await c.call_tool("audit_backtest", {})
            assert "at least one" in (r.structured_content or json.loads(r.content[0].text))["error"]

            prompts = {p.name: p for p in (await c.list_prompts()).prompts}
            assert prompts["audit_backtest_before_trusting"].title == "Audit this backtest before I rely on it"
            assert prompts["find_lookahead_bias"].title == "Find look-ahead bias in this repo"
            got = await c.get_prompt("find_lookahead_bias", {"path": "/repo", "apply_fixes": "yes"})
            text = got.messages[0].content.text
            assert "check_path" in text and "'/repo'" in text and "I asked for edits" in text and "show me" in text
            got = await c.get_prompt("audit_backtest_before_trusting", {"n_trials": "40"})
            text = got.messages[0].content.text
            assert "audit_backtest" in text and VERDICT_NOTE in text and "n_trials='40'" in text

    anyio.run(main)


# -- the agent-fix eval ------------------------------------------------------------------------------------
def test_agent_fix_eval_numbers():
    sys.path.insert(0, str(AUDITOR / "eval" / "agent_fix"))
    try:
        import run_eval
    finally:
        sys.path.pop(0)
    res = run_eval.evaluate()
    t = res["total"]
    assert t["clean_lines_changed"] == 0 and t["new_findings"] == 0 and t["unparseable_after"] == 0
    assert t["semantic_changes"] == 0  # no fix may change meaning in the red-team cases
    # the numbers quoted in eval/agent_fix/README.md (in-sample: the same person wrote the repos and the fixes)
    assert (t["files"], t["planted_leaks"], t["detected"], t["fixed_by_suggestion"], t["leaks_removed"]) == \
        (27, 23, 22, 11, 11)
    readme = (AUDITOR / "eval" / "agent_fix" / "README.md").read_text(encoding="utf-8")
    assert "| **total** | 27 | 23 | 22 | 11 | 11 | 0 | 0 | 0 | 0 |" in readme


def test_agent_fix_eval_preserves_bom_crlf_and_encoding(tmp_path):
    sys.path.insert(0, str(AUDITOR / "eval" / "agent_fix"))
    try:
        import run_eval
    finally:
        sys.path.pop(0)
    import shutil

    work = tmp_path / "encodings"
    shutil.copytree(AUDITOR / "eval" / "agent_fix" / "repos" / "encodings", work)
    applied, _ = run_eval.run_agent(work)
    assert applied == 2
    bom = (work / "bom_windows.py").read_bytes()
    assert bom.startswith(codecs.BOM_UTF8) and b"\r\n" in bom and b"\n" not in bom.replace(b"\r\n", b"")
    assert b"center=False" in bom
    lat = (work / "latin1_declared.py").read_bytes()
    assert "Bénéfice".encode("latin-1") in lat and b".ffill()" in lat
    assert (work / "momentum.ipynb").read_bytes() == \
        (AUDITOR / "eval" / "agent_fix" / "repos" / "encodings" / "momentum.ipynb").read_bytes()


def test_score_mode_counts_clean_line_changes(tmp_path):
    sys.path.insert(0, str(AUDITOR / "eval" / "agent_fix"))
    try:
        import run_eval
    finally:
        sys.path.pop(0)
    import shutil

    edited = tmp_path / "edited"
    shutil.copytree(AUDITOR / "eval" / "agent_fix" / "repos", edited)
    p = edited / "near_misses" / "fills.py"
    p.write_text(p.read_text(encoding="utf-8").replace("ffill(limit=5)", "ffill(limit=6)"), encoding="utf-8")
    p = edited / "momentum_pandas" / "features.py"
    p.write_text(p.read_text(encoding="utf-8").replace('["raw"].shift(-1)', '["raw"].shift(1)'), encoding="utf-8")
    res = run_eval.evaluate(edited)
    t = res["total"]
    assert t["clean_lines_changed"] == 1 and t["semantic_changes"] == 0 and t["leaks_removed"] == 1
    assert t["new_findings"] == 0
    assert run_eval.main(["--score-dir", str(edited)]) == 1


def test_score_mode_fails_on_a_meaning_change(tmp_path):
    """The eval must fail if a fix changes semantics in the known ways (here: mirroring a lead divided by close)."""
    sys.path.insert(0, str(AUDITOR / "eval" / "agent_fix"))
    try:
        import run_eval
    finally:
        sys.path.pop(0)
    import shutil

    edited = tmp_path / "edited"
    shutil.copytree(AUDITOR / "eval" / "agent_fix" / "repos", edited)
    p = edited / "meaning_preservation" / "shift_ratio.py"
    p.write_text(p.read_text(encoding="utf-8").replace("shift(-1)", "shift(1)"), encoding="utf-8")
    p = edited / "meaning_preservation" / "centered_label.py"
    p.write_text(p.read_text(encoding="utf-8").replace("center=True", "center=False"), encoding="utf-8")
    t = run_eval.evaluate(edited)["total"]
    assert t["semantic_changes"] == 2
    assert run_eval.main(["--score-dir", str(edited)]) == 1
