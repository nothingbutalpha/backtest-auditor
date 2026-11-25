"""The version-drift guard (scripts/check_published.py): nba_auditor/ may not differ from a published release while
pyproject.toml still names that release. auditor-ci also runs the script as its own step."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

AUDITOR = Path(__file__).resolve().parents[1]
SCRIPT = AUDITOR / "scripts" / "check_published.py"

_spec = importlib.util.spec_from_file_location("check_published", SCRIPT)
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def test_the_guard_passes_on_this_checkout():
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=AUDITOR)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_published_0_7_0_manifest_is_recorded_with_both_pypi_files():
    m = json.loads((AUDITOR / "published" / "backtest-auditor-0.7.0.json").read_text(encoding="utf-8"))
    assert m["package"] == "backtest-auditor" and m["version"] == "0.7.0"
    assert m["dists"] == {
        "backtest_auditor-0.7.0-py3-none-any.whl": "79becd3d8fce3f5dc99cf5a6bb447491f956044879123ec8fb29657512855225",
        "backtest_auditor-0.7.0.tar.gz": "d5c9ad4dd356653d5d91e206ad99a111f2af430abb109a42da4c310403b13d99",
    }
    assert "__init__.py" in m["files"] and "lint.py" in m["files"] and len(m["files"]) >= 20


def test_drift_reports_changed_added_and_removed_files():
    published = {"a.py": "1", "b.py": "2", "c.py": "3"}
    assert cp.drift(published, dict(published)) == []
    assert cp.drift(published, {"a.py": "1", "b.py": "X", "d.py": "4"}) == ["changed: b.py", "added: d.py", "removed: c.py"]


def test_the_guard_fails_when_the_version_names_a_release_and_the_code_differs(monkeypatch, capsys):
    published = cp.manifests()["0.7.0"]["files"]
    monkeypatch.setattr(cp, "repo_version", lambda: "0.7.0")
    monkeypatch.setattr(cp, "repo_files", lambda: dict(published))
    assert cp.check() == 0
    monkeypatch.setattr(cp, "repo_files", lambda: {**published, "lint.py": "0" * 64})
    assert cp.check() == 1
    assert "changed: lint.py" in capsys.readouterr().out


def test_an_unpublished_version_has_nothing_to_match(monkeypatch):
    monkeypatch.setattr(cp, "repo_version", lambda: "0.7.1.dev0")
    monkeypatch.setattr(cp, "repo_files", lambda: {"anything.py": "0"})
    assert cp.check() == 0


# --------------------------------------------------------------------------- PEP 440 comparison (security review L2)

FAKE = {"0.7.0": {"version": "0.7.0", "files": {"lint.py": "a" * 64}}}


def test_a_version_equal_under_pep440_to_a_release_must_hold_its_files(monkeypatch, capsys):
    monkeypatch.setattr(cp, "manifests", lambda: FAKE)
    monkeypatch.setattr(cp, "repo_files", lambda: {"lint.py": "b" * 64})
    for spelled in ("0.7.0", "0.7.0.0", "0.7", "v0.7.0"):
        monkeypatch.setattr(cp, "repo_version", lambda s=spelled: s)
        assert cp.check() == 1, spelled
        assert "changed: lint.py" in capsys.readouterr().out
    monkeypatch.setattr(cp, "repo_files", lambda: {"lint.py": "a" * 64})
    monkeypatch.setattr(cp, "repo_version", lambda: "0.7.0.0")
    assert cp.check() == 0


def test_post_and_dev_versions_are_other_versions_but_never_older_ones(monkeypatch):
    monkeypatch.setattr(cp, "manifests", lambda: FAKE)
    monkeypatch.setattr(cp, "repo_files", lambda: {"lint.py": "b" * 64})
    for v in ("0.7.0.post0", "0.7.1.dev0", "0.7.1"):          # newer than 0.7.0 and unrecorded: nothing to match
        monkeypatch.setattr(cp, "repo_version", lambda s=v: s)
        assert cp.check() == 0, v
    for v in ("0.6.9", "0.7.0.dev3", "not a version"):          # older than the latest release, or invalid
        monkeypatch.setattr(cp, "repo_version", lambda s=v: s)
        assert cp.check() == 1, v


def test_post0_is_refused_when_a_post0_release_is_recorded_and_the_code_differs(monkeypatch):
    monkeypatch.setattr(cp, "manifests", lambda: {"0.7.0.post0": {"version": "0.7.0.post0", "files": {"x.py": "1"}}})
    monkeypatch.setattr(cp, "repo_files", lambda: {"x.py": "2"})
    for v in ("0.7.0.post0", "0.7.0post0", "0.7.0-post0", "0.7.0.post"):
        monkeypatch.setattr(cp, "repo_version", lambda s=v: s)
        assert cp.check() == 1, v


# --------------------------------------------------------------------------- publish refusals (security review L3)


def test_release_check_refuses_dev_local_recorded_and_older_versions():
    assert cp.release_problems("0.7.1", FAKE) == []
    assert cp.release_problems("0.8.0", FAKE) == []
    for bad in ("0.7.1.dev0", "0.7.1+local", "0.7.1.dev0+x", "0.7.0", "0.7.0.0", "0.7", "0.6.9", "seven"):
        assert cp.release_problems(bad, FAKE), bad


def test_release_check_cli_exit_codes():
    ok = subprocess.run([sys.executable, str(SCRIPT), "release-check", "99.0.0"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout
    bad = subprocess.run([sys.executable, str(SCRIPT), "release-check", "0.7.0"], capture_output=True, text=True)
    assert bad.returncode == 1 and "already published" in bad.stdout


# --------------------------------------------------------------------------- listing states (security review M1)


def test_listing_states():
    rec = {"0.7.0": {}, "0.6.0": {}}
    assert cp.listing_ok("0.7.0", "0.7.0", rec) == []          # released and recorded
    assert cp.listing_ok("0.7.0", "0.7.1.dev0", rec) == []     # development after the release
    assert cp.listing_ok("0.7.0", "0.7.1", rec) == []          # next release, published or not, not yet recorded
    assert cp.listing_ok("0.7.1", "0.7.1", rec)                # the listing may not name an unrecorded version
    assert cp.listing_ok("0.6.0", "0.7.1", rec)                # nor an older release than the latest
    assert cp.listing_ok("0.7.0", "0.6.5", rec)                # the repo may not go back
    assert cp.listing_ok("0.7.0", "0.7.0", {})                 # no manifest at all


# --------------------------------------------------------------------------- installed files (security review L1)


def _fake_install(tmp_path, monkeypatch, files: dict[str, bytes], version: str = "0.7.0"):
    import importlib.machinery
    import sysconfig

    pkg = tmp_path / "nba_auditor"
    for name, data in files.items():
        (pkg / name).parent.mkdir(parents=True, exist_ok=True)
        (pkg / name).write_bytes(data)
    (pkg / "__pycache__").mkdir(exist_ok=True)
    (pkg / "__pycache__" / "lint.cpython-312.pyc").write_bytes(b"ignored")
    spec = importlib.machinery.ModuleSpec("nba_auditor", None, is_package=True)
    spec.submodule_search_locations = [str(pkg)]
    monkeypatch.setattr(cp.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})
    monkeypatch.setattr(cp.importlib.metadata, "version", lambda name: version)
    monkeypatch.setattr(cp, "manifests", lambda: {"0.7.0": {"version": "0.7.0", "files": {
        "lint.py": cp.sha256(b"published lint"), "mcp/__init__.py": cp.sha256(b"mcp")}}})


def test_installed_files_must_equal_the_manifest(tmp_path, monkeypatch):
    _fake_install(tmp_path, monkeypatch, {"lint.py": b"published lint", "mcp/__init__.py": b"mcp"})
    assert cp.installed() == 0


def test_installed_files_that_differ_fail(tmp_path, monkeypatch, capsys):
    _fake_install(tmp_path, monkeypatch, {"lint.py": b"patched lint", "mcp/__init__.py": b"mcp", "extra.py": b""})
    assert cp.installed() == 1
    out = capsys.readouterr().out
    assert "changed: lint.py" in out and "added: extra.py" in out


def test_an_installed_version_without_a_manifest_fails(tmp_path, monkeypatch):
    _fake_install(tmp_path, monkeypatch, {"lint.py": b"published lint", "mcp/__init__.py": b"mcp"}, version="0.7.1")
    assert cp.installed() == 1
