"""Version-drift and release guards for the Backtest Auditor's published releases.

PyPI files are immutable, so once backtest-auditor X is published, every later change to nba_auditor/ needs a new
version number in pyproject.toml. Otherwise the repo and PyPI hold different code under one version (it happened once:
commit 8c1511b changed lint.py after 0.7.0 was published).

Each published release has a manifest in auditor/published/backtest-auditor-<version>.json: the sha256 of the wheel
and the sdist on PyPI, and the sha256 of every file under nba_auditor/ in them (the wheel and the sdist hold the same
files). Versions are compared under PEP 440 (packaging.version), so 0.7.0, 0.7.0.0 and 0.7 are one version.

Release order (also in .github/workflows/publish-auditor.yml):
  1. bump pyproject.toml and nba_auditor/__init__.py to the release version (no .dev, no +local);
  2. publish with the publish-auditor workflow (its release-check refuses a version that is already recorded);
  3. download the wheel and the sdist from PyPI, compare their sha256 with PyPI's, run `record WHEEL SDIST`;
  4. only then set listing/server.json (and listing/metadata.json) to the new version.
Between 2 and 4 the listing keeps naming the previous recorded release; it never names an unpublished version.

Run from auditor/ (standard library plus packaging):

    python scripts/check_published.py                  # check: exit 1 if pyproject's version equals a recorded
                                                        # release and nba_auditor/ differs from it, or if it is
                                                        # older than the latest recorded release
    python scripts/check_published.py release-check V  # before a publish: refuse .dev, +local, and recorded versions
    python scripts/check_published.py installed        # the installed backtest-auditor (site-packages) must hold
                                                        # exactly its recorded release's files
    python scripts/check_published.py record WHEEL SDIST
                                                        # after a publish: write the manifest from the PyPI files

__pycache__ directories are ignored.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

from packaging.version import InvalidVersion, Version

AUDITOR = Path(__file__).resolve().parents[1]
PACKAGE = "nba_auditor"
PUBLISHED = AUDITOR / "published"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repo_version() -> str:
    return tomllib.loads((AUDITOR / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def tree_files(root: Path) -> dict[str, str]:
    """sha256 of every file under ``root``, by its path relative to ``root`` (bytes as found)."""
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def repo_files() -> dict[str, str]:
    return tree_files(AUDITOR / PACKAGE)


def manifests() -> dict[str, dict]:
    out = {}
    for path in sorted(PUBLISHED.glob("backtest-auditor-*.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        if path.name != f"backtest-auditor-{m['version']}.json":
            raise SystemExit(f"{path.name}: names a different version than its content ({m['version']})")
        out[m["version"]] = m
    return out


def recorded(version: str, recorded_: dict[str, dict]) -> dict | None:
    """The manifest of the release that ``version`` equals under PEP 440 (0.7.0.0 == 0.7.0), or None."""
    v = Version(version)
    hits = [m for k, m in recorded_.items() if Version(k) == v]
    return hits[0] if hits else None


def latest(recorded_: dict[str, dict]) -> str | None:
    return max(recorded_, key=Version) if recorded_ else None


def listing_ok(server_version: str, repo_version_: str, recorded_: dict[str, dict]) -> list[str]:
    """Problems with the listing's version (empty: fine). The listing names the latest recorded release, never an
    unpublished version; the repo is that release or newer (a .dev version, or the next release before it is recorded)."""
    problems = []
    top = latest(recorded_)
    if top is None:
        return ["no published manifest in auditor/published/"]
    if server_version != top:
        problems.append(f"server.json names {server_version}, not the latest recorded release {top}")
    if Version(repo_version_) < Version(top):
        problems.append(f"the repo version {repo_version_} is older than the latest recorded release {top}")
    return problems


def drift(published: dict[str, str], current: dict[str, str]) -> list[str]:
    return ([f"changed: {k}" for k in sorted(set(published) & set(current)) if published[k] != current[k]]
            + [f"added: {k}" for k in sorted(set(current) - set(published))]
            + [f"removed: {k}" for k in sorted(set(published) - set(current))])


def check() -> int:
    version, rec = repo_version(), manifests()
    try:
        Version(version)
    except InvalidVersion:
        print(f"FAIL: auditor/pyproject.toml version {version!r} is not a PEP 440 version")
        return 1
    top = latest(rec)
    if top is not None and Version(version) < Version(top):
        print(f"FAIL: auditor/pyproject.toml says {version}, older than the latest published release {top}")
        return 1
    m = recorded(version, rec)
    if m is None:
        print(f"ok: auditor/pyproject.toml says {version}, which equals no published release "
              f"(published: {', '.join(rec) or 'none'}); nothing to match")
        return 0
    diffs = drift(m["files"], repo_files())
    if diffs:
        print(f"FAIL: auditor/{PACKAGE}/ differs from the published backtest-auditor {m['version']} "
              f"(auditor/published/backtest-auditor-{m['version']}.json), but auditor/pyproject.toml says {version}, "
              f"the same version under PEP 440.\nBump the version (a .devN of the next release until it is published) "
              f"in pyproject.toml and {PACKAGE}/__init__.py:\n  " + "\n  ".join(diffs))
        return 1
    print(f"ok: auditor/{PACKAGE}/ equals the published backtest-auditor {m['version']} ({len(m['files'])} files)")
    return 0


def release_problems(version: str, recorded_: dict[str, dict]) -> list[str]:
    """Why ``version`` may not be published (empty: it may)."""
    try:
        v = Version(version)
    except InvalidVersion:
        return [f"{version!r} is not a PEP 440 version"]
    problems = []
    if v.is_devrelease:
        problems.append(f"{version} is a development (.dev) version")
    if v.local is not None:
        problems.append(f"{version} has a local version label (+{v.local}); PyPI refuses those")
    m = recorded(version, recorded_)
    if m is not None:
        problems.append(f"{version} equals the already published {m['version']} (auditor/published/)")
    top = latest(recorded_)
    if top is not None and m is None and v < Version(top):
        problems.append(f"{version} is older than the latest published release {top}")
    return problems


def release_check(version: str) -> int:
    problems = release_problems(version, manifests())
    if problems:
        print("Refusing to publish:\n- " + "\n- ".join(problems))
        return 1
    print(f"ok: {version} may be published")
    return 0


def installed() -> int:
    """The installed backtest-auditor (not this checkout) must hold exactly its recorded release's files."""
    import sysconfig

    spec = importlib.util.find_spec(PACKAGE)     # locates the package without importing (running) it
    if spec is None or not spec.submodule_search_locations:
        print(f"FAIL: {PACKAGE} is not installed")
        return 1
    where = Path(list(spec.submodule_search_locations)[0]).resolve()
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    if where.parent != purelib:
        print(f"FAIL: {PACKAGE} imports from {where}, not from this interpreter's purelib {purelib}")
        return 1
    version = importlib.metadata.version("backtest-auditor")
    m = recorded(version, manifests())
    if m is None:
        print(f"FAIL: the installed backtest-auditor {version} has no published manifest")
        return 1
    diffs = drift(m["files"], tree_files(where))
    if diffs:
        print(f"FAIL: the installed backtest-auditor {version} ({where}) differs from its manifest:\n  " + "\n  ".join(diffs))
        return 1
    print(f"ok: the installed backtest-auditor {version} ({where}) holds exactly the {len(m['files'])} recorded files")
    return 0


def _wheel_files(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as z:
        return {n[len(PACKAGE) + 1:]: sha256(z.read(n)) for n in sorted(z.namelist())
                if n.startswith(PACKAGE + "/") and not n.endswith("/")}


def _sdist_files(path: Path) -> dict[str, str]:
    out = {}
    with tarfile.open(path) as t:
        for m in t.getmembers():
            parts = m.name.split("/", 1)
            if m.isfile() and len(parts) == 2 and parts[1].startswith(PACKAGE + "/"):
                out[parts[1][len(PACKAGE) + 1:]] = sha256(t.extractfile(m).read())
    return dict(sorted(out.items()))


def record(wheel: Path, sdist: Path) -> int:
    version = wheel.name.split("-")[1]
    if not sdist.name == f"backtest_auditor-{version}.tar.gz":
        raise SystemExit(f"{sdist.name} is not the sdist of {version}")
    files = _wheel_files(wheel)
    if files != _sdist_files(sdist):
        raise SystemExit("the wheel and the sdist hold different nba_auditor/ files")
    manifest = {
        "package": "backtest-auditor",
        "version": version,
        "note": "The published PyPI files of this version (immutable). scripts/check_published.py fails while "
                "pyproject.toml says this version and nba_auditor/ differs from these files.",
        "dists": {wheel.name: sha256(wheel.read_bytes()), sdist.name: sha256(sdist.read_bytes())},
        "files": files,
    }
    PUBLISHED.mkdir(exist_ok=True)
    out = PUBLISHED / f"backtest-auditor-{version}.json"
    out.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(f"wrote {out} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["record"] and len(args) == 3:
        sys.exit(record(Path(args[1]), Path(args[2])))
    if args[:1] == ["release-check"] and len(args) == 2:
        sys.exit(release_check(args[1]))
    if args == ["installed"]:
        sys.exit(installed())
    if not args:
        sys.exit(check())
    sys.exit(__doc__)
