"""Docs must match shipped behavior: every code block in the quickstart and the examples is tested.

Each fenced block must be preceded by a marker comment, `<!-- test: KIND ... -->`:

- `run`            python blocks are executed; bash blocks may only call `python ...` or `nba-auditor ...`,
                   and each line is executed from the Markdown file's folder.
- `mcp-config`     a client config (`mcpServers` or VS Code's `servers`); every server in it is launched over
                   stdio with the same args, and must list the four tools and answer `explain`.
- `claude-mcp-add` a `claude mcp add NAME -- COMMAND ARGS...` line; the command after `--` is launched the same way.
- `config-only`    not run (network installs, client UI). Must give a reason in parentheses.

Paths in configs are user placeholders, so `.../nba-auditor-mcp[.exe]` is mapped to the installed console script
and `.../python[.exe]` to this interpreter. Anything else fails the test.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import anyio
import pytest
from mcp import Client, StdioServerParameters

AUDITOR = Path(__file__).resolve().parents[1]
DOC_FILES = [AUDITOR / "docs" / "quickstart.md", AUDITOR / "docs" / "walkthrough-draft-to-audit.md", *sorted((AUDITOR / "examples").glob("*/README.md"))]
EXAMPLE = AUDITOR / "examples" / "leaky_to_clean"

FENCE = re.compile(r"^```(\w*)\s*$")
MARKER = re.compile(r"^<!--\s*test:\s*([\w-]+)(.*?)-->\s*$")
KINDS = {"run", "mcp-config", "claude-mcp-add", "config-only"}
TOOLS = {"check_code", "check_path", "audit_backtest", "audit_returns", "audit_trade_log", "explain"}
SERVER_NAMES = {"nba-auditor", "backtest-auditor"}  # backtest-auditor: the `uvx` configs (package on PyPI)


def _blocks(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out, i = [], 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        start, lang = i, m.group(1)
        j = i + 1
        while not lines[j].startswith("```"):
            j += 1
        prev = next((lines[k] for k in range(start - 1, -1, -1) if lines[k].strip()), "")
        mk = MARKER.match(prev.strip())
        out.append({"file": path, "line": start + 1, "lang": lang, "code": "\n".join(lines[start + 1:j]),
                    "kind": mk.group(1) if mk else None, "note": mk.group(2).strip() if mk else ""})
        i = j + 1
    return out


BLOCKS = [b for f in DOC_FILES for b in _blocks(f)]


def _id(b: dict) -> str:
    return f"{b['file'].relative_to(AUDITOR).as_posix()}:{b['line']}:{b['kind']}"


def _console_script(name: str) -> str:
    exe = Path(sys.executable).parent / (name + (".exe" if os.name == "nt" else ""))
    found = str(exe) if exe.exists() else shutil.which(name)
    assert found, f"console script {name} is not installed next to {sys.executable}"
    return found


def backtest_auditor_entry_point() -> tuple[str, list[str]]:
    """What `uvx backtest-auditor` runs: the package's `backtest-auditor` console script with no arguments.

    Uses the installed script if there is one; otherwise runs the target named in pyproject.toml directly, so the
    test still checks the entry point that will ship (an editable install from before the rename lacks the script)."""
    exe = Path(sys.executable).parent / ("backtest-auditor" + (".exe" if os.name == "nt" else ""))
    if exe.exists():
        return str(exe), []
    import tomllib

    target = tomllib.loads((AUDITOR / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]["backtest-auditor"]
    mod, func = target.split(":")
    return sys.executable, ["-c", f"import sys; sys.argv = ['backtest-auditor']; from {mod} import {func}; "
                                  f"sys.exit({func}())"]


def _resolve(command: str, args: list[str]) -> tuple[str, list[str]]:
    """Map a documented placeholder command to the one installed here, checking its shape."""
    if command == "uvx":  # the published PyPI package
        assert args == ["backtest-auditor"], args
        return backtest_auditor_entry_point()
    assert re.match(r"^(/|[A-Za-z]:[/\\])", command), f"client configs need an absolute path, got {command!r}"
    base = re.split(r"[/\\]", command)[-1].removesuffix(".exe")
    if base == "nba-auditor-mcp":
        assert args == [], args
        return _console_script("nba-auditor-mcp"), []
    if base in {"python", "python3", "python3.12"}:
        assert args == ["-m", "nba_auditor.mcp"], args
        return sys.executable, args
    raise AssertionError(f"undocumented server command {command!r}")


def source_pythonpath() -> str:
    """PYTHONPATH that puts this source tree first, so subprocesses import the code under test, not whatever copy
    of nba_auditor is installed (e.g. another checkout's editable install)."""
    return os.pathsep.join([str(AUDITOR)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])


def source_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": source_pythonpath()}


def stdio_params(command: str, args: list[str]) -> StdioServerParameters:
    """Server launch parameters. The SDK passes only a whitelisted environment to the subprocess, so set
    PYTHONPATH explicitly: the server under test is always this source tree."""
    return StdioServerParameters(command=command, args=args, env={"PYTHONPATH": source_pythonpath()})


def _exercise_server(command: str, args: list[str]) -> None:
    params = stdio_params(command, args)

    async def main():
        async with Client(params) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert tools == TOOLS
            res = await client.call_tool("explain", {"check_id": "LA07"})
            assert not res.is_error
            payload = res.structured_content or json.loads(res.content[0].text)
            assert payload["markdown"].startswith("# LA07: Signal and fill on the same bar")

    anyio.run(main)


def test_every_block_has_a_known_marker():
    assert BLOCKS, "no code blocks found"
    bad = [_id(b) for b in BLOCKS if b["kind"] not in KINDS]
    assert not bad, f"code blocks without a valid <!-- test: KIND --> marker: {bad}"
    no_reason = [_id(b) for b in BLOCKS if b["kind"] == "config-only" and not b["note"].startswith("(")]
    assert not no_reason, f"config-only blocks need a reason in parentheses: {no_reason}"


def test_uvx_blocks_run_the_published_package():
    """0.6.0 is on PyPI: every uvx snippet runs exactly `backtest-auditor` under a uvx heading, and no doc still says
    the package is unreleased or installs from the private repository."""
    uvx = [b for b in BLOCKS if "uvx" in b["code"]]
    assert uvx, "the quickstart should show the uvx setup"
    for b in uvx:
        assert re.search(r'uvx backtest-auditor\b|"args": \["backtest-auditor"\]', b["code"]), _id(b)
        lines = b["file"].read_text(encoding="utf-8").splitlines()[:b["line"]]
        heading = next(ln for ln in reversed(lines) if ln.startswith("## "))
        assert "uvx" in heading.lower(), (_id(b), heading)
    for f in [*DOC_FILES, AUDITOR / "README.md"]:
        text = f.read_text(encoding="utf-8").lower()
        for stale in ("available after release", "not on pypi", "isn't published yet",
                      "git+https://"):
            assert stale not in text, (f.name, stale)


@pytest.mark.parametrize("block", [b for b in BLOCKS if b["kind"] == "run"], ids=_id)
def test_run_block(block, tmp_path):
    if block["lang"] == "python":
        proc = subprocess.run([sys.executable, "-c", block["code"]], cwd=tmp_path, capture_output=True, text=True,
                              timeout=300, env=source_env())
        assert proc.returncode == 0, proc.stderr
        return
    assert block["lang"] == "bash", block["lang"]
    folder = block["file"].parent
    if folder == EXAMPLE:  # don't overwrite the committed reports: run a copy
        folder = tmp_path / EXAMPLE.name
        shutil.copytree(EXAMPLE, folder, ignore=shutil.ignore_patterns("output", "__pycache__"))
    for line in block["code"].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        argv = shlex.split(line)
        if argv[0] == "python":
            argv = [sys.executable, *argv[1:]]
        elif argv[0] == "nba-auditor":
            argv = [_console_script("nba-auditor"), *argv[1:]]
        else:
            raise AssertionError(f"run blocks may only call python or nba-auditor: {line!r}")
        proc = subprocess.run(argv, cwd=folder, capture_output=True, text=True, timeout=300, env=source_env())
        assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("block", [b for b in BLOCKS if b["kind"] == "mcp-config"], ids=_id)
def test_mcp_config_block(block):
    cfg = json.loads(block["code"])
    servers = cfg.get("mcpServers") or cfg.get("servers")
    assert servers and len(servers) == 1 and set(servers) <= SERVER_NAMES, servers
    spec = next(iter(servers.values()))
    assert spec.get("type", "stdio") == "stdio"
    assert not spec.get("env"), "the server needs no environment variables"
    _exercise_server(*_resolve(spec["command"], spec.get("args", [])))


@pytest.mark.parametrize("block", [b for b in BLOCKS if b["kind"] == "claude-mcp-add"], ids=_id)
def test_claude_mcp_add_block(block):
    argv = shlex.split(block["code"].strip(), posix=True)
    assert argv[:3] == ["claude", "mcp", "add"], argv
    sep = argv.index("--")
    assert argv[sep - 1] in SERVER_NAMES, argv
    _exercise_server(*_resolve(argv[sep + 1], argv[sep + 2:]))


# -- the walkthrough's quoted results ----------------------------------------------------------------

def test_walkthrough_results_match_readme(tmp_path, capsys):
    folder = tmp_path / EXAMPLE.name
    shutil.copytree(EXAMPLE, folder, ignore=shutil.ignore_patterns("output", "__pycache__"))
    proc = subprocess.run([sys.executable, "walkthrough.py"], cwd=folder, capture_output=True, text=True, timeout=300,
                          env=source_env())
    assert proc.returncode == 0, proc.stderr
    assert "Sharpe=6.77" in proc.stdout  # the leaky sweep's gross Sharpe, quoted in the README

    fresh = {p.name: p.read_text(encoding="utf-8") for p in (folder / "output").glob("*.md")}
    committed = {p.name: p.read_text(encoding="utf-8") for p in (EXAMPLE / "output").glob("*.md")}
    assert set(fresh) == set(committed) and len(fresh) == 4
    for name in fresh:  # structure must match exactly; numbers are checked below where the README quotes them
        strip = lambda t: [ln for ln in t.splitlines() if not re.search(r"\d", ln)]  # noqa: E731
        assert strip(fresh[name]) == strip(committed[name]), f"output/{name} is stale: re-run walkthrough.py"

    leaky_code, leaky_ret = fresh["1_leaky_check_code.md"], fresh["2_leaky_audit_returns.md"]
    fixed_code, fixed_ret = fresh["3_fixed_check_code.md"], fresh["4_fixed_audit_returns.md"]
    assert "**Audit score: 52/100**" in leaky_code
    for s in ("[HIGH] LA07 Signal and fill on the same bar (line 27)", "[MEDIUM] LA11", "(line 34)",
              "[POSSIBLE] CO01 No transaction costs (line 27)", "pass n_trials>=15",
              "[MEDIUM] LA09 Global normalization over all rows (line 25)"):  # issue #3: z-score inside a function
        assert s in leaky_code, s
    assert "## Verdict: Major red flags" in leaky_ret and "annualized Sharpe of 6.2" in leaky_ret
    assert "**Audit score: 100/100** (0 high, 0 medium, 0 possible, 0 info)" in fixed_code
    assert "## Verdict: Major red flags" in fixed_ret
    for s in ("Deflated Sharpe Ratio is 0.08", "about 0.89, and this one (0.17)", "t-stat is 0.34",
              "alpha 2.0% a year, t 0.37, beta 0.03", "Positive in only 2 of 3 sub-periods"):
        assert s in fixed_ret, s
    readme = (EXAMPLE / "README.md").read_text(encoding="utf-8")
    for s in ("**6.77**", "Sharpe of 6.2", "Deflated Sharpe Ratio is 0.08", "t-stat is 0.34", "Audit score 52/100"):
        assert s in readme, s
    for report in fresh.values():
        assert "not investment advice" in report
