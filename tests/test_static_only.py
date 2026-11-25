"""The Auditor reads code, it never runs it.

Fails if anything in nba_auditor/ gains a way to execute code, spawn processes or open network connections.
"""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "nba_auditor"

FORBIDDEN_MODULES = {"subprocess", "multiprocessing", "socket", "ctypes", "pickle", "marshal", "runpy", "code",
                     "codeop", "pty", "importlib", "urllib", "requests", "httpx", "http", "ssl", "ftplib",
                     "smtplib", "telnetlib", "xmlrpc", "aiohttp", "websockets", "websocket", "urllib3", "asyncio"}
FORBIDDEN_CALLS = {"exec", "eval", "compile", "__import__", "execfile", "breakpoint"}
FORBIDDEN_ATTRS = {"system", "popen", "spawnl", "spawnv", "execv", "execve", "execvp", "fork", "startfile"}


def test_no_execution_or_network_paths():
    bad = []
    for path in sorted(PKG.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mods = [node.module or ""]
            bad += [f"{path.name}:{node.lineno} imports {m}" for m in mods if m.split(".")[0] in FORBIDDEN_MODULES]
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in FORBIDDEN_CALLS:
                    bad.append(f"{path.name}:{node.lineno} calls {fn.id}()")
                if isinstance(fn, ast.Attribute) and fn.attr in FORBIDDEN_ATTRS:
                    bad.append(f"{path.name}:{node.lineno} calls .{fn.attr}()")
    assert bad == []


def test_no_network_at_runtime(monkeypatch):
    """The privacy claim "we never receive your code (local use)" depends on the package never opening a socket."""
    import socket

    def refuse(*a, **k):
        raise AssertionError("nba_auditor tried to open a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    from nba_auditor.lint import check_code

    leaky = PKG.parent / "examples" / "leaky_to_clean"
    srcs = [p.read_text(encoding="utf-8") for p in sorted(leaky.rglob("*.py"))]
    assert srcs, "example sources missing"
    for src in srcs:
        check_code(src)
