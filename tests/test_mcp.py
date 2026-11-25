"""MCP smoke tests through the SDK's own client: in-process, and over stdio via `python -m nba_auditor.mcp`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
import numpy as np
import pandas as pd
from mcp import Client, StdioServerParameters

from nba_auditor import DISCLAIMER
from nba_auditor.mcp import server

LEAKY = (Path(__file__).parent / "corpus" / "leaky" / "l22_ai_generated_sma.py").read_text(encoding="utf-8")


def _payload(result) -> dict:
    assert not result.is_error, result
    if result.structured_content:
        return result.structured_content
    return json.loads(result.content[0].text)


def _noise_csv(n=600, seed=0) -> str:
    r = pd.Series(np.random.default_rng(seed).normal(0, 0.01, n), index=pd.bdate_range("2020-01-01", periods=n))
    return r.rename("return").rename_axis("date").reset_index().to_csv(index=False)


async def _exercise(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"check_code", "check_path", "audit_backtest", "audit_returns", "audit_trade_log", "explain"}
    for t in tools.values():
        assert t.description and t.input_schema["type"] == "object"
        assert t.annotations is not None and t.annotations.read_only_hint is True
    assert "n_trials" in tools["audit_returns"].input_schema["properties"]
    assert tools["check_code"].input_schema["required"] == ["source"]

    out = _payload(await client.call_tool("check_code", {"source": LEAKY}))
    rules = {f["rule_id"] for f in out["report"]["findings"]}
    assert "LA07" in rules
    assert out["markdown"].rstrip().endswith(DISCLAIMER)
    from nba_auditor.catalog import SCORE_CAVEAT
    assert out["report"]["audit_score_note"] == SCORE_CAVEAT and SCORE_CAVEAT in out["markdown"]

    out = _payload(await client.call_tool("audit_returns", {"returns": _noise_csv(), "n_trials": 50, "turnover": 0.2}))
    assert out["report"]["verdict"] == "Major red flags"
    assert out["report"]["disclaimer"] == DISCLAIMER
    assert out["report"]["audit_score_note"] == SCORE_CAVEAT and SCORE_CAVEAT in out["markdown"]

    trades = ("date,ticker,side,qty,price\n2024-01-02,ABC,buy,10,100\n" +
              "".join(f"2024-{m:02d}-15,ABC,{'sell' if m % 2 else 'buy'},10,{100 + m}\n" for m in range(2, 12)))
    out = _payload(await client.call_tool("audit_trade_log", {"trades": trades}))
    assert "error" in out or out["report"]["tool"] == "audit_trade_log"

    out = _payload(await client.call_tool("explain", {"check_id": "RA03"}))
    assert "Deflated Sharpe" in out["markdown"] and out["check"]["references"]
    out = _payload(await client.call_tool("explain", {"check_id": ""}))
    assert len(out["check"]["checks"]) >= 20
    out = _payload(await client.call_tool("explain", {"check_id": "NOPE"}))
    assert "Unknown check id" in out["error"]

    out = _payload(await client.call_tool("audit_returns", {"returns": "[0.01, 0.02]"}))
    assert "at least 30" in out["error"]


def test_mcp_in_process():
    async def main():
        async with Client(server) as client:
            await _exercise(client)

    anyio.run(main)


def test_mcp_stdio_subprocess():
    params = StdioServerParameters(command=sys.executable, args=["-m", "nba_auditor.mcp"],
                                   env={"PYTHONPATH": str(Path(__file__).resolve().parents[1])})  # this source tree

    async def main():
        async with Client(params) as client:
            tools = (await client.list_tools()).tools
            assert {t.name for t in tools} == {"check_code", "check_path", "audit_backtest", "audit_returns", "audit_trade_log", "explain"}
            out = _payload(await client.call_tool("check_code", {"source": LEAKY}))
            assert out["report"]["counts"]["high"] >= 1

    anyio.run(main)
