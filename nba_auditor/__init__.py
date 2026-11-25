"""Backtest Auditor: a red-team lint for trading strategies (Nothing But Alpha).

Public API:
    check_code(source)          -> CodeAudit       static lookahead / bias lint
    audit_returns(returns, ...) -> ReturnsAudit    statistical audit of a return series
    audit_trade_log(trades,...) -> ReturnsAudit    trade log -> daily PnL/turnover -> audit_returns
    explain(check_id)           -> dict            what a check does, why it matters, references
    check_path(path, ...)       -> dict            lint .py/.ipynb files on disk (read-only), with re-checked fixes
    audit_backtest(...)         -> dict            code + returns/trades + benchmark -> one combined verdict
    suggest_fixes(source)       -> [SuggestedFix]  minimal edits, each re-checked by re-linting

Every report has `.to_dict()` (JSON) and `.to_markdown()`.
"""

__version__ = "0.7.0"

from .catalog import DISCLAIMER, explain, explain_markdown  # noqa: E402
from .lint import CodeAudit, Finding, check_code  # noqa: E402
from .parsing import InputError  # noqa: E402
from .returns_audit import ReturnsAudit, audit_returns  # noqa: E402
from .trade_log import audit_trade_log  # noqa: E402
from .fixes import SuggestedFix, apply_edits, suggest_fixes  # noqa: E402
from .path_audit import check_path  # noqa: E402
from .backtest_audit import audit_backtest  # noqa: E402

__all__ = ["DISCLAIMER", "CodeAudit", "Finding", "InputError", "ReturnsAudit", "SuggestedFix", "apply_edits",
           "audit_backtest", "audit_returns", "audit_trade_log", "check_code", "check_path", "explain",
           "explain_markdown", "suggest_fixes", "__version__"]
