"""able-harness-kit — thin, backend-neutral agent-harness middlewares.

Three production-tested patterns distilled into framework-neutral middlewares
that compose on top of LangChain / deepagents agents:

- ``LoopGuardMiddleware``        — detect and intervene on repeated no-progress tool calls
- ``BinaryReadGuardMiddleware``  — fail loud on binary ``read_file`` results instead of
  feeding the model a base64 block it may never actually receive
- ``ToolResultBudgetMiddleware`` — keep oversized tool results out of the context window

See the README for rationale and the experiment behind them.
"""
from .binary_read_guard import BinaryReadGuardMiddleware, DEFAULT_DIRECTIVE, is_text_like
from .loop_guard import (
    LoopAction,
    LoopGuardMiddleware,
    LoopSignal,
    LoopTracker,
    fingerprint_tool_call,
)
from .tool_result_budget import (
    OffloadFn,
    OffloadRef,
    ToolResultBudgetMiddleware,
    apply_budget,
)

__all__ = [
    "LoopGuardMiddleware",
    "LoopSignal",
    "LoopAction",
    "LoopTracker",
    "fingerprint_tool_call",
    "BinaryReadGuardMiddleware",
    "is_text_like",
    "DEFAULT_DIRECTIVE",
    "ToolResultBudgetMiddleware",
    "OffloadRef",
    "OffloadFn",
    "apply_budget",
]

__version__ = "0.1.0"
