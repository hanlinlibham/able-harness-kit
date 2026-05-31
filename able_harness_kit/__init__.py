"""able-harness-kit — thin, backend-neutral agent-harness middlewares.

Production-distilled patterns as framework-neutral middlewares that compose on
top of LangChain / deepagents agents:

- ``ProgressAwareLoopGuardMiddleware`` — stop tool loops that repeat *and* return
  no new information (compares the result, not just the arguments, so polling /
  pagination passes through)
- ``LoopGuardMiddleware``        — cheaper loop guard on repeated arguments alone
- ``BinaryReadGuardMiddleware``  — fail loud on binary ``read_file`` results instead of
  feeding the model a base64 block it may never actually receive
- ``ToolResultBudgetMiddleware`` — keep oversized tool results out of the context window

The output-delta primitives behind the progress-aware guard
(``build_observations`` / ``ToolCallObservation`` / ``new_information_delta``) are
exported for standalone use. See the README for rationale and the experiment
behind them.
"""
from .binary_read_guard import BinaryReadGuardMiddleware, DEFAULT_DIRECTIVE, is_text_like
from .loop_guard import (
    LoopAction,
    LoopGuardMiddleware,
    LoopSignal,
    LoopTracker,
    ProgressAwareLoopGuardMiddleware,
    ProgressSignal,
    fingerprint_tool_call,
)
from .tool_observation import (
    DeltaFn,
    ToolCallObservation,
    baseline_delta,
    build_observations,
    classify_error,
    fingerprint_args,
    fingerprint_content,
    make_set_membership_delta,
    observe_call,
)
from .tool_result_budget import (
    OffloadFn,
    OffloadRef,
    ToolResultBudgetMiddleware,
    apply_budget,
)

__all__ = [
    "ProgressAwareLoopGuardMiddleware",
    "ProgressSignal",
    "LoopGuardMiddleware",
    "LoopSignal",
    "LoopAction",
    "LoopTracker",
    "fingerprint_tool_call",
    "ToolCallObservation",
    "DeltaFn",
    "build_observations",
    "observe_call",
    "baseline_delta",
    "make_set_membership_delta",
    "fingerprint_args",
    "fingerprint_content",
    "classify_error",
    "BinaryReadGuardMiddleware",
    "is_text_like",
    "DEFAULT_DIRECTIVE",
    "ToolResultBudgetMiddleware",
    "OffloadRef",
    "OffloadFn",
    "apply_budget",
]

__version__ = "0.2.0"
