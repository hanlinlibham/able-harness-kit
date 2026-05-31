"""LoopGuardMiddleware — detect and intervene on repeated no-progress tool calls.

Agents that commit to a plan can get stuck calling the same tool with the same
arguments, producing no new information, until they exhaust their budget. This
middleware fingerprints each tool call (name + normalized args), counts
consecutive repeats, and — past a threshold — emits a graded ``LoopSignal`` and
applies an action:

    observe(tool_call) -> LoopSignal -> Action(WARN | STOP)

It is intentionally thin: a fingerprint plus a consecutive-repeat counter, not a
full behavioral-scoring system. The intervention is the smallest one that fits
(log a warning, or short-circuit with a directive), and it is reversible — the
counter resets the moment a different call is seen.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger("able_harness_kit.loop_guard")


class LoopAction(str, Enum):
    NONE = "none"
    WARN = "warn"
    STOP = "stop"


@dataclass
class LoopSignal:
    fingerprint: str
    tool_name: str
    count: int
    action: LoopAction


def fingerprint_tool_call(name: str, args: Any) -> str:
    """Stable, argument-sensitive fingerprint for a tool call."""
    try:
        norm = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        norm = str(args)
    return hashlib.sha1(f"{name}\x00{norm}".encode("utf-8")).hexdigest()[:16]


class LoopTracker:
    """Consecutive-repeat tracker. One instance per agent / per turn.

    ``observe`` returns a :class:`LoopSignal` whose ``action`` escalates from
    ``NONE`` to ``WARN`` (at ``warn_at`` repeats) to ``STOP`` (at ``stop_at``).
    A different call resets the streak.
    """

    def __init__(self, warn_at: int = 3, stop_at: int = 5) -> None:
        if warn_at < 1 or stop_at < warn_at:
            raise ValueError("require 1 <= warn_at <= stop_at")
        self.warn_at = warn_at
        self.stop_at = stop_at
        self._last_fp: str | None = None
        self._count = 0

    def observe(self, name: str, args: Any) -> LoopSignal:
        fp = fingerprint_tool_call(name, args)
        if fp == self._last_fp:
            self._count += 1
        else:
            self._last_fp = fp
            self._count = 1
        if self._count >= self.stop_at:
            action = LoopAction.STOP
        elif self._count >= self.warn_at:
            action = LoopAction.WARN
        else:
            action = LoopAction.NONE
        return LoopSignal(fingerprint=fp, tool_name=name, count=self._count, action=action)


class LoopGuardMiddleware(AgentMiddleware):
    """Short-circuit no-progress tool loops with a graded, reversible intervention."""

    def __init__(
        self,
        *,
        warn_at: int = 3,
        stop_at: int = 5,
        on_signal: Callable[[LoopSignal], None] | None = None,
    ) -> None:
        super().__init__()
        self.on_signal = on_signal
        self._tracker = LoopTracker(warn_at, stop_at)

    def _name_args(self, request: ToolCallRequest) -> tuple[str, Any]:
        tc = getattr(request, "tool_call", None) or {}
        if isinstance(tc, dict):
            return tc.get("name") or "", tc.get("args")
        return "", None

    def _check(self, request: ToolCallRequest) -> LoopSignal | None:
        name, args = self._name_args(request)
        if not name:
            return None
        signal = self._tracker.observe(name, args)
        if signal.action is not LoopAction.NONE:
            logger.warning(
                "LoopGuard: %s repeated %d× -> %s",
                name,
                signal.count,
                signal.action.value,
            )
            if self.on_signal is not None:
                self.on_signal(signal)
        return signal

    def _stop_message(self, signal: LoopSignal, request: ToolCallRequest) -> ToolMessage:
        tc = getattr(request, "tool_call", None) or {}
        tcid = tc.get("id") if isinstance(tc, dict) else None
        return ToolMessage(
            content=(
                f"Loop guard: '{signal.tool_name}' has been called {signal.count}× "
                f"with identical arguments and is not making progress. Stop repeating "
                f"it — change approach, try a different tool, or report what is blocking you."
            ),
            name=signal.tool_name,
            tool_call_id=tcid or "loopguard",
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        signal = self._check(request)
        if signal is not None and signal.action is LoopAction.STOP:
            return self._stop_message(signal, request)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        signal = self._check(request)
        if signal is not None and signal.action is LoopAction.STOP:
            return self._stop_message(signal, request)
        return await handler(request)


__all__ = [
    "LoopGuardMiddleware",
    "LoopTracker",
    "LoopSignal",
    "LoopAction",
    "fingerprint_tool_call",
]
