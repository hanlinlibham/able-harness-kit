"""Loop guards — stop tool-call loops with a graded, reversible intervention.

Two guards, different signals:

- :class:`LoopGuardMiddleware` — cheap. Fires on repeated *arguments* alone
  (same tool, same args, N times in a row). No view of the result. Good when you
  only want to catch a model hammering an identical call and don't want to pay for
  output tracking.

- :class:`ProgressAwareLoopGuardMiddleware` — sharper. Runs the tool, compares the
  *result* to the previous same-tool result, and only intervenes when a call both
  repeats *and* returns no new information. Genuine progress — polling, pagination,
  stream tailing — repeats the arguments but changes the output, so it passes
  through where the cheap guard would false-positive. This is the recommended guard
  when a result is available; the output-delta machinery lives in
  ``tool_observation.py``.
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

from .tool_observation import DeltaFn, ToolCallObservation, observe_call

logger = logging.getLogger("able_harness_kit.loop_guard")


# ─────────────────────────────────────────────────────────────────
# Cheap guard: repeated arguments
# ─────────────────────────────────────────────────────────────────


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
    """Short-circuit identical-argument tool loops (no view of the result).

    Cheaper than :class:`ProgressAwareLoopGuardMiddleware` but blind to whether a
    repeat is actually making progress — prefer the progress-aware guard when the
    tool's result is available.
    """

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


# ─────────────────────────────────────────────────────────────────
# Sharp guard: repeated AND no new information (output-delta)
# ─────────────────────────────────────────────────────────────────


@dataclass
class ProgressSignal:
    """The result of observing one tool call against its same-tool predecessor."""

    tool_name: str
    repeat_count: int           # consecutive same (tool_name, args)
    new_information_delta: float  # 0.0 = identical output, 1.0 = fully new
    stalled: bool               # repeat_count >= stop_at AND delta < progress_floor


class ProgressAwareLoopGuardMiddleware(AgentMiddleware):
    """Stop loops that repeat a call *and* get no new information back.

    Runs the tool, compares the result to the previous same-tool result, and
    intervenes only when both conditions hold:

      ``same_args_repeat_count >= stop_at``  AND  ``new_information_delta < progress_floor``

    Genuine progress (polling a job, walking pagination, tailing a stream) repeats
    the arguments but changes the output, so its ``new_information_delta`` stays
    high and it passes through. The intervention replaces the stalled result with a
    model-facing directive (not a synthetic user message); the streak resets the
    moment a different call or a different result is seen.

    ``delta_overrides`` maps ``tool_name -> DeltaFn`` for tools where the default
    content-hash comparison is too lax (see
    :func:`able_harness_kit.tool_observation.make_set_membership_delta`).
    """

    def __init__(
        self,
        *,
        stop_at: int = 3,
        progress_floor: float = 0.1,
        delta_overrides: dict[str, DeltaFn] | None = None,
        on_signal: Callable[[ProgressSignal], None] | None = None,
    ) -> None:
        super().__init__()
        if stop_at < 1:
            raise ValueError("stop_at must be >= 1")
        if not 0.0 <= progress_floor <= 1.0:
            raise ValueError("progress_floor must be in [0, 1]")
        self.stop_at = stop_at
        self.progress_floor = progress_floor
        self._overrides = delta_overrides or {}
        self.on_signal = on_signal
        self._prior_obs: dict[str, ToolCallObservation] = {}
        self._prior_args: dict[str, dict] = {}

    def _tool(self, request: ToolCallRequest) -> tuple[str, Any, str | None]:
        tc = getattr(request, "tool_call", None) or {}
        if isinstance(tc, dict):
            return tc.get("name") or "", tc.get("args"), tc.get("id")
        return "", None, None

    def _record(self, name: str, args: Any, result: Any) -> ProgressSignal | None:
        if not name or not isinstance(result, ToolMessage):
            return None
        obs = observe_call(
            tool_name=name,
            args=args,
            content=result.content,
            status=getattr(result, "status", "success") or "success",
            prior_obs=self._prior_obs.get(name),
            prior_args=self._prior_args.get(name),
            delta_fn=self._overrides.get(name),
            tool_call_id=getattr(result, "tool_call_id", "") or "",
        )
        self._prior_obs[name] = obs
        self._prior_args[name] = args if isinstance(args, dict) else {}
        stalled = (
            obs["same_args_repeat_count"] >= self.stop_at
            and obs["new_information_delta"] < self.progress_floor
        )
        signal = ProgressSignal(
            tool_name=name,
            repeat_count=obs["same_args_repeat_count"],
            new_information_delta=obs["new_information_delta"],
            stalled=stalled,
        )
        if stalled:
            logger.warning(
                "ProgressAwareLoopGuard: %s repeated %d× with no new information (delta=%.2f)",
                name,
                signal.repeat_count,
                signal.new_information_delta,
            )
        if self.on_signal is not None:
            self.on_signal(signal)
        return signal

    def _stalled_message(self, signal: ProgressSignal, tcid: str | None) -> ToolMessage:
        return ToolMessage(
            content=(
                f"Progress guard: '{signal.tool_name}' has run {signal.repeat_count}× "
                f"with the same arguments and returned no new information. Stop repeating "
                f"it — change the arguments, try a different tool, or report what is blocking you."
            ),
            name=signal.tool_name,
            tool_call_id=tcid or "progressguard",
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        name, args, tcid = self._tool(request)
        result = handler(request)
        signal = self._record(name, args, result)
        if signal is not None and signal.stalled:
            return self._stalled_message(signal, tcid)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        name, args, tcid = self._tool(request)
        result = await handler(request)
        signal = self._record(name, args, result)
        if signal is not None and signal.stalled:
            return self._stalled_message(signal, tcid)
        return result


__all__ = [
    "LoopGuardMiddleware",
    "LoopTracker",
    "LoopSignal",
    "LoopAction",
    "fingerprint_tool_call",
    "ProgressAwareLoopGuardMiddleware",
    "ProgressSignal",
]
