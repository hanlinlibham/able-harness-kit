"""Progress-aware tool observation — the output-delta signal.

The cheap loop signal is "same tool, same args, N times in a row." It is also
wrong about half the time. A model polling a job, tailing a stream, or walking
pagination calls the same tool with the same (or barely different) arguments and
*should* — each call returns new information. A model stuck in a perseveration
loop calls the same tool and gets the *same result back*.

The distinction isn't in the arguments, it's in the output. This module records,
per tool call, both what went in (``args_hash``) and what came back
(``content_hash``), and derives a ``new_information_delta``: ``0.0`` when the
output is identical to the previous same-tool call, ``1.0`` when it's fully new.
Loop detection then keys off *repeats that produced no new information*, which
lets genuine progress (polling, pagination) through while still catching the spin.

Pure functions, zero IO. Distilled from a production controller. The per-tool
delta override (:func:`make_set_membership_delta`) comes from a real case where a
model rewrote the same list item with a one-character cosmetic edit — the content
hash differed every time, but no real information arrived.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Literal, TypedDict

from langchain_core.messages import AIMessage, ToolMessage

__all__ = [
    "ToolCallObservation",
    "DeltaFn",
    "DEFAULT_OBS_CAP",
    "fingerprint_args",
    "fingerprint_content",
    "baseline_delta",
    "make_set_membership_delta",
    "classify_error",
    "observe_call",
    "build_observations",
]

DEFAULT_OBS_CAP = 24


class ToolCallObservation(TypedDict, total=False):
    """One AIMessage→tool_call→ToolMessage three-tuple as a structured record.

    ``total=False`` because some fields (``error_class``) are only set on error
    branches. Consumers should use ``dict.get()`` with defaults. Stored as a plain
    dict so it survives LangGraph state checkpoint serialization without custom
    encoders.
    """

    turn: int
    tool_call_id: str
    tool_name: str
    args_hash: str
    status: Literal["success", "error"]
    error_class: str
    retry_attempt: int
    content_hash: str
    same_args_repeat_count: int   # consecutive same (tool_name, args_hash)
    new_information_delta: float   # 0.0 = identical output, 1.0 = fully new


# (cur_args, cur_content, prior_args) -> new_information_delta in [0, 1]
DeltaFn = Callable[[dict, Any, "dict | None"], float]


# ── Hashing ───────────────────────────────────────────────────────


def fingerprint_args(args: Any) -> str:
    """Short, order-insensitive hash of tool-call arguments."""
    if not args:
        return "empty"
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return hashlib.md5(args.encode(errors="replace")).hexdigest()[:8]
    if isinstance(args, dict):
        normalized = json.dumps(args, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(normalized.encode()).hexdigest()[:8]
    return hashlib.md5(str(args).encode(errors="replace")).hexdigest()[:8]


def fingerprint_content(content: Any) -> str:
    """Short hash of a tool result's content (str / content-block list / other)."""
    if content is None:
        return "empty"
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
    else:
        text = str(content)
    return hashlib.md5(text.encode(errors="replace")).hexdigest()[:8]


# ── new_information_delta ──────────────────────────────────────────


def baseline_delta(cur_content_hash: str, prior_content_hash: str | None) -> float:
    """``0.0`` when the output is byte-identical to the previous same-tool call.

    Intentionally coarse: the signal is "did anything come back that we hadn't
    seen on the last identical call." Per-tool overrides give a finer answer for
    tools where the content hash is too lax (see :func:`make_set_membership_delta`).
    """
    if prior_content_hash is None:
        return 1.0
    return 0.0 if prior_content_hash == cur_content_hash else 1.0


def make_set_membership_delta(list_field: str, item_key: str = "content") -> DeltaFn:
    """Build a delta function for tools whose args carry a *set* of items.

    Real case: a TODO-writing tool re-sent with one item renamed ``"foo"`` →
    ``"f"``. The content hash differs (so a baseline delta reports "new
    information"), but nothing meaningful changed. Comparing set membership of the
    items treats the rename as the cosmetic edit it is and reports near-zero delta
    for an otherwise-identical list — which is what the loop detector needs to fire.

    ``list_field`` is the args key holding the list (e.g. ``"todos"``); ``item_key``
    is the field on each item to compare by (e.g. ``"content"``).
    """

    def _delta(cur_args: dict, _cur_content: Any, prior_args: dict | None) -> float:
        if prior_args is None:
            return 1.0
        cur_set = {(i or {}).get(item_key, "") for i in (cur_args.get(list_field) or [])}
        prev_set = {(i or {}).get(item_key, "") for i in (prior_args.get(list_field) or [])}
        if not cur_set or not prev_set:
            return 1.0
        changed = (cur_set - prev_set) | (prev_set - cur_set)
        if not changed:
            return 0.0
        return min(1.0, len(changed) / max(len(cur_set), 1))

    return _delta


_ERROR_MARKERS = (
    "TimeoutError", "HTTPError", "ConnectionError", "ValidationError",
    "PermissionError", "FileNotFoundError", "ValueError", "KeyError",
    "TypeError", "RuntimeError",
)


def classify_error(content: Any) -> str:
    """Best-effort error-class hint from a ToolMessage's content. Not authoritative."""
    if not isinstance(content, str):
        content = str(content)
    for marker in _ERROR_MARKERS:
        if marker in content:
            return marker
    if "Traceback" in content or "Exception" in content:
        return "UnknownException"
    return "UnknownError"


# ── Single-call observation (the shared primitive) ────────────────


def observe_call(
    *,
    tool_name: str,
    args: Any,
    content: Any,
    status: str = "success",
    prior_obs: ToolCallObservation | None = None,
    prior_args: dict | None = None,
    delta_fn: DeltaFn | None = None,
    turn: int = 0,
    tool_call_id: str = "",
) -> ToolCallObservation:
    """Build one observation given the previous same-tool observation.

    ``same_args_repeat_count`` continues ``prior_obs``'s streak when the args hash
    matches, else resets to 1. ``new_information_delta`` is computed against
    ``prior_obs`` (content hash) or, when ``delta_fn`` is supplied, against
    ``prior_args``.
    """
    args_dict = args if isinstance(args, dict) else {}
    args_hash = fingerprint_args(args if args is not None else args_dict)
    content_hash = fingerprint_content(content)
    if status not in ("success", "error"):
        status = "success"

    if prior_obs is not None and prior_obs.get("args_hash") == args_hash:
        streak = int(prior_obs.get("same_args_repeat_count", 0)) + 1
    else:
        streak = 1

    if delta_fn is not None:
        delta = delta_fn(args_dict, content, prior_args)
    else:
        prior_ch = prior_obs.get("content_hash") if prior_obs else None
        delta = baseline_delta(content_hash, prior_ch)

    obs: ToolCallObservation = {
        "turn": turn,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "args_hash": args_hash,
        "status": status,  # type: ignore[typeddict-item]
        "content_hash": content_hash,
        "same_args_repeat_count": streak,
        "new_information_delta": round(float(delta), 4),
        "retry_attempt": 0,
    }
    if status == "error":
        obs["error_class"] = classify_error(content)
    return obs


# ── Stateless reconstruction from message history ─────────────────


def build_observations(
    messages: list,
    existing: list[ToolCallObservation] | None = None,
    cap: int = DEFAULT_OBS_CAP,
    delta_overrides: dict[str, DeltaFn] | None = None,
) -> list[ToolCallObservation]:
    """Reconstruct per-call observations from a message history. Stateless.

    Walks every AIMessage→ToolMessage pair not already recorded in ``existing``
    and appends a :class:`ToolCallObservation` for it. Idempotent: keyed on
    ``tool_call_id``, double-invocation never duplicates. Returns the merged list
    trimmed to ``cap`` (newest kept). Never raises on malformed messages.

    Because it rebuilds from ``messages`` it needs no instance state — the same
    history yields the same observations every time, so it is safe to run inside a
    stateless middleware scan. (Per-tool ``delta_overrides`` that need the prior
    call's *raw* args are most reliable in full-rebuild mode, i.e. ``existing=None``;
    the baseline content-hash delta always works.)
    """
    overrides = delta_overrides or {}
    existing = list(existing or [])
    existing_ids = {o.get("tool_call_id") for o in existing}

    tool_msgs_by_id: dict[str, ToolMessage] = {}
    for m in messages:
        if isinstance(m, ToolMessage):
            tcid = getattr(m, "tool_call_id", None)
            if tcid:
                tool_msgs_by_id[tcid] = m

    prior_obs_by_tool: dict[str, ToolCallObservation] = {}
    for o in existing:
        name = o.get("tool_name", "")
        if name:
            prior_obs_by_tool[name] = o
    prior_args_by_tool: dict[str, dict] = {}

    new_obs: list[ToolCallObservation] = []
    for i, m in enumerate(messages):
        if not isinstance(m, AIMessage):
            continue
        for tc in getattr(m, "tool_calls", None) or []:
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if not tc_id or tc_id in existing_ids:
                continue
            tool_msg = tool_msgs_by_id.get(tc_id)
            if tool_msg is None:
                continue  # tool hasn't completed yet; a later pass picks it up
            name = (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")) or ""
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            obs = observe_call(
                tool_name=name,
                args=args,
                content=getattr(tool_msg, "content", ""),
                status=getattr(tool_msg, "status", "success") or "success",
                prior_obs=prior_obs_by_tool.get(name),
                prior_args=prior_args_by_tool.get(name),
                delta_fn=overrides.get(name),
                turn=i,
                tool_call_id=tc_id,
            )
            new_obs.append(obs)
            existing_ids.add(tc_id)
            prior_obs_by_tool[name] = obs
            prior_args_by_tool[name] = args if isinstance(args, dict) else {}

    if not new_obs:
        return existing[-cap:] if len(existing) > cap else existing
    return (existing + new_obs)[-cap:]
