"""ToolResultBudgetMiddleware — keep oversized tool results out of the context window.

A large tool result (a 200 KB API dump, a whole file) blows the context budget
and pushes the model toward premature summarization. This middleware caps each
tool result at a character budget; oversized results are either offloaded via a
caller-supplied callback (store-and-reference) or, by default, truncated with a
retrieval hint.

The offload contract is backend-neutral:

    offload(content, tool_name) -> OffloadRef(ref, summary, retrieval_hint)

Bring your own store (filesystem, blob, vector DB); the middleware only decides
*when* to offload and how to phrase the placeholder the model sees.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger("able_harness_kit.tool_result_budget")


@dataclass
class OffloadRef:
    """A reference returned by an offload callback, surfaced to the model."""

    ref: str
    summary: str = ""
    retrieval_hint: str = ""


OffloadFn = Callable[[str, str], OffloadRef]  # (content, tool_name) -> ref


def _content_str(result: Any) -> str | None:
    if isinstance(result, ToolMessage) and isinstance(result.content, str):
        return result.content
    return None


def apply_budget(
    content: str,
    *,
    limit: int,
    tool_name: str = "",
    offload: OffloadFn | None = None,
) -> str:
    """Return ``content`` unchanged if within ``limit``, else offload or truncate."""
    if len(content) <= limit:
        return content
    if offload is not None:
        ref = offload(content, tool_name)
        parts = [
            f"[tool result offloaded — {len(content)} chars exceeded the "
            f"{limit}-char budget]",
            f"ref: {ref.ref}",
        ]
        if ref.summary:
            parts.append(f"summary: {ref.summary}")
        if ref.retrieval_hint:
            parts.append(f"retrieve: {ref.retrieval_hint}")
        return "\n".join(parts)
    head = content[:limit]
    return (
        f"{head}\n\n[truncated — {len(content)} chars total, showing the first "
        f"{limit}. The full result was not kept in context; re-run the tool with a "
        f"narrower query or pagination to see more.]"
    )


class ToolResultBudgetMiddleware(AgentMiddleware):
    """Cap oversized tool results before they enter the model context."""

    def __init__(self, *, limit: int = 16_000, offload: OffloadFn | None = None) -> None:
        super().__init__()
        self.limit = limit
        self.offload = offload

    def _tool_name_of(self, request: ToolCallRequest) -> str:
        tc = getattr(request, "tool_call", None) or {}
        return (tc.get("name") if isinstance(tc, dict) else None) or ""

    def _apply(self, request: ToolCallRequest, result: Any) -> Any:
        content = _content_str(result)
        if content is None or len(content) <= self.limit:
            return result
        tool_name = self._tool_name_of(request)
        new_content = apply_budget(
            content, limit=self.limit, tool_name=tool_name, offload=self.offload
        )
        logger.info(
            "ToolResultBudget: capped %s result %d -> %d chars",
            tool_name,
            len(content),
            len(new_content),
        )
        return ToolMessage(
            content=new_content,
            name=result.name,
            tool_call_id=result.tool_call_id,
            status=getattr(result, "status", None) or "success",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        return self._apply(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        return self._apply(request, await handler(request))


__all__ = ["ToolResultBudgetMiddleware", "OffloadRef", "OffloadFn", "apply_budget"]
