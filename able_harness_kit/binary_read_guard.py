"""BinaryReadGuardMiddleware — fail loud on binary ``read_file`` results.

The problem
===========
deepagents' ``read_file`` returns a base64 multimodal content block for non-text
files, intending the image to reach a vision-capable model. In practice some
gateways silently drop the multimodal block, and the model — receiving no usable
bytes — confidently claims it "read" the file and hallucinates content from the
surrounding context.

This guard
==========
Intercepts ``read_file`` results whose declared media type is **not** text-like
and replaces them with a structured error directing the agent to a dedicated
extractor (OCR / doc-to-markdown / type sniff). Turning a silent hallucination
into an explicit "use the right tool" nudge is strictly safer; the only cost is
one extra tool hop on the first attempt.

Backend-neutral: depends only on langchain message / middleware types. The tool
name, the metadata keys it reads, and the directive text are all configurable.
Disable per-deployment with ``BINARY_READ_GUARD_ENABLED=0``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger("able_harness_kit.binary_read_guard")

_TEXT_MIME_PREFIXES: tuple[str, ...] = ("text/",)
_TEXT_MIME_EXACT: frozenset[str] = frozenset(
    {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/javascript",
        "application/x-sh",
        "application/x-python",
        "application/toml",
        "application/x-toml",
        "application/csv",
    }
)


def is_text_like(mime_type: str | None) -> bool:
    """Return True if ``mime_type`` is safe to surface to the model as text.

    A missing / empty mime type returns ``True`` (do not block): the guard only
    intervenes when there is strong evidence the content is binary.
    """
    if not mime_type:
        return True
    if any(mime_type.startswith(p) for p in _TEXT_MIME_PREFIXES):
        return True
    return mime_type in _TEXT_MIME_EXACT


DEFAULT_DIRECTIVE = (
    "read_file: {path} is binary (mime_type={mime_type}); its bytes can't be "
    "reliably consumed as model input through the current transport — claiming "
    "to have read it would lead to hallucinated content.\n"
    "Use a dedicated extractor instead: an OCR / vision tool for images, a "
    "document-to-markdown converter for PDF / Office files, or a type-sniff tool "
    "when the type is unknown."
)


class BinaryReadGuardMiddleware(AgentMiddleware):
    """Replace binary ``read_file`` results with a directive to use an extractor."""

    def __init__(
        self,
        *,
        tool_name: str = "read_file",
        media_type_key: str = "read_file_media_type",
        path_key: str = "read_file_path",
        directive: str = DEFAULT_DIRECTIVE,
        text_like: Callable[[str | None], bool] = is_text_like,
        env_var: str = "BINARY_READ_GUARD_ENABLED",
    ) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.media_type_key = media_type_key
        self.path_key = path_key
        self.directive = directive
        self._text_like = text_like
        self.env_var = env_var

    def _enabled(self) -> bool:
        raw = os.environ.get(self.env_var, "1").strip().lower()
        return raw not in ("0", "false", "no", "off", "")

    def _tool_name_of(self, request: ToolCallRequest) -> str:
        tc = getattr(request, "tool_call", None) or {}
        return (tc.get("name") if isinstance(tc, dict) else None) or ""

    def _filter(self, tool_name: str, result: Any) -> ToolMessage | None:
        """Return a replacement ToolMessage, or None to pass the result through."""
        if tool_name != self.tool_name:
            return None
        if not isinstance(result, ToolMessage):
            return None  # e.g. a Command — leave it alone
        extra = getattr(result, "additional_kwargs", None) or {}
        media_type = extra.get(self.media_type_key)
        if self._text_like(media_type):
            return None
        path = extra.get(self.path_key, "<unknown path>")
        logger.warning(
            "BinaryReadGuard: blocked binary read_file path=%r mime=%r", path, media_type
        )
        return ToolMessage(
            content=self.directive.format(path=path, mime_type=media_type or "<unknown>"),
            name=self.tool_name,
            tool_call_id=result.tool_call_id,
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if not self._enabled():
            return handler(request)
        result = handler(request)
        replacement = self._filter(self._tool_name_of(request), result)
        return replacement if replacement is not None else result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if not self._enabled():
            return await handler(request)
        result = await handler(request)
        replacement = self._filter(self._tool_name_of(request), result)
        return replacement if replacement is not None else result


__all__ = ["BinaryReadGuardMiddleware", "is_text_like", "DEFAULT_DIRECTIVE"]
