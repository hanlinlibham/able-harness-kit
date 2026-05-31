from langchain_core.messages import ToolMessage

from able_harness_kit.tool_result_budget import (
    OffloadRef,
    ToolResultBudgetMiddleware,
    apply_budget,
)


def test_under_budget_unchanged():
    assert apply_budget("short", limit=100) == "short"


def test_over_budget_truncates_with_hint():
    out = apply_budget("x" * 200, limit=50)
    assert out.startswith("x" * 50)
    assert "truncated" in out
    assert "200 chars total" in out


def test_offload_callback_used():
    def offload(content, tool_name):
        return OffloadRef(
            ref="blob://123", summary="big dump", retrieval_hint="fetch blob://123"
        )

    out = apply_budget("y" * 200, limit=50, offload=offload)
    assert "blob://123" in out
    assert "big dump" in out
    assert "offloaded" in out


def _req(name="dump", tcid="t1"):
    return type("R", (), {"tool_call": {"name": name, "id": tcid}})()


def test_middleware_caps_large_result():
    mw = ToolResultBudgetMiddleware(limit=20)
    big = ToolMessage(content="z" * 5000, name="dump", tool_call_id="t1")
    out = mw.wrap_tool_call(_req(), lambda r: big)
    assert len(out.content) < 5000          # capped well below the original
    assert out.content.startswith("z" * 20)
    assert "truncated" in out.content
    assert out.tool_call_id == "t1"


def test_middleware_passes_small_result():
    mw = ToolResultBudgetMiddleware(limit=1000)
    small = ToolMessage(content="tiny", name="dump", tool_call_id="t1")
    out = mw.wrap_tool_call(_req(), lambda r: small)
    assert out is small  # untouched when under budget


def test_middleware_offload_path():
    def offload(content, tool_name):
        return OffloadRef(ref=f"store://{tool_name}", retrieval_hint="re-query")

    mw = ToolResultBudgetMiddleware(limit=10, offload=offload)
    big = ToolMessage(content="w" * 50, name="dump", tool_call_id="t9")
    out = mw.wrap_tool_call(_req(tcid="t9"), lambda r: big)
    assert "store://dump" in out.content
    assert out.tool_call_id == "t9"
