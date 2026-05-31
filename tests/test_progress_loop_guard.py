from langchain_core.messages import ToolMessage

from able_harness_kit import make_set_membership_delta
from able_harness_kit.loop_guard import ProgressAwareLoopGuardMiddleware, ProgressSignal


def _req(name="read", args=None, tcid="t1"):
    return type("R", (), {"tool_call": {"name": name, "args": args or {"p": "x"}, "id": tcid}})()


def _handler(contents, name="read", tcid="t1"):
    it = iter(contents)
    def h(_req):
        return ToolMessage(content=next(it), name=name, tool_call_id=tcid)
    return h


def test_validates_params():
    import pytest
    with pytest.raises(ValueError):
        ProgressAwareLoopGuardMiddleware(stop_at=0)
    with pytest.raises(ValueError):
        ProgressAwareLoopGuardMiddleware(progress_floor=2.0)


def test_stops_when_repeated_with_no_new_info():
    mw = ProgressAwareLoopGuardMiddleware(stop_at=3, progress_floor=0.1)
    h = _handler(["same", "same", "same"])
    out1 = mw.wrap_tool_call(_req(), h)   # streak 1, delta 1.0 -> pass
    out2 = mw.wrap_tool_call(_req(), h)   # streak 2, delta 0.0 -> pass (below stop_at)
    out3 = mw.wrap_tool_call(_req(), h)   # streak 3, delta 0.0 -> STALL
    assert out1.content == "same"
    assert out2.content == "same"
    assert out3.status == "error"
    assert "no new information" in out3.content


def test_passes_when_output_keeps_changing():
    # polling: same args every call, but a new result each time -> never stalls
    mw = ProgressAwareLoopGuardMiddleware(stop_at=3, progress_floor=0.1)
    h = _handler(["10%", "50%", "90%", "done"])
    out = None
    for _ in range(4):
        out = mw.wrap_tool_call(_req(), h)
    assert out.content == "done"          # the real last result, never replaced


def test_passes_when_args_differ_even_if_output_identical():
    # pagination: identical output but different args each call -> streak never grows
    mw = ProgressAwareLoopGuardMiddleware(stop_at=2, progress_floor=0.1)
    def h(_req):
        return ToolMessage(content="same", name="search", tool_call_id="t")
    out = None
    for page in (1, 2, 3, 4):
        out = mw.wrap_tool_call(_req(name="search", args={"page": page}), h)
    assert out.content == "same"          # never stalled despite identical content


def test_on_signal_reports_stall():
    seen: list[ProgressSignal] = []
    mw = ProgressAwareLoopGuardMiddleware(stop_at=2, progress_floor=0.1, on_signal=seen.append)
    h = _handler(["a", "a"])
    mw.wrap_tool_call(_req(), h)
    mw.wrap_tool_call(_req(), h)
    assert len(seen) == 2
    assert seen[-1].stalled is True
    assert seen[-1].repeat_count == 2
    assert seen[-1].new_information_delta == 0.0


def test_delta_override_catches_cosmetic_variation():
    # content differs each call (baseline would call it progress) but the todo SET
    # is identical -> the set-membership override reports no progress -> stalls.
    mw = ProgressAwareLoopGuardMiddleware(
        stop_at=2,
        progress_floor=0.1,
        delta_overrides={"write_todos": make_set_membership_delta("todos")},
    )
    args = {"todos": [{"content": "ship it"}]}
    h = _handler(["v1", "v2"], name="write_todos")
    out1 = mw.wrap_tool_call(_req(name="write_todos", args=args), h)
    out2 = mw.wrap_tool_call(_req(name="write_todos", args=args), h)
    assert out1.content == "v1"
    assert out2.status == "error"
    assert "no new information" in out2.content
