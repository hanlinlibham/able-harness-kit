from langchain_core.messages import ToolMessage

from able_harness_kit.loop_guard import (
    LoopAction,
    LoopGuardMiddleware,
    LoopTracker,
    fingerprint_tool_call,
)


def test_fingerprint_stable_and_arg_sensitive():
    a = fingerprint_tool_call("search", {"q": "x"})
    b = fingerprint_tool_call("search", {"q": "x"})
    c = fingerprint_tool_call("search", {"q": "y"})
    d = fingerprint_tool_call("other", {"q": "x"})
    assert a == b
    assert a != c
    assert a != d


def test_fingerprint_arg_order_insensitive():
    a = fingerprint_tool_call("t", {"a": 1, "b": 2})
    b = fingerprint_tool_call("t", {"b": 2, "a": 1})
    assert a == b


def test_tracker_warn_then_stop():
    t = LoopTracker(warn_at=3, stop_at=5)
    assert t.observe("search", {"q": "x"}).action is LoopAction.NONE  # 1
    assert t.observe("search", {"q": "x"}).action is LoopAction.NONE  # 2
    assert t.observe("search", {"q": "x"}).action is LoopAction.WARN  # 3
    assert t.observe("search", {"q": "x"}).action is LoopAction.WARN  # 4
    assert t.observe("search", {"q": "x"}).action is LoopAction.STOP  # 5


def test_tracker_resets_on_different_call():
    t = LoopTracker(warn_at=2, stop_at=3)
    t.observe("a", {})
    assert t.observe("a", {}).action is LoopAction.WARN
    sig = t.observe("b", {})  # different call -> reset
    assert sig.count == 1
    assert sig.action is LoopAction.NONE


def test_tracker_validates_thresholds():
    import pytest

    with pytest.raises(ValueError):
        LoopTracker(warn_at=0)
    with pytest.raises(ValueError):
        LoopTracker(warn_at=5, stop_at=3)


def _req(name="search", args=None, tcid="t1"):
    return type("R", (), {"tool_call": {"name": name, "args": args or {"q": "x"}, "id": tcid}})()


def test_middleware_stops_on_repeat():
    mw = LoopGuardMiddleware(warn_at=2, stop_at=3)
    calls = {"n": 0}

    def handler(r):
        calls["n"] += 1
        return ToolMessage(content="ok", name="search", tool_call_id="t1")

    mw.wrap_tool_call(_req(), handler)        # 1 -> runs
    mw.wrap_tool_call(_req(), handler)        # 2 -> runs (WARN)
    out = mw.wrap_tool_call(_req(), handler)  # 3 -> STOP, short-circuit
    assert calls["n"] == 2                    # handler never reached on the 3rd
    assert out.status == "error"
    assert "not making progress" in out.content


def test_middleware_on_signal_callback():
    seen = []
    mw = LoopGuardMiddleware(warn_at=2, stop_at=9, on_signal=seen.append)

    def handler(r):
        return ToolMessage(content="ok", name="search", tool_call_id="t1")

    mw.wrap_tool_call(_req(), handler)  # NONE -> no signal callback
    mw.wrap_tool_call(_req(), handler)  # WARN -> callback fires
    assert len(seen) == 1
    assert seen[0].action is LoopAction.WARN
