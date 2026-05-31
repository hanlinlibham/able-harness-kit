from langchain_core.messages import AIMessage, ToolMessage

from able_harness_kit.tool_observation import (
    baseline_delta,
    build_observations,
    fingerprint_args,
    make_set_membership_delta,
    observe_call,
)


def test_fingerprint_args_order_insensitive_and_value_sensitive():
    assert fingerprint_args({"a": 1, "b": 2}) == fingerprint_args({"b": 2, "a": 1})
    assert fingerprint_args({"a": 1}) != fingerprint_args({"a": 2})
    assert fingerprint_args(None) == fingerprint_args({}) == "empty"


def test_baseline_delta():
    assert baseline_delta("h", None) == 1.0   # no prior -> all new
    assert baseline_delta("h", "h") == 0.0     # identical output -> no progress
    assert baseline_delta("h", "g") == 1.0     # changed output -> progress


def test_set_membership_delta():
    d = make_set_membership_delta("todos")
    a1 = {"todos": [{"content": "write a"}]}
    a2 = {"todos": [{"content": "write"}]}                       # cosmetic rename
    a3 = {"todos": [{"content": "write a"}, {"content": "new"}]}  # real addition
    assert d(a1, "c", None) == 1.0     # no prior
    assert d(a1, "c", a1) == 0.0       # identical set -> no progress
    assert d(a2, "c", a1) > 0.0        # rename registers as change
    assert d(a3, "c", a1) == 0.5       # 1 changed of a 2-item set


def test_observe_call_streak_and_delta():
    o1 = observe_call(tool_name="read", args={"p": "x"}, content="A")
    assert o1["same_args_repeat_count"] == 1
    assert o1["new_information_delta"] == 1.0           # first call is always "new"

    o2 = observe_call(tool_name="read", args={"p": "x"}, content="A", prior_obs=o1)
    assert o2["same_args_repeat_count"] == 2            # same args -> streak grows
    assert o2["new_information_delta"] == 0.0           # identical output -> no progress

    o3 = observe_call(tool_name="read", args={"p": "y"}, content="A", prior_obs=o2)
    assert o3["same_args_repeat_count"] == 1            # different args -> reset


def test_observe_call_override_beats_content_hash():
    d = make_set_membership_delta("todos")
    a = {"todos": [{"content": "x"}]}
    o1 = observe_call(tool_name="w", args=a, content="v1", delta_fn=d, prior_args=None)
    # content changed v1 -> v2 (baseline would say progress) but the set is identical
    o2 = observe_call(
        tool_name="w", args=a, content="v2", delta_fn=d, prior_obs=o1, prior_args=a
    )
    assert o2["same_args_repeat_count"] == 2
    assert o2["new_information_delta"] == 0.0


# ── build_observations (stateless reconstruction from messages) ──


def _ai(calls):
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": i, "type": "tool_call"} for (i, n, a) in calls],
    )


def _tm(i, content):
    return ToolMessage(content=content, tool_call_id=i)


def test_build_observations_stuck_run():
    msgs = []
    for k in range(3):
        msgs.append(_ai([(f"t{k}", "read", {"p": "x"})]))
        msgs.append(_tm(f"t{k}", "SAME"))
    reads = [o for o in build_observations(msgs) if o["tool_name"] == "read"]
    assert [o["same_args_repeat_count"] for o in reads] == [1, 2, 3]
    assert reads[1]["new_information_delta"] == 0.0
    assert reads[2]["new_information_delta"] == 0.0


def test_build_observations_polling_keeps_delta_high():
    msgs = []
    for k, out in enumerate(["10%", "50%", "done"]):
        msgs.append(_ai([(f"p{k}", "poll", {"job": 1})]))
        msgs.append(_tm(f"p{k}", out))
    polls = [o for o in build_observations(msgs) if o["tool_name"] == "poll"]
    assert [o["same_args_repeat_count"] for o in polls] == [1, 2, 3]
    assert all(o["new_information_delta"] == 1.0 for o in polls)  # output changes = progress


def test_build_observations_idempotent():
    msgs = [_ai([("t1", "read", {"p": "x"})]), _tm("t1", "A")]
    first = build_observations(msgs)
    again = build_observations(msgs, existing=first)
    assert len(again) == len(first) == 1


def test_build_observations_rolling_cap():
    msgs = []
    for k in range(30):
        msgs.append(_ai([(f"t{k}", "read", {"p": k})]))
        msgs.append(_tm(f"t{k}", f"out{k}"))
    assert len(build_observations(msgs, cap=5)) == 5
