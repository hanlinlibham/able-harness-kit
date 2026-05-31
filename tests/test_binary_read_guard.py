from langchain_core.messages import ToolMessage

from able_harness_kit.binary_read_guard import BinaryReadGuardMiddleware, is_text_like


def test_is_text_like():
    assert is_text_like("text/plain")
    assert is_text_like("application/json")
    assert is_text_like(None)   # missing -> don't block
    assert is_text_like("")     # empty   -> don't block
    assert not is_text_like("image/png")
    assert not is_text_like("application/pdf")


def _req(name="read_file", tcid="t1"):
    return type("R", (), {"tool_call": {"name": name, "id": tcid}})()


def _read_result(content, *, media_type=None, path=None, tcid="t1"):
    extra = {}
    if media_type is not None:
        extra["read_file_media_type"] = media_type
    if path is not None:
        extra["read_file_path"] = path
    return ToolMessage(
        content=content, name="read_file", tool_call_id=tcid, additional_kwargs=extra
    )


def test_binary_result_replaced_with_error():
    mw = BinaryReadGuardMiddleware()
    binary = _read_result("<base64...>", media_type="image/png", path="/x/a.png")
    out = mw.wrap_tool_call(_req(), lambda r: binary)
    assert out.status == "error"
    assert "binary" in out.content
    assert "/x/a.png" in out.content
    assert out.tool_call_id == "t1"


def test_text_result_passes_through():
    mw = BinaryReadGuardMiddleware()
    txt = _read_result("hello", media_type="text/plain")
    out = mw.wrap_tool_call(_req(), lambda r: txt)
    assert out is txt


def test_missing_media_type_passes_through():
    mw = BinaryReadGuardMiddleware()
    txt = _read_result("hello")  # no media type declared
    out = mw.wrap_tool_call(_req(), lambda r: txt)
    assert out is txt


def test_other_tools_untouched():
    mw = BinaryReadGuardMiddleware()
    res = _read_result("<base64>", media_type="image/png")
    out = mw.wrap_tool_call(_req(name="search"), lambda r: res)
    assert out is res


def test_env_disable(monkeypatch):
    monkeypatch.setenv("BINARY_READ_GUARD_ENABLED", "0")
    mw = BinaryReadGuardMiddleware()
    binary = _read_result("<base64>", media_type="image/png", path="/x/a.png")
    out = mw.wrap_tool_call(_req(), lambda r: binary)
    assert out is binary  # disabled -> passthrough
