"""L1 纯逻辑单测:cache_control ttl 改写(版本无关)。"""
from session_affinity import SessionAffinity


def _data_with_breakpoints():
    return {
        "model": "claude-aff",
        "system": [
            {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "no-cache"},
        ],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
            ]},
            {"role": "user", "content": "plain string content"},
        ],
        "tools": [
            {"name": "t1", "cache_control": {"type": "ephemeral"}},
            {"name": "t2"},
        ],
    }


def test_rewrites_existing_breakpoints_to_1h():
    h = SessionAffinity()
    d = _data_with_breakpoints()
    h._rewrite_cache_ttl(d)
    assert d["system"][0]["cache_control"]["ttl"] == "1h"
    assert d["messages"][0]["content"][0]["cache_control"]["ttl"] == "1h"
    assert d["tools"][0]["cache_control"]["ttl"] == "1h"


def test_does_not_add_new_breakpoints():
    h = SessionAffinity()
    d = _data_with_breakpoints()
    h._rewrite_cache_ttl(d)
    # 没有 cache_control 的 block 不应被加上断点
    assert "cache_control" not in d["system"][1]
    assert "cache_control" not in d["tools"][1]


def test_str_forms_are_untouched():
    h = SessionAffinity()
    d = {
        "system": "plain system string",
        "messages": [{"role": "user", "content": "plain string"}],
    }
    h._rewrite_cache_ttl(d)  # 不应抛异常
    assert d["system"] == "plain system string"
    assert d["messages"][0]["content"] == "plain string"


def test_switch_off_does_not_rewrite():
    h = SessionAffinity()
    h.force_cache_ttl = None
    d = _data_with_breakpoints()
    h._rewrite_cache_ttl(d)
    assert "ttl" not in d["system"][0]["cache_control"]
    assert "ttl" not in d["tools"][0]["cache_control"]


def test_non_ephemeral_untouched():
    h = SessionAffinity()
    d = {"system": [{"type": "text", "text": "x",
                     "cache_control": {"type": "persistent"}}]}
    h._rewrite_cache_ttl(d)
    assert "ttl" not in d["system"][0]["cache_control"]
