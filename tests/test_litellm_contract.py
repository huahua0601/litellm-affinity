"""L2 契约测试:对“当前安装的 litellm 版本”断言我们依赖的内部接口仍在。
升级后这里变红 = 需要更新 session_affinity 里对应的内部调用。"""
import inspect


def test_hook_signature():
    from litellm.integrations.custom_logger import CustomLogger
    assert hasattr(CustomLogger, "async_pre_call_hook")
    sig = inspect.signature(CustomLogger.async_pre_call_hook)
    for p in ("data", "call_type"):
        assert p in sig.parameters, f"async_pre_call_hook 缺参数 {p},签名变了"


def test_model_list_roundtrip():
    from litellm import Router
    r = Router(model_list=[{
        "model_name": "m",
        "litellm_params": {"model": "openai/x", "api_key": "k",
                           "tags": ["acct-0"], "mock_response": "ok"},
        "model_info": {"hrw_weight": 3},
    }])
    d = r.model_list[0]
    assert d["litellm_params"].get("tags") == ["acct-0"], "tag 字段被改/吞了"
    assert (d.get("model_info") or {}).get("hrw_weight") == 3, "model_info.hrw_weight 丢了"


def test_hook_can_read_healthy_deployments():
    # 直接验证修正后的 _healthy_deployments 在当前版本能拿到“带 tag 的健康列表”。
    # 签名/返回形态再变就会在这里变红。
    import asyncio
    from litellm import Router
    from session_affinity import SessionAffinity
    r = Router(model_list=[{
        "model_name": "claude-sonnet",
        "litellm_params": {"model": "openai/fake", "api_key": "k",
                           "mock_response": "ok", "tags": [f"acct-{i}"]},
        "model_info": {"id": f"id-{i}"},
    } for i in range(3)])
    deps = asyncio.run(SessionAffinity()._healthy_deployments(r, "claude-sonnet"))
    assert isinstance(deps, list) and len(deps) == 3, f"健康列表形态不对: {type(deps)}"
    tags = {(d["litellm_params"].get("tags") or [None])[0] for d in deps}
    assert tags == {"acct-0", "acct-1", "acct-2"}, f"拿不到 tag: {tags}"


def test_tag_metadata_keys_are_covered():
    # litellm 按 function_name 决定 tag 从哪个 metadata 键读。
    # 断言所有相关端点要求的键都在我们 hook 会写的 _TAG_KEYS 里,
    # 否则某条路径(如 /v1/messages 走 litellm_metadata)会静默失去粘性。
    from litellm.router_utils.batch_utils import _get_router_metadata_variable_name
    from session_affinity import SessionAffinity
    fns = ["acompletion", "completion", "anthropic_messages",
           "_ageneric_api_call_with_fallbacks", "aadapter_completion"]
    required = {_get_router_metadata_variable_name(function_name=f) for f in fns}
    missing = required - set(SessionAffinity._TAG_KEYS)
    assert not missing, f"这些 metadata 键没被 hook 覆盖,对应端点会丢粘性: {missing}"


def test_hook_writes_tags_into_both_metadata_keys():
    import asyncio
    import litellm.proxy.proxy_server as ps
    from litellm import Router
    from session_affinity import SessionAffinity
    ps.llm_router = Router(model_list=[{
        "model_name": "claude-sonnet",
        "litellm_params": {"model": "openai/fake", "api_key": "k",
                           "mock_response": "ok", "tags": [f"acct-{i}"]},
        "model_info": {"id": f"id-{i}", "hrw_weight": [3, 1, 2][i]},
    } for i in range(3)])
    h = SessionAffinity()

    async def run(call_type, data):
        return await h.async_pre_call_hook(None, None, data, call_type)

    # 模拟 /v1/messages 的 data:只有 litellm_metadata
    d = asyncio.run(run("anthropic_messages",
                        {"model": "claude-sonnet", "system": "sys",
                         "messages": [{"role": "user", "content": "hi"}]}))
    assert d.get("litellm_metadata", {}).get("tags"), "anthropic 路径没写 litellm_metadata.tags"
    # 模拟 /v1/chat/completions 的 data
    d2 = asyncio.run(run("acompletion",
                         {"model": "claude-sonnet",
                          "messages": [{"role": "user", "content": "hi"}]}))
    assert d2.get("metadata", {}).get("tags"), "completion 路径没写 metadata.tags"


def test_healthy_method_present():
    # 探测 Router 上用于列出健康 deployment 的方法名;打印出来便于核对
    from litellm import Router
    r = Router(model_list=[{
        "model_name": "m",
        "litellm_params": {"model": "openai/x", "api_key": "k", "mock_response": "ok"},
    }])
    names = sorted(n for n in dir(r) if "healthy" in n.lower() or "cooldown" in n.lower())
    print("ROUTER health/cooldown methods:", names)
    assert any("healthy" in n for n in names), (
        "Router 上找不到 *healthy* 方法,health-aware 路径已失效,请更新 _candidates")
