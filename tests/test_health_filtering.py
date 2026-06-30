"""L3b:用真实的 cooldown 机制验证 hook 的健康过滤(进程内,直连真实 Router)。"""
import asyncio

import litellm.proxy.proxy_server as ps
from litellm import Router
from litellm.router_utils.cooldown_handlers import _set_cooldown_deployments
from session_affinity import SessionAffinity


def _make_router():
    return Router(model_list=[{
        "model_name": "claude-sonnet",
        "litellm_params": {"model": "openai/fake", "api_key": "k",
                           "mock_response": "ok", "tags": [f"acct-{i}"]},
        "model_info": {"id": f"id-acct-{i}", "hrw_weight": [3, 1, 2][i]},
    } for i in range(3)])


async def _scenario():
    r = _make_router()
    ps.llm_router = r                       # 让 hook 的 _candidates 用上这个 router
    h = SessionAffinity()

    # 健康时:三个账户都在
    before = await h._candidates("claude-sonnet")
    assert set(before) == {"acct-0", "acct-1", "acct-2"}, before

    # 记录每个会话原本落到哪
    keys = [f"sess-{i}" for i in range(3000)]
    routed_before = {k: h._select(k, before) for k in keys}

    # 真实把 id-acct-1 打进 cooldown(429);在运行中的事件循环里调
    _set_cooldown_deployments(
        litellm_router_instance=r,
        original_exception=Exception("rate limited"),
        exception_status=429,
        deployment="id-acct-1",
        time_to_cooldown=60,
    )
    await asyncio.sleep(0.05)               # 让 cooldown 写入生效

    # cooldown 后:acct-1 应从候选消失
    after = await h._candidates("claude-sonnet")
    assert set(after) == {"acct-0", "acct-2"}, f"acct-1 没被剔除: {after}"

    # 最小搬动:原本在 acct-0/acct-2 的会话一个都不该动;只有 acct-1 的会话迁走
    moved_off_healthy = 0
    for k in keys:
        new = h._select(k, after)
        if routed_before[k] != "acct-1":
            moved_off_healthy += int(new != routed_before[k])
        else:
            assert new in ("acct-0", "acct-2")     # acct-1 的会话迁到健康账户
    assert moved_off_healthy == 0, f"{moved_off_healthy} 个健康会话被无谓搬动"


def test_cooldown_removes_account_and_keeps_others():
    asyncio.run(_scenario())
