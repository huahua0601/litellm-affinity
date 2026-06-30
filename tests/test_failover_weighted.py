"""下线迁移测试:3 个加权端点(US:JP:EU=3:2:1),用真实 cooldown 机制。
验证:某端点下线后,① 它的 session 迁走;② 其他端点 session 零扰动;③ 迁走的按剩余权重分摊。
路由用的是 proxy 实际调用的同一个 hook 函数(只是不发往真实 Bedrock——路由决策本就在调用前完成)。"""
import asyncio
from collections import Counter

import litellm.proxy.proxy_server as ps
from litellm import Router
from litellm.router_utils.cooldown_handlers import _set_cooldown_deployments
from session_affinity import SessionAffinity

WEIGHTS = {"dep-us": 3, "dep-jp": 2, "dep-eu": 1}
TAG_OF = {"dep-us": "acct-0", "dep-jp": "acct-1", "dep-eu": "acct-2"}


def _router():
    return Router(model_list=[{
        "model_name": "claude-aff",
        "litellm_params": {"model": "openai/fake", "api_key": "k",
                           "mock_response": "ok", "tags": [TAG_OF[dep]]},
        "model_info": {"id": dep, "hrw_weight": w},
    } for dep, w in WEIGHTS.items()])


async def _route_all(h, keys):
    cands = await h._candidates("claude-aff")
    return {k: h._select(k, cands) for k in keys}, cands


async def _scenario():
    ps.llm_router = _router()
    h = SessionAffinity()
    keys = [f"sess-{i}" for i in range(6000)]

    before, cands0 = await _route_all(h, keys)
    assert set(cands0) == {"acct-0", "acct-1", "acct-2"}
    dist0 = Counter(before.values())
    print("全健康分布:", dict(dist0), "(期望 ~3000:2000:1000)")

    # 下线 EU(权重1)
    _set_cooldown_deployments(litellm_router_instance=ps.llm_router,
                              original_exception=Exception("429"),
                              exception_status=429, deployment="dep-eu",
                              time_to_cooldown=60)
    await asyncio.sleep(0.05)

    after, cands1 = await _route_all(h, keys)
    assert set(cands1) == {"acct-0", "acct-1"}, f"eu 未被剔除: {cands1}"

    migrated = [k for k in keys if before[k] == "acct-2"]
    untouched = [k for k in keys if before[k] != "acct-2"]

    # ② 其他端点零扰动
    moved = sum(after[k] != before[k] for k in untouched)
    assert moved == 0, f"{moved} 个 us/jp 会话被无谓搬动"

    # ① 迁走的 session 都落到健康端点
    assert all(after[k] in ("acct-0", "acct-1") for k in migrated)

    # ③ 迁走的按剩余权重 us:jp = 3:2 分摊
    split = Counter(after[k] for k in migrated)
    n = len(migrated)
    r_us = split["acct-0"] / n
    print(f"EU 下线: 迁移 {n} 个; 去向 us={split['acct-0']} jp={split['acct-1']} "
          f"(us 占比 {r_us:.2f}, 期望 0.60)")
    assert abs(r_us - 0.6) < 0.05, f"迁移未按 3:2 分摊: us 占比 {r_us:.2f}"
    print("✓ EU 下线:迁移正常 / us,jp 零扰动 / 按权重分摊")


def test_failover_eu_offline():
    asyncio.run(_scenario())


async def _scenario_big():
    # 下线最大的 US(权重3),验证大规模迁移时其他端点仍零扰动
    ps.llm_router = _router()
    h = SessionAffinity()
    keys = [f"s-{i}" for i in range(6000)]
    before, _ = await _route_all(h, keys)
    _set_cooldown_deployments(litellm_router_instance=ps.llm_router,
                              original_exception=Exception("429"),
                              exception_status=429, deployment="dep-us",
                              time_to_cooldown=60)
    await asyncio.sleep(0.05)
    after, cands = await _route_all(h, keys)
    assert set(cands) == {"acct-1", "acct-2"}
    untouched = [k for k in keys if before[k] != "acct-0"]
    assert sum(after[k] != before[k] for k in untouched) == 0, "jp/eu 被扰动"
    print("✓ US 下线:jp,eu 上的会话零扰动")


def test_failover_us_offline():
    asyncio.run(_scenario_big())
