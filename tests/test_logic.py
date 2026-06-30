"""L1 纯逻辑单测:版本无关,验证 HRW 算法本身。"""
from session_affinity import SessionAffinity

h = SessionAffinity()
C = {"acct-0": 3, "acct-1": 1, "acct-2": 2}


def test_affinity_stable():
    # 同 key + 同集合 -> 永远同一个账户
    for k in ("sess-abc", "user-42", "hello world|first turn"):
        assert h._select(k, C) == h._select(k, C)


def test_weight_distribution():
    # 流量 ~ 权重 3:1:2
    cnt = {t: 0 for t in C}
    n = 60000
    for i in range(n):
        cnt[h._select(f"s{i}", C)] += 1
    for t, w in C.items():
        share = cnt[t] / n
        assert abs(share - w / sum(C.values())) < 0.02, (t, share)


def test_minimal_disruption():
    # acct-0 进 cooldown 后,只有原本落在 acct-0 的会话该迁走,其余纹丝不动
    less = {"acct-1": 1, "acct-2": 2}
    stayed_changed = 0
    for i in range(20000):
        k = f"s{i}"
        before = h._select(k, C)
        after = h._select(k, less)
        if before != "acct-0":
            stayed_changed += int(before != after)
    assert stayed_changed == 0


def test_single_candidate():
    assert h._select("x", {"acct-1": 1}) == "acct-1"
