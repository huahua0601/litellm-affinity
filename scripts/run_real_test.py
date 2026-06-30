#!/usr/bin/env python3
"""真实 Bedrock prompt-cache 验证(配合 real_config.yaml)。

证明:① 缓存按 (账户,模型) 域隔离;② 同会话粘性 -> 命中。
用 x-litellm-tags 头强制 deployment(hook 尊重显式 tag,不覆盖)。
"""
import json
import random
import sys
import urllib.request

BASE = "http://localhost:4111"
KEY = "sk-test"

# 大 system prompt:带唯一 nonce 保证全新缓存;~3000+ token 远超门槛。
NONCE = f"RUN-{random.randint(10**8, 10**9)}"
PARA = ("You are a meticulous senior software engineer assisting with a large "
        "codebase. Follow conventions, prefer clarity, and explain trade-offs. ")
BIG_SYSTEM = f"[cache-probe {NONCE}] " + (PARA * 240)   # 远超 2048 token


def call(user_msg, tag=None):
    body = {
        "model": "claude-aff",
        "max_tokens": 16,
        "messages": [
            {"role": "system",
             "content": [{"type": "text", "text": BIG_SYSTEM,
                          "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": user_msg},
        ],
    }
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    if tag:
        headers["x-litellm-tags"] = tag
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        model_id = r.headers.get("x-litellm-model-id")
        payload = json.loads(r.read())
    u = payload.get("usage", {}) or {}
    details = u.get("prompt_tokens_details") or {}
    creation = u.get("cache_creation_input_tokens")
    read = (u.get("cache_read_input_tokens")
            if u.get("cache_read_input_tokens") is not None
            else details.get("cached_tokens"))
    return {"model_id": model_id, "prompt": u.get("prompt_tokens"),
            "creation": creation, "read": read}


def show(label, r):
    print(f"{label:<28} dep={r['model_id']:<8} "
          f"prompt={r['prompt']} creation={r['creation']} read={r['read']}")


def main():
    print(f"=== 真实 Bedrock 缓存测试 (nonce {NONCE}) ===\n")
    print("--- 缓存域隔离 ---")
    show("① 域A 首次 (acct-0)", call("question one", tag="acct-0"))
    show("② 域A 再次 (acct-0)", call("question two", tag="acct-0"))
    show("③ 域B 同 prompt (acct-1)", call("question three", tag="acct-1"))
    print("\n  期望:② read>0(命中);③ read=0(换域 miss = 跨账户等价场景)\n")

    print("--- 粘性自动命中(不带头,hook 路由)---")
    a1 = call("repeated-session-X")
    a2 = call("repeated-session-X")
    show("④ 同会话 turn1", a1)
    show("④ 同会话 turn2", a2)
    same = a1["model_id"] == a2["model_id"]
    print(f"\n  同会话两轮落同一 dep: {same};turn2 read>0: {bool(a2['read'])}")
    print("\n说明:用不同 model ID 模拟两个缓存域;真实跨账户为同一现象换一层。")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:500], file=sys.stderr)
        sys.exit(1)
