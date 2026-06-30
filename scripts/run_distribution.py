#!/usr/bin/env python3
"""通过真实 proxy 验证:多 session 是否按权重(US:JP:EU=3:2:1)路由。
用极小请求(max_tokens=1)只为读 x-litellm-model-id 路由结果,成本可忽略。"""
import json
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import os
BASE, KEY, N = "http://localhost:4111", "sk-test", int(os.environ.get("NSESS", "240"))
EXPECT = {"dep-us": 3, "dep-jp": 2, "dep-eu": 1}


def route(i):
    body = {"model": "claude-aff", "max_tokens": 1,
            "messages": [{"role": "user", "content": f"session-{i}"}]}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.headers.get("x-litellm-model-id")
    except Exception as e:
        return f"ERR:{e}"


def main():
    with ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(route, range(N)))
    c = Counter(results)
    print(f"=== {N} sessions 路由分布(真实 proxy)===")
    tot = sum(EXPECT.values())
    for dep, w in EXPECT.items():
        got, exp = c.get(dep, 0), N * w / tot
        print(f"  {dep:<8} got={got:<4} expect≈{exp:.0f}  ({100*got/N:.1f}% vs {100*w/tot:.1f}%)")
    errs = {k: v for k, v in c.items() if str(k).startswith("ERR")}
    if errs:
        print("  ERRORS:", errs)


if __name__ == "__main__":
    main()
