"""Session-affinity routing for LiteLLM via weighted Rendezvous (HRW) hashing.

把同一会话稳定路由到同一个端点/缓存域(tag),从而命中上游 prompt cache;
按 model_info.hrw_weight 加权分配;只在“当前健康(未 cooldown)”的端点里选。
所有内部 API 调用都有 except 兜底:坏了也只降级,不阻断请求。
"""
import hashlib
import inspect
import logging
import math

from litellm.integrations.custom_logger import CustomLogger

log = logging.getLogger("session_affinity")


def _unit(key, tag):
    """把 (key, tag) 的哈希映射到开区间 (0,1)。"""
    h = hashlib.sha256(f"{key}:{tag}".encode()).hexdigest()
    return (int(h[:16], 16) + 1) / (2 ** 64 + 1)


def _wscore(key, tag, weight):
    """加权 HRW 分数;weight 越大越容易成为最大值。"""
    return -weight / math.log(_unit(key, tag))


class SessionAffinity(CustomLogger):

    # 把请求里已有的 prompt-cache 断点 ttl 统一改写成这个值(如 "1h")。
    # 设为 None 关闭改写,保持客户端原样(Claude Code 默认 = 5min)。
    # 注意:仅当底层模型被 litellm 认作支持 1h(如 Claude 4.5/4.6/opus-4.7)时才真正生效,
    # 否则 litellm 会在发往 Bedrock 前剥掉 ttl 降级回 5min(不报错)。
    force_cache_ttl = "1h"

    def _rewrite_cache_ttl(self, data):
        """把 data 里已存在的 ephemeral cache_control 断点的 ttl 就地改成 force_cache_ttl。

        只改**已有断点**(不新增,避免改变客户端的缓存语义/断点数量);
        仅处理 list 形态的 system / message content(str 挂不上 cache_control);
        全程兜底,失败只降级(保持原样),绝不阻断请求。
        """
        ttl = self.force_cache_ttl
        if not ttl:
            return
        try:
            def fix(block):
                if not isinstance(block, dict):
                    return
                cc = block.get("cache_control")
                if isinstance(cc, dict) and cc.get("type") == "ephemeral":
                    cc["ttl"] = ttl

            system = data.get("system")
            if isinstance(system, list):
                for b in system:
                    fix(b)
            for m in data.get("messages") or []:
                c = m.get("content") if isinstance(m, dict) else None
                if isinstance(c, list):
                    for b in c:
                        fix(b)
            for t in data.get("tools") or []:
                fix(t)
        except Exception as e:
            log.warning("cache-ttl rewrite skipped: %r", e)

    def _tag_weight(self, dep):
        lp = dep.get("litellm_params") or {}
        mi = dep.get("model_info") or {}
        tags = lp.get("tags") or []
        tag = next((t for t in tags if t != "default"), None)
        # 权重来源(任一即可,方便 UI 配置):
        #   model_info.hrw_weight  →  litellm_params.weight  →  默认 1
        w = float(mi.get("hrw_weight") or lp.get("weight") or 1)
        return tag, w

    def _select(self, key, cands):
        """纯函数:在 {tag: weight} 里选加权 HRW 分数最大的 tag。"""
        return max(cands, key=lambda t: _wscore(key, t, cands[t]))

    async def _healthy_deployments(self, r, model):
        """拿到当前健康(未 cooldown)的 deployment 列表;兼容签名/返回形态的版本差异。

        1.90.x: _get_healthy_deployments(model, parent_otel_span) -> (healthy, all)
        旧版本: 可能只收 model,或直接返回 list。
        """
        fn = getattr(r, "_get_healthy_deployments", None)
        if fn is None:
            raise AttributeError("Router._get_healthy_deployments not found")
        try:
            res = fn(model, None)          # 当前版本需要 parent_otel_span
        except TypeError:
            res = fn(model)                # 兼容旧版只收 model
        if inspect.isawaitable(res):
            res = await res
        if isinstance(res, tuple):
            res = res[0]                   # (healthy, all) -> 取 healthy
        return res or []

    async def _candidates(self, model):
        """返回 {tag: weight};优先只含当前健康端点,读不到健康信息就退回全部。"""
        from litellm.proxy.proxy_server import llm_router as r
        if r is None:
            return {}
        full = {}
        for d in r.model_list:
            if d.get("model_name") == model:
                tag, w = self._tag_weight(d)
                if tag:
                    full[tag] = w
        try:
            healthy = await self._healthy_deployments(r, model)
            htags = {self._tag_weight(d)[0] for d in healthy}
            picked = {t: w for t, w in full.items() if t in htags}
            return picked or full
        except Exception as e:
            log.warning(
                "health-aware routing degraded (litellm internal API changed?): %r", e
            )
            return full

    def _session_key(self, data):
        for mkey in ("metadata", "litellm_metadata"):
            meta = data.get(mkey) or {}
            for k in ("user_id", "session_id"):
                if meta.get(k):
                    return str(meta[k])
        parts = []
        if isinstance(data.get("system"), str):
            parts.append(data["system"])
        for m in (data.get("messages") or [])[:2]:
            c = m.get("content")
            parts.append(c if isinstance(c, str) else str(c))
        return "|".join(parts) or "default"

    # tag 过滤按 function_name 从不同键读 tag:
    #   /v1/chat/completions -> metadata["tags"]
    #   /v1/messages         -> litellm_metadata["tags"]
    # 两个都写,确保两条路径都被采纳。
    _TAG_KEYS = ("metadata", "litellm_metadata")

    def _has_explicit_tags(self, data):
        # 请求已显式带 tag(如 x-litellm-tags 头或 body tags)时,尊重它,不覆盖。
        if data.get("tags"):
            return True
        for mkey in self._TAG_KEYS:
            md = data.get(mkey)
            if isinstance(md, dict) and md.get("tags"):
                return True
        return False

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            self._rewrite_cache_ttl(data)
            if self._has_explicit_tags(data):
                return data
            cands = await self._candidates(data.get("model"))
            if cands:
                key = self._session_key(data)
                tag = self._select(key, cands)
                data["tags"] = [tag]
                for mkey in self._TAG_KEYS:
                    md = data.setdefault(mkey, {})
                    if isinstance(md, dict):
                        md["tags"] = [tag]
                log.debug("session_affinity -> model=%s tag=%s call_type=%s",
                          data.get("model"), tag, call_type)
        except Exception as e:
            log.warning("session_affinity hook skipped: %r", e)
        return data


proxy_handler_instance = SessionAffinity()
