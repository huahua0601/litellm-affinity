#!/bin/bash
# 列出当前 litellm proxy 中配置的 Bedrock 模型 + 各自的 model_id(ARN/底层模型) + 用量信息。
#
# 用法:
#   scripts/list_bedrock_usage.sh
#   LITELLM_HOST=http://localhost:4111 LITELLM_KEY=sk-affinity-test \
#     START_DATE=2026-06-01 END_DATE=2026-07-04 scripts/list_bedrock_usage.sh
#
# 数据源(litellm 管理端点):
#   /model/info            -> 模型配置(model_name / 底层 model / model_id(ARN) / region / base_model / tags)
#   /global/spend/models   -> 按底层模型的累计花费
#   /spend/logs            -> 指定时间窗内按模型的花费(按天聚合后再求和)
set -euo pipefail

HOST="${LITELLM_HOST:-http://localhost:4111}"
KEY="${LITELLM_KEY:-sk-affinity-test}"
START_DATE="${START_DATE:-$(date -u -d '30 days ago' +%F 2>/dev/null || date -u -v-30d +%F)}"
END_DATE="${END_DATE:-$(date -u -d 'tomorrow' +%F 2>/dev/null || date -u -v+1d +%F)}"

# 找一个可用的 python(优先项目 venv)
PY="/home/ec2-user/litellm-affinity/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"

AUTH=(-H "Authorization: Bearer ${KEY}")

# 先做连通性检查,失败直接给出清晰提示。
if ! curl -s -m5 -o /dev/null "${HOST}/health/liveliness"; then
  echo "ERROR: 无法连接 litellm proxy: ${HOST}" >&2
  echo "  确认服务在运行、HOST/端口正确(LITELLM_HOST 可覆盖)。" >&2
  exit 1
fi

model_info="$(curl -s -m15 "${AUTH[@]}" "${HOST}/v1/model/info")"
spend_models="$(curl -s -m15 "${AUTH[@]}" "${HOST}/global/spend/models" || echo '[]')"
spend_logs="$(curl -s -m15 "${AUTH[@]}" "${HOST}/spend/logs?start_date=${START_DATE}&end_date=${END_DATE}" || echo '[]')"

MODEL_INFO="$model_info" SPEND_MODELS="$spend_models" SPEND_LOGS="$spend_logs" \
START="$START_DATE" END="$END_DATE" HOSTNAME_="$HOST" "$PY" - <<'PYEOF'
import json, os, sys

def load(name, default):
    raw = os.environ.get(name, "")
    try:
        return json.loads(raw) if raw else default
    except Exception:
        return default

info = load("MODEL_INFO", {})
spend_models = load("SPEND_MODELS", [])
spend_logs = load("SPEND_LOGS", [])
start, end, host = os.environ["START"], os.environ["END"], os.environ["HOSTNAME_"]

def _looks_encrypted(s):
    """litellm 对 UI/DB 加的 model 会加密存储,取回时是一长串无 '.'/'/' 的 base64 样字符。"""
    s = str(s or "")
    return len(s) > 50 and ("." not in s) and ("/" not in s)

# 累计花费(按底层 model 字符串)
cumulative = {}
for row in (spend_models if isinstance(spend_models, list) else []):
    if isinstance(row, dict) and "model" in row:
        cumulative[row["model"]] = row.get("total_spend", 0.0) or 0.0

# 时间窗花费(spend/logs 按天聚合 -> 求和到每个 model)
windowed = {}
for day in (spend_logs if isinstance(spend_logs, list) else []):
    models = (day or {}).get("models") or {}
    if isinstance(models, dict):
        for m, v in models.items():
            windowed[m] = windowed.get(m, 0.0) + (v or 0.0)

def usage_for(*keys):
    """按多个候选 key 找花费(底层 model 名可能带/不带 bedrock/ 前缀)。"""
    cum = win = None
    for k in keys:
        if k is None:
            continue
        for cand in (k, f"bedrock/{k}", k.replace("bedrock/", "")):
            if cum is None and cand in cumulative:
                cum = cumulative[cand]
            if win is None and cand in windowed:
                win = windowed[cand]
    return cum, win

rows = info.get("data", []) if isinstance(info, dict) else []
bedrock = []
for m in rows:
    lp = m.get("litellm_params") or {}
    mi = m.get("model_info") or {}
    model = str(lp.get("model") or "")
    base_model = mi.get("base_model")
    # 判定是否 bedrock:model 前缀 / provider 字段 / base_model 前缀
    is_bedrock = (
        model.startswith("bedrock/")
        or lp.get("custom_llm_provider") == "bedrock"
        or str(base_model or "").startswith(("us.", "eu.", "jp.", "au.", "global.", "anthropic.", "amazon.", "meta."))
        or "anthropic." in model or "amazon." in model
    )
    if not is_bedrock:
        continue
    model_id = lp.get("model_id")  # 显式 ARN / inference profile id(若配了)
    # model 字段被 UI 加密时会是乱码;优先展示可读来源
    display_model = model if (model and not _looks_encrypted(model)) else (base_model or "<encrypted/db>")
    cum, win = usage_for(model if not _looks_encrypted(model) else None, base_model, str(model_id or ""))
    bedrock.append({
        "model_name": m.get("model_name"),
        "id": mi.get("id"),
        "model": display_model,
        "model_id(ARN)": model_id or "-",
        "base_model": base_model or "-",
        "region": lp.get("aws_region_name") or "-",
        "tags": lp.get("tags") or [],
        "spend_total": cum,
        "spend_window": win,
    })

def _fmt(v):
    return "-" if v is None else f"${v:.6f}"

# 输出
print(f"litellm proxy: {host}")
print(f"用量时间窗: {start} ~ {end}")
print(f"Bedrock 模型条目: {len(bedrock)}")
print("=" * 100)
for b in bedrock:
    print(f"● model_name : {b['model_name']}   (deployment id: {b['id']})")
    print(f"  底层 model : {b['model']}")
    print(f"  model_id   : {b['model_id(ARN)']}")
    print(f"  base_model : {b['base_model']}   region: {b['region']}   tags: {b['tags']}")
    print(f"  用量        : 累计花费={_fmt(b['spend_total'])}   窗口内花费={_fmt(b['spend_window'])}")
    print("-" * 100)

# 末尾附全量 JSON(便于程序化消费)
print("\n[JSON]")
print(json.dumps(bedrock, ensure_ascii=False, indent=2, default=str))
PYEOF
