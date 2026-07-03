# LiteLLM 会话粘性路由 —— 多云/多端点共享 Prompt Cache

> 让同一会话稳定路由到同一个上游缓存域,命中 prompt cache、降低成本;
> 对 Claude Code 开发者和 LiteLLM 管理员双向无感。已在 **litellm 1.90.1** 上端到端验证。

中文 | [English](README.en.md)

---

## 1. 背景与问题

同一个模型在 LiteLLM 里可能配置了**多个上游端点**:多个 AWS 账户 / 多个 region 的 Bedrock Claude、多个云厂商、多个项目、多个 API key,或不同容量池(为了分摊 TPM/RPM 配额、做容灾或成本优化)。
但多数上游的 prompt cache 都按某种**缓存域**隔离,例如 Bedrock 的 `(账户, region)`,或其他云厂商/模型服务里的 `(provider, project, region, credential, deployment)` 等边界:

- 同一会话的多轮请求若被负载均衡甩到不同缓存域,后续轮次**命中不了前面写入的缓存**;
- 结果是反复付 cache **写入**(基础输入价的 1.25×),而拿不到 cache **读取**(0.1×)的便宜。

LiteLLM 默认的负载均衡策略(`simple-shuffle` 等)是**无状态**的,每个请求独立选 deployment,所以天然破坏会话粘性。

**本质**:这不是缓存配错了,而是**路由缺少会话亲和性**。

---

## 2. 方案概述

一个 LiteLLM `async_pre_call_hook` 自定义回调,在请求发往上游模型服务 **之前**:

1. 从请求里取一个**稳定的会话标识**(优先 `metadata.user_id`/`session_id`,否则用 `system + 首条消息`);
2. 用**加权 Rendezvous(HRW)哈希**在"当前健康(未 cooldown)"的端点里选一个;
3. 把选中端点的 **tag** 写进请求,交给 LiteLLM 的 tag 路由精确投递。

特性:

| 能力 | 说明 |
|---|---|
| 会话粘性 | 同一会话永远落同一缓存域/端点 → 缓存稳定命中 |
| 加权分配 | 按 `model_info.hrw_weight` 成比例分配流量(配额大的多吃) |
| 健康感知 | 端点 cooldown 时自动剔除,其会话迁移到健康端点 |
| 最小搬动 | 端点增/减只影响"原本落在它上面"的会话,其余纹丝不动 |
| 多云可用 | 只依赖 LiteLLM deployment tag,可用于 Bedrock、Vertex AI、Anthropic、OpenAI-compatible 等多端点模型组 |
| 双向无感 | 纯 `config.yaml` + 一个回调文件;开发者、管理员都不用逐人配置 |
| 安全降级 | 所有内部 API 调用都有 `except` 兜底,坏了只降级、不阻断请求 |

---

## 3. 为什么用加权 HRW(而不是取模)

会话标识哈希到端点,要满足"端点增减时尽量少搬动",否则一个端点 cooldown 会让**全员缓存失效**。

- **取模 `hash % N`**:除数一变(N→N-1),几乎所有 key 重映射 → 缓存全废。❌
- **HRW(Rendezvous)**:对每个端点算 `score(key, 端点)`,取最大者。去掉一个端点只让"原本归它"的 key 迁走,其余不变。✅
- **加权 HRW**(Resch 公式):`score = -weight / ln(u)`,`u = hash(key,tag)` 归一化到 (0,1)。
  端点 i 被选中的概率 = `w_i / Σw`,同时保留最小搬动特性。

> 健康感知 + 加权天然组合:cooldown 的端点从候选集消失,`argmax` 在剩余加权集合上重选,流量自动按**剩余权重**重新分摊,无需额外代码。

---

## 4. 部署

### 4.1 目录

```
litellm-affinity/
├── README.md
├── pytest.ini                          # pythonpath=src / testpaths=tests
├── real_config.yaml                    # 真实上游配置示例   ┐
├── test_config.yaml                    # mock 测试配置     ├ 必须和 hook 同目录(见下)
├── session_affinity.py → src/...       # 软链接到源        ┘
├── src/
│   └── session_affinity.py             # 回调 hook 源(唯一真实文件,见 §4.3)
├── tests/                              # 三层测试(见 §7)
├── scripts/                            # 端到端验证脚本
└── secrets/
    ├── .secrets.env.template
    └── .secrets.env                    # 凭证(gitignore,source 使用)
```

> **为什么 config 和 hook 软链接放在根目录、而不是 `config/` 子目录?**
> litellm 加载 `callbacks: session_affinity.proxy_handler_instance` 时,是按
> **「config 文件所在目录」+ 模块名** 拼出 `.py` 路径来加载的(**不走 `PYTHONPATH`**)。
> 把 config 放进子目录会让这个相对路径随 worker 的 CWD 失配。
> 所以约定:**config 与 `session_affinity.py`(软链接到 `src/` 的源)同放项目根,从根目录启动**。
> 源码仍只有一份在 `src/`;`tests/`、`scripts/`、`secrets/` 正常分文件夹。

### 4.2 config.yaml(关键片段)

每个缓存域/上游端点一条 deployment,**同一个 `model_name`**,各带**唯一 tag** 和权重。下面用 Bedrock 举例;换成其他云或 OpenAI-compatible 端点时,只需要把 `litellm_params` 改成对应 provider 的字段,tag/weight 规则不变:

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/us.anthropic.claude-...       # 或任意 LiteLLM 支持的 provider/model
      aws_region_name: us-east-1
      # 缓存域 A 的凭证(如 AWS 账户/region、GCP 项目/region、OpenAI-compatible base_url/api_key)
      tags: ["cache-domain-0"]
    model_info:
      id: dep-a
      hrw_weight: 3                                  # ← 相对权重(配额/容量)
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/...                           # 也可以是另一云厂商或另一容量池
      aws_region_name: ...
      tags: ["cache-domain-1"]
    model_info:
      id: dep-b
      hrw_weight: 1

router_settings:
  enable_tag_filtering: true                         # ← 必须开

litellm_settings:
  callbacks: session_affinity.proxy_handler_instance # ← 挂载回调

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

> **单一事实源**:端点数、tag、权重全在 `model_list` 里;回调启动时自己读,不会漂移。
> 改云厂商、账户、区域、容量池或权重只动 config,不动代码。

### 4.3 session_affinity.py

见同目录文件(完整可用)。核心逻辑:`_session_key`(取会话标识)→ `_candidates`(健康端点+权重)→ `_select`(加权 HRW)→ 写 tag。

### 4.4 启动

**pip**(从项目根启动,用裸文件名):
```bash
pip install 'litellm[proxy]'
cd litellm-affinity/
litellm --config real_config.yaml --port 4000
```
> 必须从「含 config + `session_affinity.py` 软链接」的目录启动;litellm 按 config 同目录解析 callback。

**Docker**(工作目录设为含 config 的目录):
```bash
docker run -p 4000:4000 -v $(pwd):/app -w /app \
  -e LITELLM_MASTER_KEY=sk-... \
  ghcr.io/berriai/litellm:<pinned-tag> --config real_config.yaml
```
> 生产请**固定镜像/版本 tag**,不要用 `main-latest`(见 §7 升级流程)。
> `session_affinity.py` 软链接指向 `src/`,需保证 `src/` 也在挂载内(整目录挂载即可)。

### 4.5 Claude Code 端(开发者只需这几行)

```bash
export ANTHROPIC_BASE_URL=http://<litellm-host>:4000
export ANTHROPIC_AUTH_TOKEN=sk-<litellm-key>
export ANTHROPIC_MODEL=claude-sonnet
export ANTHROPIC_SMALL_FAST_MODEL=claude-haiku   # 后台小模型也建议配同样的多端点组
```

### 4.6 在 LiteLLM UI 上配置(混合模式)

**核心限制**:hook 是自定义 Python 类,UI / 数据库**无法注册任意 Python 代码**,所以
`callbacks: session_affinity.proxy_handler_instance` 必须留在 config.yaml(ops 一次性)。
其余——**端点、tag、权重——可以全部在 UI 上管**,因为 hook 运行时读的是 `llm_router.model_list`,
它**同时包含 config.yaml 和 UI/DB 添加的模型**。

| 配什么 | 在哪配 | 谁 | UI 可配 |
|---|---|---|---|
| `callbacks`(hook) | config.yaml + 挂载 .py | ops 一次性 | ❌ |
| `enable_tag_filtering: true` | config.yaml `router_settings` | ops 一次性 | ⚠️ 部分版本有 Router Settings 页 |
| `store_model_in_db: true` + DB | config.yaml `general_settings` | ops 一次性 | ❌ |
| 各账户端点 + tag + 权重 | **UI / `/model/new` API** | 管理员日常 | ✅ |

**第一步:config.yaml(一次性,最小化)** —— `model_list` 可留空,端点都从 UI 加:

```yaml
litellm_settings:
  callbacks: session_affinity.proxy_handler_instance
router_settings:
  enable_tag_filtering: true
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  store_model_in_db: true          # 必开,否则 UI 加的模型重启即失
  # database_url: os.environ/DATABASE_URL
```
这份 config 仍需与 `session_affinity.py`(软链接到 `src/`)放在同目录、从该目录启动(见 §4.1 说明)。

**第二步:UI 上加端点**(Models → Add Model,每个缓存域一条):
- **Model Name**:所有端点填**同一个**(如 `claude-sonnet`)组成一个组
- **Provider / Model / 凭证 / Region / Base URL**:该缓存域对应的云厂商、模型与凭证
- **Advanced Settings → Tags**:`cache-domain-0`(每个缓存域**唯一**)
- **Advanced Settings → Weight**:`3`(hook 同时认 `litellm_params.weight` 与 `model_info.hrw_weight`)

**等价 `/model/new` API(字段最精确,版本无关)**:
```bash
curl -X POST http://<host>:4000/model/new \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{
    "model_name": "claude-sonnet",
    "litellm_params": {
      "model": "bedrock/us.anthropic.claude-opus-4-7",
      "aws_region_name": "us-east-1",
      "api_key": "os.environ/AWS_BEARER_TOKEN_BEDROCK",
      "tags": ["cache-domain-0"], "weight": 3
    },
    "model_info": { "hrw_weight": 3 }
  }'
```
重复添加 cache-domain-1 / cache-domain-2……。UI/API 加完**无需重启**,下一条请求 hook 即纳入加权 HRW。

> 权重来源优先级:`model_info.hrw_weight` → `litellm_params.weight` → 默认 1。
> 两处任填其一即可;tag 路由已把请求钉到单个 deployment,故 litellm 内建的 `weight` 加权在此是惰性的,可安全复用。

### 4.7 用 Application Inference Profile ARN(且保住 1h 缓存)

想用 Bedrock **application inference profile**(自定义推理配置,ARN 形如
`arn:aws:bedrock:us-east-1:<acct>:application-inference-profile/<opaque-id>`)作为端点时,
**不能**直接把 ARN 填进 `litellm_params.model`。原因:

- 这类 ARN 结尾是**不透明 id**(如 `raefjm3jcgd9`),不含模型名子串。
- litellm 判断"是否支持 1h 缓存"用的是 `is_claude_4_5_on_bedrock(model)`——它拿
  `litellm_params.model` 去价格表/名字里找 opus/sonnet 特征。传 ARN 时**找不到 → 判 False →
  litellm 把 `ttl: "1h"` 剥掉,静默降级回 5min**(端点仍能调通、定价靠 `base_model` 也对,唯独 1h 失效)。

**正确写法:`model` 与 `model_id` 分离**——`model` 填**明文模型名**让 litellm 识别能力,
`model_id` 填 **ARN** 指定实际调用目标:

```bash
curl -X POST http://<host>:4000/model/new \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{
    "model_name": "opus-47-appprofile",
    "litellm_params": {
      "model": "anthropic.claude-opus-4-7",
      "model_id": "arn:aws:bedrock:us-east-1:<acct>:application-inference-profile/<opaque-id>",
      "aws_region_name": "us-east-1"
    },
    "model_info": { "base_model": "us.anthropic.claude-opus-4-7" }
  }'
```

各字段职责(litellm 1.92.0,`converse_handler.py` 实测确认):

| 字段 | 作用 |
|---|---|
| `litellm_params.model` = 明文模型名 | 只用于**能力/定价/1h 判定**;让 `is_claude_4_5_on_bedrock` 认出是 opus → 放行 `ttl:1h` |
| `litellm_params.model_id` = ARN | **实际请求目标**;handler 取 `model_id` 后 `encode_model_id` 作为真正的 Bedrock `modelId`,即打到该 inference profile |
| `model_info.base_model` | 定价/能力回退锚点 |

> Application inference profile ARN **强制走 converse 路由**(ARN 无 provider 子串,invoke 路径无法解析)。
> converse 的 `/v1/chat/completions` usage 只给 `cache_read/creation`,**不给 5m/1h 分档**;要确认
> 命中的确实是 **1h 档**,用 `/v1/messages` 请求看 `usage.cache_creation.ephemeral_1h_input_tokens`。

**实测结论(opus-4-7 + 上述配置)**:请求命中该端点(`x-litellm-model-id` = 该 id),
`/v1/messages` 首轮 `ephemeral_1h_input_tokens` = 全部 prefix、次轮 `cache_read_input_tokens` 命中 →
**inference profile 生效 + 1h 缓存生效,且无需改 litellm 源码**。

---

## 5. 关键实现要点 / 踩过的坑

这几条是**实测才发现、看代码发现不了**的,务必保留:

1. **`/v1/messages` 与 `/v1/chat/completions` 读 tag 的键不同**
   - `/v1/chat/completions`(`acompletion`)从 `metadata["tags"]` 读;
   - `/v1/messages`(走 `_ageneric_api_call_with_fallbacks`)从 **`litellm_metadata["tags"]`** 读。
   - Claude Code 用的是 `/v1/messages`。**回调必须两个键都写**,否则只测 chat/completions 会以为正常、上线后 Claude Code 完全没粘性。
   - 由 `_TAG_KEYS = ("metadata", "litellm_metadata")` 覆盖。

2. **`_get_healthy_deployments` 是内部 API,签名/返回会随版本变**
   - 1.90.1:`_get_healthy_deployments(model, parent_otel_span)` → 返回 `(healthy, all)` 二元组。
   - 回调里做了兼容(补 `parent_otel_span=None`、取 `[0]`、兼容 async/旧签名),且 `except` 兜底:**这个调用坏了也只是退化成"无健康过滤",不会让请求失败**。

3. **尊重显式 tag**:请求若已带 `x-litellm-tags` 头或 body tags,回调不覆盖(`_has_explicit_tags`)。便于人工强制路由 / 调试。

---

## 6. 验证结果(litellm 1.90.1)

### 6.1 真实 Bedrock 缓存命中(opus-4-7,US / JP / EU 三个跨区 profile)

以下是 Bedrock 上的实测结果;多云场景的前提相同:上游支持 prompt cache,且不同端点之间存在独立缓存域。

| 步骤 | 端点 | cache_creation | cache_read | 结论 |
|---|---|---|---|---|
| 域A 首次 | US | 11057 | 0 | 写入缓存 |
| 域A 再次 | US | 0 | **11057** | ✅ 命中 |
| 域B 同 prompt | JP | 11057 | 0 | ❌ 未命中(换域 = 跨账户/跨云等价场景) |
| 同会话(回调路由) | 同一端点 ×2 | 0 | **11057** | ✅ 粘性带来持续命中 |

> 缓存命中后那段 prefix 成本约降到 **1/12**(读 0.1× vs 写 1.25×)。

### 6.2 加权路由(真实 proxy,权重 US:JP:EU = 3:2:1,600 会话)

```
dep-us  303  (50.5% vs 50.0%)
dep-jp  201  (33.5% vs 33.3%)
dep-eu   96  (16.0% vs 16.7%)
```
≈ 完美 3:2:1。

### 6.3 端点下线迁移(真实 cooldown 机制,6000 会话)

- 全健康:2970 / 2016 / 1014(3:2:1)
- **EU 下线**:1014 个会话全部迁走 → us=605 / jp=409(us 占比 **0.60 = 精确按剩余权重 3:2**);us/jp 会话 **0 扰动**
- **US(最大)下线**:jp/eu 会话 **0 扰动**

完整测试套件:**13 passed**。

---

## 7. 测试与升级流程

因为依赖了 litellm 内部 API,**升级前必须跑测试**;目标是把"内部 API 变了"在合并前变红,而不是 prod 静默降级。

三层测试(项目根目录下 `pytest` 一键跑;`pytest.ini` 已设 `pythonpath=src` / `testpaths=tests`):

| 层 | 文件 | 作用 |
|---|---|---|
| L1 纯逻辑 | `tests/test_logic.py` | 版本无关:粘性 / 加权分布 / 最小搬动 |
| L2 契约 | `tests/test_litellm_contract.py` | 对**当前版本**断言:hook 签名、`model_list` 字段、healthy 方法存在、**两个 metadata 键都被覆盖** |
| L3 行为 | `tests/test_health_filtering.py`、`tests/test_failover_weighted.py` | 真实 cooldown 驱动真实 hook:健康过滤 / 下线迁移 / 零扰动 / 加权重分配 |

端到端(可选,需启动 proxy):
- `scripts/run_distribution.py` —— 真实 proxy 上验证权重分布(极小请求,成本可忽略)。
- `scripts/run_real_test.py` —— 真实上游验证缓存命中(当前示例为 Bedrock,需 `secrets/.secrets.env` 凭证)。

**升级 runbook**:
1. 改 pin 到目标版本,隔离环境安装;
2. `pytest`:L1 必过 → L2 红了按报错改内部调用名/字段 → L3 行为过;
3. staging 放真流量,观察(见 §8 监控);
4. canary → 全量;
5. 把 L1/L2/L3 接进 CI,每次升 pin 自动跑。

---

## 8. 运维注意事项

- **调权重**:改 `model_info.hrw_weight` 即可,HRW 保证只挪动边际份额,不全员重排。
- **增/减端点**:加/删 `model_list` 条目即可;HRW 只影响相关会话。
- **监控静默降级**:回调在健康过滤失败时打 `WARNING("health-aware routing degraded ...")`。上线后盯三个信号:
  1. 该 WARNING 频率;
  2. `x-litellm-model-id` 分布是否仍贴近权重;
  3. 对话里 `cache_read_input_tokens` 是否 > 0。
  任一异常说明升级引入了回归。
- **in-flight 失败**:回调按健康选端点,但"刚选完才 429"的那一个请求仍需 `num_retries` / `fallbacks` 兜底;下一次请求回调已能看到 cooldown 而自动避开。如需彻底兜,可加一个不带 tag 的全端点组作为 `fallbacks` 目标(需实测 `enable_tag_filtering` 与 fallback 的交互)。

---

## 9. 已知限制

- 健康过滤依赖 litellm 内部 `_get_healthy_deployments`(有兜底,但升级需用 L2 契约测试守住)。
- 单端点精确匹配单 tag,故端点级容错靠回调的健康感知 + 上述 fallback,而非 tag 自身。
- 缓存边界由上游 provider 决定;本 hook 只能保证同一会话落到同一 LiteLLM deployment/tag,不能把不同上游缓存域合并成一个。
- Bedrock 场景下缓存边界通常是 `(账户, region)`;跨区 inference profile(`global.` 等)会让"按区隔离"失真,测试/生产请用 geo 明确的 profile(`us.` / `jp.` / `eu.` / `au.`)或直连区域模型。

---

## 10. 文件清单

| 路径 | 说明 |
|---|---|
| `README.md` / `README.en.md` | 中文 / 英文说明文档 |
| `src/session_affinity.py` | 生产级回调(粘性 + 加权 HRW + 健康感知) |
| `real_config.yaml` / `test_config.yaml`(项目根) | proxy 配置(真实 Bedrock / mock 测试) |
| `session_affinity.py`(根,软链接 → `src/`) | 让 litellm 在 config 同目录找到 callback |
| `tests/` | 三层测试(L1 逻辑 / L2 契约 / L3 行为) |
| `scripts/run_distribution.py` / `scripts/run_real_test.py` | 端到端验证脚本 |
| `secrets/.secrets.env`(本地,gitignore) | 凭证,`source` 使用,不入库 |
| `pytest.ini` | `pythonpath=src` / `testpaths=tests` |
