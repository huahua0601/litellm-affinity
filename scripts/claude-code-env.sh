# 用法:在一个新终端里  source scripts/claude-code-env.sh  然后运行  claude
# 把当前 shell 的 Claude Code 指向测试 litellm(4111),不改全局 settings.json。
# 在 EC2 本机跑用 localhost;从外部跑把 BASE_URL 换成 http://52.13.186.133:4111(需安全组放行 4111 或 SSH 隧道)。

unset ANTHROPIC_API_KEY                            # 避免与 AUTH_TOKEN 冲突(双 token 警告 / 认证错乱)
export ANTHROPIC_BASE_URL="http://localhost:4111"
export ANTHROPIC_AUTH_TOKEN="sk-affinity-test"     # litellm master key
export ANTHROPIC_MODEL="claude-aff"                # 主模型:opus-4-7,多端点加权粘性
export ANTHROPIC_SMALL_FAST_MODEL="claude-fast"    # 后台小模型:haiku(便宜)

echo "Claude Code -> $ANTHROPIC_BASE_URL  model=$ANTHROPIC_MODEL  fast=$ANTHROPIC_SMALL_FAST_MODEL"
