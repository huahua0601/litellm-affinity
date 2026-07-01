#!/bin/bash
# 启动带 DB + UI 的测试 litellm(端口 4111),用隔离的 postgres(5434)。
cd /home/ec2-user/litellm-affinity || exit 1
set -a; source secrets/.secrets.env; set +a
export DATABASE_URL="postgresql://affinity:affinity@127.0.0.1:5434/litellm_affinity"
export UI_USERNAME="admin"
export UI_PASSWORD="affinity-admin"
export STORE_MODEL_IN_DB="True"
export DISABLE_SCHEMA_UPDATE="True"   # 表已用 prisma db push 建好,跳过启动迁移
export PATH="/home/ec2-user/litellm-affinity/.venv/bin:$PATH"
exec ./.venv/bin/litellm --config config_ui.yaml --port 4111
