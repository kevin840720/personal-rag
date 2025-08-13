#!/bin/bash

set -eo pipefail  # 遇到錯誤立即退出，pipe 失敗也會退出
trap 'echo "腳本在第 $LINENO 行失敗"; exit 1' ERR

# 獲取 Elasticsearch 的 Container 名稱
SERVICE_NAME="elasticsearch"
CONTAINER_NAME=$(grep -E "container_name:\s*.*${SERVICE_NAME}.*" docker-compose.yaml | sed -E 's/.*container_name:\s*//')

# 沒有發現 container 名稱包含 elasticsearch
if [ -z "$CONTAINER_NAME" ]; then
    echo "找不到 $SERVICE_NAME 容器，請確認 docker-compose.yaml 設定。"
    exit 1
fi
# 發現多個 container 名稱包含 elasticsearch
if [ "$(echo "$CONTAINER_NAME" | wc -l)" -gt 1 ]; then
    echo "找到多個符合條件的容器："
    echo "$CONTAINER_NAME"
    echo "請確認 pattern 或 service name 唯一後再執行。"
    exit 1
fi


# 當資料夾不存在，或使用者傳入 `--recreate` 參數時，重建 folder
if [ ! -d "./.database/elastic/data" ] || [ "$1" == "--recreate" ]; then
    echo "刪除並重建 Elasticsearch 資料夾..."

    sudo rm -rf ./.database/elastic/data/
    sudo rm -rf ./.database/elastic/logs/

    sudo mkdir -p ./.database/elastic/data
    sudo mkdir -p ./.database/elastic/logs
fi

sudo chmod -R 777 ./.database/elastic/data/
sudo chmod -R 777 ./.database/elastic/logs/

sudo chown -R 1000:1000 ./.database/elastic/data/
sudo chown -R 1000:1000 ./.database/elastic/logs/

# 啟動容器
docker compose up -d

# 等待 30 秒確保 Elasticsearch 啟動
sleep 30

# 安裝 IK 分詞插件
docker exec -it "$CONTAINER_NAME" bin/elasticsearch-plugin install -b https://get.infini.cloud/elasticsearch/analysis-ik/8.17.2

# 重啟 Elasticsearch 容器
docker restart "$CONTAINER_NAME"
