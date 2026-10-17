#!/usr/bin/env bash
# Windows/Mac 側で編集したコードを講義の Docker コンテナ (pai_ros2) に反映してビルドする.
#   bash tools/sync_to_docker.sh [コンテナ名]
set -euo pipefail
CONTAINER=${1:-pai_ros2}
REPO_DIR=$(cd "$(dirname "$0")/.." && (pwd -W 2>/dev/null || pwd))  # Git Bash では E:/... 形式
DEST=/home/ubuntu/pai_ws/src/pai-final-vlm-pick-place
export MSYS_NO_PATHCONV=1   # Git Bash が /home/... を Windows パスに変換しないように

docker exec -u ubuntu "$CONTAINER" rm -rf /tmp/pai-final-sync
docker cp "$REPO_DIR/." "$CONTAINER:/tmp/pai-final-sync"
docker exec -u root "$CONTAINER" bash -c \
  "rm -rf $DEST && mv /tmp/pai-final-sync $DEST && rm -rf $DEST/.git && chown -R ubuntu:ubuntu $DEST"
docker exec -u ubuntu "$CONTAINER" bash -ic \
  "cd ~/pai_ws && colcon build --symlink-install --packages-select vlm_pick_place" 2>&1 \
  | grep -v -e 'no job control' -e 'cannot set terminal'
