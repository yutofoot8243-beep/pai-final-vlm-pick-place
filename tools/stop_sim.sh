#!/usr/bin/env bash
# コンテナ内の Gazebo / MoveIt / エージェントを全部止める.  bash tools/stop_sim.sh [コンテナ名]
# パターンを [x]yz の形にしているのは、pkill がこのコマンド自身にマッチしないようにするため.
CONTAINER=${1:-pai_ros2}
docker exec "$CONTAINER" bash -c '
  for p in "[r]os2 launch" "[i]gn gazebo" "[m]ove_group" "[r]viz2" "[v]lm_agent" \
           "[p]arameter_bridge" "[r]obot_state_publisher" "[r]qt_image_view" "[s]pawner"; do
    pkill -9 -f "$p"
  done
  sleep 1
  ps aux | grep -E "[i]gn gazebo|[m]ove_group|[v]lm_agent" | wc -l'
echo stopped
