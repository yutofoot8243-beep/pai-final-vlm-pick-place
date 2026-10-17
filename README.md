# 言葉で片付けるロボットアーム — Gemini Robotics-ER × CRANE+ V2

「ブロックを同じ色のトレイに片付けて」のように日本語で指示すると、
Gemini Robotics-ER がカメラ画像を見て「次に何をどこへ動かすか」を決め、
MoveIt で CRANE+ V2 がピック&プレースを行います（Gazebo シミュレーション）。

Physical AI 応用1講座 最終課題。

## しくみ

```
/instruction (日本語の指示)
        │
        ▼
┌───────────────────────── vlm_agent ノード ─────────────────────────┐
│ 1. 観察: アームを home に戻し、真上のカメラ画像 (image_raw) を取る    │
│ 2. 思考: Gemini Robotics-ER に画像・指示・これまでの履歴を渡し、     │
│          次の1手 {pick 点, place 点} か「完了」を JSON で受け取る     │
│ 3. 変換: 掴む点を上面の中心に補正 → 画素 → tf で base_link → 机と交差 │
│ 4. 行動: MoveIt (/move_action) で 掴む→持ち上げ→運ぶ→置く           │
│ 5. 1 に戻る（失敗しても次の観察で Gemini が気づいてやり直す）        │
└────────────────────────────────────────────────────────────────────┘
        │                                  │
        ▼                                  ▼
 Gazebo 上の CRANE+ V2              /vlm/annotated_image（動画用の可視化）
```

### 講義・サンプルからの発展点

| ベース | このリポジトリで加えたこと |
|---|---|
| 第4回: Gazebo + CRANE+ V2 + MoveIt | カメラ付きモデルと独自ワールド（色ブロック3個・トレイ3枚）|
| `crane_plus_examples/color_detection` (HSV しきい値で赤だけ検出) | VLM で任意の物体・言語指示に対応。tf を使った任意平面への逆投影 |
| `crane_plus_examples/pick_and_place_tf` (置き場所固定) | 置き場所も VLM が画像から決定。多段タスクを1手ずつ計画 |
| サンプル「Gemini Robotics-ER を試す」(画像に点を打つだけ) | 点を実際のロボット動作につなげ、観察→思考→行動のループで失敗から回復 |
| Gemini の点をそのまま使う | 点が上面の端を指して 1〜2 cm ずれ、掴み損ねていた。指された点から同色の上面を flood fill で切り出し重心に補正（誤差 1 mm 以下に）|

## リポジトリ構成

```
vlm_pick_place/                  ROS 2 (ament_python) パッケージ
  vlm_pick_place/
    vlm_agent_node.py            メインノード（観察→思考→行動ループ）
    gemini_brain.py              Gemini 呼び出し・プロンプト・JSON 解析（ROS 非依存）
    geometry.py                  画素→机上座標、把持姿勢、到達判定（ROS 非依存）
    refine.py                    Gemini の掴む点をブロック上面の中心に寄せる（ROS 非依存）
    arm_commander.py             MoveGroup アクションで arm / gripper を動かす
    skills.py                    ピック&プレースの動作手順
    pick_test_node.py            Gemini なしで把持だけ試すノード
    recorder_node.py             デモ動画 (MP4) の録画
    annotate.py                  動画用オーバーレイ
  launch/vlm_pick_place.launch.py  全部入りの起動
  launch/sim.launch.py           Gazebo + CRANE+ (カメラスタンド付き) + MoveIt
  worlds/vlm_table.sdf           tools/make_world.py で生成
  config/params.yaml             高さ・オフセット・モデルなどの調整値
  test/                          単体テスト（ROS なしで実行可）
tools/
  scene_layout.py                物体配置（ワールド生成と検証で共有）
  make_world.py                  ワールド SDF の生成
  offline_check.py               Gazebo なしで Gemini の指差し精度を検証
  sync_to_docker.sh              コードを Docker コンテナへ反映してビルド
  stop_sim.sh                    コンテナ内のシミュレーションを全停止
```

## セットアップ（第4回の Docker 環境）

講義の `airobotbook/ros2-desktop-ai-robot-book-humble` コンテナ（名前 `pai_ros2`）で動かします。
この中の crane_plus は古い版なので、ワールドとカメラは `launch/sim.launch.py` と
`worlds/vlm_table.sdf` で自前で用意しています（crane_plus 側の変更は不要）。

```bash
# --- Windows / Mac 側（このリポジトリのフォルダで）---
docker start pai_ros2
bash tools/sync_to_docker.sh            # コードをコンテナの ~/pai_ws/src にコピーしてビルド

# --- コンテナ側（ブラウザで http://localhost:6080 → ターミナル）---
pip3 install --user google-genai        # 最初の1回だけ
echo 'export GEMINI_API_KEY=ここにキー' >> ~/.bashrc   # 最初の1回だけ（Google AI Studio で発行）
echo 'source ~/pai_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

コードを書き換えたら、もう一度 `bash tools/sync_to_docker.sh` を実行します。

## 実行（コンテナのデスクトップで）

```bash
# Gazebo + MoveIt + VLM エージェント + 可視化ウィンドウを起動（15秒後にエージェントが立ち上がる）
ros2 launch vlm_pick_place vlm_pick_place.launch.py

# 別ターミナルから指示を送る
ros2 topic pub --once /instruction std_msgs/msg/String "data: 'ブロックを同じ色のトレイに片付けて'"
ros2 topic pub --once /instruction std_msgs/msg/String "data: '一番右のブロックを赤いトレイに置いて'"

# 起動と同時に実行する（動画撮影向け）
ros2 launch vlm_pick_place vlm_pick_place.launch.py auto_instruction:='ブロックを同じ色のトレイに片付けて'
```

### デモ動画を録画する

```bash
ros2 launch vlm_pick_place vlm_pick_place.launch.py record:=true video:=$HOME/videos/demo.mp4     auto_instruction:='ブロックを同じ色のトレイに片付けて'
```

左に斜めカメラ、右に Gemini の判断を重ねた真上カメラ、上に指示と進捗（日本語）を並べた
1280x560 の MP4 (H.264) を保存し、タスクが終わると自動で閉じます（シミュレーション時間で 10 fps）。
倍速版は `ffmpeg -i demo.mp4 -vf setpts=0.5*PTS -r 20 demo_2x.mp4` で作れます（動画内に倍速を明記すること）。

進捗は `ros2 topic echo /vlm/status`、判断の可視化は rqt_image_view（`/vlm/annotated_image`）で見られます。
止めるときは Ctrl+C。ゾンビが残ったら Windows 側で `bash tools/stop_sim.sh`。

### Gemini の無料枠について

`gemini-robotics-er-2-preview` の無料枠は **1日20リクエスト**（1回の片付けで 4〜5 リクエスト）。
使い切ると自動で `fallback_model_ids`（`gemini-3.5-flash` など、枠はモデルごとに別）に切り替わります。
動画撮影の日は事前のテストを控えるか、Google AI Studio で課金を有効にしておくと安心です。

## 調整のポイント

まず Gazebo なしで Gemini 部分だけ確認できます（Windows/Mac でも可）:

```bash
pip install google-genai pillow numpy
python tools/offline_check.py                       # 合成画像で指差し誤差 [mm] を表示
python tools/offline_check.py --instruction "ブロックを同じ色のトレイに片付けて"
python tools/offline_check.py --image gazebo_camera.png --instruction "..."   # 実際のカメラ画像で
```

把持だけを Gemini を使わずに試せます（API 枠を消費しない）:

```bash
ros2 launch vlm_pick_place vlm_pick_place.launch.py use_agent:=false
ros2 run vlm_pick_place pick_test --ros-args   --params-file $(ros2 pkg prefix vlm_pick_place)/share/vlm_pick_place/config/params.yaml   -p use_sim_time:=true -p pick:="[0.1414, 0.1414]" -p place:="[0.2338, 0.1350]"
```

Gazebo で掴めないときは `config/params.yaml` を調整します:

| 症状 | 調整するパラメータ |
|---|---|
| 掴みに行く途中で隣の物体を倒す | `approach_z` を高く |
| 物体の手前/奥で空振りする | `gripper_offset`（link4 から把持点までの距離, 既定 0.13） |
| 上から押しつぶす/下を掴む | `grasp_z` |
| 掴めているのに滑り落ちる | `gripper_close_deg` を大きく |
| `too far` / `too close` で飛ばされる | 物体配置（`tools/scene_layout.py` → `make_world.py`）か `reach_r_*` |
| Gemini が遅い | `thinking_budget` を小さく（0 で思考なし） |

座標変換は `ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame` で確認できます。

## テスト

```bash
cd vlm_pick_place && python -m pytest test     # ROS なしで動く
```

## ライセンス

Apache-2.0。CRANE+ V2 のモデル・サンプルは [rt-net/crane_plus](https://github.com/rt-net/crane_plus)（Apache-2.0）を利用しています。
