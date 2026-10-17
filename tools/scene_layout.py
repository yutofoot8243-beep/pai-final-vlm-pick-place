"""机上の物体配置（ワールド生成とオフライン検証で共有）.

座標は CRANE+ の base_link 基準の極座標 (r [m], 方位角 [deg]).
- ブロック: r = 0.20 （pick_and_place_tf.cpp のサンプル位置 r≈0.21 付近）
- トレイ:   r = 0.27 （ブロックより外側。上から置くので内側のブロックに当たらない）
トレイの色はブロックとわざと位置をずらしてある（「同じ色に片付けて」で推論が要る）.
"""

import math

TABLE_Z = 1.015          # base_link のワールド高さ（crane_plus_with_table.launch.py の -z）
CUBE_SIZE = 0.05
TRAY_SIZE = 0.08         # 白い縁
TRAY_INNER = 0.06        # 色付きの内側
TRAY_THICK = 0.005

COLORS = {  # RGB 0-1
    'red': (0.85, 0.10, 0.10),
    'green': (0.10, 0.70, 0.15),
    'blue': (0.10, 0.25, 0.90),
}

# カメラ: crane_plus_description/urdf/camera.urdf.xacro の camera_joint と同じ位置（base_link 基準）
CAMERA_XYZ = (0.10626, 0.10253, 0.46)
CAMERA_HFOV_DEG = 70.0
CAMERA_SIZE = (640, 480)

# 動画用の斜めカメラ（base_link 基準の位置と、注視点）. ロボットの右手前から見下ろす
SIDE_CAMERA_XYZ = (0.62, -0.42, 0.44)
SIDE_CAMERA_LOOK_AT = (0.10, 0.05, 0.08)
SIDE_CAMERA_HFOV_DEG = 60.0

CUBES = [('red', 0.20, 45), ('green', 0.20, 15), ('blue', 0.20, -15)]
TRAYS = [('blue', 0.27, 75), ('red', 0.27, 30), ('green', 0.27, -25)]


def polar_to_xy(r, deg):
    return r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg))
