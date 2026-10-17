"""ROSに依存しない幾何計算（単体テスト可能）.

座標系の約束:
- 画像座標: u = 右向き[px], v = 下向き[px]
- Gemini Robotics-ER の点: [y, x] を 0-1000 に正規化した値
- カメラ光学座標系: z = 前方（視線方向）, x = 右, y = 下
- base_link: CRANE+ V2 の根元。z = 上向き。机の天板が z = 0
"""

import math

import numpy as np


def normalized_yx_to_pixel(point_yx, width, height):
    """Gemini の [y, x] (0-1000) を画素座標 (u, v) に変換する."""
    y, x = float(point_yx[0]), float(point_yx[1])
    return x / 1000.0 * width, y / 1000.0 * height


def pixel_to_ray(u, v, k):
    """CameraInfo.k (3x3 を平坦化した9要素) を使い、画素を光学座標系の視線ベクトルに変換する."""
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    return np.array([(u - cx) / fx, (v - cy) / fy, 1.0])


def quaternion_to_matrix(x, y, z, w):
    """単位クォータニオンを 3x3 回転行列に変換する."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quaternion_from_rpy(roll, pitch, yaw):
    """tf2::Quaternion::setRPY と同じ規約 (固定軸 X-Y-Z) で (x, y, z, w) を返す."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def intersect_ray_with_plane_z(origin, direction, plane_z):
    """origin + t * direction (t > 0) と平面 z = plane_z の交点. 交わらなければ None."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if abs(direction[2]) < 1e-9:
        return None
    t = (plane_z - origin[2]) / direction[2]
    if t <= 0:
        return None
    return origin + t * direction


def pixel_to_table_point(u, v, k, cam_translation, cam_rotation, plane_z):
    """画素 (u, v) が指す机上の点を base_link 座標で返す.

    cam_translation / cam_rotation は base_link から見たカメラ光学座標系の位置・姿勢
    (tf の lookup_transform(base_link, camera_optical_frame) の結果).
    color_detection.cpp はカメラ高さを固定値で掛けていたが、ここでは tf を使って
    任意の高さの平面と交差させるので、カメラ位置を変えても動く.
    """
    ray_cam = pixel_to_ray(u, v, k)
    ray_base = np.asarray(cam_rotation) @ ray_cam
    return intersect_ray_with_plane_z(cam_translation, ray_base, plane_z)


def gripper_pose_for_target(target_xy, z, gripper_offset=0.13, pitch_deg=90.0):
    """把持点 (target_xy, 高さ z) を掴むための link4 の目標位置と RPY を返す.

    pitch 90° はグリッパー水平（crane_plus_examples/pick_and_place_tf.cpp と同じで、
    link4 を物体から gripper_offset だけ根元側に引く）. pitch を大きくすると
    グリッパーが下を向き、根元に近い物体にも届くようになる.
    """
    x, y = float(target_xy[0]), float(target_xy[1])
    theta = math.atan2(y, x)
    pitch = math.radians(pitch_deg)
    r = math.hypot(x, y) - gripper_offset * math.sin(pitch)
    position = (r * math.cos(theta), r * math.sin(theta), z - gripper_offset * math.cos(pitch))
    rpy = (0.0, pitch, theta)
    return position, rpy


def check_reachable(target_xy, r_min, r_max, yaw_limit_deg=100.0):
    """根元からの水平距離と方位角で、届くかどうかを判定する. (ok, 理由) を返す."""
    x, y = float(target_xy[0]), float(target_xy[1])
    r = math.hypot(x, y)
    yaw = math.degrees(math.atan2(y, x))
    if r < r_min:
        return False, f'too close (r={r:.3f} m < {r_min:.3f} m)'
    if r > r_max:
        return False, f'too far (r={r:.3f} m > {r_max:.3f} m)'
    if abs(yaw) > yaw_limit_deg:
        return False, f'out of joint1 range (yaw={yaw:.0f} deg)'
    return True, f'ok (r={r:.3f} m, yaw={yaw:.0f} deg)'
