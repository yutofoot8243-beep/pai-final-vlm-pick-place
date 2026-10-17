import math
import unittest

import numpy as np

from vlm_pick_place.geometry import (check_reachable, gripper_pose_for_target,
                                     normalized_yx_to_pixel, pixel_to_table_point,
                                     quaternion_from_rpy, quaternion_to_matrix)


def rpy_matrix(roll, pitch, yaw):
    """URDF の rpy 規約: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


# crane_plus_description/urdf/camera.urdf.xacro と crane_plus.gazebo.xacro の値
CAM_T = np.array([0.10626, 0.10253, 0.46])
CAM_R = rpy_matrix(0.0, math.pi, math.pi / 2)
W, H = 640, 480
F = W / (2 * math.tan(math.radians(70) / 2))
K = [F, 0, W / 2, 0, F, H / 2, 0, 0, 1]


class TestGeometry(unittest.TestCase):
    def test_normalized_to_pixel(self):
        self.assertEqual(normalized_yx_to_pixel([500, 250], W, H), (160.0, 240.0))

    def test_principal_point_hits_table_below_camera(self):
        p = pixel_to_table_point(W / 2, H / 2, K, CAM_T, CAM_R, 0.0)
        np.testing.assert_allclose(p, [CAM_T[0], CAM_T[1], 0.0], atol=1e-9)

    def test_plane_height_shrinks_offset(self):
        # 高い平面（ブロック上面）ほどカメラに近いので、同じ画素でも光軸からのずれが小さい
        low = pixel_to_table_point(600, 50, K, CAM_T, CAM_R, 0.0)
        high = pixel_to_table_point(600, 50, K, CAM_T, CAM_R, 0.05)
        self.assertAlmostEqual(high[2], 0.05)
        ratio = np.linalg.norm(high[:2] - CAM_T[:2]) / np.linalg.norm(low[:2] - CAM_T[:2])
        self.assertAlmostEqual(ratio, (0.46 - 0.05) / 0.46, places=6)

    def test_world_objects_are_visible_and_reachable(self):
        # worlds/vlm_table.sdf に置いた物体が画像内に写り、手が届く範囲にあることを確認
        inv = CAM_R.T
        for r, deg in [(0.20, 45), (0.20, 15), (0.20, -15), (0.27, 75), (0.27, 30), (0.27, -25)]:
            p = np.array([r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg)), 0.0])
            pc = inv @ (p - CAM_T)
            u = K[0] * pc[0] / pc[2] + K[2]
            v = K[4] * pc[1] / pc[2] + K[5]
            self.assertTrue(20 < u < W - 20 and 20 < v < H - 20, f'{r},{deg} -> ({u:.0f},{v:.0f})')
            self.assertTrue(check_reachable(p[:2], 0.14, 0.30)[0])

    def test_quaternion_matches_urdf_rpy(self):
        for rpy in [(0, math.pi / 2, 0.3), (0.1, -0.2, 1.0), (0, math.pi / 2, -math.pi / 4)]:
            np.testing.assert_allclose(quaternion_to_matrix(*quaternion_from_rpy(*rpy)),
                                       rpy_matrix(*rpy), atol=1e-9)

    def test_gripper_pose_matches_pick_and_place_tf(self):
        (x, y, z), (roll, pitch, yaw) = gripper_pose_for_target((0.2, 0.0), 0.04)
        self.assertAlmostEqual(x, 0.07)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(pitch, math.pi / 2)
        (x, y, _), (_, _, yaw) = gripper_pose_for_target((0.0, 0.2), 0.04)
        self.assertAlmostEqual(y, 0.07)
        self.assertAlmostEqual(yaw, math.pi / 2)

    def test_tilted_gripper_keeps_grasp_point(self):
        # どの pitch でも、link4 から把持方向に gripper_offset 進んだ点が目標になる
        for pitch in (90.0, 110.0, 130.0):
            (x, y, z), (_, p, yaw) = gripper_pose_for_target((0.12, 0.09), 0.04, 0.13, pitch)
            axis = np.array([math.sin(p) * math.cos(yaw), math.sin(p) * math.sin(yaw), math.cos(p)])
            np.testing.assert_allclose(np.array([x, y, z]) + 0.13 * axis, [0.12, 0.09, 0.04],
                                       atol=1e-9)

    def test_reachability(self):
        self.assertFalse(check_reachable((0.05, 0.0), 0.14, 0.30)[0])
        self.assertFalse(check_reachable((0.4, 0.0), 0.14, 0.30)[0])
        self.assertFalse(check_reachable((-0.2, 0.05), 0.14, 0.30)[0])
        self.assertTrue(check_reachable((0.2, 0.05), 0.14, 0.30)[0])


if __name__ == '__main__':
    unittest.main()
