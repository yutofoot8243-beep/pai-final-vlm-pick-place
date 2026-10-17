import unittest

import numpy as np

from vlm_pick_place.refine import refine_pick_pixel


def scene():
    """木目色の机に、明るい上面 (60x60) と暗い側面 (下側 15px) を持つ緑ブロック."""
    img = np.full((480, 640, 3), (40, 70, 110), np.uint8)       # BGR 茶色
    img[160:175, 340:400] = (30, 120, 30)                         # 側面（暗い）
    img[100:160, 340:400] = (80, 200, 70)                         # 上面（明るい）
    return img


class TestRefine(unittest.TestCase):
    def test_point_near_edge_moves_to_top_face_center(self):
        (u, v), ok = refine_pick_pixel(scene(), 372, 156)
        self.assertTrue(ok)
        self.assertAlmostEqual(u, 369.5, delta=1)
        self.assertAlmostEqual(v, 129.5, delta=1)

    def test_point_on_side_face_moves_to_top_face(self):
        (u, v), ok = refine_pick_pixel(scene(), 370, 163)
        self.assertTrue(ok)
        self.assertAlmostEqual(v, 129.5, delta=1)

    def test_no_object_keeps_point(self):
        (u, v), ok = refine_pick_pixel(scene(), 100, 400)
        self.assertFalse(ok)
        self.assertEqual((u, v), (100.0, 400.0))


if __name__ == '__main__':
    unittest.main()
