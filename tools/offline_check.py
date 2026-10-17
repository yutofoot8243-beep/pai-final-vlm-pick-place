"""Gazebo なしで「Gemini の指差し → 机上座標」の精度を確かめるツール.

worlds/vlm_table.sdf と同じ配置を、CRANE+ のカメラ（URDF の値）から見た画像として合成し、
Gemini Robotics-ER に見せる. 返ってきた点を逆投影して、正解座標との誤差 [mm] を表示する.

  python tools/offline_check.py                      # 全物体の指差し精度
  python tools/offline_check.py --instruction "ブロックを同じ色のトレイに片付けて"
  python tools/offline_check.py --image screenshot.png --instruction "..."   # 実画像で試す
"""

import argparse
import io
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vlm_pick_place'))
from vlm_pick_place.gemini_brain import GeminiBrain  # noqa: E402
from vlm_pick_place.geometry import normalized_yx_to_pixel, pixel_to_table_point  # noqa: E402

W, H = 640, 480
F = W / (2 * math.tan(math.radians(70) / 2))
K = [F, 0, W / 2, 0, F, H / 2, 0, 0, 1]
CAM_T = np.array([0.10626, 0.10253, 0.46])
# camera_color_optical_joint: rpy = (0, pi, pi/2) → R = Rz(pi/2) Ry(pi)
CAM_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

sys.path.insert(0, os.path.dirname(__file__))
from scene_layout import (COLORS, CUBE_SIZE, CUBES, TRAY_INNER, TRAY_SIZE, TRAY_THICK,  # noqa: E402
                          TRAYS)

# (name, color, r, deg, top_z)
OBJECTS = ([(f'{c} cube', c, r, d, CUBE_SIZE) for c, r, d in CUBES]
           + [(f'{c} tray', c, r, d, TRAY_THICK) for c, r, d in TRAYS])


def project(p):
    pc = CAM_R.T @ (np.asarray(p) - CAM_T)
    return K[0] * pc[0] / pc[2] + K[2], K[4] * pc[1] / pc[2] + K[5]


def center(r, deg, z):
    return np.array([r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg)), z])


def render():
    img = Image.new('RGB', (W, H), (150, 120, 90))   # 木の机っぽい色
    d = ImageDraw.Draw(img)
    base = [project((0.04 * math.cos(a), 0.04 * math.sin(a), 0.07))
            for a in np.linspace(0, 2 * math.pi, 24)]
    d.polygon(base, fill=(40, 40, 40))               # ロボットの根元
    def square(c, size, yaw):
        pts = []
        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            ox, oy = sx * size / 2, sy * size / 2
            pts.append(project(c + [ox * math.cos(yaw) - oy * math.sin(yaw),
                                    ox * math.sin(yaw) + oy * math.cos(yaw), 0]))
        return pts

    rgb = {k: tuple(int(v * 255) for v in c) for k, c in COLORS.items()}
    for name, color, r, deg, top_z in sorted(OBJECTS, key=lambda o: o[4]):
        c = center(r, deg, top_z)
        if 'tray' in name:
            inner = tuple(int((v * 0.7 + 0.15) * 255) for v in COLORS[color])
            d.polygon(square(c, TRAY_SIZE, math.radians(deg)), fill=(242, 242, 242),
                      outline=(120, 120, 120))
            d.polygon(square(c, TRAY_INNER, math.radians(deg)), fill=inner)
        else:
            d.polygon(square(c, CUBE_SIZE, 0.0), fill=rgb[color], outline=(30, 30, 30))
    return img


def to_jpeg(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return buf.getvalue()


def back_project(point_yx, top_z):
    u, v = normalized_yx_to_pixel(point_yx, W, H)
    return pixel_to_table_point(u, v, K, CAM_T, CAM_R, top_z)


def mark(d, point_yx, color, label):
    u, v = normalized_yx_to_pixel(point_yx, W, H)
    d.ellipse([u - 7, v - 7, u + 7, v + 7], outline=color, width=3)
    d.text((u + 9, v - 6), label, fill=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instruction', default='')
    ap.add_argument('--image', default='', help='合成画像の代わりに使う画像')
    ap.add_argument('--out', default='offline_check.png')
    ap.add_argument('--thinking-budget', type=int, default=1024)
    ap.add_argument('--model', default=None, help='既定: gemini_brain.DEFAULT_MODEL_ID')
    args = ap.parse_args()

    img = Image.open(args.image).convert('RGB').resize((W, H)) if args.image else render()
    kwargs = {'model_id': args.model} if args.model else {}
    brain = GeminiBrain(thinking_budget=args.thinking_budget, **kwargs)
    jpeg = to_jpeg(img)
    d = ImageDraw.Draw(img)

    if args.instruction:
        decision = brain.next_step(jpeg, args.instruction, [])
        print('Gemini:', decision.raw)
        if not decision.done:
            for tgt, z, col in ((decision.pick, 0.05, (0, 255, 0)), (decision.place, 0.005, (255, 0, 255))):
                p = back_project(tgt.point_yx, z)
                print(f'  {tgt.label:12s} -> base_link xy = ({p[0]:.3f}, {p[1]:.3f}) m')
                mark(d, tgt.point_yx, col, tgt.label)
    else:
        errors = []
        for name, color, r, deg, top_z in OBJECTS:
            found = brain.point(jpeg, f'the {name}')
            if not found:
                print(f'{name:12s}: not found')
                continue
            p = back_project(found[0].point_yx, top_z)
            if args.image:  # 実画像では正解が分からないので座標だけ表示
                print(f'{name:12s}: base_link xy = ({p[0]:.3f}, {p[1]:.3f}) m')
            else:
                err = np.linalg.norm(p[:2] - center(r, deg, top_z)[:2]) * 1000
                errors.append(err)
                print(f'{name:12s}: error {err:5.1f} mm  (point {found[0].point_yx})')
            mark(d, found[0].point_yx, (255, 255, 0), name)
        if errors and not args.image:
            print(f'mean error {np.mean(errors):.1f} mm, max {np.max(errors):.1f} mm '
                  '(把持の許容はおよそ 15 mm)')
    img.save(args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
