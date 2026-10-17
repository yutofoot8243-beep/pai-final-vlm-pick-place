"""デモ動画を MP4 (H.264) で保存するノード.

左: 斜め上からのカメラ (/side_camera/image_raw)、右: Gemini の判断を重ねた真上の映像
(/vlm/annotated_image)、上: 指示文と進捗（日本語）を並べた 1280x560 の映像を ffmpeg に流す.
映像は /vlm/annotated_image が届くたび（シミュレーション時間で 10 Hz）に1フレーム書くので、
再生速度はシミュレーション時間基準になる.

  ros2 run vlm_pick_place recorder --ros-args -p output:=/home/ubuntu/videos/demo.mp4
"""

import glob
import os
import subprocess
import threading
import time

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .vlm_agent_node import image_msg_to_bgr

PANEL_W, PANEL_H, HEADER_H = 640, 480, 80


def find_cjk_font():
    patterns = ['/usr/share/fonts/**/NotoSansCJK*.tt[cf]', '/usr/share/fonts/**/NotoSerifCJK*.tt[cf]',
                '/usr/share/fonts/**/*Gothic*.tt[cf]']
    for pattern in patterns:
        found = sorted(glob.glob(pattern, recursive=True))
        if found:
            return found[0]
    return None


class RecorderNode(Node):
    def __init__(self):
        super().__init__('vlm_recorder')
        self.declare_parameter('output', os.path.expanduser('~/videos/pai_demo.mp4'))
        self.declare_parameter('fps', 10)
        self.declare_parameter('stop_on_finish', True)   # 'task finished' から数秒後に自動で終了
        self.declare_parameter('tail_sec', 4.0)
        self.declare_parameter('title', 'Gemini Robotics-ER x CRANE+ V2')
        self.done = threading.Event()

        font_path = find_cjk_font()
        self._font_big = ImageFont.truetype(font_path, 26) if font_path else ImageFont.load_default()
        self._font_small = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()
        if not font_path:
            self.get_logger().warn('日本語フォントが見つかりません（文字化けします）')

        self._side = None
        self._instruction = ''
        self._status = 'waiting for instruction'
        self._finish_time = None
        self._frames = 0

        output = self.get_parameter('output').value
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        w, h = PANEL_W * 2, PANEL_H + HEADER_H
        fps = str(self.get_parameter('fps').value)
        self._output = output
        self._ffmpeg = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
             '-s', f'{w}x{h}', '-r', fps, '-i', '-', '-c:v', 'libx264', '-preset', 'medium',
             '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', output],
            stdin=subprocess.PIPE)

        self.create_subscription(Image, '/side_camera/image_raw', self._on_side, 10)
        self.create_subscription(Image, '/vlm/annotated_image', self._on_top, 10)
        self.create_subscription(String, '/vlm/status', self._on_status, 10)
        self.create_subscription(String, '/instruction', self._on_instruction, 10)
        self.get_logger().info(f'recording to {output}')

    def _on_side(self, msg):
        self._side = image_msg_to_bgr(msg)

    def _on_instruction(self, msg):
        self._instruction = msg.data

    def _on_status(self, msg):
        text = msg.data
        if text.startswith('instruction: '):
            self._instruction = text[len('instruction: '):]
        else:
            self._status = text
        if text == 'task finished' and self._finish_time is None:
            self._finish_time = time.monotonic()

    def _header(self, width):
        img = PILImage.new('RGB', (width, HEADER_H), (25, 25, 30))
        d = ImageDraw.Draw(img)
        title = self.get_parameter('title').value
        d.text((14, 8), f'指示: {self._instruction or "（待機中）"}', font=self._font_big,
               fill=(255, 255, 255))
        d.text((14, 46), self._status[:70], font=self._font_small, fill=(120, 220, 255))
        tw = d.textlength(title, font=self._font_small)
        d.text((width - tw - 14, 46), title, font=self._font_small, fill=(160, 160, 160))
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def _on_top(self, msg):
        if self.done.is_set():
            return
        top = cv2.resize(image_msg_to_bgr(msg), (PANEL_W, PANEL_H))
        side = (cv2.resize(self._side, (PANEL_W, PANEL_H)) if self._side is not None
                else np.zeros((PANEL_H, PANEL_W, 3), np.uint8))
        frame = np.vstack([self._header(PANEL_W * 2), np.hstack([side, top])])
        try:
            self._ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
            self._frames += 1
        except BrokenPipeError:
            self.get_logger().error('ffmpeg が終了しました')
            self.done.set()
            return
        if (self.get_parameter('stop_on_finish').value and self._finish_time is not None
                and time.monotonic() - self._finish_time > self.get_parameter('tail_sec').value):
            self.done.set()

    def close(self):
        if self._ffmpeg.stdin and not self._ffmpeg.stdin.closed:
            self._ffmpeg.stdin.close()
        self._ffmpeg.wait(timeout=60)
        fps = self.get_parameter('fps').value
        self.get_logger().info(f'saved {self._output} ({self._frames} frames, '
                               f'{self._frames / fps:.0f} s)')


def main():
    rclpy.init()
    node = RecorderNode()
    try:
        while rclpy.ok() and not node.done.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
