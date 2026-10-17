"""言語指示 → Gemini Robotics-ER → MoveIt で CRANE+ V2 にピック&プレースさせるノード.

使い方:
  ros2 topic pub --once /instruction std_msgs/msg/String "data: '赤いブロックを青いトレイに置いて'"

購読:  image_raw, camera_info, /instruction
配信:  /vlm/annotated_image (動画撮影用), /vlm/status
"""

import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import tf2_ros

from .annotate import draw_decision, draw_text_banner
from .arm_commander import ArmCommander
from .refine import refine_pick_pixel
from .gemini_brain import (DEFAULT_FALLBACK_MODEL_IDS, DEFAULT_MODEL_ID, DEFAULT_SCENE_HINT,
                           GeminiBrain)
from .skills import declare_motion_params, pick_and_place, read_motion_params
from .geometry import (check_reachable, normalized_yx_to_pixel, pixel_to_table_point,
                       quaternion_to_matrix)


def image_msg_to_bgr(msg):
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(msg.encoding)
    if channels is None:
        raise ValueError(f'unsupported image encoding: {msg.encoding}')
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding == 'rgb8':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if msg.encoding == 'rgba8':
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if msg.encoding == 'bgra8':
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if msg.encoding == 'mono8':
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def bgr_to_image_msg(bgr, header):
    msg = Image()
    msg.header = header
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = 'bgr8'
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(bgr).tobytes()
    return msg


class VlmAgentNode(Node):
    def __init__(self):
        super().__init__('vlm_agent')
        p = self.declare_parameter
        p('model_id', DEFAULT_MODEL_ID)
        p('fallback_model_ids', list(DEFAULT_FALLBACK_MODEL_IDS))  # 無料枠切れ時の切替先
        p('thinking_budget', 1024)
        p('temperature', 0.5)
        p('scene_hint', DEFAULT_SCENE_HINT)  # 机上の物体の見た目の説明（ワールドを変えたら書き換える）
        p('base_frame', 'base_link')
        p('camera_frame', '')            # 空なら画像ヘッダの frame_id を使う
        p('pick_surface_z', 0.05)        # 掴む物体の上面の高さ [m]（Gemini は上面中心を指す）
        p('place_surface_z', 0.005)      # 置き場所（トレイ）上面の高さ [m]
        declare_motion_params(self)
        p('refine_pick', True)           # 掴む点を画像上でブロック上面の中心に寄せる
        p('reach_r_min', 0.14)
        p('reach_r_max', 0.30)
        p('max_steps', 8)
        p('velocity_scaling', 0.6)
        p('auto_instruction', '')        # 起動時に自動実行する指示（デモ撮影用）
        gp = self.get_parameter

        self._brain = GeminiBrain(
            model_id=gp('model_id').value,
            thinking_budget=gp('thinking_budget').value,
            temperature=gp('temperature').value,
            scene_hint=gp('scene_hint').value,
            fallback_model_ids=gp('fallback_model_ids').value,
            log=self.get_logger().warn)
        self._arm = ArmCommander(self, base_frame=gp('base_frame').value,
                                 velocity_scaling=gp('velocity_scaling').value,
                                 acceleration_scaling=gp('velocity_scaling').value)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._lock = threading.Lock()
        self._latest_image = None
        self._camera_info = None
        self._overlay = None             # (decision, step) か 文字列
        self._busy = False

        self.create_subscription(Image, 'image_raw', self._on_image, 10)
        self.create_subscription(CameraInfo, 'camera_info', self._on_camera_info, 10)
        self.create_subscription(String, '/instruction', self._on_instruction, 10)
        self._annotated_pub = self.create_publisher(Image, '/vlm/annotated_image', 10)
        self._status_pub = self.create_publisher(String, '/vlm/status', 10)

        auto = gp('auto_instruction').value
        if auto:
            threading.Thread(target=self._run_task, args=(auto,), daemon=True).start()
        self.get_logger().info(f'ready (model={self._brain.model_id}). '
                               'publish an instruction to /instruction')

    # ------------------------------------------------------------ コールバック
    def _on_camera_info(self, msg):
        self._camera_info = msg

    def _on_image(self, msg):
        try:
            bgr = image_msg_to_bgr(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        with self._lock:
            self._latest_image = (msg.header, bgr)
            overlay = self._overlay
        # ライブ映像に現在の判断を重ねて配信（rqt_image_view で録画する想定）
        if isinstance(overlay, tuple):
            bgr = draw_decision(bgr, overlay[0], overlay[1])
        elif isinstance(overlay, str):
            bgr = draw_text_banner(bgr, overlay)
        self._annotated_pub.publish(bgr_to_image_msg(bgr, msg.header))

    def _on_instruction(self, msg):
        if self._busy:
            self.get_logger().warn('busy: ignoring new instruction')
            return
        threading.Thread(target=self._run_task, args=(msg.data,), daemon=True).start()

    # ------------------------------------------------------------ 補助
    def _status(self, text, overlay=None):
        self.get_logger().info(text)
        self._status_pub.publish(String(data=text))
        if overlay is not None:
            with self._lock:
                self._overlay = overlay

    def _fresh_image(self, settle_sec=1.0):
        """アームが止まってから撮った画像を返す."""
        time.sleep(settle_sec)
        with self._lock:
            return self._latest_image

    def _to_table_point(self, header, uv, label, surface_z):
        u, v = uv
        cam_frame = self.get_parameter('camera_frame').value or header.frame_id
        tf = self._tf_buffer.lookup_transform(
            self.get_parameter('base_frame').value, cam_frame,
            rclpy.time.Time(), timeout=Duration(seconds=2.0))
        t, q = tf.transform.translation, tf.transform.rotation
        point = pixel_to_table_point(
            u, v, self._camera_info.k, np.array([t.x, t.y, t.z]),
            quaternion_to_matrix(q.x, q.y, q.z, q.w), surface_z)
        if point is None:
            raise RuntimeError(f'ray for "{label}" does not hit the table')
        return point

    # ------------------------------------------------------------ メインループ
    def _run_task(self, instruction):
        self._busy = True
        try:
            self._run_task_inner(instruction)
        except Exception as e:  # noqa: BLE001  スレッドが黙って死なないようにする
            self.get_logger().error(f'task aborted: {e!r}')
            self._status(f'ERROR: {e}', overlay=f'ERROR: {e}')
        finally:
            self._busy = False

    def _run_task_inner(self, instruction):
        gp = self.get_parameter
        self._status(f'instruction: {instruction}', overlay='Instruction received')
        if not self._arm.wait_for_server(timeout=60.0):
            raise RuntimeError('move_group (/move_action) が見つかりません')
        while self._camera_info is None or self._latest_image is None:
            self.get_logger().info('waiting for camera...', throttle_duration_sec=2.0)
            time.sleep(0.2)

        self._arm.go_named('home')
        history = []
        failures = 0
        for step in range(1, gp('max_steps').value + 1):
            header, bgr = self._fresh_image()
            _, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self._status(f'[step {step}] thinking...', overlay=f'Step {step}: Gemini is thinking...')
            t0 = time.monotonic()
            decision = self._brain.next_step(jpeg.tobytes(), instruction, history)
            self.get_logger().info(f'Gemini ({time.monotonic() - t0:.1f}s): {decision.raw}')
            self._status(f'[step {step}] {decision.say_ja or decision.reason}',
                         overlay=(decision, step))
            if decision.done:
                break

            h, w = bgr.shape[:2]
            pick_uv = normalized_yx_to_pixel(decision.pick.point_yx, w, h)
            place_uv = normalized_yx_to_pixel(decision.place.point_yx, w, h)
            if gp('refine_pick').value:
                refined_uv, ok = refine_pick_pixel(bgr, *pick_uv)
                self.get_logger().info(
                    f'refine pick: ({pick_uv[0]:.0f}, {pick_uv[1]:.0f}) -> '
                    f'({refined_uv[0]:.0f}, {refined_uv[1]:.0f})' + ('' if ok else ' (no top face found)'))
                if ok:   # 動画にも補正後の点を描く
                    decision.pick.point_yx = (refined_uv[1] / h * 1000.0, refined_uv[0] / w * 1000.0)
                    pick_uv = refined_uv
                    with self._lock:
                        self._overlay = (decision, step)
            pick_xyz = self._to_table_point(header, pick_uv, decision.pick.label,
                                            gp('pick_surface_z').value)
            place_xyz = self._to_table_point(header, place_uv, decision.place.label,
                                             gp('place_surface_z').value)
            self.get_logger().info(f'pick {decision.pick.label} @ {np.round(pick_xyz, 3)}, '
                                   f'place {decision.place.label} @ {np.round(place_xyz, 3)}')
            summary = f"pick '{decision.pick.label}' -> place '{decision.place.label}'"
            reachable = [check_reachable(xyz[:2], gp('reach_r_min').value, gp('reach_r_max').value)
                         for xyz in (pick_xyz, place_xyz)]
            if not all(ok for ok, _ in reachable):
                why = '; '.join(msg for ok, msg in reachable if not ok)
                history.append(f'{summary}: SKIPPED, robot cannot reach ({why})')
                self._status(f'[step {step}] unreachable: {why}')
                continue

            ok, msg = self._pick_and_place(pick_xyz[:2], place_xyz[:2])
            history.append(f'{summary}: {msg}')
            self._status(f'[step {step}] {msg}')
            failures = 0 if ok else failures + 1
            if failures >= 3:  # 同じ失敗を繰り返して API 枠を使い切らないように打ち切る
                self._status('stopped: 3 failures in a row', overlay='Stopped: 3 failures in a row')
                break
        else:
            self._status('max_steps reached', overlay='Stopped: max steps reached')

        self._arm.go_named('home')
        self._status('task finished')

    def _pick_and_place(self, pick_xy, place_xy):
        return pick_and_place(self._arm, read_motion_params(self), pick_xy, place_xy,
                              log=self.get_logger().info)


def main():
    rclpy.init()
    node = VlmAgentNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
