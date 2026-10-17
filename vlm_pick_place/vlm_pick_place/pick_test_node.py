"""Gemini を使わずに、座標を直接与えてピック&プレースだけを試す（把持の調整用）.

  ros2 run vlm_pick_place pick_test --ros-args --params-file <params.yaml> \
      -p pick:="[0.1414, 0.1414]" -p place:="[0.2338, 0.1350]"

座標は base_link 基準 [m]. 既定値は worlds/vlm_table.sdf の赤ブロック → 赤トレイ.
"""

import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .arm_commander import ArmCommander
from .skills import declare_motion_params, pick_and_place, read_motion_params


class PickTestNode(Node):
    def __init__(self):
        super().__init__('vlm_agent')   # params.yaml の vlm_agent セクションをそのまま読む
        declare_motion_params(self)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('velocity_scaling', 0.6)
        self.declare_parameter('pick', [0.1414, 0.1414])
        self.declare_parameter('place', [0.2338, 0.1350])
        self.done = threading.Event()
        gp = self.get_parameter
        self._arm = ArmCommander(self, base_frame=gp('base_frame').value,
                                 velocity_scaling=gp('velocity_scaling').value,
                                 acceleration_scaling=gp('velocity_scaling').value)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        log = self.get_logger().info
        try:
            if not self._arm.wait_for_server(timeout=60.0):
                self.get_logger().error('move_group が見つかりません')
                return
            pick = list(self.get_parameter('pick').value)
            place = list(self.get_parameter('place').value)
            log(f'pick {pick} -> place {place}')
            self._arm.go_named('home')
            ok, msg = pick_and_place(self._arm, read_motion_params(self), pick, place, log=log)
            log(f'result: {msg}')
        finally:
            self.done.set()


def main():
    rclpy.init()
    node = PickTestNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.done.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
