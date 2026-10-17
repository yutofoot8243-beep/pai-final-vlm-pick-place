"""MoveIt (move_group の MoveGroup アクション) で CRANE+ V2 を動かす.

Humble では moveit_py が標準で入っていないため、move_group ノードの
/move_action に Python から直接ゴールを送る.
手先の位置姿勢ゴールは、先に /compute_ik（衝突チェック付き）で関節角に直してから
関節空間のゴールとして渡す（4自由度アームでは位置姿勢拘束のままだと OMPL が
ゴール状態をサンプリングできないことがあったため）.
ブロッキングで呼ぶので、Executor を回しているスレッドとは別のスレッドから使うこと.
"""

import math
import time

from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (BoundingVolume, Constraints, JointConstraint,
                             MoveItErrorCodes, OrientationConstraint, PositionConstraint,
                             PositionIKRequest)
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from shape_msgs.msg import SolidPrimitive

from .geometry import gripper_pose_for_target, quaternion_from_rpy

ARM_JOINTS = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4']
HAND_JOINT = 'crane_plus_joint_hand'

# crane_plus_moveit_config/config/crane_plus.srdf の group_state
NAMED_POSES = {
    'home': [0.0, -1.16, -2.01, -0.73],
    'vertical': [0.0, 0.0, 0.0, 0.0],
}


class ArmCommander:
    def __init__(self, node, base_frame='base_link', ee_link='crane_plus_link4',
                 velocity_scaling=0.6, acceleration_scaling=0.6):
        self._node = node
        self._log = node.get_logger()
        self.base_frame = base_frame
        self.ee_link = ee_link
        self.velocity_scaling = velocity_scaling
        self.acceleration_scaling = acceleration_scaling
        group = ReentrantCallbackGroup()
        self._client = ActionClient(node, MoveGroup, 'move_action', callback_group=group)
        self._ik_client = node.create_client(GetPositionIK, 'compute_ik', callback_group=group)

    # ---------------------------------------------------------------- 低レベル
    @staticmethod
    def _wait(future, timeout):
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    def wait_for_server(self, timeout=60.0):
        return (self._client.wait_for_server(timeout_sec=timeout)
                and self._ik_client.wait_for_service(timeout_sec=timeout))

    def compute_ik(self, xyz, rpy, timeout=30.0):
        """link4 の目標位置姿勢を満たす arm の関節角 (4要素) を返す. 解けなければ None."""
        req = GetPositionIK.Request()
        ik = PositionIKRequest()
        ik.group_name = 'arm'
        ik.ik_link_name = self.ee_link
        ik.avoid_collisions = True
        ik.robot_state.is_diff = True   # 現在の関節角を初期値に使う
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.pose.position = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        qx, qy, qz, qw = quaternion_from_rpy(*rpy)
        pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        ik.pose_stamped = pose
        ik.timeout.sec = 1
        req.ik_request = ik
        future = self._ik_client.call_async(req)
        if not self._wait(future, timeout):
            self._log.error('compute_ik timed out')
            return None
        res = future.result()
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self._log.debug(f'no IK solution for {[round(v, 3) for v in xyz]} '
                            f'(code {res.error_code.val})')
            return None
        js = res.solution.joint_state
        positions = dict(zip(js.name, js.position))
        return [positions[name] for name in ARM_JOINTS]

    # シミュレーションが重いと実時間より大幅に遅くなるので、待ち時間は長めにとる
    def _execute(self, group, goal_constraints, path_constraints=None, timeout=180.0,
                 quiet=False):
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = group
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.acceleration_scaling
        req.start_state.is_diff = True  # 現在の状態から計画する
        req.workspace_parameters.header.frame_id = self.base_frame
        req.workspace_parameters.min_corner.x = -1.0
        req.workspace_parameters.min_corner.y = -1.0
        req.workspace_parameters.min_corner.z = -1.0
        req.workspace_parameters.max_corner.x = 1.0
        req.workspace_parameters.max_corner.y = 1.0
        req.workspace_parameters.max_corner.z = 1.0
        req.goal_constraints = [goal_constraints]
        if path_constraints is not None:
            req.path_constraints = path_constraints
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        send_future = self._client.send_goal_async(goal)
        if not self._wait(send_future, timeout):
            self._log.error(f'[{group}] goal send timed out')
            return False
        handle = send_future.result()
        if not handle.accepted:
            self._log.error(f'[{group}] goal rejected')
            return False
        result_future = handle.get_result_async()
        if not self._wait(result_future, timeout):
            if not quiet:
                self._log.error(f'[{group}] execution timed out')
            return False
        code = result_future.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self._log.warn(f'[{group}] MoveIt error code {code}')
            return False
        return True

    @staticmethod
    def _joint_constraints(names, values, tolerance=0.01):
        c = Constraints()
        for name, value in zip(names, values):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = tolerance
            jc.tolerance_below = tolerance
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return c

    def _pose_constraints(self, xyz, rpy, position_tolerance=0.003, orientation_tolerance=0.05):
        c = Constraints()

        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.ee_link
        region = BoundingVolume()
        region.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE,
                                            dimensions=[position_tolerance])]
        region.primitive_poses = [Pose(position=Point(x=xyz[0], y=xyz[1], z=xyz[2]),
                                       orientation=Quaternion(w=1.0))]
        pc.constraint_region = region
        pc.weight = 1.0
        c.position_constraints.append(pc)

        qx, qy, qz, qw = quaternion_from_rpy(*rpy)
        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.ee_link
        oc.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        oc.absolute_x_axis_tolerance = orientation_tolerance
        oc.absolute_y_axis_tolerance = orientation_tolerance
        oc.absolute_z_axis_tolerance = orientation_tolerance
        oc.weight = 1.0
        c.orientation_constraints.append(oc)
        return c

    def _arm_path_constraints(self):
        # pick_and_place_tf.cpp と同じく、肘が下に折れ込む解などを避ける
        c = Constraints()
        for name, center_deg, tol_deg in (('crane_plus_joint1', 0.0, 100.0),
                                          ('crane_plus_joint3', -90.0, 90.0)):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = math.radians(center_deg)
            jc.tolerance_above = math.radians(tol_deg)
            jc.tolerance_below = math.radians(tol_deg)
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return c

    # ---------------------------------------------------------------- 高レベル
    def go_named(self, name):
        return self._execute('arm', self._joint_constraints(ARM_JOINTS, NAMED_POSES[name]))

    def move_pose(self, xyz, rpy):
        joints = self.compute_ik(xyz, rpy)
        if joints is None:
            return False
        return self._execute('arm', self._joint_constraints(ARM_JOINTS, joints))

    def move_pose_constrained(self, xyz, rpy):
        """位置姿勢拘束のまま MoveIt に渡す版（比較・参考用）."""
        return self._execute('arm', self._pose_constraints(xyz, rpy),
                             self._arm_path_constraints())

    def gripper(self, angle_rad):
        # 物体を掴むと目標角に届かず完了が返らないことがあるので、一定時間で次へ進む
        return self._execute('gripper', self._joint_constraints([HAND_JOINT], [angle_rad], 0.05),
                             timeout=30.0, quiet=True)

    def move_gripper_to(self, target_xy, z, gripper_offset, pitch_deg=90.0):
        xyz, rpy = gripper_pose_for_target(target_xy, z, gripper_offset, pitch_deg)
        return self.move_pose(xyz, rpy)

    def can_reach(self, target_xy, z, gripper_offset, pitch_deg=90.0):
        xyz, rpy = gripper_pose_for_target(target_xy, z, gripper_offset, pitch_deg)
        return self.compute_ik(xyz, rpy) is not None
