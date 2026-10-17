"""ピック&プレースの動作手順（エージェントとテスト用ノードで共通）."""

import math

# config/params.yaml で上書きできる動作パラメータとその既定値
MOTION_PARAMS = {
    'approach_z': 0.10,         # 掴む前に真上で向きを合わせる高さ（隣の物体をなぎ払わない）
    'grasp_z': 0.04,            # 掴むときの link4 の高さ
    'place_z': 0.06,            # 置くときの link4 の高さ
    'lift_z': 0.12,             # 持ち上げ・移動時の link4 の高さ
    'gripper_offset': 0.13,     # link4 から把持点までの水平距離
    'gripper_open_deg': -30.0,
    'gripper_close_deg': 10.0,
}


def declare_motion_params(node):
    for name, default in MOTION_PARAMS.items():
        node.declare_parameter(name, default)


def read_motion_params(node):
    return {name: node.get_parameter(name).value for name in MOTION_PARAMS}


PITCH_CANDIDATES_DEG = (90.0, 110.0, 130.0)   # 水平 → 下向き. 根元に近い物体ほど下向きが必要


def choose_pitch(arm, xy, heights, offset):
    """全ての高さで IK が解ける最初のグリッパー角度を返す. 無ければ None."""
    for pitch in PITCH_CANDIDATES_DEG:
        if all(arm.can_reach(xy, z, offset, pitch) for z in heights):
            return pitch
    return None


def pick_and_place(arm, cfg, pick_xy, place_xy, log=print):
    """(成功したか, 履歴用メッセージ) を返す. 掴めたかどうかはカメラで確認する."""
    offset = cfg['gripper_offset']
    open_rad = math.radians(cfg['gripper_open_deg'])
    close_rad = math.radians(cfg['gripper_close_deg'])

    # 動く前に IK で届く姿勢を選んでおく（途中で行き詰まって物体を落とさないように）.
    # 低い高さで joint1 を回すと隣の物体に当たるので、まず真上 (approach_z) で向きを合わせる
    pick_pitch = choose_pitch(arm, pick_xy, (cfg['approach_z'], cfg['grasp_z'], cfg['lift_z']),
                              offset)
    pick_heights = (cfg['approach_z'], cfg['grasp_z'])
    if pick_pitch is None:   # 真上からは無理でも、直接なら掴めることがある
        pick_pitch = choose_pitch(arm, pick_xy, (cfg['grasp_z'],), offset)
        pick_heights = (cfg['grasp_z'],)
    if pick_pitch is None:
        return False, 'FAILED: the arm cannot reach the object (no IK solution)'
    place_pitch = choose_pitch(arm, place_xy, (cfg['lift_z'], cfg['place_z']), offset)
    if place_pitch is None:
        return False, 'FAILED: the arm cannot reach the place location (no IK solution)'
    log(f'gripper pitch: pick {pick_pitch:.0f} deg, place {place_pitch:.0f} deg')

    arm.gripper(open_rad)
    for z in pick_heights:
        if not arm.move_gripper_to(pick_xy, z, offset, pick_pitch):
            arm.go_named('home')
            return False, 'FAILED: motion to the grasp pose failed'
    arm.gripper(close_rad)
    arm.move_gripper_to(pick_xy, cfg['lift_z'], offset, pick_pitch)
    if not arm.move_gripper_to(place_xy, cfg['lift_z'], offset, place_pitch):
        arm.gripper(open_rad)
        arm.go_named('home')
        return False, 'FAILED: motion to the place location failed (object dropped)'
    arm.move_gripper_to(place_xy, cfg['place_z'], offset, place_pitch)
    arm.gripper(open_rad)
    arm.move_gripper_to(place_xy, cfg['lift_z'], offset, place_pitch)
    arm.go_named('home')
    return True, 'executed (check the image to verify)'
