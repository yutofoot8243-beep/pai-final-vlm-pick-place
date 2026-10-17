"""Gazebo (sim.launch.py) と VLM エージェント・可視化ウィンドウをまとめて起動する.

  ros2 launch vlm_pick_place vlm_pick_place.launch.py
  ros2 launch vlm_pick_place vlm_pick_place.launch.py auto_instruction:='ブロックを同じ色のトレイに片付けて'
  ros2 launch vlm_pick_place vlm_pick_place.launch.py record:=true video:=$HOME/videos/demo.mp4       auto_instruction:='ブロックを同じ色のトレイに片付けて'
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('vlm_pick_place')

    declare_world = DeclareLaunchArgument(
        'world_name', default_value=os.path.join(pkg, 'worlds', 'vlm_table.sdf'))
    declare_agent = DeclareLaunchArgument(
        'use_agent', default_value='true', description='Start the VLM agent node.')
    declare_viewer = DeclareLaunchArgument(
        'use_viewer', default_value='true', description='Show /vlm/annotated_image.')
    declare_record = DeclareLaunchArgument(
        'record', default_value='false', description='Save a demo video (MP4).')
    declare_output = DeclareLaunchArgument(
        'video', default_value=os.path.expanduser('~/videos/pai_demo.mp4'))
    declare_auto = DeclareLaunchArgument(
        'auto_instruction', default_value='', description='Instruction executed on start.')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')),
        launch_arguments={'world_name': LaunchConfiguration('world_name')}.items())

    agent = Node(
        package='vlm_pick_place',
        executable='vlm_agent',
        output='screen',
        parameters=[os.path.join(pkg, 'config', 'params.yaml'),
                    {'auto_instruction': ParameterValue(
                        LaunchConfiguration('auto_instruction'), value_type=str)}],
        condition=IfCondition(LaunchConfiguration('use_agent')))

    viewer = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        arguments=['/vlm/annotated_image'],
        condition=IfCondition(LaunchConfiguration('use_viewer')))

    recorder = Node(
        package='vlm_pick_place',
        executable='recorder',
        output='screen',
        parameters=[{'output': LaunchConfiguration('video')}],
        condition=IfCondition(LaunchConfiguration('record')))

    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        declare_world,
        declare_agent,
        declare_viewer,
        declare_record,
        declare_output,
        declare_auto,
        sim,
        # コントローラと move_group が立ち上がるのを待ってから起動する
        TimerAction(period=15.0, actions=[recorder, agent, viewer]),
    ])
