"""
Vicon-only launch: starts Crazyswarm2's motion_capture_tracking node against the
project's portable motion_capture.yaml + crazyflies.yaml. No particle_life_node,
no flight commands. Use this to verify Vicon → /poses (and per-drone /cfN/pose
for any drone enabled in crazyflies.yaml) before doing anything that flies.

    ros2 launch particle_life vicon_test.launch.py
    # in another shell:
    ros2 topic echo /poses
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pl_share = get_package_share_directory('particle_life')
    crazyflies_yaml = os.path.join(pl_share, 'config', 'crazyflies.yaml')
    motion_capture_yaml = os.path.join(pl_share, 'config', 'motion_capture.yaml')

    crazyflie_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('crazyflie'), 'launch', 'launch.py')),
        launch_arguments={
            # cflib backend so motion_capture_tracking actually runs
            # (it's gated off when backend == 'sim'). All robots in our
            # crazyflies.yaml ship with enabled: false, so no radio is touched.
            'backend': 'cflib',
            'mocap': 'True',
            'crazyflies_yaml_file': crazyflies_yaml,
            'motion_capture_yaml_file': motion_capture_yaml,
        }.items()
    )

    return LaunchDescription([crazyflie_launch])
