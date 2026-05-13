"""Launch the single-named-drone hover test.

Usage:
    ros2 launch particle_life hover_one.launch.py drone:=cf4

Outputs land in <repo>/results/flights/:
  hover_one_<timestamp>/        rosbag (Vicon /poses, /cfN/pose, /tf, /rosout)
  hover_one_<timestamp>.log     tee of every launched process's stdout/stderr
"""
import datetime
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from particle_life._launch_logging import make_bag_recorder, make_stdio_tee


_THIS_FILE = os.path.realpath(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(_THIS_FILE), '..', '..', '..'))
BAG_ROOT = os.path.join(_REPO_ROOT, 'results', 'flights')


def generate_launch_description():
    pl_share = get_package_share_directory('particle_life')
    crazyflies_yaml = os.path.join(pl_share, 'config', 'crazyflies.yaml')
    motion_capture_yaml = os.path.join(pl_share, 'config', 'motion_capture.yaml')
    bag_qos_yaml = os.path.join(pl_share, 'config', 'bag_qos.yaml')

    os.makedirs(BAG_ROOT, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_path = os.path.join(BAG_ROOT, f'hover_one_{stamp}')
    log_path = f'{bag_path}.log'

    drone_arg = DeclareLaunchArgument(
        'drone',
        description="Name of the single drone to fly (must match crazyflies.yaml, e.g. cf4)")

    crazyflie_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('crazyflie'), 'launch', 'launch.py')),
        launch_arguments={
            'backend': 'cflib',
            'mocap': 'True',
            'crazyflies_yaml_file': crazyflies_yaml,
            'motion_capture_yaml_file': motion_capture_yaml,
        }.items()
    )

    hover_one_node = Node(
        package='particle_life',
        executable='hover_one',
        name='hover_one',
        output='screen',
        arguments=['--drone', LaunchConfiguration('drone')],
    )

    return LaunchDescription([
        drone_arg,
        make_stdio_tee(log_path),
        crazyflie_launch,
        hover_one_node,
        make_bag_recorder(bag_path, bag_qos_yaml),
    ])
