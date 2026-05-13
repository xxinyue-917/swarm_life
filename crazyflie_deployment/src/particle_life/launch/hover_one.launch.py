"""Launch the single-named-drone hover test.

Usage:
    ros2 launch particle_life hover_one.launch.py drone:=cf4
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pl_share = get_package_share_directory('particle_life')
    crazyflies_yaml = os.path.join(pl_share, 'config', 'crazyflies.yaml')
    motion_capture_yaml = os.path.join(pl_share, 'config', 'motion_capture.yaml')

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

    return LaunchDescription([drone_arg, crazyflie_launch, hover_one_node])
