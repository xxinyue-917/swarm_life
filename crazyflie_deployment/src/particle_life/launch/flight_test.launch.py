"""Launch the single-drone flight test alongside the Crazyswarm2 server.

Used to validate the hardware loop end-to-end (radio + Vicon + firmware) on a
single drone before bringing up the particle_life controller.
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
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
            'backend': 'cflib',
            'mocap': 'True',
            'crazyflies_yaml_file': crazyflies_yaml,
            'motion_capture_yaml_file': motion_capture_yaml,
        }.items()
    )

    flight_test_node = Node(
        package='particle_life',
        executable='flight_test',
        name='flight_test',
        output='screen',
    )

    return LaunchDescription([crazyflie_launch, flight_test_node])
