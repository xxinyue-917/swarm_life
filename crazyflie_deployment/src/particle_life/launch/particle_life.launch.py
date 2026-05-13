"""Launch the particle life controller alongside the Crazyswarm2 server.

Fully portable: all config paths resolve through the `particle_life` package's
share dir, so anyone who clones swarm_life and builds gets a working stack.
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pl_share = get_package_share_directory('particle_life')
    crazyflies_yaml = os.path.join(pl_share, 'config', 'crazyflies.yaml')
    motion_capture_yaml = os.path.join(pl_share, 'config', 'motion_capture.yaml')

    backend_arg = DeclareLaunchArgument(
        'backend', default_value='sim',
        description="'sim' for software-in-the-loop, 'cflib' or 'cpp' for real drones")

    crazyflie_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('crazyflie'), 'launch', 'launch.py')),
        launch_arguments={
            'backend': LaunchConfiguration('backend'),
            'crazyflies_yaml_file': crazyflies_yaml,
            'motion_capture_yaml_file': motion_capture_yaml,
        }.items()
    )

    particle_life_node = Node(
        package='particle_life',
        executable='particle_life_node',
        name='particle_life_controller',
        output='screen',
        # Forward backend choice to the SetpointAdapter (cmdPosition vs cmdFullState).
        additional_env={'CF_BACKEND': LaunchConfiguration('backend')},
    )

    return LaunchDescription([
        backend_arg,
        crazyflie_launch,
        particle_life_node,
    ])
