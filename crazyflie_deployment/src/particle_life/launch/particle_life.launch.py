"""Launch the particle life controller alongside the Crazyswarm2 server."""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    backend_arg = DeclareLaunchArgument(
        'backend', default_value='sim',
        description="'sim' for software-in-the-loop, 'cflib' or 'cpp' for real drones")

    crazyflie_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('crazyflie'), 'launch', 'launch.py')),
        launch_arguments={'backend': LaunchConfiguration('backend')}.items()
    )

    particle_life_node = Node(
        package='particle_life',
        executable='particle_life_node',
        name='particle_life_controller',
        output='screen',
    )

    return LaunchDescription([
        backend_arg,
        crazyflie_launch,
        particle_life_node,
    ])
