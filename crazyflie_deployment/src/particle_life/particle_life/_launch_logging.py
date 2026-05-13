"""Shared helpers for wiring rosbag + stdout/stderr capture into launch files.

Both pieces of telemetry land next to each other in `results/flights/`:

    <bag_path>/             # rosbag dir (mcap + metadata)
    <bag_path>.log          # tee of every launched process's stdout/stderr

The log file captures plain `print(...)` from non-ROS Python code and child
process startup banners — things that DON'T go through /rosout. ROS log
messages (rclpy.logger, rclcpp) DO go through /rosout, which is also recorded
into the bag so playback can recover them.
"""
import os

from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessIO


# Topics to record in every launch. /rosout captures node log messages
# (server connect/disconnect events, teleop warnings, etc).
BAG_TOPICS = [
    '/poses', '/tf', '/tf_static', '/rosout',
    '/cfx/pose',
    '/cf1/pose', '/cf2/pose', '/cf3/pose', '/cf4/pose',
    '/cf5/pose', '/cf6/pose', '/cf7/pose', '/cf9/pose',
]


def make_bag_recorder(bag_path, bag_qos_yaml):
    """ExecuteProcess that runs `ros2 bag record` for the standard topic set."""
    return ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-o', bag_path,
            '--qos-profile-overrides-path', bag_qos_yaml,
            *BAG_TOPICS,
        ],
        output='screen',
    )


def make_stdio_tee(log_path):
    """RegisterEventHandler that tees every process's stdout+stderr to log_path.

    OnProcessIO with no target_action matches all processes launched by this
    LaunchDescription (including ones spawned by IncludeLaunchDescription).
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, 'a', buffering=1)

    def _tee(event):
        name = event.process_name
        text = event.text.decode('utf-8', errors='replace')
        # Preserve any internal newlines but ensure each chunk is tagged.
        for line in text.splitlines(keepends=True):
            log_file.write(f'[{name}] {line}')
        return None

    return RegisterEventHandler(OnProcessIO(on_stdout=_tee, on_stderr=_tee))
