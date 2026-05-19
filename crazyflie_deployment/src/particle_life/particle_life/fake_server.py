#!/usr/bin/env python3
"""
fake_server.py — minimal stand-in for Crazyswarm2's `crazyflie_server`.

Speaks the same ROS service + topic + parameter interface as the real server,
but with **zero physics and zero firmware emulation**. Each drone's position is
just whatever was last commanded — perfect tracking, no PID, no dynamics.

Purpose
-------
Render what particle_life / hover_all / circle_all / flight_test / ... are
**commanding** through the existing viewer.py, without paying the cost of
`crazyflie_sim` (which simulates 9 firmware controllers + 6-DOF rigid-body
integration single-threaded and gets ~0.07 real-time factor on our laptop).

Three terminals, all existing scripts unchanged:

    Terminal 1:  ros2 run particle_life fake_server
    Terminal 2:  ros2 run particle_life hover_all      # or circle_all, etc.
    Terminal 3:  ros2 run particle_life viewer

Trajectory execution is just linear interpolation:
  takeoff(h, T) → glide from current pos to (x, y, h) over T seconds
  land(h, T)    → glide z to h over T seconds
  goTo(p, T)    → glide xyz to p over T seconds (relative or absolute)
  cmd_full_state → snap to commanded pose, cancel any active trajectory
  arm() → no-op
"""
from __future__ import annotations

import os
import yaml

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
from crazyflie_interfaces.srv import (
    Arm, GoTo, Land, NotifySetpointsStop,
    StartTrajectory, Takeoff, UploadTrajectory,
)
from crazyflie_interfaces.msg import FullState, Hover, Position, Status


DEFAULT_YAML = os.path.expanduser(
    '~/swarm_life/crazyflie_deployment/config/crazyflies.yaml')


def _duration_to_sec(d) -> float:
    return float(d.sec) + float(d.nanosec) * 1e-9


class DroneState:
    """Kinematic state + active trajectory for one drone."""

    def __init__(self, name: str, init_pos):
        self.name = name
        self.pos = np.array(init_pos, dtype=float)
        self.yaw = 0.0
        self.traj_t0 = None
        self.traj_p0 = None
        self.traj_p1 = None
        self.traj_T = 0.0

    def step(self, now: float) -> None:
        if self.traj_t0 is None:
            return
        elapsed = now - self.traj_t0
        if elapsed >= self.traj_T:
            self.pos = self.traj_p1.copy()
            self.traj_t0 = None
            return
        a = elapsed / self.traj_T
        self.pos = self.traj_p0 + a * (self.traj_p1 - self.traj_p0)

    def set_traj(self, target, duration, now):
        self.traj_t0 = now
        self.traj_p0 = self.pos.copy()
        self.traj_p1 = np.array(target, dtype=float)
        self.traj_T = max(duration, 0.001)

    def snap(self, target):
        self.pos = np.array(target, dtype=float)
        self.traj_t0 = None


class FakeServer(Node):

    def __init__(self):
        # Node name MUST be `crazyflie_server` — crazyflie_py queries
        # parameters via `/crazyflie_server/get_parameters`.
        super().__init__('crazyflie_server')

        yaml_path = os.environ.get('FAKE_SERVER_YAML', DEFAULT_YAML)
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        # Build drone state + declare the parameters crazyflie_py queries.
        self.drones: dict[str, DroneState] = {}
        for name, info in (cfg.get('robots') or {}).items():
            if not info.get('enabled', False):
                continue
            init_pos = info.get('initial_position', [0.0, 0.0, 0.0])
            uri = info.get('uri', 'radio://fake')
            self.drones[name] = DroneState(name, init_pos)
            # crazyflie_py reads these two via GetParameters:
            self.declare_parameter(f'robots.{name}.initial_position',
                                   [float(x) for x in init_pos])
            self.declare_parameter(f'robots.{name}.uri', uri)

        if not self.drones:
            self.get_logger().error("no enabled drones in YAML; nothing to do")
            return

        self.get_logger().info(
            f"FakeServer started: {len(self.drones)} drone(s) {list(self.drones)}")

        # Per-drone publishers + services + subscriptions.
        self._pose_pubs: dict[str, any] = {}
        self._status_pubs: dict[str, any] = {}
        for name in self.drones:
            self._pose_pubs[name] = self.create_publisher(
                PoseStamped, f'{name}/pose', 10)
            self._status_pubs[name] = self.create_publisher(
                Status, f'{name}/status', 10)

            # Services. crazyflie_py uses /<cf>/start_trajectory to enumerate
            # drones, and waits on several others during Crazyswarm() init.
            self.create_service(Empty, f'{name}/emergency', self._noop_srv)
            self.create_service(StartTrajectory, f'{name}/start_trajectory',
                                self._noop_srv)
            self.create_service(UploadTrajectory, f'{name}/upload_trajectory',
                                self._noop_srv)
            self.create_service(NotifySetpointsStop,
                                f'{name}/notify_setpoints_stop', self._noop_srv)
            self.create_service(Arm, f'{name}/arm', self._noop_srv)
            self.create_service(
                Takeoff, f'{name}/takeoff',
                lambda req, resp, n=name: self._takeoff(req, resp, n))
            self.create_service(
                Land, f'{name}/land',
                lambda req, resp, n=name: self._land(req, resp, n))
            self.create_service(
                GoTo, f'{name}/go_to',
                lambda req, resp, n=name: self._goto(req, resp, n))

            # cmd_full_state is what SetpointAdapter sends in sim mode.
            self.create_subscription(
                FullState, f'{name}/cmd_full_state',
                lambda msg, n=name: self._cmd_full_state(msg, n), 10)
            # cmd_position lets hardware-style scripts work too.
            self.create_subscription(
                Position, f'{name}/cmd_position',
                lambda msg, n=name: self._cmd_position(msg, n), 10)

        # Swarm-level services.
        self.create_service(Empty, 'all/emergency', self._noop_srv)
        self.create_service(Takeoff, 'all/takeoff', self._all_takeoff)
        self.create_service(Land, 'all/land', self._all_land)
        self.create_service(GoTo, 'all/go_to', self._all_goto)
        self.create_service(Arm, 'all/arm', self._noop_srv)
        self.create_service(StartTrajectory, 'all/start_trajectory',
                            self._noop_srv)

        # 50 Hz pose tick — plenty for viewer.
        self.create_timer(0.02, self._publish_poses)

    # ---------- helpers ----------

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _noop_srv(_req, resp):
        return resp

    def _publish_poses(self):
        now = self._now()
        stamp = self.get_clock().now().to_msg()
        for name, d in self.drones.items():
            d.step(now)
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = 'world'
            msg.pose.position.x = float(d.pos[0])
            msg.pose.position.y = float(d.pos[1])
            msg.pose.position.z = float(d.pos[2])
            msg.pose.orientation.w = float(np.cos(d.yaw / 2))
            msg.pose.orientation.z = float(np.sin(d.yaw / 2))
            self._pose_pubs[name].publish(msg)

    # ---------- per-drone services ----------

    def _takeoff(self, req, resp, name):
        d = self.drones[name]
        target = d.pos.copy()
        target[2] = float(req.height)
        d.set_traj(target, _duration_to_sec(req.duration), self._now())
        return resp

    def _land(self, req, resp, name):
        d = self.drones[name]
        target = d.pos.copy()
        target[2] = float(req.height)
        d.set_traj(target, _duration_to_sec(req.duration), self._now())
        return resp

    def _goto(self, req, resp, name):
        d = self.drones[name]
        g = req.goal
        target = np.array([g.x, g.y, g.z], dtype=float)
        if req.relative:
            target = d.pos + target
        d.set_traj(target, _duration_to_sec(req.duration), self._now())
        d.yaw = float(np.deg2rad(req.yaw))
        return resp

    # ---------- streaming command subscriptions ----------

    def _cmd_full_state(self, msg: FullState, name: str):
        d = self.drones[name]
        d.snap([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def _cmd_position(self, msg: Position, name: str):
        d = self.drones[name]
        d.snap([msg.x, msg.y, msg.z])
        d.yaw = float(np.deg2rad(msg.yaw))

    # ---------- swarm-level services ----------

    def _all_takeoff(self, req, resp):
        for n in self.drones:
            self._takeoff(req, resp, n)
        return resp

    def _all_land(self, req, resp):
        for n in self.drones:
            self._land(req, resp, n)
        return resp

    def _all_goto(self, req, resp):
        for n in self.drones:
            self._goto(req, resp, n)
        return resp


def main():
    rclpy.init()
    node = FakeServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
