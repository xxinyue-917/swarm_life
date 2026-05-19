#!/usr/bin/env python3
"""
Particle Life swarm controller.

Lifecycle
---------
    load config → wait for poses → takeoff → stream mode loop → land

Modes (`--mode`):
    hover : stream each drone's takeoff XY at the configured altitude
            (useful as a streaming-pipeline smoke test)
    force : 2D particle-life kernel — K_pos radial + K_rot tangential
            (the M2/M3 default)

Targets are streamed via `cmdFullState` (with zero velocity / acceleration
feedforward), so the same code-path works on:
    * real hardware (cflib backend)
    * Crazyswarm2 `sim` backend
    * `fake_server` (perfect-tracking stub for viewer dev)

Architecture
------------
    Config              dataclass; built once from arena.yaml + species.yaml.
    PoseSource          subscribes to `/cfN/pose` on the swarm node; tracks
                        per-drone last-update time for stale detection.
    ParticleLife        controller class. One streaming loop (`_stream`)
                        shared between modes; the mode just supplies a
                        per-tick `get_target` function.

Safety (driven by `arena.yaml.safety`)
-------------------------------------
    * pose-stale FREEZE     pose older than `pose_stale_hover` → hold last target.
    * pose-stale EMERGENCY  pose older than `pose_stale_land`  → land everyone.
    * MIN-SEPARATION        any two drones closer than `min_separation` → land.
    * GEOFENCE              targets clipped to arena minus `geofence_margin`.

There is **no silent fallback**: if a connected drone is missing from
`species.yaml.assignments`, startup fails with a clear message rather than
quietly assigning it to species 0.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import threading
import time

import numpy as np
import yaml

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from crazyflie_py import Crazyswarm

from particle_life import force_controller
from particle_life.flight_logger import FlightLogger


# ---------------------------------------------------------------- config


def _share(*parts):
    return os.path.join(get_package_share_directory('particle_life'), *parts)


@dataclasses.dataclass
class Config:
    altitude: float
    arena_w: float
    arena_h: float
    origin: tuple
    geofence_margin: float
    rate_hz: int
    pose_stale_freeze: float
    pose_stale_emergency: float
    min_separation: float
    v_max: float
    post_kernel_scale: float        # arena.yaml control.force_output_scale
    physics_params: dict            # passed to force_controller.compute_velocities
    takeoff_duration: float = 3.0
    land_duration: float = 3.0
    initial_pose_timeout: float = 5.0


def load_config() -> Config:
    """Build the Config from arena.yaml."""
    with open(_share('config', 'arena.yaml')) as f:
        a = yaml.safe_load(f)
    arena = a['arena']
    safety = a['safety']
    ctrl = a['control']
    return Config(
        altitude=float(arena['altitude']),
        arena_w=float(arena['width']),
        arena_h=float(arena['height']),
        origin=(float(arena['origin_x']), float(arena['origin_y'])),
        geofence_margin=float(safety['geofence_margin']),
        rate_hz=int(ctrl['rate_hz']),
        pose_stale_freeze=float(safety['pose_stale_hover']),
        pose_stale_emergency=float(safety['pose_stale_land']),
        min_separation=float(safety['min_separation']),
        v_max=float(safety['v_max_initial']),
        post_kernel_scale=float(ctrl['force_output_scale']),
        physics_params={
            'r_max': float(ctrl['r_max']),
            'beta': float(ctrl['beta']),
            'force_scale': float(ctrl['force_scale']),
            'far_attraction': float(ctrl['far_attraction']),
            'a_rot': float(ctrl['a_rot']),
        },
    )


def load_species_and_matrices():
    """Return (assignments, n_species, k_pos, k_rot, preset_label)."""
    with open(_share('config', 'species.yaml')) as f:
        cfg = yaml.safe_load(f)
    active = cfg['active']
    assignments = cfg['presets'][active]['assignments']
    n_species = max(assignments.values()) + 1

    matrix_rel = cfg.get('matrix_preset')
    if matrix_rel is None:
        zeros = np.zeros((n_species, n_species))
        return assignments, n_species, zeros, zeros, 'zeros'

    with open(_share('config', *matrix_rel.split('/'))) as f:
        mp = json.load(f)
    k_pos = np.asarray(mp['k_pos'], dtype=float)
    k_rot = np.asarray(mp['k_rot'], dtype=float)
    if k_pos.shape != (n_species, n_species):
        raise ValueError(
            f"k_pos shape {k_pos.shape} does not match n_species={n_species} "
            f"derived from species.yaml preset '{active}'.")
    if k_rot.shape != (n_species, n_species):
        raise ValueError(
            f"k_rot shape {k_rot.shape} does not match n_species={n_species}.")
    return assignments, n_species, k_pos, k_rot, os.path.basename(matrix_rel)


# ---------------------------------------------------------------- pose source


class PoseSource:
    """Subscribes to /cfN/pose on the swarm node; thread-safe latest pose + age."""

    def __init__(self, node, names):
        self.names = list(names)
        self.n = len(self.names)
        self._lock = threading.Lock()
        self._poses = np.zeros((self.n, 3))
        # last_t = 0 sentinel meaning "no pose received yet"
        self._last_t = np.zeros(self.n)

        for i, name in enumerate(self.names):
            node.create_subscription(
                PoseStamped, f'/{name}/pose',
                lambda msg, idx=i: self._on_pose(idx, msg),
                10)

    def _on_pose(self, i, msg):
        p = msg.pose.position
        with self._lock:
            self._poses[i, 0] = p.x
            self._poses[i, 1] = p.y
            self._poses[i, 2] = p.z
            self._last_t[i] = time.monotonic()

    def snapshot(self):
        with self._lock:
            return self._poses.copy(), self._last_t.copy()

    def have_all_poses(self):
        with self._lock:
            return bool(np.all(self._last_t > 0))


# ---------------------------------------------------------------- controller


class EmergencyAbort(Exception):
    """Raised by _emergency to bubble out of the run loop after landing."""


class ParticleLife:

    _ZERO3 = (0.0, 0.0, 0.0)

    def __init__(self):
        self.cfg = load_config()
        assignments, n_species, k_pos, k_rot, preset_label = \
            load_species_and_matrices()

        # --- Crazyswarm ---
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.allcfs = self.swarm.allcfs
        self.cfs = list(self.allcfs.crazyflies)
        self.names = [cf.prefix.lstrip('/') for cf in self.cfs]
        self.n = len(self.cfs)
        if self.n == 0:
            raise RuntimeError(
                "no drones connected; check `enabled: true` in crazyflies.yaml "
                "and that the Crazyswarm2 server is running.")

        # --- Strict species mapping ---
        missing = [n for n in self.names if n not in assignments]
        if missing:
            extra = [n for n in assignments if n not in self.names]
            raise RuntimeError(
                f"species.yaml mismatch — connected drones missing from "
                f"assignments: {missing}. Unused yaml entries: {extra}. "
                f"Edit config/species.yaml ('{cfg_active_hint(assignments)}' "
                f"preset) so it matches the enabled robots.")
        self.species = np.array([assignments[n] for n in self.names], dtype=int)
        self.k_pos = k_pos
        self.k_rot = k_rot

        # --- Pose subscriptions (share the swarm node so spinning is automatic) ---
        self.poses = PoseSource(self.allcfs, self.names)

        # --- Internal landed flag for clean-shutdown idempotence ---
        self._landed = False

        c = self.cfg
        print(f"[particle_life] {self.n} drones: {self.names}")
        print(f"[particle_life] species: "
              f"{dict(zip(self.names, self.species.tolist()))} "
              f"(S={n_species})")
        print(f"[particle_life] matrix preset: {preset_label}")
        print(f"[particle_life] altitude={c.altitude}m  rate={c.rate_hz}Hz  "
              f"v_max={c.v_max}m/s  arena={c.arena_w}x{c.arena_h}m  "
              f"geofence_margin={c.geofence_margin}m  "
              f"min_sep={c.min_separation}m")
        print(f"[particle_life] post_kernel_scale={c.post_kernel_scale}  "
              f"physics={c.physics_params}")

    # --- pose wait ---

    def _wait_for_initial_poses(self):
        """Block until every drone produces at least one pose, or timeout."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.cfg.initial_pose_timeout:
            self.timeHelper.sleep(0.1)
            if self.poses.have_all_poses():
                return
        _, last_t = self.poses.snapshot()
        missing = [self.names[i] for i, t in enumerate(last_t) if t == 0]
        raise RuntimeError(
            f"no pose received for {missing} within "
            f"{self.cfg.initial_pose_timeout}s; aborting takeoff. "
            f"Check Vicon Tracker (hardware) or that fake_server / sim is up.")

    # --- lifecycle ---

    def takeoff(self):
        self._wait_for_initial_poses()
        print(f"[particle_life] takeoff to {self.cfg.altitude}m")
        self.allcfs.takeoff(targetHeight=self.cfg.altitude,
                            duration=self.cfg.takeoff_duration)
        self.timeHelper.sleep(self.cfg.takeoff_duration + 0.5)

    def land(self):
        if self._landed:
            return
        self._landed = True
        print("[particle_life] landing")
        for cf in self.cfs:
            cf.notifySetpointsStop()
        self.allcfs.land(targetHeight=0.04,
                         duration=self.cfg.land_duration)
        self.timeHelper.sleep(self.cfg.land_duration + 0.5)

    def _emergency(self, reason):
        print(f"[particle_life] EMERGENCY: {reason}", flush=True)
        self.land()
        raise EmergencyAbort(reason)

    # --- safety checks ---

    def _check_pose_health(self, last_t):
        """Return (stale_mask). Raises EmergencyAbort if any pose is critically old."""
        now = time.monotonic()
        age = now - last_t
        emerg = age > self.cfg.pose_stale_emergency
        if np.any(emerg):
            bad = [self.names[i] for i in np.where(emerg)[0]]
            ages = [f'{age[i]:.2f}s' for i in np.where(emerg)[0]]
            self._emergency(
                f"pose stale > {self.cfg.pose_stale_emergency}s on "
                f"{list(zip(bad, ages))}")
        return age > self.cfg.pose_stale_freeze

    def _check_min_separation(self, poses_xyz):
        if self.cfg.min_separation <= 0 or self.n < 2:
            return
        diff = poses_xyz[:, None, :] - poses_xyz[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(d, np.inf)
        d_min = float(d.min())
        if d_min < self.cfg.min_separation:
            i, j = np.unravel_index(np.argmin(d), d.shape)
            self._emergency(
                f"min separation violated: {self.names[i]}<->{self.names[j]} "
                f"at {d_min:.2f}m < {self.cfg.min_separation:.2f}m")

    # --- streaming ---

    def _stream(self, get_target, duration, logger):
        """Shared streaming loop. `get_target(poses_xyz, tick)` → (targets_xyz, v_cmd)."""
        period = 1.0 / self.cfg.rate_hz
        n_ticks = int(duration * self.cfg.rate_hz)
        last_targets = None

        for tick in range(n_ticks):
            poses_xyz, last_t = self.poses.snapshot()
            stale = self._check_pose_health(last_t)
            self._check_min_separation(poses_xyz)

            targets_xyz, v_cmd = get_target(poses_xyz, tick)

            # Freeze stale drones to their last commanded target.
            if last_targets is not None and stale.any():
                for i in np.where(stale)[0]:
                    targets_xyz[i] = last_targets[i]
                    v_cmd[i] = 0.0

            # Geofence (XY only — Z is fixed by mode).
            targets_xyz[:, :2] = force_controller.geofence_clip(
                targets_xyz[:, :2],
                self.cfg.arena_w, self.cfg.arena_h,
                self.cfg.geofence_margin, self.cfg.origin)

            # Stream.
            for i, cf in enumerate(self.cfs):
                cf.cmdFullState(
                    tuple(targets_xyz[i]), self._ZERO3, self._ZERO3,
                    0.0, self._ZERO3)

            if logger is not None:
                logger.log_tick(tick, poses_xyz[:, :2], poses_xyz[:, 2],
                                targets_xyz, v_cmd)
            last_targets = targets_xyz
            self.timeHelper.sleep(period)

    # --- modes ---

    def run_hover(self, duration, logger=None):
        """Stream takeoff XY at fixed altitude."""
        init_xy = np.array([cf.initialPosition[:2] for cf in self.cfs])
        z = self.cfg.altitude
        z_col = np.full(self.n, z)
        zero_v = np.zeros((self.n, 2))

        def get_target(poses_xyz, tick):
            return np.column_stack([init_xy, z_col]), zero_v.copy()

        self._stream(get_target, duration, logger)

    def run_force(self, duration, logger=None, aggregate_duration=0.0):
        """K_pos + K_rot driven control loop.

        If `aggregate_duration > 0`, K_rot is held at zero for the first
        `aggregate_duration` seconds so the swarm can form pure-K_pos clusters
        before the rotational coupling kicks in.
        """
        period = 1.0 / self.cfg.rate_hz
        aggregate_ticks = int(aggregate_duration * self.cfg.rate_hz)
        z_col = np.full(self.n, self.cfg.altitude)
        k_rot_off = np.zeros_like(self.k_rot)
        phase_msg = {'rotation_started': False}

        def get_target(poses_xyz, tick):
            poses_xy = poses_xyz[:, :2]
            if tick < aggregate_ticks:
                k_rot_now = k_rot_off
            else:
                if not phase_msg['rotation_started'] and aggregate_ticks > 0:
                    print(f"[particle_life] aggregate phase done "
                          f"({aggregate_duration:.1f}s) — rotation enabled")
                    phase_msg['rotation_started'] = True
                k_rot_now = self.k_rot

            v = force_controller.compute_velocities(
                poses_xy, self.species, self.k_pos, k_rot_now,
                self.cfg.physics_params)
            v *= self.cfg.post_kernel_scale
            v = force_controller.clamp_speed(v, self.cfg.v_max)
            targets_xy = poses_xy + v * period
            return np.column_stack([targets_xy, z_col]), v

        if aggregate_duration > 0:
            print(f"[particle_life] force loop: aggregate for "
                  f"{aggregate_duration:.1f}s (K_rot=0), then rotate")
        self._stream(get_target, duration, logger)

    # --- top-level driver ---

    def run(self, mode, duration, log=True, run_tag=None,
            aggregate_duration=0.0):
        logger = None
        if log:
            logger = FlightLogger(self.names, self.species.tolist(),
                                  run_tag=run_tag or mode)
        try:
            self.takeoff()
            if mode == 'hover':
                self.run_hover(duration, logger=logger)
            else:
                self.run_force(duration, logger=logger,
                               aggregate_duration=aggregate_duration)
        except (KeyboardInterrupt, EmergencyAbort) as e:
            if isinstance(e, KeyboardInterrupt):
                print("[particle_life] interrupted")
        finally:
            try:
                self.land()
            finally:
                if logger is not None:
                    logger.close()


# Tiny utility used in the error message above.
def cfg_active_hint(assignments):
    """Best-effort guess for which preset the assignments came from."""
    return 'active' if assignments else '(empty)'


# ---------------------------------------------------------------- main


def main():
    # rclpy / Crazyswarm consume their own argv; strip ROS args before argparse.
    user_argv = [a for a in sys.argv[1:] if not a.startswith('--ros-args')]
    p = argparse.ArgumentParser(prog='particle_life')
    p.add_argument('--mode', choices=['hover', 'force'], default='force',
                   help="hover = stream takeoff XY at altitude (M1 smoke test); "
                        "force = K_pos + K_rot loop (M2/M3 default)")
    p.add_argument('--duration', type=float, default=60.0,
                   help="seconds of in-flight control after takeoff")
    p.add_argument('--aggregate-duration', type=float, default=0.0,
                   help="force mode only: hold K_rot=0 for this many seconds "
                        "at the start so clusters form before rotation enables")
    p.add_argument('--no-log', action='store_true',
                   help="disable CSV pose / target logging")
    p.add_argument('--tag', default=None,
                   help="extra suffix on the log filename for identification")
    args, _ = p.parse_known_args(user_argv)

    ctrl = ParticleLife()
    ctrl.run(args.mode, args.duration,
             log=not args.no_log, run_tag=args.tag,
             aggregate_duration=args.aggregate_duration)


if __name__ == '__main__':
    main()
