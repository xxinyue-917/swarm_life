#!/usr/bin/env python3
"""
Particle Life swarm controller for Crazyflie drones.

Milestone scaffolding:
  M1: takeoff, hover via SetpointAdapter, land
      (preserved as `--mode hover` for streaming-pipeline verification)
  M2/M3 (current default): N-drone, N-species K_pos + K_rot force control
      Poses are read each tick via `cf.position()`, velocities computed by
      `force_controller.compute_velocities`, integrated to position targets,
      streamed via SetpointAdapter (cmdFullState in sim, cmdPosition on hw).

Drone count is whatever Crazyswarm2 brings up from crazyflies.yaml — set N
by enabling the desired robots there. Species mapping comes from species.yaml
(active preset); K_pos / K_rot matrices come from the JSON preset it points to.
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

from crazyflie_py import Crazyswarm

from particle_life.setpoint_adapter import SetpointAdapter
from particle_life import force_controller


HOVER_HEIGHT = 1.0
TAKEOFF_DURATION = 3.0
LAND_DURATION = 3.0
STREAM_HZ = 30
RUN_DURATION = 30.0  # seconds of force-driven flight before landing


def _share_path(*parts):
    return os.path.join(get_package_share_directory('particle_life'), *parts)


def load_species_and_matrices():
    """Read species.yaml and the matrix preset it points to.

    Returns:
        assignments    : dict[name -> species_idx]
        n_species      : int
        k_pos, k_rot   : (S, S) numpy arrays
        matrix_label   : str (human-readable)
    """
    species_path = _share_path('config', 'species.yaml')
    with open(species_path) as f:
        cfg = yaml.safe_load(f)

    active = cfg['active']
    preset = cfg['presets'][active]
    assignments = preset['assignments']
    n_species = max(assignments.values()) + 1

    matrix_rel = cfg.get('matrix_preset')
    if matrix_rel is None:
        # Fallback: zeros matrix (drones hover with no interaction).
        return assignments, n_species, np.zeros((n_species, n_species)), np.zeros((n_species, n_species)), 'zeros'

    matrix_path = _share_path('config', *matrix_rel.split('/'))
    with open(matrix_path) as f:
        mp = json.load(f)
    k_pos = np.asarray(mp['k_pos'], dtype=float)
    k_rot = np.asarray(mp['k_rot'], dtype=float)
    assert k_pos.shape == (n_species, n_species), (
        f"k_pos shape {k_pos.shape} != ({n_species},{n_species}) from species.yaml")
    return assignments, n_species, k_pos, k_rot, os.path.basename(matrix_rel)


def load_arena():
    with open(_share_path('config', 'arena.yaml')) as f:
        return yaml.safe_load(f)


class ParticleLifeController:

    def __init__(self):
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.allcfs = self.swarm.allcfs
        self.cfs = self.allcfs.crazyflies
        self.n = len(self.cfs)

        backend = os.environ.get('CF_BACKEND', 'cflib')
        self.adapter = SetpointAdapter(backend=backend)

        # --- Load configs ---
        assignments, n_species, k_pos, k_rot, label = load_species_and_matrices()
        self.k_pos = k_pos
        self.k_rot = k_rot
        self.n_species = n_species
        arena = load_arena()
        self.arena = arena['arena']
        self.safety = arena['safety']
        self.control_cfg = arena['control']

        # Map this run's connected drones to species via name.
        # Drones not in species.yaml fall back to species 0 (warned).
        self.species = np.zeros(self.n, dtype=int)
        for i, cf in enumerate(self.cfs):
            name = self._cf_name(cf, i)
            if name in assignments:
                self.species[i] = assignments[name]
            else:
                print(f"[particle_life] WARN: '{name}' not in species.yaml → species 0")

        print(f"[particle_life] {self.n} drones connected — backend={backend}")
        print(f"[particle_life] species={self.species.tolist()}  (S={n_species})")
        print(f"[particle_life] matrix preset: {label}")
        print(f"[particle_life] arena {self.arena['width']}×{self.arena['height']} m "
              f"@ z={HOVER_HEIGHT}m, v_max={self.safety['v_max_initial']} m/s, "
              f"force_scale={self.control_cfg['force_output_scale']}")

    @staticmethod
    def _cf_name(cf, fallback_idx):
        """Best-effort name extraction matching crazyflies.yaml keys.

        Crazyswarm2 names appear at different attributes across versions:
        prefix='/cf1' (ROS namespace), id=1 (legacy int id), or cfname='cf1'.
        """
        prefix = getattr(cf, 'prefix', None)
        if prefix:
            return prefix.lstrip('/')
        for attr in ('cfname', 'name'):
            v = getattr(cf, attr, None)
            if v:
                return str(v)
        cf_id = getattr(cf, 'id', None)
        if cf_id is not None:
            return f'cf{cf_id}'
        return f'cf{fallback_idx + 1}'

    # ------------------------------------------------------------------ flight

    def takeoff(self):
        print(f"[particle_life] taking off to {HOVER_HEIGHT} m")
        self.allcfs.takeoff(targetHeight=HOVER_HEIGHT, duration=TAKEOFF_DURATION)
        self.timeHelper.sleep(TAKEOFF_DURATION + 0.5)

    def land(self):
        print("[particle_life] landing")
        for cf in self.cfs:
            cf.notifySetpointsStop()
        self.allcfs.land(targetHeight=0.04, duration=LAND_DURATION)
        self.timeHelper.sleep(LAND_DURATION + 0.5)

    # --------------------------------------------------------------- M1 hover

    def hover(self, duration: float):
        """Stream static per-drone position targets at z = HOVER_HEIGHT."""
        targets = [(cf.initialPosition[0], cf.initialPosition[1], HOVER_HEIGHT)
                   for cf in self.cfs]
        n_ticks = int(duration * STREAM_HZ)
        period = 1.0 / STREAM_HZ
        for _ in range(n_ticks):
            for cf, target in zip(self.cfs, targets):
                self.adapter.set_target(cf, target, yaw=0.0)
            self.timeHelper.sleep(period)

    # ---------------------------------------------------------- M2/M3 force

    def _read_poses_xy(self):
        """Return current XY positions (N, 2) in the arena (Vicon) frame."""
        xy = np.zeros((self.n, 2))
        for i, cf in enumerate(self.cfs):
            pos = cf.position()  # numpy (3,) — sim uses internal state, hw uses Vicon
            xy[i, 0] = pos[0]
            xy[i, 1] = pos[1]
        return xy

    def run_force_control(self, duration: float):
        """Force-driven control loop — the M2/M3 default."""
        v_max = float(self.safety['v_max_initial'])
        force_output_scale = float(self.control_cfg['force_output_scale'])
        physics_params = {
            'r_max': float(self.control_cfg['r_max']),
            'beta': float(self.control_cfg['beta']),
            'force_scale': float(self.control_cfg['force_scale']),
            'far_attraction': float(self.control_cfg['far_attraction']),
            'a_rot': float(self.control_cfg['a_rot']),
        }
        geofence_margin = float(self.safety['geofence_margin'])
        arena_w = float(self.arena['width'])
        arena_h = float(self.arena['height'])
        origin = (float(self.arena['origin_x']), float(self.arena['origin_y']))

        period = 1.0 / STREAM_HZ
        n_ticks = int(duration * STREAM_HZ)

        print(f"[particle_life] force control running for {duration:.1f}s @ {STREAM_HZ} Hz")
        for tick in range(n_ticks):
            poses = self._read_poses_xy()

            v = force_controller.compute_velocities(
                poses, self.species, self.k_pos, self.k_rot, physics_params)

            # Scale sim-units velocity → physical m/s and clamp.
            v *= force_output_scale
            v = force_controller.clamp_speed(v, v_max)

            # Integrate one tick to get the next position target.
            targets_xy = poses + v * period
            targets_xy = force_controller.geofence_clip(
                targets_xy, arena_w, arena_h, geofence_margin, origin)

            for i, cf in enumerate(self.cfs):
                self.adapter.set_target(
                    cf,
                    (targets_xy[i, 0], targets_xy[i, 1], HOVER_HEIGHT),
                    yaw=0.0,
                )
            self.timeHelper.sleep(period)

    # ---------------------------------------------------------------- driver

    def run(self, mode: str, duration: float):
        try:
            self.takeoff()
            if mode == 'hover':
                self.hover(duration)
            else:
                self.run_force_control(duration)
        except KeyboardInterrupt:
            print("[particle_life] interrupted, landing")
        finally:
            self.land()


def main():
    # Strip args bound for rclpy/Crazyswarm before argparse.
    user_argv = [a for a in sys.argv[1:] if not a.startswith('--ros-args')]
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['hover', 'force'], default='force',
                        help='hover = M1 static targets; force = M2/M3 K_pos+K_rot loop')
    parser.add_argument('--duration', type=float, default=RUN_DURATION,
                        help='seconds of flight after takeoff')
    args, _ = parser.parse_known_args(user_argv)

    ctrl = ParticleLifeController()
    ctrl.run(args.mode, args.duration)


if __name__ == '__main__':
    main()
