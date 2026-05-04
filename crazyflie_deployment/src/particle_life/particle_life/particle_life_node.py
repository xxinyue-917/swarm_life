#!/usr/bin/env python3
"""
Particle Life swarm controller for Crazyflie drones.

Milestone scaffolding:
  M1 (current): takeoff, hover via SetpointAdapter, land
                — verifies ROS <-> Crazyswarm2 streaming pipeline
  M2 (next):    add K_pos force computation for 2 drones
  M3 (final):   10-drone multi-species with K_pos + K_rot
"""

import os

from crazyflie_py import Crazyswarm

from particle_life.setpoint_adapter import SetpointAdapter


HOVER_HEIGHT = 1.0   # meters
TAKEOFF_DURATION = 3.0
HOVER_DURATION = 5.0
LAND_DURATION = 3.0
STREAM_HZ = 30


class ParticleLifeController:
    """High-level controller that drives Crazyflies through particle life behaviors."""

    def __init__(self):
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.allcfs = self.swarm.allcfs
        self.cfs = self.allcfs.crazyflies
        self.n = len(self.cfs)

        # Backend ('sim' | 'cflib' | 'cpp') is forwarded through the CF_BACKEND
        # env var by the launch file. Default to 'cflib' (real hardware).
        backend = os.environ.get('CF_BACKEND', 'cflib')
        self.adapter = SetpointAdapter(backend=backend)
        print(f"[particle_life] {self.n} drones connected — backend={backend} "
              f"({'cmdFullState' if self.adapter.is_sim else 'cmdPosition'} streaming)")

    def takeoff(self):
        print(f"[particle_life] taking off to {HOVER_HEIGHT} m")
        self.allcfs.takeoff(targetHeight=HOVER_HEIGHT, duration=TAKEOFF_DURATION)
        self.timeHelper.sleep(TAKEOFF_DURATION + 0.5)

    def hover(self, duration: float):
        """Stream a static position target per drone for `duration` seconds.

        Exercises the same SetpointAdapter path M2/M3 will use for force-driven motion.
        Initial target = current Vicon position with z = HOVER_HEIGHT (drones already
        there from takeoff).
        """
        print(f"[particle_life] hovering for {duration:.1f}s @ {STREAM_HZ} Hz "
              f"(M1 scaffold; static target per drone)")
        targets = [(cf.initialPosition[0], cf.initialPosition[1], HOVER_HEIGHT)
                   for cf in self.cfs]

        # TODO M2: replace this static-target loop with per-tick force computation:
        #   poses = read_all_poses()
        #   v = compute_velocities(poses, K_pos, K_rot)
        #   targets = poses + v * dt
        n_ticks = int(duration * STREAM_HZ)
        period = 1.0 / STREAM_HZ
        for _ in range(n_ticks):
            for cf, target in zip(self.cfs, targets):
                self.adapter.set_target(cf, target, yaw=0.0)
            self.timeHelper.sleep(period)

    def land(self):
        print("[particle_life] landing")
        # `land()` only works from high-level mode. After streaming setpoints, the
        # firmware is locked in low-level mode (per cmdFullState/cmdPosition docs).
        # notifySetpointsStop is per-Crazyflie; release the lock on each drone first.
        for cf in self.cfs:
            cf.notifySetpointsStop()
        self.allcfs.land(targetHeight=0.04, duration=LAND_DURATION)
        self.timeHelper.sleep(LAND_DURATION + 0.5)

    def run(self):
        try:
            self.takeoff()
            self.hover(HOVER_DURATION)
        except KeyboardInterrupt:
            print("[particle_life] interrupted, landing")
        finally:
            self.land()


def main():
    ctrl = ParticleLifeController()
    ctrl.run()


if __name__ == '__main__':
    main()
