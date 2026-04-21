#!/usr/bin/env python3
"""
Particle Life swarm controller for Crazyflie drones.

Milestone scaffolding:
  M1 (current): takeoff, hover, land — verifies ROS <-> Crazyswarm2 pipeline
  M2 (next):    add K_pos force computation for 2 drones
  M3 (final):   10-drone multi-species with K_pos + K_rot
"""

import rclpy
from rclpy.node import Node
from crazyflie_py import Crazyswarm
import numpy as np


HOVER_HEIGHT = 1.0   # meters
TAKEOFF_DURATION = 3.0
HOVER_DURATION = 5.0
LAND_DURATION = 3.0


class ParticleLifeController:
    """High-level controller that drives Crazyflies through particle life behaviors."""

    def __init__(self):
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.allcfs = self.swarm.allcfs
        self.n = len(self.allcfs.crazyflies)
        print(f"[particle_life] {self.n} drones connected")

    def takeoff(self):
        print(f"[particle_life] taking off to {HOVER_HEIGHT}m")
        self.allcfs.takeoff(targetHeight=HOVER_HEIGHT, duration=TAKEOFF_DURATION)
        self.timeHelper.sleep(TAKEOFF_DURATION + 0.5)

    def hover(self, duration: float):
        print(f"[particle_life] hovering for {duration}s (M1 scaffold)")
        # TODO M2: replace with force computation loop
        #   for each cf: read pose, compute F = K_pos * ..., integrate to target pos, cmdPosition(target)
        self.timeHelper.sleep(duration)

    def land(self):
        print("[particle_life] landing")
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
