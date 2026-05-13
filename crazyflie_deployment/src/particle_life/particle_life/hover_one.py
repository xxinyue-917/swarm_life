#!/usr/bin/env python3
"""
Single-drone hover test — picks one drone by name from the launched fleet
and flies only that one. Other enabled drones in crazyflies.yaml are
ignored (won't be armed / commanded).

Use this to walk through the fleet one-at-a-time and figure out which drones
are flight-worthy before trying any multi-drone scenario.

Run:
    ros2 launch particle_life hover_one.launch.py drone:=cf4
"""

import argparse
import sys

import rclpy
from crazyflie_py import Crazyswarm


HOVER_HEIGHT = 0.4
TAKEOFF_DURATION = 3.0
HOVER_DURATION = 5.0
LAND_DURATION = 3.0
ESTIMATOR_WARMUP = 8.0


def main():
    # Crazyswarm() calls rclpy.init() internally and consumes its own argv,
    # so parse the --drone arg before constructing the swarm.
    parser = argparse.ArgumentParser()
    parser.add_argument('--drone', required=True,
                        help="drone name to fly, e.g. cf4 (must match crazyflies.yaml)")
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    target_name = args.drone

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cfs_by_name = swarm.allcfs.crazyfliesByName

    cf = cfs_by_name.get(target_name)
    if cf is None:
        print(f"[hover_one] '{target_name}' not in connected fleet: "
              f"{list(cfs_by_name.keys())}")
        return

    print(f"[hover_one] flying ONLY {cf.prefix}")

    print("[hover_one] arming")
    cf.arm(True)
    timeHelper.sleep(0.5)

    print(f"[hover_one] Kalman warmup ({ESTIMATOR_WARMUP}s)")
    timeHelper.sleep(ESTIMATOR_WARMUP)

    print(f"[hover_one] takeoff -> {HOVER_HEIGHT} m")
    cf.takeoff(targetHeight=HOVER_HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 0.5)

    print(f"[hover_one] hovering for {HOVER_DURATION}s")
    timeHelper.sleep(HOVER_DURATION)

    print("[hover_one] landing")
    cf.land(targetHeight=0.04, duration=LAND_DURATION)
    timeHelper.sleep(LAND_DURATION + 0.5)

    print("[hover_one] disarming")
    cf.arm(False)


if __name__ == '__main__':
    main()
