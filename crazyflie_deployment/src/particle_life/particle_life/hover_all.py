#!/usr/bin/env python3
"""
Hover every enabled drone in crazyflies.yaml for 5 seconds, then land.

Flies whatever the Crazyswarm2 server hands us in `allcfs.crazyflies` — to
change the set of drones, edit `enabled: true/false` in crazyflies.yaml and
re-launch (no script edit needed).

Run:
    ros2 launch particle_life hover_all.launch.py
"""

from crazyflie_py import Crazyswarm


HOVER_HEIGHT = 0.4
TAKEOFF_DURATION = 3.0
HOVER_DURATION = 5.0
LAND_DURATION = 3.0
ESTIMATOR_WARMUP = 8.0   # generous: lets "fully connected" + Kalman settle


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    cfs = list(allcfs.crazyflies)
    if not cfs:
        print("[hover_all] no drones connected, aborting")
        return

    names = [cf.prefix for cf in cfs]
    print(f"[hover_all] flying {len(cfs)} drone(s): {names}")

    print("[hover_all] arming")
    allcfs.arm(True)
    timeHelper.sleep(0.5)

    print(f"[hover_all] Kalman warmup ({ESTIMATOR_WARMUP}s)")
    timeHelper.sleep(ESTIMATOR_WARMUP)

    print(f"[hover_all] takeoff -> {HOVER_HEIGHT} m")
    allcfs.takeoff(targetHeight=HOVER_HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 0.5)

    print(f"[hover_all] hovering for {HOVER_DURATION}s")
    timeHelper.sleep(HOVER_DURATION)

    print("[hover_all] landing")
    allcfs.land(targetHeight=0.04, duration=LAND_DURATION)
    timeHelper.sleep(LAND_DURATION + 0.5)

    print("[hover_all] disarming")
    allcfs.arm(False)


if __name__ == '__main__':
    main()
