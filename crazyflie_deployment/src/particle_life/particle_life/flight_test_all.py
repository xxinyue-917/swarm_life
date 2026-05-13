#!/usr/bin/env python3
"""
Multi-drone box flight: every enabled drone takes off and walks the same
0.3 m box *relative to its own start position*. With unique starts in
crazyflies.yaml the boxes are spatially separated → no collision.

Run:
    ros2 launch particle_life flight_test_all.launch.py
"""

from crazyflie_py import Crazyswarm


HOVER_HEIGHT = 0.5
TAKEOFF_DURATION = 3.0
WAYPOINT_DURATION = 3.0
LAND_DURATION = 3.0
ESTIMATOR_WARMUP = 8.0

# 0.3 m box RELATIVE to each drone's *current* position at each step.
# With z=0 the drones stay at takeoff height (HOVER_HEIGHT) throughout.
BOX_REL = [
    ( 0.3,  0.0, 0.0),
    ( 0.0,  0.3, 0.0),
    (-0.3,  0.0, 0.0),
    ( 0.0, -0.3, 0.0),
]


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    cfs = list(allcfs.crazyflies)
    if not cfs:
        print("[flight_test_all] no drones connected, aborting")
        return

    names = [cf.prefix for cf in cfs]
    print(f"[flight_test_all] flying {len(cfs)} drone(s): {names}")

    print("[flight_test_all] arming")
    allcfs.arm(True)
    timeHelper.sleep(0.5)

    print(f"[flight_test_all] Kalman warmup ({ESTIMATOR_WARMUP}s)")
    timeHelper.sleep(ESTIMATOR_WARMUP)

    print(f"[flight_test_all] takeoff -> {HOVER_HEIGHT} m")
    allcfs.takeoff(targetHeight=HOVER_HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 0.5)

    try:
        for i, wp in enumerate(BOX_REL):
            print(f"[flight_test_all] waypoint {i + 1}/{len(BOX_REL)} (relative) -> {wp}")
            for cf in cfs:
                cf.goTo(wp, yaw=0.0, duration=WAYPOINT_DURATION, relative=True)
            timeHelper.sleep(WAYPOINT_DURATION + 0.3)
    except KeyboardInterrupt:
        print("[flight_test_all] interrupted")

    print("[flight_test_all] landing")
    allcfs.land(targetHeight=0.04, duration=LAND_DURATION)
    timeHelper.sleep(LAND_DURATION + 0.5)

    print("[flight_test_all] disarming")
    allcfs.arm(False)


if __name__ == '__main__':
    main()
