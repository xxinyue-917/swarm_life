#!/usr/bin/env python3
"""
Single-drone flight test: takeoff -> small box waypoints -> land.

Uses Crazyswarm2's high-level commander (goTo) so the firmware plans a smooth
polynomial trajectory between waypoints. No setpoint streaming — keeps this
script independent of the M1 SetpointAdapter pipeline.

Run with cf1 enabled in crazyflies.yaml and Vicon publishing /cf1/pose:

    ros2 launch particle_life flight_test.launch.py
"""

from crazyflie_py import Crazyswarm


HOVER_HEIGHT = 0.5       # meters (low for first flight)
TAKEOFF_DURATION = 3.0
WAYPOINT_DURATION = 3.0  # per leg
LAND_DURATION = 3.0
ESTIMATOR_WARMUP = 8.0   # seconds; let Kalman converge on Vicon extpose
                         # before the firmware tries to control attitude

# 0.3 m box RELATIVE to the *current* position at each step. With z=0 the
# drone stays at takeoff height (HOVER_HEIGHT) throughout. Each waypoint is
# an offset from where the drone is now, not from the takeoff origin.
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
    cfs = allcfs.crazyflies

    print(f"[flight_test] {len(cfs)} drone(s) connected")

    print(f"[flight_test] warming up Kalman estimator for {ESTIMATOR_WARMUP}s "
          f"(needs Vicon extpose to converge before takeoff)")
    timeHelper.sleep(ESTIMATOR_WARMUP)

    print(f"[flight_test] takeoff to {HOVER_HEIGHT} m")
    allcfs.takeoff(targetHeight=HOVER_HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 0.5)

    try:
        for i, wp in enumerate(BOX_REL):
            print(f"[flight_test] waypoint {i + 1}/{len(BOX_REL)} (relative) -> {wp}")
            for cf in cfs:
                cf.goTo(wp, yaw=0.0, duration=WAYPOINT_DURATION, relative=True)
            timeHelper.sleep(WAYPOINT_DURATION + 0.3)
    except KeyboardInterrupt:
        print("[flight_test] interrupted")

    print("[flight_test] landing")
    allcfs.land(targetHeight=0.04, duration=LAND_DURATION)
    timeHelper.sleep(LAND_DURATION + 0.5)


if __name__ == '__main__':
    main()
