#!/usr/bin/env python3
"""
Simplest possible Crazyswarm2 hover test:
  arm -> Kalman warmup -> takeoff -> hover -> land -> disarm.

No waypoints, no setpoint streaming. Just verify the firmware will leave the
ground and stay put under PID + Vicon (single marker). If this works, the
whole hardware/Vicon/ROS pipeline is sane.

Run:
    ros2 launch particle_life hover_test.launch.py
"""

from crazyflie_py import Crazyswarm


HOVER_HEIGHT = 0.4       # meters (low for first flight)
TAKEOFF_DURATION = 3.0
HOVER_DURATION = 5.0
LAND_DURATION = 3.0
ESTIMATOR_WARMUP = 3.0


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    print(f"[hover_test] {len(allcfs.crazyflies)} drone(s) connected")

    # 2024+ firmware ships with auto-arm disabled by default. Call the Arm
    # service explicitly before any motion command.
    print("[hover_test] arming")
    allcfs.arm(True)
    timeHelper.sleep(0.5)

    print(f"[hover_test] Kalman warmup ({ESTIMATOR_WARMUP}s)")
    timeHelper.sleep(ESTIMATOR_WARMUP)

    print(f"[hover_test] takeoff -> {HOVER_HEIGHT} m")
    allcfs.takeoff(targetHeight=HOVER_HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 0.5)

    print(f"[hover_test] hovering for {HOVER_DURATION}s")
    timeHelper.sleep(HOVER_DURATION)

    print("[hover_test] landing")
    allcfs.land(targetHeight=0.04, duration=LAND_DURATION)
    timeHelper.sleep(LAND_DURATION + 0.5)

    print("[hover_test] disarming")
    allcfs.arm(False)


if __name__ == '__main__':
    main()
