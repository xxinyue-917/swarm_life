#!/usr/bin/env python3
"""
Multi-drone circle flight: every enabled drone simultaneously flies one
RADIUS-meter circle centered on its own start position. With 1m drone spacing
and 0.3m radius, neighbours stay ~0.4m apart at closest approach.

Run:
    ros2 launch particle_life circle_all.launch.py
"""

import math

from crazyflie_py import Crazyswarm


HOVER_HEIGHT = 0.5
TAKEOFF_DURATION = 3.0
LAND_DURATION = 3.0
ESTIMATOR_WARMUP = 8.0

RADIUS = 0.3           # circle radius (m)
N_STEPS = 16           # discretization of the circle
STEP_DURATION = 0.6    # seconds per step (16 * 0.6 = 9.6s per revolution)


def _circle_relative_deltas(radius, n):
    """Yield successive (dx, dy, 0) offsets that trace a closed CCW circle.

    Path from the drone's takeoff-height hover: first hop to entry point
    (radius, 0) on the circle, then walk N segments around the circle, then
    return to centre. All steps are relative to the previous waypoint, so
    they compose into the full path regardless of where the drone started.
    """
    # Entry: takeoff position -> (radius, 0) on the circle.
    deltas = [(radius, 0.0, 0.0)]
    # Around: deltas between consecutive circle points.
    prev = (radius, 0.0)
    for k in range(1, n + 1):
        theta = 2.0 * math.pi * k / n
        cur = (radius * math.cos(theta), radius * math.sin(theta))
        deltas.append((cur[0] - prev[0], cur[1] - prev[1], 0.0))
        prev = cur
    # Exit: circle entry back to centre.
    deltas.append((-radius, 0.0, 0.0))
    return deltas


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    cfs = list(allcfs.crazyflies)
    if not cfs:
        print("[circle_all] no drones connected, aborting")
        return

    names = [cf.prefix for cf in cfs]
    print(f"[circle_all] flying {len(cfs)} drone(s): {names}")
    print(f"[circle_all] radius={RADIUS} m, {N_STEPS} steps, {STEP_DURATION}s/step")

    print("[circle_all] arming")
    allcfs.arm(True)
    timeHelper.sleep(0.5)

    print(f"[circle_all] Kalman warmup ({ESTIMATOR_WARMUP}s)")
    timeHelper.sleep(ESTIMATOR_WARMUP)

    print(f"[circle_all] takeoff -> {HOVER_HEIGHT} m")
    allcfs.takeoff(targetHeight=HOVER_HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 0.5)

    deltas = _circle_relative_deltas(RADIUS, N_STEPS)
    try:
        for i, wp in enumerate(deltas):
            phase = (
                'entry' if i == 0 else
                'exit'  if i == len(deltas) - 1 else
                f'arc {i}/{N_STEPS}'
            )
            print(f"[circle_all] step {i + 1}/{len(deltas)} ({phase}) -> "
                  f"({wp[0]:+.3f}, {wp[1]:+.3f})")
            for cf in cfs:
                cf.goTo(wp, yaw=0.0, duration=STEP_DURATION, relative=True)
            timeHelper.sleep(STEP_DURATION + 0.1)
    except KeyboardInterrupt:
        print("[circle_all] interrupted")

    print("[circle_all] landing")
    allcfs.land(targetHeight=0.04, duration=LAND_DURATION)
    timeHelper.sleep(LAND_DURATION + 0.5)

    print("[circle_all] disarming")
    allcfs.arm(False)


if __name__ == '__main__':
    main()
