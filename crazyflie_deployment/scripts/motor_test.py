#!/usr/bin/env python3
"""
Bare-metal motor test — no ROS, no Vicon, no flight stack.

Spins each motor of the Crazyflie one at a time at low power, then runs a
brief 4-motor balanced thrust pulse. Use this to verify on a brand-new
Crazyflie that:
  1. The radio link works (it'll fail fast if URI is wrong)
  2. All 4 motors spin (no dead motor / cold solder)
  3. All 4 motors spin together cleanly (no flipping when balanced)
  4. Props are right side up (motors should sound the same, not labored)

Hold the drone gently or rest it on a soft surface — propwash will lift it
~1 cm during the balanced-thrust step. Stop the script with Ctrl-C anytime.

Usage:
    python3 motor_test.py
    python3 motor_test.py --uri radio://0/80/2M/E7E7E7E7E7   # default
"""

import argparse
import time

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


DEFAULT_URI = 'radio://0/80/2M/E7E7E7E7E7'
SPIN_POWER = 10000       # ~15% of max (max is 65535). Just enough to spin.
LIFT_POWER = 35000       # ~55% — should make drone twitch but not climb.
SPIN_DURATION = 1.0      # per motor
LIFT_DURATION = 0.8


def set_motor(cf, motor: int, power: int):
    """motor: 1..4 (M1=front-right, M2=back-right, M3=back-left, M4=front-left)."""
    cf.param.set_value(f'motorPowerSet.m{motor}', str(power))


def all_motors(cf, power: int):
    for m in range(1, 5):
        set_motor(cf, m, power)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', default=DEFAULT_URI)
    args = parser.parse_args()

    cflib.crtp.init_drivers()
    print(f"[motor_test] connecting to {args.uri}")

    with SyncCrazyflie(args.uri) as scf:
        cf = scf.cf
        print("[motor_test] connected. Enabling manual motor control.")
        cf.param.set_value('motorPowerSet.enable', '1')
        time.sleep(0.3)

        # 1) Per-motor spin test.
        for m in range(1, 5):
            label = {1: 'M1 (front-right)', 2: 'M2 (back-right)',
                     3: 'M3 (back-left)', 4: 'M4 (front-left)'}[m]
            print(f"[motor_test] spinning {label} @ {SPIN_POWER}")
            set_motor(cf, m, SPIN_POWER)
            time.sleep(SPIN_DURATION)
            set_motor(cf, m, 0)
            time.sleep(0.4)

        # 2) Balanced 4-motor pulse — drone may twitch / leave the ground slightly.
        print(f"[motor_test] all 4 motors @ {LIFT_POWER} for {LIFT_DURATION}s")
        all_motors(cf, LIFT_POWER)
        time.sleep(LIFT_DURATION)
        all_motors(cf, 0)

        # 3) Hand control back to the firmware.
        cf.param.set_value('motorPowerSet.enable', '0')
        print("[motor_test] done. If any motor was silent or weak, that's your problem.")
        print("[motor_test] If the drone tipped during the 4-motor pulse, a prop is")
        print("              installed wrong (M1/M3=CCW, M2/M4=CW) or upside-down.")


if __name__ == '__main__':
    main()
