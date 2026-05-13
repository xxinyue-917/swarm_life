#!/usr/bin/env python3
"""
FlightLogger — append one row per drone per tick to a CSV for offline analysis.

One file per run, named with a timestamp under `crazyflie_deployment/results/flights/`.
Schema (wide → long, so plotting filters by drone name trivially):

    t_sec, tick, drone, species, x, y, z, target_x, target_y, target_z, vx_cmd, vy_cmd, speed_cmd

The logger is intentionally dumb: no buffering games, no threads. Python's CSV
writer + flush() per tick is plenty at 30 Hz × 10 drones (≈300 rows/s).
"""

import csv
import os
import time
from datetime import datetime


COLUMNS = [
    't_sec', 'tick', 'drone', 'species',
    'x', 'y', 'z',
    'target_x', 'target_y', 'target_z',
    'vx_cmd', 'vy_cmd', 'speed_cmd',
]


class FlightLogger:

    def __init__(self, drone_names, species, out_dir=None, run_tag=None):
        """Open a new CSV file for this run.

        Args:
            drone_names : ordered list, e.g. ['cf1', 'cf2', ...].
            species     : per-drone species index (same order as drone_names).
            out_dir     : override default results/flights/ location.
            run_tag     : suffix appended to the filename for easy identification.
        """
        self.drone_names = list(drone_names)
        self.species = list(species)
        if out_dir is None:
            # results/flights/ at the repo root: ../../../results/flights from this file
            here = os.path.dirname(os.path.abspath(__file__))
            out_dir = os.path.abspath(os.path.join(
                here, '..', '..', '..', '..', 'results', 'flights'))
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'flight_{stamp}'
        if run_tag:
            fname += f'_{run_tag}'
        fname += '.csv'
        self.path = os.path.join(out_dir, fname)
        self._fh = open(self.path, 'w', newline='')
        self._writer = csv.writer(self._fh)
        self._writer.writerow(COLUMNS)
        self._t0 = time.time()
        print(f"[flight_logger] writing -> {self.path}")

    def log_tick(self, tick, poses_xy, z_actual, targets_xyz, v_cmd):
        """Write one row per drone for this tick.

        poses_xy    : (N, 2)
        z_actual    : (N,) or scalar
        targets_xyz : (N, 3) commanded target
        v_cmd       : (N, 2) commanded XY velocity (post-clamp, m/s)
        """
        t = time.time() - self._t0
        for i, name in enumerate(self.drone_names):
            zi = z_actual[i] if hasattr(z_actual, '__len__') else float(z_actual)
            vx, vy = float(v_cmd[i, 0]), float(v_cmd[i, 1])
            self._writer.writerow([
                f'{t:.4f}', tick, name, int(self.species[i]),
                f'{poses_xy[i, 0]:.4f}', f'{poses_xy[i, 1]:.4f}', f'{zi:.4f}',
                f'{targets_xyz[i, 0]:.4f}', f'{targets_xyz[i, 1]:.4f}', f'{targets_xyz[i, 2]:.4f}',
                f'{vx:.4f}', f'{vy:.4f}', f'{(vx*vx + vy*vy) ** 0.5:.4f}',
            ])
        self._fh.flush()

    def close(self):
        if not self._fh.closed:
            self._fh.close()
            print(f"[flight_logger] closed {self.path}")
