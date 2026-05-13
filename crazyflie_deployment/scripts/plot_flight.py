#!/usr/bin/env python3
"""
Plot a flight log written by FlightLogger.

Usage:
    python3 scripts/plot_flight.py results/flights/flight_20260513_171530_force.csv
    python3 scripts/plot_flight.py                      # auto-pick the latest CSV
    python3 scripts/plot_flight.py --save out.png       # write image, no GUI

Four panels:
  1. XY trajectories per drone (the bird's-eye view)
  2. Altitude (z) per drone vs time — flags loss-of-altitude crashes
  3. Commanded speed per drone — flags out-of-control accel right before a crash
  4. Pairwise minimum separation vs time — flags near-collisions
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _latest_csv(results_dir):
    csvs = sorted(glob.glob(os.path.join(results_dir, '*.csv')))
    if not csvs:
        sys.exit(f"no CSVs found under {results_dir}")
    return csvs[-1]


def _species_colors(species_per_drone):
    """Stable color per species so trajectories are easy to read."""
    cmap = plt.get_cmap('tab10')
    return {d: cmap(s % 10) for d, s in species_per_drone.items()}


def plot(csv_path, save_path=None):
    df = pd.read_csv(csv_path)
    drones = list(df['drone'].unique())
    species_map = df.drop_duplicates('drone').set_index('drone')['species'].to_dict()
    colors = _species_colors(species_map)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(os.path.basename(csv_path), fontsize=11)

    # --- 1. XY trajectories ----------------------------------------------------
    ax = axes[0, 0]
    for d in drones:
        sub = df[df['drone'] == d]
        ax.plot(sub['x'], sub['y'], color=colors[d], lw=1.2, alpha=0.8, label=d)
        ax.scatter(sub['x'].iloc[0], sub['y'].iloc[0],
                   color=colors[d], marker='o', s=40, zorder=3)  # start
        ax.scatter(sub['x'].iloc[-1], sub['y'].iloc[-1],
                   color=colors[d], marker='X', s=60, zorder=3)  # end
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('XY trajectories (o=start, X=end)')
    ax.set_aspect('equal', 'box')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='best')

    # --- 2. Altitude vs time --------------------------------------------------
    ax = axes[0, 1]
    for d in drones:
        sub = df[df['drone'] == d]
        ax.plot(sub['t_sec'], sub['z'], color=colors[d], lw=1.0, label=d)
    ax.set_xlabel('t (s)'); ax.set_ylabel('z (m)')
    ax.set_title('Altitude — sudden drops = crash candidates')
    ax.axhline(0.1, color='red', ls=':', lw=1, alpha=0.6, label='floor 0.1m')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='best')

    # --- 3. Commanded speed vs time -------------------------------------------
    ax = axes[1, 0]
    for d in drones:
        sub = df[df['drone'] == d]
        ax.plot(sub['t_sec'], sub['speed_cmd'], color=colors[d], lw=1.0, label=d)
    ax.set_xlabel('t (s)'); ax.set_ylabel('|v_cmd| (m/s)')
    ax.set_title('Commanded speed (post-clamp)')
    ax.grid(alpha=0.3)

    # --- 4. Pairwise minimum separation ---------------------------------------
    ax = axes[1, 1]
    # Build per-tick matrices; assume every drone has a row per tick.
    by_tick = df.pivot_table(index='tick', columns='drone',
                             values=['x', 'y'], aggfunc='first')
    ticks = by_tick.index.values
    t_per_tick = df.drop_duplicates('tick').sort_values('tick')['t_sec'].values
    min_sep = np.full(len(ticks), np.nan)
    for k, tick in enumerate(ticks):
        xs = by_tick.loc[tick, 'x'].values
        ys = by_tick.loc[tick, 'y'].values
        # All pairwise distances
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        d2 = dx * dx + dy * dy
        np.fill_diagonal(d2, np.inf)
        min_sep[k] = float(np.sqrt(d2.min()))
    ax.plot(t_per_tick, min_sep, 'k-', lw=1.2)
    ax.axhline(0.30, color='red', ls='--', lw=1, label='min_separation=0.30m')
    ax.set_xlabel('t (s)'); ax.set_ylabel('min pairwise dist (m)')
    ax.set_title('Closest pair — anything below the red line is a near-miss')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved -> {save_path}")
    else:
        plt.show()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.abspath(os.path.join(here, '..', 'results', 'flights'))

    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='?', default=None,
                    help='path to flight CSV (defaults to latest in results/flights/)')
    ap.add_argument('--save', default=None, help='write PNG instead of opening a window')
    args = ap.parse_args()

    csv_path = args.csv or _latest_csv(default_dir)
    plot(csv_path, save_path=args.save)


if __name__ == '__main__':
    main()
