#!/usr/bin/env python3
"""
Parse a rosbag from results/flights/ and plot Vicon (/poses) vs firmware-Kalman
(/cfN/pose) trajectories for every drone that appears in the bag.

Usage:
    python3 scripts/plot_bag.py                                # latest bag
    python3 scripts/plot_bag.py results/flights/flight_test_... # specific bag
    python3 scripts/plot_bag.py --save out.png                 # write PNG
    python3 scripts/plot_bag.py --csv                          # also dump per-drone CSVs

Panels:
  1. XY trajectory (Vicon solid, Kalman dashed) — bird's-eye
  2. z(t) — flags sudden altitude drops / crashes
  3. Vicon ‖position‖ residual from its mean — pure mocap drift/jitter
  4. Vicon − Kalman position error magnitude — estimator divergence
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


def _latest_bag(flights_dir):
    bags = sorted(
        d for d in glob.glob(os.path.join(flights_dir, '*'))
        if os.path.isdir(d) and glob.glob(os.path.join(d, 'metadata.yaml'))
    )
    if not bags:
        sys.exit(f"no bag dirs under {flights_dir}")
    return bags[-1]


def _open_reader(bag_path):
    storage = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)
    return reader


def _type_map(reader):
    return {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}


def read_bag(bag_path):
    """Return dict: drone_name -> {'vicon': (t, xyz), 'kalman': (t, xyz)}."""
    reader = _open_reader(bag_path)
    types = _type_map(reader)

    # Per-drone time + position arrays.
    vicon = defaultdict(lambda: {'t': [], 'p': []})
    kalman = defaultdict(lambda: {'t': [], 'p': []})

    t0 = None
    while reader.has_next():
        topic, raw, t_nsec = reader.read_next()
        if topic not in types:
            continue
        msg = deserialize_message(raw, types[topic])
        t_sec = t_nsec * 1e-9
        if t0 is None:
            t0 = t_sec
        t_rel = t_sec - t0

        if topic == '/poses':
            for np_ in msg.poses:
                p = np_.pose.position
                vicon[np_.name]['t'].append(t_rel)
                vicon[np_.name]['p'].append([p.x, p.y, p.z])
        elif topic.endswith('/pose') and topic.startswith('/cf'):
            name = topic.split('/')[1]
            p = msg.pose.position
            kalman[name]['t'].append(t_rel)
            kalman[name]['p'].append([p.x, p.y, p.z])

    def _pack(d):
        out = {}
        for k, v in d.items():
            if not v['t']:
                continue
            out[k] = (np.asarray(v['t']), np.asarray(v['p']))
        return out

    return _pack(vicon), _pack(kalman)


def _interp_to(t_ref, t_src, xyz_src):
    """Linearly resample xyz_src(t_src) onto t_ref."""
    return np.column_stack([np.interp(t_ref, t_src, xyz_src[:, i]) for i in range(3)])


def plot(bag_path, save_path=None, dump_csv=False):
    vicon, kalman = read_bag(bag_path)
    drones = sorted(set(vicon) | set(kalman))
    if not drones:
        sys.exit("no drone poses found in bag")

    cmap = plt.get_cmap('tab10')
    colors = {d: cmap(i % 10) for i, d in enumerate(drones)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(os.path.basename(bag_path), fontsize=11)

    # --- 1. XY trajectory ---------------------------------------------
    ax = axes[0, 0]
    for d in drones:
        if d in vicon:
            t, p = vicon[d]
            ax.plot(p[:, 0], p[:, 1], color=colors[d], lw=1.2,
                    label=f"{d} vicon")
            ax.scatter(p[0, 0], p[0, 1], color=colors[d], marker='o', s=40, zorder=3)
            ax.scatter(p[-1, 0], p[-1, 1], color=colors[d], marker='X', s=60, zorder=3)
        if d in kalman:
            _, pk = kalman[d]
            ax.plot(pk[:, 0], pk[:, 1], color=colors[d], lw=1.0, ls='--',
                    alpha=0.7, label=f"{d} kalman")
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('XY (solid=Vicon, dashed=Kalman; o=start, X=end)')
    ax.set_aspect('equal', 'box')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='best')

    # --- 2. z(t) -------------------------------------------------------
    ax = axes[0, 1]
    for d in drones:
        if d in vicon:
            t, p = vicon[d]
            ax.plot(t, p[:, 2], color=colors[d], lw=1.0, label=f"{d} vicon")
        if d in kalman:
            tk, pk = kalman[d]
            ax.plot(tk, pk[:, 2], color=colors[d], lw=1.0, ls='--', alpha=0.7,
                    label=f"{d} kalman")
    ax.set_xlabel('t (s)'); ax.set_ylabel('z (m)')
    ax.set_title('Altitude — sudden drop = crash')
    ax.axhline(0.1, color='red', ls=':', lw=1, alpha=0.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # --- 3. Vicon position residual from mean --------------------------
    # Computed over the takeoff-and-hover window: anything past the 1st sample.
    # Gives a feel for mocap jitter + true motion magnitude.
    ax = axes[1, 0]
    for d in drones:
        if d not in vicon:
            continue
        t, p = vicon[d]
        # Detrend each axis by its median (robust to a single takeoff jump).
        ctr = np.median(p, axis=0)
        resid = np.linalg.norm(p - ctr, axis=1)
        ax.plot(t, resid * 1000, color=colors[d], lw=1.0, label=d)
    ax.set_xlabel('t (s)'); ax.set_ylabel('|p − median(p)| (mm)')
    ax.set_title('Vicon position spread from median — pure motion + jitter')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # --- 4. Vicon − Kalman error ---------------------------------------
    ax = axes[1, 1]
    for d in drones:
        if d not in vicon or d not in kalman:
            continue
        tv, pv = vicon[d]
        tk, pk = kalman[d]
        # Resample Vicon onto Kalman timeline (Vicon ~100 Hz, Kalman 10 Hz).
        lo = max(tv.min(), tk.min())
        hi = min(tv.max(), tk.max())
        mask = (tk >= lo) & (tk <= hi)
        if not mask.any():
            continue
        t_eval = tk[mask]
        pv_on_tk = _interp_to(t_eval, tv, pv)
        err = np.linalg.norm(pv_on_tk - pk[mask], axis=1)
        ax.plot(t_eval, err * 1000, color=colors[d], lw=1.0, label=d)
    ax.set_xlabel('t (s)'); ax.set_ylabel('|Vicon − Kalman| (mm)')
    ax.set_title('Estimator error — Kalman drifting from Vicon truth')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if dump_csv:
        for d in drones:
            if d in vicon:
                t, p = vicon[d]
                out = os.path.join(bag_path, f'{d}_vicon.csv')
                np.savetxt(out, np.column_stack([t, p]), delimiter=',',
                           header='t,x,y,z', comments='')
                print(f"  wrote {out}")
            if d in kalman:
                t, p = kalman[d]
                out = os.path.join(bag_path, f'{d}_kalman.csv')
                np.savetxt(out, np.column_stack([t, p]), delimiter=',',
                           header='t,x,y,z', comments='')
                print(f"  wrote {out}")

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"saved -> {save_path}")
    else:
        plt.show()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.abspath(os.path.join(here, '..', 'results', 'flights'))

    ap = argparse.ArgumentParser()
    ap.add_argument('bag', nargs='?', default=None,
                    help='bag dir (defaults to latest in results/flights/)')
    ap.add_argument('--save', default=None, help='write PNG instead of opening a window')
    ap.add_argument('--csv', action='store_true', help='also dump per-drone CSVs into bag dir')
    args = ap.parse_args()

    bag_path = args.bag or _latest_bag(default_dir)
    print(f"bag: {bag_path}")
    plot(bag_path, save_path=args.save, dump_csv=args.csv)


if __name__ == '__main__':
    main()
