#!/usr/bin/env python3
"""
General 3D Vicon trajectory replay for any bag dir with per-drone CSVs.

Usage:
    python3 scripts/replay_3d.py <bag_dir> [--highlight cf7] [--t_start 0] [--t_end auto]

Outputs:
    <bag_dir>/replay_3d.png   — static 3D figure (all drones + highlight)
    <bag_dir>/replay_3d.mp4   — 30 fps animation, optionally focused on a drone
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_drones(bag_dir):
    out = {}
    for vp in sorted(glob.glob(os.path.join(bag_dir, 'cf*_vicon.csv'))):
        d = os.path.basename(vp).replace('_vicon.csv', '')
        v = np.loadtxt(vp, delimiter=',', skiprows=1)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        kp = os.path.join(bag_dir, f'{d}_kalman.csv')
        k = None
        if os.path.exists(kp):
            k = np.loadtxt(kp, delimiter=',', skiprows=1)
            if k.ndim == 1:
                k = k.reshape(1, -1)
        out[d] = {'v': (v[:, 0], v[:, 1:4]),
                  'k': (k[:, 0], k[:, 1:4]) if k is not None else None}
    return out


def _set_axes(ax):
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag_dir')
    ap.add_argument('--highlight', default=None,
                    help='drone name to draw thicker / Kalman dashed')
    ap.add_argument('--t_start', type=float, default=None)
    ap.add_argument('--t_end', type=float, default=None)
    ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()

    data = load_drones(args.bag_dir)
    if not data:
        sys.exit(f"no per-drone CSVs in {args.bag_dir}; run plot_bag.py --csv first")
    drones = sorted(data.keys())

    # Global time range across all drones
    t_lo = min(d['v'][0][0] for d in data.values())
    t_hi = max(d['v'][0][-1] for d in data.values())
    # Normalize so t=0 is bag start
    for d in drones:
        tv, p = data[d]['v']
        data[d]['v'] = (tv - t_lo, p)
        if data[d]['k'] is not None:
            tk, pk = data[d]['k']
            data[d]['k'] = (tk - t_lo, pk)
    t_hi -= t_lo
    t_lo = 0.0

    t_start = args.t_start if args.t_start is not None else t_lo
    t_end = args.t_end if args.t_end is not None else t_hi

    cmap = plt.get_cmap('tab10')
    colors = {d: cmap(i % 10) for i, d in enumerate(drones)}
    hl = args.highlight

    # ============ STATIC ============
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    _set_axes(ax)
    for d in drones:
        tv, p = data[d]['v']
        lw = 2.4 if d == hl else 1.0
        alpha = 1.0 if d == hl else 0.55
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=colors[d], lw=lw,
                alpha=alpha, label=d)
        ax.scatter(p[0, 0], p[0, 1], p[0, 2], marker='o', s=40,
                   color=colors[d], edgecolors='black')
        ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2], marker='s', s=40,
                   color=colors[d], edgecolors='black')
        if d == hl and data[d]['k'] is not None:
            tk, pk = data[d]['k']
            ax.plot(pk[:, 0], pk[:, 1], pk[:, 2], ls='--',
                    color=colors[d], lw=1.2, alpha=0.5)
    ax.legend(fontsize=9, loc='upper left')
    title = f'{os.path.basename(args.bag_dir)}'
    if hl:
        title += f'  (highlight: {hl})'
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=25, azim=-60)
    out_png = os.path.join(args.bag_dir, 'replay_3d.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f'saved {out_png}')

    # ============ ANIMATION ============
    fig2 = plt.figure(figsize=(11, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    _set_axes(ax2)

    trails = {}
    dots = {}
    for d in drones:
        lw = 2.5 if d == hl else 1.0
        alpha = 1.0 if d == hl else 0.45
        (line,) = ax2.plot([], [], [], color=colors[d], lw=lw, alpha=alpha,
                           label=d)
        (dot,) = ax2.plot([], [], [], 'o', color=colors[d],
                          markersize=14 if d == hl else 7,
                          markeredgecolor='black', mew=0.8)
        trails[d] = line
        dots[d] = dot

    if hl:
        ax2.legend(fontsize=9, loc='upper left')

    time_text = ax2.text2D(0.02, 0.97, '', transform=ax2.transAxes,
                           fontsize=13, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white',
                                     alpha=0.85))

    fps = args.fps
    frames = np.arange(t_start, t_end, 1.0 / fps)

    def update(t_now):
        for d in drones:
            tv, p = data[d]['v']
            mask = tv <= t_now
            if mask.any():
                trails[d].set_data(p[mask, 0], p[mask, 1])
                trails[d].set_3d_properties(p[mask, 2])
                dots[d].set_data([p[mask, 0][-1]], [p[mask, 1][-1]])
                dots[d].set_3d_properties([p[mask, 2][-1]])
        time_text.set_text(f't = {t_now:5.2f} s')
        ax2.view_init(elev=25, azim=-60 + (t_now - t_start) * 2.0)
        return list(trails.values()) + list(dots.values()) + [time_text]

    anim = animation.FuncAnimation(fig2, update, frames=frames,
                                   interval=1000.0 / fps, blit=False,
                                   repeat=False)

    out_mp4 = os.path.join(args.bag_dir, 'replay_3d.mp4')
    try:
        anim.save(out_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2500))
        print(f'saved {out_mp4}')
    except Exception as e:
        out_gif = os.path.join(args.bag_dir, 'replay_3d.gif')
        anim.save(out_gif, writer=animation.PillowWriter(fps=fps))
        print(f'ffmpeg unavailable ({e}); saved gif: {out_gif}')


if __name__ == '__main__':
    main()
