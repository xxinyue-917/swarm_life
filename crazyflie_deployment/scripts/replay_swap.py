#!/usr/bin/env python3
"""
Replay the hover_all_20260518_151510 Vicon swap event.

Reads per-drone CSVs already dumped into the bag dir, produces:
  swap_trajectory.png  — time-colored XY trajectories of all 9 drones with
                          Vicon (solid) vs firmware Kalman (dashed), plus
                          declared start positions and key event markers.
  swap_animation.mp4   — drone positions over time, takeoff at t=33.5s.

Run after `plot_bag.py --csv` has written the CSVs.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation

BAG = '/home/xxinyue/swarm_life/crazyflie_deployment/results/flights/hover_all_20260518_151510'

DECLARED = {
    'cf1': (1.0, 1.0), 'cf2': (1.0, 0.0), 'cf3': (1.0, -1.0),
    'cf4': (0.0, 1.0), 'cf5': (0.0, 0.0), 'cf6': (-1.0, 0.0),
    'cf7': (-1.0, 1.0), 'cf8': (-1.0, -1.0), 'cf9': (0.0, -1.0),
}
DRONES = list(DECLARED.keys())
FAILED = {'cf6', 'cf8', 'cf9'}

TAKEOFF_T = 33.5      # bag-relative seconds (takeoff command issued)
SWAP_CF8_T = 35.02    # cf8's 86 cm Vicon jump


def load(d, kind):
    p = os.path.join(BAG, f'{d}_{kind}.csv')
    if not os.path.exists(p):
        return None
    a = np.loadtxt(p, delimiter=',', skiprows=1)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a[:, 0], a[:, 1:4]


def main():
    data = {d: {'v': load(d, 'vicon'), 'k': load(d, 'kalman')} for d in DRONES}

    # =================== STATIC PLOT ===================
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, focus in zip(axes, ['all 9 drones', 'cf6 / cf8 / cf9 only']):
        ax.set_title(f'Vicon XY trajectory — {focus}\n'
                     f'(color = time; o = declared start; ■ = Vicon final)')
        # Declared starts
        for d, (x, y) in DECLARED.items():
            color = 'red' if d in FAILED else 'gray'
            ax.scatter(x, y, marker='o', s=80, facecolors='none',
                       edgecolors=color, linewidths=1.5, zorder=4)
            ax.annotate(d, (x, y), xytext=(6, 6), textcoords='offset points',
                        fontsize=9, color=color, fontweight='bold' if d in FAILED else 'normal')

        for d in DRONES:
            if focus == 'cf6 / cf8 / cf9 only' and d not in FAILED:
                continue
            v = data[d]['v']
            if v is None:
                continue
            t, p = v
            # Time-colored line via LineCollection
            pts = p[:, :2].reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap='viridis',
                                norm=plt.Normalize(t.min(), t.max()),
                                linewidth=1.5, alpha=0.85)
            lc.set_array(t[:-1])
            ax.add_collection(lc)
            ax.scatter(p[-1, 0], p[-1, 1], marker='s', s=60,
                       color='black', zorder=5)
            ax.annotate(f'{d}_end', (p[-1, 0], p[-1, 1]),
                        xytext=(8, -10), textcoords='offset points',
                        fontsize=8, color='black')

            # Kalman dashed for failed drones only (clarity)
            if d in FAILED and focus == 'cf6 / cf8 / cf9 only':
                k = data[d]['k']
                if k is not None:
                    tk, pk = k
                    ax.plot(pk[:, 0], pk[:, 1], ':', color='magenta',
                            lw=1.0, alpha=0.6, label=f'{d} Kalman' if d == 'cf8' else None)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', 'box')
        ax.grid(alpha=0.3)
        ax.set_xlim(-2.5, 2.0)
        ax.set_ylim(-2.5, 2.0)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(0, 53))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='bag time (s)')
    cbar.ax.axhline(TAKEOFF_T, color='red', linewidth=2)
    cbar.ax.text(1.5, TAKEOFF_T, ' takeoff', color='red',
                 fontsize=9, va='center', ha='left',
                 transform=cbar.ax.get_yaxis_transform())
    cbar.ax.axhline(SWAP_CF8_T, color='orange', linewidth=2)
    cbar.ax.text(1.5, SWAP_CF8_T, ' cf8 swap', color='orange',
                 fontsize=9, va='center', ha='left',
                 transform=cbar.ax.get_yaxis_transform())

    plt.suptitle('hover_all_20260518_151510 — Vicon label swap replay',
                 fontsize=12, y=0.98)
    out = os.path.join(BAG, 'swap_trajectory.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f'saved {out}')

    # =================== ANIMATION ===================
    fig2, ax2 = plt.subplots(figsize=(9, 9))
    ax2.set_xlim(-2.5, 2.0)
    ax2.set_ylim(-2.5, 2.0)
    ax2.set_aspect('equal', 'box')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.grid(alpha=0.3)

    # Declared starts in background
    for d, (x, y) in DECLARED.items():
        color = 'red' if d in FAILED else 'lightgray'
        ax2.scatter(x, y, marker='o', s=150, facecolors='none',
                    edgecolors=color, linewidths=1.5, zorder=2)
        ax2.annotate(d, (x, y), xytext=(8, 8), textcoords='offset points',
                     fontsize=10, color=color,
                     fontweight='bold' if d in FAILED else 'normal')

    # Per-drone artists
    cmap = plt.get_cmap('tab10')
    colors = {d: cmap(i % 10) for i, d in enumerate(DRONES)}
    trails = {}
    dots = {}
    for d in DRONES:
        ls = '-' if d in FAILED else '-'
        lw = 2.2 if d in FAILED else 1.0
        alpha = 1.0 if d in FAILED else 0.4
        (line,) = ax2.plot([], [], color=colors[d], lw=lw, alpha=alpha, ls=ls)
        (dot,) = ax2.plot([], [], 'o', color=colors[d],
                          markersize=12 if d in FAILED else 7,
                          markeredgecolor='black', mew=0.8)
        trails[d] = line
        dots[d] = dot

    time_text = ax2.text(0.02, 0.97, '', transform=ax2.transAxes,
                         fontsize=14, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Common timeline at 30 fps
    fps = 30
    t_start, t_end = 30.0, 40.0
    frames = np.arange(t_start, t_end, 1.0 / fps)

    def init():
        for d in DRONES:
            trails[d].set_data([], [])
            dots[d].set_data([], [])
        time_text.set_text('')
        return list(trails.values()) + list(dots.values()) + [time_text]

    def update(t_now):
        for d in DRONES:
            v = data[d]['v']
            if v is None:
                continue
            t, p = v
            mask = t <= t_now
            if mask.any():
                trails[d].set_data(p[mask, 0], p[mask, 1])
                dots[d].set_data([p[mask, 0][-1]], [p[mask, 1][-1]])
            else:
                trails[d].set_data([], [])
                dots[d].set_data([], [])

        if t_now < TAKEOFF_T:
            phase = 'pre-takeoff'
        elif t_now < SWAP_CF8_T:
            phase = 'takeoff (Vicon already drifting on cf6/cf9)'
        else:
            phase = 'AFTER cf8 swap (Vicon labels chaotic)'
        time_text.set_text(f't = {t_now:5.2f} s  —  {phase}')
        return list(trails.values()) + list(dots.values()) + [time_text]

    anim = animation.FuncAnimation(
        fig2, update, frames=frames, init_func=init,
        interval=1000.0 / fps, blit=True, repeat=False,
    )

    out_mp4 = os.path.join(BAG, 'swap_animation.mp4')
    try:
        anim.save(out_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2000))
        print(f'saved {out_mp4}')
    except Exception as e:
        out_gif = os.path.join(BAG, 'swap_animation.gif')
        anim.save(out_gif, writer=animation.PillowWriter(fps=fps))
        print(f'ffmpeg unavailable ({e}); saved gif instead: {out_gif}')


if __name__ == '__main__':
    main()
