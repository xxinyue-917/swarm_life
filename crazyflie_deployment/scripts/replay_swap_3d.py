#!/usr/bin/env python3
"""
3D version of the Vicon swap replay for hover_all_20260518_151510.

Outputs:
  swap_trajectory_3d.png  — 3D Vicon trajectories (all 9 drones), with Kalman
                            shown for cf6/cf8/cf9 as dashed lines.
  swap_animation_3d.mp4   — 3D animation t=30→40s.

The z axis is the smoking gun: cf6/cf9 stay below 0.16 m the whole time while
their xy "moves" ~1 m — a grounded drone can't slide that far, proving the
xy motion was Vicon label migration, not real drone motion.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BAG = '/home/xxinyue/swarm_life/crazyflie_deployment/results/flights/hover_all_20260518_151510'

DECLARED = {
    'cf1': (1.0, 1.0, 0.0), 'cf2': (1.0, 0.0, 0.0), 'cf3': (1.0, -1.0, 0.0),
    'cf4': (0.0, 1.0, 0.0), 'cf5': (0.0, 0.0, 0.0), 'cf6': (-1.0, 0.0, 0.0),
    'cf7': (-1.0, 1.0, 0.0), 'cf8': (-1.0, -1.0, 0.0), 'cf9': (0.0, -1.0, 0.0),
}
DRONES = list(DECLARED.keys())
FAILED = {'cf6', 'cf8', 'cf9'}

TAKEOFF_T = 33.5
SWAP_CF8_T = 35.02


def load(d, kind):
    p = os.path.join(BAG, f'{d}_{kind}.csv')
    if not os.path.exists(p):
        return None
    a = np.loadtxt(p, delimiter=',', skiprows=1)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a[:, 0], a[:, 1:4]


def _set_axes(ax):
    ax.set_xlim(-2.5, 2.0)
    ax.set_ylim(-2.5, 2.0)
    ax.set_zlim(0.0, 1.5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    # Floor grid (z=0 plane)
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.0, 8), np.linspace(-2.5, 2.0, 8))
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.05, color='gray')


def main():
    data = {d: {'v': load(d, 'vicon'), 'k': load(d, 'kalman')} for d in DRONES}

    # =================== STATIC 3D PLOT ===================
    fig = plt.figure(figsize=(18, 9))

    cmap = plt.get_cmap('tab10')
    colors = {d: cmap(i % 10) for i, d in enumerate(DRONES)}

    for sp, (title, focus_failed_only) in enumerate(
        [('all 9 drones', False), ('cf6 / cf8 / cf9 only (with Kalman dashed)', True)], 1
    ):
        ax = fig.add_subplot(1, 2, sp, projection='3d')
        ax.set_title(f'Vicon 3D trajectory — {title}', fontsize=11)
        _set_axes(ax)

        # Declared starts
        for d, (x, y, z) in DECLARED.items():
            edge = 'red' if d in FAILED else 'gray'
            ax.scatter(x, y, z, marker='o', s=80, facecolors='none',
                       edgecolors=edge, linewidths=1.5)
            ax.text(x, y, z + 0.05, d, color=edge, fontsize=9,
                    fontweight='bold' if d in FAILED else 'normal')

        for d in DRONES:
            if focus_failed_only and d not in FAILED:
                continue
            v = data[d]['v']
            if v is None:
                continue
            t, p = v
            ax.plot(p[:, 0], p[:, 1], p[:, 2],
                    color=colors[d], lw=2.0 if d in FAILED else 1.0,
                    alpha=1.0 if d in FAILED else 0.5, label=d)
            # End marker
            ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2], marker='s',
                       s=60, color=colors[d], edgecolors='black')

            if focus_failed_only and d in FAILED:
                k = data[d]['k']
                if k is not None:
                    tk, pk = k
                    ax.plot(pk[:, 0], pk[:, 1], pk[:, 2], ls='--',
                            color=colors[d], lw=1.2, alpha=0.6)

        ax.legend(fontsize=8, loc='upper left')
        ax.view_init(elev=25, azim=-60)

    plt.suptitle('hover_all_20260518_151510 — Vicon swap, 3D replay\n'
                 '(failed drones stay near z=0.1 m while their xy "moves" ~1 m → grounded label migration)',
                 fontsize=12, y=0.99)
    plt.tight_layout()
    out = os.path.join(BAG, 'swap_trajectory_3d.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f'saved {out}')

    # =================== 3D ANIMATION ===================
    fig2 = plt.figure(figsize=(11, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    _set_axes(ax2)

    # Static elements: declared start positions (as guide)
    for d, (x, y, z) in DECLARED.items():
        edge = 'red' if d in FAILED else 'lightgray'
        ax2.scatter(x, y, z, marker='o', s=120, facecolors='none',
                    edgecolors=edge, linewidths=1.5)
        ax2.text(x, y, z + 0.04, d, color=edge, fontsize=9,
                 fontweight='bold' if d in FAILED else 'normal')

    trails = {}
    dots = {}
    for d in DRONES:
        lw = 2.5 if d in FAILED else 1.0
        alpha = 1.0 if d in FAILED else 0.35
        (line,) = ax2.plot([], [], [], color=colors[d], lw=lw, alpha=alpha)
        (dot,) = ax2.plot([], [], [], 'o', color=colors[d],
                          markersize=14 if d in FAILED else 7,
                          markeredgecolor='black', mew=0.8)
        trails[d] = line
        dots[d] = dot

    time_text = ax2.text2D(0.02, 0.97, '', transform=ax2.transAxes,
                           fontsize=13, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    fps = 30
    t_start, t_end = 30.0, 40.0
    frames = np.arange(t_start, t_end, 1.0 / fps)

    def update(t_now):
        for d in DRONES:
            v = data[d]['v']
            if v is None:
                continue
            t, p = v
            mask = t <= t_now
            if mask.any():
                trails[d].set_data(p[mask, 0], p[mask, 1])
                trails[d].set_3d_properties(p[mask, 2])
                dots[d].set_data([p[mask, 0][-1]], [p[mask, 1][-1]])
                dots[d].set_3d_properties([p[mask, 2][-1]])
        if t_now < TAKEOFF_T:
            phase = 'pre-takeoff (all on ground)'
        elif t_now < SWAP_CF8_T:
            phase = 'takeoff: cf6/cf9 xy "drift" while z stays ~0.1m (impossible for real motion)'
        else:
            phase = 'AFTER cf8 swap: Vicon labels chaotic'
        time_text.set_text(f't = {t_now:5.2f} s  —  {phase}')
        # Slow camera rotation around z-axis
        ax2.view_init(elev=25, azim=-60 + (t_now - t_start) * 3.0)
        return list(trails.values()) + list(dots.values()) + [time_text]

    anim = animation.FuncAnimation(
        fig2, update, frames=frames,
        interval=1000.0 / fps, blit=False, repeat=False,
    )

    out_mp4 = os.path.join(BAG, 'swap_animation_3d.mp4')
    try:
        anim.save(out_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2500))
        print(f'saved {out_mp4}')
    except Exception as e:
        out_gif = os.path.join(BAG, 'swap_animation_3d.gif')
        anim.save(out_gif, writer=animation.PillowWriter(fps=fps))
        print(f'ffmpeg unavailable ({e}); saved gif instead: {out_gif}')


if __name__ == '__main__':
    main()
