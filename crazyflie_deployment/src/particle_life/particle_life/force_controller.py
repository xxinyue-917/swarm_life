#!/usr/bin/env python3
"""
Force controller — 2D port of swarm_life/src/particle_life.py 4-zone kernel.

Computes per-drone velocity from current poses, K_pos, K_rot at fixed altitude.
Pure numpy (no JIT, no pygame). Identical physics to the simulation reference.

Inputs (per tick):
    positions : (N, 2) float — XY in arena meters
    species   : (N,)   int   — species index per drone
    k_pos     : (S, S) float — radial attraction/repulsion matrix
    k_rot     : (S, S) float — tangential (rotation) coupling matrix
    params    : dict — r_max, beta, force_scale, far_attraction, a_rot

Returns:
    velocities : (N, 2) float — XY velocity (sim units; caller scales to m/s)
"""

import numpy as np


def compute_velocities(positions, species, k_pos, k_rot, params):
    n = positions.shape[0]
    r_max = params['r_max']
    beta = params['beta']
    force_scale = params['force_scale']
    far_attraction = params['far_attraction']
    a_rot = params['a_rot']
    inv_1_minus_beta = 1.0 / (1.0 - beta) if beta < 1.0 else 1.0
    peak_r = 0.5 * (1.0 + beta)

    velocities = np.zeros((n, 2))

    for i in range(n):
        delta = positions - positions[i]
        dist = np.linalg.norm(delta, axis=1)
        dist[i] = np.inf

        for j in range(n):
            if j == i:
                continue
            r = dist[j]
            r_norm = r / r_max
            dx, dy = delta[j]
            inv_r = 1.0 / (r + 1e-8)
            r_hat_x, r_hat_y = dx * inv_r, dy * inv_r
            t_hat_x, t_hat_y = -r_hat_y, r_hat_x

            si = species[i]
            sj = species[j]
            kp = k_pos[si, sj]
            kr = k_rot[si, sj]

            if r_norm >= 1.0:
                if far_attraction > 0.0:
                    F = kp * far_attraction
                    velocities[i, 0] += force_scale * F * r_hat_x
                    velocities[i, 1] += force_scale * F * r_hat_y
                continue

            if r_norm < beta:
                F = r_norm / beta - 1.0  # universal repulsion
            else:
                triangle = 1.0 - abs(2.0 * r_norm - 1.0 - beta) * inv_1_minus_beta
                if r_norm < peak_r:
                    F = kp * triangle
                else:
                    F = kp * max(far_attraction, triangle)

            velocities[i, 0] += force_scale * F * r_hat_x
            velocities[i, 1] += force_scale * F * r_hat_y

            # Tangential swirl (decays linearly to zero at r_max).
            swirl = kr * a_rot * max(0.0, 1.0 - r_norm)
            velocities[i, 0] += swirl * t_hat_x
            velocities[i, 1] += swirl * t_hat_y

    return velocities


def clamp_speed(velocities, v_max):
    """Clamp per-row magnitude to v_max (in m/s after caller scales)."""
    speed = np.linalg.norm(velocities, axis=1, keepdims=True)
    over = (speed > v_max).flatten()
    if np.any(over):
        velocities[over] = velocities[over] * (v_max / speed[over])
    return velocities


def geofence_clip(positions, arena_width, arena_height, margin, origin_xy=(0.0, 0.0)):
    """Clip XY positions to arena rectangle centered at origin_xy."""
    ox, oy = origin_xy
    half_w = arena_width / 2.0 - margin
    half_h = arena_height / 2.0 - margin
    positions[:, 0] = np.clip(positions[:, 0], ox - half_w, ox + half_w)
    positions[:, 1] = np.clip(positions[:, 1], oy - half_h, oy + half_h)
    return positions
