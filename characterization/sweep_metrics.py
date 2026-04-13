#!/usr/bin/env python3
"""
2-Species Metrics Sweep — Headless quantitative characterization

Sweep K_pos / K_rot parameters, compute metrics over the last 20% of
simulation timesteps, and output CSV files. Much faster than video sweep
(no rendering).

Usage:
    python characterization/sweep_metrics.py

Edit CONFIG below to change sweep type, grid resolution, physics, etc.
"""

import os
import sys

# Headless pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import csv
import json
import math
import time
from pathlib import Path
from itertools import product

import numpy as np
from tqdm import tqdm

from particle_life import Config, ParticleLife


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # ----------------------------------------------------------
    # SWEEP TYPE
    # ----------------------------------------------------------
    'sweep_type': 'kpos_offdiag',

    # ----------------------------------------------------------
    # GRID
    # ----------------------------------------------------------
    'grid_points': 21,        # 21 → [-1.0, -0.9, ..., 0.9, 1.0] (interval 0.1)
    'param_min': -1.0,
    'param_max':  1.0,

    # ----------------------------------------------------------
    # FIXED MATRIX VALUES
    # ----------------------------------------------------------
    'K11': 0.6,    'K22': 0.6,    'K12': 0.0,    'K21': 0.0,
    'R11': 0.0,    'R22': 0.0,    'R12': 0.0,    'R21': 0.0,

    # ----------------------------------------------------------
    # K_ROT CASES (for kpos_x_krot)
    # ----------------------------------------------------------
    'krot_cases': {
        'A': {'R11': 0, 'R12': 0,  'R21': 0,  'R22': 0},
        'B': {'R11': 0, 'R12': 1,  'R21': 1,  'R22': 0},
        'C': {'R11': 0, 'R12': 1,  'R21': -1, 'R22': 0},
        'D': {'R11': 0, 'R12': 1,  'R21': 0,  'R22': 0},
    },

    # ----------------------------------------------------------
    # SIMULATION
    # ----------------------------------------------------------
    'n_particles': 50,
    'sim_width': 10.0,
    'sim_height': 10.0,
    'dt': 0.05,
    'max_speed': 1.0,
    'r_max': 2.0,
    'beta': 0.2,
    'force_scale': 0.5,
    'a_rot': 1.0,
    'far_attraction': 0.1,

    # ----------------------------------------------------------
    # MEASUREMENT
    # ----------------------------------------------------------
    'total_steps': 1000,      # Total simulation steps
    'measure_fraction': 0.2,  # Measure over last 20% of steps
    'sample_stride': 5,       # Sample every N steps during measurement
    'n_seeds': 5,             # Random seeds for averaging

    # ----------------------------------------------------------
    # OUTPUT
    # ----------------------------------------------------------
    'output_dir': 'results/metrics',
}


# ============================================================
# METRICS COMPUTATION
# ============================================================

def max_pairwise_distance(X):
    """Max distance between any two points in X."""
    if len(X) < 2:
        return 0.0
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.sqrt((diff * diff).sum(axis=2))
    return float(dists.max())


def mean_pairwise_distance(X):
    """Mean distance between all pairs in X."""
    if len(X) < 2:
        return 0.0
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.sqrt((diff * diff).sum(axis=2))
    n = len(X)
    iu = np.triu_indices(n, k=1)
    return float(dists[iu].mean())


def mean_pairwise_distance_cross(Xa, Xb):
    """Mean distance between all pairs across two groups."""
    if len(Xa) == 0 or len(Xb) == 0:
        return 0.0
    diff = Xa[:, np.newaxis, :] - Xb[np.newaxis, :, :]
    dists = np.sqrt((diff * diff).sum(axis=2))
    return float(dists.mean())


def compute_metrics_snapshot(X, V, mask1, mask2, X_init, mask1_init, mask2_init):
    """Compute all 14 metrics for one simulation snapshot."""
    X1, X2 = X[mask1], X[mask2]
    V1, V2 = V[mask1], V[mask2]

    # Speeds
    speeds = np.linalg.norm(V, axis=1)
    speeds1 = np.linalg.norm(V1, axis=1)
    speeds2 = np.linalg.norm(V2, axis=1)

    # Centroids
    c1 = X1.mean(axis=0) if len(X1) > 0 else np.array([0.0, 0.0])
    c2 = X2.mean(axis=0) if len(X2) > 0 else np.array([0.0, 0.0])

    # MSD from initial positions
    disp_all = X - X_init
    msd_all = float((disp_all ** 2).sum(axis=1).mean())
    disp1 = X[mask1] - X_init[mask1_init]
    msd1 = float((disp1 ** 2).sum(axis=1).mean()) if len(disp1) > 0 else 0.0
    disp2 = X[mask2] - X_init[mask2_init]
    msd2 = float((disp2 ** 2).sum(axis=1).mean()) if len(disp2) > 0 else 0.0

    return {
        'max_d1': max_pairwise_distance(X1),
        'max_d2': max_pairwise_distance(X2),
        'centroid_dist': float(np.linalg.norm(c1 - c2)),
        'avg_speed': float(speeds.mean()),
        'avg_speed1': float(speeds1.mean()) if len(speeds1) > 0 else 0.0,
        'avg_speed2': float(speeds2.mean()) if len(speeds2) > 0 else 0.0,
        'KE': float((speeds ** 2).mean()),
        'MSD': msd_all,
        'MSD1': msd1,
        'MSD2': msd2,
        'spacing_all': mean_pairwise_distance(X),
        'spacing_same': (mean_pairwise_distance(X1) + mean_pairwise_distance(X2)) / 2,
        'spacing1': mean_pairwise_distance(X1),
        'spacing2': mean_pairwise_distance(X2),
    }


METRIC_KEYS = ['max_d1', 'max_d2', 'centroid_dist',
               'avg_speed', 'avg_speed1', 'avg_speed2', 'KE',
               'MSD', 'MSD1', 'MSD2',
               'spacing_all', 'spacing_same', 'spacing1', 'spacing2']


# ============================================================
# SIMULATION RUNNER
# ============================================================

def build_config(K_pos, K_rot, cfg):
    return Config(
        n_species=2,
        n_particles=cfg['n_particles'],
        sim_width=cfg['sim_width'],
        sim_height=cfg['sim_height'],
        dt=cfg['dt'],
        max_speed=cfg['max_speed'],
        r_max=cfg['r_max'],
        beta=cfg['beta'],
        force_scale=cfg['force_scale'],
        a_rot=cfg['a_rot'],
        far_attraction=cfg['far_attraction'],
        position_matrix=K_pos.tolist(),
        orientation_matrix=K_rot.tolist(),
    )


def run_single(config, cfg, seed):
    """Run one simulation and return averaged metrics over last 20%."""
    config.seed = int(seed)
    sim = ParticleLife(config, headless=True)

    # Scatter randomly
    m = 0.5
    sim.positions[:, 0] = np.random.uniform(m, cfg['sim_width'] - m, sim.n)
    sim.positions[:, 1] = np.random.uniform(m, cfg['sim_height'] - m, sim.n)
    sim.velocities[:] = 0.0

    species = sim.species.copy()
    mask1 = (species == 0)
    mask2 = (species == 1)

    total = cfg['total_steps']
    measure_start = int(total * (1.0 - cfg['measure_fraction']))
    stride = cfg['sample_stride']

    # Run to measurement start
    for _ in range(measure_start):
        sim.step()

    # Record initial positions for MSD (at start of measurement window)
    X_init = sim.positions.copy()

    # Measure over last 20%
    accum = {k: 0.0 for k in METRIC_KEYS}
    n_samples = 0

    for t in range(total - measure_start):
        sim.step()
        if t % stride != 0:
            continue

        m = compute_metrics_snapshot(
            sim.positions, sim.velocities,
            mask1, mask2, X_init, mask1, mask2)

        for k in METRIC_KEYS:
            accum[k] += m[k]
        n_samples += 1

    # Average
    result = {}
    for k in METRIC_KEYS:
        result[k] = accum[k] / max(n_samples, 1)
    return result


def run_averaged(config, cfg):
    """Run multiple seeds and average metrics."""
    all_results = []
    for s in range(cfg['n_seeds']):
        seed = 42 + s
        result = run_single(config, cfg, seed)
        all_results.append(result)

    avg = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in all_results if not np.isnan(r[k])]
        avg[k] = np.mean(vals) if vals else float('nan')
    return avg


# ============================================================
# SWEEP POINT GENERATION (same as video sweep)
# ============================================================

def build_matrices(K11, K12, K21, K22, R11, R12, R21, R22):
    K_pos = np.array([[K11, K12], [K21, K22]], dtype=float)
    K_rot = np.array([[R11, R12], [R21, R22]], dtype=float)
    return K_pos, K_rot


def generate_sweep_points(cfg):
    grid = np.linspace(cfg['param_min'], cfg['param_max'], cfg['grid_points'])
    sweep_type = cfg['sweep_type']
    points = []

    if sweep_type == 'kpos_offdiag':
        for k12, k21 in product(grid, grid):
            K_pos, K_rot = build_matrices(
                cfg['K11'], k12, k21, cfg['K22'],
                cfg['R11'], cfg['R12'], cfg['R21'], cfg['R22'])
            points.append(('K12', k12, 'K21', k21, '', K_pos, K_rot))

    elif sweep_type == 'kpos_diag':
        for k11, k22 in product(grid, grid):
            K_pos, K_rot = build_matrices(
                k11, cfg['K12'], cfg['K21'], k22,
                cfg['R11'], cfg['R12'], cfg['R21'], cfg['R22'])
            points.append(('K11', k11, 'K22', k22, '', K_pos, K_rot))

    elif sweep_type == 'krot_offdiag':
        for r12, r21 in product(grid, grid):
            K_pos, K_rot = build_matrices(
                cfg['K11'], cfg['K12'], cfg['K21'], cfg['K22'],
                cfg['R11'], r12, r21, cfg['R22'])
            points.append(('R12', r12, 'R21', r21, '', K_pos, K_rot))

    elif sweep_type == 'krot_diag':
        for r11, r22 in product(grid, grid):
            K_pos, K_rot = build_matrices(
                cfg['K11'], cfg['K12'], cfg['K21'], cfg['K22'],
                r11, cfg['R12'], cfg['R21'], r22)
            points.append(('R11', r11, 'R22', r22, '', K_pos, K_rot))

    elif sweep_type == 'kpos_x_krot':
        for case_name, case_vals in cfg['krot_cases'].items():
            for k12, k21 in product(grid, grid):
                K_pos, K_rot = build_matrices(
                    cfg['K11'], k12, k21, cfg['K22'],
                    case_vals['R11'], case_vals['R12'],
                    case_vals['R21'], case_vals['R22'])
                points.append(('K12', k12, 'K21', k21, case_name, K_pos, K_rot))

    elif sweep_type == 'krot_full':
        for r11, r12, r21, r22 in product(grid, grid, grid, grid):
            K_pos, K_rot = build_matrices(
                cfg['K11'], cfg['K12'], cfg['K21'], cfg['K22'],
                r11, r12, r21, r22)
            name = f"{r11:.1f}_{r12:.1f}_{r21:.1f}_{r22:.1f}"
            points.append(('R12', r12, 'R21', r21, name, K_pos, K_rot))

    return points


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str, default=None,
                        help='Override sweep_type from CONFIG')
    args = parser.parse_args()

    cfg = CONFIG.copy()
    if args.sweep:
        cfg['sweep_type'] = args.sweep
    sweep_type = cfg['sweep_type']

    # Set appropriate fixed values per sweep type
    if sweep_type in ('kpos_offdiag', 'kpos_x_krot'):
        cfg['K12'] = 0.0;  cfg['K21'] = 0.0   # these are swept
    elif sweep_type in ('krot_offdiag', 'krot_diag', 'krot_full'):
        cfg['K12'] = 0.3;  cfg['K21'] = 0.3   # fixed attractive
    elif sweep_type == 'kpos_diag':
        cfg['K12'] = 0.3;  cfg['K21'] = 0.3   # fixed cross-attraction

    base_dir = Path(__file__).parent / cfg['output_dir']
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(base_dir / f'{sweep_type}_config.json', 'w') as f:
        serializable = {k: v for k, v in cfg.items()
                        if not isinstance(v, np.ndarray)}
        json.dump(serializable, f, indent=2, default=str)

    points = generate_sweep_points(cfg)

    print("=" * 60)
    print(f"Metrics Sweep: {sweep_type}")
    print("=" * 60)
    print(f"Grid:            {cfg['grid_points']}×{cfg['grid_points']}")
    print(f"Total points:    {len(points)}")
    print(f"Seeds/point:     {cfg['n_seeds']}")
    print(f"Steps/sim:       {cfg['total_steps']}")
    print(f"Measure window:  last {cfg['measure_fraction']*100:.0f}%")
    print(f"Output:          {base_dir}")
    print("=" * 60)

    # CSV — one row per seed (not averaged)
    csv_path = base_dir / f'{sweep_type}.csv'
    fieldnames = ['param1_name', 'param1_val', 'param2_name', 'param2_val',
                  'krot_case', 'seed'] + METRIC_KEYS

    # Resume support
    existing = set()
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['param1_val'], row['param2_val'],
                       row.get('krot_case', ''), row.get('seed', ''))
                existing.add(key)
        print(f"Resuming: {len(existing)} rows done")

    write_header = not csv_path.exists() or len(existing) == 0
    csv_file = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    t_start = time.time()
    pbar = tqdm(points, desc=f"Metrics {sweep_type}")

    for p1n, p1v, p2n, p2v, kcase, K_pos, K_rot in pbar:
        pbar.set_postfix_str(f"{p1n}={p1v:.2f} {p2n}={p2v:.2f} {kcase}")

        config = build_config(K_pos, K_rot, cfg)

        for s in range(cfg['n_seeds']):
            seed = 42 + s
            key = (f"{p1v:.4f}", f"{p2v:.4f}", kcase, str(seed))
            if key in existing:
                continue

            result = run_single(config, cfg, seed)

            row = {
                'param1_name': p1n,
                'param1_val': f"{p1v:.4f}",
                'param2_name': p2n,
                'param2_val': f"{p2v:.4f}",
                'krot_case': kcase,
                'seed': seed,
        }
        for k in METRIC_KEYS:
            row[k] = f"{result[k]:.6f}"
        writer.writerow(row)
        csv_file.flush()

    csv_file.close()
    elapsed = time.time() - t_start
    print(f"\nDone! {len(points)} points in {elapsed:.0f}s")
    print(f"Results: {csv_path}")


if __name__ == '__main__':
    main()
