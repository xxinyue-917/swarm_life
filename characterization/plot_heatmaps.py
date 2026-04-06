#!/usr/bin/env python3
"""
Generate heatmaps from metrics sweep CSV files.

Usage:
    python characterization/plot_heatmaps.py

Edit PLOT_CONFIG below to select which sweep to plot.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ============================================================
# CONFIGURATION
# ============================================================

PLOT_CONFIG = {
    # Which sweep CSV to plot
    'sweep_type': 'kpos_offdiag',

    # Metrics to plot (one heatmap each)
    'metrics': [
        'max_d1', 'max_d2', 'centroid_dist',
        'avg_speed', 'avg_speed1', 'avg_speed2', 'KE',
        'MSD', 'MSD1', 'MSD2',
        'spacing_all', 'spacing_same', 'spacing1', 'spacing2',
    ],

    # Display names for metrics
    'metric_labels': {
        'max_d1': 'Max Distance (Group 1)',
        'max_d2': 'Max Distance (Group 2)',
        'centroid_dist': 'Centroid Distance',
        'avg_speed': 'Avg Speed (All)',
        'avg_speed1': 'Avg Speed (Group 1)',
        'avg_speed2': 'Avg Speed (Group 2)',
        'KE': 'Kinetic Energy',
        'MSD': 'MSD (All)',
        'MSD1': 'MSD (Group 1)',
        'MSD2': 'MSD (Group 2)',
        'spacing_all': 'Avg Spacing (All)',
        'spacing_same': 'Avg Spacing (Same Group)',
        'spacing1': 'Avg Spacing (Group 1)',
        'spacing2': 'Avg Spacing (Group 2)',
    },

    # Color maps
    'cmaps': {
        'max_d1': 'YlOrRd', 'max_d2': 'YlOrRd',
        'centroid_dist': 'plasma',
        'avg_speed': 'hot', 'avg_speed1': 'hot', 'avg_speed2': 'hot',
        'KE': 'inferno',
        'MSD': 'viridis', 'MSD1': 'viridis', 'MSD2': 'viridis',
        'spacing_all': 'YlGnBu', 'spacing_same': 'YlGnBu',
        'spacing1': 'YlGnBu', 'spacing2': 'YlGnBu',
    },

    'dpi': 150,
}


# ============================================================
# PLOTTING
# ============================================================

def plot_single_heatmap(df, metric, ax, cmap='viridis', label='', krot_case=None):
    """Plot one metric as a 2D heatmap."""
    sub = df[df['krot_case'] == krot_case].copy() if krot_case else df.copy()

    if len(sub) == 0 or metric not in sub.columns:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        return

    p1_name = sub['param1_name'].iloc[0]
    p2_name = sub['param2_name'].iloc[0]
    p1_vals = sorted(sub['param1_val'].astype(float).unique())
    p2_vals = sorted(sub['param2_val'].astype(float).unique())

    grid = np.full((len(p2_vals), len(p1_vals)), np.nan)
    for _, row in sub.iterrows():
        i = p2_vals.index(float(row['param2_val']))
        j = p1_vals.index(float(row['param1_val']))
        grid[i, j] = float(row[metric])

    im = ax.imshow(grid, origin='lower', aspect='auto', cmap=cmap,
                   extent=[p1_vals[0], p1_vals[-1], p2_vals[0], p2_vals[-1]])

    ax.set_xlabel(p1_name, fontsize=11)
    ax.set_ylabel(p2_name, fontsize=11)
    title = label or metric
    if krot_case:
        title += f' (Case {krot_case})'
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)


def main():
    cfg = PLOT_CONFIG
    base = Path(__file__).parent / 'results' / 'metrics'
    csv_path = base / f"{cfg['sweep_type']}.csv"

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        print("Run sweep_metrics.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Output directory
    out_dir = base / cfg['sweep_type']
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for K_rot cases
    has_cases = 'krot_case' in df.columns and df['krot_case'].nunique() > 1
    cases = sorted(df['krot_case'].unique()) if has_cases else [None]

    for metric in cfg['metrics']:
        if metric not in df.columns:
            print(f"  Skipping {metric} (not in data)")
            continue

        label = cfg['metric_labels'].get(metric, metric)
        cmap = cfg['cmaps'].get(metric, 'viridis')

        if has_cases:
            n = len(cases)
            fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
            if n == 1:
                axes = [axes]
            for ax, case in zip(axes, cases):
                plot_single_heatmap(df, metric, ax, cmap, label, krot_case=case)
            fig.suptitle(label, fontsize=14, y=1.02)
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            plot_single_heatmap(df, metric, ax, cmap, label)

        fig.tight_layout()
        out_path = out_dir / f'{metric}.png'
        fig.savefig(str(out_path), dpi=cfg['dpi'], bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")

    # Also generate a combined overview (all metrics in one figure)
    n_metrics = len([m for m in cfg['metrics'] if m in df.columns])
    if n_metrics > 0 and not has_cases:
        cols = 4
        rows = (n_metrics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        axes = axes.flatten()

        idx = 0
        for metric in cfg['metrics']:
            if metric not in df.columns:
                continue
            label = cfg['metric_labels'].get(metric, metric)
            cmap = cfg['cmaps'].get(metric, 'viridis')
            plot_single_heatmap(df, metric, axes[idx], cmap, label)
            idx += 1

        # Hide unused axes
        for i in range(idx, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f"All Metrics — {cfg['sweep_type']}", fontsize=16, y=1.01)
        fig.tight_layout()
        overview_path = out_dir / 'overview.png'
        fig.savefig(str(overview_path), dpi=cfg['dpi'], bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {overview_path}")

    print(f"\nAll heatmaps saved to {out_dir}")


if __name__ == '__main__':
    main()
