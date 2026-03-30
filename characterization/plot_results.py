#!/usr/bin/env python3
"""
Plot characterization sweep results as heatmaps.

Usage:
    python characterization/plot_results.py

Edit PLOT_CONFIG below to select which sweep results to plot.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ============================================================
# PLOT CONFIGURATION — Edit to customize
# ============================================================

PLOT_CONFIG = {
    # Which sweep to plot (subfolder name under results/)
    'sweep_dir': 'sweep_kpos_offdiag',

    # Which metrics to plot (one heatmap per metric)
    'metrics': ['R1', 'R2', 'Rdiff', 'K', 'd11', 'd22', 'd12',
                'revs', 'phi', 'L_norm', 'M_norm'],

    # Color maps per metric (customize as needed)
    'cmaps': {
        'R1': 'viridis', 'R2': 'viridis', 'Rdiff': 'RdBu_r',
        'K': 'hot', 'd11': 'viridis', 'd22': 'viridis', 'd12': 'plasma',
        'revs': 'RdBu_r', 'phi': 'YlOrRd', 'L_norm': 'RdBu_r',
        'M_norm': 'YlGnBu',
    },

    # Figure size per subplot
    'fig_width': 5,
    'fig_height': 4,

    # Output
    'output_dir': 'plots',
    'dpi': 150,
    'file_format': 'png',
}


# ============================================================
# PLOTTING ENGINE
# ============================================================

def load_results(sweep_dir):
    """Load CSV results and return DataFrame."""
    csv_path = sweep_dir / 'results.csv'
    if not csv_path.exists():
        print(f"No results found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def plot_heatmap(df, metric, ax, cmap='viridis', krot_case=None):
    """Plot a single metric as a 2D heatmap."""
    if krot_case:
        sub = df[df['krot_case'] == krot_case].copy()
    else:
        sub = df.copy()

    if len(sub) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        return

    p1_name = sub['param1_name'].iloc[0]
    p2_name = sub['param2_name'].iloc[0]

    p1_vals = sorted(sub['param1_val'].astype(float).unique())
    p2_vals = sorted(sub['param2_val'].astype(float).unique())

    # Build grid
    grid = np.full((len(p2_vals), len(p1_vals)), np.nan)
    for _, row in sub.iterrows():
        i = p2_vals.index(float(row['param2_val']))
        j = p1_vals.index(float(row['param1_val']))
        val = float(row[metric])
        grid[i, j] = val

    # Handle diverging colormaps (center at 0)
    if cmap in ('RdBu_r', 'RdBu', 'coolwarm', 'bwr'):
        vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        norm = None

    im = ax.imshow(grid, origin='lower', aspect='auto', cmap=cmap, norm=norm,
                   extent=[p1_vals[0], p1_vals[-1], p2_vals[0], p2_vals[-1]])

    ax.set_xlabel(p1_name, fontsize=11)
    ax.set_ylabel(p2_name, fontsize=11)

    title = metric
    if krot_case:
        title += f' (K_rot case {krot_case})'
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8)


def main():
    cfg = PLOT_CONFIG
    base = Path(__file__).parent
    # Find the results directory
    results_dir = base / 'results' / cfg['sweep_dir']
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print(f"Available: {list((base / 'results').glob('*')) if (base / 'results').exists() else 'None'}")
        sys.exit(1)

    df = load_results(results_dir)

    # Output directory
    out_dir = base / cfg['output_dir'] / cfg['sweep_dir']
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for K_rot cases
    has_cases = 'krot_case' in df.columns and df['krot_case'].nunique() > 1
    cases = sorted(df['krot_case'].unique()) if has_cases else [None]

    for metric in cfg['metrics']:
        if metric not in df.columns:
            print(f"  Skipping {metric} (not in data)")
            continue

        cmap = cfg['cmaps'].get(metric, 'viridis')

        if has_cases:
            # One subplot per K_rot case
            n_cases = len(cases)
            fig, axes = plt.subplots(1, n_cases, figsize=(cfg['fig_width'] * n_cases, cfg['fig_height']))
            if n_cases == 1:
                axes = [axes]
            for ax, case in zip(axes, cases):
                plot_heatmap(df, metric, ax, cmap, krot_case=case)
            fig.suptitle(f'{metric} — K_pos sweep × K_rot cases', fontsize=14, y=1.02)
        else:
            fig, ax = plt.subplots(figsize=(cfg['fig_width'], cfg['fig_height']))
            plot_heatmap(df, metric, ax, cmap)

        fig.tight_layout()
        out_path = out_dir / f'{metric}.{cfg["file_format"]}'
        fig.savefig(out_path, dpi=cfg['dpi'], bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")

    print(f"\nAll plots saved to {out_dir}")


if __name__ == '__main__':
    main()
