#!/usr/bin/env python3
"""
Generate 5x5 grid images from sweep screenshots with axis labels.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


RESULTS_DIR = Path(__file__).parent / 'results'
OUTPUT_DIR = Path(__file__).parent / 'results' / 'grids'
GRID_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]


# ============================================================
# Sweep definitions: (folder, param1_name, param2_name, title)
# ============================================================

SWEEPS = [
    ('kpos_offdiag', 'K₁₂', 'K₂₁',
     'Sweep 1: Cross-Species Position Coupling (K_rot = 0)'),
    ('krot_offdiag', 'R₁₂', 'R₂₁',
     'Sweep 3: Cross-Species Rotation Coupling (K_pos fixed attractive)'),
    ('kpos_diag', 'K₁₁', 'K₂₂',
     'Sweep 4: Self-Cohesion Asymmetry (K₁₂=K₂₁=0.3, K_rot=0)'),
    ('krot_diag', 'R₁₁', 'R₂₂',
     'Sweep 5: Self-Rotation (K_pos fixed attractive)'),
]

# Sweep 2 has 4 cases — one grid per case
SWEEP2_CASES = ['A', 'B', 'C', 'D']
SWEEP2_CASE_NAMES = {
    'A': 'No rotation',
    'B': 'Symmetric (R₁₂=R₂₁=+1)',
    'C': 'Antisymmetric (R₁₂=+1, R₂₁=-1)',
    'D': 'One-way (R₁₂=+1, R₂₁=0)',
}


def make_grid(sweep_dir, p1_name, p2_name, title, case=None, output_path=None):
    """Create a 5x5 grid image from screenshots."""
    screenshot_dir = RESULTS_DIR / sweep_dir / 'screenshots'

    fig, axes = plt.subplots(5, 5, figsize=(14, 14))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    for row, p2 in enumerate(reversed(GRID_VALUES)):  # y-axis top=high
        for col, p1 in enumerate(GRID_VALUES):
            ax = axes[row, col]

            # Build filename
            if case:
                fname = f"{p1:.1f}_{p2:.1f}_{case}.png"
            else:
                fname = f"{p1:.1f}_{p2:.1f}.png"

            img_path = screenshot_dir / fname
            if img_path.exists():
                img = mpimg.imread(str(img_path))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='red')

            ax.set_xticks([])
            ax.set_yticks([])

            # Column labels (top)
            if row == 0:
                ax.set_title(f'{p1:+.1f}', fontsize=11, pad=4)

            # Row labels (left)
            if col == 0:
                ax.set_ylabel(f'{p2:+.1f}', fontsize=11, rotation=0,
                              labelpad=30, va='center')

    # Axis labels
    fig.text(0.5, 0.01, p1_name, ha='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.5, p2_name, va='center', rotation='vertical',
             fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0.05, 0.03, 1.0, 0.95])

    if output_path is None:
        output_path = OUTPUT_DIR / f'{sweep_dir}.png'
    os.makedirs(output_path.parent, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sweeps 1, 3, 4, 5 — single 5x5 grid each
    for sweep_dir, p1, p2, title in SWEEPS:
        print(f"Generating: {sweep_dir}")
        make_grid(sweep_dir, p1, p2, title)

    # Sweep 2 — one grid per K_rot case
    for case in SWEEP2_CASES:
        case_title = (f'Sweep 2: K_pos × K_rot — Case {case}: '
                      f'{SWEEP2_CASE_NAMES[case]}')
        print(f"Generating: kpos_x_krot case {case}")
        output_path = OUTPUT_DIR / f'kpos_x_krot_{case}.png'
        make_grid('kpos_x_krot', 'K₁₂', 'K₂₁', case_title,
                  case=case, output_path=output_path)

    print(f"\nAll grids saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
