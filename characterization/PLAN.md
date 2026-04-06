# Characterization Plan — 2-Species Parameter Sweep

## Goal

Systematically characterize emergent behaviors in a 2-species particle life system by sweeping the 8-dimensional parameter space (4 K_pos + 4 K_rot entries) and recording video + screenshot outputs for visual analysis.

## Approach

Start with 2 groups (species) and limited parameters:
- **Position Matrix (K_pos)**: 2x2, entries K₁₁, K₁₂, K₂₁, K₂₂ each in [-1, 1]
- **Revolution Matrix (K_rot)**: 2x2, entries R₁₁, R₁₂, R₂₁, R₂₂ each in [-1, 1]
- Multiple matrix configurations to sweep (diagonal-only, off-diagonal-only, full)
- All simulations start from random particle positions

## Parameter Space

```
K_pos = [[K₁₁, K₁₂],    K_rot = [[R₁₁, R₁₂],
         [K₂₁, K₂₂]]             [R₂₁, R₂₂]]
```

Full space = 8 dimensions. We reduce dimensionality by fixing some entries.

## Sweep Configurations

### Sweep 1: `kpos_offdiag` — Cross-species position coupling
- **Swept**: K₁₂ ∈ [-1, 1], K₂₁ ∈ [-1, 1], grid = 5 points
- **Fixed**: K₁₁ = K₂₂ = 0.6, K₁₂ = K₂₁ = 0.0 (defaults), K_rot = all zeros
- **Purpose**: Baseline — how cross-species position coupling creates chase, encapsulate, separate
- **Result**: 25 videos

### Sweep 2: `kpos_x_krot` — Position × rotation cases
- **Swept**: K₁₂ ∈ [-1, 1], K₂₁ ∈ [-1, 1], grid = 5, repeated for 4 K_rot cases
- **Fixed**: K₁₁ = K₂₂ = 0.6
- **K_rot cases**:
  - A: R = [[0, 0], [0, 0]] (no rotation)
  - B: R = [[0, +1], [+1, 0]] (symmetric → collective rotation)
  - C: R = [[0, +1], [-1, 0]] (antisymmetric → translation)
  - D: R = [[0, +1], [0, 0]] (one-way)
- **Purpose**: How rotation coupling modifies positional behaviors
- **Result**: 100 videos

### Sweep 3: `krot_offdiag` — Cross-species rotation coupling
- **Swept**: R₁₂ ∈ [-1, 1], R₂₁ ∈ [-1, 1], grid = 5
- **Fixed**: K_pos = [[0.6, 0.3], [0.3, 0.6]], R₁₁ = R₂₂ = 0
- **Purpose**: Rotation behavior space with fixed attractive K_pos
- **Result**: 25 videos

### Sweep 4: `kpos_diag` — Self-cohesion asymmetry
- **Swept**: K₁₁ ∈ [-1, 1], K₂₂ ∈ [-1, 1], grid = 5
- **Fixed**: K₁₂ = K₂₁ = 0.3, K_rot = all zeros
- **Purpose**: Effect of asymmetric self-cohesion
- **Result**: 25 videos

### Sweep 5: `krot_diag` — Self-rotation
- **Swept**: R₁₁ ∈ [-1, 1], R₂₂ ∈ [-1, 1], grid = 5
- **Fixed**: K_pos = [[0.6, 0.3], [0.3, 0.6]], R₁₂ = R₂₁ = 0
- **Purpose**: Effect of self-rotation coupling
- **Result**: 25 videos

### Sweep 6: `krot_full` — All 4 K_rot entries
- **Swept**: R₁₁, R₁₂, R₂₁, R₂₂ ∈ [-1, 1], grid = 5 (5⁴ = 625 combinations)
- **Fixed**: K_pos = [[0.6, 0.3], [0.3, 0.6]]
- **Purpose**: Complete rotation characterization
- **Result**: 625 videos

## Simulation Parameters (as run)

| Parameter | Value | Notes |
|-----------|-------|-------|
| n_particles | 50 | Per species (100 total) |
| sim_width | 10.0 | Meters |
| sim_height | 10.0 | Meters |
| dt | 0.05 | Timestep |
| max_speed | 1.0 | Velocity clamp |
| r_max | 2.0 | Interaction radius |
| beta | 0.2 | Repulsion zone boundary |
| force_scale | 0.5 | Force multiplier |
| a_rot | 1.0 | Rotation coupling strength |
| far_attraction | 0.1 | Long-range attraction beyond r_max |
| video_duration | 10 | Seconds per video |
| fps | 30 | Frames per second |
| grid_points | 5 | Points per axis: [-1.0, -0.5, 0.0, 0.5, 1.0] |
| init | Random | Particles scattered uniformly across workspace |

## Output Structure

```
characterization/
├── PLAN.md                 # This document
├── sweep_2species.py       # Main sweep script (video + screenshot)
├── plot_results.py         # Metrics visualization (for later)
├── results/
│   ├── kpos_offdiag/       # Sweep 1
│   │   ├── config.json
│   │   ├── videos/         # -1.0_-1.0.mp4, -1.0_-0.5.mp4, ...
│   │   └── screenshots/    # -1.0_-1.0.png, -1.0_-0.5.png, ...
│   ├── kpos_x_krot/        # Sweep 2 (files: 0.5_-0.5_B.mp4, etc.)
│   ├── krot_offdiag/       # Sweep 3
│   ├── kpos_diag/          # Sweep 4
│   ├── krot_diag/          # Sweep 5
│   └── krot_full/          # Sweep 6 (files: 0.0_0.5_-0.5_1.0_name.mp4)
```

File naming: `{param1}_{param2}.mp4` (or `{param1}_{param2}_{case}.mp4` for multi-case sweeps). Videos show simulation with matrix values overlaid. Screenshots capture final state.

## Quantitative Metrics

All metrics measured across the **last 20%** of simulation timesteps and averaged.

| # | Metric | Symbol | Description |
|---|--------|--------|-------------|
| 1 | Max distance (group 1) | max_d1 | Max pairwise distance within species 1 — shows if group is clustered or spread |
| 2 | Max distance (group 2) | max_d2 | Max pairwise distance within species 2 |
| 3 | Centroid distance | centroid_dist | Distance between centroids of the two groups — concentric vs separated |
| 4 | Avg speed (all) | avg_speed | Mean speed across all agents |
| 5 | Avg speed (group 1) | avg_speed1 | Mean speed of species 1 |
| 6 | Avg speed (group 2) | avg_speed2 | Mean speed of species 2 |
| 7 | Kinetic energy | KE | Mean v² across all agents |
| 8 | MSD (all) | MSD | Mean square displacement from initial positions |
| 9 | MSD (group 1) | MSD1 | MSD for species 1 |
| 10 | MSD (group 2) | MSD2 | MSD for species 2 |
| 11 | Avg spacing (all) | spacing_all | Mean pairwise distance across all agents |
| 12 | Avg spacing (same group) | spacing_same | Mean pairwise distance within same species (avg of group 1 and 2) |
| 13 | Avg spacing (group 1) | spacing1 | Mean pairwise distance within species 1 |
| 14 | Avg spacing (group 2) | spacing2 | Mean pairwise distance within species 2 |

**Future**: Revolution/rotation order parameters (pending).

Each metric produces one heatmap per 2D sweep → 14 heatmaps per sweep configuration.

## Output Structure

```
characterization/
├── PLAN.md                     # This document
├── sweep_2species.py           # Video + screenshot sweep
├── sweep_metrics.py            # Headless metrics sweep (CSV output)
├── plot_heatmaps.py            # Generate heatmaps from CSV
├── make_grids.py               # Screenshot grid images
├── make_slides.py              # PowerPoint generation
├── results/
│   ├── kpos_offdiag/           # Sweep 1 videos/screenshots
│   ├── kpos_x_krot/            # Sweep 2
│   ├── krot_offdiag/           # Sweep 3
│   ├── kpos_diag/              # Sweep 4
│   ├── krot_diag/              # Sweep 5
│   ├── krot_full/              # Sweep 6
│   ├── grids/                  # 5×5 screenshot grids
│   └── metrics/                # CSV files + heatmap PNGs
│       ├── kpos_offdiag.csv
│       ├── kpos_offdiag/       # Heatmap images
│       │   ├── max_d1.png
│       │   ├── centroid_dist.png
│       │   └── ...
│       └── ...
```

## Status

- [x] Plan created
- [x] Sweep 1: K_pos off-diagonal (25 videos, 255s)
- [x] Sweep 2: K_pos × K_rot cases (100 videos, 1022s)
- [x] Sweep 3: K_rot off-diagonal (25 videos, 255s)
- [x] Sweep 4: K_pos diagonal (25 videos)
- [x] Sweep 5: K_rot diagonal (25 videos)
- [x] Sweep 6: Full K_rot (625 videos, 6473s)
- [ ] Quantitative metrics sweep (headless)
- [ ] Heatmap generation
- [ ] Behavior classification

## Total: 825 videos generated
