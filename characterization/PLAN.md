# Characterization Plan — 2-Species Parameter Sweep

## Goal

Systematically characterize emergent behaviors in a 2-species particle life system by sweeping the 8-dimensional parameter space (4 K_pos + 4 K_rot entries) and measuring quantitative metrics at each point.

## Approach

Start with 2 groups (species) and limited parameters:
- **Position Matrix (K_pos)**: 2x2, entries K₁₁, K₁₂, K₂₁, K₂₂ each in [-1, 1]
- **Revolution Matrix (K_rot)**: 2x2, entries R₁₁, R₁₂, R₂₁, R₂₂ each in [-1, 1]
- Multiple matrix configurations to sweep (diagonal-only, off-diagonal-only, full)

## Parameter Space

```
K_pos = [[K₁₁, K₁₂],    K_rot = [[R₁₁, R₁₂],
         [K₂₁, K₂₂]]             [R₂₁, R₂₂]]
```

Full space = 8 dimensions. We reduce dimensionality by fixing some entries.

## Sweep Configurations

### Sweep 1: K_pos off-diagonal (K₁₂ vs K₂₁)
- **Swept**: K₁₂ ∈ [-1, 1], K₂₁ ∈ [-1, 1]
- **Fixed**: K₁₁ = K₂₂ = 0.6 (self-cohesion), K_rot = all zeros
- **Purpose**: Baseline — how cross-species position coupling creates chase, encapsulate, separate
- **Output**: 2D heatmap grid

### Sweep 2: K_pos off-diagonal × K_rot cases
- **Swept**: K₁₂ ∈ [-1, 1], K₂₁ ∈ [-1, 1]
- **Fixed**: K₁₁ = K₂₂ = 0.6
- **K_rot cases**:
  - A: R = [[0, 0], [0, 0]] (no rotation)
  - B: R = [[0, +1], [+1, 0]] (symmetric rotation)
  - C: R = [[0, +1], [-1, 0]] (antisymmetric rotation)
  - D: R = [[0, +1], [0, 0]] (one-way rotation)
- **Purpose**: How rotation coupling modifies positional behaviors
- **Output**: 4 × 2D heatmap grids (one per K_rot case)

### Sweep 3: K_rot off-diagonal (R₁₂ vs R₂₁)
- **Swept**: R₁₂ ∈ [-1, 1], R₂₁ ∈ [-1, 1]
- **Fixed**: K_pos = [[0.6, 0.3], [0.3, 0.6]], R₁₁ = R₂₂ = 0
- **Purpose**: Rotation behavior space with fixed attractive K_pos
- **Output**: 2D heatmap grid

### Sweep 4: K_pos diagonal (K₁₁ vs K₂₂)
- **Swept**: K₁₁ ∈ [-1, 1], K₂₂ ∈ [-1, 1]
- **Fixed**: K₁₂ = K₂₁ = 0.3, K_rot = all zeros
- **Purpose**: Effect of asymmetric self-cohesion
- **Output**: 2D heatmap grid

### Sweep 5: K_rot diagonal (R₁₁ vs R₂₂)
- **Swept**: R₁₁ ∈ [-1, 1], R₂₂ ∈ [-1, 1]
- **Fixed**: K_pos = [[0.6, 0.3], [0.3, 0.6]], R₁₂ = R₂₁ = 0
- **Purpose**: Effect of self-rotation coupling
- **Output**: 2D heatmap grid

### Sweep 6: Full K_rot (R₁₁, R₁₂, R₂₁, R₂₂)
- **Swept**: All 4 R entries
- **Fixed**: K_pos = [[0.6, 0.3], [0.3, 0.6]]
- **Purpose**: Complete rotation characterization
- **Note**: 4D sweep — use coarser grid (5-7 points per axis)

## Metrics Measured

| Metric | Symbol | Description |
|--------|--------|-------------|
| Species 1 radius | R₁ | Mean distance of species 1 particles from their centroid |
| Species 2 radius | R₂ | Mean distance of species 2 particles from their centroid |
| Radius difference | Rdiff | |R₁ - R₂| — asymmetry in cluster sizes |
| Kinetic energy | K | Mean kinetic energy — activity level |
| Intra-species spacing | d₁₁, d₂₂ | Mean pairwise distance within each species |
| Inter-species spacing | d₁₂ | Mean pairwise distance between species |
| Revolutions | revs | Cumulative angular revolutions — rotational behavior |
| Polarization | Φ | Velocity alignment — flocking indicator |
| Angular momentum | L | Net rotation — orbital behavior |
| Mixing index | M | Species intermixing — segregation indicator |

## Simulation Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| n_particles | 50 | Per species (100 total) |
| sim_width | 10.0 | Meters |
| sim_height | 10.0 | Meters |
| dt | 0.05 | Timestep |
| max_speed | 5.0 | Velocity clamp |
| r_max | 5.0 | Interaction radius |
| beta | 0.3 | Repulsion zone boundary |
| a_rot | 3.0 | Rotation coupling strength |
| burnin_steps | 500 | Steps before measuring |
| measure_steps | 1000 | Steps to average metrics |
| sample_stride | 5 | Sample every N steps |
| n_seeds | 3 | Random seeds for averaging |
| grid_points | 11 | Points per swept axis: [-1, -0.8, ..., 0.8, 1] |

## Output Structure

```
characterization/
├── PLAN.md                 # This document
├── sweep_2species.py       # Main sweep script (video + screenshot output)
├── plot_results.py         # Metrics visualization (for later quantitative analysis)
├── results/
│   ├── sweep_kpos_offdiag/
│   │   ├── config.json     # Sweep parameters for reproducibility
│   │   ├── videos/         # One .mp4 per parameter combination
│   │   │   ├── K12=+0.50_K21=-0.50.mp4
│   │   │   └── ...
│   │   └── screenshots/    # Final frame of each simulation
│   │       ├── K12=+0.50_K21=-0.50.png
│   │       └── ...
│   ├── sweep_kpos_x_krot/
│   ├── sweep_krot_offdiag/
│   ├── sweep_kpos_diag/
│   └── sweep_krot_diag/
```

Each video shows the simulation with matrix values overlaid. Each screenshot captures the final state. Browse visually, then decide which metrics to quantify.

## Status

- [x] Plan created
- [x] Sweep 1: K_pos off-diagonal (25 videos, 255s)
- [x] Sweep 2: K_pos × K_rot cases (100 videos, 1022s)
- [x] Sweep 3: K_rot off-diagonal (25 videos, 255s)
- [x] Sweep 4: K_pos diagonal (25 videos)
- [x] Sweep 5: K_rot diagonal (25 videos)
- [x] Sweep 6: Full K_rot (625 videos, 6473s)
- [ ] Plotting and analysis
- [ ] Behavior classification
