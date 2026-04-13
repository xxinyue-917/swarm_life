# Characterization Plan — Multi-Species Parameter Sweeps

This document covers both **Part I: 2-Species Characterization** (baseline phase diagram) and **Part II: 3-Species Extension** (new behaviors unique to higher species counts).

---

# Part I — 2-Species Characterization

## Goal

Systematically characterize emergent behaviors in a 2-species particle life system by sweeping the 8-dimensional parameter space (4 K_pos + 4 K_rot entries) and recording video + screenshot outputs for visual analysis.

## Rationale — Why 2 Species First?

Starting with 2 species is a deliberate methodological choice with several justifications:

1. **Minimum interacting system**: 2 is the smallest number of species where cross-species interactions exist. Single-species systems only show self-cohesion and internal rotation — none of the interesting emergent behaviors (chase, encapsulation, orbital dynamics) require interaction between distinct populations.

2. **Tractable parameter space**: With 2 species, K_pos and K_rot are each 2×2 matrices, giving 8 total parameters. We can sweep 2 parameters at a time while holding the other 6 fixed, producing interpretable 2D heatmaps. With more species the parameter space explodes (3 species → 18 parameters, 4 species → 32).

3. **Clear symmetry structure**: The 2×2 case has only two meaningful symmetry classes — diagonal (self-cohesion / self-rotation) and off-diagonal (cross-species coupling). This lets us cleanly separate "within-group" dynamics from "between-group" dynamics.

4. **Reference phenomenology**: The behaviors that emerge in 2-species systems (predator-prey chase, mutual encapsulation, orbital pairs, flocking vs dispersion) form a well-defined classification scheme that grounds all later analysis in higher-species systems.

5. **Baseline for universality claim**: The paper's main argument is that minimal dual-matrix models reproduce behaviors from diverse physical systems. Establishing the behavioral phenotypes at the 2-species level provides the foundation for this claim; 3+ species extensions demonstrate how the same primitives scale.

6. **Computational budget**: 2×2 sweeps with 11-point grids produce 121 points per sweep, each taking ~15 s to simulate. Five sweeps = ~25 minutes per metric pass. This is fast enough to iterate on metric definitions, grid resolution, and physics parameters before committing to more expensive 3-species work.

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
- **Rationale**: This is the most fundamental 2-species sweep. The off-diagonal entries K₁₂ and K₂₁ encode how each species "feels" the other's presence. The four quadrants of this sweep correspond to the four classical 2-species interaction modes: mutual attraction (both positive → merger), mutual repulsion (both negative → separation), and asymmetric chase (opposite signs → predator-prey). Fixing both diagonal entries at a moderately positive value (0.6) ensures each species holds itself together, so we can isolate the effect of the cross-coupling. Zero K_rot eliminates confounding rotational effects, giving a clean baseline for comparison with Sweep 2.
- **Result**: 25 videos (+ 121 metric points)

### Sweep 2: `kpos_x_krot` — Position × rotation cases
- **Swept**: K₁₂ ∈ [-1, 1], K₂₁ ∈ [-1, 1], grid = 5, repeated for 4 K_rot cases
- **Fixed**: K₁₁ = K₂₂ = 0.6
- **K_rot cases**:
  - A: R = [[0, 0], [0, 0]] (no rotation — baseline, identical to Sweep 1)
  - B: R = [[0, +1], [+1, 0]] (symmetric → collective rotation)
  - C: R = [[0, +1], [-1, 0]] (antisymmetric → translation)
  - D: R = [[0, +1], [0, 0]] (one-way — shepherd-like)
- **Purpose**: How rotation coupling modifies positional behaviors
- **Rationale**: Sweep 1 establishes the positional phase diagram with no rotation. Sweep 2 asks: how does each of the fundamental K_rot symmetry classes perturb that phase diagram? The four K_rot cases span the space of adjacent-pair rotation patterns — zero (control), symmetric (which creates opposing tangential forces → collective rotation/bending), antisymmetric (which creates aligned tangential forces → net translation), and one-way (broken reciprocity). Each case reveals whether rotation coupling can transform one behavior into another (e.g., does adding symmetric rotation turn a chase into an orbit?). Case A duplicates Sweep 1 as a control — comparing A with the other three isolates the rotation contribution. This factorial design is far more informative than sweeping K_rot independently because it exposes interactions between positional and rotational couplings.
- **Result**: 100 videos (+ 484 metric points)

### Sweep 3: `krot_offdiag` — Cross-species rotation coupling
- **Swept**: R₁₂ ∈ [-1, 1], R₂₁ ∈ [-1, 1], grid = 5
- **Fixed**: K_pos = [[0.6, 0.3], [0.3, 0.6]], R₁₁ = R₂₂ = 0
- **Purpose**: Rotation behavior space with fixed attractive K_pos
- **Rationale**: The mirror of Sweep 1 — instead of sweeping position coupling with no rotation, we sweep rotation coupling with a mildly attractive position baseline. The fixed K_pos = [[0.6, 0.3], [0.3, 0.6]] ensures the species stay together (so rotation has something to act on) while being weakly coupled (so rotation can still reshape the formation). The four R₁₂/R₂₁ quadrants reveal the rotational analogues of attraction/repulsion: (+,+) is symmetric rotation (both species orbit in the same direction), (+,-) is antisymmetric (translation), (-,-) is counter-rotation, and mixed signs are intermediate. Together with Sweep 1, this isolates the "pure" effect of each matrix type. Self-rotation (R₁₁, R₂₂) is zeroed out here and explored separately in Sweep 5.
- **Result**: 25 videos (+ 121 metric points)

### Sweep 4: `kpos_diag` — Self-cohesion asymmetry
- **Swept**: K₁₁ ∈ [-1, 1], K₂₂ ∈ [-1, 1], grid = 5
- **Fixed**: K₁₂ = K₂₁ = 0.3, K_rot = all zeros
- **Purpose**: Effect of asymmetric self-cohesion
- **Rationale**: Sweeps 1-3 all fixed self-cohesion at 0.6, implicitly assuming that "each species holds itself together" is a constant condition. This sweep relaxes that assumption and asks: what if the two species differ in how strongly they self-attract? The four quadrants reveal behaviors that cannot occur with symmetric self-cohesion: (large K₁₁, small K₂₂) gives a tight species 1 cluster surrounded by a diffuse species 2 cloud (encapsulation precursor); (negative K₁₁, positive K₂₂) gives a "gas" species 1 mixed with a "liquid" species 2 (phase separation). This sweep is essential for showing that the model can reproduce heterogeneous swarm behaviors where different populations have qualitatively different internal dynamics — a key feature of biological and robotic swarms with different agent types. Cross-attraction fixed at 0.3 keeps the species interacting (otherwise they would just decouple).
- **Result**: 25 videos (+ 121 metric points)

### Sweep 5: `krot_diag` — Self-rotation
- **Swept**: R₁₁ ∈ [-1, 1], R₂₂ ∈ [-1, 1], grid = 5
- **Fixed**: K_pos = [[0.6, 0.3], [0.3, 0.6]], R₁₂ = R₂₁ = 0
- **Purpose**: Effect of self-rotation coupling
- **Rationale**: The diagonal of K_rot controls how each species rotates around itself — independent of the other species. In the microrobot paper, this corresponds to each species spinning in place (creating local vorticity). This sweep asks whether independent self-rotation of the two species produces emergent collective behaviors, or whether it stays decoupled. Because cross-rotation is zero, any global rotation observed here must come from hydrodynamic-like coupling through the radial forces (when two spinning species are nearby, their positions evolve correlated motions). This sweep also tests whether asymmetric self-rotation (e.g., species 1 spinning while species 2 is static) creates shearing or other non-equilibrium behaviors.
- **Result**: 25 videos (+ 121 metric points)

### Sweep 6: `krot_full` — All 4 K_rot entries
- **Swept**: R₁₁, R₁₂, R₂₁, R₂₂ ∈ [-1, 1], grid = 5 (5⁴ = 625 combinations)
- **Fixed**: K_pos = [[0.6, 0.3], [0.3, 0.6]]
- **Purpose**: Complete rotation characterization
- **Rationale**: Sweeps 3 and 5 explore the off-diagonal and diagonal components of K_rot separately. Sweep 6 is the exhaustive 4D sweep that captures all combinations, including cases where self-rotation and cross-rotation interact non-trivially. This is computationally expensive (625 points vs 121) but necessary for completeness: some behaviors (e.g., a species spinning while also being rotated by another species) cannot be predicted from the 2D projections alone. The output is too large for direct visualization as a single heatmap, so we analyze it through slices (holding 2 parameters fixed and viewing the other 2) or dimensionality reduction.
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
├── PLAN.md                     # This document
├── sweep_2species.py           # Video + screenshot sweep
├── sweep_metrics.py            # Headless metrics sweep (CSV output)
├── plot_heatmaps.py            # Generate heatmaps from CSV
├── make_grids.py               # Screenshot grid images
├── make_slides.py              # PowerPoint with video grids
├── make_metric_slides.py       # PowerPoint with metric heatmaps
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

### Rationale — Why These Metrics?

The characterization needs metrics that (a) can distinguish qualitatively different behaviors from simulation data alone, (b) have direct physical interpretations usable in a research paper, and (c) are robust to random initial conditions. The chosen metrics fall into four complementary categories:

1. **Spatial spread** (max distances, centroid distance) — captures the "shape" of each species and their relative positions. Distinguishes clustered vs dispersed formations, concentric vs separated groups.

2. **Speed and energy** (avg_speed, KE) — captures the "activity level" of the system. High values indicate non-equilibrium dynamics (chase, milling); low values indicate static or slowly relaxing configurations.

3. **Mean square displacement** (MSD) — captures "transport behavior". Small MSD indicates confined motion (trapped in a cluster); large MSD indicates diffusive or ballistic transport. The difference between per-group MSDs reveals whether one species is more mobile than the other.

4. **Inter-particle spacing** (spacing_*) — captures "local density". Complements max distances by showing typical (not extreme) inter-particle distances.

All metrics are measured across the **last 20%** of simulation timesteps and averaged. The burn-in period (first 80%) lets the system settle from random initial conditions, ensuring measurements reflect the asymptotic behavior rather than transients.


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

---

# Part II — 3-Species Extension

## Goal

Extend the 2-species characterization to 3-species systems, revealing emergent behaviors that are unique to higher species counts — especially cyclic dynamics (rock-paper-scissors chase), hierarchical encapsulation, and three-body orbital phenomena.

## Rationale — Why Extend to 3 Species?

The 2-species characterization establishes a behavioral phase diagram for the simplest non-trivial case. Extending to 3 species is essential for several reasons:

1. **Qualitatively new behaviors**: Some collective behaviors are impossible in 2-species systems and require at least 3 species to emerge. The canonical example is **cyclic chase** (species 1 chases 2 chases 3 chases 1), which has no 2-species analogue. Nested encapsulation (A inside B inside C) and three-body orbital systems also require 3+ species.

2. **Test of universality**: If the dual-matrix model is a universal framework for collective behaviors, it must handle systems with more than 2 interacting populations. Many biological and robotic swarms have multiple distinct agent types (e.g., leaders and followers, workers and queens, different sensor classes). Showing that our 2×2 characterization generalizes to 3×3 demonstrates that the parameter space of K_pos and K_rot captures behaviors at arbitrary species counts with the same primitives.

3. **Symmetry breaking**: With 2 species, the only symmetry classes are "symmetric" (A↔B equivalent) and "asymmetric" (A≠B). With 3 species, we get richer symmetries: fully symmetric (all equivalent), dihedral (two equivalent + one different), cyclic (A→B→C→A), and fully asymmetric. These richer symmetries allow qualitatively new equilibrium configurations.

4. **Parameter-space expansion**: The 2-species K_pos has 4 entries; the 3-species K_pos has 9. This gives us more degrees of freedom to encode complex interactions, enabling behaviors like "species 1 attracts 2 but repels 3" that cannot be expressed in 2-species matrices.

5. **Reference for larger systems**: Our formation demos (shape_formation, formation_from_image) already use many species (50-100). Understanding the 3-species case bridges the gap between the minimal 2-species phase diagram and the high-species formation studies — it reveals which effects scale and which are peculiar to the minimum-interaction case.

## Challenge: Parameter Space Explosion

3 species → **18 parameters total** (9 K_pos + 9 K_rot). This is too large for exhaustive sweeps:
- Full pairwise 2D sweeps: C(18,2) = 153 possible parameter pairs
- Full 4D sweeps like Sweep 6 in 2-species: 5⁴ = 625 points × multiple configs
- Full 6D sweep over K_pos alone: 5⁶ = 15,625 points

**Strategy**: Pick meaningful 2D slices based on symmetry and physical interpretation. Prefer parameter combinations that either (a) preserve some symmetry (reducing effective dimensionality) or (b) are known to produce interesting behavior classes.

## 3-Species Parameter Space

```
         [K₁₁ K₁₂ K₁₃]           [R₁₁ R₁₂ R₁₃]
K_pos =  [K₂₁ K₂₂ K₂₃]   K_rot = [R₂₁ R₂₂ R₂₃]
         [K₃₁ K₃₂ K₃₃]           [R₃₁ R₃₂ R₃₃]
```

## 3-Species Sweep Configurations

### Sweep A1: `3s_pairwise_12` — Isolated pair interaction (1↔2 with 3 as observer)
- **Swept**: K₁₂ ∈ [-1, 1], K₂₁ ∈ [-1, 1], grid = 5
- **Fixed**: K₁₁ = K₂₂ = K₃₃ = 0.6 (self-cohesion)
  K₁₃ = K₃₁ = K₂₃ = K₃₂ = 0.3 (third species as mild attractor)
  K_rot = all zeros
- **Purpose**: Reproduce the 2-species Sweep 1 behavior in the presence of a third species.
- **Rationale**: The most direct extension of the 2-species study. Answers the question: "does adding a third species perturb the 1↔2 phase diagram, or does the pair still behave like an isolated 2-species system?" Positive answer means the behavior is robust to the addition of new populations; negative answer means 3-species systems require full characterization. By fixing species 3's interactions as weakly attractive, we give it a concrete but non-dominating role — it provides a gentle background potential without dictating the outcome. This reveals whether the presence of a third population amplifies, dampens, or reshapes pairwise behaviors.

### Sweep A2: `3s_third_species` — Third species as variable
- **Swept**: K₁₃ ∈ [-1, 1], K₂₃ ∈ [-1, 1], grid = 5
- **Fixed**: K₁₁ = K₂₂ = K₃₃ = 0.6
  K₁₂ = K₂₁ = 0.5 (species 1 and 2 mutually attract)
  K₃₁ = K₃₂ = 0.3 (species 3's reciprocal)
  K_rot = all zeros
- **Purpose**: How does species 3 interact with a pre-bound 1+2 collective?
- **Rationale**: Sweep A1 fixes species 3's role and varies the 1↔2 pair. This sweep does the opposite: pin a "pair" (species 1 and 2 attractively coupled) and vary how species 3 relates to each. The four quadrants reveal: attracted to both (3 joins the collective), repelled by both (3 separates), attracted to one / repelled by the other (asymmetric shepherding), and mixed. This pattern is the minimal 3-species test for hierarchical formations — structures where one species mediates the interaction between the other two. It is the building block for encapsulation, where one species surrounds another.

### Sweep B1: `3s_cyclic` — Rock-paper-scissors (cyclic chase)
- **Swept**: K_forward (K₁₂ = K₂₃ = K₃₁) ∈ [-1, 1], K_backward (K₂₁ = K₃₂ = K₁₃) ∈ [-1, 1]
- **Fixed**: K₁₁ = K₂₂ = K₃₃ = 0.6 (self-cohesion)
  K_rot = all zeros
- **Purpose**: Map the cyclic chase behavior space — uniquely 3-species.
- **Rationale**: The most distinctive 3-species behavior is cyclic pursuit, where species 1 chases 2, 2 chases 3, and 3 chases 1. This is impossible in 2-species systems (there is no "third direction"). By parameterizing the 9 off-diagonal K_pos entries with just 2 numbers (K_forward for the cyclic direction and K_backward for the reverse), we reduce dimensionality while preserving the cyclic symmetry. The four quadrants show: (+,+) mutual attraction (no cycle), (+,0) pure forward cycle (classic rock-paper-scissors chase), (+,-) strong directional chase, and (-,-) mutual repulsion (no cycle). This sweep is the 3-species analogue of Sweep 1 in 2 species — the canonical phase diagram specific to this species count.

### Sweep C1: `3s_symmetric` — Fully symmetric parameterization
- **Swept**: K_self (K₁₁ = K₂₂ = K₃₃) ∈ [-1, 1], K_cross (all 6 off-diagonals) ∈ [-1, 1]
- **Fixed**: K_rot = all zeros
- **Purpose**: Aggregate behavior when all species are equivalent.
- **Rationale**: When all off-diagonal entries are equal and all diagonal entries are equal, the 3-species system is effectively a single-species system with a 3-way label distinction. This sweep answers: "what is the baseline behavior when species 3 is just a relabeled version of species 1 and 2?" The 2D slice (K_self, K_cross) is a direct analogue of Sweep 4 in the 2-species study. Comparing this with the 2-species diagonal sweep tests whether adding species changes the basic phase structure. This is the 3-species "mean-field" characterization — it smooths over species-specific effects to reveal the aggregate dynamics.

### Sweep C2: `3s_krot_symmetric` — Symmetric K_rot with fixed attractive K_pos
- **Swept**: R_cross (all 6 off-diagonal K_rot entries) ∈ [-1, 1],
  R_self (R₁₁ = R₂₂ = R₃₃) ∈ [-1, 1]
- **Fixed**: K_pos = diag 0.6, off-diag 0.3 (mild attraction)
- **Purpose**: Aggregate rotation behavior for 3 species.
- **Rationale**: The rotation analogue of Sweep C1. Fixes K_pos at a mild attractive baseline and sweeps the symmetric version of K_rot. Reveals whether 3 species collectively rotate as a single body (symmetric R_cross), rotate independently (R_self only), or translate as a unit. This is the 3-species analogue of Sweep 3 and Sweep 5 combined.

### Sweep D1: `3s_hierarchical` — Nested encapsulation candidates
- **Swept**: K₁₃ ∈ [-1, 1] (species 1 attracted to 3), K₂₃ ∈ [-1, 1] (species 2 attracted to 3)
- **Fixed**: K₁₁ = K₂₂ = K₃₃ = 0.6
  K₁₂ = K₂₁ = -0.5 (species 1 and 2 mutually repel)
  K₃₁ = K₃₂ = 0.6 (species 3 strongly attracts both)
  K_rot = all zeros
- **Purpose**: Find encapsulation regimes where one species mediates between two mutually repulsive species.
- **Rationale**: Classical encapsulation behavior requires a specific K_pos structure: two "outer" species that repel each other but both attract (or are attracted by) a "core" species. The core effectively sticks the otherwise-separating outer species together. This sweep asks which combinations of K₁₃ and K₂₃ produce stable three-layer formations. It is a targeted probe for a specific emergent structure, rather than a blind sweep. Results here would directly demonstrate that the dual-matrix model can reproduce containment behaviors from swarm robotics (where one robot population surrounds and isolates another).

## 3-Species Budget

| Sweep | Grid | Points | Est. time (metrics, headless) |
|-------|------|--------|------------------------------|
| A1 `3s_pairwise_12` | 11×11 | 121 | ~30 min |
| A2 `3s_third_species` | 11×11 | 121 | ~30 min |
| B1 `3s_cyclic` | 11×11 | 121 | ~30 min |
| C1 `3s_symmetric` | 11×11 | 121 | ~30 min |
| C2 `3s_krot_symmetric` | 11×11 | 121 | ~30 min |
| D1 `3s_hierarchical` | 11×11 | 121 | ~30 min |
| **Total** | | **726** | **~3 hours** |

## 3-Species Metrics

Extend the 14 two-species metrics with per-species and per-pair versions:

| Category | New metrics |
|----------|-------------|
| Max distance | max_d3 (add to max_d1, max_d2) |
| Avg speed | avg_speed3 |
| MSD | MSD3 |
| Spacing (within group) | spacing3 |
| Spacing (between groups) | spacing_12, spacing_13, spacing_23 (split cross-spacing) |
| Centroid distances | centroid_12, centroid_13, centroid_23 |

Total: ~20 metrics per 3-species sweep.

## 3-Species Implementation Strategy

1. **Extend `sweep_metrics.py`** to support arbitrary n_species via a new CONFIG entry. Add 3-species sweep types (`3s_pairwise_12`, `3s_third_species`, `3s_cyclic`, `3s_symmetric`, `3s_krot_symmetric`, `3s_hierarchical`).

2. **Extend `compute_metrics_snapshot()`** to handle 3 species: loop over species indices for per-group metrics, compute all 3 pairwise centroid distances and cross-spacings.

3. **Update `plot_heatmaps.py`** to use the extended metric set when a 3-species CSV is loaded. Grid layout becomes larger (~20 metrics instead of 14).

4. **Run sweeps sequentially** starting with B1 (cyclic) — the most interesting 3-species-specific behavior.

5. **Generate slides** using a script similar to `make_metric_slides.py`, adapted for 3-species metric layout.

---

## Status

### 2-Species
- [x] Plan created
- [x] Sweep 1: K_pos off-diagonal (25 videos, 255s)
- [x] Sweep 2: K_pos × K_rot cases (100 videos, 1022s)
- [x] Sweep 3: K_rot off-diagonal (25 videos, 255s)
- [x] Sweep 4: K_pos diagonal (25 videos)
- [x] Sweep 5: K_rot diagonal (25 videos)
- [x] Sweep 6: Full K_rot (625 videos, 6473s)
- [x] Quantitative metrics sweep (headless) — all 5 sweeps, 21×21 grid, 5 seeds averaged
- [x] Heatmap generation — all 14 metrics, bicubic interpolation
- [x] Metric slides with descriptions and example matrices
- [ ] **Re-run metrics with per-seed CSV** — sweep_metrics.py updated to save each seed as its own row (5 rows per parameter point with `seed` column). plot_heatmaps.py auto-averages across seeds. Enables variance analysis and error bars. Pending re-run.
- [ ] Behavior classification

### 3-Species
- [x] Plan created
- [ ] Extend sweep_metrics.py for n_species=3
- [ ] Extend compute_metrics_snapshot() for 3-species metrics
- [ ] Run Sweep B1 (cyclic) — first priority
- [ ] Run Sweep A1, A2 (pairwise)
- [ ] Run Sweep C1, C2 (symmetric)
- [ ] Run Sweep D1 (hierarchical)
- [ ] Generate heatmaps and slides

## Total: 825 videos generated (2-species)
