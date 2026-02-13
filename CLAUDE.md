# Swarm Life Research Project

## Research Context

This project investigates **emergent collective behaviors** in decentralized multi-agent systems using particle life simulation as a computational model for swarm robotics. The core research question: **How do simple local interaction rules between agents produce complex, coordinated global behaviors without centralized control?**

### Relevance to Robotics

Particle life serves as a **low-dimensional proxy** for swarm robotics:
- Particles → Physical robots with limited sensing range
- Interaction matrices → Local communication/sensing protocols
- Emergent patterns → Desired collective behaviors (formation, pursuit, encapsulation)

Understanding parameter-behavior mappings in simulation enables **principled design** of swarm control laws that can be transferred to physical robot teams.

---

## Physics Model

### Dual-Force Architecture

Each particle experiences two force components from neighbors:

**1. Radial Force (Position Matrix K_pos)**
```
F_radial = (K_pos[i,j] × a_att - a_rep/√r) × r̂
```
- Controls attraction/repulsion between species
- K_pos > 0: net attraction dominates at distance
- K_pos < 0: net repulsion dominates
- Creates: clustering, separation, pursuit, encapsulation

**2. Tangential Force (Orientation Matrix K_rot)**
```
F_tangential = -10 × K_rot[i,j] × (ω_j/ω_max) × (a_rot/r) × t̂
```
- Controls rotational/swirling dynamics
- Couples angular velocity between species
- Creates: collective rotation, vortices, orbital motion

### Key Parameters

| Parameter | Symbol | Role |
|-----------|--------|------|
| `a_att` | Attraction strength | Base attraction coefficient |
| `a_rep` | Repulsion strength | Short-range repulsion (prevents collision) |
| `a_rot` | Rotation coupling | Strength of angular velocity influence |
| `max_speed` | v_max | Velocity clamp for stability |
| `max_angular_speed` | ω_max | Angular velocity normalization |
| `dt` | Δt | Integration timestep |

---

## Codebase Architecture

```
src/
├── particle_life.py    # Core simulation + interactive visualization
├── metrics.py          # Quantitative behavior characterization
├── sweep.py            # Parameter space exploration
├── plot_heatmaps.py    # Visualization of sweep results
└── save_videos.py      # Video generation from presets

presets/                # Discovered behavior configurations (JSON)
```

### Core Classes

**Config** (`particle_life.py:18-71`)
- Dataclass holding all simulation parameters
- JSON serialization for preset save/load
- Stores both interaction matrices

**ParticleLife** (`particle_life.py:73-868`)
- Main simulation class with pygame visualization
- `compute_velocities()`: Force calculation (lines 357-418)
- `step()`: Euler integration + boundary handling
- Interactive matrix editing during runtime

### Reusable Drawing API (ParticleLife base class)

All demos inherit from `ParticleLife`. The following methods are available for any new demo to call directly via `self.`:

| Method | Description | Example |
|--------|-------------|---------|
| `to_screen(pos)` | Convert sim position (meters) → screen pixels | `px = self.to_screen([1.0, 2.0])` |
| `draw_particles()` | Draw all particles with anti-aliased circles | `self.draw_particles()` |
| `draw_particle(x, y, color, r=None)` | Draw a single anti-aliased circle at screen coords | `self.draw_particle(100, 200, (255,0,0))` |
| `draw_pause_indicator()` | Show "PAUSED" text when `self.paused` is True | `self.draw_pause_indicator()` |
| `draw_centroid_spine(line_width=3)` | Draw line connecting species centroids; returns screen pts | `pts = self.draw_centroid_spine()` |
| `draw_centroid_markers(pts=None, head_r=10, tail_r=6)` | Draw colored circles at species centroids | `self.draw_centroid_markers(pts)` |
| `draw_swarm_centroid()` | Draw hollow circle at overall swarm centroid | `self.draw_swarm_centroid()` |

**Typical demo `draw()` pattern:**
```python
def draw(self):
    self.screen.fill((255, 255, 255))
    self.draw_particles()
    # --- demo-specific elements (walls, waypoints, etc.) ---
    pts = self.draw_centroid_spine()
    self.draw_centroid_markers(pts)
    self.draw_swarm_centroid()
    self.draw_pause_indicator()
```

**Also available but not yet in base class** (duplicated in `snake_demo`, `multi_species_demo`, `shape_formation`):
- `draw_single_matrix()` / `draw_matrix_viz()` — matrix heatmap with edit highlighting
- `draw_control_indicator()` — turn/speed joystick widget

These depend on demo-specific state (`matrix_edit_mode`, `turn_input`, `speed_input`). Copy from an existing demo if needed.

### Metrics System (`metrics.py`)

Quantitative descriptors for characterizing emergent behaviors:

| Metric | Description | What it reveals |
|--------|-------------|-----------------|
| `R1, R2` | Average radius from centroid | Cluster compactness |
| `Rdiff` | R1 - R2 | Relative clustering asymmetry |
| `K` | Kinetic energy | Activity level |
| `d11, d22` | Intra-species spacing | Same-species cohesion |
| `d12` | Inter-species spacing | Cross-species mixing/separation |
| `revs` | Cumulative revolutions | Rotational behavior strength |

### Parameter Sweep (`sweep.py`)

Systematically explores the (k12_pos, k21_pos) × orientation_case space:

**Orientation Cases:**
- **A**: No orientation coupling (k12_ori=0, k21_ori=0)
- **B**: Symmetric attraction (k12_ori=1, k21_ori=1)
- **C**: Antisymmetric/lanes (k12_ori=1, k21_ori=-1)
- **D**: One-way influence (k12_ori=1, k21_ori=0)

**Usage:**
```bash
python src/sweep.py --grid 10 --ori-cases A,B,C,D --burnin 500 --steps 1000 --out results/
python src/plot_heatmaps.py --csv results/results.csv --out plots/
```

---

## Discovered Behaviors (Presets)

### Two-Species (2_*)

| Preset | K_pos pattern | Behavior |
|--------|---------------|----------|
| `2_chase` | Asymmetric | Predator-prey pursuit |
| `2_encapsulate` | Strong diagonal, mixed cross | One species surrounds another |
| `2_move_together` | Symmetric positive | Cohesive flocking |
| `2_sun_earth` | With orientation matrix | Orbital dynamics |
| `2_dynamically_rotate` | With orientation | Dynamic rotation |

### Three-Species (3_*)

| Preset | Description |
|--------|-------------|
| `3_chase` | Cyclic pursuit (A→B→C→A) |
| `3_encapsulate` | Nested containment shells |
| `3_planet` | Multi-body orbital system |
| `3_rotate_together` | Collective synchronized rotation |

---

## Research Directions

### 1. Behavior Classification
- **Goal**: Map (K_pos, K_rot) → behavior categories automatically
- **Approach**: Cluster metric vectors from sweeps
- **Output**: Phase diagrams of swarm behaviors

### 2. Inverse Design
- **Goal**: Given desired behavior, find interaction matrices
- **Approach**: Optimization/search in parameter space
- **Application**: Design swarm protocols for specific tasks

### 3. Stability Analysis
- **Goal**: Understand which configurations are stable vs transient
- **Approach**: Track metric variance over time, perturbation response
- **Relevance**: Robust swarm behaviors for real robots

### 4. Scalability
- **Goal**: How do behaviors change with N (particle count)?
- **Questions**:
  - Do patterns persist at different scales?
  - What are finite-size effects?

### 5. Robot Transfer
- **Goal**: Validate simulation findings on physical robot platforms
- **Considerations**:
  - Sensing noise and range limitations
  - Communication delays
  - Physical constraints (collision, inertia)

### 6. Extended Physics
- **Additional forces**: Lennard-Jones, Morse potential
- **Heterogeneous parameters**: Per-particle variation
- **Time-varying matrices**: Adaptive behaviors
- **3D extension**: Volumetric swarm dynamics

---

## Development Commands

```bash
# Interactive simulation
python src/particle_life.py
python src/particle_life.py --load presets/3_planet.json

# Parameter sweep (headless)
python src/sweep.py --grid 10 --ori-cases A,B --burnin 200 --steps 500 --seeds 3

# Generate heatmaps
python src/plot_heatmaps.py --csv sweep_out/results.csv --out plots/

# Record videos from presets
python src/save_videos.py                              # All presets
python src/save_videos.py --load presets/3_chase.json  # Single preset

# Dependencies
pip install pygame numpy matplotlib pandas tqdm
pip install opencv-python  # For video recording
```

---

## Key Code Locations

| Feature | File | Lines |
|---------|------|-------|
| Force computation | `particle_life.py` | 357-418 |
| Boundary conditions | `particle_life.py` | 456-467 |
| Matrix editing UI | `particle_life.py` | 556-628 |
| Metric calculations | `metrics.py` | All |
| Sweep configuration | `sweep.py` | 36-53 |
| Orientation cases | `sweep.py` | 25-34 |

---

## Experimental Notes

*Space for recording observations, hypotheses, and findings:*

### Observations
- Asymmetric K_pos matrices tend to produce more dynamic (non-equilibrium) behaviors
- Strong diagonal values (same-species attraction) create tight clusters
- Orientation matrix effects are most visible when radial forces are balanced

### Hypotheses to Test
- [ ] Is there a critical K_pos threshold for chase vs encapsulate transition?
- [ ] Do 3-species systems show qualitatively different phase diagrams than 2-species?
- [ ] How does orientation coupling modify stability boundaries?

### TODO
- [ ] Run comprehensive sweep with finer grid resolution
- [ ] Implement additional metrics (polarization, angular momentum)
- [ ] Compare simulation results with physical robot experiments
- [ ] Explore higher species counts (N > 3)
