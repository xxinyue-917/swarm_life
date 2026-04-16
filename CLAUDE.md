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
├── particle_life.py       # Core 2D simulation + interactive visualization
├── particle_life_3d.py    # Core 3D simulation with orthographic projection
├── multi_species_demo.py  # N-species swarm with PD heading controller
├── snake_demo.py          # Chain steering with delayed turn propagation
├── shape_formation.py     # PID joint-angle controller for arbitrary shapes
├── formation_locomotion.py# Shape + locomotion (K_rot forward + K_pos lateral)
├── formation_transport.py # Shape + locomotion + object pick-up and delivery
├── formation_from_image.py# Shape from image via K_pos topology (Gaussian kernel)
├── snake_demo_3d.py       # 3D snake navigation
└── snake_maze_3d.py       # 3D maze with intro animation + autopilot

└── snake_maze_3d.py       # 3D maze with intro animation + autopilot

characterization/              # 2-species parameter sweep study
├── PLAN.md                    # Sweep configurations and status
├── sweep_2species.py          # Video + screenshot sweep engine
├── plot_results.py            # Metrics visualization (for later)
└── results/                   # 825 videos + screenshots (gitignored)

presets/                       # Discovered behavior configurations (JSON)
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

### Event Handling — Custom Keys in Subclasses

If a demo subclass adds custom WASD or arrow-key handlers in `handle_events`, gate them on `not self.show_matrix` so the parent's matrix editor (M key) still works:

```python
def handle_events(self):
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            # Custom controls ONLY when matrix editor is closed.
            if not getattr(self, 'show_matrix', False):
                if event.key == pygame.K_UP:    ...   # e.g. adjust parameter
                elif event.key == pygame.K_w:   ...
                # (continue/return here)

            # Preset/toggle keys that don't conflict can stay unconditional:
            if event.key == pygame.K_1:  self._load_preset(...)
        pygame.event.post(event)
    return super().handle_events()
```

Without this gate the subclass intercepts WASD before the parent sees it, and the matrix cursor can't be moved. Symptom: "W/S changes the number instead of moving location in the matrix." Reference implementation: `behavior_reproduction/planet.py`.

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

## Formation Locomotion (`formation_locomotion.py`)

Combines shape formation with two independent locomotion channels:

### Locomotion Mechanisms

| Channel | Matrix | Direction | Mechanism |
|---------|--------|-----------|-----------|
| **Forward** | K_rot (antisymmetric) | Perpendicular to chain | Equal tangential forces on both species in each pair |
| **Lateral** | K_pos (forward_bias) | Along chain axis | Asymmetric attraction: stronger toward tail → drift toward head |

**Key insight**: Antisymmetric K_rot and symmetric K_rot (PID shape) are independent — antisymmetric forces shift all segment angles θ equally, so joint angles φ = θ[j+1] − θ[j] are unchanged. The PID shape controller and locomotion don't interfere.

### Compression Effect on Curved Chains

On a C-shape, uniform antisymmetric K_rot creates tangential forces perpendicular to each local segment. Because segments point in different directions on a curve, these forces have a net inward component → the formation compresses. This is the true physics of local-only tangential drive and cannot be eliminated without global knowledge. Shape compression is resisted by K_pos cross-attraction (local inter-species radial forces).

### Controls

```
↑/↓:   Forward speed (K_rot antisymmetric, 0 to 0.02, step 0.005)
←/→:   Lateral speed (K_pos forward_bias, step 0.02)
1-4:   Shape pattern (STRAIGHT/U/M/HUG)
G/F:   Draw open shape / filled contour
[/]:   Adjust curvature (phi0)
T:     Toggle trajectory display
+/-:   Species count (2-10)
C:     Toggle PID control
```

---

## Formation Transport (`formation_transport.py`)

Extends formation locomotion with a passive object pick-up and delivery task. Demonstrates **collective caging transport** — a classic swarm robotics problem.

### Object Physics

The object uses **hard-contact collision response** (not force-based):
1. **Position correction**: If a particle overlaps the object, the object is pushed out immediately — no overlap allowed
2. **Velocity transfer**: The object receives 30% of each contacting particle's approach velocity — smooth pushing at the speed of the swarm

This prevents the "overlap then impulse" artifact of force-based models.

### Task Flow

1. Particles spawn in a constrained region (left side)
2. PID forms them into a C-shape
3. User moves the C toward the orange object using ↑/↓/←/→
4. Tighten the C with `]` to capture (increase phi0)
5. Transport the object to the green goal zone
6. Loosen with `[` to release

### Controls (additional)

```
N:     New random object position
[/]:   Tighten/loosen C-shape (capture/release)
T:     Toggle full trajectory display
```

---

## Formation From Image (`formation_from_image.py`)

Self-organize particles into arbitrary shapes using only K_pos — no K_rot, no PID. Upload a black-and-white image or use built-in shapes.

### Pipeline

1. **Image → mask**: Black pixels = shape (threshold at 128)
2. **Sample N points**: Lloyd's relaxation (centroidal Voronoi tessellation) for uniform spacing — random init then iteratively move each point to its Voronoi region centroid (20 iterations, CV ≈ 0.11)
3. **Gaussian K_pos**: `K_pos[i,j] = 0.3 × exp(-dist²/2σ²)` — close species attract strongly, far species weakly
4. **Self-organization**: Radial forces find equilibrium that approximates the target shape

**Key insight**: K_pos acts as a **weighted adjacency graph** encoding spatial topology. The shape emerges from which species are connected and how strongly — not from explicit coordinate targets.

**σ auto-tuning**: σ = median(nearest-neighbor distances) × scale_factor. Adjustable at runtime with `[`/`]`.

### Controls

```
L:     Load image (type path in console)
1-9:   Built-in: Circle/Star/Square/Ring/Triangle/Cross/Crescent/L/Arrow
R:     Reset near target positions
S:     Scatter randomly (test self-organization)
+/-:   Species count (step 5/10)
[/]:   Adjust sigma (neighborhood size)
T:     Toggle target overlay
V/I/H: Centroids / Info / Hide GUI
```

---

## Characterization Study (`characterization/`)

Systematic parameter sweeps that map K_pos and K_rot interaction matrices to emergent swarm behaviors. The characterization produces both qualitative output (videos/screenshots for visual inspection) and quantitative output (metric heatmaps as CSV + PNG).

### Structure

```
characterization/
├── PLAN.md                # 2-species + 3-species plans with full rationales
├── sweep_2species.py      # Video/screenshot sweep (pygame window visible)
├── sweep_metrics.py       # Headless metric sweep (CSV output, ~10x faster)
├── plot_heatmaps.py       # Generate heatmap PNGs from metric CSVs
├── make_grids.py          # Create 5×5 screenshot grid composites
├── make_slides.py         # PowerPoint slides with video grids
├── make_metric_slides.py  # PowerPoint slides with metric heatmaps and descriptions
└── results/               # All outputs (gitignored)
    ├── {sweep_name}/      # Video sweeps: videos/ + screenshots/
    ├── grids/             # Composite grid images
    └── metrics/           # Metric CSVs + per-sweep heatmap PNGs
```

### 2-Species Sweeps (complete)

6 sweep types, each varying 2 parameters while fixing the rest:
1. **kpos_offdiag** — K₁₂ vs K₂₁ (cross-attraction)
2. **kpos_x_krot** — K₁₂ vs K₂₁ with 4 K_rot cases (A/B/C/D)
3. **krot_offdiag** — R₁₂ vs R₂₁ (cross-rotation)
4. **kpos_diag** — K₁₁ vs K₂₂ (self-cohesion asymmetry)
5. **krot_diag** — R₁₁ vs R₂₂ (self-rotation)
6. **krot_full** — full 4D K_rot sweep (625 points)

**Status**: 825 videos + 5 metric CSVs (11×11 grids) generated. Metric heatmaps complete for all 14 metrics (max distances, centroid distance, speeds, KE, MSDs, spacings).

### 3-Species Sweeps (planned)

6 sweep types designed to reveal behaviors unique to 3+ species:
- **3s_pairwise_12** — reproduce 2-species sweep with 3rd species as observer
- **3s_third_species** — vary species 3's interactions with a pre-bound 1+2 pair
- **3s_cyclic** — rock-paper-scissors chase (K_forward vs K_backward)
- **3s_symmetric** — all-species-equivalent parameterization
- **3s_krot_symmetric** — symmetric rotation baseline
- **3s_hierarchical** — encapsulation/containment regimes

**Status**: Plan documented in `characterization/PLAN.md` Part II. Implementation pending.

### Metric Definitions

All metrics measured over the last 20% of simulation timesteps and averaged:

| Category | Metrics |
|----------|---------|
| **Spatial spread** | max_d1, max_d2, centroid_dist |
| **Speed & energy** | avg_speed, avg_speed1, avg_speed2, KE |
| **Mean square displacement** | MSD, MSD1, MSD2 |
| **Inter-particle spacing** | spacing_all, spacing_same, spacing1, spacing2 |

### Running a Sweep

```bash
# Video sweep (visual output)
python characterization/sweep_2species.py        # uses CONFIG['sweep_type']

# Headless metric sweep (CSV output)
python characterization/sweep_metrics.py --sweep kpos_offdiag

# Generate heatmaps from CSV
python characterization/plot_heatmaps.py --sweep kpos_offdiag

# Generate PowerPoint slides
python characterization/make_metric_slides.py
```

Edit the CONFIG dict at the top of each script to tune grid resolution, physics parameters, or fixed matrix values.

---

## Behavior Reproduction (`behavior_reproduction/`)

Reproduce collective behaviors from other swarm systems (microrobots, flocking, bacterial colonies) using only the dual-matrix model. Demonstrates that the same behavioral primitives emerge from radically different physical mechanisms.

### Structure

```
behavior_reproduction/
├── PLAN.md                    # Target behaviors and mapping
├── microrobot_collectives.py  # Reproduce Gardi et al. 2022 microrobot modes
├── flocking.py                # Flocking presets (separation, aggregation, cohesion)
├── galaxy.py                  # Galaxy morphology (elliptical, disk, ring, merger, diff. rotation)
├── planet.py                  # Planetary systems (per-pair beta — scoped physics exception)
├── research_papers/           # Reference papers (gitignored)
├── presets/                   # Discovered parameter configs
└── results/                   # Videos/screenshots (gitignored)
```

### Targets

**Microrobot Collectives** (Gardi et al., Nature Communications 2022):
- Rotation modes (tight/medium/wide) — symmetric K_rot + attractive K_pos
- Chain formation (connected/separated) — tridiagonal K_pos with many species
- Oscillation — time-varying sign-flipped K_rot

**Flocking** (Reynolds/Vicsek):
- Separation → Zone 1 short-range repulsion (built into kernel)
- Aggregation → K_pos positive cross-attraction
- Cohesion → chain K_pos forward_bias + antisymmetric K_rot

**Galaxy Morphology**:
- Elliptical → single-species self-attracting blob
- Rotating disk → bulge + disk with antisymmetric K_rot
- Ring galaxy (Cartwheel) → intruder species repels disk outward
- Galaxy merger → two rotating clusters with weak cross-attraction
- Differential rotation → inner disk K_rot > outer disk K_rot

**Planetary Systems** (`planet.py` — scoped physics exception):
- The shared engine uses a global scalar `beta`, which forces every bound pair to collapse to the same limit-cycle radius `r_eq ≈ beta × r_max` — so multi-planet systems with differentiated orbital radii are impossible under the shared engine.
- `planet.py` is the only file allowed to override `compute_velocities`. It defines a local JIT kernel identical to the shared engine except that `beta[i, j]` is a per-pair matrix. Each species pair now has its own limit-cycle radius.
- Presets: Sun-Earth (r≈3.0) · Sun-Earth-Moon (hierarchical: r_Earth≈3.6, r_Moon≈1.2 from Earth) · Solar System (4 planets at r=1.5/2.5/3.5/4.5 via staggered beta) · Solar + Moon · Saturn + rings.
- Hierarchical orbit design: Moon has `K_pos[Moon, Sun] = 0` so it ignores Sun directly; it couples only to Earth and tracks Earth's moving position automatically (overdamped dynamics, no inertia needed).
- Moon orbit must start *perpendicular* to the Sun-Earth axis to be visible — colinear initial conditions make Earth's and Moon's tangential directions coincide, producing visual translation instead of orbit.
- Intra-species cohesion uses diagonal `K_pos = 0.9–1.0` so each body stays a tight cluster while orbiting.
- Rationale documented in `behavior_reproduction/PLAN.md` (Planetary System Implementation Notes) including the multi-agent debate that established per-pair beta as the only viable approach.

### Status

- [x] Microrobot rotation modes implemented
- [x] Microrobot chain mode (tridiagonal K_pos)
- [x] Oscillation with sinusoidal K_rot
- [x] Flocking (separation, aggregation, cohesion presets)
- [x] Galaxy (elliptical, rotating disk, ring, merger, differential rotation)
- [x] Planetary systems (via per-pair beta, scoped physics exception in planet.py only)
- [ ] Quantitative comparison with published results

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

# Demo simulations
python src/multi_species_demo.py       # PD heading controller demo
python src/snake_demo.py               # Snake with delayed turn propagation
python src/shape_formation.py          # PID shape formation + mouse drawing
python src/formation_locomotion.py     # Shape + locomotion (K_rot + K_pos)
python src/formation_transport.py      # Shape + locomotion + object transport
python src/formation_from_image.py     # Shape from image via K_pos topology
python src/snake_maze_3d.py            # 3D maze with intro + autopilot

# Behavior reproduction
python behavior_reproduction/microrobot_collectives.py  # Reproduce Gardi et al. 2022

# Characterization sweeps
python characterization/sweep_2species.py                      # Video sweep (CONFIG-driven)
python characterization/sweep_metrics.py --sweep kpos_offdiag  # Headless metric sweep
python characterization/plot_heatmaps.py --sweep kpos_offdiag  # Generate heatmaps from CSV
python characterization/make_metric_slides.py                  # PowerPoint with descriptions


# Dependencies
pip install pygame numpy matplotlib pandas tqdm
pip install opencv-python  # For video recording
```

---

## Key Code Locations

| Feature | File | Lines |
|---------|------|-------|
| Force computation | `particle_life.py` | 449-542 |
| Boundary conditions | `particle_life.py` | 563-578 |
| Matrix editing UI | `particle_life.py` | 556-628 |
| Metric sweep (headless) | `characterization/sweep_metrics.py` | All |
| Video sweep | `characterization/sweep_2species.py` | All |
| Heatmap plotting | `characterization/plot_heatmaps.py` | All |

---

## Experimental Notes

*Space for recording observations, hypotheses, and findings:*

### Observations
- Asymmetric K_pos matrices tend to produce more dynamic (non-equilibrium) behaviors
- Strong diagonal values (same-species attraction) create tight clusters
- Orientation matrix effects are most visible when radial forces are balanced
- **Locomotion orthogonality**: Antisymmetric K_rot (tangential) and K_pos forward_bias (radial) create motion in perpendicular directions — this gives full 2D locomotion control for formations
- **Compression on curved chains**: Uniform antisymmetric K_rot on a C-shape creates net inward force because tangential directions diverge on curves. This is a fundamental tradeoff: local-only tangential drive cannot produce shape-preserving translation on curved formations without global coordination
- **Chain self-intersection**: With >10 species starting from random positions, chains almost always cross themselves. Non-adjacent species have K_pos=0, so there's no force to untangle crossings
- **PID jitter sources**: Integral term (ki) causes oscillation through windup; velocity-based heading measurement is noisy (use centroid displacement or chain axis orientation instead); dead zones on error help eliminate micro-corrections
- **Object transport physics**: Force-based pushing causes overlap→impulse artifacts. Hard-contact collision response (position correction + velocity transfer) is much smoother. Object receives 30% of contacting particle's approach velocity — moves at swarm speed, no bouncing
- **Caging transport**: C-shape can capture and transport objects by surrounding them. Tighter C (higher phi0) = more secure grip. The object's motion is an emergent result of collective contact forces — no explicit grasping mechanism needed
- **K_pos topology encoding**: Gaussian kernel on sampled point distances encodes local neighborhood structure. Species close on the target shape attract strongly; the equilibrium approximates the original shape without explicit coordinates
- **Sigma tradeoff**: Too small σ → disconnected clusters; too large → everything collapses to a blob. Auto-sigma from median nearest-neighbor distance × 1.5 works well for most shapes
- **Behavioral universality**: The dual-matrix model (K_pos + K_rot) can reproduce behavioral phenotypes from physically diverse systems (microrobot collectives, biological flocking, bacterial swarming). The same behavioral classes — aggregation, vortex, chain, pursuit, transport — emerge from universal interaction primitives (radial + tangential coupling) regardless of the underlying physical mechanism. See `claudedocs/behavioral_universality.md` for full analysis.
- **Non-uniform density in formation_from_image**: Even though Lloyd's relaxation produces uniformly sampled target points, the equilibrium particle distribution is non-uniform — species in sparse regions (e.g., star tips, thin protrusions) settle more loosely, while species in dense regions (e.g., shape center) pack more compactly. This is because central species have more neighbors (higher total K_pos attraction) pulling them inward, while peripheral species have fewer neighbors and weaker total attraction
- **Global-beta constraint on multi-orbit systems**: In the shared overdamped kernel, the stable limit cycle for any bound pair is at `r = beta × r_max`. Because `beta` is a global scalar, ALL bound pairs converge to the same radius — multi-planet systems with differentiated orbital radii cannot be reproduced by matrix design alone. `planet.py` resolves this via a scoped override that treats `beta[i,j]` as a per-pair matrix (see `behavior_reproduction/PLAN.md`). This is the ONLY file with a physics exception; the shared engine remains scalar-beta.
- **Overdamped orbit ≠ Newtonian orbit**: Because `v = F` (no inertia), there is no angular momentum and no centripetal equilibrium. Radial force is zero only at `r_norm = beta` (stable) or `r_norm = 1` (unstable). K_rot tangential forcing produces tangential-only motion at `r_norm = beta` → the orbit is a *limit cycle*, not a Keplerian conic. K_rot controls speed/eccentricity; beta × r_max sets mean radius.
- **Colinear-orbit degeneracy in hierarchical systems**: If Sun, Earth, Moon are initialized colinear (all on x-axis), then Earth's tangential direction from Sun and Moon's tangential direction from Earth both point +y on step 1. The Moon translates upward with Earth instead of orbiting it. Offsetting the Moon perpendicular to the Sun-Earth axis at t=0 breaks the degeneracy and makes the nested orbit visible immediately.

### Hypotheses to Test
- [ ] Is there a critical K_pos threshold for chase vs encapsulate transition?
- [ ] Do 3-species systems show qualitatively different phase diagrams than 2-species?
- [ ] How does orientation coupling modify stability boundaries?
- [ ] Can compression on curved chains be quantified as a function of curvature × move_speed?
- [ ] Does K_pos cross-attraction strength have a critical threshold for resisting locomotion compression?
- [ ] Can non-adjacent weak repulsion untangle chain crossings without destabilizing the formation?

### TODO
- [ ] Run comprehensive sweep with finer grid resolution
- [ ] Implement additional metrics (polarization, angular momentum)
- [ ] Compare simulation results with physical robot experiments
- [ ] Explore higher species counts (N > 3)
- [ ] Quantify shape fidelity vs locomotion speed tradeoff curve
- [ ] Test formation locomotion with drawn shapes (F key) under various speeds
- [ ] Measure transport efficiency: object delivery time vs species count, phi0, move_speed
- [ ] Test robustness: can the C-shape recover if object escapes during transport?
- [ ] Multi-object transport: can multiple C-shapes coordinate to move multiple objects?
