# Behavior Reproduction — Reproducing Collective Behaviors from Other Systems

## Goal

Demonstrate that the dual-matrix particle life model (K_pos + K_rot) can reproduce behavioral phenotypes from physically diverse systems, showing the **universality of the interaction primitives** (radial attraction/repulsion + tangential coupling).

## Rationale

The central argument of the research paper is that collective behaviors from physically diverse systems can be reproduced by a single minimal model parameterized by two interaction matrices. This folder implements reproductions that support that claim.

The goal is **not** to claim the particle life model IS any of the target systems — the physics are completely different (magnetic fields, hydrodynamics, vision-based alignment). The goal is to show that the same **behavioral phenotypes** emerge from a minimal dual-matrix model, suggesting that these collective behaviors arise from universal interaction primitives regardless of the underlying physical mechanism.

This has direct precedents:
- Reynolds' boids (3 rules → flocking)
- Vicsek model (1 rule → order-disorder phase transition)
- Active Brownian particles (bacterial colony dynamics)

All showed that complex emergent behaviors reduce to minimal local rules. Our work extends this by showing a **single parameterized framework** captures behaviors from multiple source systems.

## Target Systems

### 1. Microrobot Collectives
**Reference**: Gardi, Ceron, Wang, Petersen, Sitti — *Microrobot collectives with reconfigurable morphologies, behaviors, and functions* — Nature Communications 2022

**Why this paper?** It explicitly documents 4+ emergent modes (rotation, oscillation, chain, self-propelling pairs) with clear parameter-behavior mappings. The paper uses magnetic field frequencies as the control parameter; we use K_pos/K_rot. If we can reproduce the same behaviors with a simpler model, we demonstrate that the specific physical mechanism is not essential.

**Behaviors to reproduce**:
- [x] Rotation (tight / medium / wide) — symmetric K_rot + attractive K_pos, varied strength
- [x] Chain mode (connected / separated) — tridiagonal K_pos with many species
- [x] Oscillation — time-varying K_rot (sinusoidal sign flip)
- [ ] Dispersion / gas-like — negative K_pos cross-coupling
- [ ] Reconfigurable morphology switching — runtime parameter changes
- [ ] Collective transport (via formation_transport demo)

### 2. Flocking (Reynolds / Vicsek)
**References**: Reynolds (1987) "Flocks, herds and schools"; Vicsek et al. (1995) "Novel type of phase transition in a system of self-driven particles"

**Mapping**:
| Flocking rule | Particle life equivalent |
|---|---|
| Cohesion (move toward neighbors' center) | Positive K_pos |
| Separation (avoid crowding) | Zone 1 short-range repulsion (built into kernel) |
| Alignment (match neighbor velocities) | K_rot tangential coupling aligns velocities |

**Behaviors to reproduce**:
- [ ] Ordered flocking (aligned collective motion)
- [ ] Disordered swarming (no alignment)
- [ ] Order-disorder phase transition (continuous change in order parameter)
- [ ] Milling / rotating flock

### 3. Galaxy Morphology
**References**: Hubble classification (elliptical, lenticular, ring, spiral); Cartwheel galaxy; Antennae collision

**Behaviors to reproduce**:
- [x] Elliptical — single-species self-attracting blob
- [x] Rotating disk (S0) — bulge + disk with antisymmetric K_rot
- [x] Ring galaxy (Cartwheel) — intruder species repels disk outward
- [x] Galaxy merger — two rotating clusters with weak cross-attraction
- [x] Differential rotation — inner-disk K_rot > outer-disk K_rot

### 4. Planetary Systems
**References**: Solar system hierarchy; Sun-Earth-Moon three-body; Saturn + rings

**Why this is special in the framework**: multi-planet orbital systems with *differentiated radii* are the one morphological class that cannot be reproduced by matrix design alone under the shared physics engine. The limit-cycle radius for any bound pair is `r_eq ≈ beta × r_max`, and `beta` is a global scalar — so Mercury, Venus, Earth, Mars would all collapse to the same orbital radius. This was established by a multi-agent debate (see Debate Notes below).

**Scoped physics exception**: `planet.py` overrides `compute_velocities` locally to turn `beta` into a per-pair matrix `beta[i,j]`. `src/particle_life.py` and every other demo remain unchanged. With per-pair beta, each species pair has its own limit-cycle radius.

**Behaviors reproduced**:
- [x] Sun-Earth (2-species) — single orbital pair
- [x] Sun-Earth-Moon (3-species) — hierarchical orbit, Moon tracks Earth's moving position
- [x] Solar system (5-species) — 4 planets with distinct radii via staggered beta
- [x] Solar + Moon (6-species) — full hierarchy with planet-moon coupling
- [x] Saturn's rings (3-species) — self-repulsion in ring species produces radial spread

### 5. Other Systems (future)
- [ ] Bacterial colony expansion (active Brownian particles)
- [ ] Fish schooling (similar to flocking but with visual field constraints)
- [ ] Ant foraging trails (stigmergic communication proxy)

## Approach

For each target behavior:
1. **Identify qualitative features** from the source paper (what does it look like?)
2. **Map features to K_pos / K_rot configurations** based on behavioral universality principles
3. **Implement a preset** that reproduces the behavior
4. **Validate visually** against the source paper's figures
5. **Quantitative comparison** — measure order parameters (polarization, angular momentum, cluster count) and compare with published values
6. **Record video** for side-by-side comparison

## File Structure

```
behavior_reproduction/
├── PLAN.md                    # This document
├── microrobot_collectives.py  # Microrobot behavior presets (rotation, chain, oscillation)
├── flocking.py                # Flocking presets (separation, aggregation, cohesion)
├── galaxy.py                  # Galaxy morphology (elliptical, disk, ring, merger, differential)
├── planet.py                  # Planetary systems — per-pair beta (scoped physics override)
├── research_papers/           # Reference PDFs (gitignored)
├── results/                   # Videos and screenshots (gitignored)
└── presets/                   # Discovered parameter configurations (JSON)
```

## Implementation Notes

### Microrobot Rotation Mode
- Symmetric K_rot[A,A] (or K_rot[A,B] = K_rot[B,A]) creates opposing tangential forces between particles, which causes orbital rotation
- Combined with attractive K_pos, particles stay bound while rotating
- Tuning: higher K_rot → wider orbit (matches paper's frequency-dependent spread)

### Microrobot Chain Mode
- Particle life is isotropic — a single species can only form clusters, not chains
- To get chains, we break isotropy via species topology: many species with tridiagonal K_pos
- Each species only attracts its immediate neighbors (i-1 and i+1)
- Forms a single linear chain because of the topological ordering

### Microrobot Oscillation Mode
- Time-varying K_rot: K_rot(t) = base × sin(2π t / period)
- The sign flip causes the collective to alternate rotation direction
- Matches the paper's "oscillating about center of mass" behavior

### Flocking
- Cohesion and separation are easy (positive K_pos + Zone 1 repulsion)
- Alignment is harder: particle life's K_rot creates rotation, not direct velocity matching
- Need to explore whether symmetric K_rot coupling produces the same order-disorder phase transition as Vicsek

### Galaxy Implementation Notes

5 presets reproducing morphological analogs of galaxy phenomena:

| Preset | Species | Mechanism | Galaxy analog |
|--------|---------|-----------|---------------|
| Elliptical | 1 (200 particles) | Self-attraction blob, no rotation | E0-E7 elliptical galaxies |
| Rotating Disk | 2 (bulge + disk) | Antisymmetric K_rot → disk orbits bulge | Lenticular (S0) galaxy |
| Ring Galaxy | 3 (bulge + disk + intruder) | Intruder repels disk outward → expanding ring | Cartwheel galaxy |
| Merger | 4 (2 per galaxy) | Two rotating clusters with weak cross-attraction merge | Antennae-like collision |
| Differential Rotation | 3 (bulge + inner + outer) | K_rot[0,1]=0.7 vs K_rot[0,2]=0.3 → inner orbits faster | Spiral galaxy rotation curve |

**Honest framing**: These are topological/morphological analogs, not gravitational simulations. The model lacks inertia, angular momentum conservation, and 1/r² gravity. True spiral arms and tidal tails are not achievable. The value is showing that visually complex rotational structures emerge from simple species-species coupling matrices.

**Critical rule**: compute_velocities() and step physics remain identical to particle_life.py. All behaviors come from matrix design only.

### Flocking Implementation Notes
- 10 species, 15 particles each (150 total), toroidal wrapping
- Chain-structured K_pos with forward_bias creates continuous forward motion
- Antisymmetric K_rot between adjacent species creates perpendicular translation
- Both mechanisms are identical to the formation_locomotion demo — no physics modifications
- **Critical rule**: compute_velocities() and step physics must remain identical to particle_life.py. All behaviors come from matrix design only.

### Planetary System Implementation Notes

#### Why per-pair beta is necessary (debate rationale)

A 3-round / 5-agent debate (Systems Thinker, Pragmatist, Edge-Case Finder, Domain Expert, Contrarian) established the following about multi-planet orbits under the shared physics engine:

1. **Overdamped dynamics has no true orbits.** Velocity equals force (no inertia, no conserved angular momentum). There is no Newtonian centripetal equilibrium.
2. **Limit cycle radius is fixed by beta.** For any bound pair (K_pos > 0), the radial force F_r = 0 at exactly two normalized radii: `r_norm = beta` (stable) and `r_norm = 1` (unstable). With nonzero K_rot, the particle settles into a tangential-only limit cycle at `r_norm = beta`, i.e., physical radius `r_eq ≈ beta × r_max`.
3. **K_rot controls orbital speed, not radius.** Stronger swirl makes the orbit faster and more eccentric, but the limit cycle still sits at `r ≈ beta × r_max`.
4. **With a global scalar beta, all planets converge to the same radius.** Mercury, Venus, Earth, Mars cannot have distinct stable orbits — only distinct transients before collapse.
5. **Conclusion**: the ONLY way to get differentiated stable orbital radii is per-pair beta.

#### Scoped physics override

`planet.py` defines a local JIT kernel `_compute_velocities_beta_matrix_jit` that mirrors `src/particle_life.py::_compute_velocities_jit` exactly except that it reads `beta[si, sj]` from a matrix instead of a scalar. `PlanetDemo.compute_velocities()` overrides the parent method to call this local kernel with `self.beta_matrix`. Nothing else in the shared engine changes.

The override is recorded as an explicit exception in the user's memory (`feedback_no_physics_change.md`) — this is the ONLY file with a physics exception; every other demo still uses the shared scalar-beta engine.

#### Config choices

| Parameter | Value | Reason |
|---|---|---|
| `r_max` | 8.0 | Large enough that `beta ∈ [0.1, 0.6]` gives orbital radii `r_eq ∈ [0.8, 4.8]` with headroom |
| `beta` (scalar fallback) | 0.2 | Used only when a preset omits the `beta` matrix |
| `a_rot` | 0.5 | Tangential strength — reduced from default for calmer orbits |
| `far_attraction` | 0.05 | Weak zone-4 recovery so drifting planets don't escape |
| `force_scale` | 0.5 | Lowers overall force magnitude for stable integration at `dt=0.05` |

#### Preset design

| Preset | Species | Key parameters | What it shows |
|---|---|---|---|
| `1_sun_earth` | 2 | beta[Sun,Earth]=0.38 → r≈3.0 | Simplest orbital pair |
| `2_sun_earth_moon` | 3 | beta[Sun,Earth]=0.45 (r≈3.6); beta[Earth,Moon]=0.15 (r≈1.2) | Hierarchical orbit — K_pos[Moon,Sun]=0 so the Moon only couples to Earth |
| `3_solar_system` | 5 | beta[Sun,i] = 0.19 / 0.31 / 0.44 / 0.56 → r = 1.5 / 2.5 / 3.5 / 4.5 | 4 planets at distinct radii via staggered beta |
| `4_solar_moon` | 6 | Above + beta[Earth,Moon]=0.10 | Full hierarchy with planet-moon coupling |
| `5_saturn_rings` | 3 | beta[Saturn,Ring]=0.44 (r≈3.5); K_pos[Ring,Ring]=−0.05 | Ring species uses self-repulsion to spread radially |

#### Key design details

- **Intra-species cohesion**: diagonal `K_pos = 0.9–1.0` for all bodies (Sun 1.0, planets/moons 0.9) keeps each body a tight cluster. Diagonal `beta = 0.10` gives intra-body contact radius ≈ 0.8.
- **Hierarchical orbit (Sun-Earth-Moon)**: Moon has `K_pos[Moon, Sun] = 0` (ignores Sun directly) but `K_pos[Moon, Earth] > 0`. In overdamped dynamics this automatically makes the Moon track Earth's moving position — no inertia needed.
- **Moon orbit visibility**: Moon's initial position must be *perpendicular* to the Sun-Earth axis. If Sun, Earth, Moon are colinear at t=0, then Earth's tangential direction from Sun and Moon's tangential direction from Earth both point the same way — the Moon visually translates with Earth instead of orbiting it. Preset 2 places Moon at `(Earth_x, Earth_y + 1.2)` to break this degeneracy.
- **Moon orbital speed**: `K_rot[Moon, Earth] = 0.9` (vs `K_rot[Earth, Sun] = 0.3`). Moon angular velocity around Earth is ~6–20× Earth's around Sun, so the nested orbit is unambiguous over a few seconds of simulation.
- **Staggered orbital speeds (Preset 3/4)**: `K_rot[Sun, planet_i]` decreases with radius (0.4 Mercury → 0.15 Mars) — inner planets orbit faster, matching real solar-system rotation curves qualitatively.

#### UI controls

Planet-tuning keys are gated on `not self.show_matrix` so the matrix editor (M key) works correctly:

| Key | Effect |
|---|---|
| `←/→` | Select planet (skip Sun; only off-Sun entries are tunable) |
| `↑/↓` | Adjust `beta[selected, Sun]` by ±0.02 (kept symmetric) — changes orbital *radius* |
| `W/S` | Adjust `K_rot[selected, Sun]` by ±0.02 — changes orbital *speed* |
| `M` | Toggle full matrix editor (WASD then navigates cells; E/X adjusts values) |
| `1–5` | Load preset |
| `R` | Reset positions |

When matrix editor is open, none of ←/→/↑/↓/W/S intercept — they all fall through to parent `ParticleLife.handle_events` for cell navigation.

#### Honest framing

This is a *morphological analog*, not gravitational simulation. The kernel has no inertia, no angular momentum conservation, no 1/r² law. "Orbits" are limit cycles of an overdamped force balance. Kepler's laws, elliptical precession, and tidal effects do not apply. The value is showing that visually hierarchical orbital structures (planet-moon-sun) emerge from simple per-pair coupling matrices — not that the simulation is physically accurate.

## Status

- [x] Plan created
- [x] Microrobot rotation modes (3 presets)
- [x] Microrobot chain mode (tridiagonal K_pos)
- [x] Microrobot oscillation (sinusoidal K_rot)
- [x] Flocking — separation, aggregation, cohesion, full Reynolds presets
- [x] Galaxy — elliptical, rotating disk, ring galaxy, merger, differential rotation
- [x] Planetary systems — Sun-Earth, Sun-Earth-Moon, Solar System, Solar+Moon, Saturn's rings (via per-pair beta)
- [ ] Microrobot — tune presets to quantitatively match paper figures
- [ ] Microrobot — reproduce dispersion and reconfiguration modes
- [ ] Flocking — tune for more realistic boids-like motion. Current issue: particles aggregate but don't show convincing flock-like collective motion (moving together in one direction). The overdamped velocity model makes sustained directional movement hard — particles settle at equilibrium rather than flying. Need to explore stronger forward_bias, higher max_speed, or different matrix structures to get persistent group motion with toroidal wrapping.
- [ ] Quantitative comparison with published results
- [ ] Slides with side-by-side comparison

## Related Documents

- `claudedocs/behavioral_universality.md` — Full theoretical argument and universality analysis
- `characterization/PLAN.md` — 2-species and 3-species parameter sweeps that identify behavioral regions
- `CLAUDE.md` — Codebase architecture and research context
