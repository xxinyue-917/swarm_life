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

### 3. Other Systems (future)
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

## Status

- [x] Plan created
- [x] Microrobot rotation modes (3 presets)
- [x] Microrobot chain mode (tridiagonal K_pos)
- [x] Microrobot oscillation (sinusoidal K_rot)
- [x] Flocking — separation, aggregation, cohesion, full Reynolds presets
- [x] Galaxy — elliptical, rotating disk, ring galaxy, merger, differential rotation
- [ ] Microrobot — tune presets to quantitatively match paper figures
- [ ] Microrobot — reproduce dispersion and reconfiguration modes
- [ ] Flocking — tune for more realistic boids-like motion. Current issue: particles aggregate but don't show convincing flock-like collective motion (moving together in one direction). The overdamped velocity model makes sustained directional movement hard — particles settle at equilibrium rather than flying. Need to explore stronger forward_bias, higher max_speed, or different matrix structures to get persistent group motion with toroidal wrapping.
- [ ] Quantitative comparison with published results
- [ ] Slides with side-by-side comparison

## Related Documents

- `claudedocs/behavioral_universality.md` — Full theoretical argument and universality analysis
- `characterization/PLAN.md` — 2-species and 3-species parameter sweeps that identify behavioral regions
- `CLAUDE.md` — Codebase architecture and research context
