# Behavior Reproduction — Reproducing Collective Behaviors from Other Systems

## Goal

Demonstrate that the dual-matrix particle life model (K_pos + K_rot) can reproduce behavioral phenotypes from physically diverse systems, showing the universality of the interaction primitives.

## Target Systems

### 1. Microrobot Collectives
**Reference**: "Microrobot collectives with reconfigurable morphologies, behaviors, and functions" (Nature)

Behaviors to reproduce:
- [ ] Aggregation / clustering
- [ ] Chain / line formation
- [ ] Vortex / milling
- [ ] Dispersion
- [ ] Reconfigurable morphology switching
- [ ] Collective transport

### 2. Flocking (Reynolds / Vicsek)
**References**: Reynolds (1987), Vicsek et al. (1995)

Behaviors to reproduce:
- [ ] Ordered flocking (aligned collective motion)
- [ ] Disordered swarming
- [ ] Order-disorder phase transition
- [ ] Milling / rotating flock

### 3. Other Systems (future)
- [ ] Bacterial colony expansion
- [ ] Fish schooling
- [ ] Ant foraging trails

## Approach

For each target behavior:
1. Identify the key qualitative features (what does it look like?)
2. Map those features to K_pos / K_rot configurations
3. Implement a preset that reproduces the behavior
4. Validate visually and with metrics (polarization, angular momentum, etc.)
5. Record video for comparison with the original system

## File Structure

```
behavior_reproduction/
├── PLAN.md                    # This document
├── microrobot_collectives.py  # Microrobot behavior presets + sweep
├── flocking.py                # Flocking behavior presets + sweep
├── results/                   # Videos and screenshots (gitignored)
└── presets/                   # Discovered parameter configurations (JSON)
```

## Status

- [x] Plan created
- [x] Microrobot collectives — initial implementation with 6 presets
- [ ] Microrobot collectives — tune presets to match paper figures
- [ ] Flocking implementation
- [ ] Comparison analysis
