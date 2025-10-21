#!/usr/bin/env python3
"""Direct test of simulation to diagnose particle movement."""

import numpy as np
from backend.simulation import Simulation
from backend.config import SimConfig

# Create config with aggressive settings
config = SimConfig(
    n_species=5,
    n_particles=100,  # Fewer particles for easier debugging
    dt=0.05,
    damping=0.995,  # Very little damping
    max_speed=300.0,
    seed=42
)

# Create simulation
sim = Simulation(config)

# Apply chaos matrix
chaos_matrix = np.array([
    [+0.5, -0.9, +0.7, -0.4, +0.6],
    [+0.8, +0.4, -0.6, +0.5, -0.7],
    [-0.7, +0.6, +0.3, -0.8, +0.4],
    [+0.4, -0.5, +0.9, +0.2, -0.6],
    [-0.6, +0.7, -0.4, +0.8, +0.3],
])
sim.set_matrix(chaos_matrix)

print("Initial state:")
print(f"  Position 0: {sim.positions[0]}")
print(f"  Velocity 0: {sim.velocities[0]}")
print(f"  Species 0: {sim.species[0]}")
print()

# Run simulation for 100 steps
for i in range(100):
    sim.step()
    if i % 10 == 9:
        print(f"Step {i+1}: pos={sim.positions[0]}, vel={sim.velocities[0]}")

print(f"\nFinal state after 100 steps:")
print(f"  Position 0: {sim.positions[0]}")
print(f"  Velocity 0: {sim.velocities[0]}")

# Check average velocity magnitude
vel_mags = np.linalg.norm(sim.velocities, axis=1)
print(f"\nAverage velocity magnitude: {vel_mags.mean():.2f}")
print(f"Max velocity magnitude: {vel_mags.max():.2f}")
print(f"Min velocity magnitude: {vel_mags.min():.2f}")