"""Simple configuration for particle life simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimConfig:
    """Configuration for particle life simulation."""

    # World
    width: float = 800.0
    height: float = 600.0

    # Particles
    n_species: int = 3
    n_particles: int = 450

    # Dynamics
    dt: float = 0.05
    damping: float = 0.995  # Much less damping!
    max_speed: float = 300.0

    # Simulation
    frame_interval: float = 0.03  # Time between WebSocket frames
    seed: Optional[int] = None


DEFAULT_CONFIG = SimConfig()
