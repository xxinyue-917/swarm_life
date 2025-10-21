"""Preset scenarios with different species counts and interaction matrices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .simulation import KernelParams


@dataclass
class Preset:
    """A preset scenario with species count and interaction matrix."""
    name: str
    description: str
    n_species: int
    matrix: np.ndarray  # Shape (n_species, n_species)
    kernel_params: KernelParams = None  # Optional custom kernel params

    def __post_init__(self) -> None:
        if self.kernel_params is None:
            self.kernel_params = KernelParams()


# ============================================================================
# Preset Definitions
# ============================================================================

# 2-species: Guards and Workers (Containment)
GUARDS_WORKERS = Preset(
    name="guards_workers",
    description="2-species containment: Guards form ring, Workers cluster inside",
    n_species=2,
    matrix=np.array([
        [+0.9, +0.6],   # Guards: repel Guards, repel Workers
        [-0.3, +0.2],   # Workers: attract to Guards, repel Workers
    ]),
)

# 3-species: Cyclic interactions (Rock-Paper-Scissors)
CYCLIC = Preset(
    name="cyclic",
    description="3-species cyclic chase (rock-paper-scissors dynamics)",
    n_species=3,
    matrix=np.array([
        [-0.2,  +0.9, -0.4],
        [-0.4, -0.2,  +0.9],
        [+0.9, -0.4, -0.2],
    ]),
)

# 3-species: Flocking behavior
FLOCKING = Preset(
    name="flocking",
    description="3-species flocking with mild inter-species repulsion",
    n_species=3,
    matrix=np.array([
        [-0.6, -0.3, -0.3],
        [-0.3, -0.6, -0.3],
        [-0.3, -0.3, -0.6],
    ]),
)

# 4-species: Complex ecosystem
ECOSYSTEM = Preset(
    name="ecosystem",
    description="4-species complex ecosystem with mixed interactions",
    n_species=4,
    matrix=np.array([
        [+0.2, -0.8, +0.4, -0.3],
        [+0.6, +0.3, -0.7, +0.2],
        [-0.4, +0.5, +0.1, -0.6],
        [+0.3, -0.5, +0.7, +0.2],
    ]),
)

# 5-species: Chaos
CHAOS = Preset(
    name="chaos",
    description="5-species high-energy chaotic interactions",
    n_species=5,
    matrix=np.array([
        [+0.5, -0.9, +0.7, -0.4, +0.6],
        [+0.8, +0.4, -0.6, +0.5, -0.7],
        [-0.7, +0.6, +0.3, -0.8, +0.4],
        [+0.4, -0.5, +0.9, +0.2, -0.6],
        [-0.6, +0.7, -0.4, +0.8, +0.3],
    ]),
)


# ============================================================================
# Preset Registry
# ============================================================================

PRESETS: Dict[str, Preset] = {
    "guards_workers": GUARDS_WORKERS,
    "cyclic": CYCLIC,
    "flocking": FLOCKING,
    "ecosystem": ECOSYSTEM,
    "chaos": CHAOS,
}


def list_presets() -> List[Preset]:
    """Return list of all available presets."""
    return list(PRESETS.values())


def get_preset(name: str) -> Preset:
    """Get preset by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
