"""Unified particle life simulation with 3-piece radial kernel."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import SimConfig


# ============================================================================
# Radial Kernel (3-piece smooth interaction force)
# ============================================================================

@dataclass
class KernelParams:
    """Parameters for the 3-piece smooth radial kernel."""
    r_rep: float = 4.0       # repulsion radius
    r_att: float = 24.0      # attraction radius
    r_cut: float = 36.0      # cutoff radius
    a_rep: float = 5.0       # repulsion strength
    a_att: float = 2.0       # attraction strength


def radial_kernel(r: float, p: KernelParams) -> float:
    """
    Return scalar gain g(r); + means repulsive (push away), - means attractive (pull).

    The kernel has three pieces:
    - r < r_rep: strong linear repulsion
    - r_rep <= r < r_att: cosine-windowed attraction
    - r_att <= r < r_cut: decaying attraction
    - r >= r_cut: zero force

    Args:
        r: Distance between particles
        p: Kernel parameters

    Returns:
        Force magnitude (positive = repulsive, negative = attractive)
    """
    if r <= 0.0:
        return p.a_rep
    if r < p.r_rep:
        # Near: strong repulsion (linear hat)
        return p.a_rep * (1.0 - r / p.r_rep)
    if r < p.r_att:
        # Mid: attraction (cosine window)
        x = (r - p.r_rep) / (p.r_att - p.r_rep)
        return -p.a_att * 0.5 * (1.0 + math.cos(math.pi * x))
    if r < p.r_cut:
        # Far: decaying attraction
        x = (r - p.r_att) / (p.r_cut - p.r_att)
        return -p.a_att * 0.2 * (1.0 - x)
    return 0.0


class Simulation:
    """
    Particle life simulation with smooth 3-piece radial kernel.

    Supports N species with configurable interaction matrix.
    """

    def __init__(
        self,
        config: SimConfig,
        matrix: Optional[np.ndarray] = None,
        kernel_params: Optional[KernelParams] = None
    ):
        self.config = config

        # Interaction matrix K[i][j] = effect of species j on species i
        if matrix is not None:
            self.matrix = np.array(matrix, dtype=float)
        else:
            self.matrix = self._default_matrix()

        # Validate matrix
        if self.matrix.shape != (config.n_species, config.n_species):
            raise ValueError(
                f"Matrix shape {self.matrix.shape} doesn't match "
                f"species count {config.n_species}"
            )

        # Kernel parameters
        self.kernel_params = kernel_params or KernelParams()

        # Initialize RNG
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

        # Agent state
        self.n_agents = config.n_particles
        self.species = self._init_species()
        self.positions = self._init_positions()
        self.velocities = np.random.uniform(-20.0, 20.0, (self.n_agents, 2))

        # Time
        self.t = 0.0

    def _default_matrix(self) -> np.ndarray:
        """Generate default interaction matrix."""
        n = self.config.n_species
        # Simple pattern: repel same species, attract different
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 0.5  # Mild repulsion within species
                else:
                    matrix[i, j] = -0.8  # Attraction to other species
        return matrix

    def _init_species(self) -> np.ndarray:
        """Assign species to agents uniformly."""
        species = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            species[i] = i % self.config.n_species
        return species

    def _init_positions(self) -> np.ndarray:
        """Initialize positions randomly within world."""
        return np.random.uniform(
            [0, 0],
            [self.config.width, self.config.height],
            (self.n_agents, 2)
        )

    def step(self) -> None:
        """Advance simulation by one timestep."""
        dt = self.config.dt

        # Compute forces for all agents
        forces = self._compute_forces()

        # Update velocities
        self.velocities += dt * forces
        self.velocities *= self.config.damping

        # Enforce speed limit
        speeds = np.linalg.norm(self.velocities, axis=1)
        over_limit = speeds > self.config.max_speed
        if np.any(over_limit):
            scale = self.config.max_speed / speeds[over_limit]
            self.velocities[over_limit] *= scale[:, np.newaxis]

        # Update positions
        self.positions += dt * self.velocities

        # Apply boundary conditions
        self._apply_boundaries()

        # Increment time
        self.t += dt

    def _compute_forces(self) -> np.ndarray:
        """Compute interaction forces for all agents."""
        forces = np.zeros((self.n_agents, 2))

        # Build spatial grid for efficient neighbor search
        cell_size = self.kernel_params.r_cut
        grid = self._build_grid(cell_size)

        # Compute pairwise forces
        for i in range(self.n_agents):
            cell = self._get_cell(self.positions[i], cell_size)

            # Check neighboring cells
            for neighbor_cell in self._neighbor_cells(cell):
                if neighbor_cell not in grid:
                    continue

                for j in grid[neighbor_cell]:
                    if i == j:
                        continue

                    # Compute relative position
                    dx = self.positions[j] - self.positions[i]
                    r = np.linalg.norm(dx)

                    # Skip if beyond cutoff
                    if r >= self.kernel_params.r_cut or r < 1e-6:
                        continue

                    # Compute force
                    direction = dx / r
                    s_i = self.species[i]
                    s_j = self.species[j]

                    # Interaction strength from matrix
                    k_ij = self.matrix[s_i, s_j]

                    # Radial kernel (positive = repulsion, negative = attraction)
                    g_r = radial_kernel(r, self.kernel_params)

                    # Total force: repulsion always active, attraction modulated by matrix
                    if g_r > 0:  # Repulsive part
                        f_mag = g_r
                    else:  # Attractive part
                        f_mag = k_ij * abs(g_r)  # k_ij modulates attraction strength

                    # Accumulate force
                    forces[i] += f_mag * direction

        return forces

    def _build_grid(self, cell_size: float) -> Dict[Tuple[int, int], List[int]]:
        """Build spatial grid for neighbor search."""
        grid: Dict[Tuple[int, int], List[int]] = {}
        for i in range(self.n_agents):
            cell = self._get_cell(self.positions[i], cell_size)
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(i)
        return grid

    def _get_cell(self, pos: np.ndarray, cell_size: float) -> Tuple[int, int]:
        """Get grid cell for position."""
        return (int(pos[0] // cell_size), int(pos[1] // cell_size))

    def _neighbor_cells(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get neighboring cells (including self)."""
        cx, cy = cell
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbors.append((cx + dx, cy + dy))
        return neighbors

    def _apply_boundaries(self) -> None:
        """Apply reflective boundary conditions."""
        # X boundaries
        left = self.positions[:, 0] < 0
        self.positions[left, 0] = -self.positions[left, 0]
        self.velocities[left, 0] = np.abs(self.velocities[left, 0])

        right = self.positions[:, 0] > self.config.width
        self.positions[right, 0] = 2 * self.config.width - self.positions[right, 0]
        self.velocities[right, 0] = -np.abs(self.velocities[right, 0])

        # Y boundaries
        bottom = self.positions[:, 1] < 0
        self.positions[bottom, 1] = -self.positions[bottom, 1]
        self.velocities[bottom, 1] = np.abs(self.velocities[bottom, 1])

        top = self.positions[:, 1] > self.config.height
        self.positions[top, 1] = 2 * self.config.height - self.positions[top, 1]
        self.velocities[top, 1] = -np.abs(self.velocities[top, 1])

    def get_state(self) -> dict:
        """Get current state for API/visualization."""
        return {
            "width": self.config.width,
            "height": self.config.height,
            "t": self.t,
            "n_species": self.config.n_species,
            "particles": [
                {
                    "id": i,
                    "species": int(self.species[i]),
                    "x": float(self.positions[i, 0]),
                    "y": float(self.positions[i, 1]),
                }
                for i in range(self.n_agents)
            ],
        }

    def set_matrix(self, matrix: np.ndarray) -> None:
        """Update interaction matrix."""
        matrix = np.array(matrix, dtype=float)
        if matrix.shape != (self.config.n_species, self.config.n_species):
            raise ValueError(
                f"Matrix shape {matrix.shape} doesn't match "
                f"species count {self.config.n_species}"
            )
        self.matrix = matrix

    def reset(self) -> None:
        """Reset simulation to initial state."""
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        self.species = self._init_species()
        self.positions = self._init_positions()
        self.velocities = np.random.uniform(-20.0, 20.0, (self.n_agents, 2))
        self.t = 0.0
