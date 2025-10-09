from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .config import SimulationConfig


@dataclass
class Particle:
    species: int
    x: float
    y: float
    vx: float
    vy: float


class Simulation:
    """Particle Life style simulation with a uniform grid neighbour search."""

    def __init__(self, config: SimulationConfig, matrix: Optional[List[List[float]]] = None):
        self.config = config
        if matrix is not None:
            self._validate_matrix(matrix, config.species_count)
            self.matrix = [row[:] for row in matrix]
        else:
            self.matrix = config.base_matrix()
        self.cell_size = config.interaction_radius
        self._radius_sq = config.interaction_radius * config.interaction_radius
        if config.seed is not None:
            random.seed(config.seed)
        self.particles = self._spawn_particles()

    def _spawn_particles(self) -> List[Particle]:
        def velocity() -> float:
            return random.uniform(-20.0, 20.0)

        particles = []
        for idx in range(self.config.particle_count):
            species = idx % self.config.species_count
            particles.append(
                Particle(
                    species=species,
                    x=random.uniform(0.0, self.config.width),
                    y=random.uniform(0.0, self.config.height),
                    vx=velocity(),
                    vy=velocity(),
                )
            )
        return particles

    def set_matrix(self, matrix: List[List[float]]) -> None:
        self._validate_matrix(matrix)
        self.matrix = [row[:] for row in matrix]

    def reset(self, config: SimulationConfig, matrix: Optional[List[List[float]]] = None) -> None:
        self.config = config
        if matrix is not None:
            self._validate_matrix(matrix, config.species_count)
            self.matrix = [row[:] for row in matrix]
        else:
            self.matrix = config.base_matrix()
        self.cell_size = config.interaction_radius
        self._radius_sq = config.interaction_radius * config.interaction_radius
        if config.seed is not None:
            random.seed(config.seed)
        self.particles = self._spawn_particles()

    def get_state(self) -> dict:
        return {
            "width": self.config.width,
            "height": self.config.height,
            "particles": [
                {
                    "id": index,
                    "species": particle.species,
                    "x": particle.x,
                    "y": particle.y,
                }
                for index, particle in enumerate(self.particles)
            ],
        }

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            grid = self._build_grid()
            for index, particle in enumerate(self.particles):
                ax, ay = self._compute_acceleration(index, particle, grid)
                self._integrate(particle, ax, ay)

    def _build_grid(self) -> Dict[Tuple[int, int], List[int]]:
        grid: Dict[Tuple[int, int], List[int]] = {}
        for index, particle in enumerate(self.particles):
            key = self._cell_index(particle.x, particle.y)
            grid.setdefault(key, []).append(index)
        return grid

    def _cell_index(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))

    def _neighbor_cells(self, cell: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
        cx, cy = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield (cx + dx, cy + dy)

    def _compute_acceleration(
        self, index: int, particle: Particle, grid: Dict[Tuple[int, int], List[int]]
    ) -> Tuple[float, float]:
        ax = 0.0
        ay = 0.0
        for cell in self._neighbor_cells(self._cell_index(particle.x, particle.y)):
            for neighbor_index in grid.get(cell, []):
                if neighbor_index == index:
                    continue
                neighbor = self.particles[neighbor_index]
                dx = neighbor.x - particle.x
                dy = neighbor.y - particle.y
                distance_sq = dx * dx + dy * dy
                if distance_sq == 0.0 or distance_sq > self._radius_sq:
                    continue
                distance = math.sqrt(distance_sq)
                influence = self.matrix[particle.species][neighbor.species]
                falloff = 1.0 - (distance / self.config.interaction_radius)
                scale = influence * falloff
                ax += scale * (dx / distance)
                ay += scale * (dy / distance)
        limit = self.config.acceleration_limit
        magnitude = math.sqrt(ax * ax + ay * ay)
        if magnitude > limit:
            factor = limit / magnitude
            ax *= factor
            ay *= factor
        return ax, ay

    def _integrate(self, particle: Particle, ax: float, ay: float) -> None:
        dt = self.config.time_step
        particle.vx = (particle.vx + dt * ax) * self.config.velocity_decay
        particle.vy = (particle.vy + dt * ay) * self.config.velocity_decay
        speed = math.sqrt(particle.vx * particle.vx + particle.vy * particle.vy)
        if speed > self.config.max_speed:
            factor = self.config.max_speed / speed
            particle.vx *= factor
            particle.vy *= factor
        particle.x += dt * particle.vx
        particle.y += dt * particle.vy
        self._reflect(particle)

    def _reflect(self, particle: Particle) -> None:
        max_x = self.config.width
        max_y = self.config.height
        if particle.x < 0.0:
            particle.x = -particle.x
            particle.vx = abs(particle.vx)
        elif particle.x > max_x:
            particle.x = 2 * max_x - particle.x
            particle.vx = -abs(particle.vx)
        if particle.y < 0.0:
            particle.y = -particle.y
            particle.vy = abs(particle.vy)
        elif particle.y > max_y:
            particle.y = 2 * max_y - particle.y
            particle.vy = -abs(particle.vy)

    def _validate_matrix(self, matrix: List[List[float]], species_count: Optional[int] = None) -> None:
        expected = species_count or self.config.species_count
        if len(matrix) != expected:
            raise ValueError(f"Matrix requires {expected} rows")
        for row in matrix:
            if len(row) != expected:
                raise ValueError(f"Matrix rows must contain {expected} values")
