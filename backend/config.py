from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class SimulationConfig:
    """Holds runtime parameters that control the particle simulation."""

    width: float = 800.0
    height: float = 600.0
    species_count: int = 3
    particle_count: int = 450
    interaction_radius: float = 32.0
    time_step: float = 0.12
    velocity_decay: float = 0.96
    max_speed: float = 220.0
    frame_interval: float = 0.03
    seed: Optional[int] = None
    acceleration_limit: float = 600.0
    _default_matrix: List[List[float]] = field(
        default_factory=lambda: [
            [0.0, 3.5, -6.0],
            [-6.0, 0.0, 4.5],
            [3.0, -4.0, 0.0],
        ]
    )

    def as_dict(self) -> dict:
        """Return a serialisable representation without private fields."""
        data = asdict(self)
        data.pop("_default_matrix", None)
        return data

    def base_matrix(self, species_count: Optional[int] = None) -> List[List[float]]:
        """Return a square interaction matrix sized for the requested species count."""
        size = species_count or self.species_count
        template = self._default_matrix
        template_size = len(template)
        matrix: List[List[float]] = []
        for row_index in range(size):
            template_row = template[row_index % template_size]
            matrix.append(
                [template_row[col_index % template_size] for col_index in range(size)]
            )
        return matrix


DEFAULT_CONFIG = SimulationConfig()
