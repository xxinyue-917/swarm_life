from dataclasses import dataclass
from typing import Dict, List

from .config import DEFAULT_CONFIG


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    matrix: List[List[float]]


def _scale(matrix: List[List[float]], factor: float) -> List[List[float]]:
    return [[value * factor for value in row] for row in matrix]


PRESETS: Dict[str, Preset] = {
    "flocking": Preset(
        name="flocking",
        description="Soft attraction between like species, mild repulsion otherwise.",
        matrix=[
            [0.0, -1.2, -1.2],
            [-1.2, 0.0, -1.2],
            [-1.2, -1.2, 0.0],
        ],
    ),
    "predator_prey": Preset(
        name="predator_prey",
        description="Species 0 chases 1, 1 chases 2, 2 avoids 0 to form loops.",
        matrix=[
            [0.0, 6.5, -4.8],
            [-5.0, 0.0, 6.0],
            [3.0, -6.5, 0.0],
        ],
    ),
    "chaos": Preset(
        name="chaos",
        description="High-energy asymmetric pushes for dynamic swirls.",
        matrix=_scale(DEFAULT_CONFIG.base_matrix(), 1.35),
    ),
}


def list_presets() -> List[Preset]:
    return list(PRESETS.values())


def get_preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown preset '{name}'") from exc
