import math

import pytest

from backend.config import SimulationConfig
from backend.presets import get_preset
from backend.simulation import Particle, Simulation


def make_config(**overrides):
    base = SimulationConfig(
        width=20.0,
        height=20.0,
        interaction_radius=10.0,
        species_count=2,
        particle_count=2,
        time_step=1.0,
        velocity_decay=1.0,
        max_speed=500.0,
        acceleration_limit=500.0,
        frame_interval=0.02,
    )
    data = base.as_dict()
    data.update(overrides)
    return SimulationConfig(**data)


def test_particles_reflect_off_boundaries():
    config = make_config(species_count=1, particle_count=1, width=10.0, height=10.0)
    simulation = Simulation(config, [[0.0]])
    particle = Particle(species=0, x=9.5, y=5.0, vx=8.0, vy=0.0)
    simulation.particles = [particle]

    simulation.step()

    assert 0.0 <= particle.x <= config.width
    assert particle.vx < 0.0


def test_matrix_influences_velocity_direction():
    config = make_config()
    matrix = [
        [0.0, 6.0],
        [6.0, 0.0],
    ]
    simulation = Simulation(config, matrix)
    particle = Particle(species=0, x=4.0, y=10.0, vx=0.0, vy=0.0)
    neighbor = Particle(species=1, x=12.0, y=10.0, vx=0.0, vy=0.0)
    simulation.particles = [particle, neighbor]

    simulation.step()

    assert particle.vx > 0.0
    assert math.isclose(particle.vy, 0.0, abs_tol=1e-6)


def test_chaos_preset_matches_species_count():
    preset = get_preset("chaos")
    assert len(preset.matrix) == len(preset.matrix[0])


def test_base_matrix_expands_with_species_count():
    config = SimulationConfig(species_count=5)
    matrix = config.base_matrix(species_count=5)
    assert len(matrix) == 5
    assert all(len(row) == 5 for row in matrix)
    assert matrix[3][4] == matrix[0][1]


def test_invalid_matrix_size_raises_error():
    config = make_config(species_count=3, particle_count=3)
    simulation = Simulation(config, [[0.0, 1.0, 2.0]] * 3)
    with pytest.raises(ValueError):
        simulation.set_matrix([[0.0, 1.0], [2.0, 3.0]])
