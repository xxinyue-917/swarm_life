#!/usr/bin/env python3
"""
Planetary System — Sun-Earth-Moon and multi-planet orbital dynamics

Reproduce hierarchical orbital behavior using K_pos (radial attraction)
and K_rot (tangential forcing). Each celestial body is a species cluster.

Key insight: the limit-cycle orbital radius for a bound species pair (i,j)
is r_eq ≈ beta[i,j] * r_max. This file OVERRIDES particle_life's physics
(only here, not the shared engine) so that beta is a per-pair matrix
instead of a global scalar — enabling distinct orbital radii per planet.

Presets:
    1: Sun-Earth (basic orbit)
    2: Sun-Earth-Moon (hierarchical orbit)
    3: Solar System (Sun + 4 planets at staggered betas)
    4: Solar System + Moon
    5: Saturn's Rings

Controls:
    1-5:   Presets
    ←/→:   Select planet (for beta / K_rot adjustment)
    ↑/↓:   Adjust beta[sel, Sun]  → farther/closer orbit
    W/S:   Adjust K_rot[sel, Sun] → faster/slower orbit
    M:     Matrix edit (K_pos / K_rot, standard controls)
    R:     Reset positions
    SPACE: Pause
    Q:     Quit
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import numpy as np
import pygame
from particle_life import Config, ParticleLife

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# ============================================================
# PHYSICS OVERRIDE — per-pair beta (repulsion parameter)
# ============================================================
# Scoped physics change (allowed only in planet.py): beta becomes a
# per-pair matrix beta[i,j] instead of a global scalar. The limit-cycle
# orbital radius for bound pair (i,j) is r_eq = beta[i,j] * r_max,
# so per-pair beta gives per-planet orbital radii.
#
# Kernel is otherwise identical to src/particle_life.py:_compute_velocities_jit

@njit(cache=True)
def _compute_velocities_beta_matrix_jit(positions, species, k_pos, k_rot, beta_mat,
                                        n, r_max, force_scale, far_attraction, a_rot):
    new_velocities = np.zeros((n, 2), dtype=np.float64)

    for i in range(n):
        vx = 0.0
        vy = 0.0
        si = species[i]

        for j in range(n):
            if j == i:
                continue

            sj = species[j]
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            r = math.sqrt(dx * dx + dy * dy)

            if r < 1e-8:
                continue

            r_norm = r / r_max
            beta_ij = beta_mat[si, sj]
            inv_1_minus_beta = 1.0 / (1.0 - beta_ij) if beta_ij < 1.0 else 1.0

            if r_norm >= 1.0:
                if far_attraction > 0:
                    kp = k_pos[si, sj]
                    inv_r = 1.0 / r
                    F = kp * far_attraction
                    vx += force_scale * F * dx * inv_r
                    vy += force_scale * F * dy * inv_r
                continue

            kp = k_pos[si, sj]
            kr = k_rot[si, sj]

            inv_r = 1.0 / r
            r_hat_x = dx * inv_r
            r_hat_y = dy * inv_r
            t_hat_x = -r_hat_y
            t_hat_y = r_hat_x

            if r_norm < beta_ij:
                F = r_norm / beta_ij - 1.0
            else:
                triangle = 1.0 - abs(2.0 * r_norm - 1.0 - beta_ij) * inv_1_minus_beta
                peak_r = 0.5 * (1.0 + beta_ij)
                if r_norm < peak_r:
                    F = kp * triangle
                else:
                    F = kp * max(far_attraction, triangle)

            vx += force_scale * F * r_hat_x
            vy += force_scale * F * r_hat_y

            swirl_weight = 1.0 - r_norm
            if swirl_weight < 0.0:
                swirl_weight = 0.0
            swirl_gain = kr * a_rot * swirl_weight

            vx += swirl_gain * t_hat_x
            vy += swirl_gain * t_hat_y

        new_velocities[i, 0] = vx
        new_velocities[i, 1] = vy

    return new_velocities


# ============================================================
# PRESETS
# ============================================================

PRESETS = {
    # With per-pair beta, limit-cycle radius for pair (i,j) ≈ beta[i,j] * r_max.
    # r_max = 8.0 throughout, so beta=0.2 → r_eq=1.6, beta=0.5 → r_eq=4.0, etc.
    '1_sun_earth': {
        'name': 'Sun-Earth',
        'n_species': 2,
        'n_particles': 40,
        'K_pos': np.array([
            [1.0, 0.3],
            [0.3, 0.9],
        ]),
        'K_rot': np.array([
            [0.0, 0.0],
            [0.3, 0.0],
        ]),
        'beta': np.array([
            [0.10, 0.38],   # Sun bulge tight; Sun-Earth pair → r_eq ≈ 3.0
            [0.38, 0.10],
        ]),
        'description': 'Basic orbit — Earth (S1) orbits Sun (S0) at r ≈ beta·r_max = 3.0',
        'bodies': [
            {'name': 'Sun',   'count': 10, 'pos': (0, 0),    'spread': 0.3},
            {'name': 'Earth', 'count': 6,  'pos': (3.0, 0),  'spread': 0.2},
        ],
    },
    '2_sun_earth_moon': {
        'name': 'Sun-Earth-Moon',
        'n_species': 3,
        'n_particles': 30,
        'K_pos': np.array([
            [1.0,  0.25, 0.0 ],   # Sun tight self-cohesion, attracts Earth
            [0.25, 0.9,  0.3 ],   # Earth tight self-cohesion
            [0.0,  0.3,  0.9 ],   # Moon tight self-cohesion
        ]),
        'K_rot': np.array([
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.0, 0.9, 0.0],   # Moon orbits Earth ~20x faster than Earth orbits Sun
        ]),
        'beta': np.array([
            [0.10, 0.45, 0.10],   # Sun-Earth pair → r_eq = 3.6
            [0.45, 0.10, 0.15],   # Earth-Moon pair → r_eq = 1.2
            [0.10, 0.15, 0.10],
        ]),
        'description': 'Moon (r≈1.2) orbits Earth, Earth (r≈3.6) orbits Sun — per-pair beta',
        'bodies': [
            # Moon offset PERPENDICULAR to Sun-Earth axis so its orbit is visible
            {'name': 'Sun',   'count': 8, 'pos': (0, 0),     'spread': 0.3},
            {'name': 'Earth', 'count': 4, 'pos': (3.6, 0),   'spread': 0.2},
            {'name': 'Moon',  'count': 2, 'pos': (3.6, 1.2), 'spread': 0.1},
        ],
    },
    '3_solar_system': {
        'name': 'Solar System (4 planets)',
        'n_species': 5,
        'n_particles': 20,
        'K_pos': np.array([
            # Sun    Merc   Venus  Earth  Mars
            [1.0,   0.3,   0.3,   0.3,   0.3 ],   # Sun tight self-cohesion
            [0.3,   0.9,   0.0,   0.0,   0.0 ],   # Mercury tight self-cohesion
            [0.3,   0.0,   0.9,   0.0,   0.0 ],
            [0.3,   0.0,   0.0,   0.9,   0.0 ],
            [0.3,   0.0,   0.0,   0.0,   0.9 ],
        ]),
        'K_rot': np.array([
            [0.0,  0.0,  0.0,  0.0,  0.0 ],
            [0.4,  0.0,  0.0,  0.0,  0.0 ],
            [0.3,  0.0,  0.0,  0.0,  0.0 ],
            [0.2,  0.0,  0.0,  0.0,  0.0 ],
            [0.15, 0.0,  0.0,  0.0,  0.0 ],
        ]),
        'beta': np.array([
            # Sun   Merc   Venus  Earth  Mars   — per-pair limit cycle radii
            [0.10, 0.19,  0.31,  0.44,  0.56],   # Sun row
            [0.19, 0.10,  0.10,  0.10,  0.10],
            [0.31, 0.10,  0.10,  0.10,  0.10],
            [0.44, 0.10,  0.10,  0.10,  0.10],
            [0.56, 0.10,  0.10,  0.10,  0.10],
        ]),
        'description': '4 planets with distinct per-pair beta → r=1.5/2.5/3.5/4.5',
        'bodies': [
            {'name': 'Sun',     'count': 8, 'pos': (0, 0),     'spread': 0.3},
            {'name': 'Mercury', 'count': 2, 'pos': (1.5, 0),   'spread': 0.1},
            {'name': 'Venus',   'count': 3, 'pos': (2.5, 0),   'spread': 0.15},
            {'name': 'Earth',   'count': 4, 'pos': (3.5, 0),   'spread': 0.15},
            {'name': 'Mars',    'count': 3, 'pos': (4.5, 0),   'spread': 0.1},
        ],
    },
    '4_solar_moon': {
        'name': 'Solar System + Moon',
        'n_species': 6,
        'n_particles': 15,
        'K_pos': np.array([
            # Sun   Merc  Venus Earth Mars  Moon
            [1.0,  0.3,  0.3,  0.3,  0.3,  0.0 ],   # Sun tight
            [0.3,  0.9,  0.0,  0.0,  0.0,  0.0 ],
            [0.3,  0.0,  0.9,  0.0,  0.0,  0.0 ],
            [0.3,  0.0,  0.0,  0.9,  0.0,  0.3 ],
            [0.3,  0.0,  0.0,  0.0,  0.9,  0.0 ],
            [0.0,  0.0,  0.0,  0.3,  0.0,  0.9 ],
        ]),
        'K_rot': np.array([
            [0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
            [0.4,  0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3,  0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2,  0.0, 0.0, 0.0, 0.0, 0.0],
            [0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0,  0.0, 0.0, 0.4, 0.0, 0.0],
        ]),
        'beta': np.array([
            # Sun   Merc  Venus Earth Mars  Moon
            [0.10, 0.19, 0.31, 0.44, 0.56, 0.10],
            [0.19, 0.10, 0.10, 0.10, 0.10, 0.10],
            [0.31, 0.10, 0.10, 0.10, 0.10, 0.10],
            [0.44, 0.10, 0.10, 0.10, 0.10, 0.10],   # Earth-Moon at 0.10 → r_eq = 0.8
            [0.56, 0.10, 0.10, 0.10, 0.10, 0.10],
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        ]),
        'description': '4 planets + Moon orbiting Earth — all radii via per-pair beta',
        'bodies': [
            {'name': 'Sun',     'count': 6, 'pos': (0, 0),     'spread': 0.3},
            {'name': 'Mercury', 'count': 2, 'pos': (1.5, 0),   'spread': 0.1},
            {'name': 'Venus',   'count': 3, 'pos': (2.5, 0),   'spread': 0.1},
            {'name': 'Earth',   'count': 3, 'pos': (3.5, 0),   'spread': 0.15},
            {'name': 'Mars',    'count': 2, 'pos': (4.5, 0),   'spread': 0.1},
            {'name': 'Moon',    'count': 2, 'pos': (4.3, 0),   'spread': 0.05},
        ],
    },
    '5_saturn_rings': {
        'name': "Saturn's Rings",
        'n_species': 3,
        'n_particles': 40,
        'K_pos': np.array([
            [1.0,  0.3,  0.3 ],   # Saturn tight self-cohesion
            [0.3,  0.9,  0.0 ],   # Moons tight self-cohesion
            [0.3,  0.0, -0.05],   # Ring self-repulsion → spreads (intentional)
        ]),
        'K_rot': np.array([
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ]),
        'beta': np.array([
            [0.10, 0.25, 0.44],   # Saturn-disk r_eq=2.0, Saturn-ring r_eq=3.5
            [0.25, 0.10, 0.10],
            [0.44, 0.10, 0.10],
        ]),
        'description': "Saturn + inner moons (r≈2.0) + ring (r≈3.5) via per-pair beta",
        'bodies': [
            {'name': 'Saturn', 'count': 8,  'pos': (0, 0),   'spread': 0.3},
            {'name': 'Moons',  'count': 4,  'pos': (2.0, 0), 'spread': 0.3},
            {'name': 'Ring',   'count': 15, 'pos': (3.5, 0), 'spread': 1.5},
        ],
    },
}


# ============================================================
# DEMO
# ============================================================

class PlanetDemo(ParticleLife):
    """Planetary system reproduction via particle life."""

    def __init__(self):
        preset = PRESETS['2_sun_earth_moon']
        self.current_preset = '2_sun_earth_moon'
        self.preset_name = preset['name']

        config = Config(
            n_species=preset['n_species'],
            n_particles=preset['n_particles'],
            sim_width=15.0,
            sim_height=15.0,
            r_max=8.0,
            beta=0.2,   # scalar fallback; actual physics uses self.beta_matrix per-pair
            force_scale=0.5,
            max_speed=2.0,
            a_rot=0.5,
            far_attraction=0.05,
            position_matrix=preset['K_pos'].tolist(),
            orientation_matrix=preset['K_rot'].tolist(),
        )

        super().__init__(config, headless=False)

        self.beta_matrix = self._make_beta_matrix(preset)
        self._init_bodies(preset['bodies'])
        self.selected_planet = 1  # Start with first planet (not Sun)

        pygame.display.set_caption("Planetary System — Behavior Reproduction")
        self._print_help()

    def _make_beta_matrix(self, preset):
        """Build per-pair beta matrix from preset, or uniform fallback from config.beta."""
        n = preset['n_species']
        if 'beta' in preset:
            return np.asarray(preset['beta'], dtype=np.float64).copy()
        return np.full((n, n), self.config.beta, dtype=np.float64)

    def compute_velocities(self):
        """Override parent physics: use per-pair beta matrix instead of scalar beta."""
        return _compute_velocities_beta_matrix_jit(
            self.positions.astype(np.float64),
            self.species.astype(np.int64),
            self.matrix.astype(np.float64),
            self.alignment_matrix.astype(np.float64),
            self.beta_matrix.astype(np.float64),
            self.n,
            self.config.r_max, self.config.force_scale,
            self.config.far_attraction, self.config.a_rot,
        )

    def _init_bodies(self, bodies):
        """Place particles for each celestial body at specified positions."""
        sw, sh = self.config.sim_width, self.config.sim_height
        cx, cy = sw / 2, sh / 2

        idx = 0
        for sid, body in enumerate(bodies):
            count = body['count']
            ox, oy = body['pos']
            spread = body['spread']
            for _ in range(count):
                if idx >= self.n:
                    break
                self.species[idx] = sid
                self.positions[idx, 0] = cx + ox + np.random.normal(0, spread)
                self.positions[idx, 1] = cy + oy + np.random.normal(0, spread)
                self.velocities[idx] = [0.0, 0.0]
                idx += 1

        # Clamp
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0.5, sw - 0.5)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0.5, sh - 0.5)

    def _load_preset(self, key):
        if key not in PRESETS:
            return
        preset = PRESETS[key]
        self.current_preset = key
        self.preset_name = preset['name']

        n_species = preset['n_species']
        n_particles = preset['n_particles']

        self.config.n_species = n_species
        self.config.n_particles = n_particles
        self.n_species = n_species
        self.n = sum(b['count'] for b in preset['bodies'])

        self.matrix = preset['K_pos'].copy()
        self.alignment_matrix = preset['K_rot'].copy()
        self.beta_matrix = self._make_beta_matrix(preset)

        self.colors = []
        # Custom colors for planets
        planet_colors = [
            (255, 200, 50),   # Sun — yellow
            (180, 180, 180),  # Mercury — gray
            (230, 180, 100),  # Venus — tan
            (80, 130, 230),   # Earth — blue
            (200, 100, 80),   # Mars — red
            (200, 200, 200),  # Moon — light gray
        ]
        for i in range(n_species):
            if i < len(planet_colors):
                self.colors.append(planet_colors[i])
            else:
                hue = i / max(n_species, 1)
                color = pygame.Color(0)
                color.hsva = (hue * 360, 70, 90, 100)
                self.colors.append((color.r, color.g, color.b))

        self.initialize_particles()
        self._init_bodies(preset['bodies'])

        print(f"Preset: {preset['name']} — {preset['description']}")

    def draw(self):
        """Standard draw + planet info overlay."""
        super().draw()

        if self.show_info:
            self._draw_planet_info()

    def _draw_planet_info(self):
        preset = PRESETS.get(self.current_preset, {})
        desc = preset.get('description', '')
        bodies = preset.get('bodies', [])

        # Selected planet info
        sel = min(self.selected_planet, self.n_species - 1)
        sel_name = bodies[sel]['name'] if sel < len(bodies) else f"S{sel}"
        beta_to_sun = self.beta_matrix[sel, 0] if sel > 0 else 0
        krot_to_sun = self.alignment_matrix[sel, 0] if sel > 0 else 0
        r_eq = beta_to_sun * self.config.r_max

        lines = [
            f"Preset: {self.preset_name}",
            f"  {desc}",
            "",
            f"Selected: [{sel_name}]  (←/→ to switch)",
            f"  beta[{sel_name},Sun] = {beta_to_sun:.3f}  r_eq ≈ {r_eq:.2f}  (↑/↓ adjust)",
            f"  K_rot[{sel_name},Sun] = {krot_to_sun:.3f}  (W/S adjust speed)",
            "",
            "1-5: Presets  M: K_pos/K_rot edit",
        ]
        y = self.config.height - len(lines) * 20 - 10
        for line in lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 20

    def _print_help(self):
        print()
        print("  Planetary System — per-pair beta enabled")
        print("  ─────────────────────────────────────────")
        print("  1  Sun-Earth          4  Solar System + Moon")
        print("  2  Sun-Earth-Moon     5  Saturn's Rings")
        print("  3  Solar System       M  Edit K_pos / K_rot")
        print()
        print("  ←/→ select planet    ↑/↓ adjust beta  (r_eq = beta * r_max)")
        print("  W/S  adjust K_rot    R  reset positions")
        print()
        print("  NOTE: planet.py overrides physics to use per-pair beta[i,j].")
        print("  Limit-cycle radius for pair (i,j) ≈ beta[i,j] * r_max.")
        print()

    def handle_events(self):
        """Extend parent with preset keys."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                # Planet tuning keys are ONLY active when the matrix editor is closed.
                # When show_matrix is True, let WASD/arrows fall through to parent for
                # cell navigation and E/X adjustment.
                if not getattr(self, 'show_matrix', False):
                    if event.key == pygame.K_LEFT:
                        self.selected_planet = max(1, self.selected_planet - 1)
                        bodies = PRESETS.get(self.current_preset, {}).get('bodies', [])
                        name = bodies[self.selected_planet]['name'] if self.selected_planet < len(bodies) else f"S{self.selected_planet}"
                        print(f"Selected: {name}")
                        continue
                    elif event.key == pygame.K_RIGHT:
                        self.selected_planet = min(self.n_species - 1, self.selected_planet + 1)
                        bodies = PRESETS.get(self.current_preset, {}).get('bodies', [])
                        name = bodies[self.selected_planet]['name'] if self.selected_planet < len(bodies) else f"S{self.selected_planet}"
                        print(f"Selected: {name}")
                        continue
                    elif event.key == pygame.K_UP:
                        # Increase beta[sel, Sun] → farther orbit (r_eq = beta * r_max)
                        sel = self.selected_planet
                        if sel > 0:
                            new_b = min(0.9, self.beta_matrix[sel, 0] + 0.02)
                            self.beta_matrix[sel, 0] = new_b
                            self.beta_matrix[0, sel] = new_b
                            r_eq = new_b * self.config.r_max
                            print(f"beta[{sel},Sun] = {new_b:.3f}  r_eq ≈ {r_eq:.2f} (farther)")
                        continue
                    elif event.key == pygame.K_DOWN:
                        # Decrease beta[sel, Sun] → closer orbit
                        sel = self.selected_planet
                        if sel > 0:
                            new_b = max(0.05, self.beta_matrix[sel, 0] - 0.02)
                            self.beta_matrix[sel, 0] = new_b
                            self.beta_matrix[0, sel] = new_b
                            r_eq = new_b * self.config.r_max
                            print(f"beta[{sel},Sun] = {new_b:.3f}  r_eq ≈ {r_eq:.2f} (closer)")
                        continue
                    elif event.key == pygame.K_w:
                        # Increase K_rot → faster orbit
                        sel = self.selected_planet
                        if sel > 0:
                            self.alignment_matrix[sel, 0] = min(1.0, self.alignment_matrix[sel, 0] + 0.02)
                            print(f"K_rot[{sel},Sun] = {self.alignment_matrix[sel,0]:.3f} (faster)")
                        continue
                    elif event.key == pygame.K_s:
                        # Decrease K_rot → slower orbit
                        sel = self.selected_planet
                        if sel > 0:
                            self.alignment_matrix[sel, 0] = max(0.0, self.alignment_matrix[sel, 0] - 0.02)
                            print(f"K_rot[{sel},Sun] = {self.alignment_matrix[sel,0]:.3f} (slower)")
                        continue

                # Presets (always active, independent of matrix mode)
                if event.key == pygame.K_1:
                    self._load_preset('1_sun_earth')
                    continue
                elif event.key == pygame.K_2:
                    self._load_preset('2_sun_earth_moon')
                    continue
                elif event.key == pygame.K_3:
                    self._load_preset('3_solar_system')
                    continue
                elif event.key == pygame.K_4:
                    self._load_preset('4_solar_moon')
                    continue
                elif event.key == pygame.K_5:
                    self._load_preset('5_saturn_rings')
                    continue

            pygame.event.post(event)

        return super().handle_events()


def main():
    demo = PlanetDemo()
    demo.run()


if __name__ == '__main__':
    main()
