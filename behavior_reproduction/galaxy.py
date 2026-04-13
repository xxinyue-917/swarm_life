#!/usr/bin/env python3
"""
Galaxy Behavior Reproduction — Morphological analogs via particle life

Reproduce the VISUAL MORPHOLOGY of galaxy phenomena using K_pos and K_rot.
These are NOT gravitational simulations — they are topological analogs that
produce similar shapes through different mechanisms (local interaction rules
instead of 1/r² gravity + inertia).

Presets:
    1: Elliptical galaxy — self-attracting pressure-supported blob
    2: Rotating disk — bulge + orbiting disk (lenticular S0)
    3: Ring galaxy (Cartwheel) — intruder passes through disk, creates expanding ring
    4: Galaxy merger — two rotating disk galaxies merge
    5: Differential rotation — inner disk orbits faster than outer

Controls:
    1-5:   Presets
    ↑/↓:   Adjust K_rot strength (rotation speed)
    ←/→:   Adjust K_pos cross-attraction
    M:     Matrix edit mode (WASD nav, E/X adjust)
    R:     Reset (randomize)
    T:     Toggle trajectory
    I:     Info panel
    V:     Show/hide matrix
    SPACE: Pause
    Q:     Quit
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pygame
from particle_life import Config, ParticleLife


# ============================================================
# PRESETS
# ============================================================

PRESETS = {
    '1_elliptical': {
        'name': 'Elliptical Galaxy',
        'n_species': 1,
        'n_particles': 200,
        'K_pos': np.array([[0.6]]),
        'K_rot': np.array([[0.0]]),
        'description': 'Self-attracting blob — pressure-supported, no rotation',
        'init': 'gaussian',  # Gaussian cluster at center
    },
    '2_rotating_disk': {
        'name': 'Rotating Disk (S0)',
        'n_species': 2,
        'n_particles': 100,
        'K_pos': np.array([
            [0.8, 0.3],
            [0.3, 0.15],
        ]),
        'K_rot': np.array([
            [0.0,  0.6],
            [-0.6, 0.0],
        ]),
        'description': 'Bulge (S0) + disk (S1). Antisymmetric K_rot → disk orbits bulge',
        'init': 'disk',
    },
    '3_ring_galaxy': {
        'name': 'Ring Galaxy (Cartwheel)',
        'n_species': 3,
        'n_particles': 80,
        'K_pos': np.array([
            [0.8,  0.3,  0.0],   # Bulge: self-cohesion, attracts disk
            [0.3,  0.15, -0.5],   # Disk: attracted to bulge, REPELLED by intruder
            [0.4,  0.0,  0.8],   # Intruder: attracted to bulge, self-cohesion
        ]),
        'K_rot': np.array([
            [0.0,  0.4,  0.0],
            [-0.4, 0.0,  0.0],
            [0.0,  0.0,  0.0],
        ]),
        'description': 'Intruder (S2) passes through disk → repulsion creates expanding ring',
        'init': 'ring_galaxy',
    },
    '4_merger': {
        'name': 'Galaxy Merger',
        'n_species': 4,
        'n_particles': 50,
        'K_pos': np.array([
            # Galaxy A: S0 bulge, S1 disk
            # Galaxy B: S2 bulge, S3 disk
            [0.8, 0.3, 0.1, 0.0],   # A-bulge: self, attracts A-disk, weakly attracts B-bulge
            [0.3, 0.15, 0.0, 0.0],  # A-disk: attracted to A-bulge
            [0.1, 0.0, 0.8, 0.3],   # B-bulge: weakly attracted to A-bulge, self, attracts B-disk
            [0.0, 0.0, 0.3, 0.15],  # B-disk: attracted to B-bulge
        ]),
        'K_rot': np.array([
            # Intra-galaxy rotation, no cross-galaxy rotation
            [0.0,  0.5,  0.0,  0.0],
            [-0.5, 0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.5],
            [0.0,  0.0, -0.5,  0.0],
        ]),
        'description': 'Two rotating galaxies (A: S0+S1, B: S2+S3) merge via weak cross-attraction',
        'init': 'two_galaxies',
    },
    '5_differential': {
        'name': 'Differential Rotation',
        'n_species': 3,
        'n_particles': 80,
        'K_pos': np.array([
            [0.8, 0.3, 0.15],   # Bulge attracts all
            [0.3, 0.15, 0.1],   # Inner disk
            [0.15, 0.1, 0.1],   # Outer disk
        ]),
        'K_rot': np.array([
            [0.0,  0.7,  0.3],   # Bulge drives inner fast, outer slow
            [-0.7, 0.0,  0.0],
            [-0.3, 0.0,  0.0],
        ]),
        'description': 'Inner disk orbits faster than outer → differential rotation',
        'init': 'disk_3species',
    },
}


# ============================================================
# DEMO
# ============================================================

class GalaxyDemo(ParticleLife):
    """Galaxy morphology reproduction via particle life."""

    def __init__(self):
        preset = PRESETS['2_rotating_disk']
        self.current_preset = '2_rotating_disk'
        self.preset_name = preset['name']

        config = Config(
            n_species=preset['n_species'],
            n_particles=preset['n_particles'],
            sim_width=15.0,
            sim_height=15.0,
            r_max=3.0,
            beta=0.3,
            force_scale=0.5,
            max_speed=1.5,
            a_rot=2.0,
            far_attraction=0.05,
            position_matrix=preset['K_pos'].tolist(),
            orientation_matrix=preset['K_rot'].tolist(),
        )

        super().__init__(config, headless=False)

        self._init_positions(preset['init'])

        pygame.display.set_caption("Galaxy — Behavior Reproduction")
        self._print_help()

    def _init_positions(self, mode):
        """Initialize particle positions based on preset mode."""
        sw, sh = self.config.sim_width, self.config.sim_height
        cx, cy = sw / 2, sh / 2

        if mode == 'gaussian':
            # Gaussian blob at center
            spread = min(sw, sh) * 0.15
            self.positions[:, 0] = np.random.normal(cx, spread, self.n)
            self.positions[:, 1] = np.random.normal(cy, spread, self.n)

        elif mode == 'disk':
            # Bulge (S0) tight at center, Disk (S1) spread in ring
            pps = self.n // self.n_species
            for i in range(self.n):
                sid = min(i // pps, self.n_species - 1)
                if sid == 0:  # Bulge
                    self.positions[i] = [cx + np.random.normal(0, 0.5),
                                         cy + np.random.normal(0, 0.5)]
                else:  # Disk
                    angle = np.random.uniform(0, 2 * np.pi)
                    r = np.random.uniform(1.5, 4.0)
                    self.positions[i] = [cx + r * np.cos(angle),
                                         cy + r * np.sin(angle)]

        elif mode == 'ring_galaxy':
            # Bulge at center, disk spread, intruder at edge
            pps = self.n // self.n_species
            for i in range(self.n):
                sid = min(i // pps, self.n_species - 1)
                if sid == 0:  # Bulge
                    self.positions[i] = [cx + np.random.normal(0, 0.3),
                                         cy + np.random.normal(0, 0.3)]
                elif sid == 1:  # Disk
                    angle = np.random.uniform(0, 2 * np.pi)
                    r = np.random.uniform(1.5, 4.0)
                    self.positions[i] = [cx + r * np.cos(angle),
                                         cy + r * np.sin(angle)]
                else:  # Intruder — starts at top edge
                    self.positions[i] = [cx + np.random.normal(0, 0.3),
                                         1.5 + np.random.normal(0, 0.3)]

        elif mode == 'two_galaxies':
            # Two galaxies offset from center
            pps = self.n // self.n_species
            offsets = [(-3, -2), (-3, -2), (3, 2), (3, 2)]  # A left, B right
            spreads = [0.3, 2.0, 0.3, 2.0]  # bulge tight, disk wide
            for i in range(self.n):
                sid = min(i // pps, self.n_species - 1)
                ox, oy = offsets[sid]
                spread = spreads[sid]
                if spread < 1.0:  # Bulge
                    self.positions[i] = [cx + ox + np.random.normal(0, spread),
                                         cy + oy + np.random.normal(0, spread)]
                else:  # Disk
                    angle = np.random.uniform(0, 2 * np.pi)
                    r = np.random.uniform(0.5, spread)
                    self.positions[i] = [cx + ox + r * np.cos(angle),
                                         cy + oy + r * np.sin(angle)]

        elif mode == 'disk_3species':
            # Bulge + inner ring + outer ring
            pps = self.n // self.n_species
            for i in range(self.n):
                sid = min(i // pps, self.n_species - 1)
                if sid == 0:  # Bulge
                    self.positions[i] = [cx + np.random.normal(0, 0.3),
                                         cy + np.random.normal(0, 0.3)]
                elif sid == 1:  # Inner disk
                    angle = np.random.uniform(0, 2 * np.pi)
                    r = np.random.uniform(1.0, 2.5)
                    self.positions[i] = [cx + r * np.cos(angle),
                                         cy + r * np.sin(angle)]
                else:  # Outer disk
                    angle = np.random.uniform(0, 2 * np.pi)
                    r = np.random.uniform(3.0, 5.0)
                    self.positions[i] = [cx + r * np.cos(angle),
                                         cy + r * np.sin(angle)]
        else:
            # Random scatter
            self.positions[:, 0] = np.random.uniform(1, sw - 1, self.n)
            self.positions[:, 1] = np.random.uniform(1, sh - 1, self.n)

        # Clamp to workspace
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0.5, sw - 0.5)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0.5, sh - 0.5)
        self.velocities[:] = 0.0

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
        self.n = n_species * n_particles

        self.matrix = preset['K_pos'].copy()
        self.alignment_matrix = preset['K_rot'].copy()

        self.colors = []
        for i in range(n_species):
            hue = i / max(n_species, 1)
            color = pygame.Color(0)
            color.hsva = (hue * 360, 70, 90, 100)
            self.colors.append((color.r, color.g, color.b))

        self.initialize_particles()
        self._init_positions(preset['init'])

        print(f"Preset: {preset['name']} — {preset['description']}")

    # ================================================================
    # Drawing
    # ================================================================

    def draw(self):
        # Standard ParticleLife rendering
        super().draw()

        # Add galaxy-specific info overlay
        if self.show_info:
            self._draw_galaxy_info()

    def _draw_galaxy_info(self):
        preset = PRESETS.get(self.current_preset, {})
        desc = preset.get('description', '')
        lines = [
            f"Preset: {self.preset_name}",
            f"  {desc}",
            "",
            "1: Elliptical  2: Rotating Disk  3: Ring Galaxy",
            "4: Merger  5: Differential Rotation",
        ]
        y = self.config.height - len(lines) * 20 - 10
        for line in lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 20

    def _print_help(self):
        print()
        print("  Galaxy Behavior Reproduction")
        print("  ────────────────────────────")
        print("  1  Elliptical       5  Differential rotation")
        print("  2  Rotating disk    M  Edit matrix")
        print("  3  Ring galaxy      R  Reset positions")
        print("  4  Galaxy merger    All particle_life keys work")
        print()

    def handle_events(self):
        """Extend parent's event handling with preset keys."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                # Preset keys
                if event.key == pygame.K_1:
                    self._load_preset('1_elliptical')
                    continue
                elif event.key == pygame.K_2:
                    self._load_preset('2_rotating_disk')
                    continue
                elif event.key == pygame.K_3:
                    self._load_preset('3_ring_galaxy')
                    continue
                elif event.key == pygame.K_4:
                    self._load_preset('4_merger')
                    continue
                elif event.key == pygame.K_5:
                    self._load_preset('5_differential')
                    continue

            # Post the event back so parent can handle it
            pygame.event.post(event)

        # Let parent handle all standard controls
        return super().handle_events()


def main():
    demo = GalaxyDemo()
    demo.run()


if __name__ == '__main__':
    main()
