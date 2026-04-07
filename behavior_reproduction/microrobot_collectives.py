#!/usr/bin/env python3
"""
Reproduce behaviors from:
"Microrobot collectives with reconfigurable morphologies, behaviors, and functions"
(Gardi, Ceron, Wang, Petersen, Sitti — Nature Communications 2022)

Target behaviors:
1. ROTATION: All agents orbit around a common center of mass
   - Each agent spins, creating tangential flow → collective orbits
   - Higher frequency → more spread out, slower orbiting
2. CHAIN: Agents align into linear chains
   - Agents attract along one axis, forming connected or separated chains
   - Low frequency → connected chains; high frequency → separated chains

In our model:
- Rotation → symmetric K_rot (creates orbital motion) + attractive K_pos
- Chain → strong diagonal K_pos (self-cohesion) + tridiagonal K_pos (neighbor attraction)

Controls:
    1:     Rotation mode (low speed — tight cluster, fast orbit)
    2:     Rotation mode (medium speed — expanded, moderate orbit)
    3:     Rotation mode (high speed — spread out, slow orbit)
    4:     Chain mode (connected — low separation)
    5:     Chain mode (separated — high separation)
    6:     Oscillation mode (collective oscillates about center)
    ↑/↓:   Adjust K_rot strength (rotation speed)
    ←/→:   Adjust K_pos cross-attraction (cohesion)
    R:     Reset (random positions)
    I:     Toggle info
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
# PRESETS — Edit these to tune behaviors
# ============================================================

PRESETS = {
    # Rotation modes: symmetric K_rot + attractive K_pos
    # Increasing rot_strength → faster individual spin → wider collective
    '1_rotation_tight': {
        'name': 'Rotation (tight)',
        'n_species': 1,
        'n_particles': 120,
        'K_pos': [[0.5]],
        'K_rot': [[0.3]],
        'description': 'Tight rotating cluster — low rotation speed',
    },
    '2_rotation_medium': {
        'name': 'Rotation (medium)',
        'n_species': 1,
        'n_particles': 120,
        'K_pos': [[0.3]],
        'K_rot': [[0.6]],
        'description': 'Expanded rotation — moderate speed',
    },
    '3_rotation_wide': {
        'name': 'Rotation (wide)',
        'n_species': 1,
        'n_particles': 120,
        'K_pos': [[0.15]],
        'K_rot': [[0.9]],
        'description': 'Spread out rotation — high speed, slow orbit',
    },
    '4_chain_connected': {
        'name': 'Chain (connected)',
        'chain': True,
        'n_species': 20,
        'n_particles': 6,
        'self_cohesion': 0.8,
        'neighbor_attraction': 0.6,  # Strong → connected chain
        'description': 'Connected chain — 20 species, tridiagonal K_pos, strong neighbor attraction',
    },
    '5_chain_separated': {
        'name': 'Chain (separated)',
        'chain': True,
        'n_species': 20,
        'n_particles': 6,
        'self_cohesion': 0.8,
        'neighbor_attraction': 0.2,  # Weak → separated beads
        'description': 'Separated chain — 20 species, weak neighbor attraction creates separated beads',
    },
    '6_oscillation': {
        'name': 'Oscillation',
        'n_species': 1,
        'n_particles': 120,
        'K_pos': [[0.4]],
        'K_rot': [[0.5]],
        'oscillation': True,
        'description': 'Oscillating collective — K_rot flips sign periodically (CW↔CCW)',
    },
}


# ============================================================
# DEMO
# ============================================================

class MicrorobotDemo(ParticleLife):
    """Reproduce microrobot collective behaviors."""

    def __init__(self):
        # Start with rotation preset
        preset = PRESETS['1_rotation_tight']
        self.current_preset = '1_rotation_tight'
        self.preset_name = preset['name']

        config = Config(
            n_species=preset['n_species'],
            n_particles=preset['n_particles'],
            sim_width=10.0,
            sim_height=10.0,
            r_max=2.0,
            beta=0.6,
            force_scale=1.0,
            max_speed=1.0,
            a_rot=1.0,
            far_attraction=0.05,
            position_matrix=preset['K_pos'],
            orientation_matrix=preset['K_rot'],
        )

        super().__init__(config, headless=False)

        self.hide_gui = False
        self.oscillation_mode = False  # Time-varying K_rot for oscillation
        self.oscillation_period = 10   # Frames per half-cycle
        self.oscillation_base_krot = None  # Base K_rot to oscillate around
        self.frame_count = 0
        self._scatter()

        pygame.display.set_caption("Microrobot Collectives — Behavior Reproduction")
        self._print_help()

    def _scatter(self):
        """Random positions across workspace."""
        m = 1.0
        sw, sh = self.config.sim_width, self.config.sim_height
        # Cluster near center (like microrobot experiments)
        cx, cy = sw / 2, sh / 2
        spread = min(sw, sh) * 0.25
        self.positions[:, 0] = np.random.normal(cx, spread, self.n)
        self.positions[:, 1] = np.random.normal(cy, spread, self.n)
        # Clamp to bounds
        self.positions[:, 0] = np.clip(self.positions[:, 0], m, sw - m)
        self.positions[:, 1] = np.clip(self.positions[:, 1], m, sh - m)
        self.velocities[:] = 0.0

    def _load_preset(self, key):
        """Load a preset configuration."""
        if key not in PRESETS:
            return
        preset = PRESETS[key]
        self.current_preset = key
        self.preset_name = preset['name']

        n_species = preset['n_species']
        n_particles = preset['n_particles']

        # Update config
        self.config.n_species = n_species
        self.config.n_particles = n_particles
        self.n_species = n_species
        self.n = n_species * n_particles

        # Build matrices — either explicit or chain-generated
        if preset.get('chain', False):
            # Tridiagonal K_pos: each species attracts only its neighbors
            K_pos = np.zeros((n_species, n_species))
            sc = preset['self_cohesion']
            na = preset['neighbor_attraction']
            for i in range(n_species):
                K_pos[i, i] = sc
                if i > 0:
                    K_pos[i, i - 1] = na
                if i < n_species - 1:
                    K_pos[i, i + 1] = na
            K_rot = np.zeros((n_species, n_species))
        else:
            K_pos = np.array(preset['K_pos'], dtype=float)
            K_rot = np.array(preset['K_rot'], dtype=float)

        self.matrix = K_pos
        self.alignment_matrix = K_rot

        # Regenerate colors
        self.colors = []
        for i in range(n_species):
            hue = i / max(n_species, 1)
            color = pygame.Color(0)
            color.hsva = (hue * 360, 70, 90, 100)
            self.colors.append((color.r, color.g, color.b))

        # Reinitialize particles
        self.initialize_particles()
        self._scatter()

        # Oscillation mode
        if preset.get('oscillation', False):
            self.oscillation_mode = True
            self.oscillation_base_krot = K_rot.copy()
            self.frame_count = 0
        else:
            self.oscillation_mode = False
            self.oscillation_base_krot = None

        print(f"Preset: {preset['name']} — {preset['description']}")

    def _adjust_krot(self, delta):
        """Adjust all K_rot values."""
        self.alignment_matrix += delta
        self.alignment_matrix = np.clip(self.alignment_matrix, -1.0, 1.0)
        print(f"K_rot adjusted by {delta:+.05f}, max={self.alignment_matrix.max():.2f}")

    def _adjust_kpos_cross(self, delta):
        """Adjust off-diagonal K_pos values."""
        n = self.n_species
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.matrix[i, j] = np.clip(self.matrix[i, j] + delta, -1.0, 1.0)
        if n > 1:
            print(f"K_pos cross adjusted by {delta:+.05f}, val={self.matrix[0,1]:.2f}")

    def step(self):
        """Override to handle oscillation — flip K_rot sign periodically."""
        if self.oscillation_mode and self.oscillation_base_krot is not None and not self.paused:
            self.frame_count += 1
            # Sinusoidal oscillation: K_rot = base * sin(2π * t / period)
            phase = np.sin(2 * np.pi * self.frame_count / (2 * self.oscillation_period))
            self.alignment_matrix[:] = self.oscillation_base_krot * phase
        super().step()

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.draw_particles()

        if self.hide_gui:
            return

        if self.show_info:
            self._draw_info()
        self.draw_pause_indicator()

    def _draw_matrix_heatmap(self, matrix, label, x0, y0, cell_size=40):
        """Draw a matrix as a colored heatmap with values."""
        n = matrix.shape[0]
        grey = (80, 80, 80)

        # Label
        lbl = self.font.render(label, True, grey)
        self.screen.blit(lbl, (x0, y0))
        y0 += 20

        # Species color indicators (top)
        for j in range(n):
            cx = x0 + 15 + j * cell_size + cell_size // 2
            pygame.draw.circle(self.screen, self.colors[j % len(self.colors)], (cx, y0), 5)
        y0 += 12

        for i in range(n):
            # Row color indicator
            pygame.draw.circle(self.screen, self.colors[i % len(self.colors)],
                               (x0 + 6, y0 + cell_size // 2), 5)
            for j in range(n):
                x = x0 + 15 + j * cell_size
                y = y0
                val = matrix[i, j]

                # Color: green=positive, red=negative, gray=zero
                if val > 0.01:
                    intensity = int(min(255, abs(val) * 255))
                    color = (0, intensity, 0)
                elif val < -0.01:
                    intensity = int(min(255, abs(val) * 255))
                    color = (intensity, 0, 0)
                else:
                    color = (200, 200, 200)

                pygame.draw.rect(self.screen, color,
                                 (x, y, cell_size - 2, cell_size - 2))
                pygame.draw.rect(self.screen, (160, 160, 160),
                                 (x, y, cell_size - 2, cell_size - 2), 1)

                # Value text
                txt = self.font.render(f"{val:+.2f}", True, (255, 255, 255))
                tr = txt.get_rect(center=(x + cell_size // 2 - 1, y + cell_size // 2 - 1))
                self.screen.blit(txt, tr)

            y0 += cell_size

        return y0

    def _draw_info(self):
        # Text info (top left)
        lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Preset: {self.preset_name}",
            f"Species: {self.n_species}  Particles: {self.n}",
            "",
            "1-6: Presets",
            "Up/Dn: K_rot  L/R: K_pos cross",
            "R: Reset  I: Info  Space: Pause",
        ]
        y = 10
        for line in lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 20

        # Matrix heatmaps (top right)
        n = self.n_species
        cell_size = min(40, 200 // max(n, 1))
        x0 = self.config.width - 15 - n * cell_size - 20
        y_after = self._draw_matrix_heatmap(self.matrix, "K_pos:", x0, 10, cell_size)
        self._draw_matrix_heatmap(self.alignment_matrix, "K_rot:", x0, y_after + 10, cell_size)

    def _print_help(self):
        print("=" * 60)
        print("Microrobot Collectives — Behavior Reproduction")
        print("=" * 60)
        print("  1   Rotation (tight)")
        print("  2   Rotation (medium)")
        print("  3   Rotation (wide)")
        print("  4   Chain (connected)")
        print("  5   Chain (separated)")
        print("  6   Oscillation")
        print("  ↑/↓ Adjust K_rot strength")
        print("  ←/→ Adjust K_pos cross-attraction")
        print("  R   Reset positions")
        print("=" * 60)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._scatter()
                    print("Reset")
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_h:
                    self.hide_gui = not self.hide_gui

                # Presets
                elif event.key == pygame.K_1:
                    self._load_preset('1_rotation_tight')
                elif event.key == pygame.K_2:
                    self._load_preset('2_rotation_medium')
                elif event.key == pygame.K_3:
                    self._load_preset('3_rotation_wide')
                elif event.key == pygame.K_4:
                    self._load_preset('4_chain_connected')
                elif event.key == pygame.K_5:
                    self._load_preset('5_chain_separated')
                elif event.key == pygame.K_6:
                    self._load_preset('6_oscillation')

                # Tuning
                elif event.key == pygame.K_UP:
                    self._adjust_krot(0.05)
                elif event.key == pygame.K_DOWN:
                    self._adjust_krot(-0.05)
                elif event.key == pygame.K_RIGHT:
                    self._adjust_kpos_cross(0.05)
                elif event.key == pygame.K_LEFT:
                    self._adjust_kpos_cross(-0.05)

        return True

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.step()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()


def main():
    demo = MicrorobotDemo()
    demo.run()


if __name__ == '__main__':
    main()
