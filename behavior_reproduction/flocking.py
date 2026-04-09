#!/usr/bin/env python3
"""
Flocking Behavior Reproduction — Reynolds Boids via Particle Life

Reproduce the three classic Reynolds flocking rules using only K_pos and K_rot:
1. Separation — avoid crowding neighbors (Zone 1 repulsion)
2. Aggregation — steer toward neighbors (K_pos positive attraction)
3. Cohesion — combined flocking with velocity alignment (K_rot coupling)

The workspace uses toroidal wrapping (no walls — particles wrap around edges).

Reference: Craig W. Reynolds, "Flocks, Herds, and Schools: A Distributed
Behavioral Model" (SIGGRAPH 1987)

Controls:
    1:     Separation only (repulsion, particles spread out)
    2:     Aggregation only (attraction, particles cluster)
    3:     Cohesion / flocking (attraction + alignment via K_rot)
    4:     Cohesion + separation (full Reynolds-like behavior)
    ↑/↓:   Adjust K_pos (attraction strength)
    ←/→:   Adjust K_rot (alignment / rotation strength)
    B:     Adjust beta (repulsion zone size)
    R:     Reset (random positions and velocities)
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
# PRESETS
# ============================================================

N_SPECIES = 10
N_PARTICLES_PER = 15  # 150 total


def make_kpos_chain(diag, adjacent, forward_bias=0.0):
    """Build chain K_pos: diagonal + adjacent attraction with forward bias.

    Same mechanism as snake_demo: asymmetric K_pos creates drift along chain.
    K[i, i-1] = adjacent (strong, toward tail)
    K[i, i+1] = adjacent - forward_bias (weaker, toward head)
    → net drift toward head = continuous forward motion.
    """
    K = np.zeros((N_SPECIES, N_SPECIES))
    for i in range(N_SPECIES):
        K[i, i] = diag
        if i > 0:
            K[i, i - 1] = adjacent  # toward tail (strong)
        if i < N_SPECIES - 1:
            K[i, i + 1] = adjacent - forward_bias  # toward head (weaker)
    return K


def make_krot_antisymmetric(strength):
    """Antisymmetric K_rot between adjacent species → forward translation."""
    K = np.zeros((N_SPECIES, N_SPECIES))
    for i in range(N_SPECIES - 1):
        K[i, i + 1] = +strength
        K[i + 1, i] = -strength
    return K


PRESETS = {
    '1_separation': {
        'name': 'Separation',
        'K_pos': make_kpos_chain(diag=0.4, adjacent=-0.1, forward_bias=0.0),
        'K_rot': np.zeros((N_SPECIES, N_SPECIES)),
        'beta': 0.5,
        'force_scale': 1.0,
        'description': 'Repulsion between species — groups spread apart',
    },
    '2_aggregation': {
        'name': 'Aggregation',
        'K_pos': make_kpos_chain(diag=0.5, adjacent=0.3, forward_bias=0.0),
        'K_rot': np.zeros((N_SPECIES, N_SPECIES)),
        'beta': 0.2,
        'force_scale': 0.5,
        'description': 'Attraction between adjacent species — groups merge',
    },
    '3_cohesion': {
        'name': 'Cohesion (flocking)',
        'K_pos': make_kpos_chain(diag=0.5, adjacent=0.3, forward_bias=0.1),
        'K_rot': make_krot_antisymmetric(0.3),
        'beta': 0.3,
        'force_scale': 0.5,
        'description': 'Forward bias K_pos + antisymmetric K_rot — collective motion',
    },
    '4_full_reynolds': {
        'name': 'Full Reynolds',
        'K_pos': make_kpos_chain(diag=0.5, adjacent=0.2, forward_bias=0.1),
        'K_rot': make_krot_antisymmetric(0.3),
        'beta': 0.4,
        'force_scale': 0.8,
        'description': 'Separation + cohesion + two-channel locomotion',
    },
}


# ============================================================
# DEMO
# ============================================================

class FlockingDemo(ParticleLife):
    """Reproduce Reynolds flocking using particle life with toroidal wrapping."""

    def __init__(self):
        preset = PRESETS['1_separation']
        self.current_preset = '1_separation'
        self.preset_name = preset['name']

        sim_w, sim_h = 15.0, 15.0

        config = Config(
            n_species=N_SPECIES,
            n_particles=N_PARTICLES_PER,
            sim_width=sim_w,
            sim_height=sim_h,
            r_max=3.0,
            beta=preset['beta'],
            force_scale=preset['force_scale'],
            max_speed=2.0,
            a_rot=1.0,
            far_attraction=0.0,
            position_matrix=preset['K_pos'].tolist(),
            orientation_matrix=preset['K_rot'].tolist(),
        )

        super().__init__(config, headless=False)

        self.hide_gui = False
        self._scatter_with_velocity()

        pygame.display.set_caption("Flocking — Reynolds Boids Reproduction")
        self._print_help()

    def _scatter_with_velocity(self):
        """Random positions and random initial velocities."""
        sw, sh = self.config.sim_width, self.config.sim_height
        self.positions[:, 0] = np.random.uniform(0, sw, self.n)
        self.positions[:, 1] = np.random.uniform(0, sh, self.n)
        # Random initial velocities (boids start moving)
        angles = np.random.uniform(0, 2 * np.pi, self.n)
        speed = 0.5
        self.velocities[:, 0] = speed * np.cos(angles)
        self.velocities[:, 1] = speed * np.sin(angles)

    def _load_preset(self, key):
        if key not in PRESETS:
            return
        preset = PRESETS[key]
        self.current_preset = key
        self.preset_name = preset['name']

        self.matrix[:] = preset['K_pos']
        self.alignment_matrix[:] = preset['K_rot']
        self.config.beta = preset['beta']
        self.config.force_scale = preset['force_scale']

        print(f"Preset: {preset['name']} — {preset['description']}")

    def step(self):
        """Standard particle life step with toroidal wrapping."""
        if self.paused:
            return

        # Same velocity computation as particle_life.py
        self.velocities = self.compute_velocities()

        # Clamp speed
        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = np.where(
            speed > self.config.max_speed,
            self.velocities * self.config.max_speed / speed,
            self.velocities
        )

        # Update positions
        self.positions += self.velocities * self.config.dt

        # Toroidal wrapping (instead of reflection)
        sw, sh = self.config.sim_width, self.config.sim_height
        self.positions[:, 0] = self.positions[:, 0] % sw
        self.positions[:, 1] = self.positions[:, 1] % sh

    def draw(self):
        self.screen.fill((255, 255, 255))

        # Draw particles (standard particle life circles)
        self.draw_particles()

        if self.hide_gui:
            return

        if self.show_info:
            self._draw_matrix_heatmap()
            self._draw_info()
        self.draw_pause_indicator()

    def _draw_matrix_heatmap(self):
        """Draw K_pos and K_rot as colored matrix cells (top right)."""
        grey = (80, 80, 80)
        n = self.n_species
        # Scale cell size to fit — smaller for more species
        cell_size = max(12, min(30, 300 // max(n, 1)))
        mat_width = 15 + n * cell_size + 10
        x0 = self.config.width - mat_width

        # Calculate vertical positions so they don't overlap
        mat_height = 20 + 12 + n * cell_size + 10
        y_kpos = 10
        y_krot = y_kpos + mat_height

        for mat, label, y0 in [(self.matrix, "K_pos:", y_kpos),
                                (self.alignment_matrix, "K_rot:", y_krot)]:
            lbl = self.font.render(label, True, grey)
            self.screen.blit(lbl, (x0, y0))
            y0 += 20

            for j in range(n):
                cx = x0 + 15 + j * cell_size + cell_size // 2
                pygame.draw.circle(self.screen, self.colors[j % len(self.colors)], (cx, y0), 5)
            y0 += 12

            for i in range(n):
                pygame.draw.circle(self.screen, self.colors[i % len(self.colors)],
                                   (x0 + 6, y0 + cell_size // 2), 5)
                for j in range(n):
                    x = x0 + 15 + j * cell_size
                    y = y0
                    val = mat[i, j]

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

                    # Only show value text if cells are large enough
                    if cell_size >= 25:
                        small_font = pygame.font.Font(None, max(12, cell_size // 2))
                        txt = small_font.render(f"{val:+.1f}", True, (255, 255, 255))
                        tr = txt.get_rect(center=(x + cell_size // 2 - 1, y + cell_size // 2 - 1))
                        self.screen.blit(txt, tr)
                y0 += cell_size

    def _draw_info(self):
        lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Preset: {self.preset_name}",
            f"Species: {N_SPECIES}  Particles: {self.n}",
            f"K_pos diag: {self.matrix[0,0]:.2f}  off: {self.matrix[0,1]:.2f}",
            f"K_rot max: {self.alignment_matrix.max():.2f}",
            f"Beta: {self.config.beta:.2f}",
            "",
            "1: Separation  2: Aggregation",
            "3: Cohesion    4: Full Reynolds",
            "Up/Dn: K_pos cross  L/R: K_rot",
            "B: Beta  R: Reset",
        ]
        y = 10
        for line in lines:
            if line:
                text = self.font.render(line, True, (180, 180, 180))
                self.screen.blit(text, (10, y))
            y += 20

    def _print_help(self):
        print("=" * 60)
        print("Flocking — Reynolds Boids Reproduction")
        print("=" * 60)
        print("  1   Separation (repulsion only)")
        print("  2   Aggregation (attraction only)")
        print("  3   Cohesion / flocking (attraction + alignment)")
        print("  4   Full Reynolds (separation + cohesion + alignment)")
        print("  ↑/↓ K_pos (attraction)")
        print("  ←/→ K_rot (alignment)")
        print("  R   Reset with random velocities")
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
                    self._scatter_with_velocity()
                    print("Reset")
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_h:
                    self.hide_gui = not self.hide_gui

                # Presets
                elif event.key == pygame.K_1:
                    self._load_preset('1_separation')
                elif event.key == pygame.K_2:
                    self._load_preset('2_aggregation')
                elif event.key == pygame.K_3:
                    self._load_preset('3_cohesion')
                elif event.key == pygame.K_4:
                    self._load_preset('4_full_reynolds')

                # Tuning — adjust off-diagonal K_pos uniformly
                elif event.key == pygame.K_UP:
                    n = N_SPECIES
                    for i in range(n):
                        for j in range(n):
                            if i != j:
                                self.matrix[i, j] = min(1.0, self.matrix[i, j] + 0.05)
                    print(f"K_pos off-diag: {self.matrix[0,1]:.2f}")
                elif event.key == pygame.K_DOWN:
                    n = N_SPECIES
                    for i in range(n):
                        for j in range(n):
                            if i != j:
                                self.matrix[i, j] = max(-1.0, self.matrix[i, j] - 0.05)
                    print(f"K_pos off-diag: {self.matrix[0,1]:.2f}")
                # Adjust K_rot strength (scale all nonzero entries)
                elif event.key == pygame.K_RIGHT:
                    self.alignment_matrix *= 1.1
                    self.alignment_matrix = np.clip(self.alignment_matrix, -1.0, 1.0)
                    print(f"K_rot scaled up, max={self.alignment_matrix.max():.2f}")
                elif event.key == pygame.K_LEFT:
                    self.alignment_matrix *= 0.9
                    print(f"K_rot scaled down, max={self.alignment_matrix.max():.2f}")
                elif event.key == pygame.K_b:
                    self.config.beta = min(0.8, self.config.beta + 0.05)
                    print(f"Beta: {self.config.beta:.2f}")

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
    demo = FlockingDemo()
    demo.run()


if __name__ == '__main__':
    main()
