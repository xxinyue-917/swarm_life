#!/usr/bin/env python3
"""
Flocking Behavior Reproduction — Boids Aggregation via Per-Agent Chase Network

Reproduce Reynolds-boids-style aggregation using 1 particle per species and
50 species. Each K_pos row encodes WHICH specific agents that agent chases.
The chase-network topology IS the behavior.

Design: each agent chases k=5 targets spread uniformly across the population
(offsets N/k, 2N/k, ..., (k-1)N/k mod N). This makes the graph k-regular in
both directions — every agent has k targets AND is chased by k others. Every
agent experiences k inward-pulling forces from different directions, which
balance near the population centroid → cohesive aggregated flock, matching
Reynolds' aggregation rule.

Workspace uses toroidal wrapping.

Reference: Craig W. Reynolds, "Flocks, Herds, and Schools" (SIGGRAPH 1987)

Controls:
    R:     Reset (random positions and velocities)
    I:     Toggle info
    V:     Toggle matrix display
    M:     Matrix edit mode (WASD navigate, E/X adjust)
    ↑/↓:   Scale nonzero K_pos entries
    B:     Adjust beta (repulsion zone size)
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
# BOIDS AGGREGATION NETWORK
# ============================================================

N_SPECIES = 50        # 50 species, 1 agent each
N_PARTICLES = 1
K_TARGETS = 5         # each agent chases k targets spread across population


def make_aggregation_network(n, k=5, cohesion=0.4, flow=1.0, flow_reach=3):
    """Asymmetric chase network — produces cohesive flock WITH directional drift.

    Two channels in ONE K_pos matrix:

      1. Cohesion (symmetric, weak): every agent has k chase targets at offsets
         round(m·N/(k+1)) mod N for m=1..k. Distributed inward pulls from all
         directions → keeps the flock clumped.

      2. Flow (asymmetric, strong): every agent strongly chases the next few
         agents in id-space (offsets +1, +2, ..., +flow_reach) — but NOT the
         ones behind (offsets -1, -2, ...). This breaks symmetry.

    Why this produces coherent drift:
      - Each agent i is pulled toward agents (i+1)..(i+flow_reach)
      - Agent i is pulled BY agents (i-1)..(i-flow_reach), but those forces
        act on i-1..i-flow_reach, not on i. So i feels net force only from
        its forward targets.
      - As the flock moves, agent 0's position becomes "wherever agent 1 is"
        becomes "wherever agent 2 is"... creating a traveling wave through
        id-space that manifests as coherent flock motion in real space.

    Same mechanism as snake_demo.py's `make_kpos_chain(forward_bias=...)`,
    but extended over a chase graph of 50 agents instead of 10.
    """
    K = np.zeros((n, n))

    # Channel 1: weak symmetric cohesion
    offsets = [round(m * n / (k + 1)) for m in range(1, k + 1)]
    offsets = sorted(set(o % n for o in offsets if o % n != 0))
    for i in range(n):
        for d in offsets:
            j = (i + d) % n
            K[i, j] = cohesion

    # Channel 2: strong ASYMMETRIC forward chase — each agent pulled toward
    # the next few ids only. Overwrites cohesion at those edges with a higher value.
    # Use linearly decreasing strength so nearest forward neighbor pulls hardest.
    for i in range(n):
        for d in range(1, flow_reach + 1):
            j = (i + d) % n
            # Strongest pull at d=1, linearly decreasing
            K[i, j] = flow * (1.0 - (d - 1) / flow_reach)

    return K


PRESET = {
    'name': 'Boids Aggregation + Flow',
    'K_pos': make_aggregation_network(N_SPECIES, K_TARGETS, cohesion=0.6, flow=0.0, flow_reach=1),
    'K_rot': np.zeros((N_SPECIES, N_SPECIES)),
    'beta': 0.12,
    'force_scale': 1.0,
    'description': (
        f'{N_SPECIES} agents. K_pos k={K_TARGETS} symmetric chase network → cohesive aggregation. '
        'Intrinsic forward drift (like real boids) gives the flock a coherent direction.'
    ),
}

# Intrinsic per-agent forward speed (like Reynolds boids and Vicsek particles).
# This is a post-force VELOCITY term, not a physics modification — each agent
# has its own preferred forward direction (set at init from random angles) and
# cruises at this speed in addition to whatever forces pull it.
INTRINSIC_SPEED = 0.6


# ============================================================
# DEMO
# ============================================================

class FlockingDemo(ParticleLife):
    """Boids aggregation via a per-agent k-regular chase network."""

    def __init__(self):
        self.preset_name = PRESET['name']

        sim_w, sim_h = 15.0, 15.0

        config = Config(
            n_species=N_SPECIES,
            n_particles=N_PARTICLES,
            sim_width=sim_w,
            sim_height=sim_h,
            r_max=8.0,   # large enough that all 5 chase targets are reachable across the swarm
            beta=PRESET['beta'],
            force_scale=PRESET['force_scale'],
            max_speed=2.0,
            a_rot=1.0,
            far_attraction=0.0,
            position_matrix=PRESET['K_pos'].tolist(),
            orientation_matrix=PRESET['K_rot'].tolist(),
        )

        super().__init__(config, headless=False)

        self.hide_gui = False
        self.show_matrix = True
        self.matrix_edit_mode = False
        self._tiny_font = None
        self._tiny_font_size = 0
        self.edit_row = 0
        self.edit_col = 0
        self.editing_k_rot = False
        self._scatter_with_velocity()

        pygame.display.set_caption("Flocking — Boids Aggregation via Chase Network")
        self._print_help()

    def _scatter_with_velocity(self):
        """Random positions in a centered cloud. Assign each agent a persistent
        forward heading (used by the intrinsic-drift term in step)."""
        sw, sh = self.config.sim_width, self.config.sim_height
        cloud_size = min(sw, sh) * 0.3
        cx, cy = sw / 2, sh / 2
        self.positions[:, 0] = np.random.uniform(cx - cloud_size, cx + cloud_size, self.n)
        self.positions[:, 1] = np.random.uniform(cy - cloud_size, cy + cloud_size, self.n)
        # Pick ONE shared heading for all agents — they'll drift as a coherent flock
        # in a random direction each reset. K_pos cohesion holds them together.
        shared_angle = np.random.uniform(0, 2 * np.pi)
        self.heading = np.array([np.cos(shared_angle), np.sin(shared_angle)], dtype=np.float64)
        self.velocities[:] = 0

    def step(self):
        """Standard particle life step with toroidal wrapping + intrinsic boids drift."""
        if self.paused:
            return

        self.velocities = self.compute_velocities()

        # Reynolds/Vicsek-style intrinsic forward speed: every agent has a preferred
        # cruise velocity. Combined with K_pos cohesion (which holds the flock
        # together), the group drifts coherently in a single direction.
        self.velocities += INTRINSIC_SPEED * self.heading

        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        speed_safe = np.where(speed > 1e-8, speed, 1.0)
        self.velocities = np.where(
            speed > self.config.max_speed,
            self.velocities * self.config.max_speed / speed_safe,
            self.velocities
        )

        self.positions += self.velocities * self.config.dt

        sw, sh = self.config.sim_width, self.config.sim_height
        self.positions[:, 0] = self.positions[:, 0] % sw
        self.positions[:, 1] = self.positions[:, 1] % sh

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.draw_particles()

        if self.hide_gui:
            return

        if self.show_matrix:
            self._draw_matrix_heatmap()

        if self.show_info:
            self._draw_info()
        self.draw_pause_indicator()

    def _draw_matrix_heatmap(self):
        """Draw K_pos matrix (top right). K_rot hidden (always zero here)."""
        grey = (80, 80, 80)
        n = self.n_species
        cell_size = max(6, min(30, 300 // max(n, 1)))
        mat_width = 15 + n * cell_size + 10
        x0 = self.config.width - mat_width

        y0 = 10
        lbl_text = "K_pos:" + (" (EDIT)" if self.matrix_edit_mode else "")
        lbl_color = (100, 100, 200) if self.matrix_edit_mode else grey
        lbl = self.font.render(lbl_text, True, lbl_color)
        self.screen.blit(lbl, (x0, y0))
        y0 += 20

        # Column color markers (skip for large n to save space)
        if cell_size >= 10:
            for j in range(n):
                cx = x0 + 15 + j * cell_size + cell_size // 2
                pygame.draw.circle(self.screen, self.colors[j % len(self.colors)], (cx, y0), 3)
            y0 += 8

        mat = self.matrix
        for i in range(n):
            if cell_size >= 10:
                pygame.draw.circle(self.screen, self.colors[i % len(self.colors)],
                                   (x0 + 6, y0 + cell_size // 2), 3)
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
                    color = (220, 220, 220)

                pygame.draw.rect(self.screen, color,
                                 (x, y, cell_size - 1, cell_size - 1))
                if self.matrix_edit_mode and i == self.edit_row and j == self.edit_col:
                    pygame.draw.rect(self.screen, (255, 255, 0),
                                     (x, y, cell_size - 1, cell_size - 1), 2)
            y0 += cell_size

    def _draw_info(self):
        lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Preset: {self.preset_name}",
            f"Species: {self.n_species}  Particles: {self.n}",
            f"K_pos chase strength: {np.max(self.matrix):.2f}",
            f"Targets per agent: {K_TARGETS}",
            f"Beta: {self.config.beta:.2f}",
            "",
            "Up/Dn: scale K_pos   B: beta",
            "V: matrix  M: edit  R: reset",
            "TAB switches — but only K_pos is used here",
        ]
        y = 10
        for line in lines:
            if line:
                text = self.font.render(line, True, (180, 180, 180))
                self.screen.blit(text, (10, y))
            y += 20

    def _print_help(self):
        print("=" * 60)
        print("Flocking — Boids Aggregation (k-regular chase network)")
        print("=" * 60)
        print(f"  {N_SPECIES} species, 1 particle each, k={K_TARGETS} targets per agent")
        print("  Targets at uniform offsets across population id-space")
        print("  → every agent feels k inward pulls → cohesive aggregation")
        print()
        print("  ↑/↓ scale K_pos strength")
        print("  B   adjust beta (repulsion zone)")
        print("  R   reset with random positions + velocities")
        print("  V   toggle matrix display")
        print("  M   matrix edit mode (WASD navigate, E/X +/-)")
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
                    print(f"Reset. Heading = ({self.heading[0]:.2f}, {self.heading[1]:.2f})")
                elif event.key == pygame.K_n:
                    # Pick a new drift heading without resetting positions
                    angle = np.random.uniform(0, 2 * np.pi)
                    self.heading = np.array([np.cos(angle), np.sin(angle)])
                    print(f"New heading: ({self.heading[0]:.2f}, {self.heading[1]:.2f})")
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_h:
                    self.hide_gui = not self.hide_gui
                elif event.key == pygame.K_v:
                    self.show_matrix = not self.show_matrix

                # Matrix editing
                elif event.key == pygame.K_m:
                    self.matrix_edit_mode = not self.matrix_edit_mode
                    print(f"Matrix edit: {'ON' if self.matrix_edit_mode else 'OFF'}")
                elif self.matrix_edit_mode:
                    if event.key == pygame.K_w:
                        self.edit_row = max(0, self.edit_row - 1)
                    elif event.key == pygame.K_s:
                        self.edit_row = min(self.n_species - 1, self.edit_row + 1)
                    elif event.key == pygame.K_a:
                        self.edit_col = max(0, self.edit_col - 1)
                    elif event.key == pygame.K_d:
                        self.edit_col = min(self.n_species - 1, self.edit_col + 1)
                    elif event.key in (pygame.K_e, pygame.K_EQUALS):
                        self.matrix[self.edit_row, self.edit_col] = min(
                            1.0, self.matrix[self.edit_row, self.edit_col] + 0.1)
                        print(f"[{self.edit_row},{self.edit_col}] = {self.matrix[self.edit_row, self.edit_col]:.2f}")
                    elif event.key in (pygame.K_x, pygame.K_MINUS):
                        self.matrix[self.edit_row, self.edit_col] = max(
                            -1.0, self.matrix[self.edit_row, self.edit_col] - 0.1)
                        print(f"[{self.edit_row},{self.edit_col}] = {self.matrix[self.edit_row, self.edit_col]:.2f}")

                # Scale nonzero K_pos entries
                elif event.key == pygame.K_UP:
                    off_diag = ~np.eye(self.n_species, dtype=bool)
                    nonzero_mask = off_diag & (np.abs(self.matrix) > 1e-6)
                    self.matrix[nonzero_mask] = np.clip(self.matrix[nonzero_mask] * 1.1, -1.0, 1.0)
                    print(f"K_pos scaled up, max={np.abs(self.matrix).max():.2f}")
                elif event.key == pygame.K_DOWN:
                    off_diag = ~np.eye(self.n_species, dtype=bool)
                    nonzero_mask = off_diag & (np.abs(self.matrix) > 1e-6)
                    self.matrix[nonzero_mask] = self.matrix[nonzero_mask] * 0.9
                    print(f"K_pos scaled down, max={np.abs(self.matrix).max():.2f}")
                elif event.key == pygame.K_b:
                    self.config.beta = min(0.8, self.config.beta + 0.05)
                    print(f"Beta: {self.config.beta:.2f}")

        # Held keys — continuous matrix editing
        if self.matrix_edit_mode:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_e] or keys[pygame.K_EQUALS]:
                self.matrix[self.edit_row, self.edit_col] = min(
                    1.0, self.matrix[self.edit_row, self.edit_col] + 0.02)
            if keys[pygame.K_x] or keys[pygame.K_MINUS]:
                self.matrix[self.edit_row, self.edit_col] = max(
                    -1.0, self.matrix[self.edit_row, self.edit_col] - 0.02)

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
