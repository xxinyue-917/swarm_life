#!/usr/bin/env python3
"""
Formation Fill Demo

Fills 2D shapes with many coloured particle species.
Each species (small cluster) occupies a target position inside the shape.
Spring forces pull every particle toward its species' target while
short-range repulsion prevents overlap.

Shapes: Mickey Mouse, Heart, Star, Circle, Snowman

Controls:
    1-5:   Switch shape (particles animate to new positions)
    S:     Scatter particles randomly
    R:     Reset particles to target positions
    T:     Toggle target markers
    +/-:   Adjust spring strength
    [/]:   Decrease / increase grid density (rebuild)
    C:     Cycle colour scheme
    SPACE: Pause / Resume
    I:     Toggle info panel
    Q/ESC: Quit
"""

import pygame
import pygame.gfxdraw
import numpy as np
import colorsys
from particle_life import Config, ParticleLife


# ── Shape testers ───────────────────────────────────────────────────

def _in_circle(px, py, cx, cy, r):
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r


def shape_mickey(px, py, sw, sh):
    cx, cy = sw / 2, sh / 2
    if _in_circle(px, py, cx, cy + 0.8, 2.0):
        return True
    if _in_circle(px, py, cx - 1.7, cy - 1.2, 1.1):
        return True
    if _in_circle(px, py, cx + 1.7, cy - 1.2, 1.1):
        return True
    return False


def shape_heart(px, py, sw, sh):
    x = (px - sw / 2) / 2.3
    y = -(py - sh / 2) / 2.3 - 0.15
    return (x ** 2 + y ** 2 - 1) ** 3 - x ** 2 * y ** 3 <= 0


def shape_star(px, py, sw, sh):
    cx, cy = sw / 2, sh / 2
    dx, dy = px - cx, py - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)
    outer, inner = 2.8, 1.1
    sector = (theta + np.pi) % (2 * np.pi / 5)
    half = np.pi / 5
    if sector < half:
        boundary = outer - (outer - inner) * sector / half
    else:
        boundary = inner + (outer - inner) * (sector - half) / half
    return r <= boundary


def shape_circle(px, py, sw, sh):
    return _in_circle(px, py, sw / 2, sh / 2, 2.8)


def shape_snowman(px, py, sw, sh):
    cx = sw / 2
    cy = sh / 2
    if _in_circle(px, py, cx, cy + 2.0, 1.6):
        return True
    if _in_circle(px, py, cx, cy, 1.2):
        return True
    if _in_circle(px, py, cx, cy - 1.8, 0.85):
        return True
    return False


SHAPES = {
    1: ("Mickey Mouse", shape_mickey),
    2: ("Heart",        shape_heart),
    3: ("Star",         shape_star),
    4: ("Circle",       shape_circle),
    5: ("Snowman",      shape_snowman),
}


# ── Sampling & colouring ────────────────────────────────────────────

def sample_targets(shape_fn, sw, sh, spacing):
    """Return (M, 2) array of grid points inside *shape_fn*."""
    pts = []
    margin = 0.3
    y = margin
    while y < sh - margin:
        x = margin
        while x < sw - margin:
            if shape_fn(x, y, sw, sh):
                pts.append([x, y])
            x += spacing
        y += spacing
    if not pts:
        pts = [[sw / 2, sh / 2], [sw / 2 + 0.1, sh / 2]]
    return np.array(pts)


def _hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


COLOR_SCHEMES = ["rainbow", "gradient", "cluster"]


def make_colors(targets, sw, sh, scheme="rainbow"):
    cx, cy = sw / 2, sh / 2
    colors = []
    for tx, ty in targets:
        if scheme == "rainbow":
            angle = np.arctan2(ty - cy, tx - cx)
            hue = (angle + np.pi) / (2 * np.pi)
            colors.append(_hsv(hue, 0.85, 0.92))
        elif scheme == "gradient":
            t = ty / sh
            colors.append(_hsv(0.6 - 0.6 * t, 0.8, 0.95))
        else:  # cluster
            dist = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
            hue = dist / 4.0
            colors.append(_hsv(hue, 0.9, 0.9))
    return colors


# ── Demo ─────────────────────────────────────────────────────────────

class FormationFillDemo(ParticleLife):
    """Fill 2D shapes with coloured particle clusters."""

    PPS = 3  # particles per species

    def __init__(self, shape_id=1, spacing=0.45):
        self.shape_id = shape_id
        self.spacing = spacing
        self.show_targets = True
        self.spring_k = 2.0
        self.color_scheme = 0  # index into COLOR_SCHEMES

        sw, sh = 10.0, 10.0
        _, shape_fn = SHAPES[shape_id]
        targets = sample_targets(shape_fn, sw, sh, spacing)
        n_sp = len(targets)

        config = Config(
            n_particles=self.PPS,
            n_species=n_sp,
            sim_width=sw,
            sim_height=sh,
            r_max=1.0,
            beta=0.3,
            force_scale=0.5,
            max_speed=0.8,
            far_attraction=0.0,
            a_rot=0.0,
        )

        super().__init__(config, headless=False)
        pygame.display.set_caption("Formation Fill Demo")

        self.targets = targets
        self._apply_colors()

        # start scattered → particles flow into shape
        self._scatter()

        self._print_status()

    # ── colour helpers ───────────────────────────────────────────────

    def _apply_colors(self):
        scheme = COLOR_SCHEMES[self.color_scheme]
        self.target_colors = make_colors(
            self.targets, self.config.sim_width, self.config.sim_height, scheme
        )
        self.colors = self.target_colors

    # ── rebuild ──────────────────────────────────────────────────────

    def _rebuild(self, shape_id):
        self.shape_id = shape_id
        _, shape_fn = SHAPES[shape_id]
        sw = self.config.sim_width
        sh = self.config.sim_height

        targets = sample_targets(shape_fn, sw, sh, self.spacing)
        n_sp = len(targets)
        n_total = n_sp * self.PPS

        self.n_species = n_sp
        self.config.n_species = n_sp
        self.n = n_total

        self.positions = np.random.uniform(0.5, sw - 0.5, (n_total, 2))
        self.velocities = np.zeros((n_total, 2))
        self.orientations = np.zeros(n_total)

        self.species = np.repeat(np.arange(n_sp), self.PPS)

        self.matrix = np.zeros((n_sp, n_sp))
        self.alignment_matrix = np.zeros((n_sp, n_sp))

        self.targets = targets
        self._apply_colors()
        self._print_status()

    def _print_status(self):
        name = SHAPES[self.shape_id][0]
        print(f"Shape: {name} | {self.n_species} species × {self.PPS} = {self.n} particles | spacing {self.spacing:.2f}")

    def _switch_shape(self, shape_id):
        if shape_id not in SHAPES or shape_id == self.shape_id:
            return
        self._rebuild(shape_id)

    # ── particle placement ───────────────────────────────────────────

    def _scatter(self):
        m = 0.5
        sw, sh = self.config.sim_width, self.config.sim_height
        self.positions[:, 0] = np.random.uniform(m, sw - m, self.n)
        self.positions[:, 1] = np.random.uniform(m, sh - m, self.n)
        self.velocities[:] = 0

    def _reset_to_targets(self):
        for i in range(self.n):
            t = self.targets[self.species[i]]
            self.positions[i] = t + np.random.uniform(-0.08, 0.08, 2)
        self.velocities[:] = 0

    # ── physics ──────────────────────────────────────────────────────

    def compute_velocities(self):
        """Spring toward targets + vectorised short-range repulsion."""
        pos = self.positions
        r_max = self.config.r_max
        beta = self.config.beta
        fs = self.config.force_scale

        # spring: v = k * (target − pos)
        target_pos = self.targets[self.species]
        spring = self.spring_k * (target_pos - pos)

        # pairwise repulsion (zone 1 only)
        delta = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (n,n,2)
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)

        r_norm = dist / r_max
        mask = r_norm < beta
        F = np.where(mask, r_norm / beta - 1.0, 0.0)

        inv_d = 1.0 / (dist + 1e-8)
        rhat = delta * inv_d[:, :, np.newaxis]
        repulsion = fs * (F[:, :, np.newaxis] * rhat).sum(axis=1)

        return spring + repulsion

    def step(self):
        if self.paused:
            return

        self.velocities = self.compute_velocities()

        # damping
        self.velocities *= 0.90

        # speed clamp
        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        too_fast = speed > self.config.max_speed
        self.velocities = np.where(
            too_fast,
            self.velocities * self.config.max_speed / speed,
            self.velocities,
        )

        self.positions += self.velocities * self.config.dt

        # boundary reflection
        margin = 0.05
        for ax in range(2):
            hi = (self.config.sim_width if ax == 0 else self.config.sim_height) - margin
            lo_mask = self.positions[:, ax] < margin
            self.positions[lo_mask, ax] = margin
            self.velocities[lo_mask, ax] = abs(self.velocities[lo_mask, ax])
            hi_mask = self.positions[:, ax] > hi
            self.positions[hi_mask, ax] = hi
            self.velocities[hi_mask, ax] = -abs(self.velocities[hi_mask, ax])

    # ── drawing ──────────────────────────────────────────────────────

    def draw(self):
        self.screen.fill((255, 255, 255))  # White background (matches particle_life)

        # target dots (faded)
        if self.show_targets:
            rt = max(2, int(0.025 * self.ppu * self.zoom))
            for si in range(len(self.targets)):
                sx, sy = self.to_screen(self.targets[si])
                c = self.target_colors[si]
                faded = (
                    c[0] // 4 + 190,
                    c[1] // 4 + 190,
                    c[2] // 4 + 190,
                )
                faded = tuple(min(255, v) for v in faded)
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, rt, faded)

        # particles
        self.draw_particles()

        if self.show_info:
            self._draw_info()
        self.draw_pause_indicator()

    def _draw_info(self):
        name = SHAPES[self.shape_id][0]
        scheme = COLOR_SCHEMES[self.color_scheme]
        lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Workspace: {self.config.sim_width:.1f}x{self.config.sim_height:.1f}m ({self.config.width}x{self.config.height}px)",
            f"Shape: {name}",
            f"Species: {self.n_species}   Particles: {self.n}",
            f"Spring: {self.spring_k:.1f}   Grid: {self.spacing:.2f}",
            f"Colours: {scheme}",
            "",
            "Controls:",
            "1-5 - Switch shape",
            "S - Scatter particles",
            "R - Reset to targets",
            "T - Toggle target markers",
            "+/- - Spring strength",
            "[/] - Grid density",
            "C - Cycle colour scheme",
            "SPACE - Pause/Resume",
            "I - Toggle info",
            "Q/ESC - Quit",
        ]
        y = 10
        for line in lines:
            if line:
                text = self.font.render(line, True, (200, 200, 200))
                self.screen.blit(text, (10, y))
            y += 25

    # ── events ───────────────────────────────────────────────────────

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if not self._on_key(event.key):
                    return False
        return True

    def _on_key(self, key):
        """Return False to quit."""
        if key in (pygame.K_q, pygame.K_ESCAPE):
            return False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_i:
            self.show_info = not self.show_info
        elif key == pygame.K_t:
            self.show_targets = not self.show_targets
        elif key == pygame.K_s:
            self._scatter()
        elif key == pygame.K_r:
            self._reset_to_targets()
        elif key == pygame.K_c:
            self.color_scheme = (self.color_scheme + 1) % len(COLOR_SCHEMES)
            self._apply_colors()
            print(f"Colour scheme: {COLOR_SCHEMES[self.color_scheme]}")
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.spring_k = min(10.0, self.spring_k + 0.5)
            print(f"Spring: {self.spring_k:.1f}")
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.spring_k = max(0.0, self.spring_k - 0.5)
            print(f"Spring: {self.spring_k:.1f}")
        elif key == pygame.K_RIGHTBRACKET:
            self.spacing = max(0.25, self.spacing - 0.05)
            self._rebuild(self.shape_id)
        elif key == pygame.K_LEFTBRACKET:
            self.spacing = min(1.0, self.spacing + 0.05)
            self._rebuild(self.shape_id)
        elif pygame.K_1 <= key <= pygame.K_5:
            self._switch_shape(key - pygame.K_0)
        return True

    # ── main loop ────────────────────────────────────────────────────

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
    demo = FormationFillDemo(shape_id=1, spacing=0.45)
    demo.run()


if __name__ == "__main__":
    main()
