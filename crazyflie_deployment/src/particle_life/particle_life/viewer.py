#!/usr/bin/env python3
"""
Top-down pygame viewer for the Crazyflie swarm.

Subscribes to `/cfN/pose` (PoseStamped) for every enabled drone in
`crazyflies.yaml` and renders their live XY positions on a 2D arena.
Works with any pose source — Crazyswarm2 hardware backend, sim backend,
or `fake_server` (perfect-tracking stub).

Render
------
* Arena box matching `arena.yaml` (width × height, centered at origin_x/y).
* 0.5 m grid for spatial reference.
* Each drone:
    - circle colored by species (from `species.yaml`)
    - drone name label
    - altitude bar to the right (0 → 1.5 m)
    - fading trail of recent positions
* Status bar: FPS, "N / M receiving", per-drone stale flag (>1 s since last
  pose).

Keyboard
--------
    T   toggle trails
    L   toggle drone labels
    G   toggle grid
    C   clear trails
    ESC / Q   quit
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import pygame
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


# ---------------------------------------------------------------- paths

_REPO_CONFIG = os.path.expanduser(
    '~/swarm_life/crazyflie_deployment/config')


def _find(*candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _config_path(name):
    """Resolve a config file from package share (preferred) or repo source."""
    # When run from install/ via ros2 run, the configs ship in pkg share.
    share = None
    try:
        from ament_index_python.packages import get_package_share_directory
        share = os.path.join(
            get_package_share_directory('particle_life'), 'config', name)
    except Exception:
        pass
    return _find(share, os.path.join(_REPO_CONFIG, name))


# ---------------------------------------------------------------- config

def load_drone_roster():
    """Return (names, init_positions) for every enabled drone in crazyflies.yaml."""
    path = _config_path('crazyflies.yaml')
    with open(path) as f:
        cfg = yaml.safe_load(f)
    names, init = [], {}
    for name, info in (cfg.get('robots') or {}).items():
        if info.get('enabled', False):
            names.append(name)
            init[name] = info.get('initial_position', [0.0, 0.0, 0.0])
    return names, init


def load_species(names):
    """Return (species_per_drone, n_species)."""
    path = _config_path('species.yaml')
    if path is None:
        return [0] * len(names), 1
    with open(path) as f:
        cfg = yaml.safe_load(f)
    active = cfg.get('active')
    preset = cfg.get('presets', {}).get(active, {})
    assignments = preset.get('assignments', {})
    n_species = (max(assignments.values()) + 1) if assignments else 1
    return [assignments.get(n, 0) for n in names], n_species


def load_arena():
    """Return dict with width, height, origin_x, origin_y. Defaults if missing."""
    path = _config_path('arena.yaml')
    if path is None:
        return {'width': 3.0, 'height': 3.0, 'origin_x': 0.0, 'origin_y': 0.0}
    with open(path) as f:
        cfg = yaml.safe_load(f)
    a = cfg.get('arena', {})
    return {
        'width': float(a.get('width', 3.0)),
        'height': float(a.get('height', 3.0)),
        'origin_x': float(a.get('origin_x', 0.0)),
        'origin_y': float(a.get('origin_y', 0.0)),
    }


# ---------------------------------------------------------------- colors

# tab10-ish palette (RGB 0-255), drone-species mapping.
SPECIES_COLORS = [
    (31, 119, 180),    # blue
    (255, 127, 14),    # orange
    (44, 160, 44),     # green
    (214, 39, 40),     # red
    (148, 103, 189),   # purple
    (140, 86, 75),     # brown
    (227, 119, 194),   # pink
    (127, 127, 127),   # gray
    (188, 189, 34),    # yellow-green
    (23, 190, 207),    # cyan
]

BG = (245, 245, 248)
ARENA_LINE = (60, 60, 60)
GRID_LINE = (220, 220, 225)
TEXT_DARK = (40, 40, 50)
TEXT_DIM = (150, 150, 160)
STALE = (190, 190, 190)
STATUS_BG = (28, 30, 36)
STATUS_TEXT = (235, 235, 240)
Z_BAR_BG = (210, 210, 215)


# ---------------------------------------------------------------- node

class ViewerNode(Node):
    """ROS subscriber side: collects poses into a thread-safe state dict."""

    def __init__(self, names):
        super().__init__('cf_viewer')
        self.names = names
        self.lock = threading.Lock()
        # x, y, z, last_update_wall_t per drone
        self.state = {n: (0.0, 0.0, 0.0, 0.0) for n in names}

        for name in names:
            self.create_subscription(
                PoseStamped, f'/{name}/pose',
                lambda msg, n=name: self._update(n, msg), 10)
        self.get_logger().info(f"viewer subscribing to {len(names)} drones: {names}")

    def _update(self, name, msg):
        with self.lock:
            self.state[name] = (
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                time.monotonic(),
            )

    def snapshot(self):
        with self.lock:
            return dict(self.state)


# ---------------------------------------------------------------- render

class Renderer:

    Z_MAX = 1.5            # m, top of altitude bar
    TRAIL_LEN = 60         # samples kept per drone
    STALE_AFTER = 1.0      # s; older poses → dim

    def __init__(self, names, species, n_species, arena, window=(960, 980)):
        self.names = names
        self.species = species
        self.n_species = n_species
        self.arena = arena
        self.show_trails = True
        self.show_labels = True
        self.show_grid = True

        self.window_w, self.window_h = window
        self.status_h = 36
        self.plot_h = self.window_h - self.status_h
        # Square plot area centered horizontally.
        self.plot_size = min(self.window_w - 40, self.plot_h - 40)
        self.plot_x0 = (self.window_w - self.plot_size) // 2
        self.plot_y0 = self.status_h + (self.plot_h - self.plot_size) // 2

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption("Crazyflie Particle Life Viewer")
        self.font = pygame.font.SysFont('DejaVu Sans', 14)
        self.font_small = pygame.font.SysFont('DejaVu Sans', 11)
        self.font_status = pygame.font.SysFont('DejaVu Sans', 13, bold=True)

        self.trails = {n: deque(maxlen=self.TRAIL_LEN) for n in names}

    # ---------------- coordinate transform ----------------

    def to_screen(self, x, y):
        """World (m) → screen pixel. Arena centered at origin_x/y."""
        nx = (x - self.arena['origin_x']) / self.arena['width'] + 0.5
        ny = (y - self.arena['origin_y']) / self.arena['height'] + 0.5
        # Pygame y axis is flipped relative to math y axis.
        return (
            int(self.plot_x0 + nx * self.plot_size),
            int(self.plot_y0 + (1 - ny) * self.plot_size),
        )

    # ---------------- drawing primitives ----------------

    def _draw_grid(self):
        # 0.5 m grid lines.
        step = 0.5
        ox, oy = self.arena['origin_x'], self.arena['origin_y']
        w, h = self.arena['width'], self.arena['height']
        x_lines = np.arange(np.ceil((ox - w / 2) / step) * step,
                            ox + w / 2 + 1e-9, step)
        y_lines = np.arange(np.ceil((oy - h / 2) / step) * step,
                            oy + h / 2 + 1e-9, step)
        for x in x_lines:
            p0 = self.to_screen(x, oy - h / 2)
            p1 = self.to_screen(x, oy + h / 2)
            color = ARENA_LINE if abs(x) < 1e-6 else GRID_LINE
            pygame.draw.line(self.screen, color, p0, p1, 1)
        for y in y_lines:
            p0 = self.to_screen(ox - w / 2, y)
            p1 = self.to_screen(ox + w / 2, y)
            color = ARENA_LINE if abs(y) < 1e-6 else GRID_LINE
            pygame.draw.line(self.screen, color, p0, p1, 1)

    def _draw_arena_box(self):
        p0 = self.to_screen(self.arena['origin_x'] - self.arena['width'] / 2,
                            self.arena['origin_y'] - self.arena['height'] / 2)
        p1 = self.to_screen(self.arena['origin_x'] + self.arena['width'] / 2,
                            self.arena['origin_y'] + self.arena['height'] / 2)
        rect = pygame.Rect(p0[0], p1[1], p1[0] - p0[0], p0[1] - p1[1])
        pygame.draw.rect(self.screen, ARENA_LINE, rect, 2)

    def _draw_axis_ticks(self):
        # Axis labels at corners.
        w, h = self.arena['width'], self.arena['height']
        ox, oy = self.arena['origin_x'], self.arena['origin_y']
        for (x, y, anchor) in [
            (ox - w / 2, oy - h / 2, ('left', 'top')),
            (ox + w / 2, oy - h / 2, ('right', 'top')),
            (ox - w / 2, oy + h / 2, ('left', 'bottom')),
            (ox + w / 2, oy + h / 2, ('right', 'bottom')),
        ]:
            sx, sy = self.to_screen(x, y)
            label = self.font_small.render(
                f'({x:+.1f}, {y:+.1f})', True, TEXT_DIM)
            r = label.get_rect()
            r.left = sx + 4 if anchor[0] == 'left' else sx - r.width - 4
            r.top = sy + 4 if anchor[1] == 'top' else sy - r.height - 4
            self.screen.blit(label, r)

    def _draw_trail(self, name, color):
        pts = self.trails[name]
        if len(pts) < 2:
            return
        screen_pts = [self.to_screen(x, y) for (x, y) in pts]
        # Fade older points by drawing with progressively lighter color.
        n = len(screen_pts)
        for i in range(1, n):
            alpha = i / n  # 0..1, recent = brighter
            c = tuple(int(BG[k] * (1 - alpha) + color[k] * alpha) for k in range(3))
            pygame.draw.line(self.screen, c, screen_pts[i - 1], screen_pts[i], 2)

    def _draw_drone(self, name, sp_idx, x, y, z, stale):
        color = STALE if stale else SPECIES_COLORS[sp_idx % len(SPECIES_COLORS)]
        sx, sy = self.to_screen(x, y)
        # Trail
        if self.show_trails:
            self._draw_trail(name, color)
        # Circle
        pygame.draw.circle(self.screen, color, (sx, sy), 11)
        pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), 11, 1)
        # Label
        if self.show_labels:
            txt = self.font.render(name, True,
                                   TEXT_DIM if stale else TEXT_DARK)
            self.screen.blit(txt, (sx + 14, sy - 8))
        # Z bar (small vertical bar 14px right of the dot)
        bar_x = sx + 22
        bar_y0 = sy - 18
        bar_h = 36
        pygame.draw.rect(self.screen, Z_BAR_BG,
                         (bar_x, bar_y0, 5, bar_h))
        z_norm = max(0.0, min(1.0, z / self.Z_MAX))
        z_pix = int(bar_h * z_norm)
        pygame.draw.rect(self.screen, color,
                         (bar_x, bar_y0 + bar_h - z_pix, 5, z_pix))

    def _draw_status(self, fps, n_recv, n_total, stale_names):
        pygame.draw.rect(self.screen, STATUS_BG,
                         (0, 0, self.window_w, self.status_h))
        parts = [
            f"{fps:5.1f} FPS",
            f"poses {n_recv}/{n_total}",
        ]
        if stale_names:
            parts.append(f"STALE: {', '.join(sorted(stale_names))}")
        x = 14
        for txt in parts:
            surf = self.font_status.render(txt, True, STATUS_TEXT)
            self.screen.blit(surf, (x, 10))
            x += surf.get_width() + 30

        keys = "T trails  L labels  G grid  C clear  ESC quit"
        surf = self.font_small.render(keys, True, STATUS_TEXT)
        self.screen.blit(surf, (self.window_w - surf.get_width() - 14, 12))

    # ---------------- top-level frame ----------------

    def draw(self, snap, fps):
        now = time.monotonic()
        self.screen.fill(BG)
        if self.show_grid:
            self._draw_grid()
        self._draw_arena_box()
        self._draw_axis_ticks()

        n_recv = 0
        stale_names = set()
        for i, name in enumerate(self.names):
            x, y, z, last_t = snap[name]
            received = last_t > 0
            stale = received and (now - last_t > self.STALE_AFTER)
            if received:
                n_recv += 1
                # Update trail at most every render frame.
                self.trails[name].append((x, y))
            if stale:
                stale_names.add(name)
            self._draw_drone(name, self.species[i], x, y, z,
                             stale=(not received) or stale)
            if not received:
                # Pin "waiting" badge over the drone start position guess.
                pass

        self._draw_status(fps, n_recv, len(self.names), stale_names)
        pygame.display.flip()

    # ---------------- key handlers ----------------

    def handle_key(self, key):
        if key == pygame.K_t:
            self.show_trails = not self.show_trails
        elif key == pygame.K_l:
            self.show_labels = not self.show_labels
        elif key == pygame.K_g:
            self.show_grid = not self.show_grid
        elif key == pygame.K_c:
            for d in self.trails.values():
                d.clear()


# ---------------------------------------------------------------- driver

def main():
    names, _ = load_drone_roster()
    if not names:
        print("[viewer] no enabled drones in crazyflies.yaml; nothing to show.")
        return

    species, n_species = load_species(names)
    arena = load_arena()

    rclpy.init()
    node = ViewerNode(names)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    renderer = Renderer(names, species, n_species, arena)
    clock = pygame.time.Clock()
    target_fps = 60

    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        return
                    renderer.handle_key(ev.key)

            renderer.draw(node.snapshot(), fps=clock.get_fps())
            clock.tick(target_fps)
    finally:
        pygame.quit()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
