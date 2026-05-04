#!/usr/bin/env python3
"""
Pygame-based custom viewer for the Crazyflie swarm.

Reuses the existing ParticleLife rendering (anti-aliased circles, species
colors, centroid markers) by replacing its physics step with live Vicon poses
streamed from /cfN/pose topics.

Designed per crazyflie_deployment/SIM_DESIGN.md (Option A — pygame reuse).

Usage:
    ros2 run particle_life viewer
    # In a separate terminal, run Crazyswarm2 (sim or hardware)
"""

import os
import importlib.util
import threading
import yaml
import numpy as np
import pygame
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# Load the swarm_life ParticleLife module directly from its file
# (cannot use plain `import particle_life` — collides with this ROS package's name)
_SWARM_LIFE_FILE = os.path.expanduser('~/swarm_life/src/particle_life.py')
_spec = importlib.util.spec_from_file_location('swarm_life_pl', _SWARM_LIFE_FILE)
_pl_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pl_mod)
ParticleLife = _pl_mod.ParticleLife
Config = _pl_mod.Config


# Arena geometry — must match crazyflie_deployment/config/arena.yaml
ARENA_SIZE = 3.0  # meters; arena is centered at (0,0) in Vicon frame


def load_species_config(path=None):
    """Read drone → species mapping from config/species.yaml."""
    if path is None:
        # Default: alongside the package config dir
        candidates = [
            os.path.expanduser('~/swarm_life/crazyflie_deployment/config/species.yaml'),
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
    if path is None or not os.path.exists(path):
        # Fallback: empty mapping; viewer will assign all drones to species 0
        return {}, 1
    with open(path) as f:
        cfg = yaml.safe_load(f)
    active = cfg.get('active', list(cfg.get('presets', {}).keys())[0])
    preset = cfg['presets'][active]
    assignments = preset['assignments']
    n_species = max(assignments.values()) + 1
    return assignments, n_species


class CFViewer(Node):
    """ROS2 node that streams live drone poses into a pygame window."""

    def __init__(self, drone_names, species_assignments, n_species):
        super().__init__('cf_viewer')

        self.drone_names = drone_names  # ordered list, e.g. ['cf1', 'cf5', 'cf7']
        self.n_drones = len(drone_names)
        self.species_assignments = species_assignments
        self.lock = threading.Lock()

        # Build a ParticleLife instance with N=n_drones, initialized headless
        cfg = Config(
            n_species=n_species,
            n_particles=self.n_drones // n_species + 1,
            sim_width=ARENA_SIZE,
            sim_height=ARENA_SIZE,
            width=900,
            height=900,
        )
        self.sim = ParticleLife(cfg, headless=False)
        # Override n and species to exactly match our drone roster
        self.sim.n = self.n_drones
        self.sim.positions = np.full((self.n_drones, 2), ARENA_SIZE / 2)  # center
        self.sim.velocities = np.zeros((self.n_drones, 2))
        self.sim.orientations = np.zeros(self.n_drones)
        self.sim.species = np.array(
            [species_assignments.get(name, 0) for name in drone_names], dtype=int
        )

        # Per-drone pose subscribers
        self._poses_received = {name: False for name in drone_names}
        for idx, name in enumerate(drone_names):
            self.create_subscription(
                PoseStamped,
                f'/{name}/pose',
                lambda msg, i=idx, n=name: self._update_pose(i, n, msg),
                10,
            )

        self.get_logger().info(
            f"viewer started — tracking {self.n_drones} drones: {drone_names}"
        )

    def _update_pose(self, idx, name, msg):
        with self.lock:
            # Vicon arena is centered at (0,0); sim arena is (0..ARENA_SIZE)
            self.sim.positions[idx, 0] = msg.pose.position.x + ARENA_SIZE / 2
            self.sim.positions[idx, 1] = msg.pose.position.y + ARENA_SIZE / 2
            self._poses_received[name] = True

    def snapshot(self):
        """Thread-safe copy of state for the render thread."""
        with self.lock:
            return self.sim.positions.copy(), self.sim.species.copy()


def draw_arena_box(sim):
    """Draw a black rectangle around the arena bounds for visual reference."""
    p0 = sim.to_screen([0, 0])
    p1 = sim.to_screen([ARENA_SIZE, ARENA_SIZE])
    rect = pygame.Rect(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1])
    pygame.draw.rect(sim.screen, (0, 0, 0), rect, 2)


def draw_drone_labels(sim, names):
    """Draw drone name next to each particle."""
    for i in range(sim.n):
        x, y = sim.to_screen(sim.positions[i])
        txt = sim.font.render(names[i], True, (50, 50, 50))
        sim.screen.blit(txt, (x + 12, y - 8))


def main():
    # Choose drone roster: defaults to the 7-drone fleet from FLEET.md
    drone_names = ['cf1', 'cf2', 'cf3', 'cf4', 'cf5', 'cf7', 'cf9']
    assignments, n_species = load_species_config()

    rclpy.init()
    viewer = CFViewer(drone_names, assignments, n_species)

    # ROS spinning in a background thread; pygame on main thread
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(viewer,), daemon=True
    )
    spin_thread.start()

    sim = viewer.sim
    pygame.display.set_caption("Crazyflie Particle Life Viewer")
    clock = pygame.time.Clock()

    try:
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    running = False

            # Snapshot under lock, then render
            with viewer.lock:
                sim.screen.fill((255, 255, 255))
                draw_arena_box(sim)
                sim.draw_particles()
                draw_drone_labels(sim, drone_names)
                # Optional: draw centroid markers (helps see species clustering)
                try:
                    pts = sim.draw_centroid_spine()
                    sim.draw_centroid_markers(pts)
                except Exception:
                    pass

            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()
        viewer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
