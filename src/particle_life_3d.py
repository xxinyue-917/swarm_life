#!/usr/bin/env python3
"""
3D Particle Life Simulation with Pygame

3D extension of the particle life simulation with isometric projection,
depth cues, and camera controls.
"""

import pygame
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import colorsys
import math
import json
import argparse
import os
from datetime import datetime
from scipy.spatial.transform import Rotation


@dataclass
class Config3D:
    """3D Simulation configuration"""
    # Display (pixels) — only for pygame window
    width: int = 1200
    height: int = 800
    # Simulation space (meters) - now 3D
    sim_width: float = 8.0      # X dimension
    sim_height: float = 8.0     # Y dimension
    sim_depth: float = 8.0      # Z dimension (NEW)
    # Physics params (all in meters / meters per second)
    init_space_size: float = 1.5    # spawn area half-size in meters
    n_species: int = 3
    n_particles: int = 20           # particles per species
    dt: float = 0.05
    max_speed: float = 1.0          # meters/sec
    r_max: float = 2.0              # interaction radius in meters
    beta: float = 0.2               # repulsion threshold (dimensionless)
    force_scale: float = 0.5        # force multiplier
    far_attraction: float = 0.1     # long-range attraction strength beyond r_max
    seed: int = 42
    max_angular_speed: float = 20.0
    a_rot: float = 1.0
    # Matrices (initialized as None, will be set during initialization)
    position_matrix: Optional[List[List[float]]] = None
    orientation_matrix: Optional[List[List[float]]] = None

    def to_dict(self):
        """Convert config to dictionary for JSON serialization"""
        d = asdict(self)
        if hasattr(self, '_position_matrix_np'):
            d['position_matrix'] = self._position_matrix_np.tolist()
        if hasattr(self, '_orientation_matrix_np'):
            d['orientation_matrix'] = self._orientation_matrix_np.tolist()
        return d

    @classmethod
    def from_dict(cls, data):
        """Create Config3D from dictionary"""
        pos_matrix = data.pop('position_matrix', None)
        ori_matrix = data.pop('orientation_matrix', None)

        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_keys}

        config = cls(**data)
        config.position_matrix = pos_matrix
        config.orientation_matrix = ori_matrix
        return config

    def save(self, filepath):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ParticleLife3D:
    """
    3D Particle Life Simulation with isometric projection.

    Features:
    - 3D positions and velocities
    - Quaternion-based orientations
    - Camera rotation (yaw/pitch) with mouse drag
    - Zoom control with scroll wheel
    - Depth cues (size and alpha)
    - Dual interaction matrices (position and orientation)
    """

    def __init__(self, config: Config3D, headless: bool = False):
        self.config = config
        self.headless = headless
        self.rng = np.random.RandomState(config.seed)

        # Initialize particles
        self.n = config.n_particles * config.n_species
        self.n_species = config.n_species

        # Colors for species
        self.colors = self.generate_colors(self.n_species)

        # Initialize matrices from config or generate default
        if config.position_matrix is not None:
            self.matrix = np.array(config.position_matrix)
        else:
            self.matrix = np.zeros((self.n_species, self.n_species))

        if config.orientation_matrix is not None:
            self.alignment_matrix = np.array(config.orientation_matrix)
        else:
            self.alignment_matrix = np.zeros((self.n_species, self.n_species))

        # Store matrices in config for saving
        self.config._position_matrix_np = self.matrix
        self.config._orientation_matrix_np = self.alignment_matrix

        # Initialize all particle states
        self.initialize_particles()

        # Camera state
        self.cam_yaw = np.pi / 6      # Rotation around Y axis
        self.cam_pitch = np.pi / 6    # Rotation around X axis
        self.cam_zoom = 1.0
        self.cam_pan = np.array([0.0, 0.0])  # Screen pan offset

        # Display center and scale
        self.update_display_params()

        # Initialize display variables
        self.screen = None
        self.clock = None
        self.font = None

        # UI state
        self.paused = False
        self.show_info = True
        self.show_matrix = False
        self.matrix_cursor = [0, 0]
        self.current_matrix = "position"
        self.show_axes = True  # Show 3D axes
        self.fullscreen = False

        # Mouse state for camera control
        self.mouse_dragging = False
        self.last_mouse_pos = None
        self.right_dragging = False  # For panning

        # Set up display unless headless
        if not headless:
            self.setup_display()

    def update_display_params(self):
        """Update display parameters based on config"""
        # Center of screen
        self.cx = self.config.width // 2
        self.cy = self.config.height // 2
        # Pixels per unit (base scale)
        max_dim = max(self.config.sim_width, self.config.sim_height, self.config.sim_depth)
        self.ppu = min(self.config.width, self.config.height) / (max_dim * 2.5)

    def setup_display(self, title: str = "3D Particle Life Simulation"):
        """Initialize pygame display."""
        if not pygame.get_init():
            pygame.init()
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.headless = False

    def generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for species"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors

    def initialize_particles(self, count=None, reset_all=True):
        """Initialize 3D particle states."""
        if count is None:
            count = self.n

        # Calculate spawn area (center of 3D simulation space)
        center = np.array([
            self.config.sim_width / 2,
            self.config.sim_height / 2,
            self.config.sim_depth / 2
        ])
        size = self.config.init_space_size

        # Generate initial states - 3D positions
        positions = self.rng.uniform(
            center - size,
            center + size,
            (count, 3)
        )
        velocities = np.zeros((count, 3))  # Start with zero velocity

        # Quaternion orientations - random initial orientations
        # Store as (N, 4) array: [w, x, y, z]
        orientations = self.random_quaternions(count)

        # 3D angular velocities
        angular_velocities = self.rng.randn(count, 3) * 0.1

        # Equal distribution of species
        species = np.zeros(count, dtype=int)
        particles_per_species = count // self.n_species
        remainder = count % self.n_species

        idx = 0
        for s in range(self.n_species):
            n_for_this_species = particles_per_species + (1 if s < remainder else 0)
            species[idx:idx + n_for_this_species] = s
            idx += n_for_this_species

        self.rng.shuffle(species)

        if reset_all:
            self.n = count
            self.positions = positions
            self.velocities = velocities
            self.orientations = orientations
            self.angular_velocities = angular_velocities
            self.species = species
        else:
            return positions, velocities, orientations, angular_velocities, species

    def random_quaternions(self, n: int) -> np.ndarray:
        """Generate n random unit quaternions."""
        # Use scipy for proper random rotations
        rotations = Rotation.random(n, random_state=self.rng)
        return rotations.as_quat()  # Returns [x, y, z, w] format

    def quaternion_rotate_vector(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate vector v by quaternion q.
        q: quaternion [x, y, z, w]
        v: vector [x, y, z]
        """
        rotation = Rotation.from_quat(q)
        return rotation.apply(v)

    def get_tangent_direction(self, i: int, r_hat: np.ndarray) -> np.ndarray:
        """
        Get consistent tangent direction for particle i perpendicular to r_hat.
        Uses the particle's orientation to define a reference direction,
        then computes the tangent in the plane perpendicular to r_hat.
        """
        # Get particle's "up" direction from its orientation
        q = self.orientations[i]
        up = self.quaternion_rotate_vector(q, np.array([0.0, 1.0, 0.0]))

        # Compute tangent: t = r_hat × up, then normalize
        t = np.cross(r_hat, up)
        t_norm = np.linalg.norm(t)

        if t_norm < 1e-8:
            # r_hat is parallel to up, use forward direction instead
            forward = self.quaternion_rotate_vector(q, np.array([0.0, 0.0, 1.0]))
            t = np.cross(r_hat, forward)
            t_norm = np.linalg.norm(t)
            if t_norm < 1e-8:
                # Fallback: arbitrary perpendicular
                if abs(r_hat[0]) < 0.9:
                    t = np.cross(r_hat, np.array([1.0, 0.0, 0.0]))
                else:
                    t = np.cross(r_hat, np.array([0.0, 1.0, 0.0]))
                t_norm = np.linalg.norm(t)

        return t / t_norm

    # =========================================================================
    # 3D Projection and Camera
    # =========================================================================

    def project_3d_to_2d(self, pos_3d: np.ndarray) -> Tuple[float, float, float]:
        """
        Project 3D position to 2D screen coordinates with camera rotation.
        Returns (screen_x, screen_y, depth) where depth is used for sorting and depth cues.
        """
        # Center the position around simulation center
        center = np.array([
            self.config.sim_width / 2,
            self.config.sim_height / 2,
            self.config.sim_depth / 2
        ])
        pos = pos_3d - center

        # Rotate around Y axis (yaw)
        cos_yaw = np.cos(self.cam_yaw)
        sin_yaw = np.sin(self.cam_yaw)
        x1 = pos[0] * cos_yaw - pos[2] * sin_yaw
        z1 = pos[0] * sin_yaw + pos[2] * cos_yaw
        y1 = pos[1]

        # Rotate around X axis (pitch)
        cos_pitch = np.cos(self.cam_pitch)
        sin_pitch = np.sin(self.cam_pitch)
        y2 = y1 * cos_pitch - z1 * sin_pitch
        z2 = y1 * sin_pitch + z1 * cos_pitch
        x2 = x1

        # Project to screen (orthographic projection)
        screen_x = self.cx + x2 * self.ppu * self.cam_zoom + self.cam_pan[0]
        screen_y = self.cy - y2 * self.ppu * self.cam_zoom + self.cam_pan[1]
        depth = z2

        return screen_x, screen_y, depth

    def project_batch(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch project all positions to 2D.
        Returns (screen_x, screen_y, depth) arrays.
        """
        # Center the positions
        center = np.array([
            self.config.sim_width / 2,
            self.config.sim_height / 2,
            self.config.sim_depth / 2
        ])
        pos = positions - center

        # Rotate around Y axis (yaw)
        cos_yaw = np.cos(self.cam_yaw)
        sin_yaw = np.sin(self.cam_yaw)
        x1 = pos[:, 0] * cos_yaw - pos[:, 2] * sin_yaw
        z1 = pos[:, 0] * sin_yaw + pos[:, 2] * cos_yaw
        y1 = pos[:, 1]

        # Rotate around X axis (pitch)
        cos_pitch = np.cos(self.cam_pitch)
        sin_pitch = np.sin(self.cam_pitch)
        y2 = y1 * cos_pitch - z1 * sin_pitch
        z2 = y1 * sin_pitch + z1 * cos_pitch
        x2 = x1

        # Project to screen
        screen_x = self.cx + x2 * self.ppu * self.cam_zoom + self.cam_pan[0]
        screen_y = self.cy - y2 * self.ppu * self.cam_zoom + self.cam_pan[1]

        return screen_x, screen_y, z2

    # =========================================================================
    # Physics
    # =========================================================================

    def compute_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 3D velocities using the same force kernel as 2D.
        """
        new_velocities = np.zeros_like(self.velocities)
        new_angular_velocities = np.zeros_like(self.angular_velocities)

        r_max = self.config.r_max
        beta = self.config.beta
        inv_1_minus_beta = 1.0 / (1.0 - beta) if beta < 1.0 else 1.0
        force_scale = self.config.force_scale
        far_attraction = self.config.far_attraction

        for i in range(self.n):
            # Vector from particle i to all others
            delta = self.positions - self.positions[i]

            # 3D distances
            dist = np.linalg.norm(delta, axis=1)
            dist[i] = np.inf  # Avoid self-interaction

            velocity_sum = np.zeros(3)

            for j in range(self.n):
                if j == i:
                    continue
                r = dist[j]
                r_norm = r / r_max

                if r_norm >= 1.0:
                    # Long-range attraction beyond r_max
                    if far_attraction > 0:
                        si = self.species[i]
                        sj = self.species[j]
                        k_pos = self.matrix[si, sj]
                        inv_r = 1.0 / (r + 1e-8)
                        r_hat = delta[j] * inv_r
                        F = k_pos * far_attraction
                        velocity_sum += force_scale * F * r_hat
                    continue

                # Species lookups
                si = self.species[i]
                sj = self.species[j]
                k_pos = self.matrix[si, sj]
                k_rot = self.alignment_matrix[si, sj]

                # Unit radial direction
                inv_r = 1.0 / (r + 1e-8)
                r_hat = delta[j] * inv_r

                # 3D tangent direction (using particle orientation)
                t_hat = self.get_tangent_direction(i, r_hat)

                # Piecewise linear radial force (4 zones)
                if r_norm < beta:
                    # Zone 1: universal repulsion
                    F = r_norm / beta - 1.0
                else:
                    triangle = 1.0 - abs(2.0 * r_norm - 1.0 - beta) * inv_1_minus_beta
                    peak_r = 0.5 * (1.0 + beta)
                    if r_norm < peak_r:
                        # Zone 2: rising triangle
                        F = k_pos * triangle
                    else:
                        # Zone 3/4: falling triangle with floor
                        F = k_pos * max(far_attraction, triangle)

                velocity_sum += force_scale * F * r_hat

                # Tangential swirl force
                omega_norm = np.clip(
                    np.linalg.norm(self.angular_velocities[j]) / self.config.max_angular_speed,
                    -1.0, 1.0
                )
                swirl_weight = np.clip(1.0 - r_norm, 0.0, 1.0)
                swirl_gain = k_rot * omega_norm * self.config.a_rot * swirl_weight

                velocity_sum += swirl_gain * t_hat

            new_velocities[i] = velocity_sum
            new_angular_velocities[i] = np.ones(3)  # Natural frequency

        return new_velocities, new_angular_velocities

    def step(self):
        """Perform one simulation step"""
        if self.paused:
            return

        # Compute new velocities
        new_velocities, new_angular_velocities = self.compute_velocities()

        self.velocities = new_velocities
        self.angular_velocities = new_angular_velocities

        # Clamp linear speed
        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        speed_safe = np.maximum(speed, 1e-8)  # Avoid division by zero
        self.velocities = np.where(
            speed > self.config.max_speed,
            self.velocities * self.config.max_speed / speed_safe,
            self.velocities
        )

        # Clamp angular speed
        ang_speed = np.linalg.norm(self.angular_velocities, axis=1, keepdims=True)
        ang_speed_safe = np.maximum(ang_speed, 1e-8)  # Avoid division by zero
        self.angular_velocities = np.where(
            ang_speed > self.config.max_angular_speed,
            self.angular_velocities * self.config.max_angular_speed / ang_speed_safe,
            self.angular_velocities
        )

        # Update positions
        self.positions += self.velocities * self.config.dt

        # Update orientations using angular velocities
        for i in range(self.n):
            omega = self.angular_velocities[i]
            omega_mag = np.linalg.norm(omega)
            if omega_mag > 1e-8:
                axis = omega / omega_mag
                angle = omega_mag * self.config.dt
                delta_rot = Rotation.from_rotvec(axis * angle)
                current_rot = Rotation.from_quat(self.orientations[i])
                new_rot = delta_rot * current_rot
                self.orientations[i] = new_rot.as_quat()

        # 3D Boundary conditions (reflection on 6 faces)
        margin = 0.05
        bounds = [
            (0, self.config.sim_width),
            (1, self.config.sim_height),
            (2, self.config.sim_depth)
        ]

        for dim, limit in bounds:
            # Lower boundary
            mask = self.positions[:, dim] < margin
            self.positions[mask, dim] = margin
            self.velocities[mask, dim] = abs(self.velocities[mask, dim])

            # Upper boundary
            upper = limit - margin
            mask = self.positions[:, dim] > upper
            self.positions[mask, dim] = upper
            self.velocities[mask, dim] = -abs(self.velocities[mask, dim])

    # =========================================================================
    # Drawing
    # =========================================================================

    def draw(self):
        """Draw the 3D simulation with projection"""
        self.screen.fill((255, 255, 255))  # White background

        # Draw bounding box
        if self.show_axes:
            self.draw_bounding_box()

        # Project all particles
        screen_x, screen_y, depth = self.project_batch(self.positions)

        # Sort by depth (draw far particles first)
        indices = np.argsort(-depth)

        # Fixed particle size
        size = max(4, int(4 * self.cam_zoom))

        # Draw particles
        for i in indices:
            color = self.colors[self.species[i]]
            x, y = screen_x[i], screen_y[i]
            pygame.draw.circle(self.screen, color, (int(x), int(y)), size)

        # Draw info panel
        if self.show_info:
            self.draw_info()

        # Draw matrix editor
        if self.show_matrix:
            self.draw_matrix()

    def draw_bounding_box(self):
        """Draw 3D bounding box and axes"""
        w, h, d = self.config.sim_width, self.config.sim_height, self.config.sim_depth

        # Box corners
        corners = [
            [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],  # Front face
            [0, 0, d], [w, 0, d], [w, h, d], [0, h, d]   # Back face
        ]

        # Project corners
        projected = []
        for c in corners:
            px, py, _ = self.project_3d_to_2d(np.array(c))
            projected.append((int(px), int(py)))

        # Draw edges
        box_color = (180, 180, 200)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        for e in edges:
            pygame.draw.line(self.screen, box_color, projected[e[0]], projected[e[1]], 1)

        # Draw axes at origin
        origin = [0, 0, 0]
        ox, oy, _ = self.project_3d_to_2d(np.array(origin))
        axis_len = 1.5

        # X axis (red)
        ax, ay, _ = self.project_3d_to_2d(np.array([axis_len, 0, 0]))
        pygame.draw.line(self.screen, (200, 50, 50), (int(ox), int(oy)), (int(ax), int(ay)), 2)

        # Y axis (green)
        ax, ay, _ = self.project_3d_to_2d(np.array([0, axis_len, 0]))
        pygame.draw.line(self.screen, (50, 200, 50), (int(ox), int(oy)), (int(ax), int(ay)), 2)

        # Z axis (blue)
        ax, ay, _ = self.project_3d_to_2d(np.array([0, 0, axis_len]))
        pygame.draw.line(self.screen, (50, 50, 200), (int(ox), int(oy)), (int(ax), int(ay)), 2)

    def draw_info(self):
        """Draw information panel"""
        info_lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"3D Space: {self.config.sim_width:.1f}x{self.config.sim_height:.1f}x{self.config.sim_depth:.1f}m",
            f"Particles: {self.config.n_particles}/species x {self.n_species} = {self.n}",
            f"Species: {self.n_species}",
            f"Camera: yaw={np.degrees(self.cam_yaw):.0f}° pitch={np.degrees(self.cam_pitch):.0f}°",
            f"Zoom: {self.cam_zoom:.2f}x",
            f"Matrix: {'POSITION' if self.current_matrix == 'position' else 'ORIENTATION'}",
            "",
            "Controls:",
            "Left drag - Rotate camera",
            "Right drag - Pan view",
            "Scroll - Zoom",
            "UP/DOWN - Species count",
            "LEFT/RIGHT - Particle count",
            "M - Matrix editor | TAB - Switch matrix",
            "A - Toggle axes | SPACE - Pause",
            "R - Reset particles",
            "S - Save config | I - Toggle info",
            "Q/ESC - Quit",
        ]

        y = 10
        for line in info_lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 22

        if self.paused:
            pause_text = self.font.render("PAUSED", True, (200, 50, 50))
            rect = pause_text.get_rect(center=(self.config.width // 2, 30))
            self.screen.blit(pause_text, rect)

    def draw_matrix(self):
        """Draw the interaction matrix for editing"""
        matrix_x = self.config.width - 320
        matrix_y = 100
        cell_size = 36

        if self.current_matrix == "position":
            matrix = self.matrix
            matrix_name = "POSITION MATRIX (K_pos)"
            title_color = (100, 255, 100)
        else:
            matrix = self.alignment_matrix
            matrix_name = "ORIENTATION MATRIX (K_rot)"
            title_color = (100, 150, 255)

        title = self.font.render(matrix_name, True, title_color)
        self.screen.blit(title, (matrix_x, matrix_y - 40))

        for i in range(self.n_species):
            color = self.colors[i]
            label = self.font.render(f"S{i+1}", True, color)
            self.screen.blit(label, (matrix_x - 30, matrix_y + 12 + i * cell_size))
            label = self.font.render(f"S{i+1}", True, color)
            self.screen.blit(label, (matrix_x + 12 + i * cell_size, matrix_y - 18))

            for j in range(self.n_species):
                x = matrix_x + j * cell_size
                y = matrix_y + i * cell_size
                value = matrix[i, j]

                if [i, j] == self.matrix_cursor:
                    pygame.draw.rect(self.screen, (100, 100, 255), (x, y, cell_size, cell_size), 3)
                else:
                    pygame.draw.rect(self.screen, (60, 60, 60), (x, y, cell_size, cell_size), 1)

                if value > 0:
                    intensity = min(255, int(abs(value) * 255))
                    color = (0, intensity, 0)
                elif value < 0:
                    intensity = min(255, int(abs(value) * 255))
                    color = (intensity, 0, 0)
                else:
                    color = (100, 100, 100)

                text = self.font.render(f"{value:.1f}", True, color)
                text_rect = text.get_rect(center=(x + cell_size // 2, y + cell_size // 2))
                self.screen.blit(text, text_rect)

        instructions = [
            "TAB: Switch matrix",
            "WASD: Navigate",
            "+/-: Change value"
        ]
        y_offset = matrix_y + (self.n_species + 1) * cell_size
        for i, instruction in enumerate(instructions):
            text = self.font.render(instruction, True, (100, 100, 100))
            self.screen.blit(text, (matrix_x, y_offset + i * 22))

    # =========================================================================
    # Event Handling
    # =========================================================================

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click - rotate
                    self.mouse_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 3:  # Right click - pan
                    self.right_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
                elif event.button == 3:
                    self.right_dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    # Rotate camera
                    pos = pygame.mouse.get_pos()
                    if self.last_mouse_pos:
                        dx = pos[0] - self.last_mouse_pos[0]
                        dy = pos[1] - self.last_mouse_pos[1]
                        self.cam_yaw += dx * 0.01
                        self.cam_pitch -= dy * 0.01
                        # Clamp pitch to avoid flipping
                        self.cam_pitch = np.clip(self.cam_pitch, -np.pi / 2 + 0.1, np.pi / 2 - 0.1)
                    self.last_mouse_pos = pos
                elif self.right_dragging:
                    # Pan view
                    pos = pygame.mouse.get_pos()
                    if self.last_mouse_pos:
                        dx = pos[0] - self.last_mouse_pos[0]
                        dy = pos[1] - self.last_mouse_pos[1]
                        self.cam_pan[0] += dx
                        self.cam_pan[1] += dy
                    self.last_mouse_pos = pos

            elif event.type == pygame.MOUSEWHEEL:
                # Zoom
                zoom_factor = 1.1 if event.y > 0 else 0.9
                self.cam_zoom *= zoom_factor
                self.cam_zoom = np.clip(self.cam_zoom, 0.2, 5.0)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self.initialize_particles()
                    print("Reset particles")

                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info

                elif event.key == pygame.K_a and not self.show_matrix:
                    self.show_axes = not self.show_axes

                elif event.key == pygame.K_m:
                    self.show_matrix = not self.show_matrix
                    if self.show_matrix:
                        self.matrix_cursor[0] = min(self.matrix_cursor[0], self.n_species - 1)
                        self.matrix_cursor[1] = min(self.matrix_cursor[1], self.n_species - 1)

                elif event.key == pygame.K_TAB:
                    if self.show_matrix:
                        self.current_matrix = "orientation" if self.current_matrix == "position" else "position"
                        print(f"Editing: {self.current_matrix} matrix")

                elif event.key == pygame.K_UP and not self.show_matrix:
                    self.change_species_count(1)

                elif event.key == pygame.K_DOWN and not self.show_matrix:
                    self.change_species_count(-1)

                elif event.key == pygame.K_LEFT and not self.show_matrix:
                    self.change_particle_count(-50)

                elif event.key == pygame.K_RIGHT and not self.show_matrix:
                    self.change_particle_count(50)

                elif self.show_matrix:
                    if event.key == pygame.K_w:
                        self.matrix_cursor[0] = max(0, self.matrix_cursor[0] - 1)
                    elif event.key == pygame.K_s:
                        self.matrix_cursor[0] = min(self.n_species - 1, self.matrix_cursor[0] + 1)
                    elif event.key == pygame.K_a:
                        self.matrix_cursor[1] = max(0, self.matrix_cursor[1] - 1)
                    elif event.key == pygame.K_d:
                        self.matrix_cursor[1] = min(self.n_species - 1, self.matrix_cursor[1] + 1)
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        i, j = self.matrix_cursor
                        if self.current_matrix == "position":
                            self.matrix[i, j] = min(1.0, self.matrix[i, j] + 0.1)
                            self.config._position_matrix_np = self.matrix
                        else:
                            self.alignment_matrix[i, j] = min(1.0, self.alignment_matrix[i, j] + 0.1)
                            self.config._orientation_matrix_np = self.alignment_matrix
                    elif event.key == pygame.K_MINUS:
                        i, j = self.matrix_cursor
                        if self.current_matrix == "position":
                            self.matrix[i, j] = max(-1.0, self.matrix[i, j] - 0.1)
                            self.config._position_matrix_np = self.matrix
                        else:
                            self.alignment_matrix[i, j] = max(-1.0, self.alignment_matrix[i, j] - 0.1)
                            self.config._orientation_matrix_np = self.alignment_matrix

                elif event.key == pygame.K_s:
                    filename = self.save_current_config()
                    print(f"Saved: {filename}")

                elif event.key == pygame.K_HOME:
                    # Reset camera
                    self.cam_yaw = np.pi / 6
                    self.cam_pitch = np.pi / 6
                    self.cam_zoom = 1.0
                    self.cam_pan = np.array([0.0, 0.0])

        return True

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def change_species_count(self, delta: int):
        """Change the number of species"""
        new_count = self.n_species + delta
        new_count = max(2, min(10, new_count))

        if new_count == self.n_species:
            return

        self.n_species = new_count
        self.n = self.config.n_particles * self.n_species
        self.colors = self.generate_colors(self.n_species)
        self.matrix = np.zeros((self.n_species, self.n_species))
        self.alignment_matrix = np.zeros((self.n_species, self.n_species))
        self.config._position_matrix_np = self.matrix
        self.config._orientation_matrix_np = self.alignment_matrix
        self.initialize_particles()
        print(f"Species: {self.n_species}, Total: {self.n}")

    def change_particle_count(self, delta: int):
        """Change the number of particles per species"""
        new_per_species = self.config.n_particles + delta
        new_per_species = max(5, min(200, new_per_species))

        if new_per_species == self.config.n_particles:
            return

        self.config.n_particles = new_per_species
        self.n = new_per_species * self.n_species
        self.initialize_particles()
        print(f"Particles: {self.config.n_particles}/species x {self.n_species} = {self.n}")

    def save_current_config(self) -> str:
        """Save current configuration"""
        self.config.n_species = self.n_species
        self.config.n_particles = self.n // self.n_species
        self.config._position_matrix_np = self.matrix
        self.config._orientation_matrix_np = self.alignment_matrix

        os.makedirs("presets3d", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"presets3d/config3d_{timestamp}.json"
        self.config.save(filename)
        return filename

    def get_swarm_centroid(self) -> np.ndarray:
        """Get 3D centroid of all particles."""
        return self.positions.mean(axis=0)

    def get_species_centroids(self) -> List[np.ndarray]:
        """Get 3D centroid of each species."""
        centroids = []
        center = np.array([
            self.config.sim_width / 2,
            self.config.sim_height / 2,
            self.config.sim_depth / 2
        ])
        for s in range(self.n_species):
            mask = self.species == s
            if mask.any():
                centroids.append(self.positions[mask].mean(axis=0))
            else:
                centroids.append(center.copy())
        return centroids

    # =========================================================================
    # Main Loop
    # =========================================================================

    def run(self):
        """Main simulation loop"""
        running = True

        while running:
            running = self.handle_events()
            self.step()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='3D Particle Life Simulation')
    parser.add_argument('--load', type=str, help='Path to configuration file to load')
    args = parser.parse_args()

    print("Starting 3D Particle Life Simulation...")
    print("Left drag to rotate | Right drag to pan | Scroll to zoom")
    print("Press M for matrix editor, SPACE to pause, Q to quit")

    if args.load:
        if os.path.exists(args.load):
            print(f"Loading configuration from {args.load}")
            config = Config3D.load(args.load)
        else:
            print(f"Config file {args.load} not found, using default")
            config = Config3D()
    else:
        config = Config3D()

    sim = ParticleLife3D(config)
    sim.run()


if __name__ == "__main__":
    main()
