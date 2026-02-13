#!/usr/bin/env python3
"""
Standalone Particle Life Simulation with Pygame
Interactive particle simulation with real-time parameter adjustment
"""

import pygame
import pygame.gfxdraw
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import colorsys
import math
import json
import argparse
import os
from datetime import datetime

@dataclass
class Config:
    """Simulation configuration"""
    # Display (pixels) — only for pygame window
    width: int = 1500
    height: int = 800
    # Simulation space (meters)
    sim_width: float = 10.0     # simulation width in meters
    sim_height: float = 10.0    # simulation height in meters
    # Physics params (all in meters / meters per second)
    init_space_size: float = 1.0    # spawn area half-size in meters
    n_species: int = 3
    n_particles: int = 20              # particles per species
    dt: float = 0.05
    max_speed: float = 1.0         # meters/sec
    r_max: float = 2.0             # interaction radius in meters
    beta: float = 0.2              # repulsion threshold (dimensionless)
    force_scale: float = 0.5       # force multiplier
    far_attraction: float = 0.1    # long-range attraction strength beyond r_max (0 = no long-range)
    seed: int = 42
    a_rot: float = 1.0
    # Matrices (initialized as None, will be set during initialization)
    position_matrix: Optional[List[List[float]]] = None
    orientation_matrix: Optional[List[List[float]]] = None

    def to_dict(self):
        """Convert config to dictionary for JSON serialization"""
        d = asdict(self)
        # Convert numpy arrays to lists if they exist
        if hasattr(self, '_position_matrix_np'):
            d['position_matrix'] = self._position_matrix_np.tolist()
        if hasattr(self, '_orientation_matrix_np'):
            d['orientation_matrix'] = self._orientation_matrix_np.tolist()
        return d

    @classmethod
    def from_dict(cls, data):
        """Create Config from dictionary"""
        # Extract matrices if they exist
        pos_matrix = data.pop('position_matrix', None)
        ori_matrix = data.pop('orientation_matrix', None)

        # Filter out unknown keys (e.g. old params from saved presets)
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

class ParticleLife:
    """
    Main particle life simulation.

    Can be used in two modes:
    1. Interactive mode (headless=False): Full pygame display with UI
    2. Headless mode (headless=True): Pure simulation, no display
       - Use this when importing as a library for experiments
       - Call setup_display() later if you need visualization
    """

    def __init__(self, config: Config, headless: bool = False):
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
            # Generate default zero matrix
            self.matrix = np.zeros((self.n_species, self.n_species))

        if config.orientation_matrix is not None:
            self.alignment_matrix = np.array(config.orientation_matrix)
        else:
            # Generate default zero matrix
            self.alignment_matrix = np.zeros((self.n_species, self.n_species))

        # Store matrices in config for saving
        self.config._position_matrix_np = self.matrix
        self.config._orientation_matrix_np = self.alignment_matrix

        # Initialize all particle states
        self.initialize_particles()

        # Initialize display variables (may be set up later)
        self.screen = None
        self.clock = None
        self.font = None

        # UI state
        self.paused = False
        self.show_info = True
        self.show_matrix = False
        self.matrix_cursor = [0, 0]  # Row, Column for matrix editing
        self.current_matrix = "position"  # Which matrix is being edited: "position" or "orientation"
        self.show_orientations = True  # Whether to show particle orientations
        self.fullscreen = False  # Fullscreen mode flag
        self.zoom = 1.0  # Zoom factor for display scaling
        self.base_width = config.width  # Store base dimensions for scaling
        self.base_height = config.height
        # Pixels per meter — converts simulation coords to screen coords
        self.ppu = min(config.width / config.sim_width,
                       config.height / config.sim_height)
        self.dragging_edge = None  # Track which edge is being dragged for resize
        self.drag_start_pos = None  # Mouse position when drag started
        self.drag_start_size = None  # Window size when drag started

        # Set up display unless headless
        if not headless:
            self.setup_display()

    def setup_display(self, title: str = "Particle Life Simulation"):
        """
        Initialize pygame display. Called automatically unless headless=True.
        Can be called manually after construction for headless instances.
        """
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
        """
        Initialize particle states with consistent spawn pattern.

        Args:
            count: Number of particles to initialize (None = use self.n)
            reset_all: If True, reset all particles. If False, return new arrays for appending.

        Returns:
            If reset_all=False, returns tuple of (positions, velocities, orientations, species)
        """
        if count is None:
            count = self.n

        # Calculate spawn area (center of simulation space, in meters)
        center_x = self.config.sim_width / 2
        center_y = self.config.sim_height / 2

        # Generate initial states
        positions = self.rng.uniform(
            [center_x - self.config.init_space_size, center_y - self.config.init_space_size],
            [center_x + self.config.init_space_size, center_y + self.config.init_space_size],
            (count, 2)
        )
        velocities = np.zeros((count, 2))  # Start with zero velocity
        orientations = self.rng.uniform(0, 2 * np.pi, count)

        # Equal distribution of species
        species = np.zeros(count, dtype=int)
        particles_per_species = count // self.n_species
        remainder = count % self.n_species

        # Assign particles to species equally
        idx = 0
        for s in range(self.n_species):
            # Give this species its share, plus one extra if there's remainder
            n_for_this_species = particles_per_species + (1 if s < remainder else 0)
            species[idx:idx + n_for_this_species] = s
            idx += n_for_this_species

        # Shuffle the species assignments to mix them spatially
        self.rng.shuffle(species)

        if reset_all:
            # Full reset: replace all particle states
            self.n = count
            self.positions = positions
            self.velocities = velocities
            self.orientations = orientations
            self.species = species
        else:
            # Partial: return new arrays for appending
            return positions, velocities, orientations, species

    # =========================================================================
    # Utility Methods (for use in experiments)
    # =========================================================================

    def get_swarm_centroid(self) -> np.ndarray:
        """Get centroid of all particles."""
        return self.positions.mean(axis=0)

    def get_species_centroids(self) -> List[np.ndarray]:
        """Get centroid of each species."""
        centroids = []
        for s in range(self.n_species):
            mask = self.species == s
            if mask.any():
                centroids.append(self.positions[mask].mean(axis=0))
            else:
                centroids.append(np.array([self.config.sim_width / 2, self.config.sim_height / 2]))
        return centroids

    def get_species_mask(self, species_id: int) -> np.ndarray:
        """Get boolean mask for particles of a specific species."""
        return self.species == species_id

    def get_average_velocity(self) -> np.ndarray:
        """Get average velocity of all particles."""
        return self.velocities.mean(axis=0)

    def get_species_velocities(self) -> List[np.ndarray]:
        """Get average velocity for each species."""
        velocities = []
        for s in range(self.n_species):
            mask = self.species == s
            if mask.any():
                velocities.append(self.velocities[mask].mean(axis=0))
            else:
                velocities.append(np.zeros(2))
        return velocities

    def set_position_matrix(self, matrix: np.ndarray):
        """Set the position interaction matrix."""
        self.matrix = np.array(matrix)
        self.config._position_matrix_np = self.matrix

    def set_orientation_matrix(self, matrix: np.ndarray):
        """Set the orientation interaction matrix."""
        self.alignment_matrix = np.array(matrix)
        self.config._orientation_matrix_np = self.alignment_matrix

    def save_current_config(self):
        """Save current configuration with timestamp"""
        # Update config with current values
        self.config.n_species = self.n_species
        self.config.n_particles = self.n // self.n_species
        self.config._position_matrix_np = self.matrix
        self.config._orientation_matrix_np = self.alignment_matrix

        # Create presets directory if it doesn't exist
        os.makedirs("presets", exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"presets/config_{timestamp}.json"

        # Save configuration
        self.config.save(filename)
        return filename

    def change_species_count(self, delta: int):
        """Change the number of species by delta"""
        new_count = self.n_species + delta

        # Clamp to valid range (2-10 species)
        new_count = max(2, min(10, new_count))

        if new_count == self.n_species:
            return

        self.n_species = new_count
        self.n = self.config.n_particles * self.n_species

        # Regenerate colors
        self.colors = self.generate_colors(self.n_species)

        # Regenerate matrices for new species count (start with zeros)
        self.matrix = np.zeros((self.n_species, self.n_species))
        self.alignment_matrix = np.zeros((self.n_species, self.n_species))

        # Update config with new matrices
        self.config._position_matrix_np = self.matrix
        self.config._orientation_matrix_np = self.alignment_matrix

        # Reset all particles with new species distribution
        self.initialize_particles()

        # Show distribution info
        print(f"Reset: {self.n_species} species x {self.config.n_particles} = {self.n} particles")

    def change_particle_count(self, delta: int):
        """Change the number of particles per species by delta"""
        new_per_species = self.config.n_particles + delta
        new_per_species = max(5, min(200, new_per_species))  # Clamp per species

        if new_per_species == self.config.n_particles:
            return

        self.config.n_particles = new_per_species
        self.n = new_per_species * self.n_species
        self.initialize_particles()

        print(f"Particles: {self.config.n_particles}/species x {self.n_species} = {self.n} total")

    def change_workspace_size(self, width_delta: int = 0, height_delta: int = 0):
        """Change the workspace size"""
        new_width = self.config.width + width_delta
        new_height = self.config.height + height_delta

        # Clamp to reasonable sizes
        new_width = max(600, min(1920, new_width))
        new_height = max(400, min(1080, new_height))

        if new_width == self.config.width and new_height == self.config.height:
            return

        self.config.width = new_width
        self.config.height = new_height

        # Update base dimensions for zoom calculations
        self.base_width = new_width
        self.base_height = new_height

        # Resize the display (reset zoom when changing workspace size)
        self.screen = pygame.display.set_mode((new_width, new_height))
        self.zoom = 1.0
        self.fullscreen = False

        # Recalculate pixels per unit
        self.ppu = min(new_width / self.config.sim_width,
                       new_height / self.config.sim_height)

        # Keep particles within sim bounds (in meters)
        margin = 0.05
        self.positions[:, 0] = np.clip(self.positions[:, 0], margin, self.config.sim_width - margin)
        self.positions[:, 1] = np.clip(self.positions[:, 1], margin, self.config.sim_height - margin)

        print(f"Workspace size: {new_width}x{new_height}")

    def toggle_fullscreen(self):
        """Toggle fullscreen mode with zoom"""
        self.fullscreen = not self.fullscreen

        if self.fullscreen:
            # Get display info for fullscreen size
            info = pygame.display.Info()
            screen_width = info.current_w
            screen_height = info.current_h
            self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)

            # Recalculate ppu for fullscreen resolution
            self.ppu = min(screen_width / self.config.sim_width,
                           screen_height / self.config.sim_height)

            # Calculate zoom to fit the workspace in the screen
            zoom_x = screen_width / self.base_width
            zoom_y = screen_height / self.base_height
            self.zoom = min(zoom_x, zoom_y)

            print(f"Fullscreen ON: {screen_width}x{screen_height}, Zoom: {self.zoom:.2f}x")
        else:
            # Return to windowed mode with original size
            self.screen = pygame.display.set_mode((self.base_width, self.base_height))
            self.ppu = min(self.base_width / self.config.sim_width,
                           self.base_height / self.config.sim_height)
            self.zoom = 1.0
            print(f"Fullscreen OFF: {self.base_width}x{self.base_height}, Zoom: 1.0x")

        # Simulation space stays the same, only display scaling changes

    def detect_edge(self, mouse_pos):
        """Detect which edge the mouse is near (for resizing)"""
        if self.fullscreen:
            return None

        x, y = mouse_pos
        edge_threshold = 15  # Pixels from edge to activate resize

        edges = []
        if x <= edge_threshold:
            edges.append("left")
        elif x >= self.config.width - edge_threshold:
            edges.append("right")

        if y <= edge_threshold:
            edges.append("top")
        elif y >= self.config.height - edge_threshold:
            edges.append("bottom")

        if edges:
            return "-".join(edges)  # e.g., "left-top" for corner
        return None

    def update_cursor_for_edge(self, edge):
        """Update mouse cursor based on which edge we're near"""
        if edge is None:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        elif "left" in edge and "top" in edge:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENWSE)
        elif "right" in edge and "bottom" in edge:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENWSE)
        elif "left" in edge and "bottom" in edge:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENESW)
        elif "right" in edge and "top" in edge:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENESW)
        elif "left" in edge or "right" in edge:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEWE)
        elif "top" in edge or "bottom" in edge:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENS)

    def compute_velocities(self) -> tuple:
        """Compute velocities using continuous four-zone force kernel.

        Let tri(r) = 1 - |2r-1-β|/(1-β), fa = far_attraction, r_p = (1+β)/2

        F(r,a) = { r/β - 1,                           r < β         (repulsion)
                 { a·tri(r),                           β ≤ r < r_p   (rising triangle)
                 { a·max(fa, tri(r)),                  r_p ≤ r < 1   (falling + floor)
                 { a·fa,                               r ≥ 1         (constant long-range)

        where r is normalized distance (raw/r_max), a is k_pos matrix value.
        All boundaries are continuous. Set far_attraction=0 to disable zones 3-4 floor.
        """
        new_velocities = np.zeros_like(self.velocities)

        r_max = self.config.r_max
        beta = self.config.beta
        inv_1_minus_beta = 1.0 / (1.0 - beta) if beta < 1.0 else 1.0
        force_scale = self.config.force_scale
        far_attraction = self.config.far_attraction

        for i in range(self.n):
            # Vector from particle i to all others
            delta = self.positions - self.positions[i]

            # Distances
            dist = np.linalg.norm(delta, axis=1)
            dist[i] = np.inf  # Avoid self-interaction

            velocity_sum = np.zeros(2)

            for j in range(self.n):
                if j == i:
                    continue  # Skip self-interaction
                r = dist[j]
                r_norm = r / r_max

                dx, dy = delta[j]

                if r_norm >= 1.0:
                    # ---- Long-range attraction beyond r_max ----
                    if far_attraction > 0:
                        si = self.species[i]; sj = self.species[j]
                        k_pos = self.matrix[si, sj]
                        inv_r = 1.0 / (r + 1e-8)
                        r_hat_x, r_hat_y = dx * inv_r, dy * inv_r
                        # Constant attraction beyond r_max
                        F = k_pos * far_attraction
                        velocity_sum[0] += force_scale * F * r_hat_x
                        velocity_sum[1] += force_scale * F * r_hat_y
                    continue

                # Species lookups
                si = self.species[i]; sj = self.species[j]
                k_pos = self.matrix[si, sj]
                k_rot = self.alignment_matrix[si, sj]

                # Unit directions
                inv_r = 1.0 / (r + 1e-8)
                r_hat_x, r_hat_y = dx * inv_r, dy * inv_r
                t_hat_x, t_hat_y = -r_hat_y, r_hat_x

                # ---- Piecewise linear radial force (4 zones) ----
                if r_norm < beta:
                    # Zone 1: universal repulsion
                    F = r_norm / beta - 1.0
                else:
                    triangle = 1.0 - abs(2.0 * r_norm - 1.0 - beta) * inv_1_minus_beta
                    peak_r = 0.5 * (1.0 + beta)
                    if r_norm < peak_r:
                        # Zone 2: rising triangle (no floor, continuous from zone 1)
                        F = k_pos * triangle
                    else:
                        # Zone 3/4: falling triangle with floor
                        F = k_pos * max(far_attraction, triangle)

                velocity_sum[0] += force_scale * F * r_hat_x
                velocity_sum[1] += force_scale * F * r_hat_y

                # ---- tangential swirl: Δẋ_i += μ_swirl * k_rot * (ω_j/ω_max) * g_t(r) * t̂ ----
                swirl_weight = np.clip(1.0 - r_norm, 0.0, 1.0)
                swirl_gain = k_rot * self.config.a_rot * swirl_weight

                velocity_sum[0] += swirl_gain * t_hat_x
                velocity_sum[1] += swirl_gain * t_hat_y



            # Normalize by number of other particles
            # velocity_sum /= (self.n - 1)

            new_velocities[i] = velocity_sum

        return new_velocities

    def step(self):
        """Perform one simulation step"""
        if self.paused:
            return

        # Compute new velocities directly
        self.velocities = self.compute_velocities()

        # Clamp linear speed
        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = np.where(
            speed > self.config.max_speed,
            self.velocities * self.config.max_speed / speed,
            self.velocities
        )

        # Update positions
        self.positions += self.velocities * self.config.dt

        # Boundary conditions (reflection) — in meters
        margin = 0.05
        for i in range(2):
            # Left/top boundary
            mask = self.positions[:, i] < margin
            self.positions[mask, i] = margin
            self.velocities[mask, i] = abs(self.velocities[mask, i])

            # Right/bottom boundary
            limit = self.config.sim_width - margin if i == 0 else self.config.sim_height - margin
            mask = self.positions[:, i] > limit
            self.positions[mask, i] = limit
            self.velocities[mask, i] = -abs(self.velocities[mask, i])

    def draw(self):
        """Draw the simulation"""
        # Clear screen
        self.screen.fill((255, 255, 255))  # White background

        # Draw particles with orientation (meters → pixels via ppu)
        for i in range(self.n):
            color = self.colors[self.species[i]]
            pos = self.positions[i]

            # Convert meters to screen pixels
            x = int(pos[0] * self.ppu * self.zoom)
            y = int(pos[1] * self.ppu * self.zoom)

            r = max(3, int(0.04 * self.ppu * self.zoom))
            pygame.gfxdraw.aacircle(self.screen, x, y, r, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, color)

        # Draw info panel if enabled
        if self.show_info:
            self.draw_info()

        # Draw matrix editor if enabled
        if self.show_matrix:
            self.draw_matrix()

    def draw_info(self):
        """Draw information panel"""
        info_lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Workspace: {self.config.sim_width:.1f}x{self.config.sim_height:.1f}m ({self.config.width}x{self.config.height}px)",
            f"Particles: {self.config.n_particles}/species x {self.n_species} = {self.n}",
            f"Species: {self.n_species} (2-10)",
            f"Active Matrix: {'POSITION' if self.current_matrix == 'position' else 'ORIENTATION'}",
            "",
            "Controls:",
            "F11/F - Toggle fullscreen",
            "Mouse at edges - Drag or scroll to resize",
            "UP/DOWN - Change species count",
            "LEFT/RIGHT - Change particle count (-/+ 50)",
            "SHIFT+LEFT/RIGHT - Change workspace width",
            "SHIFT+UP/DOWN - Change workspace height",
            "M - Toggle matrix editor",
            "TAB - Switch matrix (Position/Orientation)",
            "O - Toggle orientation display",
            "SPACE - Pause/Resume",
            "R - Reset positions",
            "S - Save current configuration",
            "I - Toggle info",
            "Q/ESC - Quit",
            "",
            "Matrix Editor (when M pressed):",
            "TAB - Switch between matrices",
            "WASD - Navigate matrix",
            "+/- - Modify value"
        ]

        y = 10
        for line in info_lines:
            if line:
                text = self.font.render(line, True, (200, 200, 200))
                self.screen.blit(text, (10, y))
            y += 25

        # Draw pause indicator
        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 100, 100))
            rect = pause_text.get_rect(center=(self.config.width // 2, 30))
            self.screen.blit(pause_text, rect)

    def draw_matrix(self):
        """Draw the interaction matrix for editing"""
        # Matrix position on screen
        matrix_x = self.config.width - 350
        matrix_y = 100  # Moved down from 50
        cell_size = 40

        # Select which matrix to display
        if self.current_matrix == "position":
            matrix = self.matrix
            matrix_name = "POSITION MATRIX (K_pos)"
            title_color = (100, 255, 100)  # Green for position
        else:
            matrix = self.alignment_matrix
            matrix_name = "ORIENTATION MATRIX (K_rot)"
            title_color = (100, 150, 255)  # Blue for orientation

        # Title
        title = self.font.render(matrix_name, True, title_color)
        self.screen.blit(title, (matrix_x, matrix_y - 45))  # Moved up from -30

        # Draw species labels and matrix
        for i in range(self.n_species):
            # Row label (FROM species)
            color = self.colors[i]
            label = self.font.render(f"S{i+1}", True, color)
            self.screen.blit(label, (matrix_x - 35, matrix_y + 15 + i * cell_size))

            # Column label (TO species)
            label = self.font.render(f"S{i+1}", True, color)
            self.screen.blit(label, (matrix_x + 15 + i * cell_size, matrix_y - 20))

            for j in range(self.n_species):
                x = matrix_x + j * cell_size
                y = matrix_y + i * cell_size
                value = matrix[i, j]

                # Draw cell background
                if [i, j] == self.matrix_cursor:
                    # Highlight selected cell
                    pygame.draw.rect(self.screen, (100, 100, 255), (x, y, cell_size, cell_size), 3)
                else:
                    pygame.draw.rect(self.screen, (60, 60, 60), (x, y, cell_size, cell_size), 1)

                # Color code the value
                if value > 0:
                    # Positive = attraction (green)
                    intensity = min(255, int(abs(value) * 255))
                    color = (0, intensity, 0)
                elif value < 0:
                    # Negative = repulsion (red)
                    intensity = min(255, int(abs(value) * 255))
                    color = (intensity, 0, 0)
                else:
                    # Neutral (gray)
                    color = (100, 100, 100)

                # Draw value
                text = self.font.render(f"{value:.1f}", True, color)
                text_rect = text.get_rect(center=(x + cell_size // 2, y + cell_size // 2))
                self.screen.blit(text, text_rect)

        # Instructions
        instructions = [
            "TAB: Switch matrix",
            "WASD: Navigate cells",
            "+/-: Change value",
            "M: Hide matrix editor"
        ]
        y_offset = matrix_y + (self.n_species + 1) * cell_size
        for i, instruction in enumerate(instructions):
            text = self.font.render(instruction, True, (180, 180, 180))
            self.screen.blit(text, (matrix_x, y_offset + i * 25))

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = pygame.mouse.get_pos()

                # Handle window edge resizing
                if self.dragging_edge:
                    # We're dragging to resize
                    dx = mouse_pos[0] - self.drag_start_pos[0]
                    dy = mouse_pos[1] - self.drag_start_pos[1]

                    new_width = self.drag_start_size[0]
                    new_height = self.drag_start_size[1]

                    if "right" in self.dragging_edge:
                        new_width += dx
                    elif "left" in self.dragging_edge:
                        new_width -= dx

                    if "bottom" in self.dragging_edge:
                        new_height += dy
                    elif "top" in self.dragging_edge:
                        new_height -= dy

                    # Apply size constraints
                    new_width = max(600, min(1920, new_width))
                    new_height = max(400, min(1080, new_height))

                    # Resize if changed
                    if new_width != self.config.width or new_height != self.config.height:
                        self.change_workspace_size(new_width - self.config.width, new_height - self.config.height)
                else:
                    # Just update cursor based on edge proximity
                    edge = self.detect_edge(mouse_pos)
                    self.update_cursor_for_edge(edge)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_pos = pygame.mouse.get_pos()
                    edge = self.detect_edge(mouse_pos)
                    if edge:
                        # Start dragging to resize
                        self.dragging_edge = edge
                        self.drag_start_pos = mouse_pos
                        self.drag_start_size = (self.config.width, self.config.height)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    # Stop dragging
                    self.dragging_edge = None
                    self.drag_start_pos = None
                    self.drag_start_size = None

            elif event.type == pygame.MOUSEWHEEL:
                # Allow mouse wheel resizing when at edges
                mouse_pos = pygame.mouse.get_pos()
                edge = self.detect_edge(mouse_pos)
                if edge and not self.fullscreen:
                    # Resize based on scroll direction
                    resize_amount = 25 * event.y  # Positive for scroll up, negative for down

                    if "left" in edge or "right" in edge:
                        self.change_workspace_size(resize_amount, 0)
                    if "top" in edge or "bottom" in edge:
                        self.change_workspace_size(0, resize_amount)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    # Reset all particles to initial state
                    self.initialize_particles()
                    print("Reset particles to initial state")

                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info

                elif event.key == pygame.K_o:
                    # Toggle orientation display
                    self.show_orientations = not self.show_orientations
                    print(f"Show orientations: {self.show_orientations}")

                elif event.key == pygame.K_F11 or event.key == pygame.K_f:
                    # Toggle fullscreen mode
                    self.toggle_fullscreen()

                elif event.key == pygame.K_TAB:
                    # Switch between position and orientation matrices
                    if self.show_matrix:
                        self.current_matrix = "orientation" if self.current_matrix == "position" else "position"
                        print(f"Editing: {self.current_matrix} matrix")

                elif event.key == pygame.K_m:
                    # Toggle matrix editor
                    self.show_matrix = not self.show_matrix
                    if self.show_matrix:
                        # Ensure cursor is in bounds
                        self.matrix_cursor[0] = min(self.matrix_cursor[0], self.n_species - 1)
                        self.matrix_cursor[1] = min(self.matrix_cursor[1], self.n_species - 1)

                elif event.key == pygame.K_UP and not self.show_matrix:
                    # Check if SHIFT is held for workspace size change
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        # Decrease workspace height
                        self.change_workspace_size(height_delta=-50)
                    else:
                        # Increase species count
                        self.change_species_count(1)
                        print(f"Species count: {self.n_species}")

                elif event.key == pygame.K_DOWN and not self.show_matrix:
                    # Check if SHIFT is held for workspace size change
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        # Increase workspace height
                        self.change_workspace_size(height_delta=50)
                    else:
                        # Decrease species count
                        self.change_species_count(-1)
                        print(f"Species count: {self.n_species}")

                elif event.key == pygame.K_LEFT and not self.show_matrix:
                    # Check if SHIFT is held for workspace size change
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        # Decrease workspace width
                        self.change_workspace_size(width_delta=-50)
                    else:
                        # Decrease particle count
                        self.change_particle_count(-50)

                elif event.key == pygame.K_RIGHT and not self.show_matrix:
                    # Check if SHIFT is held for workspace size change
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        # Increase workspace width
                        self.change_workspace_size(width_delta=50)
                    else:
                        # Increase particle count
                        self.change_particle_count(50)

                # Matrix editor navigation (WASD)
                elif self.show_matrix:
                    if event.key == pygame.K_w:
                        # Move up in matrix
                        self.matrix_cursor[0] = max(0, self.matrix_cursor[0] - 1)
                    elif event.key == pygame.K_s:
                        # Move down in matrix
                        self.matrix_cursor[0] = min(self.n_species - 1, self.matrix_cursor[0] + 1)
                    elif event.key == pygame.K_a:
                        # Move left in matrix
                        self.matrix_cursor[1] = max(0, self.matrix_cursor[1] - 1)
                    elif event.key == pygame.K_d:
                        # Move right in matrix
                        self.matrix_cursor[1] = min(self.n_species - 1, self.matrix_cursor[1] + 1)
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        # Increase matrix value
                        i, j = self.matrix_cursor
                        if self.current_matrix == "position":
                            self.matrix[i, j] = min(1.0, self.matrix[i, j] + 0.1)
                            self.config._position_matrix_np = self.matrix  # Update config
                            print(f"Position Matrix[{i},{j}] = {self.matrix[i, j]:.2f}")
                        else:
                            self.alignment_matrix[i, j] = min(1.0, self.alignment_matrix[i, j] + 0.1)
                            self.config._orientation_matrix_np = self.alignment_matrix  # Update config
                            print(f"Orientation Matrix[{i},{j}] = {self.alignment_matrix[i, j]:.2f}")
                    elif event.key == pygame.K_MINUS:
                        # Decrease matrix value
                        i, j = self.matrix_cursor
                        if self.current_matrix == "position":
                            self.matrix[i, j] = max(-1.0, self.matrix[i, j] - 0.1)
                            self.config._position_matrix_np = self.matrix  # Update config
                            print(f"Position Matrix[{i},{j}] = {self.matrix[i, j]:.2f}")
                        else:
                            self.alignment_matrix[i, j] = max(-1.0, self.alignment_matrix[i, j] - 0.1)
                            self.config._orientation_matrix_np = self.alignment_matrix  # Update config
                            print(f"Orientation Matrix[{i},{j}] = {self.alignment_matrix[i, j]:.2f}")

                elif event.key == pygame.K_s:
                    # Save current configuration
                    filename = self.save_current_config()
                    print(f"Configuration saved to {filename}")

        return True

    def run(self):
        """Main simulation loop"""
        running = True

        while running:
            # Handle events
            running = self.handle_events()

            # Update simulation
            self.step()

            # Draw everything
            self.draw()

            # Update display
            pygame.display.flip()

            # Control frame rate
            self.clock.tick(60)  # 60 FPS

        pygame.quit()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Particle Life Simulation')
    parser.add_argument('--load', type=str, help='Path to configuration file to load')
    args = parser.parse_args()

    print("Starting Particle Life Simulation...")
    print("Press SPACE to pause, I to toggle info, Q to quit")

    # Load configuration or create default
    if args.load:
        if os.path.exists(args.load):
            print(f"Loading configuration from {args.load}")
            config = Config.load(args.load)
        else:
            print(f"Configuration file {args.load} not found, using default")
            config = Config()
    else:
        print("Starting with default configuration")
        config = Config()

    sim = ParticleLife(config)
    sim.run()

if __name__ == "__main__":
    main()