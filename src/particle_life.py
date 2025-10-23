#!/usr/bin/env python3
"""
Standalone Particle Life Simulation with Pygame
Interactive particle simulation with real-time parameter adjustment
"""

import pygame
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import colorsys
import math

@dataclass
class Config:
    """Simulation configuration"""
    width: int = 200
    height: int = 200
    n_species: int = 2
    n_particles: int = 20
    dt: float = 0.05
    damping: float = 0.995
    max_speed: float = 300.0
    r_rep: float = 5.0
    r_att: float = 50.0
    r_cut: float = 100.0
    a_rep: float = 5.0
    a_att: float = 2.0
    seed: int = 42
    angular_damping: float = 0.98  # Damping for angular velocity
    max_angular_speed: float = 5.0  # Max angular velocity (radians/sec)
    alignment_strength: float = 0.5  # Strength of orientation alignment
    blend_factor: float = 0.15  # Blending factor for velocity updates (0-1)

class ParticleLife:
    """Main particle life simulation"""

    def __init__(self, config: Config):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        # Initialize particles
        self.n = config.n_particles
        self.n_species = config.n_species

        # Positions and velocities
        self.positions = self.rng.uniform(
            [50, 50],
            [config.width - 50, config.height - 50],
            (self.n, 2)
        )
        self.velocities = self.rng.randn(self.n, 2) * 10

        # Orientations and angular velocities
        self.orientations = self.rng.uniform(0, 2 * np.pi, self.n)
        self.angular_velocities = self.rng.randn(self.n) * 0.1

        # Species assignment
        self.species = self.rng.randint(0, self.n_species, self.n)

        # Interaction matrices
        self.matrix = self.generate_matrix("chaos")  # Position attraction/repulsion matrix
        self.alignment_matrix = self.generate_matrix("symbiosis")  # Orientation alignment matrix

        # Colors for species
        self.colors = self.generate_colors(self.n_species)

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption("Particle Life Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        # UI state
        self.paused = False
        self.show_info = True
        self.show_matrix = False
        self.selected_preset = "chaos"
        self.matrix_cursor = [0, 0]  # Row, Column for matrix editing
        self.current_matrix = "position"  # Which matrix is being edited: "position" or "orientation"
        self.show_orientations = True  # Whether to show particle orientations
        self.fullscreen = False  # Fullscreen mode flag
        self.zoom = 1.0  # Zoom factor for display scaling
        self.base_width = config.width  # Store base dimensions for scaling
        self.base_height = config.height
        self.dragging_edge = None  # Track which edge is being dragged for resize
        self.drag_start_pos = None  # Mouse position when drag started
        self.drag_start_size = None  # Window size when drag started

    def generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for species"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors

    def generate_matrix(self, preset: str) -> np.ndarray:
        """Generate interaction matrix based on preset"""
        if preset == "chaos":
            # Generate chaotic matrix for any size
            matrix = self.rng.uniform(-0.9, 0.9, (self.n_species, self.n_species))
            np.fill_diagonal(matrix, self.rng.uniform(0.2, 0.5, self.n_species))
            return matrix
        elif preset == "symbiosis":
            # Symbiotic relationships
            matrix = np.zeros((self.n_species, self.n_species))
            for i in range(self.n_species):
                for j in range(self.n_species):
                    if i == j:
                        matrix[i, j] = 0.2  # Mild self-attraction
                    else:
                        matrix[i, j] = 0.6 if (i + j) % 2 == 0 else -0.4
            return matrix
        elif preset == "predator":
            # Predator-prey dynamics
            matrix = np.zeros((self.n_species, self.n_species))
            for i in range(self.n_species):
                j = (i + 1) % self.n_species
                matrix[i, j] = 0.9  # Chase next species
                matrix[j, i] = -0.8  # Flee from previous
            return matrix
        elif preset == "random":
            return self.rng.uniform(-1, 1, (self.n_species, self.n_species))
        else:
            # Default neutral
            return np.zeros((self.n_species, self.n_species))

    def change_species_count(self, delta: int):
        """Change the number of species by delta"""
        new_count = self.n_species + delta

        # Clamp to valid range (2-10 species)
        new_count = max(2, min(10, new_count))

        if new_count == self.n_species:
            return

        self.n_species = new_count

        # Regenerate colors
        self.colors = self.generate_colors(self.n_species)

        # Regenerate matrices for new species count
        self.matrix = self.generate_matrix(self.selected_preset)
        self.alignment_matrix = self.generate_matrix("symbiosis")

        # Reassign particles to species (keeping existing assignments where possible)
        for i in range(self.n):
            if self.species[i] >= self.n_species:
                # Reassign particles that are now out of range
                self.species[i] = self.rng.randint(0, self.n_species)

    def change_particle_count(self, delta: int):
        """Change the number of particles by delta"""
        new_count = self.n + delta
        new_count = max(50, min(2000, new_count))  # Clamp to 50-2000

        if new_count == self.n:
            return

        if new_count > self.n:
            # Add new particles
            num_new = new_count - self.n
            new_positions = self.rng.uniform(
                [50, 50],
                [self.config.width - 50, self.config.height - 50],
                (num_new, 2)
            )
            new_velocities = self.rng.randn(num_new, 2) * 10
            new_orientations = self.rng.uniform(0, 2 * np.pi, num_new)
            new_angular_velocities = self.rng.randn(num_new) * 0.1
            new_species = self.rng.randint(0, self.n_species, num_new)

            self.positions = np.vstack([self.positions, new_positions])
            self.velocities = np.vstack([self.velocities, new_velocities])
            self.orientations = np.concatenate([self.orientations, new_orientations])
            self.angular_velocities = np.concatenate([self.angular_velocities, new_angular_velocities])
            self.species = np.concatenate([self.species, new_species])
        else:
            # Remove particles
            self.positions = self.positions[:new_count]
            self.velocities = self.velocities[:new_count]
            self.orientations = self.orientations[:new_count]
            self.angular_velocities = self.angular_velocities[:new_count]
            self.species = self.species[:new_count]

        self.n = new_count
        print(f"Particle count: {self.n}")

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

        # Keep particles within new bounds
        self.positions[:, 0] = np.clip(self.positions[:, 0], 10, new_width - 10)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 10, new_height - 10)

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

            # Calculate zoom to fit the workspace in the screen
            zoom_x = screen_width / self.base_width
            zoom_y = screen_height / self.base_height
            self.zoom = min(zoom_x, zoom_y)  # Use the smaller zoom to fit everything

            print(f"Fullscreen ON: {screen_width}x{screen_height}, Zoom: {self.zoom:.2f}x")
        else:
            # Return to windowed mode with original size
            self.screen = pygame.display.set_mode((self.base_width, self.base_height))
            self.zoom = 1.0
            print(f"Fullscreen OFF: {self.base_width}x{self.base_height}, Zoom: 1.0x")

        # Note: We don't change config.width/height or particle positions
        # The simulation space stays the same, only the display is scaled

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
        """Compute velocities directly from neighbor interactions"""
        new_velocities = np.zeros_like(self.velocities)
        new_angular_velocities = np.zeros(self.n)

        for i in range(self.n):
            # Vector from particle i to all others
            delta = self.positions - self.positions[i]

            # Distances
            dist = np.linalg.norm(delta, axis=1)
            dist[i] = np.inf  # Avoid self-interaction

            # Find neighbors within cutoff
            neighbors = np.where(dist < self.config.r_cut)[0]

            if len(neighbors) == 0:
                continue

            velocity_sum = np.zeros(2)
            angular_vel_sum = 0.0

            for j in neighbors:
                r = dist[j]
                dx, dy = delta[j]

                # Species interaction matrices
                si = self.species[i]
                sj = self.species[j]
                k_pos = self.matrix[si, sj]  # Position interaction
                k_rot = self.alignment_matrix[si, sj]  # Rotation interaction

                # Linear velocity contribution
                # Attraction/repulsion part
                if r < self.config.r_rep:
                    # Strong repulsion when too close
                    repulsion = -self.config.a_rep * (1 - r / self.config.r_rep)
                    velocity_sum[0] += repulsion * dx / r
                    velocity_sum[1] += repulsion * dy / r
                else:
                    # Attraction based on position matrix
                    attraction = k_pos * self.config.a_att * (1 - r / self.config.r_cut)
                    velocity_sum[0] += attraction * dx / r
                    velocity_sum[1] += attraction * dy / r

                # Rotation coupling to linear velocity
                # k_rot * neighbor's angular velocity / max angular velocity * dist_vector / dist^3
                if r > 0.1:  # Avoid division issues
                    angular_coupling = k_rot * (self.angular_velocities[j] / self.config.max_angular_speed) * 10.0  # Scale up for visibility
                    velocity_sum[0] += angular_coupling * dx / (r * r)
                    velocity_sum[1] += angular_coupling * dy / (r * r)

                # Angular velocity contribution
                # k_rot * (angular_velocity_j - angular_velocity_i) / dist
                angular_diff = self.angular_velocities[j] - self.angular_velocities[i]
                angular_vel_sum += 0.1 + k_rot * angular_diff / r

            # Set new velocities
            new_velocities[i] = velocity_sum
            new_angular_velocities[i] = angular_vel_sum / len(neighbors)

        return new_velocities, new_angular_velocities

    def step(self):
        """Perform one simulation step"""
        if self.paused:
            return

        # Compute new velocities directly
        new_velocities, new_angular_velocities = self.compute_velocities()

        # Blend with current velocities for smoothness
        self.velocities = self.config.blend_factor * new_velocities + (1 - self.config.blend_factor) * self.velocities

        # Blend angular velocities for smoother rotation
        self.angular_velocities = 0.3 * new_angular_velocities + 0.7 * self.angular_velocities

        # Apply damping for stability
        self.velocities *= self.config.damping
        self.angular_velocities *= self.config.angular_damping

        # Clamp linear speed
        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = np.where(
            speed > self.config.max_speed,
            self.velocities * self.config.max_speed / speed,
            self.velocities
        )

        # Clamp angular speed
        self.angular_velocities = np.clip(
            self.angular_velocities,
            -self.config.max_angular_speed,
            self.config.max_angular_speed
        )

        # Update positions and orientations
        self.positions += self.velocities * self.config.dt
        self.orientations += self.angular_velocities * self.config.dt

        # Normalize orientations to [0, 2*pi]
        self.orientations = np.mod(self.orientations, 2 * np.pi)

        # Boundary conditions (reflection)
        for i in range(2):
            # Left/top boundary
            mask = self.positions[:, i] < 10
            self.positions[mask, i] = 10
            self.velocities[mask, i] = abs(self.velocities[mask, i])

            # Right/bottom boundary
            limit = self.config.width - 10 if i == 0 else self.config.height - 10
            mask = self.positions[:, i] > limit
            self.positions[mask, i] = limit
            self.velocities[mask, i] = -abs(self.velocities[mask, i])

    def draw(self):
        """Draw the simulation"""
        # Clear screen
        self.screen.fill((255, 255, 255))  # White background

        # Draw particles with orientation (with zoom scaling)
        for i in range(self.n):
            color = self.colors[self.species[i]]
            pos = self.positions[i]

            # Apply zoom to position
            x = int(pos[0] * self.zoom)
            y = int(pos[1] * self.zoom)

            if self.show_orientations:
                # Draw as oriented triangle (scaled with zoom)
                angle = self.orientations[i]
                size = 6 * self.zoom  # Scale size with zoom

                # Calculate triangle points
                front_x = x + size * np.cos(angle)
                front_y = y + size * np.sin(angle)

                left_angle = angle + 2.5
                left_x = x + size * 0.7 * np.cos(left_angle)
                left_y = y + size * 0.7 * np.sin(left_angle)

                right_angle = angle - 2.5
                right_x = x + size * 0.7 * np.cos(right_angle)
                right_y = y + size * 0.7 * np.sin(right_angle)

                # Draw triangle
                points = [(front_x, front_y), (left_x, left_y), (right_x, right_y)]
                pygame.draw.polygon(self.screen, color, points)

                # Draw small circle at center (scaled)
                pygame.draw.circle(self.screen, color, (x, y), max(1, int(2 * self.zoom)))
            else:
                # Draw as simple circle (scaled)
                pygame.draw.circle(self.screen, color, (x, y), max(1, int(4 * self.zoom)))

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
            f"Workspace: {self.config.width}x{self.config.height}",
            f"Particles: {self.n} (50-2000)",
            f"Species: {self.n_species} (2-10)",
            f"Preset: {self.selected_preset}",
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
            "1-5 - Select presets",
            "I - Toggle info",
            "Q/ESC - Quit",
            "",
            "Matrix Editor (when M pressed):",
            "TAB - Switch between matrices",
            "WASD - Navigate matrix",
            "+/- - Modify value",
            "",
            "Presets:",
            "1 - Chaos",
            "2 - Symbiosis",
            "3 - Predator-Prey",
            "4 - Random",
            "5 - Neutral"
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
        matrix_y = 50
        cell_size = 40

        # Select which matrix to display
        if self.current_matrix == "position":
            matrix = self.matrix
            matrix_name = "Position Matrix"
        else:
            matrix = self.alignment_matrix
            matrix_name = "Orientation Matrix"

        # Title
        title = self.font.render(matrix_name, True, (255, 255, 255))
        self.screen.blit(title, (matrix_x, matrix_y - 30))

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
            "WASD: Navigate",
            "+/-: Change value",
            "M: Hide matrix"
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
                    # Reset positions and orientations
                    self.positions = self.rng.uniform(
                        [50, 50],
                        [self.config.width - 50, self.config.height - 50],
                        (self.n, 2)
                    )
                    self.velocities = self.rng.randn(self.n, 2) * 10
                    self.orientations = self.rng.uniform(0, 2 * np.pi, self.n)
                    self.angular_velocities = self.rng.randn(self.n) * 0.1

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
                            print(f"Position Matrix[{i},{j}] = {self.matrix[i, j]:.2f}")
                        else:
                            self.alignment_matrix[i, j] = min(1.0, self.alignment_matrix[i, j] + 0.1)
                            print(f"Orientation Matrix[{i},{j}] = {self.alignment_matrix[i, j]:.2f}")
                    elif event.key == pygame.K_MINUS:
                        # Decrease matrix value
                        i, j = self.matrix_cursor
                        if self.current_matrix == "position":
                            self.matrix[i, j] = max(-1.0, self.matrix[i, j] - 0.1)
                            print(f"Position Matrix[{i},{j}] = {self.matrix[i, j]:.2f}")
                        else:
                            self.alignment_matrix[i, j] = max(-1.0, self.alignment_matrix[i, j] - 0.1)
                            print(f"Orientation Matrix[{i},{j}] = {self.alignment_matrix[i, j]:.2f}")

                elif event.key == pygame.K_1:
                    self.selected_preset = "chaos"
                    self.matrix = self.generate_matrix("chaos")

                elif event.key == pygame.K_2:
                    self.selected_preset = "symbiosis"
                    self.matrix = self.generate_matrix("symbiosis")

                elif event.key == pygame.K_3:
                    self.selected_preset = "predator"
                    self.matrix = self.generate_matrix("predator")

                elif event.key == pygame.K_4:
                    self.selected_preset = "random"
                    self.matrix = self.generate_matrix("random")

                elif event.key == pygame.K_5:
                    self.selected_preset = "neutral"
                    self.matrix = self.generate_matrix("neutral")

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
    print("Starting Particle Life Simulation...")
    print("Press SPACE to pause, I to toggle info, Q to quit")

    config = Config()
    sim = ParticleLife(config)
    sim.run()

if __name__ == "__main__":
    main()