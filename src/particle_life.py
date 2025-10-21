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

@dataclass
class Config:
    """Simulation configuration"""
    width: int = 1200
    height: int = 800
    n_species: int = 5
    n_particles: int = 500
    dt: float = 0.05
    damping: float = 0.995
    max_speed: float = 300.0
    r_rep: float = 4.0
    r_att: float = 24.0
    r_cut: float = 36.0
    a_rep: float = 5.0
    a_att: float = 2.0
    seed: int = 42
    angular_damping: float = 0.98  # Damping for angular velocity
    max_angular_speed: float = 5.0  # Max angular velocity (radians/sec)
    alignment_strength: float = 0.5  # Strength of orientation alignment

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

    def compute_forces(self) -> tuple:
        """Compute forces and torques between all particles"""
        forces = np.zeros_like(self.positions)
        torques = np.zeros(self.n)

        for i in range(self.n):
            # Vector from particle i to all others
            delta = self.positions - self.positions[i]

            # Distances
            dist = np.linalg.norm(delta, axis=1)
            dist[i] = np.inf  # Avoid self-interaction

            # Find neighbors within cutoff
            neighbors = np.where(dist < self.config.r_cut)[0]

            for j in neighbors:
                r = dist[j]
                dx, dy = delta[j]

                # Unit vector
                if r > 0:
                    dx /= r
                    dy /= r

                # Species interaction
                si = self.species[i]
                sj = self.species[j]
                k = self.matrix[si, sj]

                # Radial force
                if r < self.config.r_rep:
                    # Strong repulsion
                    f_mag = -self.config.a_rep * (1 - r / self.config.r_rep)
                elif r < self.config.r_att:
                    # Attraction/repulsion based on matrix
                    f_mag = k * self.config.a_att * (r - self.config.r_rep) / (self.config.r_att - self.config.r_rep)
                else:
                    # Weak tail
                    f_mag = k * self.config.a_att * 0.2 * (1 - (r - self.config.r_att) / (self.config.r_cut - self.config.r_att))

                # Apply force
                forces[i, 0] += f_mag * dx
                forces[i, 1] += f_mag * dy

                # Compute orientation alignment torque
                angle_diff = self.orientations[j] - self.orientations[i]
                # Normalize angle difference to [-pi, pi]
                angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

                # Get alignment strength from alignment matrix
                alignment_k = self.alignment_matrix[si, sj]

                # Apply alignment torque (scaled by distance)
                torque_mag = alignment_k * self.config.alignment_strength * (1.0 - r / self.config.r_cut)
                torques[i] += torque_mag * angle_diff

        return forces, torques

    def step(self):
        """Perform one simulation step"""
        if self.paused:
            return

        # Compute forces and torques
        forces, torques = self.compute_forces()

        # Update linear velocities
        self.velocities += forces * self.config.dt

        # Update angular velocities
        self.angular_velocities += torques * self.config.dt

        # Apply damping
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

        # Draw particles with orientation
        for i in range(self.n):
            color = self.colors[self.species[i]]
            pos = self.positions[i]
            x, y = int(pos[0]), int(pos[1])

            # Always draw as circle
            pygame.draw.circle(self.screen, color, (x, y), 5)

            if self.show_orientations:
                # Draw black orientation line inside the circle
                angle = self.orientations[i]
                line_length = 4  # Keep line inside the circle (radius is 5)
                end_x = x + line_length * np.cos(angle)
                end_y = y + line_length * np.sin(angle)
                pygame.draw.line(self.screen, (0, 0, 0), (x, y), (int(end_x), int(end_y)), 2)

        # Draw info panel if enabled
        if self.show_info:
            self.draw_info()

        # Draw matrix editor if enabled
        if self.show_matrix:
            self.draw_matrix()

    def draw_info(self):
        """Draw information panel"""
        info_lines = [
            f"Particles: {self.n}",
            f"Species: {self.n_species} (2-10)",
            f"Preset: {self.selected_preset}",
            "",
            "Controls:",
            "UP/DOWN - Change species count",
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
                text = self.font.render(line, True, (50, 50, 50))  # Dark text for white background
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
        title = self.font.render(matrix_name, True, (0, 0, 0))  # Black text for white background
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
                    pygame.draw.rect(self.screen, (200, 200, 200), (x, y, cell_size, cell_size), 1)  # Light gray for white background

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
            text = self.font.render(instruction, True, (80, 80, 80))  # Dark gray for white background
            self.screen.blit(text, (matrix_x, y_offset + i * 25))

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

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
                    # Increase species count
                    self.change_species_count(1)
                    print(f"Species count: {self.n_species}")

                elif event.key == pygame.K_DOWN and not self.show_matrix:
                    # Decrease species count
                    self.change_species_count(-1)
                    print(f"Species count: {self.n_species}")

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