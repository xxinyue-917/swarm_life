#!/usr/bin/env python3
"""
Multi-Species Manual Control Demo

Demonstrates manual control of N species (2-10) arranged side-by-side.
Arrow keys control swarm movement:
- Left/Right: Turn the swarm
- Up/Down: Adjust speed

The K_rot matrix is dynamically updated based on input:
- Translation mode: Adjacent species have opposite-sign K_rot entries
- Rotation mode: All species have same-sign K_rot entries
- Blending: Arrow keys smoothly blend between modes

Controls:
    ←/→: Turn left/right
    ↑/↓: Speed up/slow down
    +/-: Add/remove species (2-10)
    C: Converge formation
    R: Reset positions
    SPACE: Pause/Resume
    I: Toggle info panel
    Q/ESC: Quit
"""

import pygame
import numpy as np
from particle_life import Config, ParticleLife


def generate_translation_matrix(n_species: int, strength: float) -> np.ndarray:
    """
    Generate K_rot matrix for forward translation.

    Only adjacent species pairs are coupled (like joints in a chain).
    Antisymmetric: K[i, i+1] = +strength, K[i+1, i] = -strength
    This creates net forward motion along the chain.
    """
    K = np.zeros((n_species, n_species))

    for i in range(n_species - 1):
        K[i, i + 1] = +strength  # 前一个对后一个：正
        K[i + 1, i] = -strength  # 后一个对前一个：负

    return K


def generate_rotation_matrix(n_species: int, strength: float) -> np.ndarray:
    """
    Generate K_rot matrix for collective rotation.

    Only adjacent species pairs are coupled (like joints in a chain).
    Symmetric: K[i, i+1] = K[i+1, i] = strength
    """
    K = np.zeros((n_species, n_species))

    if n_species <= 1:
        return K

    for i in range(n_species - 1):
        K[i, i + 1] = strength
        K[i + 1, i] = strength

    return K


def generate_position_matrix(n_species: int,
                             self_cohesion: float = 0.6,
                             cross_attraction: float = 0.4) -> np.ndarray:
    """
    Generate K_pos matrix for species cohesion.

    Args:
        n_species: Number of species
        self_cohesion: Diagonal values (attraction within species)
        cross_attraction: Off-diagonal values (attraction between neighbors)
    """
    K = np.zeros((n_species, n_species))
    for i in range(n_species):
        K[i, i] = self_cohesion
        for j in range(n_species):
            if abs(i - j) == 1:  # Only neighbors
                K[i, j] = cross_attraction
    return K


class MultiSpeciesDemo(ParticleLife):
    """
    Demo for manual control of N species swarm.

    Species are arranged side-by-side horizontally.
    Arrow keys control turn and speed via K_rot matrix manipulation.
    """

    def __init__(self, n_species: int = 3, n_particles: int = 150):
        # Create config with initial species count
        config = Config(
            n_particles=n_particles,
            n_species=n_species,
            position_matrix=generate_position_matrix(n_species).tolist(),
            orientation_matrix=generate_translation_matrix(n_species, 0.5).tolist(),
        )

        # Initialize base simulation
        super().__init__(config, headless=False)

        # Control state
        self.turn_input = 0.0       # -1 (full left) to +1 (full right)
        self.speed_input = 1.0      # 0 (stop) to 1 (full speed)
        self.base_k_rot = 0.8      # Base rotation matrix strength

        # Input smoothing
        self.turn_decay = 0.95      # How quickly turn input returns to 0
        self.speed_decay = 0.98     # How quickly speed input stabilizes

        # Formation parameters
        self.group_spacing = 80     # Pixels between species group centers

        # Converge mode
        self.converge_active = False
        self.normal_cross_attraction = 0.4  # Increased for better line cohesion
        self.converge_cross_attraction = 0.6

        # Matrix editing mode
        self.matrix_edit_mode = False
        self.edit_row = 0
        self.edit_col = 0
        self.editing_k_rot = True  # True = K_rot, False = K_pos

        # GUI visibility
        self.hide_gui = False

        # Initialize particles in side-by-side formation
        self._initialize_side_by_side()

        # Pre-compute base matrices
        self._update_base_matrices()

        # Override window title
        pygame.display.set_caption("Multi-Species Manual Control Demo")

        print("=" * 60)
        print("Multi-Species Manual Control Demo")
        print("=" * 60)
        print(f"Species: {self.n_species}  Particles: {self.n}")
        print("")
        print("Controls:")
        print("  ←/→     Turn left/right")
        print("  ↑/↓     Speed up/slow down")
        print("  +/-     Add/remove species")
        print("  C       Hold to converge formation")
        print("  R       Reset positions")
        print("  SPACE   Pause")
        print("  H       Hide/show all GUI")
        print("  I       Toggle info panel")
        print("")
        print("Matrix Editing:")
        print("  M       Toggle matrix edit mode")
        print("  TAB     Switch K_rot/K_pos")
        print("  WASD    Navigate cells")
        print("  E/X     Increase/decrease value")
        print("=" * 60)

    def _initialize_side_by_side(self):
        """Arrange species in horizontal line formation."""
        center_x = self.config.width / 2
        center_y = self.config.height / 2

        # Calculate total width of formation
        total_width = (self.n_species - 1) * self.group_spacing
        start_x = center_x - total_width / 2

        particles_per_species = self.n // self.n_species

        for i in range(self.n):
            species_id = min(i // particles_per_species, self.n_species - 1)
            self.species[i] = species_id

            # Group center for this species
            group_center_x = start_x + species_id * self.group_spacing
            group_center_y = center_y

            # Random offset within group
            self.positions[i, 0] = group_center_x + self.rng.uniform(-20, 20)
            self.positions[i, 1] = group_center_y + self.rng.uniform(-20, 20)

            # Initial velocity (slight forward motion)
            self.velocities[i] = np.array([0.0, -1.0])  # Moving up initially

    def _update_base_matrices(self):
        """Pre-compute translation base matrix (rotation computed dynamically)."""
        self.K_translation = generate_translation_matrix(self.n_species, 1.0)

    def update_matrices_from_input(self):
        """
        Update K_rot matrix based on current turn and speed input.

        Blends between translation mode (turn_input ≈ 0) and
        rotation mode (turn_input ≈ ±1).

        Uses differential rotation: outer species in turn arc get more force
        to maintain line formation.
        """
        # Skip auto-update when in matrix edit mode (allow manual edits to persist)
        if self.matrix_edit_mode:
            return

        # Blend factor: 0 = pure translation, 1 = pure rotation
        blend = min(1.0, abs(self.turn_input) * 2)

        # Direction of rotation
        turn_direction = np.sign(self.turn_input) if self.turn_input != 0 else 0

        # Effective strength based on speed
        effective_strength = self.base_k_rot * self.speed_input

        # Compute blended matrix
        K_trans = self.K_translation * effective_strength

        # Generate rotation matrix (direction applied by multiplying by turn_direction)
        K_rotation = generate_rotation_matrix(self.n_species, 1.0)
        K_rot = K_rotation * effective_strength * turn_direction

        # Blend: when not turning, use translation; when turning, add rotation
        K_blended = K_trans * (1 - blend) + K_rot * blend

        # Apply to alignment matrix (only K_rot changes based on input)
        for i in range(self.n_species):
            for j in range(self.n_species):
                self.alignment_matrix[i, j] = np.clip(K_blended[i, j], -1.0, 1.0)

        # K_pos only changes when converge mode is active (C key held)
        cross_attraction = (self.converge_cross_attraction if self.converge_active
                           else self.normal_cross_attraction)
        for i in range(self.n_species):
            for j in range(self.n_species):
                if abs(i - j) == 1:  # Only neighbors
                    self.matrix[i, j] = cross_attraction
                elif i != j:
                    self.matrix[i, j] = 0.0  # Non-neighbors don't attract

    def step(self):
        """Perform one simulation step with control updates."""
        if self.paused:
            return

        # Apply input decay (return to neutral over time)
        self.turn_input *= self.turn_decay
        if abs(self.turn_input) < 0.01:
            self.turn_input = 0.0

        # Update matrices based on current input
        self.update_matrices_from_input()

        # Call parent step for physics
        super().step()

    def draw(self):
        """Draw the simulation with control overlay."""
        self.screen.fill((255, 255, 255))

        # Draw particles
        for i in range(self.n):
            color = self.colors[self.species[i]]
            pos = self.positions[i]
            x = int(pos[0] * self.zoom)
            y = int(pos[1] * self.zoom)

            if self.show_orientations:
                angle = self.orientations[i]
                radius = 5 * self.zoom
                pygame.draw.circle(self.screen, color, (x, y), max(1, int(radius)))
                line_length = radius * 0.8
                end_x = x + line_length * np.cos(angle)
                end_y = y + line_length * np.sin(angle)
                pygame.draw.line(self.screen, (0, 0, 0), (x, y), (end_x, end_y),
                               max(1, int(self.zoom)))
            else:
                pygame.draw.circle(self.screen, color, (x, y), max(1, int(4 * self.zoom)))

        # Skip GUI elements if hidden
        if self.hide_gui:
            return

        # Draw swarm centroid
        centroid = self.get_swarm_centroid().astype(int)
        pygame.draw.circle(self.screen, (0, 0, 0), tuple(centroid), 8, 2)

        # Draw control indicators
        self.draw_control_indicator()

        # Draw info panel
        if self.show_info:
            self.draw_info_panel()

    def draw_control_indicator(self):
        """Draw visual indicator for current turn/speed input."""
        # Position in bottom center
        cx = self.config.width // 2
        cy = self.config.height - 60

        # Background circle
        pygame.draw.circle(self.screen, (230, 230, 230), (cx, cy), 50)
        pygame.draw.circle(self.screen, (200, 200, 200), (cx, cy), 50, 2)

        # Turn indicator (horizontal bar)
        turn_width = int(self.turn_input * 40)
        if turn_width != 0:
            color = (100, 150, 255) if turn_width < 0 else (255, 150, 100)
            bar_x = cx if turn_width > 0 else cx + turn_width
            pygame.draw.rect(self.screen, color, (bar_x, cy - 5, abs(turn_width), 10))

        # Speed indicator (vertical bar)
        speed_height = int(self.speed_input * 40)
        pygame.draw.rect(self.screen, (100, 200, 100),
                        (cx - 5, cy - speed_height, 10, speed_height))

        # Center dot
        pygame.draw.circle(self.screen, (50, 50, 50), (cx, cy), 5)

        # Labels
        font = self.font
        turn_text = font.render(f"Turn: {self.turn_input:.2f}", True, (100, 100, 100))
        speed_text = font.render(f"Speed: {self.speed_input:.2f}", True, (100, 100, 100))
        self.screen.blit(turn_text, (cx - 40, cy + 55))
        self.screen.blit(speed_text, (cx - 45, cy + 75))

    def draw_info_panel(self):
        """Draw information panel."""
        info_lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Species: {self.n_species}",
            f"Particles: {self.n}",
            "",
            f"Turn Input: {self.turn_input:+.2f}",
            f"Speed Input: {self.speed_input:.2f}",
            f"Converge: {'ON' if self.converge_active else 'OFF'}",
            "",
            "Controls:",
            "←/→: Turn left/right",
            "↑/↓: Speed up/down",
            "+/-: Add/remove species",
            "C: Hold to converge",
            "R: Reset  SPACE: Pause",
            "I: Toggle info  Q: Quit",
        ]

        y = 10
        for line in info_lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 22

        # Draw K_rot matrix visualization
        self.draw_matrix_viz()

        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 100, 100))
            rect = pause_text.get_rect(center=(self.config.width // 2, 30))
            self.screen.blit(pause_text, rect)

    def draw_single_matrix(self, matrix, label_text, x_start, y_start, is_editing=False):
        """Draw a single matrix visualization with values and species colors."""
        cell_size = 35
        color_indicator_size = 12

        # Label color
        if is_editing:
            label_color = (100, 100, 200)
            label_text = label_text + " (EDIT)"
        else:
            label_color = (100, 100, 100)

        label = self.font.render(label_text, True, label_color)
        self.screen.blit(label, (x_start, y_start))
        y_start += 25

        # Draw column color indicators (top)
        for j in range(self.n_species):
            cx = x_start + j * cell_size + cell_size // 2 - 1
            cy = y_start - 10
            pygame.draw.circle(self.screen, self.colors[j], (cx, cy), color_indicator_size // 2)
            pygame.draw.circle(self.screen, (100, 100, 100), (cx, cy), color_indicator_size // 2, 1)

        for i in range(self.n_species):
            # Draw row color indicator (left side)
            rx = x_start - 15
            ry = y_start + i * cell_size + cell_size // 2 - 1
            pygame.draw.circle(self.screen, self.colors[i], (rx, ry), color_indicator_size // 2)
            pygame.draw.circle(self.screen, (100, 100, 100), (rx, ry), color_indicator_size // 2, 1)

            for j in range(self.n_species):
                x = x_start + j * cell_size
                y = y_start + i * cell_size
                value = matrix[i, j]

                # Color based on value
                if value > 0.01:
                    intensity = int(min(255, abs(value) * 255))
                    color = (0, intensity, 0)
                elif value < -0.01:
                    intensity = int(min(255, abs(value) * 255))
                    color = (intensity, 0, 0)
                else:
                    color = (180, 180, 180)

                pygame.draw.rect(self.screen, color, (x, y, cell_size - 2, cell_size - 2))

                # Highlight selected cell in edit mode
                if is_editing and i == self.edit_row and j == self.edit_col:
                    pygame.draw.rect(self.screen, (255, 255, 0),
                                   (x, y, cell_size - 2, cell_size - 2), 3)
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200),
                                   (x, y, cell_size - 2, cell_size - 2), 1)

                # Always show value text
                val_text = self.font.render(f"{value:.1f}", True, (255, 255, 255))
                text_rect = val_text.get_rect(center=(x + cell_size//2 - 1, y + cell_size//2 - 1))
                self.screen.blit(val_text, text_rect)

        return y_start + self.n_species * cell_size

    def draw_matrix_viz(self):
        """Draw visualization of both K_rot and K_pos matrices."""
        cell_size = 35
        x_start = self.config.width - 30 - self.n_species * cell_size
        y_start = 10

        # Draw K_rot matrix
        is_editing_krot = self.matrix_edit_mode and self.editing_k_rot
        y_after_krot = self.draw_single_matrix(
            self.alignment_matrix, "K_rot:", x_start, y_start, is_editing_krot
        )

        # Draw K_pos matrix below K_rot
        y_pos_start = y_after_krot + 20
        is_editing_kpos = self.matrix_edit_mode and not self.editing_k_rot
        y_after_kpos = self.draw_single_matrix(
            self.matrix, "K_pos:", x_start, y_pos_start, is_editing_kpos
        )

        # Draw edit mode instructions
        if self.matrix_edit_mode:
            instr_y = y_after_kpos + 10
            instr = self.font.render("WASD:move E/X:+/- TAB:switch M:exit", True, (100, 100, 100))
            self.screen.blit(instr, (x_start - 50, instr_y))

    def change_species_count(self, new_count: int):
        """Change the number of species and reinitialize."""
        new_count = max(2, min(10, new_count))
        if new_count == self.n_species:
            return

        print(f"Changing species count: {self.n_species} → {new_count}")

        # Update config
        self.config.n_species = new_count
        self.n_species = new_count

        # Regenerate matrices
        self.matrix = generate_position_matrix(new_count)
        self.alignment_matrix = generate_translation_matrix(new_count, self.base_k_rot)

        # Update base matrices
        self._update_base_matrices()

        # Regenerate colors
        self.colors = []
        for i in range(new_count):
            hue = i / new_count
            color = pygame.Color(0)
            color.hsva = (hue * 360, 70, 90, 100)
            self.colors.append((color.r, color.g, color.b))

        # Reinitialize particle positions
        self._initialize_side_by_side()

        # Reset control state
        self.turn_input = 0.0
        self.edit_row = 0
        self.edit_col = 0

    def _adjust_matrix_value(self, delta: float):
        """Adjust the selected matrix cell value."""
        if self.editing_k_rot:
            current = self.alignment_matrix[self.edit_row, self.edit_col]
            new_val = np.clip(current + delta, -1.0, 1.0)
            self.alignment_matrix[self.edit_row, self.edit_col] = new_val
            print(f"K_rot[{self.edit_row},{self.edit_col}] = {new_val:.2f}")
        else:
            current = self.matrix[self.edit_row, self.edit_col]
            new_val = np.clip(current + delta, -1.0, 1.0)
            self.matrix[self.edit_row, self.edit_col] = new_val
            print(f"K_pos[{self.edit_row},{self.edit_col}] = {new_val:.2f}")

    def handle_events(self) -> bool:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self._initialize_side_by_side()
                    self.turn_input = 0.0
                    print("Reset positions")

                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info

                elif event.key == pygame.K_o:
                    self.show_orientations = not self.show_orientations

                elif event.key == pygame.K_h:
                    self.hide_gui = not self.hide_gui

                elif event.key == pygame.K_c:
                    self.converge_active = True
                    print("Converge: ON")

                # Matrix editing controls
                elif event.key == pygame.K_m:
                    self.matrix_edit_mode = not self.matrix_edit_mode
                    print(f"Matrix edit mode: {'ON' if self.matrix_edit_mode else 'OFF'}")

                elif event.key == pygame.K_TAB and self.matrix_edit_mode:
                    self.editing_k_rot = not self.editing_k_rot
                    print(f"Editing: {'K_rot' if self.editing_k_rot else 'K_pos'}")

                elif self.matrix_edit_mode:
                    # WASD navigation
                    if event.key == pygame.K_w:
                        self.edit_row = max(0, self.edit_row - 1)
                    elif event.key == pygame.K_s:
                        self.edit_row = min(self.n_species - 1, self.edit_row + 1)
                    elif event.key == pygame.K_a:
                        self.edit_col = max(0, self.edit_col - 1)
                    elif event.key == pygame.K_d:
                        self.edit_col = min(self.n_species - 1, self.edit_col + 1)
                    # E/X or +/- to adjust value
                    elif event.key == pygame.K_e or event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self._adjust_matrix_value(0.1)
                    elif event.key == pygame.K_x or event.key == pygame.K_MINUS:
                        self._adjust_matrix_value(-0.1)

                # Species count adjustment (only when not in matrix edit mode)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.change_species_count(self.n_species + 1)

                elif event.key == pygame.K_MINUS:
                    self.change_species_count(self.n_species - 1)

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_c:
                    self.converge_active = False
                    print("Converge: OFF")

        # Handle held keys for continuous control
        keys = pygame.key.get_pressed()

        # Turn control
        if keys[pygame.K_LEFT]:
            self.turn_input = max(-1.0, self.turn_input - 0.05)
        if keys[pygame.K_RIGHT]:
            self.turn_input = min(1.0, self.turn_input + 0.05)

        # Speed control
        if keys[pygame.K_UP]:
            self.speed_input = min(1.0, self.speed_input + 0.02)
        if keys[pygame.K_DOWN]:
            self.speed_input = max(0.0, self.speed_input - 0.02)

        return True

    def run(self):
        """Main simulation loop."""
        running = True

        while running:
            running = self.handle_events()
            self.step()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    demo = MultiSpeciesDemo(n_species=4, n_particles=150)
    demo.run()


if __name__ == "__main__":
    main()
