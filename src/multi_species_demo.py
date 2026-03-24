#!/usr/bin/env python3
"""
Multi-Species Manual Control Demo

Demonstrates manual control of N species (2-10) arranged side-by-side.
A PD heading controller steers the swarm toward a target direction.

The K_rot matrix is dynamically updated:
- Translation mode: Adjacent species have opposite-sign K_rot entries
- Rotation mode: All species have same-sign K_rot entries
- PD controller blends between modes to track target heading

Controls:
    ←/→: Rotate target heading left/right
    ↑/↓: Speed up/slow down
    +/-: Add/remove species (2-10)
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
        K[i, i + 1] = +strength
        K[i + 1, i] = -strength

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
                             cross_attraction: float = 0.1) -> np.ndarray:
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

    def __init__(self, n_species: int = 3, n_particles: int = 20):
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
        self.speed_input = 0.4      # 0 (stop) to 1 (full speed)
        self.base_k_rot = 0.8      # Base rotation matrix strength

        # PD heading controller
        self.target_heading = -np.pi / 2  # Target direction (radians, 0=right, -pi/2=up)
        self.current_heading = -np.pi / 2  # Measured from centroid velocity
        self.prev_heading_error = 0.0
        self.heading_kp = 0.5       # Proportional gain (gentle correction)
        self.heading_kd = 0.4       # Derivative gain (damp oscillation)
        self.heading_turn_rate = 0.03  # Radians per frame when key held
        self.heading_deadzone = 0.05  # Ignore errors smaller than this (radians, ~3°)

        # Formation parameters
        self.group_spacing = 1.0    # Meters between species group centers

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
        print("  ←/→     Rotate target heading")
        print("  ↑/↓     Speed up/slow down")
        print("  +/-     Add/remove species")
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
        center_x = self.config.sim_width / 2
        center_y = self.config.sim_height / 2

        # Calculate total width of formation
        total_width = (self.n_species - 1) * self.group_spacing
        start_x = center_x - total_width / 2

        particles_per_species = self.n // self.n_species

        for i in range(self.n):
            species_id = min(i // particles_per_species, self.n_species - 1)
            self.species[i] = species_id

            # Group center for this species (meters)
            group_center_x = start_x + species_id * self.group_spacing
            group_center_y = center_y

            # Random offset within group (meters)
            self.positions[i, 0] = group_center_x + self.rng.uniform(-0.2, 0.2)
            self.positions[i, 1] = group_center_y + self.rng.uniform(-0.2, 0.2)

            # Initial velocity (slight forward motion)
            self.velocities[i] = np.array([0.0, -0.1])

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

    def _measure_heading(self):
        """Measure heading from chain axis orientation.

        The swarm moves perpendicular to the chain (species 0 → species N-1).
        This is stable because it depends on cluster positions, not velocities.
        """
        head_mask = self.species == 0
        tail_mask = self.species == self.n_species - 1
        if not head_mask.any() or not tail_mask.any():
            return  # No particles in head/tail species
        head_pos = self.positions[head_mask].mean(axis=0)
        tail_pos = self.positions[tail_mask].mean(axis=0)
        chain_vec = tail_pos - head_pos
        if np.linalg.norm(chain_vec) < 0.01:
            return  # Degenerate chain, keep previous heading
        chain_angle = np.arctan2(chain_vec[1], chain_vec[0])
        # Motion is perpendicular to chain axis
        self.current_heading = self._wrap_angle(chain_angle - np.pi / 2)

    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _pd_heading_control(self):
        """PD controller: heading error -> turn_input."""
        error = self._wrap_angle(self.target_heading - self.current_heading)

        # Dead zone: ignore tiny errors to prevent jitter
        if abs(error) < self.heading_deadzone:
            error = 0.0

        d_error = self._wrap_angle(error - self.prev_heading_error)
        self.prev_heading_error = error

        output = self.heading_kp * error + self.heading_kd * d_error
        self.turn_input = np.clip(output, -1.0, 1.0)

    def step(self):
        """Perform one simulation step with PD heading control."""
        if self.paused:
            return

        # Measure current heading from chain axis orientation
        self._measure_heading()

        # PD controller drives turn_input
        self._pd_heading_control()

        # Update matrices based on current input
        self.update_matrices_from_input()

        # Call parent step for physics
        super().step()

    def draw(self):
        """Draw the simulation with control overlay."""
        self.screen.fill((255, 255, 255))

        # Draw particles
        self.draw_particles()

        # Skip GUI elements if hidden
        if self.hide_gui:
            return

        # Draw swarm centroid
        if self.show_centroids:
            self.draw_swarm_centroid()

        # Draw info panel and control indicators
        if self.show_info:
            self.draw_control_indicator()
            self.draw_info_panel()

    def draw_control_indicator(self):
        """Draw heading compass and speed indicator."""
        cx = self.config.width // 2
        cy = self.config.height - 70
        radius = 50

        # Background circle (compass)
        pygame.draw.circle(self.screen, (240, 240, 240), (cx, cy), radius)
        pygame.draw.circle(self.screen, (200, 200, 200), (cx, cy), radius, 2)

        # Target heading arrow (blue, thick)
        tx = cx + int(radius * 0.85 * np.cos(self.target_heading))
        ty = cy + int(radius * 0.85 * np.sin(self.target_heading))
        pygame.draw.line(self.screen, (60, 100, 220), (cx, cy), (tx, ty), 3)
        pygame.draw.circle(self.screen, (60, 100, 220), (tx, ty), 5)

        # Current heading arrow (red, thin)
        ax = cx + int(radius * 0.7 * np.cos(self.current_heading))
        ay = cy + int(radius * 0.7 * np.sin(self.current_heading))
        pygame.draw.line(self.screen, (220, 80, 80), (cx, cy), (ax, ay), 2)
        pygame.draw.circle(self.screen, (220, 80, 80), (ax, ay), 3)

        # Center dot
        pygame.draw.circle(self.screen, (50, 50, 50), (cx, cy), 4)

        # Speed bar (right side of compass)
        bar_x = cx + radius + 15
        bar_h = int(self.speed_input * radius * 2)
        bar_top = cy + radius - bar_h
        pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, cy - radius, 10, radius * 2), 1)
        pygame.draw.rect(self.screen, (100, 200, 100), (bar_x, bar_top, 10, bar_h))

        # Labels
        heading_deg = np.degrees(self.target_heading)
        lbl1 = self.font.render(f"Target: {heading_deg:+.0f}\u00b0", True, (60, 100, 220))
        lbl2 = self.font.render(f"Actual: {np.degrees(self.current_heading):+.0f}\u00b0", True, (220, 80, 80))
        lbl3 = self.font.render(f"Speed: {self.speed_input:.2f}", True, (100, 100, 100))
        self.screen.blit(lbl1, (cx - 50, cy + radius + 8))
        self.screen.blit(lbl2, (cx - 50, cy + radius + 28))
        self.screen.blit(lbl3, (cx - 50, cy + radius + 48))

    def draw_info_panel(self):
        """Draw information panel."""
        error = self._wrap_angle(self.target_heading - self.current_heading)
        info_lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Species: {self.n_species}",
            f"Particles: {self.n}",
            "",
            f"Target:  {np.degrees(self.target_heading):+.0f}\u00b0",
            f"Actual:  {np.degrees(self.current_heading):+.0f}\u00b0",
            f"Error:   {np.degrees(error):+.1f}\u00b0",
            f"PD out:  {self.turn_input:+.2f}",
            f"Kp={self.heading_kp:.1f}  Kd={self.heading_kd:.1f}",
            "",
            "Controls:",
            "\u2190/\u2192: Rotate target heading",
            "\u2191/\u2193: Speed up/down",
            "+/-: Add/remove species",
            "R: Reset  SPACE: Pause",
            "V: Centroids  I: Info  Q: Quit",
        ]

        y = 10
        for line in info_lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 22

        # Draw K_rot matrix visualization
        self.draw_matrix_viz()

        self.draw_pause_indicator()

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
        self.n = self.config.n_particles * self.n_species

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

        # Reinitialize particles (reallocates arrays for new total)
        self.initialize_particles()
        self._initialize_side_by_side()

        # Reset control state
        self.turn_input = 0.0
        self.target_heading = -np.pi / 2
        self.current_heading = -np.pi / 2
        self.prev_heading_error = 0.0
        self.edit_row = 0
        self.edit_col = 0

        print(f"Species: {self.n_species} x {self.config.n_particles} = {self.n} particles")

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
                    self.target_heading = -np.pi / 2
                    self.current_heading = -np.pi / 2
                    self.prev_heading_error = 0.0
                    print("Reset positions")

                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info

                elif event.key == pygame.K_v:
                    self.show_centroids = not self.show_centroids

                elif event.key == pygame.K_o:
                    self.show_orientations = not self.show_orientations

                elif event.key == pygame.K_h:
                    self.hide_gui = not self.hide_gui

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

        # Handle held keys for continuous control
        keys = pygame.key.get_pressed()

        # Heading control — Left/Right rotate target heading
        if keys[pygame.K_LEFT]:
            self.target_heading -= self.heading_turn_rate
        if keys[pygame.K_RIGHT]:
            self.target_heading += self.heading_turn_rate
        self.target_heading = self._wrap_angle(self.target_heading)

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
    demo = MultiSpeciesDemo(n_species=4, n_particles=20)
    demo.run()


if __name__ == "__main__":
    main()
