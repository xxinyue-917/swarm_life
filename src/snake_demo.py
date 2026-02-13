#!/usr/bin/env python3
"""
Snake Demo — Direct Head Steering

A chain of particle clusters (species) connected by position-matrix attraction.
Arrow keys steer the head left/right by manipulating the K_rot matrix.
Up/Down arrows control forward movement speed.

Controls:
    ←/→:   Steer head left/right
    ↑/↓:   Increase/decrease forward speed
    R:     Reset positions
    SPACE: Pause/Resume
    O:     Toggle orientation display
    H:     Hide/show all GUI
    I:     Toggle info panel

Matrix Editing:
    M:     Toggle matrix edit mode
    TAB:   Switch K_rot/K_pos
    WASD:  Navigate cells
    E/X:   Increase/decrease value
    Q/ESC: Quit
"""

import pygame
import numpy as np
from particle_life import Config, ParticleLife


# =============================================================================
# Matrix generators
# =============================================================================

def generate_position_matrix(n_species: int,
                             self_cohesion: float = 0.8,
                             adjacent_attraction: float = 0.3,
                             forward_bias: float = 0.1) -> np.ndarray:
    """
    Generate K_pos matrix for chain structure.

    Diagonal = self-cohesion, off-diagonal only for adjacent species.

    Args:
        forward_bias: Asymmetry that creates forward motion.
                      Positive = move toward head (species 0)
                      Zero = stationary
                      Negative = move toward tail
    """
    K = np.zeros((n_species, n_species))
    for i in range(n_species):
        K[i, i] = self_cohesion
        if i > 0:
            # Attraction to species behind (toward tail)
            K[i, i - 1] = adjacent_attraction
        if i < n_species - 1:
            # Attraction to species ahead (toward head) - reduced by forward_bias
            K[i, i + 1] = adjacent_attraction - forward_bias
    return K


def generate_rotation_matrix(n_species: int, strength: float) -> np.ndarray:
    """
    Generate K_rot matrix for collective rotation.

    Symmetric: K[i, i+1] = K[i+1, i] = strength
    Only adjacent species are coupled (chain joints).
    """
    K = np.zeros((n_species, n_species))
    for i in range(n_species - 1):
        K[i, i + 1] = strength
        K[i + 1, i] = strength
    return K


# =============================================================================
# Snake Demo
# =============================================================================

class SnakeDemo(ParticleLife):
    """
    Snake demo with direct head steering via K_rot manipulation.

    Species are arranged as a horizontal chain.
    Arrow keys control turn via K_rot matrix (symmetric rotation component).
    """

    def __init__(self, n_species: int = 6, n_particles: int = 20):
        config = Config(
            n_species=n_species,
            n_particles=n_particles,
            a_rot=3.0,
            position_matrix=generate_position_matrix(n_species).tolist(),
            orientation_matrix=np.zeros((n_species, n_species)).tolist(),
        )

        super().__init__(config, headless=False)

        # Control state
        self.turn_input = 0.0       # -1 (full left) to +1 (full right)
        self.base_k_rot = 1.0       # Base rotation matrix strength
        self.forward_speed = 0.1    # Forward bias (0 = stationary, positive = move forward)

        # Input smoothing
        self.turn_decay = 0.92      # How quickly turn input returns to 0

        # Delay propagation: each joint gets the head's turn signal
        # delayed by joint_delay steps per joint down the chain
        self.joint_delay = 8        # Steps of delay between adjacent joints
        n_joints = n_species - 1
        history_len = self.joint_delay * n_joints + 1
        self.turn_history = np.zeros(history_len)  # Ring buffer of past turn inputs
        self.history_idx = 0

        # Formation parameters
        self.group_spacing = 0.8    # Meters between species group centers

        # Matrix editing mode
        self.matrix_edit_mode = False
        self.edit_row = 0
        self.edit_col = 0
        self.editing_k_rot = True   # True = K_rot, False = K_pos

        # GUI visibility
        self.hide_gui = False

        # Initialize particles in chain formation
        self._initialize_chain()

        # Override window title
        pygame.display.set_caption("Snake Demo — Arrow Keys to Steer")

        print("=" * 60)
        print("Snake Demo — Direct Head Steering")
        print("=" * 60)
        print(f"Species: {self.n_species}  Particles: {self.n}")
        print("")
        print("Controls:")
        print("  ←/→     Steer left/right")
        print("  ↑/↓     Forward speed (+/-)")
        print("  R       Reset positions")
        print("  SPACE   Pause")
        print("  O       Toggle orientation display")
        print("  H       Hide/show all GUI")
        print("  I       Toggle info panel")
        print("")
        print("Matrix Editing:")
        print("  M       Toggle matrix edit mode")
        print("  TAB     Switch K_rot/K_pos")
        print("  WASD    Navigate cells")
        print("  E/X     Increase/decrease value")
        print("=" * 60)

    def _initialize_chain(self):
        """Arrange species in horizontal chain formation."""
        center_x = self.config.sim_width / 2
        center_y = self.config.sim_height / 2

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
            self.positions[i, 0] = group_center_x + self.rng.uniform(-0.15, 0.15)
            self.positions[i, 1] = group_center_y + self.rng.uniform(-0.15, 0.15)

            # Zero initial velocity (stay in place)
            self.velocities[i] = np.array([0.0, 0.0])

    def update_matrices_from_input(self):
        """
        Update K_rot matrix based on delayed turn input per joint.

        Each joint (between species i and i+1) receives the head's turn
        signal delayed by joint_delay * i steps, creating a wave that
        propagates from head to tail.
        """
        if self.matrix_edit_mode:
            return

        # Record current turn input into history
        self.turn_history[self.history_idx] = self.turn_input
        self.history_idx = (self.history_idx + 1) % len(self.turn_history)

        # Build K_rot with per-joint delayed signals
        K_rot = np.zeros((self.n_species, self.n_species))
        n_joints = self.n_species - 1

        for joint in range(n_joints):
            # Joint 0 (head-segment1) uses current input (no delay)
            # Joint 1 uses input from joint_delay steps ago, etc.
            delay = joint * self.joint_delay
            idx = (self.history_idx - 1 - delay) % len(self.turn_history)
            delayed_input = self.turn_history[idx]

            strength = -self.base_k_rot * delayed_input
            K_rot[joint, joint + 1] = strength
            K_rot[joint + 1, joint] = strength

        # Apply to alignment matrix
        for i in range(self.n_species):
            for j in range(self.n_species):
                self.alignment_matrix[i, j] = np.clip(K_rot[i, j], -1.0, 1.0)

    def update_forward_speed(self, delta: float):
        """
        Adjust forward speed and update position matrix.

        Args:
            delta: Change in forward speed (positive = faster forward)
        """
        self.forward_speed = np.clip(self.forward_speed + delta, -0.3, 0.5)

        # Regenerate position matrix with new forward bias
        new_matrix = generate_position_matrix(
            self.n_species,
            forward_bias=self.forward_speed
        )
        self.matrix[:] = new_matrix
        print(f"Forward speed: {self.forward_speed:.2f}")

    def step(self):
        """Perform one simulation step with control updates."""
        if self.paused:
            return

        # Apply input decay (return to neutral over time)
        self.turn_input *= self.turn_decay
        if abs(self.turn_input) < 0.01:
            self.turn_input = 0.0

        # Update K_rot from input
        self.update_matrices_from_input()

        # Call parent step for physics
        super().step()

    def draw(self):
        """Draw the simulation with control overlay."""
        self.screen.fill((255, 255, 255))

        # Draw particles
        self.draw_particles()

        if self.hide_gui:
            return

        # Draw centroid spine, markers, and swarm centroid
        pts = self.draw_centroid_spine()
        self.draw_centroid_markers(pts)
        self.draw_swarm_centroid()

        # Control indicator
        self.draw_control_indicator()

        # Info panel
        if self.show_info:
            self.draw_info_panel()

    def draw_control_indicator(self):
        """Draw visual indicator for current turn input."""
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

        # Center dot
        pygame.draw.circle(self.screen, (50, 50, 50), (cx, cy), 5)

        # Label
        turn_text = self.font.render(f"Turn: {self.turn_input:.2f}", True, (100, 100, 100))
        self.screen.blit(turn_text, (cx - 40, cy + 55))

    def draw_info_panel(self):
        """Draw information panel."""
        info_lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Species: {self.n_species}",
            f"Particles: {self.n}",
            "",
            f"Turn Input: {self.turn_input:+.2f}",
            f"Forward Speed: {self.forward_speed:+.2f}",
            "",
            "Controls:",
            "←/→: Steer left/right",
            "↑/↓: Forward speed",
            "R: Reset  SPACE: Pause",
            "O: Orientations  H: GUI",
            "I: Toggle info  Q: Quit",
        ]

        y = 10
        for line in info_lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 22

        # Draw both matrices
        self.draw_matrix_viz()

        self.draw_pause_indicator()

    def draw_single_matrix(self, matrix, label_text, x_start, y_start, is_editing=False):
        """Draw a single matrix visualization with values and species colors."""
        cell_size = 35
        color_indicator_size = 12

        # Label
        if is_editing:
            label_color = (100, 100, 200)
            label_text = label_text + " (EDIT)"
        else:
            label_color = (100, 100, 100)

        label = self.font.render(label_text, True, label_color)
        self.screen.blit(label, (x_start, y_start))
        y_start += 25

        # Column color indicators
        for j in range(self.n_species):
            cx = x_start + j * cell_size + cell_size // 2 - 1
            cy = y_start - 10
            pygame.draw.circle(self.screen, self.colors[j], (cx, cy), color_indicator_size // 2)
            pygame.draw.circle(self.screen, (100, 100, 100), (cx, cy), color_indicator_size // 2, 1)

        for i in range(self.n_species):
            # Row color indicator
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

                # Value text
                val_text = self.font.render(f"{value:.1f}", True, (255, 255, 255))
                text_rect = val_text.get_rect(center=(x + cell_size // 2 - 1,
                                                       y + cell_size // 2 - 1))
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

        # Draw K_pos matrix below
        y_pos_start = y_after_krot + 20
        is_editing_kpos = self.matrix_edit_mode and not self.editing_k_rot
        y_after_kpos = self.draw_single_matrix(
            self.matrix, "K_pos:", x_start, y_pos_start, is_editing_kpos
        )

        # Edit mode instructions
        if self.matrix_edit_mode:
            instr_y = y_after_kpos + 10
            instr = self.font.render("WASD:move E/X:+/- TAB:switch M:exit", True, (100, 100, 100))
            self.screen.blit(instr, (x_start - 50, instr_y))

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
                    self._initialize_chain()
                    self.turn_input = 0.0
                    self.turn_history[:] = 0.0
                    self.history_idx = 0
                    print("Reset positions")

                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info

                elif event.key == pygame.K_o:
                    self.show_orientations = not self.show_orientations
                    print(f"Show orientations: {self.show_orientations}")

                elif event.key == pygame.K_h:
                    self.hide_gui = not self.hide_gui

                # Forward speed controls
                elif event.key == pygame.K_UP:
                    self.update_forward_speed(0.05)
                elif event.key == pygame.K_DOWN:
                    self.update_forward_speed(-0.05)

                # Matrix editing controls
                elif event.key == pygame.K_m:
                    self.matrix_edit_mode = not self.matrix_edit_mode
                    print(f"Matrix edit mode: {'ON' if self.matrix_edit_mode else 'OFF'}")

                elif event.key == pygame.K_TAB and self.matrix_edit_mode:
                    self.editing_k_rot = not self.editing_k_rot
                    print(f"Editing: {'K_rot' if self.editing_k_rot else 'K_pos'}")

                elif self.matrix_edit_mode:
                    if event.key == pygame.K_w:
                        self.edit_row = max(0, self.edit_row - 1)
                    elif event.key == pygame.K_s:
                        self.edit_row = min(self.n_species - 1, self.edit_row + 1)
                    elif event.key == pygame.K_a:
                        self.edit_col = max(0, self.edit_col - 1)
                    elif event.key == pygame.K_d:
                        self.edit_col = min(self.n_species - 1, self.edit_col + 1)
                    elif event.key in (pygame.K_e, pygame.K_PLUS, pygame.K_EQUALS):
                        self._adjust_matrix_value(0.1)
                    elif event.key in (pygame.K_x, pygame.K_MINUS):
                        self._adjust_matrix_value(-0.1)

        # Continuous arrow key input for steering
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.turn_input = max(-1.0, self.turn_input - 0.05)
        if keys[pygame.K_RIGHT]:
            self.turn_input = min(1.0, self.turn_input + 0.05)

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
    demo = SnakeDemo(n_species=8, n_particles=20)
    demo.run()


if __name__ == "__main__":
    main()
