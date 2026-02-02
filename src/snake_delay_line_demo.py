#!/usr/bin/env python3
"""
Snake Delay Line Demo

Implements snake-like following behavior using a delay line approach.
The head receives turn commands, and each subsequent joint follows
with a time delay, creating a slithering motion.

Only uses:
- Fixed K_pos matrix for chain structure
- Time-varying K_rot matrix for turning (via delay line)

No additional forces or physics modifications.

Controls:
    ←/→: Turn left/right (keyboard mode)
    ↑/↓: Adjust forward speed
    A: Toggle auto/keyboard mode
    R: Reset positions
    M: Toggle matrix visualization
    I: Toggle info panel
    SPACE: Pause/Resume
    Q/ESC: Quit
"""

import sys
import os
from collections import deque
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from particle_life import Config, ParticleLife


# =============================================================================
# Configuration
# =============================================================================

# Chain structure
S = 5                   # Number of species (segments), species 0 is head
P = 30                  # Particles per species
L = 1.0                 # Spacing between segment centroids (meters)

# Delay line parameters
DELAY_STEPS = 15        # Steps delay between each joint
U_GAIN = 0.6            # Turn signal gain
TAIL_SCALE = 0.7        # Taper factor for tail (1.0 = no taper)
U_CLIP = 1.0            # Max absolute value for joint control

# Forward motion
BASE_FORWARD = 0.0      # Base forward speed (K_rot antisymmetric strength) - 0 to stay in place

# K_pos parameters (fixed)
K_SELF = 0.6            # Self-cohesion (diagonal)
K_ADJ = 0.4             # Adjacent attraction

# Auto mode parameters
AUTO_AMPLITUDE = 0.8    # Amplitude of auto turn signal
AUTO_FREQUENCY = 0.003  # Frequency of sine wave (cycles per step)

# Simulation
DT = 0.1
SEED = 42


# =============================================================================
# Helper functions
# =============================================================================

def build_k_pos(n_species: int, k_self: float, k_adj: float) -> np.ndarray:
    """Build fixed K_pos matrix for chain structure (adjacent-only)."""
    K = np.zeros((n_species, n_species))
    for i in range(n_species):
        K[i, i] = k_self
        if i > 0:
            K[i, i-1] = k_adj
        if i < n_species - 1:
            K[i, i+1] = k_adj
    return K


def build_k_rot_translation(n_species: int, strength: float) -> np.ndarray:
    """
    Build K_rot matrix for forward translation.
    Antisymmetric: K[i, i+1] = +strength, K[i+1, i] = -strength
    """
    K = np.zeros((n_species, n_species))
    for i in range(n_species - 1):
        K[i, i + 1] = +strength
        K[i + 1, i] = -strength
    return K


def build_k_rot_from_joints(n_species: int, joint_values: np.ndarray) -> np.ndarray:
    """
    Build K_rot matrix from per-joint values.

    For forward + turning:
    - Base: antisymmetric for translation
    - Turning: symmetric component added based on joint_values

    joint_values[i] controls the turning at joint i (between species i and i+1).
    Positive = turn one direction, negative = turn other direction.
    """
    K = np.zeros((n_species, n_species))
    for i in range(n_species - 1):
        # Symmetric turning component
        K[i, i + 1] = joint_values[i]
        K[i + 1, i] = joint_values[i]
    return K


# =============================================================================
# Snake Demo Class
# =============================================================================

class SnakeDelayLineDemo(ParticleLife):
    """Snake demo with delay line control."""

    def __init__(self):
        self.S = S
        self.P = P
        self.L = L

        # Create config (n_particles is per species)
        config = Config(
            n_species=self.S,
            n_particles=self.P,
            dt=DT,
            seed=SEED,
        )

        # Initialize parent
        super().__init__(config, headless=False)

        # Override positions with chain initialization
        self._initialize_chain()

        # Set fixed K_pos
        K_pos = build_k_pos(self.S, K_SELF, K_ADJ)
        self.set_position_matrix(K_pos)

        # Initialize K_rot to forward motion
        K_rot = build_k_rot_translation(self.S, BASE_FORWARD)
        self.set_orientation_matrix(K_rot)

        # Delay line state
        self.n_joints = self.S - 1
        self.delay_steps = DELAY_STEPS
        self.hist_len = self.n_joints * self.delay_steps + 1
        self.head_history = deque([0.0] * self.hist_len, maxlen=self.hist_len)

        # Control parameters
        self.u_gain = U_GAIN
        self.tail_scale = TAIL_SCALE
        self.u_clip = U_CLIP
        self.base_forward = BASE_FORWARD

        # Taper for tail (optional: makes tail movements smaller)
        self.taper = np.linspace(1.0, self.tail_scale, self.n_joints)

        # Input state
        self.turn_input = 0.0       # Current turn command (-1 to +1)
        self.speed_input = 1.0      # Speed multiplier (0 to 1)
        self.auto_mode = True       # Auto vs keyboard mode

        # Step counter
        self.current_step = 0

        # Show matrix by default
        self.show_matrix = True

        # Override window title
        pygame.display.set_caption("Snake Delay Line Demo")

        print("=" * 60)
        print("Snake Delay Line Demo")
        print("=" * 60)
        print(f"Species: {self.S}, Particles: {self.P}/species = {self.n} total")
        print(f"Delay steps: {self.delay_steps}, History length: {self.hist_len}")
        print("")
        print("Controls:")
        print("  ←/→     Turn left/right (keyboard mode)")
        print("  ↑/↓     Speed up/slow down")
        print("  A       Toggle auto/keyboard mode")
        print("  R       Reset")
        print("  M       Toggle matrix  I: Toggle info")
        print("  SPACE   Pause  Q: Quit")
        print("=" * 60)

    def _initialize_chain(self):
        """Initialize particles as a horizontal chain."""
        center_x = self.config.sim_width / 2
        center_y = self.config.sim_height / 2
        total_width = (self.S - 1) * self.L
        start_x = center_x - total_width / 2

        positions = []
        species = []
        sigma = 0.2  # meters

        for s in range(self.S):
            cx = start_x + s * self.L
            cy = center_y
            for _ in range(self.P):
                px = cx + self.rng.randn() * sigma
                py = cy + self.rng.randn() * sigma
                positions.append([px, py])
                species.append(s)

        self.positions = np.array(positions)
        self.species = np.array(species, dtype=int)
        self.velocities = np.zeros_like(self.positions)
        self.orientations = self.rng.uniform(0, 2 * np.pi, len(positions))
        self.angular_velocities = np.zeros(len(positions))
        self.n = len(positions)

    def get_auto_turn_input(self) -> float:
        """Generate automatic turn input (sine wave)."""
        return AUTO_AMPLITUDE * np.sin(2 * np.pi * AUTO_FREQUENCY * self.current_step)

    def update_delay_line(self):
        """
        Update the delay line and compute K_rot.

        The head receives turn_input, and each joint follows with delay.
        """
        # Get current head turn input
        if self.auto_mode:
            head_turn = self.get_auto_turn_input()
        else:
            head_turn = self.turn_input

        # Push to history (newest at the end)
        self.head_history.append(head_turn)

        # Extract delayed values for each joint
        # Joint i uses head_history[-(1 + i * delay_steps)]
        joint_values = np.zeros(self.n_joints)
        for i in range(self.n_joints):
            delay_idx = i * self.delay_steps
            if delay_idx < len(self.head_history):
                # Read from history (older values have larger delay_idx)
                joint_values[i] = self.head_history[-(1 + delay_idx)]
            else:
                joint_values[i] = 0.0

        # Apply gain and taper
        joint_values = joint_values * self.u_gain * self.taper

        # Clip to prevent instability
        joint_values = np.clip(joint_values, -self.u_clip, self.u_clip)

        # Build K_rot: translation base + turning
        # Forward: antisymmetric component
        K_forward = build_k_rot_translation(self.S, self.base_forward * self.speed_input)

        # Turning: symmetric component based on joint values
        K_turn = build_k_rot_from_joints(self.S, joint_values)

        # Combine: forward motion + turning bias
        K_rot = K_forward + K_turn

        # Clip final values
        K_rot = np.clip(K_rot, -1.5, 1.5)

        # Apply
        self.set_orientation_matrix(K_rot)

        return joint_values

    def step(self):
        """Perform one simulation step."""
        if self.paused:
            return

        # Update delay line and K_rot
        self.current_joint_values = self.update_delay_line()

        # Call parent step
        super().step()

        self.current_step += 1

    def draw(self):
        """Draw simulation."""
        self.screen.fill((255, 255, 255))

        # Draw particles (meters → pixels via ppu)
        for i in range(self.n):
            color = self.colors[self.species[i]]
            pos = self.positions[i]
            x = int(pos[0] * self.ppu * self.zoom)
            y = int(pos[1] * self.ppu * self.zoom)
            pygame.draw.circle(self.screen, color, (x, y), max(2, int(0.06 * self.ppu * self.zoom)))

        # Draw centroids and connecting line (meters → pixels)
        centroids = self.get_species_centroids()
        centroid_points = [(int(c[0] * self.ppu * self.zoom), int(c[1] * self.ppu * self.zoom)) for c in centroids]

        if len(centroid_points) >= 2:
            pygame.draw.lines(self.screen, (0, 0, 0), False, centroid_points, 3)

        # Draw centroid markers (head is larger)
        for i, (cx, cy) in enumerate(centroid_points):
            radius = 10 if i == 0 else 6  # Head is larger
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), radius + 2)
            pygame.draw.circle(self.screen, self.colors[i], (cx, cy), radius)

        # Draw info panel
        if self.show_info:
            self._draw_info_panel()

        # Draw matrix visualization
        if self.show_matrix:
            self._draw_matrix_panel()

        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 100, 100))
            rect = pause_text.get_rect(center=(self.config.width // 2, 30))
            self.screen.blit(pause_text, rect)

    def _draw_info_panel(self):
        """Draw info panel."""
        mode_str = "AUTO" if self.auto_mode else "KEYBOARD"

        # Get current head turn
        if self.auto_mode:
            head_turn = self.get_auto_turn_input()
        else:
            head_turn = self.turn_input

        # Joint values display
        if hasattr(self, 'current_joint_values'):
            jv = self.current_joint_values
            jv_str = ' '.join([f"{v:+.2f}" for v in jv[:min(6, len(jv))]])
            if len(jv) > 6:
                jv_str += "..."
        else:
            jv_str = "N/A"

        info_lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Step: {self.current_step}",
            f"Mode: {mode_str}",
            "",
            f"Head turn: {head_turn:+.2f}",
            f"Speed: {self.speed_input:.2f}",
            "",
            f"Joint values:",
            f"  {jv_str}",
            "",
            "Controls:",
            "A: Toggle auto/keyboard",
            "←/→: Turn  ↑/↓: Speed",
            "R: Reset  M: Matrix",
        ]

        y = 10
        for line in info_lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 22

    def _draw_matrix_panel(self):
        """Draw K_rot matrix visualization."""
        cell_size = 30 if self.S <= 8 else 25

        x_start = self.config.width - cell_size * self.S - 50
        y_start = 10

        # Title
        title = self.font.render("K_rot (delay line)", True, (100, 100, 200))
        self.screen.blit(title, (x_start, y_start))
        y_start += 25

        # Column indicators
        for j in range(self.S):
            cx = x_start + j * cell_size + cell_size // 2
            pygame.draw.circle(self.screen, self.colors[j], (cx, y_start), 5)
        y_start += 15

        for i in range(self.S):
            # Row indicator
            pygame.draw.circle(self.screen, self.colors[i],
                             (x_start - 10, y_start + i * cell_size + cell_size // 2), 5)

            for j in range(self.S):
                x = x_start + j * cell_size
                y = y_start + i * cell_size
                value = self.alignment_matrix[i, j]

                # Background color
                if abs(value) < 0.01:
                    bg_color = (220, 220, 220)
                elif value > 0:
                    intensity = min(255, int(abs(value) * 200))
                    bg_color = (255 - intensity//2, 255, 255 - intensity//2)
                else:
                    intensity = min(255, int(abs(value) * 200))
                    bg_color = (255, 255 - intensity//2, 255 - intensity//2)

                pygame.draw.rect(self.screen, bg_color, (x, y, cell_size - 1, cell_size - 1))

                # Highlight adjacent pairs
                if abs(i - j) == 1:
                    pygame.draw.rect(self.screen, (100, 100, 255),
                                   (x, y, cell_size - 1, cell_size - 1), 2)

                # Value text
                val_str = f"{value:.1f}"
                text_color = (0, 100, 0) if value > 0 else (150, 0, 0) if value < 0 else (150, 150, 150)
                val_surf = self.font.render(val_str, True, text_color)
                text_rect = val_surf.get_rect(center=(x + cell_size//2, y + cell_size//2))
                self.screen.blit(val_surf, text_rect)

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
                    self.head_history = deque([0.0] * self.hist_len, maxlen=self.hist_len)
                    self.current_step = 0
                    self.turn_input = 0.0
                    print("Reset")
                elif event.key == pygame.K_a:
                    self.auto_mode = not self.auto_mode
                    self.turn_input = 0.0
                    print(f"Mode: {'AUTO' if self.auto_mode else 'KEYBOARD'}")
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_m:
                    self.show_matrix = not self.show_matrix

        # Continuous keyboard input (only in keyboard mode)
        if not self.auto_mode:
            keys = pygame.key.get_pressed()

            # Turn input
            if keys[pygame.K_LEFT]:
                self.turn_input = max(-1.0, self.turn_input - 0.05)
            elif keys[pygame.K_RIGHT]:
                self.turn_input = min(1.0, self.turn_input + 0.05)
            else:
                # Decay turn input back to 0
                self.turn_input *= 0.95
                if abs(self.turn_input) < 0.01:
                    self.turn_input = 0.0

        # Speed input (both modes)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.speed_input = min(1.5, self.speed_input + 0.02)
        if keys[pygame.K_DOWN]:
            self.speed_input = max(0.0, self.speed_input - 0.02)

        return True

    def run(self):
        """Main loop."""
        running = True
        while running:
            running = self.handle_events()
            self.step()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()


def main():
    demo = SnakeDelayLineDemo()
    demo.run()


if __name__ == "__main__":
    main()
