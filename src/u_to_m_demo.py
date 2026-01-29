#!/usr/bin/env python3
"""
U-shape to M-shape Morphing Demo (Real-time Pygame)

Demonstrates shape transformation from U-shape to M-shape using only
time-varying K_rot matrix (no additional forces).

Controls:
    SPACE: Pause/Resume
    R: Reset
    M: Toggle matrix visualization
    I: Toggle info panel
    Q/ESC: Quit
"""

import sys
import os
import numpy as np
import pygame

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from particle_life import Config, ParticleLife


# =============================================================================
# Configuration (edit these values directly)
# =============================================================================

# Species and particles
S = 9                # Number of species (chain segments)
P = 20               # Particles per species
L = 90.0            # Spacing between species centroids

# Simulation
DT = 0.1             # Timestep
SEED = 42            # Random seed

# Schedule (in simulation steps)
T_U = 500            # U-shape hold
T_MORPH = 300        # Morphing: interpolate U -> M
T_HOLD = 500         # M-shape hold

# Control strength
U0 = 0.8             # Joint control strength (K_rot magnitude)

# K_pos matrix parameters (fixed, maintains chain structure)
K_SELF = 0.6         # Self-cohesion (diagonal)
K_ADJ = 0.4          # Adjacent attraction (adjacent-only)

# =============================================================================


def smoothstep(t: float) -> float:
    """Smooth interpolation from 0 to 1 using cubic Hermite."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def generate_u_pattern(n_joints: int) -> np.ndarray:
    """
    Generate U-shape joint pattern.
    Left half bends one way (+), right half bends opposite (-).
    This makes both ends point upward.
    """
    pattern = np.ones(n_joints)
    mid = n_joints // 2
    pattern[mid:] = -1
    return pattern


def generate_m_pattern(n_joints: int) -> np.ndarray:
    """
    Generate M-shape joint pattern (two peaks pointing up).
    Pattern: -, -, +, +, -, -, +, + (inverted from W)
    """
    pattern = -np.ones(n_joints)
    if n_joints < 4:
        return pattern

    quarter = n_joints // 4
    # Flip: middle sections positive
    pattern[quarter:2*quarter] = 1
    pattern[3*quarter:] = 1

    return pattern


def build_k_pos(n_species: int, k_self: float, k_adj: float) -> np.ndarray:
    """Build fixed K_pos matrix for chain structure (adjacent-only)."""
    K = np.zeros((n_species, n_species))
    for i in range(n_species):
        K[i, i] = k_self  # Self-cohesion
        if i > 0:
            K[i, i-1] = k_adj  # Adjacent attraction
        if i < n_species - 1:
            K[i, i+1] = k_adj  # Adjacent attraction
    return K


def build_k_rot(n_species: int, joint_values: np.ndarray) -> np.ndarray:
    """Build K_rot matrix from joint values (symmetric for pure rotation, no translation)."""
    K = np.zeros((n_species, n_species))
    for s in range(n_species - 1):
        K[s, s+1] = joint_values[s]
        K[s+1, s] = joint_values[s]  # symmetric: rotation only
    return K


def compute_turning_angles(centroids: list) -> np.ndarray:
    """Compute turning angles between consecutive centroid segments."""
    if len(centroids) < 3:
        return np.array([])
    angles = []
    for i in range(1, len(centroids) - 1):
        v1 = centroids[i] - centroids[i-1]
        v2 = centroids[i+1] - centroids[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = np.dot(v1, v2)
        angle = np.arctan2(cross, dot)
        angles.append(angle)
    return np.array(angles)


class UtoMDemo(ParticleLife):
    """Real-time U-to-M morphing demo using pygame."""

    def __init__(self):
        # Use global config constants
        self.S = S
        self.P = P
        self.L = L

        # Time schedule
        self.T_U = T_U
        self.T_morph = T_MORPH
        self.T_hold = T_HOLD
        self.total_steps = self.T_U + self.T_morph + self.T_hold

        # Control parameters
        self.u0 = U0
        self.k_self = K_SELF
        self.k_adj = K_ADJ

        # Joint patterns
        self.n_joints = self.S - 1
        self.u_U = generate_u_pattern(self.n_joints) * self.u0
        self.u_M = generate_m_pattern(self.n_joints) * self.u0

        # Create config (use defaults from particle_life.py, only override what's needed)
        n_particles = self.S * self.P
        config = Config(
            n_species=self.S,
            n_particles=n_particles,
            dt=DT,
            seed=SEED,
        )

        # Initialize parent class
        super().__init__(config, headless=False)

        # Override particle positions with chain initialization
        self._initialize_chain()

        # Set fixed K_pos (adjacent-only)
        K_pos = build_k_pos(self.S, self.k_self, self.k_adj)
        self.set_position_matrix(K_pos)

        # Initialize K_rot to zero
        self.set_orientation_matrix(np.zeros((self.S, self.S)))

        # Step counter
        self.current_step = 0

        # Override window title
        pygame.display.set_caption("U-to-M Morphing Demo")

        # Show matrix by default
        self.show_matrix = True

        print("=" * 60)
        print("U-to-M Morphing Demo")
        print("=" * 60)
        print(f"Species: {self.S}, Particles: {n_particles}")
        print(f"Schedule: U={self.T_U}, Morph={self.T_morph}, Hold={self.T_hold}")
        print(f"U-pattern: {self.u_U}")
        print(f"M-pattern: {self.u_M}")
        print("")
        print("Controls: SPACE=Pause, R=Reset, M=Matrix, I=Info, Q=Quit")
        print("=" * 60)

    def _initialize_chain(self):
        """Initialize particles as a horizontal chain of clusters."""
        center_x = self.config.width / 2
        center_y = self.config.height / 2
        total_width = (self.S - 1) * self.L
        start_x = center_x - total_width / 2

        positions = []
        species = []
        sigma = 30.0  # Spread of particles around centroid

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

    def get_current_joint_values(self) -> np.ndarray:
        """Get joint control values based on current step."""
        step = self.current_step
        t1 = self.T_U
        t2 = t1 + self.T_morph

        if step < t1:
            return self.u_U.copy()
        elif step < t2:
            alpha = smoothstep((step - t1) / self.T_morph)
            return (1 - alpha) * self.u_U + alpha * self.u_M
        else:
            return self.u_M.copy()

    def get_phase_name(self) -> str:
        """Get current phase name."""
        step = self.current_step
        t1 = self.T_U
        t2 = t1 + self.T_morph

        if step < t1:
            return "U-SHAPE"
        elif step < t2:
            return "MORPHING"
        else:
            return "M-SHAPE"

    def step(self):
        """Perform one simulation step with K_rot update."""
        if self.paused:
            return

        # Update K_rot based on schedule
        joint_values = self.get_current_joint_values()
        K_rot = build_k_rot(self.S, joint_values)
        self.set_orientation_matrix(K_rot)

        # Call parent step
        super().step()

        # Increment step counter (loop back to M-shape hold)
        self.current_step += 1
        if self.current_step >= self.total_steps:
            self.current_step = self.T_U + self.T_morph  # Stay in M-shape

    def draw(self):
        """Draw simulation with centroid overlay."""
        self.screen.fill((255, 255, 255))

        # Draw particles
        for i in range(self.n):
            color = self.colors[self.species[i]]
            pos = self.positions[i]
            x, y = int(pos[0] * self.zoom), int(pos[1] * self.zoom)
            pygame.draw.circle(self.screen, color, (x, y), max(2, int(8 * self.zoom)))

        # Draw centroids and connecting line
        centroids = self.get_species_centroids()
        centroid_points = [(int(c[0] * self.zoom), int(c[1] * self.zoom)) for c in centroids]

        # Draw connecting line
        if len(centroid_points) >= 2:
            pygame.draw.lines(self.screen, (0, 0, 0), False, centroid_points, 3)

        # Draw centroid markers
        for i, (cx, cy) in enumerate(centroid_points):
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), 8)
            pygame.draw.circle(self.screen, self.colors[i], (cx, cy), 6)

        # Draw info panel
        if self.show_info:
            self._draw_info_panel(centroids)

        # Draw matrix visualization
        if self.show_matrix:
            self._draw_matrix_panel()

        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 100, 100))
            rect = pause_text.get_rect(center=(self.config.width // 2, 30))
            self.screen.blit(pause_text, rect)

    def _draw_info_panel(self, centroids):
        """Draw info panel with phase and metrics."""
        phase = self.get_phase_name()
        progress = self.current_step / self.total_steps * 100

        # Compute angle pattern
        angles = compute_turning_angles(np.array(centroids))
        if len(angles) > 0:
            sign_pattern = ''.join(['+' if a > 0.05 else ('-' if a < -0.05 else '0') for a in angles])
        else:
            sign_pattern = ''

        # Get current joint values
        joint_vals = self.get_current_joint_values()
        joint_str = ' '.join([f"{v:+.1f}" for v in joint_vals])

        info_lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Step: {self.current_step}/{self.total_steps} ({progress:.0f}%)",
            f"Phase: {phase}",
            "",
            f"Joint values:",
            f"  {joint_str}",
            "",
            f"Angle pattern: {sign_pattern}",
            "",
            "Controls: SPACE R M I Q",
        ]

        y = 10
        for line in info_lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 22

        # Draw phase indicator bar
        bar_x, bar_y = self.config.width - 220, 10
        bar_w, bar_h = 200, 20

        pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, bar_y, bar_w, bar_h))

        # Color sections for each phase
        phases = [
            (self.T_U, (100, 200, 100), "U"),
            (self.T_morph, (200, 200, 100), "→"),
            (self.T_hold, (100, 100, 200), "M"),
        ]

        x_offset = bar_x
        for duration, color, label in phases:
            w = int(bar_w * duration / self.total_steps)
            pygame.draw.rect(self.screen, color, (x_offset, bar_y, w, bar_h))
            label_text = self.font.render(label, True, (50, 50, 50))
            self.screen.blit(label_text, (x_offset + w//2 - 5, bar_y + 2))
            x_offset += w

        # Current position marker
        marker_x = bar_x + int(bar_w * self.current_step / self.total_steps)
        pygame.draw.line(self.screen, (255, 0, 0), (marker_x, bar_y - 5), (marker_x, bar_y + bar_h + 5), 2)

    def _draw_matrix_panel(self):
        """Draw K_rot and K_pos matrix visualization."""
        cell_size = 40 if self.S <= 6 else 35 if self.S <= 8 else 30

        # Position matrices at bottom right
        panel_width = cell_size * self.S + 60
        panel_height = cell_size * self.S * 2 + 100

        x_start = self.config.width - panel_width - 10
        y_start = self.config.height - panel_height - 10

        # Draw K_rot matrix
        self._draw_single_matrix(
            self.alignment_matrix, "K_rot (time-varying)",
            x_start, y_start, cell_size, highlight_antisym=True
        )

        # Draw K_pos matrix below
        y_pos = y_start + cell_size * self.S + 50
        self._draw_single_matrix(
            self.matrix, "K_pos (fixed)",
            x_start, y_pos, cell_size, highlight_antisym=False
        )

    def _draw_single_matrix(self, matrix, title, x_start, y_start, cell_size, highlight_antisym=False):
        """Draw a single matrix with color coding."""
        # Title
        title_color = (100, 100, 200) if "rot" in title.lower() else (100, 150, 100)
        title_surf = self.font.render(title, True, title_color)
        self.screen.blit(title_surf, (x_start, y_start - 20))

        # Draw column indicators (species colors)
        for j in range(self.S):
            cx = x_start + j * cell_size + cell_size // 2
            cy = y_start + 5
            pygame.draw.circle(self.screen, self.colors[j], (cx, cy), 5)

        y_offset = y_start + 15

        for i in range(self.S):
            # Row indicator
            rx = x_start - 12
            ry = y_offset + i * cell_size + cell_size // 2
            pygame.draw.circle(self.screen, self.colors[i], (rx, ry), 5)

            for j in range(self.S):
                x = x_start + j * cell_size
                y = y_offset + i * cell_size
                value = matrix[i, j]

                # Background color based on value
                if abs(value) < 0.01:
                    bg_color = (220, 220, 220)
                elif value > 0:
                    intensity = min(255, int(abs(value) * 255))
                    bg_color = (255 - intensity//2, 255, 255 - intensity//2)  # Green tint
                else:
                    intensity = min(255, int(abs(value) * 255))
                    bg_color = (255, 255 - intensity//2, 255 - intensity//2)  # Red tint

                pygame.draw.rect(self.screen, bg_color, (x, y, cell_size - 1, cell_size - 1))

                # Highlight antisymmetric pairs (adjacent off-diagonals)
                if highlight_antisym and abs(i - j) == 1:
                    pygame.draw.rect(self.screen, (100, 100, 255), (x, y, cell_size - 1, cell_size - 1), 2)

                # Value text - always show (no +/- sign)
                val_str = f"{value:.2f}"
                if abs(value) < 0.01:
                    text_color = (150, 150, 150)  # Gray for zero
                elif value > 0:
                    text_color = (0, 100, 0)  # Green for positive
                else:
                    text_color = (180, 0, 0)  # Red for negative
                val_surf = self.font.render(val_str, True, text_color)
                text_rect = val_surf.get_rect(center=(x + cell_size//2, y + cell_size//2))
                self.screen.blit(val_surf, text_rect)

        # Draw joint values indicator for K_rot
        if highlight_antisym:
            joint_y = y_offset + self.S * cell_size + 5
            joint_vals = self.get_current_joint_values()
            joint_str = "Joints: " + " ".join([f"{v:+.1f}" for v in joint_vals[:min(6, len(joint_vals))]])
            if len(joint_vals) > 6:
                joint_str += "..."
            joint_surf = self.font.render(joint_str, True, (80, 80, 80))
            self.screen.blit(joint_surf, (x_start - 10, joint_y))

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
                    self.current_step = 0
                    print("Reset")
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_m:
                    self.show_matrix = not self.show_matrix
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
    demo = UtoMDemo()
    demo.run()


if __name__ == "__main__":
    main()
