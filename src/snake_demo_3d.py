#!/usr/bin/env python3
"""
3D Snake Demo — Steerable Chain in 3D Space

A chain of particle clusters (species) connected by position-matrix attraction,
extended to 3D with yaw (left/right) and pitch (up/down) steering.

Controls:
    ←/→:     Yaw steering (horizontal turn)
    ↑/↓:     Pitch steering (vertical turn)
    Z/X:     Roll steering (barrel roll)
    +/-:     Increase/decrease forward speed
    R:       Reset positions
    SPACE:   Pause/Resume
    H:       Hide/show all GUI
    I:       Toggle info panel
    V:       Toggle centroid spine
    A:       Toggle 3D axes/bounding box
    HOME:    Reset camera

Camera:
    Left drag:   Rotate camera
    Right drag:  Pan view
    Scroll:      Zoom
    Q/ESC:       Quit
"""

import pygame
import pygame.gfxdraw
import numpy as np
from particle_life_3d import Config3D, ParticleLife3D
from snake_demo import generate_position_matrix


class SnakeDemo3D(ParticleLife3D):
    """
    3D snake demo with yaw/pitch/roll steering via K_rot_y, K_rot_x, K_rot_z.

    Species are arranged as a chain along the X axis.
    Left/Right arrows control yaw (Y-axis rotation).
    Up/Down arrows control pitch (X-axis rotation).
    Z/X keys control roll (Z-axis rotation).
    +/- keys control forward movement speed.
    """

    def __init__(self, n_species: int = 6, n_particles: int = 20):
        # Build K_pos chain matrix (reuse 2D generator — it's species-indexed)
        k_pos = generate_position_matrix(n_species)
        zeros = np.zeros((n_species, n_species)).tolist()

        config = Config3D(
            n_species=n_species,
            n_particles=n_particles,
            sim_width=16.0,
            sim_height=16.0,
            sim_depth=16.0,
            a_rot=3.0,
            position_matrix=k_pos.tolist(),
            orientation_matrix_x=zeros,
            orientation_matrix_y=zeros,
            orientation_matrix_z=zeros,
        )

        super().__init__(config, headless=False)

        # Control state — three steering axes
        self.yaw_input = 0.0        # -1 (full left) to +1 (full right)
        self.pitch_input = 0.0      # -1 (full down) to +1 (full up)
        self.roll_input = 0.0       # -1 (roll left) to +1 (roll right)
        self.base_k_rot = 0.05      # Base rotation matrix strength
        self.forward_speed = 0.1    # Forward bias in K_pos

        # Input smoothing
        self.turn_decay = 0.92

        # Delay propagation: each joint gets the head's signal
        # delayed by joint_delay steps per joint down the chain
        self.joint_delay = 8
        n_joints = n_species - 1
        history_len = self.joint_delay * n_joints + 1
        self.yaw_history = np.zeros(history_len)
        self.pitch_history = np.zeros(history_len)
        self.roll_history = np.zeros(history_len)
        self.history_idx = 0

        # Formation parameters
        self.group_spacing = 0.8

        # GUI state
        self.hide_gui = False
        self.show_centroids = True

        # Initialize particles in chain formation
        self._initialize_chain()

        # Override window title
        pygame.display.set_caption("3D Snake Demo — Arrow Keys to Steer")

        print("=" * 60)
        print("3D Snake Demo — Yaw/Pitch/Roll Steering")
        print("=" * 60)
        print(f"Species: {self.n_species}  Particles: {self.n}")
        print(f"Space: {self.config.sim_width}x{self.config.sim_height}x{self.config.sim_depth}m")
        print("")
        print("Controls:")
        print("  ←/→       Yaw (horizontal turn)")
        print("  ↑/↓       Pitch (vertical turn)")
        print("  Z/X       Roll (barrel roll)")
        print("  +/-       Forward speed")
        print("  R         Reset positions")
        print("  SPACE     Pause")
        print("  H         Hide/show GUI")
        print("  I         Toggle info panel")
        print("  V         Toggle centroid spine")
        print("  A         Toggle axes/bounding box")
        print("  HOME      Reset camera")
        print("")
        print("Camera:")
        print("  Left drag   Rotate")
        print("  Right drag  Pan")
        print("  Scroll      Zoom")
        print("=" * 60)

    def _initialize_chain(self):
        """Arrange species in a chain along the X axis at the 3D center."""
        center_x = self.config.sim_width / 2
        center_y = self.config.sim_height / 2
        center_z = self.config.sim_depth / 2

        total_width = (self.n_species - 1) * self.group_spacing
        start_x = center_x - total_width / 2

        particles_per_species = self.n // self.n_species

        for i in range(self.n):
            species_id = min(i // particles_per_species, self.n_species - 1)
            self.species[i] = species_id

            # Group center along X axis
            group_x = start_x + species_id * self.group_spacing

            # Small 3D random offset within group
            self.positions[i, 0] = group_x + self.rng.uniform(-0.15, 0.15)
            self.positions[i, 1] = center_y + self.rng.uniform(-0.15, 0.15)
            self.positions[i, 2] = center_z + self.rng.uniform(-0.15, 0.15)

            self.velocities[i] = np.array([0.0, 0.0, 0.0])

    def update_matrices_from_input(self):
        """
        Update K_rot_y (yaw), K_rot_x (pitch), K_rot_z (roll) from delayed steering.

        Each joint receives the head's signal delayed by
        joint_delay * joint_index steps, creating a wave from head to tail.
        """
        # Record current inputs into histories
        self.yaw_history[self.history_idx] = self.yaw_input
        self.pitch_history[self.history_idx] = self.pitch_input
        self.roll_history[self.history_idx] = self.roll_input
        self.history_idx = (self.history_idx + 1) % len(self.yaw_history)

        n_joints = self.n_species - 1
        K_rot_y = np.zeros((self.n_species, self.n_species))
        K_rot_x = np.zeros((self.n_species, self.n_species))
        K_rot_z = np.zeros((self.n_species, self.n_species))

        for joint in range(n_joints):
            delay = joint * self.joint_delay
            idx = (self.history_idx - 1 - delay) % len(self.yaw_history)

            # Yaw: Y-axis rotation
            delayed_yaw = self.yaw_history[idx]
            strength_y = -self.base_k_rot * delayed_yaw
            K_rot_y[joint, joint + 1] = strength_y
            K_rot_y[joint + 1, joint] = strength_y

            # Pitch: X-axis rotation
            delayed_pitch = self.pitch_history[idx]
            strength_x = -self.base_k_rot * delayed_pitch
            K_rot_x[joint, joint + 1] = strength_x
            K_rot_x[joint + 1, joint] = strength_x

            # Roll: Z-axis rotation
            delayed_roll = self.roll_history[idx]
            strength_z = -self.base_k_rot * delayed_roll
            K_rot_z[joint, joint + 1] = strength_z
            K_rot_z[joint + 1, joint] = strength_z

        self.alignment_matrix_y[:] = np.clip(K_rot_y, -1.0, 1.0)
        self.alignment_matrix_x[:] = np.clip(K_rot_x, -1.0, 1.0)
        self.alignment_matrix_z[:] = np.clip(K_rot_z, -1.0, 1.0)

    def update_forward_speed(self, delta: float):
        """Adjust forward speed and regenerate K_pos."""
        self.forward_speed = np.clip(self.forward_speed + delta, -0.3, 0.5)
        new_matrix = generate_position_matrix(
            self.n_species, forward_bias=self.forward_speed
        )
        self.matrix[:] = new_matrix
        print(f"Forward speed: {self.forward_speed:.2f}")

    def step(self):
        """Perform one simulation step with control updates."""
        if self.paused:
            return

        # Decay all inputs toward neutral
        self.yaw_input *= self.turn_decay
        if abs(self.yaw_input) < 0.01:
            self.yaw_input = 0.0
        self.pitch_input *= self.turn_decay
        if abs(self.pitch_input) < 0.01:
            self.pitch_input = 0.0
        self.roll_input *= self.turn_decay
        if abs(self.roll_input) < 0.01:
            self.roll_input = 0.0

        # Update rotation matrices from steering
        self.update_matrices_from_input()

        # Call parent step for 3D physics
        super().step()

    # ================================================================
    # Drawing
    # ================================================================

    def draw(self):
        """Draw the 3D simulation with snake overlay."""
        self.screen.fill((255, 255, 255))

        # Draw bounding box and axes
        if self.show_axes:
            self.draw_bounding_box()

        # Project all particles
        screen_x, screen_y, depth = self.project_batch(self.positions)

        # Sort by depth (draw far particles first)
        indices = np.argsort(-depth)

        # Draw particles
        r = max(3, int(0.04 * self.ppu * self.cam_zoom))
        for i in indices:
            color = self.colors[self.species[i]]
            sx, sy = int(screen_x[i]), int(screen_y[i])
            try:
                pygame.gfxdraw.aacircle(self.screen, sx, sy, r, color)
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, r, color)
            except OverflowError:
                pass

        if self.hide_gui:
            return

        # Draw centroid spine (3D projected)
        if self.show_centroids:
            self.draw_centroid_spine_3d()

        # HUD overlay
        if self.show_info:
            self.draw_control_indicator()
            self.draw_info_panel()

        if self.paused:
            pause_text = self.font.render("PAUSED", True, (200, 50, 50))
            rect = pause_text.get_rect(center=(self.config.width // 2, 30))
            self.screen.blit(pause_text, rect)

    def draw_centroid_spine_3d(self):
        """Draw line connecting species centroids, projected to 2D."""
        centroids = self.get_species_centroids()
        if len(centroids) < 2:
            return

        # Project each centroid
        pts = []
        for c in centroids:
            sx, sy, _ = self.project_3d_to_2d(c)
            pts.append((int(sx), int(sy)))

        # Draw connecting line
        pygame.draw.lines(self.screen, (0, 0, 0), False, pts, 2)

        # Draw colored markers at each centroid
        head_r, tail_r = 8, 5
        for i, (cx, cy) in enumerate(pts):
            r = head_r if i == 0 else tail_r
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), r + 2)
            pygame.draw.circle(self.screen, self.colors[i], (cx, cy), r)

    def draw_control_indicator(self):
        """Draw crosshair indicator for yaw and pitch input."""
        cx = self.config.width // 2
        cy = self.config.height - 70

        # Background circle
        pygame.draw.circle(self.screen, (230, 230, 230), (cx, cy), 50)
        pygame.draw.circle(self.screen, (200, 200, 200), (cx, cy), 50, 2)

        # Crosshair lines
        pygame.draw.line(self.screen, (210, 210, 210), (cx - 40, cy), (cx + 40, cy), 1)
        pygame.draw.line(self.screen, (210, 210, 210), (cx, cy - 40), (cx, cy + 40), 1)

        # Yaw indicator (horizontal bar)
        yaw_width = int(self.yaw_input * 40)
        if yaw_width != 0:
            color = (100, 150, 255) if yaw_width < 0 else (255, 150, 100)
            bar_x = cx if yaw_width > 0 else cx + yaw_width
            pygame.draw.rect(self.screen, color, (bar_x, cy - 4, abs(yaw_width), 8))

        # Pitch indicator (vertical bar)
        pitch_height = int(-self.pitch_input * 40)  # Invert: positive pitch = up on screen
        if pitch_height != 0:
            color = (100, 255, 150) if pitch_height < 0 else (255, 200, 100)
            bar_y = cy if pitch_height > 0 else cy + pitch_height
            pygame.draw.rect(self.screen, color, (cx - 4, bar_y, 8, abs(pitch_height)))

        # Roll indicator (diagonal bar, bottom-left to top-right)
        roll_len = int(self.roll_input * 30)
        if roll_len != 0:
            color = (200, 100, 255) if roll_len < 0 else (255, 100, 200)
            # Draw along the diagonal
            dx = abs(roll_len)
            sign = 1 if roll_len > 0 else -1
            pygame.draw.line(self.screen, color,
                             (cx, cy),
                             (cx + sign * dx, cy - dx), 4)

        # Center dot
        pygame.draw.circle(self.screen, (50, 50, 50), (cx, cy), 4)

        # Labels
        yaw_text = self.font.render(f"Yaw: {self.yaw_input:+.2f}", True, (100, 100, 100))
        pitch_text = self.font.render(f"Pitch: {self.pitch_input:+.2f}", True, (100, 100, 100))
        roll_text = self.font.render(f"Roll: {self.roll_input:+.2f}", True, (100, 100, 100))
        self.screen.blit(yaw_text, (cx - 70, cy + 55))
        self.screen.blit(pitch_text, (cx - 70, cy + 75))
        self.screen.blit(roll_text, (cx - 70, cy + 95))

    def draw_info_panel(self):
        """Draw information panel."""
        info_lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Species: {self.n_species}  Particles: {self.n}",
            f"Camera: yaw={np.degrees(self.cam_yaw):.0f}° pitch={np.degrees(self.cam_pitch):.0f}°",
            "",
            f"Yaw Input: {self.yaw_input:+.2f}",
            f"Pitch Input: {self.pitch_input:+.2f}",
            f"Roll Input: {self.roll_input:+.2f}",
            f"Forward Speed: {self.forward_speed:+.2f}",
            "",
            "Controls:",
            "←/→: Yaw  ↑/↓: Pitch  Z/X: Roll",
            "+/-: Speed  R: Reset",
            "SPACE: Pause  A: Axes",
            "H: GUI  I: Info  V: Spine",
            "HOME: Reset camera  Q: Quit",
        ]

        y = 10
        for line in info_lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 22

    # ================================================================
    # Event handling
    # ================================================================

    def handle_events(self) -> bool:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # --- Mouse: camera controls (from ParticleLife3D) ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 3:
                    self.right_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
                elif event.button == 3:
                    self.right_dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    pos = pygame.mouse.get_pos()
                    if self.last_mouse_pos:
                        dx = pos[0] - self.last_mouse_pos[0]
                        dy = pos[1] - self.last_mouse_pos[1]
                        self.cam_yaw += dx * 0.01
                        self.cam_pitch -= dy * 0.01
                        self.cam_pitch = np.clip(self.cam_pitch,
                                                 -np.pi / 2 + 0.1,
                                                 np.pi / 2 - 0.1)
                    self.last_mouse_pos = pos
                elif self.right_dragging:
                    pos = pygame.mouse.get_pos()
                    if self.last_mouse_pos:
                        dx = pos[0] - self.last_mouse_pos[0]
                        dy = pos[1] - self.last_mouse_pos[1]
                        self.cam_pan[0] += dx
                        self.cam_pan[1] += dy
                    self.last_mouse_pos = pos

            elif event.type == pygame.MOUSEWHEEL:
                zoom_factor = 1.1 if event.y > 0 else 0.9
                self.cam_zoom *= zoom_factor
                self.cam_zoom = np.clip(self.cam_zoom, 0.2, 5.0)

            # --- Keyboard ---
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self._initialize_chain()
                    self.yaw_input = 0.0
                    self.pitch_input = 0.0
                    self.roll_input = 0.0
                    self.yaw_history[:] = 0.0
                    self.pitch_history[:] = 0.0
                    self.roll_history[:] = 0.0
                    self.history_idx = 0
                    print("Reset positions")

                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info

                elif event.key == pygame.K_v:
                    self.show_centroids = not self.show_centroids

                elif event.key == pygame.K_a:
                    self.show_axes = not self.show_axes

                elif event.key == pygame.K_h:
                    self.hide_gui = not self.hide_gui

                elif event.key == pygame.K_HOME:
                    self.cam_yaw = np.pi / 6
                    self.cam_pitch = np.pi / 6
                    self.cam_zoom = 1.0
                    self.cam_pan = np.array([0.0, 0.0])

                # Forward speed
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.update_forward_speed(0.05)
                elif event.key == pygame.K_MINUS:
                    self.update_forward_speed(-0.05)

        # Continuous key input for steering
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.yaw_input = max(-1.0, self.yaw_input - 0.05)
        if keys[pygame.K_RIGHT]:
            self.yaw_input = min(1.0, self.yaw_input + 0.05)
        if keys[pygame.K_UP]:
            self.pitch_input = min(1.0, self.pitch_input + 0.05)
        if keys[pygame.K_DOWN]:
            self.pitch_input = max(-1.0, self.pitch_input - 0.05)
        if keys[pygame.K_z]:
            self.roll_input = max(-1.0, self.roll_input - 0.05)
        if keys[pygame.K_x]:
            self.roll_input = min(1.0, self.roll_input + 0.05)

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
    demo = SnakeDemo3D(n_species=8, n_particles=20)
    demo.run()


if __name__ == "__main__":
    main()
