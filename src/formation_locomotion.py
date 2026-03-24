#!/usr/bin/env python3
"""
Formation Locomotion Demo — Shape + Movement

Combines shape formation (PID joint-angle control) with forward motion via
asymmetric K_pos (position matrix), same mechanism as the snake demo.

- K_rot: PID maintains shape (C, U, custom, etc.)
- K_pos: asymmetric forward_bias creates drift along chain axis toward head
- The shape's orientation doesn't change — only its position moves.

Controls:
    ↑/↓:   Increase/decrease forward speed (K_pos forward_bias)
    1-4:   Shape pattern (STRAIGHT/U/M/HUG)
    G:     Draw open shape (mouse)
    F:     Draw filled contour (mouse)
    [/]:   Adjust curvature (phi0)
    C:     Toggle PID control mode
    +/-:   Add/remove species (2-10)
    R:     Reset positions
    S:     Scatter randomly
    SPACE: Pause/Resume
    I:     Toggle info panel
    Q/ESC: Quit
"""

import pygame
import numpy as np
from particle_life import ParticleLife
from shape_formation import (
    MultiSpeciesDemo as ShapeFormationDemo,
    wrap_to_pi,
)


class FormationLocomotion(ShapeFormationDemo):
    """
    Shape formation + locomotion via K_pos forward bias.

    Inherits all shape formation capabilities (PID, drawing, patterns).
    Forward motion comes from asymmetric K_pos (snake demo mechanism):
    each species is more attracted to the one behind than ahead, creating
    net drift along the chain axis toward species 0 (head).
    """

    def __init__(self, n_species: int = 6, n_particles: int = 20):
        super().__init__(n_species=n_species, n_particles=n_particles)

        # Lateral motion via antisymmetric K_rot (perpendicular to chain axis)
        self.lateral_speed = 0.0
        self.lateral_speed_max = 0.02
        self.lateral_speed_step = 0.005

        # Forward motion via K_pos asymmetry (along chain axis toward head)
        self.forward_bias = 0.0
        self.forward_bias_max = 0.25
        self.forward_bias_step = 0.02

        # Trajectory tracking
        self.show_trajectory = False
        self.trajectory = []          # list of (x, y) centroid positions
        self.trajectory_max = 500     # max points to keep

        # Start in control mode (PID active)
        self.control_mode = True
        self.show_centroids = False
        self.phi0 = 0.25

        # PD gains (no integral)
        self.kp = 0.4
        self.ki = 0.0
        self.kd = 3.5
        self.u_max = 0.3
        self.ctrl_alpha = 0.05
        self.e_deadzone = 0.05

        # Base K_pos parameters
        self.self_cohesion = 0.6
        self.cross_attraction = 0.25
        self._update_kpos()

        # Override window title
        pygame.display.set_caption("Formation Locomotion Demo")

        print()
        print("=" * 60)
        print("Formation Locomotion — Shape + Movement")
        print("=" * 60)
        print("  ↑/↓     Lateral speed (K_rot, perpendicular to chain)")
        print("  ←/→     Forward speed (K_pos, along chain)")
        print("  1-4     Shape: STRAIGHT/U/M/HUG")
        print("  G/F     Draw shape / filled contour")
        print("  [/]     Adjust curvature (phi0)")
        print("  C       Toggle PID control")
        print("=" * 60)

    def _update_kpos(self):
        """Regenerate K_pos with current forward_bias."""
        n = self.n_species
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = self.self_cohesion
            if i > 0:
                K[i, i - 1] = self.cross_attraction  # attraction to species behind
            if i < n - 1:
                K[i, i + 1] = self.cross_attraction - self.forward_bias  # weaker ahead
        self.matrix[:] = K

    def change_species_count(self, new_count: int):
        """Override: cap at 10, start from random positions."""
        new_count = max(2, min(10, new_count))
        if new_count == self.n_species:
            return
        super().change_species_count(new_count)
        self._randomize()
        self._seed_phi_prev()
        self.forward_bias = 0.0
        self._update_kpos()

    def update_matrices_control_mode(self):
        """Override PID with heavier filtering, dead zone, no integral."""
        S = self.n_species
        if S < 3:
            self.alignment_matrix[:] = 0.0
            return

        n_joints = S - 2
        n_edges = S - 1

        from shape_formation import compute_segment_angles, compute_joint_angles

        centroids = np.array(self.get_species_centroids())
        theta = compute_segment_angles(centroids)
        phi = compute_joint_angles(theta)

        phi_star = self._get_target_profile()

        # Error with dead zone
        error = wrap_to_pi(phi_star - phi)
        error = np.where(np.abs(error) < self.e_deadzone, 0.0, error)
        error = np.clip(error, -self.e_max, self.e_max)

        # Heavy low-pass filter on phi
        alpha_filt = 0.15
        phi_filt = (1 - alpha_filt) * self.phi_filtered + alpha_filt * phi
        phi_dot = wrap_to_pi(phi_filt - self.phi_filtered)

        self.phi_filtered = phi_filt.copy()
        self.phi_prev = phi.copy()
        self.ctrl_mean_error = float(np.mean(np.abs(error)))

        # PD law (no integral)
        u_joint = self.kp * error - self.kd * phi_dot
        u_joint = np.clip(u_joint, -self.u_max, self.u_max)

        # Map to edges
        u_edge = np.zeros(n_edges)
        for j in range(n_joints):
            u_edge[j] += u_joint[j]
            u_edge[j + 1] -= u_joint[j]

        # Heavy exponential smoothing
        a = self.ctrl_alpha
        u_edge = (1 - a) * self.u_edge_prev + a * u_edge
        self.u_edge_prev = u_edge.copy()

        # Write K_rot (symmetric = bending, shape only)
        self.alignment_matrix[:] = 0.0
        for i in range(n_edges):
            val = np.clip(u_edge[i], -1.0, 1.0)
            self.alignment_matrix[i, i + 1] = val
            self.alignment_matrix[i + 1, i] = val

    def _apply_lateral(self):
        """Add antisymmetric K_rot for lateral motion (perpendicular to chain)."""
        if abs(self.lateral_speed) < 1e-6:
            return
        n_edges = self.n_species - 1
        for i in range(n_edges):
            self.alignment_matrix[i, i + 1] = np.clip(
                self.alignment_matrix[i, i + 1] - self.lateral_speed, -1.0, 1.0
            )
            self.alignment_matrix[i + 1, i] = np.clip(
                self.alignment_matrix[i + 1, i] + self.lateral_speed, -1.0, 1.0
            )

    def step(self):
        """Shape PID + K_pos locomotion + physics."""
        if self.paused:
            return

        if self.control_mode:
            self.head_bias *= self.turn_decay
            if abs(self.head_bias) < 0.01:
                self.head_bias = 0.0
            self.update_matrices_control_mode()
        else:
            self.turn_input *= self.turn_decay
            if abs(self.turn_input) < 0.01:
                self.turn_input = 0.0
            self.update_matrices_from_input()

        # Add lateral locomotion (antisymmetric K_rot)
        self._apply_lateral()

        # Physics
        ParticleLife.step(self)

        # Record trajectory
        if self.show_trajectory:
            centroid = self.positions.mean(axis=0)
            sx = int(centroid[0] * self.ppu)
            sy = int(centroid[1] * self.ppu)
            self.trajectory.append((sx, sy))
            if len(self.trajectory) > self.trajectory_max:
                self.trajectory.pop(0)

    def _draw_trajectory(self):
        """Draw the swarm centroid trajectory as a fading trail."""
        n = len(self.trajectory)
        if n < 2:
            return
        for i in range(1, n):
            alpha = int(80 + 175 * i / n)  # fade from dim to bright
            color = (alpha, alpha // 2, alpha // 3)
            pygame.draw.line(self.screen, color,
                             self.trajectory[i - 1], self.trajectory[i], 2)

    def draw(self):
        """Draw with trajectory and locomotion status."""
        self.screen.fill((255, 255, 255))

        # Trajectory underneath everything
        if self.show_trajectory:
            self._draw_trajectory()

        # Particles
        self.draw_particles()

        # Shape extraction overlay
        if self.shape_vis is not None:
            self.draw_shape_extraction_overlay()

        if self.hide_gui:
            return

        if self.show_centroids:
            pts = self.draw_centroid_spine()
            self.draw_centroid_markers(pts)
            self.draw_swarm_centroid()

        if self.show_info:
            self.draw_control_indicator()
            self.draw_info_panel()

        self.draw_pause_indicator()
        self._draw_locomotion_status()

    def _draw_locomotion_status(self):
        """Draw forward and lateral indicators (bottom left)."""
        x = 10
        y = self.config.height - 100
        bar_w, bar_h = 120, 12

        # Forward (K_pos)
        pygame.draw.rect(self.screen, (220, 220, 220), (x, y, bar_w, bar_h))
        if abs(self.forward_bias) > 1e-6:
            fill_w = int(abs(self.forward_bias) / self.forward_bias_max * bar_w / 2)
            cx = x + bar_w // 2
            color = (80, 180, 80) if self.forward_bias > 0 else (180, 80, 80)
            if self.forward_bias > 0:
                pygame.draw.rect(self.screen, color, (cx, y, fill_w, bar_h))
            else:
                pygame.draw.rect(self.screen, color, (cx - fill_w, y, fill_w, bar_h))
        pygame.draw.line(self.screen, (100, 100, 100),
                         (x + bar_w // 2, y), (x + bar_w // 2, y + bar_h), 2)
        text = self.font.render(f"Forward (K_pos): {self.forward_bias:+.3f}", True, (80, 80, 80))
        self.screen.blit(text, (x, y - 18))

        # Lateral (K_rot)
        y2 = y + 30
        pygame.draw.rect(self.screen, (220, 220, 220), (x, y2, bar_w, bar_h))
        if abs(self.lateral_speed) > 1e-6:
            fill_w = int(abs(self.lateral_speed) / self.lateral_speed_max * bar_w / 2)
            cx = x + bar_w // 2
            color = (100, 130, 220) if self.lateral_speed > 0 else (220, 130, 100)
            if self.lateral_speed > 0:
                pygame.draw.rect(self.screen, color, (cx, y2, fill_w, bar_h))
            else:
                pygame.draw.rect(self.screen, color, (cx - fill_w, y2, fill_w, bar_h))
        pygame.draw.line(self.screen, (100, 100, 100),
                         (x + bar_w // 2, y2), (x + bar_w // 2, y2 + bar_h), 2)
        text = self.font.render(f"Lateral (K_rot): {self.lateral_speed:.4f}", True, (80, 80, 80))
        self.screen.blit(text, (x, y2 - 18))

    def handle_events(self) -> bool:
        """Handle events with K_pos locomotion controls."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # Mouse events for shape drawing
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.drawing_mode or self.fill_draw_mode:
                    self.mouse_drawing = True
                    self.drawing_points = [event.pos]

            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_drawing:
                    self.drawing_points.append(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if self.mouse_drawing:
                    self.mouse_drawing = False
                    if self.fill_draw_mode:
                        self._process_drawn_contour_fill()
                        self.fill_draw_mode = False
                    elif self.drawing_mode:
                        self._process_drawn_shape()
                        self.drawing_mode = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self._initialize_side_by_side()
                    self.forward_bias = 0.0
                    self.lateral_speed = 0.0
                    self.trajectory.clear()
                    self._update_kpos()
                    self._init_control_state()
                    self._seed_phi_prev()
                    print("Reset positions")

                elif event.key == pygame.K_s and not self.matrix_edit_mode:
                    self._randomize()
                    self._init_control_state()
                    self._seed_phi_prev()
                    print("Scattered")

                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_v:
                    self.show_centroids = not self.show_centroids
                elif event.key == pygame.K_h:
                    self.hide_gui = not self.hide_gui
                elif event.key == pygame.K_o:
                    self.show_orientations = not self.show_orientations
                elif event.key == pygame.K_t:
                    self.show_trajectory = not self.show_trajectory
                    if not self.show_trajectory:
                        self.trajectory.clear()

                elif event.key == pygame.K_c:
                    self.control_mode = not self.control_mode
                    print(f"Control mode: {'ON' if self.control_mode else 'OFF'}")

                # Locomotion (discrete steps per keypress)
                elif event.key == pygame.K_UP:
                    self.lateral_speed = min(self.lateral_speed_max,
                                             self.lateral_speed + 0.005)
                    print(f"Forward (K_rot): {self.lateral_speed:.4f}")
                elif event.key == pygame.K_DOWN:
                    self.lateral_speed = max(0.0,
                                             self.lateral_speed - 0.005)
                    print(f"Forward (K_rot): {self.lateral_speed:.4f}")
                elif event.key == pygame.K_RIGHT:
                    self.forward_bias = min(self.forward_bias_max,
                                            self.forward_bias + 0.02)
                    if abs(self.forward_bias) < 0.01:
                        self.forward_bias = 0.0
                    self._update_kpos()
                    print(f"Lateral (K_pos): {self.forward_bias:+.3f}")
                elif event.key == pygame.K_LEFT:
                    self.forward_bias = max(-self.forward_bias_max,
                                            self.forward_bias - 0.02)
                    if abs(self.forward_bias) < 0.01:
                        self.forward_bias = 0.0
                    self._update_kpos()
                    print(f"Lateral (K_pos): {self.forward_bias:+.3f}")

                elif event.key == pygame.K_g:
                    self.drawing_mode = True
                    self.fill_draw_mode = False
                    print("Draw open shape (click and drag)")
                elif event.key == pygame.K_f and not self.matrix_edit_mode:
                    self.fill_draw_mode = True
                    self.drawing_mode = False
                    print("Draw filled contour (click and drag)")

                # Species count
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.change_species_count(self.n_species + 1)
                elif event.key == pygame.K_MINUS:
                    self.change_species_count(self.n_species - 1)

                # Pattern and PID tuning (only in control mode)
                elif self.control_mode:
                    if event.key == pygame.K_1:
                        self.pattern_index = 0
                        print("Pattern: STRAIGHT")
                    elif event.key == pygame.K_2:
                        self.pattern_index = 1
                        print("Pattern: U")
                    elif event.key == pygame.K_3:
                        self.pattern_index = 2
                        print("Pattern: M")
                    elif event.key == pygame.K_4:
                        self.pattern_index = 3
                        print("Pattern: HUG")
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.phi0 = min(np.pi, self.phi0 + 0.05)
                        print(f"phi0: {self.phi0:.2f}")
                    elif event.key == pygame.K_LEFTBRACKET:
                        self.phi0 = max(0.0, self.phi0 - 0.05)
                        print(f"phi0: {self.phi0:.2f}")
                    elif event.key == pygame.K_k:
                        self.kp = min(10.0, self.kp + 0.2)
                        print(f"kp: {self.kp:.1f}")
                    elif event.key == pygame.K_j:
                        self.kp = max(0.0, self.kp - 0.2)
                        print(f"kp: {self.kp:.1f}")

                # Matrix editing
                elif event.key == pygame.K_m:
                    self.matrix_edit_mode = not self.matrix_edit_mode
                    print(f"Matrix edit: {'ON' if self.matrix_edit_mode else 'OFF'}")
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

        return True


def main():
    demo = FormationLocomotion(n_species=6, n_particles=20)
    demo.run()


if __name__ == "__main__":
    main()
