#!/usr/bin/env python3
"""
Formation Locomotion Demo — Shape + Movement

Combines shape formation (PID joint-angle control) with snake-style locomotion.
Particles first form a shape (C, U, custom drawn, etc.), then the whole
formation can move and steer while maintaining its shape.

Key insight: K_rot = K_shape (symmetric, PID) + K_locomotion (antisymmetric)
These are independent — antisymmetric forces don't change joint angles,
so the PID and locomotion don't fight each other.

Steering is achieved by adding a uniform bias to all target joint angles,
which curves the chain → the formation follows a curved path.

Controls:
    ←/→:   Steer (bias all target joint angles)
    ↑/↓:   Move speed (forward/backward)
    1-4:   Shape pattern (STRAIGHT/U/M/HUG)
    G:     Draw open shape (mouse)
    F:     Draw filled contour (mouse)
    [/]:   Adjust curvature (phi0)
    C:     Toggle PID control mode
    +/-:   Add/remove species
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
    generate_position_matrix,
    generate_translation_matrix,
    wrap_to_pi,
)


class FormationLocomotion(ShapeFormationDemo):
    """
    Shape formation + locomotion demo.

    Inherits all shape formation capabilities (PID, drawing, patterns).
    Adds antisymmetric K_rot for forward movement and uniform target-angle
    bias for steering.
    """

    def __init__(self, n_species: int = 6, n_particles: int = 20):
        super().__init__(n_species=n_species, n_particles=n_particles)

        # Locomotion parameters
        self.move_speed = 0.0       # Antisymmetric K_rot strength (0 = stationary)
        self.move_speed_max = 0.3   # Max locomotion strength
        self.move_speed_step = 0.005  # Speed change per frame when key held

        # Steering: uniform bias added to ALL target joint angles
        self.steer_bias = 0.0       # Radians added to all phi*
        self.steer_bias_max = 0.3   # Max steering curvature bias
        self.steer_bias_step = 0.01  # Bias change per frame when key held
        self.steer_decay = 0.95     # Bias decays when keys released

        # Start in control mode (PID active)
        self.control_mode = True
        self.show_centroids = False
        self.phi0 = 0.4             # Lower curvature → wider, more open shapes

        # PD gains (no integral — ki causes jitter from windup)
        self.kp = 0.4
        self.ki = 0.0
        self.kd = 3.5
        self.u_max = 0.3
        self.ctrl_alpha = 0.05      # Heavy output smoothing
        self.e_deadzone = 0.05      # Ignore errors < ~3° to prevent jitter

        # Stronger cross-attraction to resist compression during locomotion
        # (purely local: each species only attracts/repels its neighbors)
        self.matrix = generate_position_matrix(
            n_species, self_cohesion=0.6, cross_attraction=0.25,
            particles_per_species=n_particles
        )

        # Override window title
        pygame.display.set_caption("Formation Locomotion Demo")

        print()
        print("=" * 60)
        print("Formation Locomotion — Shape + Movement")
        print("=" * 60)
        print("  ←/→     Steer formation")
        print("  ↑/↓     Forward speed")
        print("  1-4     Shape: STRAIGHT/U/M/HUG")
        print("  G/F     Draw shape / filled contour")
        print("  [/]     Adjust curvature (phi0)")
        print("  C       Toggle PID control")
        print("=" * 60)

    def change_species_count(self, new_count: int):
        """Override: cap at 10, start from random positions."""
        new_count = max(2, min(10, new_count))
        if new_count == self.n_species:
            return
        super().change_species_count(new_count)
        # Randomize instead of side-by-side
        self._randomize()
        self._seed_phi_prev()
        self.move_speed = 0.0
        self.steer_bias = 0.0

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

        # Heavy low-pass filter on phi (0.15 = 85% old + 15% new)
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

        # Write K_rot (symmetric = bending)
        self.alignment_matrix[:] = 0.0
        for i in range(n_edges):
            val = np.clip(u_edge[i], -1.0, 1.0)
            self.alignment_matrix[i, i + 1] = val
            self.alignment_matrix[i + 1, i] = val

    def _get_target_profile(self):
        """Override: add uniform steer_bias to all target joint angles."""
        phi_star = super()._get_target_profile()
        if self.steer_bias != 0.0 and len(phi_star) > 0:
            phi_star += self.steer_bias
        return phi_star

    def _apply_locomotion(self):
        """Add uniform antisymmetric locomotion component to K_rot.

        For each edge i: K[i,i+1] += move_speed, K[i+1,i] -= move_speed.
        This creates same-direction tangential forces on both species in each
        pair → net forward translation perpendicular to each local segment.

        Note: on curved chains, the tangential directions diverge, creating a
        net inward (compressive) force component. This is the true physics of
        local-only tangential drive on curved formations. Shape compression is
        resisted by K_pos (inter-species radial attraction/repulsion), not by
        global correction.
        """
        if abs(self.move_speed) < 1e-6:
            return

        n_edges = self.n_species - 1
        for i in range(n_edges):
            self.alignment_matrix[i, i + 1] = np.clip(
                self.alignment_matrix[i, i + 1] + self.move_speed, -1.0, 1.0
            )
            self.alignment_matrix[i + 1, i] = np.clip(
                self.alignment_matrix[i + 1, i] - self.move_speed, -1.0, 1.0
            )

    def step(self):
        """Shape PID + locomotion + physics."""
        if self.paused:
            return

        # Steer bias decays when keys released
        self.steer_bias *= self.steer_decay
        if abs(self.steer_bias) < 0.001:
            self.steer_bias = 0.0

        if self.control_mode:
            # PID writes symmetric K_rot for shape maintenance
            self.head_bias *= self.turn_decay
            if abs(self.head_bias) < 0.01:
                self.head_bias = 0.0
            self.update_matrices_control_mode()
        else:
            self.turn_input *= self.turn_decay
            if abs(self.turn_input) < 0.01:
                self.turn_input = 0.0
            self.update_matrices_from_input()

        # Add antisymmetric locomotion on top of shape PID
        self._apply_locomotion()

        # Physics (skip ShapeFormationDemo.step, call grandparent directly)
        ParticleLife.step(self)

    def draw(self):
        """Draw with locomotion status."""
        # Call parent draw (particles, centroids, info, shape overlay)
        super().draw()

        if self.hide_gui:
            return

        # Locomotion status (bottom right)
        self._draw_locomotion_status()

    def _draw_locomotion_status(self):
        """Draw locomotion speed and steering indicators (bottom left)."""
        x = 10
        y = self.config.height - 100

        # Speed bar
        bar_w, bar_h = 120, 12
        pygame.draw.rect(self.screen, (220, 220, 220), (x, y, bar_w, bar_h))
        if abs(self.move_speed) > 1e-6:
            fill_w = int(abs(self.move_speed) / self.move_speed_max * bar_w / 2)
            cx = x + bar_w // 2
            if self.move_speed > 0:
                pygame.draw.rect(self.screen, (80, 180, 80), (cx, y, fill_w, bar_h))
            else:
                pygame.draw.rect(self.screen, (180, 80, 80), (cx - fill_w, y, fill_w, bar_h))
        # Center tick
        pygame.draw.line(self.screen, (100, 100, 100),
                         (x + bar_w // 2, y), (x + bar_w // 2, y + bar_h), 2)

        speed_text = self.font.render(f"Speed: {self.move_speed:+.3f}", True, (80, 80, 80))
        self.screen.blit(speed_text, (x, y - 18))

        # Steer bar
        y2 = y + 30
        pygame.draw.rect(self.screen, (220, 220, 220), (x, y2, bar_w, bar_h))
        if abs(self.steer_bias) > 1e-4:
            fill_w = int(abs(self.steer_bias) / self.steer_bias_max * bar_w / 2)
            cx = x + bar_w // 2
            color = (100, 130, 220) if self.steer_bias > 0 else (220, 130, 100)
            if self.steer_bias > 0:
                pygame.draw.rect(self.screen, color, (cx, y2, fill_w, bar_h))
            else:
                pygame.draw.rect(self.screen, color, (cx - fill_w, y2, fill_w, bar_h))
        pygame.draw.line(self.screen, (100, 100, 100),
                         (x + bar_w // 2, y2), (x + bar_w // 2, y2 + bar_h), 2)

        steer_text = self.font.render(
            f"Steer: {np.degrees(self.steer_bias):+.1f}\u00b0", True, (80, 80, 80))
        self.screen.blit(steer_text, (x, y2 - 18))

    def handle_events(self) -> bool:
        """Override arrow key behavior for locomotion."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # Mouse events for shape drawing (delegate to parent logic)
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
                    self.move_speed = 0.0
                    self.steer_bias = 0.0
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

                elif event.key == pygame.K_c:
                    self.control_mode = not self.control_mode
                    print(f"Control mode: {'ON' if self.control_mode else 'OFF'}")

                elif event.key == pygame.K_g:
                    self.drawing_mode = True
                    self.fill_draw_mode = False
                    print("Draw open shape (click and drag)")
                elif event.key == pygame.K_f and not self.matrix_edit_mode:
                    self.fill_draw_mode = True
                    self.drawing_mode = False
                    print("Draw filled contour (click and drag)")

                # Species count (before control_mode block so +/- always works)
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

        # Held keys — locomotion
        keys = pygame.key.get_pressed()

        # Up/Down: forward speed
        if keys[pygame.K_UP]:
            self.move_speed = min(self.move_speed_max,
                                  self.move_speed + self.move_speed_step)
        if keys[pygame.K_DOWN]:
            self.move_speed = max(-self.move_speed_max,
                                  self.move_speed - self.move_speed_step)

        # Left/Right: steer (bias all target angles)
        if keys[pygame.K_LEFT]:
            self.steer_bias = max(-self.steer_bias_max,
                                  self.steer_bias - self.steer_bias_step)
        if keys[pygame.K_RIGHT]:
            self.steer_bias = min(self.steer_bias_max,
                                  self.steer_bias + self.steer_bias_step)

        return True


def main():
    demo = FormationLocomotion(n_species=6, n_particles=20)
    demo.run()


if __name__ == "__main__":
    main()
