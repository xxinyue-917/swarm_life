#!/usr/bin/env python3
"""
Formation Transport Demo — Pick up and deliver an object

Particles form a C-shape, approach a passive object, capture it by
tightening the formation, transport it to a goal, and release it.

The object is pushed by nearby swarm particles via short-range repulsion.
When the C-shape surrounds the object, forces from multiple sides create
a net force that moves the object with the formation.

Controls:
    ↑/↓:   Forward speed (K_rot)
    ←/→:   Lateral speed (K_pos)
    [/]:   Tighten/loosen C-shape (capture/release)
    1-4:   Shape pattern
    T:     Toggle trajectory
    N:     New random object position
    R:     Reset everything
    +/-:   Species count (2-10)
    SPACE: Pause
    I:     Info panel
    Q/ESC: Quit
"""

import pygame
import numpy as np
from formation_locomotion import FormationLocomotion


class FormationTransport(FormationLocomotion):
    """
    Formation locomotion + passive object transport.

    A passive object is pushed by nearby swarm particles. The C-shape
    formation can capture, transport, and release the object.
    """

    def __init__(self, n_species: int = 10, n_particles: int = 20):
        super().__init__(n_species=n_species, n_particles=n_particles)

        sim_w = self.config.sim_width
        sim_h = self.config.sim_height

        # Passive object
        self.object_pos = np.array([sim_w * 0.7, sim_h * 0.5])
        self.object_vel = np.array([0.0, 0.0])
        self.object_radius = 0.3       # Visual/collision radius
        self.particle_radius = 0.08   # Approximate particle visual radius in sim coords
        self.object_damping = 0.92    # Velocity damping per frame

        # Goal
        self.goal_pos = np.array([sim_w * 0.15, sim_h * 0.5])
        self.goal_radius = 1.2
        self.task_complete = False

        # Start with U-shape pattern for C-shape formation
        self.pattern_index = 1
        self.phi0 = 0.3

        # Keep full trajectory (no limit)
        self.trajectory_max = 999999

        # Place particles in a small region on the left
        self._initialize_constrained()

        pygame.display.set_caption("Formation Transport Demo")

        print()
        print("=" * 60)
        print("Formation Transport — Pick up & Deliver")
        print("=" * 60)
        print("  1. Form C-shape (press 2 for U-shape)")
        print("  2. Move toward the orange object")
        print("  3. Tighten with ] to capture")
        print("  4. Transport to the green goal")
        print("  5. Loosen with [ to release")
        print("")
        print("  ↑/↓     Forward/backward")
        print("  ←/→     Lateral movement")
        print("  [/]     Loosen/tighten C-shape")
        print("  N       New object position")
        print("=" * 60)

    def _initialize_constrained(self):
        """Place all particles in a small region on the left side."""
        cx = self.config.sim_width * 0.25
        cy = self.config.sim_height * 0.5
        spread = 1.5  # meters radius

        particles_per_species = self.n // self.n_species
        for i in range(self.n):
            species_id = min(i // particles_per_species, self.n_species - 1)
            self.species[i] = species_id
            self.positions[i, 0] = cx + np.random.uniform(-spread, spread)
            self.positions[i, 1] = cy + np.random.uniform(-spread, spread)
            self.velocities[i] = np.array([0.0, 0.0])

    def _update_object(self):
        """Update object via hard-contact collision response.

        For each particle overlapping the object:
        1. Position correction — push object out so no overlap
        2. Velocity transfer — object receives particle's approach velocity

        This prevents overlap entirely and creates smooth pushing.
        """

        contact_dist = self.object_radius + self.particle_radius

        for i in range(self.n):
            delta = self.object_pos - self.positions[i]
            dist = np.linalg.norm(delta)

            if dist < contact_dist and dist > 1e-6:
                normal = delta / dist
                overlap = contact_dist - dist

                # 1. Position correction — eliminate overlap
                self.object_pos += normal * overlap

                # 2. Velocity transfer — smooth push
                particle_vel = self.velocities[i]
                rel_vel = particle_vel - self.object_vel
                approach = np.dot(rel_vel, normal)

                if approach > 0:
                    # Particle moving toward object — transfer velocity
                    self.object_vel += normal * approach * 0.3

        # Damping (friction)
        self.object_vel *= self.object_damping

        # Clamp speed
        speed = np.linalg.norm(self.object_vel)
        if speed > 1.5:
            self.object_vel = self.object_vel / speed * 1.5

        # Update position
        self.object_pos += self.object_vel * self.config.dt

        # Boundary reflection
        margin = 0.3
        sim_w, sim_h = self.config.sim_width, self.config.sim_height
        for dim, lim in enumerate([sim_w, sim_h]):
            if self.object_pos[dim] < margin:
                self.object_pos[dim] = margin
                self.object_vel[dim] = abs(self.object_vel[dim]) * 0.5
            elif self.object_pos[dim] > lim - margin:
                self.object_pos[dim] = lim - margin
                self.object_vel[dim] = -abs(self.object_vel[dim]) * 0.5

        # Check goal
        dist_to_goal = np.linalg.norm(self.object_pos - self.goal_pos)
        if dist_to_goal < self.goal_radius and not self.task_complete:
            self.task_complete = True
            print("Task complete! Object delivered to goal.")

    def step(self):
        """Parent step + object physics."""
        super().step()
        if not self.paused:
            self._update_object()

    def _draw_object(self):
        """Draw the passive object."""
        z = getattr(self, 'zoom', 1.0)
        ox = int(self.object_pos[0] * self.ppu * z)
        oy = int(self.object_pos[1] * self.ppu * z)
        r = max(4, int(self.object_radius * self.ppu * z))

        # Object body
        color = (240, 160, 50) if not self.task_complete else (100, 200, 100)
        pygame.draw.circle(self.screen, color, (ox, oy), r)
        pygame.draw.circle(self.screen, (180, 120, 30), (ox, oy), r, 2)


    def _draw_goal(self):
        """Draw the goal zone."""
        z = getattr(self, 'zoom', 1.0)
        gx = int(self.goal_pos[0] * self.ppu * z)
        gy = int(self.goal_pos[1] * self.ppu * z)
        gr = max(6, int(self.goal_radius * self.ppu * z))

        # Dashed circle effect — draw ring segments
        color = (80, 200, 80) if not self.task_complete else (50, 255, 50)
        pygame.draw.circle(self.screen, color, (gx, gy), gr, 2)

        # Inner fill (very transparent effect via lighter color)
        fill_color = (200, 240, 200) if not self.task_complete else (180, 255, 180)
        s = pygame.Surface((gr * 2, gr * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*fill_color, 60), (gr, gr), gr)
        self.screen.blit(s, (gx - gr, gy - gr))

        # Label
        label = self.font.render("GOAL", True, (60, 150, 60))
        self.screen.blit(label, label.get_rect(center=(gx, gy - gr - 10)))

    def _draw_task_status(self):
        """Draw distance to goal and task status."""
        dist = np.linalg.norm(self.object_pos - self.goal_pos)

        x = self.config.width // 2 - 60
        y = 10

        if self.task_complete:
            text = self.font.render("DELIVERED!", True, (50, 180, 50))
            self.screen.blit(text, (x, y))
        else:
            text = self.font.render(f"Dist to goal: {dist:.1f}m", True, (100, 100, 100))
            self.screen.blit(text, (x, y))

    def draw(self):
        """Draw everything."""
        self.screen.fill((255, 255, 255))

        # Trajectory underneath
        if self.show_trajectory:
            self._draw_trajectory()

        # Goal (underneath particles)
        self._draw_goal()

        # Object
        self._draw_object()

        # Particles
        self.draw_particles()

        # Shape overlay
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
        self._draw_task_status()

    def handle_events(self) -> bool:
        """Add N key for new object position."""
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

                elif event.key == pygame.K_n:
                    # New random object position
                    m = 2.0
                    sw, sh = self.config.sim_width, self.config.sim_height
                    self.object_pos = np.array([
                        np.random.uniform(m, sw - m),
                        np.random.uniform(m, sh - m)
                    ])
                    self.object_vel = np.array([0.0, 0.0])
                    self.task_complete = False
                    self.trajectory.clear()
                    print(f"New object at ({self.object_pos[0]:.1f}, {self.object_pos[1]:.1f})")

                elif event.key == pygame.K_r:
                    self._initialize_constrained()
                    self.forward_bias = 0.0
                    self.lateral_speed = 0.0
                    self.trajectory.clear()
                    self._update_kpos()
                    self._init_control_state()
                    self._seed_phi_prev()
                    # Reset object and goal
                    sw, sh = self.config.sim_width, self.config.sim_height
                    self.object_pos = np.array([sw * 0.7, sh * 0.5])
                    self.object_vel = np.array([0.0, 0.0])
                    self.task_complete = False
                    print("Reset")

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
                    print(f"Trajectory: {'ON' if self.show_trajectory else 'OFF'}")

                elif event.key == pygame.K_c:
                    self.control_mode = not self.control_mode
                    print(f"Control mode: {'ON' if self.control_mode else 'OFF'}")

                elif event.key == pygame.K_g:
                    self.drawing_mode = True
                    self.fill_draw_mode = False
                elif event.key == pygame.K_f and not self.matrix_edit_mode:
                    self.fill_draw_mode = True
                    self.drawing_mode = False

                # Locomotion (discrete steps)
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

                # Species count
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.change_species_count(self.n_species + 1)
                elif event.key == pygame.K_MINUS:
                    self.change_species_count(self.n_species - 1)

                # Pattern and PID tuning
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
                        print(f"phi0: {self.phi0:.2f} (tighter)")
                    elif event.key == pygame.K_LEFTBRACKET:
                        self.phi0 = max(0.0, self.phi0 - 0.05)
                        print(f"phi0: {self.phi0:.2f} (looser)")
                    elif event.key == pygame.K_k:
                        self.kp = min(10.0, self.kp + 0.2)
                        print(f"kp: {self.kp:.1f}")
                    elif event.key == pygame.K_j:
                        self.kp = max(0.0, self.kp - 0.2)
                        print(f"kp: {self.kp:.1f}")

                # Matrix editing
                elif event.key == pygame.K_m:
                    self.matrix_edit_mode = not self.matrix_edit_mode
                elif event.key == pygame.K_TAB and self.matrix_edit_mode:
                    self.editing_k_rot = not self.editing_k_rot
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
    demo = FormationTransport(n_species=10, n_particles=20)
    demo.run()


if __name__ == "__main__":
    main()
