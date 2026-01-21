#!/usr/bin/env python3
"""
Waypoint Following Demo via Orientation-Based Locomotion

Demonstrates how two species groups can follow waypoints by tuning
the orientation matrix (K_rot). The mechanism:
1. Position matrix creates two cohesive groups
2. Orientation matrix with OPPOSITE signs creates net translation
3. Dual PID control system:
   - Steering PID: adjusts K_rot asymmetry for turning
   - Speed PID: adjusts K_rot magnitude based on distance
4. When swarm reaches a waypoint, it advances to the next one

Controls:
    1/2/3/4/5: Switch path shape (circle, figure-8, line, square, spiral)
    +/-: Adjust waypoint reach threshold
    [/]: Adjust number of waypoints
    P: Toggle PID control on/off
    N: Skip to next waypoint
    SPACE: Pause/Resume
    R: Reset positions
    T: Toggle path visualization
    I: Toggle info panel
    Q/ESC: Quit
"""

import pygame
import numpy as np
from dataclasses import dataclass
from typing import List

# Import from the base simulation module
from particle_life import Config, ParticleLife


@dataclass
class ScalarPID:
    """
    PID controller for scalar values.

    Used for both steering (angular error) and speed (distance error) control.
    """
    kp: float = 1.0          # Proportional gain
    ki: float = 0.01         # Integral gain
    kd: float = 0.1          # Derivative gain
    integral_limit: float = 10.0  # Anti-windup limit

    def __post_init__(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_output = 0.0  # For debugging/display

    def compute(self, error: float, dt: float) -> float:
        """
        Compute PID output given scalar error.

        Args:
            error: Current error value (setpoint - measured)
            dt: Time step

        Returns:
            PID output value
        """
        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral

        # Derivative term (on error, not measurement)
        derivative = (error - self.prev_error) / max(dt, 0.001)
        self.prev_error = error
        d_term = self.kd * derivative

        # Combined output
        self.last_output = p_term + i_term + d_term
        return self.last_output

    def reset(self):
        """Reset PID state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_output = 0.0

    def get_terms(self) -> tuple:
        """Get individual PID terms for debugging."""
        return (
            self.kp * self.prev_error,  # P term (approximate)
            self.ki * self.integral,     # I term
            self.last_output - self.kp * self.prev_error - self.ki * self.integral  # D term (approximate)
        )


class PathGenerator:
    """Generates waypoints along predefined path shapes."""

    def __init__(self, center: np.ndarray, scale: float = 200.0):
        self.center = center
        self.scale = scale

    def circle(self, t: float) -> np.ndarray:
        """Point on circular path."""
        return self.center + self.scale * np.array([np.cos(t), np.sin(t)])

    def figure8(self, t: float) -> np.ndarray:
        """Point on figure-8 (lemniscate) path."""
        return self.center + self.scale * np.array([np.sin(t), np.sin(2 * t) / 2])

    def line(self, t: float) -> np.ndarray:
        """Point on horizontal line path."""
        x = self.scale * np.cos(t)  # Use cos for smooth back-and-forth
        return self.center + np.array([x, 0])

    def square(self, t: float) -> np.ndarray:
        """Point on square path."""
        t_mod = t % (2 * np.pi)
        segment = int(t_mod / (np.pi / 2)) % 4
        frac = (t_mod % (np.pi / 2)) / (np.pi / 2)

        corners = [
            np.array([1, 1]), np.array([-1, 1]),
            np.array([-1, -1]), np.array([1, -1])
        ]

        start = corners[segment]
        end = corners[(segment + 1) % 4]
        pos = start + frac * (end - start)

        return self.center + self.scale * 0.7 * pos

    def spiral(self, t: float, n_turns: float = 3.0) -> np.ndarray:
        """
        Point on spiral path starting from center.

        Args:
            t: Parameter [0, 2π] mapped to full spiral
            n_turns: Number of spiral rotations
        """
        # t goes from 0 to 2π, map to spiral
        # radius grows linearly from 0 to scale
        progress = t / (2 * np.pi)  # 0 to 1
        radius = self.scale * progress
        angle = n_turns * 2 * np.pi * progress  # multiple rotations

        return self.center + radius * np.array([np.cos(angle), np.sin(angle)])

    def get_point(self, path_type: str, t: float) -> np.ndarray:
        """Get point on path at parameter t."""
        if path_type == "circle":
            return self.circle(t)
        elif path_type == "figure8":
            return self.figure8(t)
        elif path_type == "line":
            return self.line(t)
        elif path_type == "square":
            return self.square(t)
        elif path_type == "spiral":
            return self.spiral(t)
        else:
            return self.circle(t)

    def generate_waypoints(self, path_type: str, n_waypoints: int = 16) -> List[np.ndarray]:
        """Generate a list of waypoints along the path."""
        waypoints = []
        for i in range(n_waypoints):
            t = 2 * np.pi * i / n_waypoints
            waypoints.append(self.get_point(path_type, t))
        return waypoints


class WaypointDemo(ParticleLife):
    """
    Demo showing waypoint following via orientation matrix control.

    Extends ParticleLife to add:
    - Waypoint generation along various path shapes
    - PID control for K_rot adjustment to steer toward waypoints
    - Automatic waypoint advancement when reached
    """

    def __init__(self, config: Config = None):
        # Create default config for waypoint demo
        if config is None:
            config = Config(
                width=1000,
                height=1000,
                n_particles=100,
                n_species=2,
                # Position matrix: two cohesive groups with moderate cross-attraction
                position_matrix=[[0.6, 0.3], [0.3, 0.6]],
                # Orientation matrix: start with opposite signs for translation
                orientation_matrix=[[0.0, -0.3], [0.3, 0.0]],
            )

        # Initialize base simulation
        super().__init__(config, headless=False)

        # Path generator
        self.path_generator = PathGenerator(
            center=np.array([config.width / 2, config.height / 2]),
            scale=250.0
        )

        # Waypoint settings
        self.path_type = "spiral"
        self.n_waypoints = 25
        self.waypoint_threshold = 10.0  # Distance to consider waypoint "reached"
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoints_completed = 0  # Total waypoints reached (for stats)

        # PID controllers (must be initialized before generate_waypoints())
        #
        # Steering PID: controls turn adjustment based on angular error
        # - Input: cross product (sin of angle between current and desired direction)
        #   Range: [-1, 1] where positive means "need to turn left (CCW)"
        # - Output: turn_adjustment, used to create K_rot asymmetry
        #   Target range: [-0.8, 0.8]
        self.steering_pid = ScalarPID(
            kp=0.8,           # Strong proportional response to angular error
            ki=0.05,          # Moderate integral to correct persistent heading errors
            kd=0.3,           # Derivative to dampen steering oscillations
            integral_limit=2.0
        )

        # Speed PID: controls speed factor based on distance to waypoint
        # - Input: distance in pixels (always positive)
        #   Range: [0, ~500] typically
        # - Output: speed_factor multiplier for base_k_rot
        #   Target range: [0.3, 1.5]
        self.speed_pid = ScalarPID(
            kp=0.008,         # Maps ~100px distance to ~0.8 speed factor contribution
            ki=0.0005,        # Slowly increase effort if stuck far away
            kd=0.002,         # Reduce speed when approaching quickly (prevents overshoot)
            integral_limit=50.0
        )

        self.pid_enabled = True
        self.base_k_rot = 0.6  # Base rotation strength

        # Generate initial waypoints
        self.generate_waypoints()

        # UI state
        self.show_path = True
        self.waypoint_color = (200, 200, 200)      # Gray for waypoints
        self.current_wp_color = (255, 100, 100)    # Red for current target
        self.reached_wp_color = (100, 255, 100)    # Green for reached
        self.path_color = (230, 230, 230)          # Light gray for path line

        # Override window title
        pygame.display.set_caption("Waypoint Following Demo - Orientation-Based Locomotion")

        # Initialize particles with two separate groups
        self._initialize_two_groups()

    def generate_waypoints(self):
        """Generate waypoints along the current path type."""
        self.waypoints = self.path_generator.generate_waypoints(
            self.path_type, self.n_waypoints
        )
        self.current_waypoint_idx = 0
        self.steering_pid.reset()
        self.speed_pid.reset()
        print(f"Generated {len(self.waypoints)} waypoints for {self.path_type} path")

    def _initialize_two_groups(self):
        """Initialize particles as two separated groups."""
        center = np.array([self.config.width / 2, self.config.height / 2])
        group_offset = 50
        particles_per_species = self.n // self.n_species

        for i in range(self.n):
            species_id = min(i // particles_per_species, self.n_species - 1)
            self.species[i] = species_id

            if species_id == 0:
                group_center = center + np.array([-group_offset, 0])
            else:
                group_center = center + np.array([group_offset, 0])

            self.positions[i] = group_center + self.rng.uniform(-30, 30, 2)

    def get_current_target(self) -> np.ndarray:
        """Get current waypoint position."""
        if len(self.waypoints) == 0:
            return np.array([self.config.width / 2, self.config.height / 2])
        return self.waypoints[self.current_waypoint_idx]

    def check_waypoint_reached(self) -> bool:
        """Check if current waypoint is reached, advance if so. Returns True if reached."""
        if len(self.waypoints) == 0:
            return False

        centroid = self.get_swarm_centroid()
        target = self.get_current_target()
        distance = np.linalg.norm(target - centroid)

        if distance < self.waypoint_threshold:
            old_idx = self.current_waypoint_idx
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            self.waypoints_completed += 1
            print(f"Waypoint {old_idx + 1} reached! "
                  f"Next: {self.current_waypoint_idx + 1}/{len(self.waypoints)} "
                  f"(Total: {self.waypoints_completed})")
            return True
        return False

    def advance_waypoint(self):
        """Manually advance to next waypoint."""
        self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
        self.steering_pid.reset()
        self.speed_pid.reset()
        print(f"Skipped to waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)}")

    def update_orientation_matrix_pid(self, target_pos: np.ndarray):
        """
        Update orientation matrix using dual PID control.

        Two separate PIDs are used:
        1. Steering PID: Controls turn_adjustment based on angular error
           - Input: cross product of current_direction × desired_direction
           - This equals sin(angle), positive = need to turn left (CCW)
           - Output: turn_adjustment to create K_rot asymmetry

        2. Speed PID: Controls speed_factor based on distance
           - Input: distance to waypoint in pixels
           - Output: speed_factor multiplier for base_k_rot

        KEY PHYSICS:
        K_rot[0,1] and K_rot[1,0] must have OPPOSITE signs for translation!
        - K_rot[0,1] = NEGATIVE (how species 0 rotates due to species 1)
        - K_rot[1,0] = POSITIVE (how species 1 rotates due to species 0)
        Asymmetric magnitudes create turning.
        """
        swarm_centroid = self.get_swarm_centroid()
        error_vec = target_pos - swarm_centroid
        distance = np.linalg.norm(error_vec)

        # Avoid division by zero
        if distance < 1e-6:
            return

        # Desired direction (unit vector toward waypoint)
        desired_direction = error_vec / distance

        # Current velocity direction
        avg_velocity = self.get_average_velocity()
        current_speed = np.linalg.norm(avg_velocity)

        # Use velocity direction if moving, otherwise use direction between species centroids
        if current_speed > 5.0:
            # Moving fast enough - use velocity direction
            current_direction = avg_velocity / current_speed
        else:
            # Moving slowly (rotating) - use inter-group direction as proxy for "facing"
            # The line between the two groups indicates translation direction
            mask0 = self.species == 0
            mask1 = self.species == 1
            centroid0 = self.positions[mask0].mean(axis=0)
            centroid1 = self.positions[mask1].mean(axis=0)
            group_vector = centroid1 - centroid0
            group_dist = np.linalg.norm(group_vector)
            if group_dist > 1e-6:
                # Perpendicular to inter-group line is the "facing" direction
                current_direction = np.array([-group_vector[1], group_vector[0]]) / group_dist
            else:
                current_direction = desired_direction  # Fallback

        # ========== STEERING PID ==========
        # Angular error via cross product: current × desired = sin(angle)
        # Positive cross = target is to the left = need to turn CCW
        # Negative cross = target is to the right = need to turn CW
        angular_error = (current_direction[0] * desired_direction[1] -
                         current_direction[1] * desired_direction[0])

        if True:  # Always run control (removed speed check)

            # PID output for steering
            turn_adjustment = self.steering_pid.compute(angular_error, self.config.dt)
            turn_adjustment = np.clip(turn_adjustment, -0.8, 0.8)

            # ========== SPEED PID ==========
            # Distance error: we want distance to be 0
            # Larger distance → larger PID output → faster movement
            speed_output = self.speed_pid.compute(distance, self.config.dt)
            # Map PID output to reasonable speed factor range
            speed_factor = np.clip(0.3 + speed_output, 0.3, 1.5)

            # ========== DECELERATION ZONE ==========
            # Slow down when approaching waypoint to avoid overshoot
            decel_radius = self.waypoint_threshold * 3  # Start slowing at 3x threshold
            if distance < decel_radius:
                # Linear deceleration: full speed at decel_radius, min speed at threshold
                decel_factor = 0.3 + 0.7 * (distance / decel_radius)
                speed_factor *= decel_factor

            # ========== APPLY TO K_ROT MATRIX ==========
            effective_k_rot = self.base_k_rot * speed_factor

            # Blend between translation and rotation based on angular error
            # - Small error: translation mode (opposite signs, asymmetric for steering)
            # - Large error: rotation mode (same signs, symmetric for true vortex)
            angular_error_mag = abs(angular_error)

            # Blend factor: 0 = pure translation, 1 = pure rotation
            rotation_blend = np.clip(angular_error_mag * 2 - 0.5, 0, 1)

            # Translation component (opposite signs)
            k01_trans = -effective_k_rot * (1 - turn_adjustment * 0.5)
            k10_trans = effective_k_rot * (1 + turn_adjustment * 0.5)

            # Rotation component (same signs, symmetric, direction based on error)
            rotation_dir = np.sign(angular_error)
            k01_rot = effective_k_rot * rotation_dir
            k10_rot = effective_k_rot

            # Blend between modes
            k01 = k01_trans * (1 - rotation_blend) + k01_rot * rotation_blend
            k10 = k10_trans * (1 - rotation_blend) + k10_rot * rotation_blend

            self.alignment_matrix[0, 1] = np.clip(k01, -1.0, 1.0)
            self.alignment_matrix[1, 0] = np.clip(k10, -1.0, 1.0)

    def step(self):
        """Perform one simulation step with waypoint control."""
        if self.paused:
            return

        # Check if we've reached current waypoint
        self.check_waypoint_reached()

        # Get current waypoint as target
        target_pos = self.get_current_target()

        # Update orientation matrix if PID is enabled
        if self.pid_enabled:
            self.update_orientation_matrix_pid(target_pos)

        # Call parent step for physics simulation
        super().step()

    def draw(self):
        """Draw the simulation with waypoint overlay."""
        self.screen.fill((255, 255, 255))

        # Draw path and waypoints
        if self.show_path:
            self.draw_path_and_waypoints()

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
                pygame.draw.line(self.screen, (0, 0, 0), (x, y), (end_x, end_y), max(1, int(self.zoom)))
            else:
                pygame.draw.circle(self.screen, color, (x, y), max(1, int(4 * self.zoom)))

        # Draw swarm centroid
        centroid = self.get_swarm_centroid().astype(int)
        pygame.draw.circle(self.screen, (0, 0, 0), centroid, 8, 2)

        # Draw info panel
        if self.show_info:
            self.draw_info_panel()

    def draw_path_and_waypoints(self):
        """Draw the path line and waypoints."""
        if len(self.waypoints) < 2:
            return

        # Draw path line connecting waypoints
        points = [wp.astype(int).tolist() for wp in self.waypoints]
        pygame.draw.lines(self.screen, self.path_color, True, points, 2)

        # Draw waypoints
        for i, wp in enumerate(self.waypoints):
            pos = wp.astype(int)

            if i == self.current_waypoint_idx:
                # Current target - large red circle
                pygame.draw.circle(self.screen, self.current_wp_color, pos, 12)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, 8)
                pygame.draw.circle(self.screen, self.current_wp_color, pos, 4)
            else:
                # Other waypoints - small gray circles
                pygame.draw.circle(self.screen, self.waypoint_color, pos, 6)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, 4)

            # Draw waypoint number
            text = self.font.render(str(i + 1), True, (150, 150, 150))
            self.screen.blit(text, (pos[0] + 10, pos[1] - 10))

        # Draw line from centroid to current target
        centroid = self.get_swarm_centroid().astype(int)
        target = self.get_current_target().astype(int)
        pygame.draw.line(self.screen, (255, 200, 200), tuple(centroid), tuple(target), 1)

    def draw_info_panel(self):
        """Draw information panel."""
        target = self.get_current_target()
        centroid = self.get_swarm_centroid()
        distance = np.linalg.norm(target - centroid)

        info_lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Path: {self.path_type.upper()}",
            f"Waypoint: {self.current_waypoint_idx + 1}/{len(self.waypoints)}",
            f"Total Reached: {self.waypoints_completed}",
            f"Distance to WP: {distance:.0f} px",
            f"Threshold: {self.waypoint_threshold:.0f} px",
            f"PID: {'ON' if self.pid_enabled else 'OFF'}",
            f"K_rot: [{self.alignment_matrix[0,1]:.2f}, {self.alignment_matrix[1,0]:.2f}]",
            "",
            f"Steering PID: {self.steering_pid.last_output:.3f}",
            f"  (I={self.steering_pid.integral:.3f})",
            f"Speed PID: {self.speed_pid.last_output:.3f}",
            f"  (I={self.speed_pid.integral:.1f})",
            "",
            "Controls:",
            "1-5: Circle/Figure-8/Line/Square/Spiral",
            "[/]: Fewer/More waypoints",
            "+/-: Smaller/Larger threshold",
            "N: Skip to next waypoint",
            "P: Toggle PID",
            "T: Toggle path display",
            "SPACE: Pause  R: Reset",
            "Q/ESC: Quit",
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

    def draw_matrix_viz(self):
        """Draw small visualization of K_rot matrix."""
        x_start = self.config.width - 150
        y_start = 10
        cell_size = 40

        label = self.font.render("K_rot Matrix:", True, (100, 100, 100))
        self.screen.blit(label, (x_start, y_start))
        y_start += 25

        for i in range(2):
            for j in range(2):
                x = x_start + j * cell_size
                y = y_start + i * cell_size
                value = self.alignment_matrix[i, j]

                if value > 0:
                    intensity = int(min(255, value * 255))
                    color = (0, intensity, 0)
                elif value < 0:
                    intensity = int(min(255, abs(value) * 255))
                    color = (intensity, 0, 0)
                else:
                    color = (128, 128, 128)

                pygame.draw.rect(self.screen, (200, 200, 200), (x, y, cell_size, cell_size), 1)
                text = self.font.render(f"{value:.2f}", True, color)
                text_rect = text.get_rect(center=(x + cell_size//2, y + cell_size//2))
                self.screen.blit(text, text_rect)

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
                    self.initialize_particles()
                    self._initialize_two_groups()
                    self.current_waypoint_idx = 0
                    self.waypoints_completed = 0
                    self.steering_pid.reset()
                    self.speed_pid.reset()
                    print("Reset particles and waypoints")

                elif event.key == pygame.K_p:
                    self.pid_enabled = not self.pid_enabled
                    self.steering_pid.reset()
                    self.speed_pid.reset()
                    if not self.pid_enabled:
                        self.alignment_matrix[0, 1] = -self.base_k_rot
                        self.alignment_matrix[1, 0] = self.base_k_rot
                    print(f"PID control: {'ON' if self.pid_enabled else 'OFF'}")

                elif event.key == pygame.K_t:
                    self.show_path = not self.show_path

                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info

                elif event.key == pygame.K_o:
                    self.show_orientations = not self.show_orientations

                elif event.key == pygame.K_n:
                    self.advance_waypoint()

                # Path shape selection
                elif event.key == pygame.K_1:
                    self.path_type = "circle"
                    self.generate_waypoints()

                elif event.key == pygame.K_2:
                    self.path_type = "figure8"
                    self.generate_waypoints()

                elif event.key == pygame.K_3:
                    self.path_type = "line"
                    self.generate_waypoints()

                elif event.key == pygame.K_4:
                    self.path_type = "square"
                    self.generate_waypoints()

                elif event.key == pygame.K_5:
                    self.path_type = "spiral"
                    self.n_waypoints = 24  # Dense waypoints for spiral
                    self.generate_waypoints()

                # Adjust number of waypoints
                elif event.key == pygame.K_LEFTBRACKET:
                    self.n_waypoints = max(4, self.n_waypoints - 2)
                    self.generate_waypoints()

                elif event.key == pygame.K_RIGHTBRACKET:
                    self.n_waypoints = min(32, self.n_waypoints + 2)
                    self.generate_waypoints()

                # Adjust waypoint threshold
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.waypoint_threshold = min(150, self.waypoint_threshold + 10)
                    print(f"Waypoint threshold: {self.waypoint_threshold:.0f} px")

                elif event.key == pygame.K_MINUS:
                    self.waypoint_threshold = max(20, self.waypoint_threshold - 10)
                    print(f"Waypoint threshold: {self.waypoint_threshold:.0f} px")

        return True

    def run(self):
        """Main simulation loop."""
        running = True

        print("=" * 60)
        print("Waypoint Following Demo")
        print("=" * 60)
        print("Mechanism: Orientation matrix with OPPOSITE signs creates")
        print("           net translation. Dual PID control:")
        print("           - Steering PID: adjusts K_rot asymmetry for turning")
        print("           - Speed PID: adjusts K_rot magnitude based on distance")
        print("=" * 60)

        while running:
            running = self.handle_events()
            self.step()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


# Keep the old class name as alias for backwards compatibility
TrajectoryDemo = WaypointDemo


def main():
    demo = WaypointDemo()
    demo.run()


if __name__ == "__main__":
    main()
