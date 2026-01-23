"""
Control Barrier Function (CBF) based obstacle avoidance for swarm control.

This module implements CBF-QP (Quadratic Programming) safety filtering
to ensure the swarm avoids circular obstacles while following waypoints.

Mathematical formulation:
- Barrier function: h(x) = ||p - c||² - r²  (positive = safe)
- CBF constraint: ḣ + α·h ≥ 0
- QP: minimize ||u - u_nominal||² subject to CBF constraints
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy.optimize import minimize, Bounds


@dataclass
class CircularObstacle:
    """
    Circular obstacle with safety margin.

    Attributes:
        center: [x, y] position of obstacle center
        radius: obstacle radius in pixels
        safety_margin: additional buffer distance (δ)
    """
    center: np.ndarray
    radius: float
    safety_margin: float = 20.0

    def __post_init__(self):
        self.center = np.array(self.center, dtype=float)

    @property
    def effective_radius(self) -> float:
        """Radius including safety margin: r + δ"""
        return self.radius + self.safety_margin

    def barrier_value(self, position: np.ndarray) -> float:
        """
        Compute barrier function h(x) for this obstacle.

        h(x) = ||p - c||² - (r + δ)²

        Returns:
            h > 0: safe (outside obstacle + margin)
            h ≤ 0: collision/violation
        """
        diff = position - self.center
        return float(np.sum(diff**2) - self.effective_radius**2)

    def barrier_gradient(self, position: np.ndarray) -> np.ndarray:
        """
        Compute gradient of barrier function: ∇h = 2(p - c)
        """
        return 2.0 * (position - self.center)

    def distance_to_surface(self, position: np.ndarray) -> float:
        """Distance from position to obstacle surface (negative if inside)."""
        return float(np.linalg.norm(position - self.center) - self.effective_radius)


@dataclass
class CBFConfig:
    """
    Configuration for CBF-QP controller.

    Attributes:
        alpha: CBF aggressiveness parameter (higher = stronger safety)
        b_long: longitudinal control effectiveness
        b_lat: lateral control effectiveness
        activation_distance: CBF only active within this distance
        tangent_blend_distance: range for tangent navigation blending
        tangent_gain: strength of tangential deflection [0, 1]
    """
    # CBF parameters
    alpha: float = 1.0

    # Control effectiveness (maps control to velocity change)
    b_long: float = 50.0   # longitudinal gain
    b_lat: float = 30.0    # lateral gain

    # Activation thresholds
    activation_distance: float = 150.0
    tangent_blend_distance: float = 200.0

    # Tangent navigation
    tangent_gain: float = 0.8

    # Control bounds
    u_long_min: float = 0.0
    u_long_max: float = 1.5
    u_lat_min: float = -0.8
    u_lat_max: float = 0.8

    # Solver settings
    max_iterations: int = 100


class ObstacleManager:
    """Manages multiple obstacles for the simulation."""

    def __init__(self):
        self.obstacles: List[CircularObstacle] = []

    def add_obstacle(self, center: np.ndarray, radius: float,
                     safety_margin: float = 20.0) -> int:
        """
        Add obstacle and return its index.

        Args:
            center: [x, y] position
            radius: obstacle radius
            safety_margin: additional buffer

        Returns:
            Index of the new obstacle
        """
        obs = CircularObstacle(
            center=np.array(center, dtype=float),
            radius=float(radius),
            safety_margin=float(safety_margin)
        )
        self.obstacles.append(obs)
        return len(self.obstacles) - 1

    def remove_obstacle(self, index: int) -> bool:
        """Remove obstacle by index. Returns True if successful."""
        if 0 <= index < len(self.obstacles):
            del self.obstacles[index]
            return True
        return False

    def remove_nearest(self, position: np.ndarray) -> bool:
        """Remove the obstacle nearest to the given position."""
        if not self.obstacles:
            return False

        min_dist = float('inf')
        min_idx = -1
        for i, obs in enumerate(self.obstacles):
            dist = np.linalg.norm(position - obs.center)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        if min_idx >= 0:
            del self.obstacles[min_idx]
            return True
        return False

    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles.clear()

    def get_active_obstacles(self, position: np.ndarray,
                             activation_distance: float) -> List[CircularObstacle]:
        """
        Get obstacles within activation distance of position.

        Args:
            position: current swarm centroid
            activation_distance: CBF activation range

        Returns:
            List of obstacles that could affect the swarm
        """
        active = []
        for obs in self.obstacles:
            dist = np.linalg.norm(position - obs.center)
            if dist < activation_distance + obs.effective_radius:
                active.append(obs)
        return active

    def check_collision(self, position: np.ndarray) -> bool:
        """Check if position is inside any obstacle (including margin)."""
        for obs in self.obstacles:
            if obs.barrier_value(position) <= 0:
                return True
        return False

    def get_min_barrier_value(self, position: np.ndarray) -> Tuple[float, int]:
        """
        Get minimum barrier value and corresponding obstacle index.

        Returns:
            (min_h, obstacle_index) or (inf, -1) if no obstacles
        """
        min_h = float('inf')
        min_idx = -1
        for i, obs in enumerate(self.obstacles):
            h = obs.barrier_value(position)
            if h < min_h:
                min_h = h
                min_idx = i
        return min_h, min_idx


def path_intersects_obstacle(p_start: np.ndarray, p_end: np.ndarray,
                             obstacle: CircularObstacle) -> bool:
    """
    Check if line segment from p_start to p_end intersects obstacle.

    Uses quadratic formula to find intersection with circle.
    """
    d = p_end - p_start
    f = p_start - obstacle.center
    r = obstacle.effective_radius

    a = np.dot(d, d)
    if a < 1e-10:
        return False  # Zero-length segment

    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - r**2

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return False  # No intersection

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)

    # Check if intersection is on the segment [0, 1]
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)


def compute_tangent_points(p: np.ndarray,
                          obstacle: CircularObstacle) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute tangent points from position p to obstacle circle.

    Returns:
        (t1, t2): Two tangent points (left and right)
    """
    c = obstacle.center
    r = obstacle.effective_radius

    pc = c - p
    d = np.linalg.norm(pc)

    if d <= r:
        # Inside obstacle - return perpendicular escape directions
        if d < 1e-6:
            return c + np.array([r, 0]), c + np.array([-r, 0])
        perp = np.array([-pc[1], pc[0]]) / d
        return c + r * perp, c - r * perp

    # Angle from center line to tangent
    theta = np.arcsin(min(1.0, r / d))

    # Angle of center line
    phi = np.arctan2(pc[1], pc[0])

    # Tangent point angles (perpendicular to tangent line at circle)
    angle1 = phi + np.pi/2 - theta
    angle2 = phi - np.pi/2 + theta

    t1 = c + r * np.array([np.cos(angle1), np.sin(angle1)])
    t2 = c + r * np.array([np.cos(angle2), np.sin(angle2)])

    return t1, t2


def blend_direction_with_tangent(p: np.ndarray, target: np.ndarray,
                                 obstacles: List[CircularObstacle],
                                 config: CBFConfig) -> np.ndarray:
    """
    Blend desired direction with tangent when path is blocked.

    Args:
        p: current swarm position
        target: target waypoint position
        obstacles: list of obstacles to consider
        config: CBF configuration

    Returns:
        Blended direction unit vector
    """
    desired = target - p
    desired_dist = np.linalg.norm(desired)
    if desired_dist < 1e-6:
        return np.array([1.0, 0.0])
    desired_dir = desired / desired_dist

    # Find blocking obstacles
    blocking = []
    for obs in obstacles:
        if path_intersects_obstacle(p, target, obs):
            dist_to_obs = obs.distance_to_surface(p)
            if dist_to_obs < config.tangent_blend_distance:
                blocking.append((obs, dist_to_obs))

    if not blocking:
        return desired_dir

    # Use nearest blocking obstacle for tangent
    blocking.sort(key=lambda x: x[1])
    nearest_obs = blocking[0][0]
    dist_to_obs = max(0.1, blocking[0][1])  # Avoid division by zero

    # Compute tangent points
    t1, t2 = compute_tangent_points(p, nearest_obs)

    # Choose tangent point that leads closer to target
    # (after going around the obstacle)
    d1 = np.linalg.norm(target - t1)
    d2 = np.linalg.norm(target - t2)
    tangent_point = t1 if d1 < d2 else t2

    # Direction to tangent point
    tangent_vec = tangent_point - p
    tangent_dist = np.linalg.norm(tangent_vec)
    if tangent_dist < 1e-6:
        return desired_dir
    tangent_dir = tangent_vec / tangent_dist

    # Blend based on distance (closer to obstacle = more tangent)
    blend_factor = 1.0 - min(1.0, dist_to_obs / config.tangent_blend_distance)
    blend_factor *= config.tangent_gain

    blended = (1 - blend_factor) * desired_dir + blend_factor * tangent_dir
    norm = np.linalg.norm(blended)
    if norm < 1e-6:
        return desired_dir
    return blended / norm


class CBFController:
    """
    Control Barrier Function based QP safety filter.

    Filters nominal control inputs to ensure safety constraints
    (obstacle avoidance) are satisfied.
    """

    def __init__(self, config: CBFConfig):
        self.config = config
        self._last_solve_success = True
        self._last_barrier_values: dict = {}
        self._last_u_nominal = np.zeros(2)
        self._last_u_safe = np.zeros(2)

    def _rotation_matrix(self, theta: float) -> np.ndarray:
        """2D rotation matrix for angle theta."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def _control_effectiveness_matrix(self, heading: float) -> np.ndarray:
        """
        Compute control effectiveness matrix in world frame.

        B_world = R(θ) @ diag([b_long, b_lat]) @ R(θ)^T

        This maps control inputs [u_long, u_lat] to velocity change.
        """
        R = self._rotation_matrix(heading)
        B_body = np.diag([self.config.b_long, self.config.b_lat])
        return R @ B_body @ R.T

    def build_cbf_constraints(self,
                              pos: np.ndarray,
                              vel: np.ndarray,
                              heading: float,
                              obstacles: List[CircularObstacle],
                              dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build linear constraint matrices for CBF-QP.

        The CBF constraint ḣ + α·h ≥ 0 becomes:
            A_i @ u ≥ b_i

        Args:
            pos: swarm centroid position
            vel: swarm centroid velocity
            heading: current heading angle
            obstacles: active obstacles
            dt: time step

        Returns:
            A: constraint matrix (M x 2)
            b: constraint RHS (M,)
        """
        if not obstacles:
            return np.empty((0, 2)), np.empty(0)

        B_world = self._control_effectiveness_matrix(heading)

        A_rows = []
        b_vals = []

        self._last_barrier_values.clear()

        for obs in obstacles:
            # Barrier value: h = ||p-c||² - r²
            h = obs.barrier_value(pos)

            # Barrier gradient: ∇h = 2(p - c)
            grad_h = obs.barrier_gradient(pos)

            # Current barrier derivative: ḣ = ∇h · v
            h_dot = np.dot(grad_h, vel)

            # Linearized constraint:
            # ḣ(u) ≈ ḣ + ∇h · (dt · B_world · u)
            # Constraint: A_i @ u ≥ b_i
            # A_i = 2·dt·(p-c)^T @ B_world
            A_i = 2.0 * dt * grad_h @ B_world

            # b_i = -ḣ - α·h
            b_i = -h_dot - self.config.alpha * h

            A_rows.append(A_i)
            b_vals.append(b_i)

            # Store for debugging
            self._last_barrier_values[id(obs)] = h

        return np.array(A_rows), np.array(b_vals)

    def solve_qp(self,
                 u_nominal: np.ndarray,
                 A: np.ndarray,
                 b: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Solve CBF-QP using scipy.optimize.minimize with SLSQP.

        minimize ||u - u_nominal||²
        s.t.     A @ u ≥ b
                 u_min ≤ u ≤ u_max

        Args:
            u_nominal: nominal control from PID
            A: constraint matrix
            b: constraint RHS

        Returns:
            (u_safe, success): safe control and solve status
        """
        # Objective: ||u - u_nom||²
        def objective(u):
            diff = u - u_nominal
            return np.sum(diff**2)

        def gradient(u):
            return 2.0 * (u - u_nominal)

        # Bounds
        bounds = Bounds(
            [self.config.u_long_min, self.config.u_lat_min],
            [self.config.u_long_max, self.config.u_lat_max]
        )

        # Constraints: A @ u ≥ b => A @ u - b ≥ 0
        constraints = []
        if len(A) > 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda u, A=A, b=b: A @ u - b,
                'jac': lambda u, A=A: A
            })

        # Solve
        result = minimize(
            objective,
            x0=np.clip(u_nominal,
                      [self.config.u_long_min, self.config.u_lat_min],
                      [self.config.u_long_max, self.config.u_lat_max]),
            jac=gradient,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations, 'ftol': 1e-6}
        )

        return result.x, result.success

    def filter_control(self,
                       pos: np.ndarray,
                       vel: np.ndarray,
                       heading: float,
                       u_nominal: np.ndarray,
                       obstacles: List[CircularObstacle],
                       dt: float) -> np.ndarray:
        """
        Main CBF-QP filter: returns safe control input.

        Args:
            pos: swarm centroid position [x, y]
            vel: swarm centroid velocity [vx, vy]
            heading: current heading angle (radians)
            u_nominal: nominal control from PID [u_long, u_lat]
            obstacles: list of active obstacles
            dt: time step

        Returns:
            u_safe: safe control input [u_long, u_lat]
        """
        self._last_u_nominal = u_nominal.copy()

        if not obstacles:
            self._last_solve_success = True
            self._last_u_safe = u_nominal.copy()
            return u_nominal

        # Build constraints
        A, b = self.build_cbf_constraints(pos, vel, heading, obstacles, dt)

        # Solve QP
        u_safe, success = self.solve_qp(u_nominal, A, b)

        self._last_solve_success = success
        self._last_u_safe = u_safe.copy()

        if not success:
            # Emergency: reduce speed significantly
            print("CBF-QP infeasible! Emergency braking.")
            return np.array([0.1, 0.0])

        return u_safe

    @property
    def last_solve_success(self) -> bool:
        """Whether the last QP solve was successful."""
        return self._last_solve_success

    @property
    def last_barrier_values(self) -> dict:
        """Dictionary of barrier values from last solve."""
        return self._last_barrier_values

    @property
    def last_u_nominal(self) -> np.ndarray:
        """Last nominal control input."""
        return self._last_u_nominal

    @property
    def last_u_safe(self) -> np.ndarray:
        """Last safe (filtered) control input."""
        return self._last_u_safe
