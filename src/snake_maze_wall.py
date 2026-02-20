#!/usr/bin/env python3
"""
Snake Maze Demo (Wall Particles) — Navigate a snake through maze environments

Same as snake_maze but walls are represented as static repulsive particles,
creating a smooth force field that pushes the snake away from walls.

Controls:
    ←/→:   Steer head left/right
    1-6:   Select maze layout (6=Random)
    G:     Generate new random maze
    R:     Reset positions
    SPACE: Pause/Resume
    M:     Toggle goal marker
    W:     Toggle wall particle visibility
    H:     Hide/show all GUI
    I:     Toggle info panel
    Q/ESC: Quit
"""

import heapq
from math import ceil

import pygame
import numpy as np
from snake_demo import SnakeDemo, generate_position_matrix
from particle_life import Config, ParticleLife
from snake_maze import (
    Wall, MAZE_LAYOUTS,
    create_maze_open, create_maze_slalom, create_maze_spiral,  # noqa: F401 — used via MAZE_LAYOUTS
    create_maze_rooms, create_maze_labyrinth, create_maze_random,  # noqa: F401
)


# =============================================================================
# Snake Maze Demo with Wall Particles
# =============================================================================

class SnakeMazeWallDemo(SnakeDemo):
    """
    Snake demo with maze environment using wall particles for collision avoidance.

    Walls are represented as static particles of a special "wall species" that
    repels all snake species via the K_pos interaction matrix. This creates a
    smooth force field around walls instead of hard collision boundaries.
    """

    # Wall particle parameters
    WALL_SPACING = 0.1          # Distance between wall particles (meters)
    WALL_REPULSION = 0.0       # K_pos value for wall→snake repulsion (gentle push)
    WALL_COLOR = (90, 100, 120) # Color for wall particles when drawn

    def __init__(self, snake_n_species: int = 6, n_particles: int = 15, maze_id: int = 1):
        # Track snake vs wall species separately
        self.snake_n_species = snake_n_species
        self.wall_species_idx = snake_n_species  # Wall is the last species
        self.show_wall_particles = True

        # Maze state (set before super().__init__ since _initialize_chain is called there)
        self.walls = []
        self.current_maze_id = maze_id
        self.wall_margin = 0.08

        # Start and goal positions
        self.start_position = np.array([1.0, 1.0])
        self.goal_position = np.array([9.0, 9.0])
        self.goal_radius = 0.6
        self.show_goal = True
        self.goal_reached = False

        # Total species = snake species + 1 wall species
        total_species = snake_n_species + 1

        # Build K_pos: snake chain + wall repulsion
        snake_k_pos = generate_position_matrix(snake_n_species)
        k_pos = np.zeros((total_species, total_species))
        k_pos[:snake_n_species, :snake_n_species] = snake_k_pos
        # All snake species are repelled by wall species
        for i in range(snake_n_species):
            k_pos[i, self.wall_species_idx] = self.WALL_REPULSION
        # Wall doesn't respond to anything
        k_pos[self.wall_species_idx, :] = 0.0

        # Build K_rot: snake chain only, wall species has no rotation coupling
        k_rot = np.zeros((total_species, total_species))

        # Create config with total species count
        config = Config(
            n_species=total_species,
            n_particles=n_particles,
            a_rot=3.0,
            position_matrix=k_pos.tolist(),
            orientation_matrix=k_rot.tolist(),
        )

        # Call ParticleLife.__init__ directly (skip SnakeDemo.__init__ which
        # would generate wrong matrices for snake_n_species)
        ParticleLife.__init__(self, config, headless=False)

        # --- Replicate SnakeDemo init state ---
        self.turn_input = 0.0
        self.base_k_rot = 0.05
        self.forward_speed = 0.1
        self.turn_decay = 0.92

        self.joint_delay = 8
        n_joints = snake_n_species - 1
        history_len = self.joint_delay * n_joints + 1
        self.turn_history = np.zeros(history_len)
        self.history_idx = 0

        self.group_spacing = 0.8
        self.matrix_edit_mode = False
        self.edit_row = 0
        self.edit_col = 0
        self.editing_k_rot = True
        self.hide_gui = False

        # --- Maze-specific init ---

        # Load maze and place wall particles
        self._load_maze_and_rebuild(maze_id)

        # Override window title
        pygame.display.set_caption("Snake Maze (Wall Particles) — Navigate to Goal")

        # --- Autopilot (A* pathfinding) ---
        self.autopilot_active = False
        self.autopilot_waypoints = []
        self.autopilot_wp_idx = 0
        self.autopilot_wp_threshold = 0.2

        # Pathfinding grid
        self.cell_size = 0.2
        self.pathfinding_margin = 0.35
        self.occupancy_grid = None
        self.grid_rows = 0
        self.grid_cols = 0
        self.build_occupancy_grid()

        print("=" * 60)
        print("Snake Maze Demo (Wall Particles)")
        print("=" * 60)
        print(f"Maze: {MAZE_LAYOUTS[maze_id][0]}")
        print(f"Snake species: {snake_n_species}, Wall species: 1")
        print("")
        print("Controls:")
        print("  ←/→     Steer left/right")
        print("  ↑/↓     Increase/decrease forward speed")
        print("  1-6     Select maze layout (6=Random)")
        print("  G       Generate new random maze")
        print("  A       Toggle autopilot (A* pathfinding)")
        print("  W       Toggle wall particle visibility")
        print("  R       Reset positions")
        print("  M       Toggle goal marker")
        print("  SPACE   Pause")
        print("  Q/ESC   Quit")
        print("=" * 60)

    # ================================================================
    # Wall particle placement
    # ================================================================

    def _sample_wall_positions(self, wall: Wall) -> np.ndarray:
        """Sample particle positions along a wall's surface.

        For thin walls (one dimension < 2*spacing), place along centerline.
        For thick walls, fill with a grid.
        """
        spacing = self.WALL_SPACING
        positions = []

        # Determine if wall is thin in either dimension
        n_cols = max(1, int(wall.width / spacing))
        n_rows = max(1, int(wall.height / spacing))

        for r in range(n_rows):
            for c in range(n_cols):
                px = wall.x + (c + 0.5) * wall.width / n_cols
                py = wall.y + (r + 0.5) * wall.height / n_rows
                positions.append([px, py])

        if not positions:
            # Fallback: single particle at center
            positions.append([wall.x + wall.width / 2, wall.y + wall.height / 2])

        return np.array(positions)

    def _compute_all_wall_positions(self) -> np.ndarray:
        """Compute wall particle positions for all walls in the maze."""
        all_positions = []
        for wall in self.walls:
            wp = self._sample_wall_positions(wall)
            all_positions.append(wp)

        if all_positions:
            return np.vstack(all_positions)
        return np.zeros((0, 2))

    def _load_maze_and_rebuild(self, maze_id: int):
        """Load maze, compute wall particles, and rebuild all arrays."""
        if maze_id not in MAZE_LAYOUTS:
            maze_id = 1

        self.current_maze_id = maze_id
        name, generator = MAZE_LAYOUTS[maze_id]
        self.walls = generator(self.config.sim_width, self.config.sim_height)

        # Add boundary walls only where the maze doesn't already cover the edge
        w, h = self.config.sim_width, self.config.sim_height
        t = 0.15
        edge_coverage = [0.0, 0.0, 0.0, 0.0]  # bottom, top, left, right
        for wall in self.walls:
            # bottom edge: wall near y=0, horizontal
            if wall.y <= t and wall.height <= t * 2:
                edge_coverage[0] += wall.width
            # top edge: wall near y=h, horizontal
            if wall.y + wall.height >= h - t and wall.height <= t * 2:
                edge_coverage[1] += wall.width
            # left edge: wall near x=0, vertical
            if wall.x <= t and wall.width <= t * 2:
                edge_coverage[2] += wall.height
            # right edge: wall near x=w, vertical
            if wall.x + wall.width >= w - t and wall.width <= t * 2:
                edge_coverage[3] += wall.height
        # Add boundary if less than 50% of that edge is covered
        if edge_coverage[0] < w * 0.5:
            self.walls.append(Wall(0, 0, w, t))
        if edge_coverage[1] < w * 0.5:
            self.walls.append(Wall(0, h - t, w, t))
        if edge_coverage[2] < h * 0.5:
            self.walls.append(Wall(0, 0, t, h))
        if edge_coverage[3] < h * 0.5:
            self.walls.append(Wall(w - t, 0, t, h))
        self.start_position = np.array([1.0, 1.0])
        self.goal_position = np.array([w - 1.0, h - 1.0])

        # Compute wall particle positions
        wall_positions = self._compute_all_wall_positions()
        self.n_wall_particles = len(wall_positions)
        self.snake_count = self.snake_n_species * self.config.n_particles

        # Total particle count
        total_n = self.snake_count + self.n_wall_particles
        self.n = total_n

        # Rebuild arrays
        self.positions = np.zeros((total_n, 2))
        self.velocities = np.zeros((total_n, 2))
        self.species = np.zeros(total_n, dtype=int)

        # Place snake particles at start
        self._initialize_at_start()

        # Place wall particles
        if self.n_wall_particles > 0:
            start_idx = self.snake_count
            self.positions[start_idx:] = wall_positions
            self.velocities[start_idx:] = 0.0
            self.species[start_idx:] = self.wall_species_idx

        self.goal_reached = False
        print(f"Loaded maze: {name} ({len(self.walls)} walls, "
              f"{self.n_wall_particles} wall particles)")

    def _initialize_at_start(self):
        """Initialize snake particles at maze start position."""
        start_x = self.start_position[0]
        start_y = self.start_position[1]

        particles_per_species = self.config.n_particles

        for i in range(self.snake_count):
            species_id = min(i // particles_per_species, self.snake_n_species - 1)
            self.species[i] = species_id

            group_x = start_x + species_id * self.group_spacing * 0.4
            group_y = start_y

            self.positions[i, 0] = group_x + self.rng.uniform(-0.08, 0.08)
            self.positions[i, 1] = group_y + self.rng.uniform(-0.08, 0.08)
            self.velocities[i] = np.array([0.0, 0.0])

        self.goal_reached = False

    def get_species_centroids(self):
        """Return centroids for snake species only (exclude wall species)."""
        centroids = []
        for s in range(self.snake_n_species):
            mask = self.species[:self.snake_count] == s
            if mask.any():
                centroids.append(self.positions[:self.snake_count][mask].mean(axis=0))
            else:
                centroids.append(np.array([self.config.sim_width / 2,
                                           self.config.sim_height / 2]))
        return centroids

    # ================================================================
    # Override matrix updates to scope to snake species only
    # ================================================================

    def update_matrices_from_input(self):
        """Update K_rot for snake joints only (wall species stays 0)."""
        if self.matrix_edit_mode:
            return

        # Record current turn input into history
        self.turn_history[self.history_idx] = self.turn_input
        self.history_idx = (self.history_idx + 1) % len(self.turn_history)

        # Build K_rot with per-joint delayed signals (snake joints only)
        n_joints = self.snake_n_species - 1

        for joint in range(n_joints):
            delay = joint * self.joint_delay
            idx = (self.history_idx - 1 - delay) % len(self.turn_history)
            delayed_input = self.turn_history[idx]

            strength = np.clip(-self.base_k_rot * delayed_input, -1.0, 1.0)
            self.alignment_matrix[joint, joint + 1] = strength
            self.alignment_matrix[joint + 1, joint] = strength

        # Ensure wall species row/col stays 0
        self.alignment_matrix[self.wall_species_idx, :] = 0.0
        self.alignment_matrix[:, self.wall_species_idx] = 0.0

    def update_forward_speed(self, delta: float):
        """Adjust forward speed — only update snake portion of K_pos."""
        self.forward_speed = np.clip(self.forward_speed + delta, -0.3, 0.5)

        new_snake_matrix = generate_position_matrix(
            self.snake_n_species,
            forward_bias=self.forward_speed
        )
        # Only update the snake sub-matrix, preserve wall entries
        self.matrix[:self.snake_n_species, :self.snake_n_species] = new_snake_matrix

    # ================================================================
    # Override compute_velocities for performance
    # ================================================================

    def compute_velocities(self) -> np.ndarray:
        """Compute velocities — only for snake particles, using all as sources.

        Wall particles always have zero velocity. We skip computing their
        velocities for performance (avoids O(wall_count * total) extra work).
        """
        new_velocities = np.zeros_like(self.velocities)

        r_max = self.config.r_max
        beta = self.config.beta
        inv_1_minus_beta = 1.0 / (1.0 - beta) if beta < 1.0 else 1.0
        force_scale = self.config.force_scale
        far_attraction = self.config.far_attraction

        # Only compute for snake particles (indices 0..snake_count-1)
        for i in range(self.snake_count):
            delta = self.positions - self.positions[i]
            dist = np.linalg.norm(delta, axis=1)
            dist[i] = np.inf

            velocity_sum = np.zeros(2)

            for j in range(self.n):
                if j == i:
                    continue
                r = dist[j]
                r_norm = r / r_max

                dx, dy = delta[j]

                if r_norm >= 1.0:
                    if far_attraction > 0:
                        si = self.species[i]; sj = self.species[j]
                        k_pos = self.matrix[si, sj]
                        inv_r = 1.0 / (r + 1e-8)
                        r_hat_x, r_hat_y = dx * inv_r, dy * inv_r
                        F = k_pos * far_attraction
                        velocity_sum[0] += force_scale * F * r_hat_x
                        velocity_sum[1] += force_scale * F * r_hat_y
                    continue

                si = self.species[i]; sj = self.species[j]
                k_pos = self.matrix[si, sj]
                k_rot = self.alignment_matrix[si, sj]

                inv_r = 1.0 / (r + 1e-8)
                r_hat_x, r_hat_y = dx * inv_r, dy * inv_r
                t_hat_x, t_hat_y = -r_hat_y, r_hat_x

                # Piecewise linear radial force (4 zones)
                if r_norm < beta:
                    F = r_norm / beta - 1.0
                else:
                    triangle = 1.0 - abs(2.0 * r_norm - 1.0 - beta) * inv_1_minus_beta
                    peak_r = 0.5 * (1.0 + beta)
                    if r_norm < peak_r:
                        F = k_pos * triangle
                    else:
                        F = k_pos * max(far_attraction, triangle)

                velocity_sum[0] += force_scale * F * r_hat_x
                velocity_sum[1] += force_scale * F * r_hat_y

                # Tangential swirl
                swirl_weight = np.clip(1.0 - r_norm, 0.0, 1.0)
                swirl_gain = k_rot * self.config.a_rot * swirl_weight
                velocity_sum[0] += swirl_gain * t_hat_x
                velocity_sum[1] += swirl_gain * t_hat_y

            new_velocities[i] = velocity_sum

        # Wall particles: velocity stays zero (already initialized to 0)
        return new_velocities

    # ================================================================
    # Override step to keep wall particles static
    # ================================================================

    def step(self):
        """Perform one simulation step — wall particles stay fixed."""
        if self.paused:
            return

        # Apply input decay
        self.turn_input *= self.turn_decay
        if abs(self.turn_input) < 0.01:
            self.turn_input = 0.0

        # Autopilot overrides turn_input
        if self.autopilot_active:
            self.update_autopilot()

        # Update K_rot from input
        self.update_matrices_from_input()

        # --- Physics step (inline, only for snake particles) ---
        self.velocities = self.compute_velocities()

        # Clamp snake particle speed
        snake_v = self.velocities[:self.snake_count]
        speed = np.linalg.norm(snake_v, axis=1, keepdims=True)
        self.velocities[:self.snake_count] = np.where(
            speed > self.config.max_speed,
            snake_v * self.config.max_speed / speed,
            snake_v
        )

        # Update snake positions only
        self.positions[:self.snake_count] += (
            self.velocities[:self.snake_count] * self.config.dt
        )

        # Boundary conditions for snake particles
        margin = 0.05
        for dim in range(2):
            mask = self.positions[:self.snake_count, dim] < margin
            self.positions[:self.snake_count][mask, dim] = margin
            self.velocities[:self.snake_count][mask, dim] = abs(
                self.velocities[:self.snake_count][mask, dim])

            limit = (self.config.sim_width - margin if dim == 0
                     else self.config.sim_height - margin)
            mask = self.positions[:self.snake_count, dim] > limit
            self.positions[:self.snake_count][mask, dim] = limit
            self.velocities[:self.snake_count][mask, dim] = -abs(
                self.velocities[:self.snake_count][mask, dim])

        # Wall particles stay fixed (velocities already 0, positions unchanged)

        # Check goal
        self.check_goal()

        # Disengage autopilot when goal reached
        if self.goal_reached and self.autopilot_active:
            self.autopilot_active = False
            self.turn_input = 0.0
            print("Autopilot: Goal reached, disengaging.")

    # ================================================================
    # Goal checking
    # ================================================================

    def check_goal(self):
        """Check if snake head has reached the goal."""
        if not self.show_goal:
            return

        head_mask = self.species[:self.snake_count] == 0
        if not head_mask.any():
            return

        head_centroid = self.positions[:self.snake_count][head_mask].mean(axis=0)
        dist_to_goal = np.linalg.norm(head_centroid - self.goal_position)

        if dist_to_goal < self.goal_radius and not self.goal_reached:
            self.goal_reached = True
            print("Goal reached!")

    # ================================================================
    # A* Autopilot (reused from snake_maze)
    # ================================================================

    def build_occupancy_grid(self):
        """Build a boolean grid where True = blocked (wall or boundary)."""
        sw = self.config.sim_width
        sh = self.config.sim_height
        self.grid_cols = ceil(sw / self.cell_size)
        self.grid_rows = ceil(sh / self.cell_size)
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)

        boundary = 0.1
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cx = (c + 0.5) * self.cell_size
                cy = (r + 0.5) * self.cell_size
                if (cx < boundary or cx > sw - boundary or
                        cy < boundary or cy > sh - boundary):
                    grid[r, c] = True
                    continue
                for wall in self.walls:
                    if wall.contains_point(cx, cy, self.pathfinding_margin):
                        grid[r, c] = True
                        break
        self.occupancy_grid = grid

    def sim_to_grid(self, x, y):
        c = int(x / self.cell_size)
        r = int(y / self.cell_size)
        return (max(0, min(r, self.grid_rows - 1)),
                max(0, min(c, self.grid_cols - 1)))

    def grid_to_sim(self, r, c):
        return np.array([(c + 0.5) * self.cell_size,
                         (r + 0.5) * self.cell_size])

    def _find_nearest_free(self, r, c):
        if not self.occupancy_grid[r, c]:
            return (r, c)
        from collections import deque
        visited = {(r, c)}
        queue = deque([(r, c)])
        while queue:
            cr, cc = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) in visited:
                        continue
                    if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                        visited.add((nr, nc))
                        if not self.occupancy_grid[nr, nc]:
                            return (nr, nc)
                        queue.append((nr, nc))
        return (r, c)

    def astar_pathfind(self, start_sim, goal_sim):
        sr, sc = self.sim_to_grid(start_sim[0], start_sim[1])
        gr, gc = self.sim_to_grid(goal_sim[0], goal_sim[1])
        sr, sc = self._find_nearest_free(sr, sc)
        gr, gc = self._find_nearest_free(gr, gc)

        start = (sr, sc)
        goal = (gr, gc)
        if start == goal:
            return [start]

        def h(a):
            dr = abs(a[0] - goal[0])
            dc = abs(a[1] - goal[1])
            return max(dr, dc) + (1.414 - 1.0) * min(dr, dc)

        DIRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

        g_score = {start: 0.0}
        came_from = {}
        counter = 0
        open_set = [(h(start), counter, start)]
        closed = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            if current in closed:
                continue
            closed.add(current)

            cr, cc = current
            for dr, dc, cost in DIRS:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols):
                    continue
                if self.occupancy_grid[nr, nc]:
                    continue
                if dr != 0 and dc != 0:
                    if self.occupancy_grid[cr + dr, cc] or self.occupancy_grid[cr, cc + dc]:
                        continue
                neighbor = (nr, nc)
                if neighbor in closed:
                    continue
                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_set, (tentative_g + h(neighbor), counter, neighbor))

        return []

    def line_of_sight(self, cell_a, cell_b):
        r0, c0 = cell_a
        r1, c1 = cell_b
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc

        while True:
            if self.occupancy_grid[r0, c0]:
                return False
            if r0 == r1 and c0 == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r0 += sr
            if e2 < dr:
                err += dr
                c0 += sc
        return True

    def simplify_path(self, raw_path):
        if len(raw_path) <= 2:
            return list(raw_path)
        simplified = [raw_path[0]]
        current_idx = 0
        while current_idx < len(raw_path) - 1:
            farthest_idx = current_idx + 1
            for candidate_idx in range(len(raw_path) - 1, current_idx, -1):
                if self.line_of_sight(raw_path[current_idx], raw_path[candidate_idx]):
                    farthest_idx = candidate_idx
                    break
            simplified.append(raw_path[farthest_idx])
            current_idx = farthest_idx
        return simplified

    def plan_autopilot_path(self):
        self.build_occupancy_grid()
        head_mask = self.species[:self.snake_count] == 0
        head_centroid = self.positions[:self.snake_count][head_mask].mean(axis=0)

        raw_path = self.astar_pathfind(head_centroid, self.goal_position)
        if not raw_path:
            print("Autopilot: No path found!")
            self.autopilot_active = False
            self.autopilot_waypoints = []
            return

        simplified = self.simplify_path(raw_path)
        self.autopilot_waypoints = [self.grid_to_sim(r, c) for r, c in simplified]
        self.autopilot_wp_idx = 0
        print(f"Autopilot: Path planned — {len(self.autopilot_waypoints)} waypoints "
              f"(from {len(raw_path)} grid cells)")

    def get_head_heading(self):
        head_mask = self.species[:self.snake_count] == 0
        neck_mask = self.species[:self.snake_count] == 1
        head_centroid = self.positions[:self.snake_count][head_mask].mean(axis=0)
        neck_centroid = self.positions[:self.snake_count][neck_mask].mean(axis=0)
        heading = neck_centroid - head_centroid
        norm = np.linalg.norm(heading)
        if norm < 1e-6:
            return np.array([1.0, 0.0])
        return heading / norm

    def update_autopilot(self):
        if not self.autopilot_waypoints:
            return

        head_mask = self.species[:self.snake_count] == 0
        head_centroid = self.positions[:self.snake_count][head_mask].mean(axis=0)

        current_wp = self.autopilot_waypoints[self.autopilot_wp_idx]
        dist_to_wp = np.linalg.norm(head_centroid - current_wp)

        if dist_to_wp < self.autopilot_wp_threshold:
            if self.autopilot_wp_idx < len(self.autopilot_waypoints) - 1:
                self.autopilot_wp_idx += 1
                current_wp = self.autopilot_waypoints[self.autopilot_wp_idx]
                dist_to_wp = np.linalg.norm(head_centroid - current_wp)
            else:
                self.turn_input = 0.0
                return

        desired = current_wp - head_centroid
        desired_norm = np.linalg.norm(desired)
        if desired_norm < 1e-6:
            return
        desired_dir = desired / desired_norm

        heading = self.get_head_heading()
        cross = heading[0] * desired_dir[1] - heading[1] * desired_dir[0]
        dot = heading[0] * desired_dir[0] + heading[1] * desired_dir[1]
        angular_error = np.arctan2(cross, dot)

        nudge = 0.03
        dead_zone = 0.1
        if angular_error > dead_zone:
            self.turn_input = max(-1.0, self.turn_input - nudge)
        elif angular_error < -dead_zone:
            self.turn_input = min(1.0, self.turn_input + nudge)

    # ================================================================
    # Drawing
    # ================================================================

    def draw(self):
        """Draw simulation with maze walls, wall particles, and goal."""
        self.screen.fill((250, 250, 252))

        # Draw start zone
        if self.show_goal:
            self.draw_start()

        # Draw wall particles (replace solid wall rectangles)
        if self.show_wall_particles:
            self.draw_wall_particles()
        else:
            # Fallback: draw solid wall rectangles when particles hidden
            self.draw_walls()

        # Draw autopilot path
        if self.autopilot_active and self.autopilot_waypoints:
            self.draw_autopilot_path()

        # Draw goal
        if self.show_goal:
            self.draw_goal()

        # Draw snake particles only
        self.draw_snake_particles()

        if self.hide_gui:
            return

        # Draw centroid spine and markers
        if self.show_centroids:
            pts = self.draw_centroid_spine(line_width=2)
            self.draw_centroid_markers(pts, head_r=8, tail_r=5)

        # Info panel and control indicator
        if self.show_info:
            self.draw_control_indicator()
            self.draw_maze_info()

        # Always show the interaction matrix (including wall species)
        self.draw_matrix_viz()

        # Goal reached message
        if self.goal_reached:
            self.draw_goal_message()

    def draw_snake_particles(self):
        """Draw only snake particles (not wall particles)."""
        r = max(3, int(0.04 * self.ppu * self.zoom))
        for i in range(self.snake_count):
            x = int(self.positions[i, 0] * self.ppu * self.zoom)
            y = int(self.positions[i, 1] * self.ppu * self.zoom)
            color = self.colors[self.species[i] % len(self.colors)]
            pygame.gfxdraw.aacircle(self.screen, x, y, r, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, color)

    def draw_wall_particles(self):
        """Draw wall particles as visible dots on top of wall rectangles."""
        r = max(3, int(0.04 * self.ppu * self.zoom))
        for i in range(self.snake_count, self.n):
            x = int(self.positions[i, 0] * self.ppu * self.zoom)
            y = int(self.positions[i, 1] * self.ppu * self.zoom)
            pygame.gfxdraw.aacircle(self.screen, x, y, r, self.WALL_COLOR)
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.WALL_COLOR)

    def draw_walls(self):
        """Draw maze walls as filled rectangles."""
        wall_color = (70, 80, 100)
        border_color = (50, 55, 70)
        for wall in self.walls:
            x = int(wall.x * self.ppu * self.zoom)
            y = int(wall.y * self.ppu * self.zoom)
            w = max(1, int(wall.width * self.ppu * self.zoom))
            h = max(1, int(wall.height * self.ppu * self.zoom))
            pygame.draw.rect(self.screen, wall_color, (x, y, w, h))
            pygame.draw.rect(self.screen, border_color, (x, y, w, h), 2)

    def draw_start(self):
        sx = int(self.start_position[0] * self.ppu * self.zoom)
        sy = int(self.start_position[1] * self.ppu * self.zoom)
        sr = int(0.6 * self.ppu * self.zoom)
        pygame.draw.circle(self.screen, (200, 220, 255), (sx, sy), sr)
        pygame.draw.circle(self.screen, (150, 180, 220), (sx, sy), sr, 2)
        label = self.font.render("START", True, (100, 120, 150))
        label_rect = label.get_rect(center=(sx, sy))
        self.screen.blit(label, label_rect)

    def draw_goal(self):
        gx = int(self.goal_position[0] * self.ppu * self.zoom)
        gy = int(self.goal_position[1] * self.ppu * self.zoom)
        gr = int(self.goal_radius * self.ppu * self.zoom)
        if self.goal_reached:
            color, border = (150, 255, 150), (100, 200, 100)
        else:
            color, border = (255, 230, 180), (220, 180, 100)
        pygame.draw.circle(self.screen, color, (gx, gy), gr)
        pygame.draw.circle(self.screen, border, (gx, gy), gr, 2)
        label = self.font.render("GOAL", True, (120, 100, 60))
        label_rect = label.get_rect(center=(gx, gy))
        self.screen.blit(label, label_rect)

    def draw_goal_message(self):
        msg = self.font.render("GOAL REACHED! Press R to restart", True, (50, 150, 50))
        rect = msg.get_rect(center=(self.config.width // 2, 60))
        bg_rect = rect.inflate(20, 10)
        pygame.draw.rect(self.screen, (220, 255, 220), bg_rect)
        pygame.draw.rect(self.screen, (100, 200, 100), bg_rect, 2)
        self.screen.blit(msg, rect)

    def draw_autopilot_path(self):
        if len(self.autopilot_waypoints) >= 2:
            points = [self.to_screen(wp) for wp in self.autopilot_waypoints]
            pygame.draw.lines(self.screen, (180, 190, 210), False, points, 2)

        for i, wp in enumerate(self.autopilot_waypoints):
            sx, sy = self.to_screen(wp)
            if i < self.autopilot_wp_idx:
                pygame.draw.circle(self.screen, (150, 220, 150), (sx, sy), 5)
            elif i == self.autopilot_wp_idx:
                pygame.draw.circle(self.screen, (255, 200, 200), (sx, sy), 12, 2)
                pygame.draw.circle(self.screen, (255, 80, 80), (sx, sy), 6)
            else:
                pygame.draw.circle(self.screen, (240, 200, 80), (sx, sy), 5)

        if self.autopilot_waypoints and self.autopilot_wp_idx < len(self.autopilot_waypoints):
            head_mask = self.species[:self.snake_count] == 0
            head_centroid = self.positions[:self.snake_count][head_mask].mean(axis=0)
            head_screen = self.to_screen(head_centroid)
            wp_screen = self.to_screen(self.autopilot_waypoints[self.autopilot_wp_idx])
            pygame.draw.line(self.screen, (255, 120, 120), head_screen, wp_screen, 2)

    def draw_maze_info(self):
        maze_name = MAZE_LAYOUTS[self.current_maze_id][0]
        auto_status = "ON" if self.autopilot_active else "OFF"
        info_lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Maze: {maze_name}",
            f"Autopilot: {auto_status}",
            f"Wall particles: {self.n_wall_particles}",
            "",
            f"Turn: {self.turn_input:+.2f}",
            f"Speed: {self.forward_speed:+.2f}",
        ]
        if self.autopilot_active and self.autopilot_waypoints:
            info_lines.append(
                f"Waypoint: {self.autopilot_wp_idx + 1}/{len(self.autopilot_waypoints)}")
        info_lines += [
            "",
            "Controls:",
            "←/→: Steer | ↑/↓: Speed",
            "1-6: Maze | G: New Random",
            "A: Autopilot | W: Wall dots",
            "R: Reset | SPACE: Pause | Q: Quit",
        ]

        y = 10
        for line in info_lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 22

        self.draw_pause_indicator()
        self.draw_maze_info_footer()

    def draw_matrix_viz(self):
        """Draw K_rot and K_pos matrices including wall species."""
        cell_size = 35
        x_start = self.config.width - 30 - self.n_species * cell_size
        y_start = 10

        # Draw K_rot
        is_editing_krot = self.matrix_edit_mode and self.editing_k_rot
        y_after_krot = self._draw_matrix_with_wall(
            self.alignment_matrix, "K_rot:", x_start, y_start, is_editing_krot
        )

        # Draw K_pos below
        y_pos_start = y_after_krot + 20
        is_editing_kpos = self.matrix_edit_mode and not self.editing_k_rot
        y_after_kpos = self._draw_matrix_with_wall(
            self.matrix, "K_pos:", x_start, y_pos_start, is_editing_kpos
        )

        # Edit mode instructions
        if self.matrix_edit_mode:
            instr_y = y_after_kpos + 10
            instr = self.font.render("WASD:move E/X:+/- TAB:switch M:exit", True, (100, 100, 100))
            self.screen.blit(instr, (x_start - 50, instr_y))

    def _draw_matrix_with_wall(self, matrix, label_text, x_start, y_start, is_editing=False):
        """Draw a matrix with species color indicators and 'W' label for wall species."""
        cell_size = 35
        color_indicator_size = 12
        wall_idx = self.wall_species_idx

        # Label
        label_color = (100, 100, 200) if is_editing else (100, 100, 100)
        if is_editing:
            label_text = label_text + " (EDIT)"
        label = self.font.render(label_text, True, label_color)
        self.screen.blit(label, (x_start, y_start))
        y_start += 25

        # Column headers
        for j in range(self.n_species):
            cx = x_start + j * cell_size + cell_size // 2 - 1
            cy = y_start - 10
            if j == wall_idx:
                # Wall species: gray "W" label
                wlabel = self.font.render("W", True, self.WALL_COLOR)
                wr = wlabel.get_rect(center=(cx, cy))
                self.screen.blit(wlabel, wr)
            else:
                pygame.draw.circle(self.screen, self.colors[j], (cx, cy), color_indicator_size // 2)
                pygame.draw.circle(self.screen, (100, 100, 100), (cx, cy), color_indicator_size // 2, 1)

        for i in range(self.n_species):
            # Row header
            rx = x_start - 15
            ry = y_start + i * cell_size + cell_size // 2 - 1
            if i == wall_idx:
                wlabel = self.font.render("W", True, self.WALL_COLOR)
                wr = wlabel.get_rect(center=(rx, ry))
                self.screen.blit(wlabel, wr)
            else:
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

    def draw_maze_info_footer(self):
        if self.autopilot_active:
            label = self.font.render("AUTOPILOT", True, (50, 120, 50))
            bg_rect = label.get_rect(topright=(self.config.width - 10, self.config.height - 35))
            bg = bg_rect.inflate(12, 6)
            pygame.draw.rect(self.screen, (220, 255, 220), bg)
            pygame.draw.rect(self.screen, (100, 200, 100), bg, 2)
            self.screen.blit(label, bg_rect)

    # ================================================================
    # Event handling
    # ================================================================

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self._initialize_at_start()
                    self.turn_input = 0.0
                    self.turn_history[:] = 0.0
                    self.history_idx = 0
                    if self.autopilot_active:
                        self.plan_autopilot_path()
                    print("Reset positions")

                elif event.key == pygame.K_g:
                    self._load_maze_and_rebuild(6)
                    self.build_occupancy_grid()
                    if self.autopilot_active:
                        self.plan_autopilot_path()
                    print("Generated new random maze")

                elif event.key == pygame.K_m:
                    self.show_goal = not self.show_goal
                    print(f"Show goal: {self.show_goal}")

                elif event.key == pygame.K_w:
                    self.show_wall_particles = not self.show_wall_particles
                    print(f"Show wall particles: {self.show_wall_particles}")

                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info

                elif event.key == pygame.K_v:
                    self.show_centroids = not self.show_centroids

                elif event.key == pygame.K_h:
                    self.hide_gui = not self.hide_gui

                elif event.key == pygame.K_a:
                    if not self.goal_reached:
                        self.autopilot_active = not self.autopilot_active
                        if self.autopilot_active:
                            self.plan_autopilot_path()
                        else:
                            self.turn_input = 0.0
                        print(f"Autopilot: {'ON' if self.autopilot_active else 'OFF'}")

                # Maze selection (1-6)
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3,
                                   pygame.K_4, pygame.K_5, pygame.K_6):
                    maze_num = event.key - pygame.K_0
                    self._load_maze_and_rebuild(maze_num)
                    self.build_occupancy_grid()
                    if self.autopilot_active:
                        self.plan_autopilot_path()

                # Forward speed adjustment
                elif event.key == pygame.K_UP:
                    self.update_forward_speed(0.05)
                elif event.key == pygame.K_DOWN:
                    self.update_forward_speed(-0.05)

        # Continuous arrow key input for steering (disabled during autopilot)
        if not self.autopilot_active:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.turn_input = max(-1.0, self.turn_input - 0.05)
            if keys[pygame.K_RIGHT]:
                self.turn_input = min(1.0, self.turn_input + 0.05)

        return True


def main():
    demo = SnakeMazeWallDemo(snake_n_species=6, n_particles=15, maze_id=1)
    demo.run()


if __name__ == "__main__":
    main()
