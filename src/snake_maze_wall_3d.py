#!/usr/bin/env python3
"""
3D Snake Maze Demo (Wall Particles) — Navigate a snake through a true 3D maze

Walls are generated as geometric panels (Wall3D) and also sampled as particles.
The geometric walls provide Y-level slice visualization (same as snake_maze_3d),
while wall particles create smooth repulsive force fields for the snake.

Each snake particle only interacts with its nearest wall particle, dramatically
reducing computation cost from O(S*N) to O(S*(S+1)).

Controls:
    Arrow keys: Yaw/Pitch steering
    Z/X:   Roll steering
    +/-:   Forward speed
    P:     Toggle autopilot (A* pathfinding)
    G:     Generate new random maze
    R:     Reset positions
    W:     Toggle wall particle visibility
    M:     Toggle goal marker
    SPACE: Pause/Resume
    H/I/V/A: GUI toggles
    HOME:  Reset camera
    Q/ESC: Quit

Camera:
    Left drag:   Rotate camera
    Right drag:  Pan view
    Scroll:      Zoom
"""

import heapq
from math import ceil

import pygame
import pygame.gfxdraw
import numpy as np
from particle_life_3d import Config3D, ParticleLife3D
from snake_demo import generate_position_matrix
from snake_demo_3d import SnakeDemo3D
from snake_maze_3d import (Wall3D, create_maze_3d_random,
                           ORIENT_FILL, ORIENT_BORDER, SLICE_THICKNESS)


# =============================================================================
# Wall helpers
# =============================================================================

def _wall_orient(wall):
    """Return 'x', 'y', or 'z' for the wall's thin dimension."""
    if wall.width <= wall.height and wall.width <= wall.depth:
        return 'x'
    if wall.height <= wall.width and wall.height <= wall.depth:
        return 'y'
    return 'z'


def _wall_main_face(wall):
    """Get 4 corners of the wall's main face (perpendicular to thin axis)."""
    o = _wall_orient(wall)
    x, y, z = wall.x, wall.y, wall.z
    w, h, d = wall.width, wall.height, wall.depth
    if o == 'x':
        cx = x + w / 2
        return np.array([[cx, y, z], [cx, y+h, z],
                         [cx, y+h, z+d], [cx, y, z+d]])
    elif o == 'y':
        cy = y + h / 2
        return np.array([[x, cy, z], [x+w, cy, z],
                         [x+w, cy, z+d], [x, cy, z+d]])
    else:
        cz = z + d / 2
        return np.array([[x, y, cz], [x+w, y, cz],
                         [x+w, y+h, cz], [x, y+h, cz]])


def _sample_wall_particles(walls, spacing):
    """Sample particle positions on Wall3D faces.

    For each wall, sample a grid of points on the wall's main face
    (the large face perpendicular to the thin dimension).
    """
    positions = []
    for wall in walls:
        o = _wall_orient(wall)
        if o == 'x':
            cx = wall.x + wall.width / 2
            n_y = max(1, int(wall.height / spacing))
            n_z = max(1, int(wall.depth / spacing))
            for iy in range(n_y):
                for iz in range(n_z):
                    py = wall.y + (iy + 0.5) * wall.height / n_y
                    pz = wall.z + (iz + 0.5) * wall.depth / n_z
                    positions.append([cx, py, pz])
        elif o == 'y':
            cy = wall.y + wall.height / 2
            n_x = max(1, int(wall.width / spacing))
            n_z = max(1, int(wall.depth / spacing))
            for ix in range(n_x):
                for iz in range(n_z):
                    px = wall.x + (ix + 0.5) * wall.width / n_x
                    pz = wall.z + (iz + 0.5) * wall.depth / n_z
                    positions.append([px, cy, pz])
        else:  # 'z'
            cz = wall.z + wall.depth / 2
            n_x = max(1, int(wall.width / spacing))
            n_y = max(1, int(wall.height / spacing))
            for ix in range(n_x):
                for iy in range(n_y):
                    px = wall.x + (ix + 0.5) * wall.width / n_x
                    py = wall.y + (iy + 0.5) * wall.height / n_y
                    positions.append([px, py, cz])
    return np.array(positions) if positions else np.zeros((0, 3))


# =============================================================================
# 3D Snake Maze Demo (Wall Particles)
# =============================================================================

class SnakeMazeWall3D(SnakeDemo3D):
    """
    3D Snake demo with maze environment using wall particles for physics
    and geometric wall rendering with Y-level slice visualization.

    Each snake particle interacts with its K nearest wall particles
    for efficient computation.
    """

    WALL_SPACING = 0.3
    WALL_REPULSION = 0.0        # Only universal short-range repulsion (Zone 1)
    WALL_COLOR = (90, 100, 120)
    N_NEAREST_WALLS = 10        # Number of nearest wall particles per snake particle

    def __init__(self, snake_n_species: int = 6, n_particles: int = 15):
        self.snake_n_species = snake_n_species
        self.wall_species_idx = snake_n_species
        self.show_wall_particles = True   # Show wall particles (toggle with W)

        # Goal
        self.goal_radius = 0.6
        self.show_goal = True
        self.goal_reached = False

        # Total species = snake species + 1 wall species
        total_species = snake_n_species + 1

        # Build K_pos: snake chain + wall repulsion
        snake_k_pos = generate_position_matrix(snake_n_species)
        k_pos = np.zeros((total_species, total_species))
        k_pos[:snake_n_species, :snake_n_species] = snake_k_pos
        for i in range(snake_n_species):
            k_pos[i, self.wall_species_idx] = self.WALL_REPULSION
        k_pos[self.wall_species_idx, :] = 0.0

        zeros = np.zeros((total_species, total_species)).tolist()

        # 3D workspace (same as snake_maze_3d: 10x10x10)
        sim_w, sim_h, sim_d = 10.0, 10.0, 10.0

        config = Config3D(
            n_species=total_species,
            n_particles=n_particles,
            sim_width=sim_w,
            sim_height=sim_h,
            sim_depth=sim_d,
            a_rot=3.0,
            max_speed=0.15,
            position_matrix=k_pos.tolist(),
            orientation_matrix_x=zeros,
            orientation_matrix_y=zeros,
            orientation_matrix_z=zeros,
        )

        # Call ParticleLife3D.__init__ directly (skip SnakeDemo3D.__init__)
        ParticleLife3D.__init__(self, config, headless=False)

        # --- Replicate SnakeDemo3D control state ---
        self.yaw_input = 0.0
        self.pitch_input = 0.0
        self.roll_input = 0.0
        self.base_k_rot = 0.08
        self.forward_speed = 0.25
        self.turn_decay = 0.92

        self.joint_delay = 8
        n_joints = snake_n_species - 1
        history_len = self.joint_delay * n_joints + 1
        self.yaw_history = np.zeros(history_len)
        self.pitch_history = np.zeros(history_len)
        self.roll_history = np.zeros(history_len)
        self.history_idx = 0

        self.group_spacing = 0.8
        self.hide_gui = False
        self.show_centroids = True

        # Start and goal (set before maze load)
        self.start_position = np.array([1.0, 1.0, 1.0])
        self.goal_position = np.array([sim_w - 1.0, sim_h - 1.0, sim_d - 1.0])

        # Autopilot (3D A* pathfinding)
        self.autopilot_active = False
        self.autopilot_waypoints = []
        self.autopilot_wp_idx = 0
        self.autopilot_wp_threshold = 0.2
        self.cell_size = 1.0
        self.pathfinding_margin = 0.3
        self.occupancy_grid = None
        self.grid_rows = 0
        self.grid_cols = 0
        self.grid_layers = 0
        self._autopilot_best_dist = float('inf')
        self._autopilot_stall_counter = 0

        # Translucent wall overlay surface
        self._wall_overlay = pygame.Surface(
            (config.width, config.height), pygame.SRCALPHA)

        # Maze state
        self.walls_3d = []  # Wall3D objects for visualization

        # --- Load maze ---
        self._load_maze_and_rebuild()

        pygame.display.set_caption("3D Snake Maze (Wall Particles)")

        print("=" * 60)
        print("3D Snake Maze Demo (Wall Particles)")
        print("=" * 60)
        print(f"Snake species: {snake_n_species}, Wall species: 1")
        print(f"Space: {sim_w}x{sim_h}x{sim_d}m")
        print("")
        print("Controls:")
        print("  \u2190/\u2192       Yaw (horizontal turn)")
        print("  \u2191/\u2193       Pitch (vertical turn)")
        print("  Z/X       Roll (barrel roll)")
        print("  +/-       Forward speed")
        print("  P         Toggle autopilot (A* pathfinding)")
        print("  G         Generate new random maze")
        print("  W         Toggle wall particle visibility")
        print("  R         Reset positions")
        print("  M         Toggle goal marker")
        print("  SPACE     Pause")
        print("  H/I/V/A   GUI toggles")
        print("  HOME      Reset camera  Q: Quit")
        print("=" * 60)

    # ================================================================
    # Maze loading
    # ================================================================

    def _load_maze_and_rebuild(self):
        """Generate 3D maze, create wall particles, rebuild arrays."""
        w = self.config.sim_width
        h = self.config.sim_height
        d = self.config.sim_depth

        # Generate Wall3D objects (same maze as snake_maze_3d)
        self.walls_3d = create_maze_3d_random(w, h, d)

        # Sample wall particles from Wall3D faces
        wall_positions = _sample_wall_particles(self.walls_3d, self.WALL_SPACING)

        self.n_wall_particles = len(wall_positions)
        self.snake_count = self.snake_n_species * self.config.n_particles

        # Start/goal at cell centers (3x3x3 grid)
        cell_w = w / 3
        cell_h = h / 3
        cell_d = d / 3
        self.start_position = np.array([cell_w / 2, cell_h / 2, cell_d / 2])
        self.goal_position = np.array([w - cell_w / 2, h - cell_h / 2,
                                       d - cell_d / 2])

        # Rebuild arrays
        total_n = self.snake_count + self.n_wall_particles
        self.n = total_n
        self.positions = np.zeros((total_n, 3))
        self.velocities = np.zeros((total_n, 3))
        self.species = np.zeros(total_n, dtype=int)

        # Place snake at start
        self._initialize_at_start()

        # Place wall particles
        if self.n_wall_particles > 0:
            start_idx = self.snake_count
            self.positions[start_idx:] = wall_positions
            self.velocities[start_idx:] = 0.0
            self.species[start_idx:] = self.wall_species_idx

        self.goal_reached = False
        self.autopilot_active = False
        self.autopilot_waypoints = []
        print(f"Loaded 3D maze ({len(self.walls_3d)} walls, "
              f"{self.n_wall_particles} wall particles)")

    def _initialize_at_start(self):
        """Initialize snake particles at start position."""
        sx, sy, sz = self.start_position
        particles_per_species = self.config.n_particles

        for i in range(self.snake_count):
            species_id = min(i // particles_per_species,
                             self.snake_n_species - 1)
            self.species[i] = species_id
            group_x = sx + species_id * self.group_spacing * 0.4
            self.positions[i, 0] = group_x + self.rng.uniform(-0.08, 0.08)
            self.positions[i, 1] = sy + self.rng.uniform(-0.08, 0.08)
            self.positions[i, 2] = sz + self.rng.uniform(-0.08, 0.08)
            self.velocities[i] = [0.0, 0.0, 0.0]

        self.goal_reached = False

    # ================================================================
    # Species centroids (snake only)
    # ================================================================

    def get_species_centroids(self):
        """Return centroids for snake species only."""
        centroids = []
        center = np.array([
            self.config.sim_width / 2,
            self.config.sim_height / 2,
            self.config.sim_depth / 2
        ])
        for s in range(self.snake_n_species):
            mask = self.species[:self.snake_count] == s
            if mask.any():
                centroids.append(
                    self.positions[:self.snake_count][mask].mean(axis=0))
            else:
                centroids.append(center.copy())
        return centroids

    # ================================================================
    # Matrix updates (snake species only)
    # ================================================================

    def update_matrices_from_input(self):
        """Update K_rot for snake joints only (wall species stays 0)."""
        self.yaw_history[self.history_idx] = self.yaw_input
        self.pitch_history[self.history_idx] = self.pitch_input
        self.roll_history[self.history_idx] = self.roll_input
        self.history_idx = (self.history_idx + 1) % len(self.yaw_history)

        n_joints = self.snake_n_species - 1
        total = self.config.n_species
        K_rot_y = np.zeros((total, total))
        K_rot_x = np.zeros((total, total))
        K_rot_z = np.zeros((total, total))

        for joint in range(n_joints):
            delay = joint * self.joint_delay
            idx = (self.history_idx - 1 - delay) % len(self.yaw_history)

            delayed_yaw = self.yaw_history[idx]
            strength_y = -self.base_k_rot * delayed_yaw
            K_rot_y[joint, joint + 1] = strength_y
            K_rot_y[joint + 1, joint] = strength_y

            delayed_pitch = self.pitch_history[idx]
            strength_x = -self.base_k_rot * delayed_pitch
            K_rot_x[joint, joint + 1] = strength_x
            K_rot_x[joint + 1, joint] = strength_x

            delayed_roll = self.roll_history[idx]
            strength_z = -self.base_k_rot * delayed_roll
            K_rot_z[joint, joint + 1] = strength_z
            K_rot_z[joint + 1, joint] = strength_z

        self.alignment_matrix_y[:] = np.clip(K_rot_y, -1.0, 1.0)
        self.alignment_matrix_x[:] = np.clip(K_rot_x, -1.0, 1.0)
        self.alignment_matrix_z[:] = np.clip(K_rot_z, -1.0, 1.0)

    def update_forward_speed(self, delta: float):
        """Adjust forward speed — only update snake portion of K_pos."""
        self.forward_speed = np.clip(self.forward_speed + delta, -0.3, 0.5)
        new_snake_matrix = generate_position_matrix(
            self.snake_n_species, forward_bias=self.forward_speed
        )
        self.matrix[:self.snake_n_species,
                     :self.snake_n_species] = new_snake_matrix
        print(f"Forward speed: {self.forward_speed:.2f}")

    # ================================================================
    # 3D A* Autopilot
    # ================================================================

    def build_occupancy_grid_3d(self):
        """Build a 3D boolean grid where True = blocked (wall or boundary)."""
        sw = self.config.sim_width
        sh = self.config.sim_height
        sd = self.config.sim_depth
        self.grid_cols = ceil(sw / self.cell_size)
        self.grid_rows = ceil(sh / self.cell_size)
        self.grid_layers = ceil(sd / self.cell_size)

        grid = np.zeros((self.grid_layers, self.grid_rows, self.grid_cols),
                         dtype=bool)

        boundary = 0.1
        for l in range(self.grid_layers):
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    cx = (c + 0.5) * self.cell_size
                    cy = (r + 0.5) * self.cell_size
                    cz = (l + 0.5) * self.cell_size
                    if (cx < boundary or cx > sw - boundary or
                            cy < boundary or cy > sh - boundary or
                            cz < boundary or cz > sd - boundary):
                        grid[l, r, c] = True
                        continue
                    for wall in self.walls_3d:
                        if wall.contains_point(cx, cy, cz,
                                               self.pathfinding_margin):
                            grid[l, r, c] = True
                            break

        self.occupancy_grid = grid

    def sim_to_grid(self, x, y, z):
        c = int(x / self.cell_size)
        r = int(y / self.cell_size)
        l = int(z / self.cell_size)
        return (max(0, min(l, self.grid_layers - 1)),
                max(0, min(r, self.grid_rows - 1)),
                max(0, min(c, self.grid_cols - 1)))

    def grid_to_sim(self, l, r, c):
        return np.array([(c + 0.5) * self.cell_size,
                         (r + 0.5) * self.cell_size,
                         (l + 0.5) * self.cell_size])

    def _find_nearest_free_3d(self, l, r, c):
        """Find nearest unblocked cell using BFS."""
        if not self.occupancy_grid[l, r, c]:
            return l, r, c
        visited = set()
        queue = [(l, r, c)]
        visited.add((l, r, c))
        while queue:
            cl, cr, cc = queue.pop(0)
            for dl, dr, dc in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),
                                (0,0,-1),(0,0,1)]:
                nl, nr, nc = cl + dl, cr + dr, cc + dc
                if (0 <= nl < self.grid_layers and
                        0 <= nr < self.grid_rows and
                        0 <= nc < self.grid_cols and
                        (nl, nr, nc) not in visited):
                    if not self.occupancy_grid[nl, nr, nc]:
                        return nl, nr, nc
                    visited.add((nl, nr, nc))
                    queue.append((nl, nr, nc))
        return l, r, c

    def astar_pathfind_3d(self, start_sim, goal_sim):
        """3D A* pathfinding on occupancy grid (6-connected)."""
        sl, sr, sc = self.sim_to_grid(*start_sim)
        gl, gr, gc = self.sim_to_grid(*goal_sim)
        sl, sr, sc = self._find_nearest_free_3d(sl, sr, sc)
        gl, gr, gc = self._find_nearest_free_3d(gl, gr, gc)

        start = (sl, sr, sc)
        goal = (gl, gr, gc)
        if start == goal:
            return [start]

        def h(node):
            return (abs(node[0] - goal[0]) + abs(node[1] - goal[1])
                    + abs(node[2] - goal[2]))

        DIRS = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]

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

            cl, cr, cc = current
            for dl, dr, dc in DIRS:
                nl, nr, nc = cl + dl, cr + dr, cc + dc
                if not (0 <= nl < self.grid_layers and
                        0 <= nr < self.grid_rows and
                        0 <= nc < self.grid_cols):
                    continue
                if self.occupancy_grid[nl, nr, nc]:
                    continue
                neighbor = (nl, nr, nc)
                if neighbor in closed:
                    continue
                tentative_g = g_score[current] + 1.0
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(
                        open_set,
                        (tentative_g + h(neighbor), counter, neighbor))

        return []

    def simplify_path_3d(self, path):
        """Reduce path waypoints by removing collinear intermediate points."""
        if len(path) <= 2:
            return path
        result = [path[0]]
        for i in range(1, len(path) - 1):
            dl0 = path[i][0] - path[i-1][0]
            dr0 = path[i][1] - path[i-1][1]
            dc0 = path[i][2] - path[i-1][2]
            dl1 = path[i+1][0] - path[i][0]
            dr1 = path[i+1][1] - path[i][1]
            dc1 = path[i+1][2] - path[i][2]
            if (dl0, dr0, dc0) != (dl1, dr1, dc1):
                result.append(path[i])
        result.append(path[-1])
        return result

    def plan_autopilot_path(self):
        """Build occupancy grid, run A*, convert to sim-space waypoints."""
        self.build_occupancy_grid_3d()

        head_mask = self.species[:self.snake_count] == 0
        head_centroid = self.positions[:self.snake_count][head_mask].mean(
            axis=0)

        raw_path = self.astar_pathfind_3d(head_centroid, self.goal_position)
        if not raw_path:
            print("Autopilot: No path found!")
            self.autopilot_active = False
            self.autopilot_waypoints = []
            return

        simplified = self.simplify_path_3d(raw_path)
        self.autopilot_waypoints = [self.grid_to_sim(l, r, c)
                                     for l, r, c in simplified]
        self.autopilot_wp_idx = 0
        self._autopilot_best_dist = float('inf')
        self._autopilot_stall_counter = 0
        print(f"Autopilot: Path planned \u2014 {len(self.autopilot_waypoints)}"
              f" waypoints (from {len(raw_path)} grid cells)")

    def get_chain_backward(self):
        """Get backward direction along chain (neck - head), normalized."""
        head_mask = self.species[:self.snake_count] == 0
        neck_mask = self.species[:self.snake_count] == 1
        head_centroid = self.positions[:self.snake_count][head_mask].mean(
            axis=0)
        neck_centroid = self.positions[:self.snake_count][neck_mask].mean(
            axis=0)
        backward = neck_centroid - head_centroid
        norm = np.linalg.norm(backward)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return backward / norm

    def update_autopilot(self):
        """Steer yaw/pitch/roll toward the next waypoint.

        Uses lever-arm physics: for each rotation axis, compute how much
        the resulting force on the head aligns with the steering direction.
        Includes anti-parallel escape for U-turns, speed modulation, and
        stall recovery with replanning.
        """
        if not self.autopilot_waypoints:
            return

        head_mask = self.species[:self.snake_count] == 0
        head_centroid = self.positions[:self.snake_count][head_mask].mean(
            axis=0)

        current_wp = self.autopilot_waypoints[self.autopilot_wp_idx]
        dist_to_wp = np.linalg.norm(head_centroid - current_wp)

        if dist_to_wp < self.autopilot_wp_threshold:
            if self.autopilot_wp_idx < len(self.autopilot_waypoints) - 1:
                self.autopilot_wp_idx += 1
                self._autopilot_best_dist = float('inf')
                self._autopilot_stall_counter = 0
                current_wp = self.autopilot_waypoints[self.autopilot_wp_idx]
                dist_to_wp = np.linalg.norm(head_centroid - current_wp)
            else:
                self.yaw_input = 0.0
                self.pitch_input = 0.0
                self.roll_input = 0.0
                return

        # Stall detection — replan if stuck for too long
        if dist_to_wp < self._autopilot_best_dist - 0.05:
            self._autopilot_best_dist = dist_to_wp
            self._autopilot_stall_counter = 0
        else:
            self._autopilot_stall_counter += 1

        if self._autopilot_stall_counter > 800:
            self.plan_autopilot_path()
            if not self.autopilot_waypoints:
                return
            current_wp = self.autopilot_waypoints[self.autopilot_wp_idx]
            dist_to_wp = np.linalg.norm(head_centroid - current_wp)

        desired = current_wp - head_centroid
        desired_norm = np.linalg.norm(desired)
        if desired_norm < 1e-6:
            return
        desired_dir = desired / desired_norm

        backward = self.get_chain_backward()
        alignment = np.dot(backward, desired_dir)

        # Anti-parallel escape: steer perpendicular for U-turn
        if alignment > 0.5:
            perp = desired_dir - alignment * backward
            perp_norm = np.linalg.norm(perp)
            if perp_norm < 1e-3:
                perp = np.cross(backward, np.array([1.0, 0.0, 0.0]))
                if np.linalg.norm(perp) < 0.1:
                    perp = np.cross(backward, np.array([0.0, 1.0, 0.0]))
                perp_norm = np.linalg.norm(perp)
            steer_dir = perp / perp_norm
        else:
            steer_dir = desired_dir

        axes = [
            ('yaw',   np.array([0.0, 1.0, 0.0])),
            ('pitch', np.array([1.0, 0.0, 0.0])),
            ('roll',  np.array([0.0, 0.0, 1.0])),
        ]

        nudge = 0.12 if alignment > 0.3 else 0.08
        dead_zone = 0.02

        for name, axis in axes:
            lever_raw = np.cross(backward, axis)
            lever_norm = np.linalg.norm(lever_raw)
            if lever_norm < 0.1:
                continue
            head_force_dir = -lever_raw / lever_norm
            leverage = np.dot(steer_dir, head_force_dir)
            if abs(leverage) < dead_zone:
                continue
            strength = nudge * min(1.0, abs(leverage) / 0.2)
            delta = strength if leverage > 0 else -strength

            if name == 'yaw':
                self.yaw_input = np.clip(
                    self.yaw_input + delta, -1.0, 1.0)
            elif name == 'pitch':
                self.pitch_input = np.clip(
                    self.pitch_input + delta, -1.0, 1.0)
            elif name == 'roll':
                self.roll_input = np.clip(
                    self.roll_input + delta, -1.0, 1.0)

    # ================================================================
    # Physics — optimized: nearest wall particle only
    # ================================================================

    def compute_velocities(self) -> np.ndarray:
        """Compute 3D velocities for snake particles.

        Optimization: each snake particle interacts with all other snake
        particles but only the K nearest wall particles (instead of all
        wall particles), reducing force computation from O(S*N) to
        O(S*(S+K)).
        """
        new_velocities = np.zeros_like(self.velocities)

        S = self.snake_count
        K = self.N_NEAREST_WALLS
        snake_pos = self.positions[:S]  # (S, 3)
        snake_species = self.species[:S]  # (S,)

        # Pre-compute K nearest wall particles for each snake particle
        nearest_wall_pos = None  # (S, K, 3) if walls exist
        if self.n_wall_particles > 0:
            wall_pos = self.positions[S:]  # (M, 3)
            M = len(wall_pos)
            K_actual = min(K, M)
            # Vectorized distance: (S, M)
            diff_sw = snake_pos[:, None, :] - wall_pos[None, :, :]
            dist_sw = np.linalg.norm(diff_sw, axis=2)
            # K nearest indices per snake particle: (S, K_actual)
            nearest_idx = np.argpartition(dist_sw, K_actual, axis=1)[:, :K_actual]
            nearest_wall_pos = wall_pos[nearest_idx]  # (S, K_actual, 3)

        r_max = self.config.r_max
        beta = self.config.beta
        inv_1_minus_beta = 1.0 / (1.0 - beta) if beta < 1.0 else 1.0
        force_scale = self.config.force_scale
        far_attraction = self.config.far_attraction
        a_rot = self.config.a_rot
        peak_r = 0.5 * (1.0 + beta)

        axes = [np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0])]
        rot_matrices = [self.alignment_matrix_x,
                        self.alignment_matrix_y,
                        self.alignment_matrix_z]

        wall_species_arr = np.full(K if nearest_wall_pos is not None
                                   else 0, self.wall_species_idx, dtype=int)

        for i in range(S):
            # Build interaction set: all snake particles + K nearest walls
            if nearest_wall_pos is not None:
                interact_pos = np.vstack(
                    [snake_pos, nearest_wall_pos[i]])          # (S+K, 3)
                interact_species = np.concatenate(
                    [snake_species, wall_species_arr])          # (S+K,)
            else:
                interact_pos = snake_pos                       # (S, 3)
                interact_species = snake_species               # (S,)

            N_local = len(interact_pos)
            delta = interact_pos - self.positions[i]           # (N_local, 3)
            dist = np.linalg.norm(delta, axis=1)               # (N_local,)
            dist[i] = np.inf  # Exclude self

            r_norm = dist / r_max
            inv_r = 1.0 / (dist + 1e-8)
            r_hat = delta * inv_r[:, None]

            si = self.species[i]
            k_pos = self.matrix[si, interact_species]

            # --- Radial force magnitudes ---
            F = np.zeros(N_local)

            # Zone 1: universal repulsion (r < beta*r_max)
            near = r_norm < beta
            F[near] = r_norm[near] / beta - 1.0

            # Zone 2-4: attraction/repulsion (beta <= r < 1)
            mid = (r_norm >= beta) & (r_norm < 1.0)
            if mid.any():
                tri = (1.0 - np.abs(2.0 * r_norm[mid] - 1.0 - beta)
                       * inv_1_minus_beta)
                below_peak = r_norm[mid] < peak_r
                F[mid] = np.where(
                    below_peak,
                    k_pos[mid] * tri,
                    k_pos[mid] * np.maximum(far_attraction, tri))

            # Far zone (r >= r_max)
            if far_attraction > 0:
                far = r_norm >= 1.0
                far[i] = False
                F[far] = k_pos[far] * far_attraction

            # Radial contribution
            velocity_sum = (force_scale * F[:, None] * r_hat).sum(axis=0)

            # --- Tangential (swirl) forces — only within r_max ---
            active = near | mid
            if active.any():
                sw = np.clip(1.0 - r_norm, 0.0, 1.0)
                for rot_mat, axis in zip(rot_matrices, axes):
                    k_rot = rot_mat[si, interact_species]
                    has_rot = active & (np.abs(k_rot) > 1e-8)
                    if not has_rot.any():
                        continue
                    t = np.cross(r_hat[has_rot], axis)
                    t_norm = np.linalg.norm(t, axis=1)
                    valid = t_norm > 1e-8
                    if not valid.any():
                        continue
                    coeff = (k_rot[has_rot][valid] * a_rot
                             * sw[has_rot][valid] / t_norm[valid])
                    velocity_sum += (coeff[:, None] * t[valid]).sum(axis=0)

            new_velocities[i] = velocity_sum

        return new_velocities

    # ================================================================
    # Step override — wall particles stay fixed
    # ================================================================

    def step(self):
        """Perform one simulation step — wall particles stay fixed."""
        if self.paused:
            return

        # Autopilot overrides steering
        if self.autopilot_active:
            self.update_autopilot()

        self.yaw_input *= self.turn_decay
        if abs(self.yaw_input) < 0.01:
            self.yaw_input = 0.0
        self.pitch_input *= self.turn_decay
        if abs(self.pitch_input) < 0.01:
            self.pitch_input = 0.0
        self.roll_input *= self.turn_decay
        if abs(self.roll_input) < 0.01:
            self.roll_input = 0.0

        self.update_matrices_from_input()
        self.velocities = self.compute_velocities()

        snake_v = self.velocities[:self.snake_count]
        speed = np.linalg.norm(snake_v, axis=1, keepdims=True)
        self.velocities[:self.snake_count] = np.where(
            speed > self.config.max_speed,
            snake_v * self.config.max_speed / speed,
            snake_v
        )

        self.positions[:self.snake_count] += (
            self.velocities[:self.snake_count] * self.config.dt
        )

        margin = 0.05
        bounds = [
            (0, self.config.sim_width),
            (1, self.config.sim_height),
            (2, self.config.sim_depth)
        ]
        for dim, limit in bounds:
            mask = self.positions[:self.snake_count, dim] < margin
            self.positions[:self.snake_count][mask, dim] = margin
            self.velocities[:self.snake_count][mask, dim] = abs(
                self.velocities[:self.snake_count][mask, dim])

            upper = limit - margin
            mask = self.positions[:self.snake_count, dim] > upper
            self.positions[:self.snake_count][mask, dim] = upper
            self.velocities[:self.snake_count][mask, dim] = -abs(
                self.velocities[:self.snake_count][mask, dim])

        self.check_goal()

    # ================================================================
    # Goal checking
    # ================================================================

    def check_goal(self):
        if not self.show_goal:
            return
        head_mask = self.species[:self.snake_count] == 0
        if not head_mask.any():
            return
        head = self.positions[:self.snake_count][head_mask].mean(axis=0)
        if np.linalg.norm(head - self.goal_position) < self.goal_radius:
            if not self.goal_reached:
                self.goal_reached = True
                print("Goal reached!")

    # ================================================================
    # Drawing
    # ================================================================

    def draw(self):
        self.screen.fill((250, 250, 252))

        if self.show_axes:
            self.draw_bounding_box()

        if self.show_goal:
            self.draw_start_3d()
            self.draw_goal_3d()

        # Geometric wall visualization (slice at snake Y-level)
        self.draw_walls_3d()

        if self.autopilot_active and self.autopilot_waypoints:
            self.draw_autopilot_path()

        # Snake particles
        screen_x, screen_y, depth = self.project_batch(
            self.positions[:self.snake_count])
        indices = np.argsort(-depth)
        r = max(3, int(0.04 * self.ppu * self.cam_zoom))

        for i in indices:
            color = self.colors[self.species[i] % len(self.colors)]
            sx, sy = int(screen_x[i]), int(screen_y[i])
            try:
                pygame.gfxdraw.aacircle(self.screen, sx, sy, r, color)
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, r, color)
            except OverflowError:
                pass

        # Optional: wall particles as dots (toggle with W)
        if self.show_wall_particles and self.n_wall_particles > 0:
            wall_pos = self.positions[self.snake_count:]
            wx, wy, wd = self.project_batch(wall_pos)
            wall_r = max(2, int(0.02 * self.ppu * self.cam_zoom))
            w_indices = np.argsort(-wd)
            for i in w_indices:
                sx, sy = int(wx[i]), int(wy[i])
                try:
                    pygame.gfxdraw.aacircle(
                        self.screen, sx, sy, wall_r, self.WALL_COLOR)
                    pygame.gfxdraw.filled_circle(
                        self.screen, sx, sy, wall_r, self.WALL_COLOR)
                except OverflowError:
                    pass

        if self.hide_gui:
            return

        if self.show_centroids:
            self.draw_centroid_spine_3d()

        if self.show_info:
            self.draw_control_indicator()
            self.draw_maze_info()

        if self.goal_reached:
            self.draw_goal_message()

        if self.paused:
            text = self.font.render("PAUSED", True, (200, 50, 50))
            self.screen.blit(text, text.get_rect(
                center=(self.config.width // 2, 30)))

    # ================================================================
    # Wall drawing — Y-level slice (same as snake_maze_3d)
    # ================================================================

    def draw_walls_3d(self):
        """Draw walls at the snake's current Y-level (horizontal slice).

        Only walls intersecting a horizontal slab around the snake head
        are shown. Vertical walls (X/Z-perpendicular) are blue; floors/
        ceilings (Y-perpendicular) are green. Boundary floors skipped.
        """
        if not self.walls_3d:
            return

        head_mask = self.species[:self.snake_count] == 0
        head_pos = self.positions[:self.snake_count][head_mask].mean(axis=0)
        snake_y = head_pos[1]

        bnd_margin = 0.5  # skip boundary floor/ceiling walls
        sim_h = self.config.sim_height

        visible = []
        for wall in self.walls_3d:
            # Skip top/bottom boundary Y-walls
            if _wall_orient(wall) == 'y':
                cy = wall.y + wall.height / 2
                if cy < bnd_margin or cy > sim_h - bnd_margin:
                    continue

            wall_y_min = wall.y
            wall_y_max = wall.y + wall.height
            if wall_y_max < snake_y - SLICE_THICKNESS:
                continue
            if wall_y_min > snake_y + SLICE_THICKNESS:
                continue
            cx = wall.x + wall.width / 2
            cz = wall.z + wall.depth / 2
            dist_xz = np.sqrt((cx - head_pos[0])**2 +
                              (cz - head_pos[2])**2)
            visible.append((wall, dist_xz))

        if not visible:
            return

        n = len(visible)
        all_corners = np.zeros((n * 4, 3))
        for i, (wall, _) in enumerate(visible):
            all_corners[i*4:(i+1)*4] = _wall_main_face(wall)

        sx, sy, dp = self.project_batch(all_corners)
        self._wall_overlay.fill((0, 0, 0, 0))

        faces = []
        for i, (wall, _) in enumerate(visible):
            pts = [(int(sx[i*4+j]), int(sy[i*4+j])) for j in range(4)]
            avg_d = dp[i*4:(i+1)*4].mean()
            orient = _wall_orient(wall)
            base = ORIENT_FILL[orient]
            bord = ORIENT_BORDER[orient]

            if orient == 'y':
                fa, ba = 70, 160
            else:
                fa, ba = 100, 220
            faces.append((avg_d, pts, (*base, fa), (*bord, ba)))

        faces.sort(key=lambda f: -f[0])
        for _, pts, fill, border in faces:
            pygame.draw.polygon(self._wall_overlay, fill, pts)
            pygame.draw.polygon(self._wall_overlay, border, pts, 2)

        self.screen.blit(self._wall_overlay, (0, 0))

    # ================================================================
    # HUD
    # ================================================================

    def draw_start_3d(self):
        sx, sy, _ = self.project_3d_to_2d(self.start_position)
        sx, sy = int(sx), int(sy)
        sr = int(0.6 * self.ppu * self.cam_zoom)
        pygame.draw.circle(self.screen, (200, 220, 255), (sx, sy), sr)
        pygame.draw.circle(self.screen, (150, 180, 220), (sx, sy), sr, 2)
        label = self.font.render("START", True, (100, 120, 150))
        self.screen.blit(label, label.get_rect(center=(sx, sy)))

    def draw_goal_3d(self):
        gx, gy, _ = self.project_3d_to_2d(self.goal_position)
        gx, gy = int(gx), int(gy)
        gr = int(self.goal_radius * self.ppu * self.cam_zoom)
        if self.goal_reached:
            color, border = (150, 255, 150), (100, 200, 100)
        else:
            color, border = (255, 230, 180), (220, 180, 100)
        pygame.draw.circle(self.screen, color, (gx, gy), gr)
        pygame.draw.circle(self.screen, border, (gx, gy), gr, 2)
        label = self.font.render("GOAL", True, (120, 100, 60))
        self.screen.blit(label, label.get_rect(center=(gx, gy)))

    def draw_goal_message(self):
        msg = self.font.render("GOAL REACHED! Press R to restart",
                               True, (50, 150, 50))
        rect = msg.get_rect(center=(self.config.width // 2, 60))
        bg = rect.inflate(20, 10)
        pygame.draw.rect(self.screen, (220, 255, 220), bg)
        pygame.draw.rect(self.screen, (100, 200, 100), bg, 2)
        self.screen.blit(msg, rect)

    def draw_autopilot_path(self):
        """Draw autopilot waypoint path as projected dots + lines."""
        wps = self.autopilot_waypoints
        if not wps:
            return

        n = len(wps)
        pts_3d = np.array(wps)
        sx, sy, _dp = self.project_batch(pts_3d)

        for i in range(n - 1):
            p1 = (int(sx[i]), int(sy[i]))
            p2 = (int(sx[i+1]), int(sy[i+1]))
            if i < self.autopilot_wp_idx:
                color = (180, 180, 190)
            else:
                color = (50, 200, 220)
            pygame.draw.line(self.screen, color, p1, p2, 2)

        for i in range(n):
            x, y = int(sx[i]), int(sy[i])
            if i == self.autopilot_wp_idx:
                pygame.draw.circle(self.screen, (255, 100, 50), (x, y), 5)
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 5, 1)
            elif i > self.autopilot_wp_idx:
                pygame.draw.circle(self.screen, (50, 200, 220), (x, y), 3)

    def draw_maze_info(self):
        ap_str = "ON" if self.autopilot_active else "OFF"
        lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"3D Maze ({len(self.walls_3d)} walls, "
            f"{self.n_wall_particles} particles)",
            f"Autopilot: {ap_str}",
            f"Camera: yaw={np.degrees(self.cam_yaw):.0f}\u00b0"
            f" pitch={np.degrees(self.cam_pitch):.0f}\u00b0",
            "",
            f"Yaw: {self.yaw_input:+.2f}",
            f"Pitch: {self.pitch_input:+.2f}",
            f"Roll: {self.roll_input:+.2f}",
            f"Speed: {self.forward_speed:+.2f}",
            "",
            "Controls:",
            "\u2190/\u2192: Yaw  \u2191/\u2193: Pitch  Z/X: Roll",
            "+/-: Speed  P: Autopilot  G: New maze",
            "W: Wall dots  R: Reset  M: Goal",
            "SPACE: Pause  H/I/V/A: GUI",
            "HOME: Reset camera  Q: Quit",
        ]
        y = 10
        for line in lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 22

    # ================================================================
    # Event handling
    # ================================================================

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

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
                        self.cam_pitch = np.clip(
                            self.cam_pitch,
                            -np.pi / 2 + 0.1, np.pi / 2 - 0.1)
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
                self.cam_zoom *= 1.1 if event.y > 0 else 0.9
                self.cam_zoom = np.clip(self.cam_zoom, 0.2, 5.0)

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self._initialize_at_start()
                    self.yaw_input = 0.0
                    self.pitch_input = 0.0
                    self.roll_input = 0.0
                    self.yaw_history[:] = 0.0
                    self.pitch_history[:] = 0.0
                    self.roll_history[:] = 0.0
                    self.history_idx = 0
                    self.autopilot_active = False
                    self.autopilot_waypoints = []
                    print("Reset positions")

                elif event.key == pygame.K_p:
                    if not self.goal_reached:
                        self.autopilot_active = not self.autopilot_active
                        if self.autopilot_active:
                            self.plan_autopilot_path()
                        else:
                            self.yaw_input = 0.0
                            self.pitch_input = 0.0
                            self.roll_input = 0.0
                        print(f"Autopilot: "
                              f"{'ON' if self.autopilot_active else 'OFF'}")

                elif event.key == pygame.K_g:
                    self._load_maze_and_rebuild()
                    print("Generated new 3D random maze")

                elif event.key == pygame.K_m:
                    self.show_goal = not self.show_goal

                elif event.key == pygame.K_w:
                    self.show_wall_particles = not self.show_wall_particles

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

        keys = pygame.key.get_pressed()
        if not self.autopilot_active:
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
        running = True
        while running:
            running = self.handle_events()
            self.step()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()


def main():
    demo = SnakeMazeWall3D(snake_n_species=6, n_particles=15)
    demo.run()


if __name__ == "__main__":
    main()
