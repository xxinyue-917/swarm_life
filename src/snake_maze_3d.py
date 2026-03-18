#!/usr/bin/env python3
"""
3D Snake Maze Demo — Navigate a snake through a true 3D maze

Walls are solid geometric boxes with collision detection. The maze is
generated using 3D recursive backtracking (DFS) with passages winding
through all three dimensions.

Controls:
    Arrow keys: Yaw/Pitch steering
    Z/X:   Roll steering
    +/-:   Forward speed
    P:     Toggle autopilot (A* pathfinding)
    G:     Generate new random maze
    R:     Reset positions
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
import random as _random
from math import ceil

import pygame
import pygame.gfxdraw
import numpy as np
from particle_life_3d import Config3D, ParticleLife3D
from snake_demo import generate_position_matrix
from snake_demo_3d import SnakeDemo3D


# =============================================================================
# 3D Wall
# =============================================================================

class Wall3D:
    """A 3D axis-aligned box obstacle."""

    def __init__(self, x, y, z, width, height, depth):
        self.x, self.y, self.z = x, y, z
        self.width, self.height, self.depth = width, height, depth

    def contains_point(self, px, py, pz, margin=0.0):
        return (self.x - margin <= px <= self.x + self.width + margin and
                self.y - margin <= py <= self.y + self.height + margin and
                self.z - margin <= pz <= self.z + self.depth + margin)

    def get_collision_response(self, px, py, pz, vx, vy, vz, margin=0.05):
        if not self.contains_point(px, py, pz, margin):
            return px, py, pz, vx, vy, vz

        # Find nearest face and push out
        bounds_lo = [self.x, self.y, self.z]
        bounds_hi = [self.x + self.width, self.y + self.height, self.z + self.depth]
        pos = [px, py, pz]
        vel = [vx, vy, vz]

        best_dim, best_sign, best_dist = 0, -1, float('inf')
        for dim in range(3):
            d_lo = pos[dim] - bounds_lo[dim]
            d_hi = bounds_hi[dim] - pos[dim]
            if d_lo < best_dist:
                best_dist, best_dim, best_sign = d_lo, dim, -1
            if d_hi < best_dist:
                best_dist, best_dim, best_sign = d_hi, dim, 1

        if best_sign == -1:
            pos[best_dim] = bounds_lo[best_dim] - margin
            vel[best_dim] = -abs(vel[best_dim])
        else:
            pos[best_dim] = bounds_hi[best_dim] + margin
            vel[best_dim] = abs(vel[best_dim])

        return pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]


# Colors for wall orientations (RGBA — alpha set at draw time)
ORIENT_FILL = {
    'x': (100, 150, 220),   # Blue - vertical walls
    'y': (60, 150, 80),     # Darker green - floors/ceilings (Y-perpendicular)
    'z': (100, 150, 220),   # Blue - vertical walls (same as X)
}
ORIENT_BORDER = {
    'x': (60, 100, 170),
    'y': (30, 100, 50),
    'z': (60, 100, 170),    # Same as X
}

# Slice mode settings — show walls at the snake's current Y-level
SLICE_THICKNESS = 0.5   # Only show walls within this Y-range of the snake


# =============================================================================
# 3D Maze Generator
# =============================================================================

def create_maze_3d_random(sim_w, sim_h, sim_d, cols=3, rows=3, layers=3):
    """True 3D perfect maze using recursive backtracking (DFS).

    Generates passages winding through all three dimensions.
    Returns list of Wall3D objects (thin slabs on cell faces).
    Start cell: (0,0,0). Goal cell: (cols-1, rows-1, layers-1).
    """
    cell_w = sim_w / cols
    cell_h = sim_h / rows
    cell_d = sim_d / layers
    t = 0.15  # wall thickness

    # Track removed walls for each cell
    removed = [[[set() for _ in range(cols)]
                for _ in range(rows)]
               for _ in range(layers)]

    # DFS maze generation
    visited = [[[False] * cols for _ in range(rows)] for _ in range(layers)]
    stack = [(0, 0, 0)]  # (row, col, layer)
    visited[0][0][0] = True

    directions = {
        'N': (1, 0, 0), 'S': (-1, 0, 0),
        'E': (0, 1, 0), 'W': (0, -1, 0),
        'U': (0, 0, 1), 'D': (0, 0, -1),
    }
    opposite = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E', 'U': 'D', 'D': 'U'}

    while stack:
        r, c, l = stack[-1]
        neighbors = []
        for d, (dr, dc, dl) in directions.items():
            nr, nc, nl = r + dr, c + dc, l + dl
            if 0 <= nr < rows and 0 <= nc < cols and 0 <= nl < layers:
                if not visited[nl][nr][nc]:
                    neighbors.append((d, nr, nc, nl))

        if neighbors:
            d, nr, nc, nl = _random.choice(neighbors)
            removed[l][r][c].add(d)
            removed[nl][nr][nc].add(opposite[d])
            visited[nl][nr][nc] = True
            stack.append((nr, nc, nl))
        else:
            stack.pop()

    # Convert remaining walls to Wall3D objects
    walls = []

    for l in range(layers):
        for r in range(rows):
            for c in range(cols):
                x0 = c * cell_w
                y0 = r * cell_h
                z0 = l * cell_d

                # South wall (Y = y0) — only for row 0 boundary
                if r == 0 and 'S' not in removed[l][r][c]:
                    walls.append(Wall3D(x0, y0 - t / 2, z0, cell_w, t, cell_d))

                # North wall (Y = y0 + cell_h)
                if r == rows - 1:
                    if 'N' not in removed[l][r][c]:
                        walls.append(Wall3D(x0, y0 + cell_h - t / 2, z0,
                                            cell_w, t, cell_d))
                elif 'N' not in removed[l][r][c]:
                    walls.append(Wall3D(x0, y0 + cell_h - t / 2, z0,
                                        cell_w, t, cell_d))

                # West wall (X = x0) — only for col 0 boundary
                if c == 0 and 'W' not in removed[l][r][c]:
                    walls.append(Wall3D(x0 - t / 2, y0, z0, t, cell_h, cell_d))

                # East wall (X = x0 + cell_w)
                if c == cols - 1:
                    if 'E' not in removed[l][r][c]:
                        walls.append(Wall3D(x0 + cell_w - t / 2, y0, z0,
                                            t, cell_h, cell_d))
                elif 'E' not in removed[l][r][c]:
                    walls.append(Wall3D(x0 + cell_w - t / 2, y0, z0,
                                        t, cell_h, cell_d))

                # Down wall (Z = z0) — only for layer 0 boundary
                if l == 0 and 'D' not in removed[l][r][c]:
                    walls.append(Wall3D(x0, y0, z0 - t / 2, cell_w, cell_h, t))

                # Up wall (Z = z0 + cell_d)
                if l == layers - 1:
                    if 'U' not in removed[l][r][c]:
                        walls.append(Wall3D(x0, y0, z0 + cell_d - t / 2,
                                            cell_w, cell_h, t))
                elif 'U' not in removed[l][r][c]:
                    walls.append(Wall3D(x0, y0, z0 + cell_d - t / 2,
                                        cell_w, cell_h, t))

    return walls


# =============================================================================
# SnakeMaze3D
# =============================================================================

class SnakeMaze3D(SnakeDemo3D):
    """3D Snake with solid geometric maze walls and collision detection."""

    def __init__(self, n_species: int = 6, n_particles: int = 15):
        # Build chain K_pos
        k_pos = generate_position_matrix(n_species)
        zeros = np.zeros((n_species, n_species)).tolist()

        sim_w, sim_h, sim_d = 10.0, 10.0, 10.0

        config = Config3D(
            n_species=n_species,
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

        # Call ParticleLife3D directly (skip SnakeDemo3D to set workspace)
        ParticleLife3D.__init__(self, config, headless=False)

        # Replicate SnakeDemo3D control state
        self.yaw_input = 0.0
        self.pitch_input = 0.0
        self.roll_input = 0.0
        self.base_k_rot = 0.08
        self.forward_speed = 0.25
        self.turn_decay = 0.92

        self.joint_delay = 8
        n_joints = n_species - 1
        history_len = self.joint_delay * n_joints + 1
        self.yaw_history = np.zeros(history_len)
        self.pitch_history = np.zeros(history_len)
        self.roll_history = np.zeros(history_len)
        self.history_idx = 0

        self.group_spacing = 0.8
        self.hide_gui = False
        self.show_centroids = True

        # Maze state
        self.walls_3d = []
        self.wall_margin = 0.08

        # Goal
        self.goal_radius = 0.6
        self.show_goal = True
        self.goal_reached = False
        self.start_position = np.array([1.0, 1.0, sim_d / 2])
        self.goal_position = np.array([sim_w - 1.0, sim_h - 1.0, sim_d / 2])

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
        # Stall detection
        self._autopilot_best_dist = float('inf')
        self._autopilot_stall_counter = 0

        # Translucent wall overlay surface (reused each frame)
        self._wall_overlay = pygame.Surface(
            (config.width, config.height), pygame.SRCALPHA)

        # Load maze and place snake
        self.load_maze()
        self._initialize_at_start()

        pygame.display.set_caption("3D Snake Maze \u2014 Navigate to Goal")

        print("=" * 60)
        print("3D Snake Maze Demo")
        print("=" * 60)
        print(f"Space: {sim_w}x{sim_h}x{sim_d}m  |  {len(self.walls_3d)} walls")
        print("")
        print("Controls:")
        print("  \u2190/\u2192       Yaw (horizontal turn)")
        print("  \u2191/\u2193       Pitch (vertical turn)")
        print("  Z/X       Roll (barrel roll)")
        print("  +/-       Forward speed")
        print("  P         Toggle autopilot (A* pathfinding)")
        print("  G         Generate new random maze")
        print("  R         Reset positions")

        print("  M         Toggle goal marker")
        print("  SPACE     Pause")
        print("  H/I/V/A   GUI toggles")
        print("  HOME      Reset camera  Q: Quit")
        print("=" * 60)

    # ================================================================
    # Maze loading
    # ================================================================

    def load_maze(self):
        w = self.config.sim_width
        h = self.config.sim_height
        d = self.config.sim_depth

        self.walls_3d = create_maze_3d_random(w, h, d)

        cell_w = w / 3
        cell_h = h / 3
        cell_d = d / 3
        self.start_position = np.array([cell_w / 2, cell_h / 2, cell_d / 2])
        self.goal_position = np.array([w - cell_w / 2, h - cell_h / 2,
                                       d - cell_d / 2])
        self.goal_reached = False
        self.autopilot_active = False
        self.autopilot_waypoints = []
        print(f"Loaded 3D maze ({len(self.walls_3d)} walls)")

    def _initialize_at_start(self):
        sx, sy, sz = self.start_position
        particles_per_species = self.n // self.n_species

        # Fit chain within the first cell — compute max spacing that stays inside
        cell_w = self.config.sim_width / 3
        margin = 0.3
        max_chain_len = cell_w - 2 * margin  # usable width within cell
        spacing = min(self.group_spacing * 0.4,
                      max_chain_len / max(1, self.n_species - 1))

        for i in range(self.n):
            species_id = min(i // particles_per_species, self.n_species - 1)
            self.species[i] = species_id
            # Center the chain within the cell
            chain_len = (self.n_species - 1) * spacing
            start_x = sx - chain_len / 2
            group_x = start_x + species_id * spacing
            self.positions[i, 0] = group_x + self.rng.uniform(-0.08, 0.08)
            self.positions[i, 1] = sy + self.rng.uniform(-0.08, 0.08)
            self.positions[i, 2] = sz + self.rng.uniform(-0.08, 0.08)
            self.velocities[i] = [0.0, 0.0, 0.0]

        self.goal_reached = False

    # ================================================================
    # Wall collision
    # ================================================================

    def handle_wall_collisions(self):
        for i in range(self.n):
            px, py, pz = self.positions[i]
            vx, vy, vz = self.velocities[i]
            for wall in self.walls_3d:
                if wall.contains_point(px, py, pz, self.wall_margin):
                    px, py, pz, vx, vy, vz = wall.get_collision_response(
                        px, py, pz, vx, vy, vz, self.wall_margin)
                    self.positions[i] = [px, py, pz]
                    self.velocities[i] = [vx, vy, vz]

    # ================================================================
    # Goal
    # ================================================================

    def check_goal(self):
        if not self.show_goal:
            return
        head_mask = self.species == 0
        if not head_mask.any():
            return
        head = self.positions[head_mask].mean(axis=0)
        if np.linalg.norm(head - self.goal_position) < self.goal_radius:
            if not self.goal_reached:
                self.goal_reached = True
                self.autopilot_active = False
                print("Goal reached!")

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
                    # Boundary check
                    if (cx < boundary or cx > sw - boundary or
                            cy < boundary or cy > sh - boundary or
                            cz < boundary or cz > sd - boundary):
                        grid[l, r, c] = True
                        continue
                    # Wall check
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
            for dl, dr, dc in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
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
        """3D A* pathfinding on occupancy grid.

        Uses 6-connected neighbors (face-adjacent cells).
        Returns list of (layer, row, col) grid coordinates.
        """
        sl, sr, sc = self.sim_to_grid(*start_sim)
        gl, gr, gc = self.sim_to_grid(*goal_sim)
        sl, sr, sc = self._find_nearest_free_3d(sl, sr, sc)
        gl, gr, gc = self._find_nearest_free_3d(gl, gr, gc)

        start = (sl, sr, sc)
        goal = (gl, gr, gc)
        if start == goal:
            return [start]

        def h(node):
            return abs(node[0] - goal[0]) + abs(node[1] - goal[1]) + abs(node[2] - goal[2])

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
                    heapq.heappush(open_set,
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

        head_mask = self.species == 0
        head_centroid = self.positions[head_mask].mean(axis=0)

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
        print(f"Autopilot: Path planned \u2014 {len(self.autopilot_waypoints)} waypoints "
              f"(from {len(raw_path)} grid cells)")

    def get_chain_backward(self):
        """Get backward direction along chain (neck − head), normalized."""
        head_mask = self.species == 0
        neck_mask = self.species == 1
        head_centroid = self.positions[head_mask].mean(axis=0)
        neck_centroid = self.positions[neck_mask].mean(axis=0)
        backward = neck_centroid - head_centroid
        norm = np.linalg.norm(backward)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return backward / norm

    def update_autopilot(self):
        """Steer yaw/pitch/roll toward the next waypoint using all three axes.

        Physics: the force on the head from the neck for rotation axis_i is
            F = k_rot * cross(backward, axis_i)
        where k_rot = -base_k_rot * input.  So positive input creates a
        force on the head in the direction  -cross(backward, axis_i).

        For each axis, we compute how much -cross(backward, axis_i) aligns
        with the steering direction.  Positive leverage → positive input.

        Key features:
        - Anti-parallel escape: when forward points away from target, steer
          toward a perpendicular direction to initiate a U-turn.
        - Speed modulation: slow down when turning to tighten the radius.
        - Stall recovery: replan path if stuck for 800+ steps.
        """
        if not self.autopilot_waypoints:
            return

        head_mask = self.species == 0
        head_centroid = self.positions[head_mask].mean(axis=0)

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

        # Chain backward direction (neck − head) determines lever arms
        backward = self.get_chain_backward()
        alignment = np.dot(backward, desired_dir)

        # Determine effective steering direction
        if alignment > 0.5:
            # Forward is pointing AWAY from target — need U-turn.
            # Steer toward a perpendicular escape direction.
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

        # For each rotation axis, compute lever direction and apply input.
        axes = [
            ('yaw',   np.array([0.0, 1.0, 0.0])),   # Y axis
            ('pitch', np.array([1.0, 0.0, 0.0])),   # X axis
            ('roll',  np.array([0.0, 0.0, 1.0])),   # Z axis
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
                self.yaw_input = np.clip(self.yaw_input + delta, -1.0, 1.0)
            elif name == 'pitch':
                self.pitch_input = np.clip(self.pitch_input + delta, -1.0, 1.0)
            elif name == 'roll':
                self.roll_input = np.clip(self.roll_input + delta, -1.0, 1.0)

    # ================================================================
    # Step
    # ================================================================

    def step(self):
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
        ParticleLife3D.step(self)
        self.handle_wall_collisions()
        self.check_goal()

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

        self.draw_walls_3d()

        if self.autopilot_active and self.autopilot_waypoints:
            self.draw_autopilot_path()

        # Particles on top
        screen_x, screen_y, depth = self.project_batch(self.positions)
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
    # Wall helpers
    # ================================================================

    def _wall_orient(self, wall):
        """Return 'x', 'y', or 'z' for the wall's thin dimension."""
        if wall.width <= wall.height and wall.width <= wall.depth:
            return 'x'
        if wall.height <= wall.width and wall.height <= wall.depth:
            return 'y'
        return 'z'

    def _wall_main_face(self, wall):
        """Get 4 corners of the wall's main face (perpendicular to thin axis)."""
        o = self._wall_orient(wall)
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

    # ================================================================
    # Wall drawing — translucent panels (see-through structure)
    # ================================================================

    def draw_walls_3d(self):
        """Draw walls at the snake's current Y-level (horizontal slice).

        Only walls intersecting a horizontal slab around the snake head
        are shown, giving a clear floor-plan view of the current layer.
        Vertical walls are drawn solid; floors/ceilings are faint.
        """
        if not self.walls_3d:
            return

        head_mask = self.species == 0
        head_pos = self.positions[head_mask].mean(axis=0)
        snake_y = head_pos[1]

        margin = 0.5  # skip boundary floor/ceiling walls
        sim_h = self.config.sim_height

        visible = []
        for wall in self.walls_3d:
            # Skip top/bottom boundary walls (floors/ceilings at edges)
            if self._wall_orient(wall) == 'y':
                cy = wall.y + wall.height / 2
                if cy < margin or cy > sim_h - margin:
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
            all_corners[i*4:(i+1)*4] = self._wall_main_face(wall)

        sx, sy, dp = self.project_batch(all_corners)
        self._wall_overlay.fill((0, 0, 0, 0))

        faces = []
        for i, (wall, _) in enumerate(visible):
            pts = [(int(sx[i*4+j]), int(sy[i*4+j])) for j in range(4)]
            avg_d = dp[i*4:(i+1)*4].mean()
            orient = self._wall_orient(wall)
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
    # Autopilot path drawing
    # ================================================================

    def draw_autopilot_path(self):
        """Draw autopilot waypoint path as projected dots + lines."""
        wps = self.autopilot_waypoints
        if not wps:
            return

        n = len(wps)
        pts_3d = np.array(wps)
        sx, sy, _dp = self.project_batch(pts_3d)

        # Draw path lines
        for i in range(n - 1):
            p1 = (int(sx[i]), int(sy[i]))
            p2 = (int(sx[i+1]), int(sy[i+1]))
            if i < self.autopilot_wp_idx:
                color = (180, 180, 190)  # Already passed — dim
            else:
                color = (50, 200, 220)   # Ahead — cyan
            pygame.draw.line(self.screen, color, p1, p2, 2)

        # Draw waypoint dots
        for i in range(n):
            x, y = int(sx[i]), int(sy[i])
            if i == self.autopilot_wp_idx:
                # Current target — bright
                pygame.draw.circle(self.screen, (255, 100, 50), (x, y), 5)
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 5, 1)
            elif i > self.autopilot_wp_idx:
                pygame.draw.circle(self.screen, (50, 200, 220), (x, y), 3)

    # ================================================================
    # HUD elements
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

    def draw_maze_info(self):
        ap_str = "ON" if self.autopilot_active else "OFF"
        lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"3D Maze ({len(self.walls_3d)} walls)",
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
            "R: Reset  M: Goal  SPACE: Pause  Q: Quit",
            "H/I/V/A: GUI  HOME: Camera",
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
                            self.cam_pitch, -np.pi / 2 + 0.1, np.pi / 2 - 0.1)
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
                    self.yaw_input = self.pitch_input = self.roll_input = 0.0
                    self.yaw_history[:] = 0.0
                    self.pitch_history[:] = 0.0
                    self.roll_history[:] = 0.0
                    self.history_idx = 0
                    self.autopilot_active = False
                    self.autopilot_waypoints = []
                    print("Reset positions")

                elif event.key == pygame.K_g:
                    self.load_maze()
                    self._initialize_at_start()
                    print("Generated new 3D random maze")

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

                elif event.key == pygame.K_m:
                    self.show_goal = not self.show_goal

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
    demo = SnakeMaze3D(n_species=6, n_particles=15)
    demo.run()


if __name__ == "__main__":
    main()
