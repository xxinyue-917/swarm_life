#!/usr/bin/env python3
"""
Snake Maze Demo — Navigate a snake through maze environments

A chain of particle clusters navigates through a maze with walls.
Arrow keys steer the head, walls block movement.

Controls:
    ←/→:   Steer head left/right
    1-4:   Select maze layout
    R:     Reset positions
    SPACE: Pause/Resume
    G:     Toggle goal marker
    H:     Hide/show all GUI
    I:     Toggle info panel
    Q/ESC: Quit
"""

import heapq
from math import ceil

import pygame
import numpy as np
from snake_demo import SnakeDemo, generate_position_matrix


# =============================================================================
# Wall and Maze Classes
# =============================================================================

class Wall:
    """A rectangular wall obstacle."""

    def __init__(self, x: float, y: float, width: float, height: float):
        """
        Create a wall.

        Args:
            x, y: Bottom-left corner position (in simulation meters)
            width, height: Wall dimensions (in meters)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def contains_point(self, px: float, py: float, margin: float = 0.0) -> bool:
        """Check if a point is inside the wall (with optional margin)."""
        return (self.left - margin <= px <= self.right + margin and
                self.top - margin <= py <= self.bottom + margin)

    def get_collision_response(self, px: float, py: float, vx: float, vy: float,
                                margin: float = 0.05) -> tuple:
        """
        Compute collision response for a particle near/inside the wall.

        Returns:
            (new_x, new_y, new_vx, new_vy) - corrected position and velocity
        """
        new_x, new_y = px, py
        new_vx, new_vy = vx, vy

        # Check if particle is inside wall bounds
        if not self.contains_point(px, py, margin):
            return new_x, new_y, new_vx, new_vy

        # Find nearest edge and push particle out
        dist_left = px - self.left
        dist_right = self.right - px
        dist_top = py - self.top
        dist_bottom = self.bottom - py

        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        if min_dist == dist_left:
            # Push left
            new_x = self.left - margin
            new_vx = -abs(vx)
        elif min_dist == dist_right:
            # Push right
            new_x = self.right + margin
            new_vx = abs(vx)
        elif min_dist == dist_top:
            # Push up (toward top)
            new_y = self.top - margin
            new_vy = -abs(vy)
        else:
            # Push down (toward bottom)
            new_y = self.bottom + margin
            new_vy = abs(vy)

        return new_x, new_y, new_vx, new_vy


def create_maze_simple(sim_width: float, sim_height: float) -> list:
    """
    Simple maze: Single obstacle in center with clear path around.
    Start: left middle, Goal: right middle
    """
    walls = []
    t = 0.2  # wall thickness
    cx, cy = sim_width / 2, sim_height / 2

    # Central block obstacle - snake must go around it
    block_w, block_h = 2.0, 4.0
    walls.append(Wall(cx - block_w/2, cy - block_h/2, block_w, block_h))

    return walls


def create_maze_corridor(sim_width: float, sim_height: float) -> list:
    """
    S-Curve corridor: Smooth path requiring steering.
    Wide enough for snake (2.5m corridors).
    """
    walls = []
    t = 0.2  # wall thickness

    # Upper boundary
    walls.append(Wall(0, 1.0, sim_width, t))
    # Lower boundary
    walls.append(Wall(0, sim_height - 1.0 - t, sim_width, t))

    # First bend: wall from top, gap at bottom
    walls.append(Wall(3.0, 1.0, t, 4.5))

    # Second bend: wall from bottom, gap at top
    walls.append(Wall(5.5, sim_height - 5.5, t, 4.5))

    # Third bend: wall from top, gap at bottom
    walls.append(Wall(8.0, 1.0, t, 4.5))

    return walls


def create_maze_zigzag(sim_width: float, sim_height: float) -> list:
    """
    Zigzag maze: Alternating horizontal barriers with gaps.
    Forces snake to navigate up and down.
    """
    walls = []
    t = 0.2  # wall thickness
    gap = 2.5  # gap width for snake to pass

    # Row 1: wall from left, gap on right
    walls.append(Wall(0.5, 2.5, sim_width - gap - 1.0, t))

    # Row 2: wall from right, gap on left
    walls.append(Wall(gap + 0.5, 5.0, sim_width - gap - 1.0, t))

    # Row 3: wall from left, gap on right
    walls.append(Wall(0.5, 7.5, sim_width - gap - 1.0, t))

    return walls


def create_maze_chambers(sim_width: float, sim_height: float) -> list:
    """
    Chamber maze: Three connected rooms with doorways.
    Clear structure, requires navigation through doorways.
    """
    walls = []
    t = 0.2  # wall thickness
    door_width = 2.5  # doorway width

    cy = sim_height / 2

    # First vertical wall (between room 1 and 2) with door at top
    x1 = 3.5
    walls.append(Wall(x1, cy - door_width/2 + 1.0, t, cy - door_width/2))  # bottom part
    walls.append(Wall(x1, 0.5, t, cy - door_width/2 - 0.5))  # top part... wait let me recalculate

    # Let me think more carefully:
    # Room divider at x=3.5, door in upper half
    door_top = 2.0  # door starts at y=2.0
    door_bottom = door_top + door_width  # door ends at y=4.5
    walls.append(Wall(x1, 0.5, t, door_top - 0.5))  # wall above door
    walls.append(Wall(x1, door_bottom, t, sim_height - door_bottom - 0.5))  # wall below door

    # Second vertical wall (between room 2 and 3) with door at bottom
    x2 = 6.5
    door_top2 = sim_height - door_width - 2.0  # door in lower portion
    door_bottom2 = door_top2 + door_width
    walls.append(Wall(x2, 0.5, t, door_top2 - 0.5))  # wall above door
    walls.append(Wall(x2, door_bottom2, t, sim_height - door_bottom2 - 0.5))  # wall below door

    return walls


def create_maze_z_tunnel(sim_width: float, sim_height: float) -> list:
    """
    Z-shaped tunnel: A confined Z-shaped corridor.

    Layout (10m x 10m):
        ENTRY →═══════════╗
                          ║
                 ╔════════╝
                 ║
                 ╚════════→ EXIT

    The snake must navigate through the Z-shaped tunnel.
    Tunnel width: ~2m for comfortable snake navigation.
    """
    walls = []
    t = 0.2  # wall thickness
    tw = 2.0  # tunnel width

    # Z-tunnel layout:
    # Top corridor:    y = 1.5 to 3.5, x = entry(open) to 7.5
    # Vertical:        x = 5.5 to 7.5, y = 3.5 to 6.5
    # Bottom corridor: y = 6.5 to 8.5, x = 2.5 to exit(open)

    # === TOP CORRIDOR ===
    # Upper wall of top corridor (from left edge to right turn)
    walls.append(Wall(0.0, 1.5, 7.5 + t, t))

    # Lower wall of top corridor (stops before vertical connector)
    walls.append(Wall(0.0, 3.5, 5.5, t))

    # === VERTICAL CONNECTOR ===
    # Left wall of vertical section
    walls.append(Wall(5.5, 3.5 + t, t, 3.0 - t))

    # Right wall of vertical section (connects top to bottom)
    walls.append(Wall(7.5, 1.5 + t, t, 5.0))

    # === BOTTOM CORRIDOR ===
    # Upper wall of bottom corridor (starts after vertical connector)
    walls.append(Wall(2.5, 6.5, 3.0, t))

    # Lower wall of bottom corridor (from left cap to right edge)
    walls.append(Wall(2.5, 8.5, 7.5, t))

    # === ENCLOSURE WALLS ===
    # Left cap of bottom corridor
    walls.append(Wall(2.5, 6.5 + t, t, tw - t))

    # Left boundary wall (connects top corridor to bottom corridor area)
    walls.append(Wall(0.0, 3.5 + t, t, 3.0 - t))

    # Bottom-left corner (connects left boundary to bottom corridor)
    walls.append(Wall(0.0, 6.5 + t, 2.5, t))
    walls.append(Wall(0.0, 8.5, 2.5 + t, t))

    return walls


MAZE_LAYOUTS = {
    1: ("Simple", create_maze_simple),
    2: ("S-Curve", create_maze_corridor),
    3: ("Zigzag", create_maze_zigzag),
    4: ("Chambers", create_maze_chambers),
    5: ("Z-Tunnel", create_maze_z_tunnel),
}


# =============================================================================
# Snake Maze Demo
# =============================================================================

class SnakeMazeDemo(SnakeDemo):
    """
    Snake demo with maze environment.

    Extends SnakeDemo with:
    - Wall obstacles that block particle movement
    - Multiple maze layouts
    - Goal marker for navigation target
    """

    def __init__(self, n_species: int = 6, n_particles: int = 15, maze_id: int = 1):
        super().__init__(n_species=n_species, n_particles=n_particles)

        # Maze state
        self.walls = []
        self.current_maze_id = maze_id

        # Start and goal positions (will be set by load_maze)
        self.start_position = np.array([1.5, self.config.sim_height / 2])
        self.goal_position = np.array([self.config.sim_width - 1.5,
                                        self.config.sim_height / 2])
        self.goal_radius = 0.6  # meters
        self.show_goal = True
        self.goal_reached = False

        # Load maze (this also sets start/goal positions)
        self.load_maze(maze_id)

        # Wall collision margin
        self.wall_margin = 0.08

        # Override window title
        pygame.display.set_caption("Snake Maze Demo — Navigate to Goal")

        # Re-initialize snake at start position
        self._initialize_at_start()

        # --- Autopilot (A* pathfinding) ---
        self.autopilot_active = False
        self.autopilot_path_sim = []       # Raw A* path in sim coords
        self.autopilot_waypoints = []      # Simplified waypoints in sim coords
        self.autopilot_wp_idx = 0          # Current target waypoint index
        self.autopilot_wp_threshold = 0.2  # Meters to consider waypoint reached

        # Pathfinding grid
        self.cell_size = 0.2               # Grid resolution (meters)
        self.pathfinding_margin = 0.35     # Wall clearance for grid cells
        self.occupancy_grid = None
        self.grid_rows = 0
        self.grid_cols = 0
        self.build_occupancy_grid()

        print("=" * 60)
        print("Snake Maze Demo")
        print("=" * 60)
        print(f"Maze: {MAZE_LAYOUTS[maze_id][0]}")
        print("")
        print("Controls:")
        print("  ←/→     Steer left/right")
        print("  ↑/↓     Increase/decrease forward speed")
        print("  1-5     Select maze layout")
        print("  A       Toggle autopilot (A* pathfinding)")
        print("  R       Reset positions")
        print("  G       Toggle goal marker")
        print("  SPACE   Pause")
        print("  Q/ESC   Quit")
        print("=" * 60)

    def load_maze(self, maze_id: int):
        """Load a maze layout with appropriate start/goal positions."""
        if maze_id not in MAZE_LAYOUTS:
            maze_id = 1

        self.current_maze_id = maze_id
        name, generator = MAZE_LAYOUTS[maze_id]
        self.walls = generator(self.config.sim_width, self.config.sim_height)

        # Set start and goal positions based on maze type
        w, h = self.config.sim_width, self.config.sim_height

        if maze_id == 1:  # Simple - center obstacle
            self.start_position = np.array([1.5, h / 2])
            self.goal_position = np.array([w - 1.5, h / 2])
        elif maze_id == 2:  # S-Curve
            self.start_position = np.array([1.5, h / 2])
            self.goal_position = np.array([w - 1.5, h / 2])
        elif maze_id == 3:  # Zigzag
            self.start_position = np.array([1.5, 1.5])
            self.goal_position = np.array([w - 1.5, h - 1.5])
        elif maze_id == 4:  # Chambers
            self.start_position = np.array([1.5, h / 2])
            self.goal_position = np.array([w - 1.5, h / 2])
        elif maze_id == 5:  # Z-Tunnel
            # Start at top-left entry, goal at bottom-right exit
            self.start_position = np.array([1.0, 2.5])  # middle of top corridor (y=1.5 to 3.5)
            self.goal_position = np.array([w - 1.0, 7.5])  # middle of bottom corridor (y=6.5 to 8.5)

        print(f"Loaded maze: {name} ({len(self.walls)} walls)")

    def _initialize_at_start(self):
        """Initialize snake at maze start position."""
        start_x = self.start_position[0]
        start_y = self.start_position[1]

        particles_per_species = self.n // self.n_species

        for i in range(self.n):
            species_id = min(i // particles_per_species, self.n_species - 1)
            self.species[i] = species_id

            # Arrange horizontally from start (head first)
            group_x = start_x + species_id * self.group_spacing * 0.4
            group_y = start_y

            self.positions[i, 0] = group_x + self.rng.uniform(-0.08, 0.08)
            self.positions[i, 1] = group_y + self.rng.uniform(-0.08, 0.08)
            self.velocities[i] = np.array([0.0, 0.0])

        self.goal_reached = False

    def check_goal(self):
        """Check if snake head has reached the goal."""
        if not self.show_goal:
            return

        # Get head centroid (species 0)
        head_mask = self.species == 0
        if not head_mask.any():
            return

        head_centroid = self.positions[head_mask].mean(axis=0)
        dist_to_goal = np.linalg.norm(head_centroid - self.goal_position)

        if dist_to_goal < self.goal_radius and not self.goal_reached:
            self.goal_reached = True
            print("Goal reached!")

    def handle_wall_collisions(self):
        """Handle collisions between particles and walls."""
        for i in range(self.n):
            px, py = self.positions[i]
            vx, vy = self.velocities[i]

            for wall in self.walls:
                if wall.contains_point(px, py, self.wall_margin):
                    new_x, new_y, new_vx, new_vy = wall.get_collision_response(
                        px, py, vx, vy, self.wall_margin
                    )
                    self.positions[i] = np.array([new_x, new_y])
                    self.velocities[i] = np.array([new_vx, new_vy])
                    px, py = new_x, new_y
                    vx, vy = new_vx, new_vy

    # ================================================================
    # A* Autopilot
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
                # Block boundary
                if (cx < boundary or cx > sw - boundary or
                        cy < boundary or cy > sh - boundary):
                    grid[r, c] = True
                    continue
                # Block cells near walls
                for wall in self.walls:
                    if wall.contains_point(cx, cy, self.pathfinding_margin):
                        grid[r, c] = True
                        break
        self.occupancy_grid = grid

    def sim_to_grid(self, x, y):
        """Convert sim coords (meters) to grid cell (row, col)."""
        c = int(x / self.cell_size)
        r = int(y / self.cell_size)
        return (max(0, min(r, self.grid_rows - 1)),
                max(0, min(c, self.grid_cols - 1)))

    def grid_to_sim(self, r, c):
        """Convert grid cell (row, col) to sim coords (center of cell)."""
        return np.array([(c + 0.5) * self.cell_size,
                         (r + 0.5) * self.cell_size])

    def _find_nearest_free(self, r, c):
        """Find the nearest free grid cell to (r, c) via BFS."""
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
        return (r, c)  # fallback

    def astar_pathfind(self, start_sim, goal_sim):
        """A* search from start to goal in sim coordinates.

        Returns list of (row, col) grid cells from start to goal,
        or empty list if no path found.
        """
        sr, sc = self.sim_to_grid(start_sim[0], start_sim[1])
        gr, gc = self.sim_to_grid(goal_sim[0], goal_sim[1])

        # Snap to nearest free cell if start/goal is inside a wall
        sr, sc = self._find_nearest_free(sr, sc)
        gr, gc = self._find_nearest_free(gr, gc)

        start = (sr, sc)
        goal = (gr, gc)

        if start == goal:
            return [start]

        # Octile heuristic
        def h(a):
            dr = abs(a[0] - goal[0])
            dc = abs(a[1] - goal[1])
            return max(dr, dc) + (1.414 - 1.0) * min(dr, dc)

        # 8-directional neighbors
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
                # Reconstruct path
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
                # Prevent corner cutting for diagonals
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

        return []  # No path found

    def line_of_sight(self, cell_a, cell_b):
        """Check if all cells on the Bresenham line are free."""
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
        """Reduce raw grid path to sparse waypoints via line-of-sight thinning."""
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
        """Plan A* path from current head position to goal."""
        self.build_occupancy_grid()

        head_mask = self.species == 0
        head_centroid = self.positions[head_mask].mean(axis=0)

        raw_path = self.astar_pathfind(head_centroid, self.goal_position)
        if not raw_path:
            print("Autopilot: No path found!")
            self.autopilot_active = False
            self.autopilot_path_sim = []
            self.autopilot_waypoints = []
            return

        self.autopilot_path_sim = [self.grid_to_sim(r, c) for r, c in raw_path]

        simplified = self.simplify_path(raw_path)
        self.autopilot_waypoints = [self.grid_to_sim(r, c) for r, c in simplified]
        self.autopilot_wp_idx = 0

        print(f"Autopilot: Path planned — {len(self.autopilot_waypoints)} waypoints "
              f"(from {len(raw_path)} grid cells)")

    def get_head_heading(self):
        """Get unit heading vector of the snake's head (species 0 → species 1 direction)."""
        head_mask = self.species == 0
        neck_mask = self.species == 1
        head_centroid = self.positions[head_mask].mean(axis=0)
        neck_centroid = self.positions[neck_mask].mean(axis=0)
        heading = neck_centroid - head_centroid
        norm = np.linalg.norm(heading)
        if norm < 1e-6:
            return np.array([1.0, 0.0])
        return heading / norm

    def update_autopilot(self):
        """Compute and apply autopilot steering signals."""
        if not self.autopilot_waypoints:
            return

        # Head position
        head_mask = self.species == 0
        head_centroid = self.positions[head_mask].mean(axis=0)

        # Check waypoint reached
        current_wp = self.autopilot_waypoints[self.autopilot_wp_idx]
        dist_to_wp = np.linalg.norm(head_centroid - current_wp)

        if dist_to_wp < self.autopilot_wp_threshold:
            if self.autopilot_wp_idx < len(self.autopilot_waypoints) - 1:
                self.autopilot_wp_idx += 1
                current_wp = self.autopilot_waypoints[self.autopilot_wp_idx]
                dist_to_wp = np.linalg.norm(head_centroid - current_wp)
            else:
                # Final waypoint reached
                self.turn_input = 0.0
                return

        # Desired direction
        desired = current_wp - head_centroid
        desired_norm = np.linalg.norm(desired)
        if desired_norm < 1e-6:
            return
        desired_dir = desired / desired_norm

        # Current heading
        heading = self.get_head_heading()

        # Full signed angular error via atan2 (handles 180° correctly)
        cross = heading[0] * desired_dir[1] - heading[1] * desired_dir[0]
        dot = heading[0] * desired_dir[0] + heading[1] * desired_dir[1]
        angular_error = np.arctan2(cross, dot)  # range [-pi, pi]

        # Nudge turn_input like pressing arrow keys — small increments per frame
        # Let the existing turn_decay handle smoothing
        nudge = 0.03
        dead_zone = 0.1
        if angular_error > dead_zone:
            # Target is to the left — nudge left
            self.turn_input = max(-1.0, self.turn_input - nudge)
        elif angular_error < -dead_zone:
            # Target is to the right — nudge right
            self.turn_input = min(1.0, self.turn_input + nudge)
        # Otherwise: do nothing, let turn_decay bring it back to 0

    def draw_autopilot_path(self):
        """Draw the A* path and waypoints on screen."""
        # Waypoint path (straight lines between waypoints)
        if len(self.autopilot_waypoints) >= 2:
            points = [self.to_screen(wp) for wp in self.autopilot_waypoints]
            pygame.draw.lines(self.screen, (180, 190, 210), False, points, 2)

        # Waypoints
        for i, wp in enumerate(self.autopilot_waypoints):
            sx, sy = self.to_screen(wp)
            if i < self.autopilot_wp_idx:
                # Past (green)
                pygame.draw.circle(self.screen, (150, 220, 150), (sx, sy), 5)
            elif i == self.autopilot_wp_idx:
                # Current target (red with ring)
                pygame.draw.circle(self.screen, (255, 200, 200), (sx, sy), 12, 2)
                pygame.draw.circle(self.screen, (255, 80, 80), (sx, sy), 6)
            else:
                # Future (yellow)
                pygame.draw.circle(self.screen, (240, 200, 80), (sx, sy), 5)

        # Line from head to current waypoint
        if self.autopilot_waypoints and self.autopilot_wp_idx < len(self.autopilot_waypoints):
            head_mask = self.species == 0
            head_centroid = self.positions[head_mask].mean(axis=0)
            head_screen = self.to_screen(head_centroid)
            wp_screen = self.to_screen(self.autopilot_waypoints[self.autopilot_wp_idx])
            pygame.draw.line(self.screen, (255, 120, 120), head_screen, wp_screen, 2)

        # "AUTOPILOT" badge
        label = self.font.render("AUTOPILOT", True, (50, 120, 50))
        bg_rect = label.get_rect(topright=(self.config.width - 10, self.config.height - 35))
        bg = bg_rect.inflate(12, 6)
        pygame.draw.rect(self.screen, (220, 255, 220), bg)
        pygame.draw.rect(self.screen, (100, 200, 100), bg, 2)
        self.screen.blit(label, bg_rect)

    def step(self):
        """Perform one simulation step with wall collisions."""
        if self.paused:
            return

        # Apply input decay
        self.turn_input *= self.turn_decay
        if abs(self.turn_input) < 0.01:
            self.turn_input = 0.0

        # Autopilot overrides turn_input before it reaches K_rot
        if self.autopilot_active:
            self.update_autopilot()

        # Update K_rot from input
        self.update_matrices_from_input()

        # Call grandparent step for physics (skip SnakeDemo.step to avoid double decay)
        # This calls ParticleLife.step()
        from particle_life import ParticleLife
        ParticleLife.step(self)

        # Handle wall collisions
        self.handle_wall_collisions()

        # Check goal
        self.check_goal()

        # Disengage autopilot when goal reached
        if self.goal_reached and self.autopilot_active:
            self.autopilot_active = False
            self.turn_input = 0.0
            print("Autopilot: Goal reached, disengaging.")

    def draw(self):
        """Draw simulation with maze and goal."""
        self.screen.fill((250, 250, 252))  # Slightly off-white background

        # Draw start zone
        if self.show_goal:
            self.draw_start()

        # Draw walls
        self.draw_walls()

        # Draw autopilot path (between walls and goal layers)
        if self.autopilot_active and self.autopilot_waypoints:
            self.draw_autopilot_path()

        # Draw goal
        if self.show_goal:
            self.draw_goal()

        # Draw particles
        self.draw_particles()

        if self.hide_gui:
            return

        # Draw centroid spine and markers
        if self.show_centroids:
            pts = self.draw_centroid_spine(line_width=2)
            self.draw_centroid_markers(pts, head_r=8, tail_r=5)

        # Control indicator
        self.draw_control_indicator()

        # Info panel
        if self.show_info:
            self.draw_maze_info()

        # Goal reached message
        if self.goal_reached:
            self.draw_goal_message()

    def draw_walls(self):
        """Draw maze walls with nice styling."""
        wall_color = (70, 80, 100)
        border_color = (50, 55, 70)

        for wall in self.walls:
            x = int(wall.x * self.ppu * self.zoom)
            y = int(wall.y * self.ppu * self.zoom)
            w = max(1, int(wall.width * self.ppu * self.zoom))
            h = max(1, int(wall.height * self.ppu * self.zoom))

            # Draw wall with border
            pygame.draw.rect(self.screen, wall_color, (x, y, w, h))
            pygame.draw.rect(self.screen, border_color, (x, y, w, h), 2)

    def draw_start(self):
        """Draw start zone marker."""
        sx = int(self.start_position[0] * self.ppu * self.zoom)
        sy = int(self.start_position[1] * self.ppu * self.zoom)
        sr = int(0.6 * self.ppu * self.zoom)

        # Start circle (light blue)
        pygame.draw.circle(self.screen, (200, 220, 255), (sx, sy), sr)
        pygame.draw.circle(self.screen, (150, 180, 220), (sx, sy), sr, 2)

        # Start label
        label = self.font.render("START", True, (100, 120, 150))
        label_rect = label.get_rect(center=(sx, sy))
        self.screen.blit(label, label_rect)

    def draw_goal(self):
        """Draw goal marker."""
        gx = int(self.goal_position[0] * self.ppu * self.zoom)
        gy = int(self.goal_position[1] * self.ppu * self.zoom)
        gr = int(self.goal_radius * self.ppu * self.zoom)

        # Goal circle
        if self.goal_reached:
            color = (150, 255, 150)
            border = (100, 200, 100)
        else:
            color = (255, 230, 180)
            border = (220, 180, 100)

        pygame.draw.circle(self.screen, color, (gx, gy), gr)
        pygame.draw.circle(self.screen, border, (gx, gy), gr, 2)

        # Goal label
        label = self.font.render("GOAL", True, (120, 100, 60))
        label_rect = label.get_rect(center=(gx, gy))
        self.screen.blit(label, label_rect)

    def draw_goal_message(self):
        """Draw goal reached message."""
        msg = self.font.render("GOAL REACHED! Press R to restart", True, (50, 150, 50))
        rect = msg.get_rect(center=(self.config.width // 2, 60))

        # Background
        bg_rect = rect.inflate(20, 10)
        pygame.draw.rect(self.screen, (220, 255, 220), bg_rect)
        pygame.draw.rect(self.screen, (100, 200, 100), bg_rect, 2)

        self.screen.blit(msg, rect)

    def draw_maze_info(self):
        """Draw information panel."""
        maze_name = MAZE_LAYOUTS[self.current_maze_id][0]

        auto_status = "ON" if self.autopilot_active else "OFF"
        info_lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Maze: {maze_name}",
            f"Autopilot: {auto_status}",
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
            "1-5: Maze | A: Autopilot",
            "R: Reset | G: Goal | V: Centroids",
            "SPACE: Pause | Q: Quit",
        ]

        y = 10
        for line in info_lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 22

        self.draw_pause_indicator()

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
                    self._initialize_at_start()
                    self.turn_input = 0.0
                    self.turn_history[:] = 0.0
                    self.history_idx = 0
                    if self.autopilot_active:
                        self.plan_autopilot_path()
                    print("Reset positions")

                elif event.key == pygame.K_g:
                    self.show_goal = not self.show_goal
                    print(f"Show goal: {self.show_goal}")

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

                # Maze selection (1-5)
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3,
                                   pygame.K_4, pygame.K_5):
                    maze_num = event.key - pygame.K_0
                    self.load_maze(maze_num)
                    self._initialize_at_start()
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
    demo = SnakeMazeDemo(n_species=6, n_particles=15, maze_id=1)
    demo.run()


if __name__ == "__main__":
    main()
