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

        print("=" * 60)
        print("Snake Maze Demo")
        print("=" * 60)
        print(f"Maze: {MAZE_LAYOUTS[maze_id][0]}")
        print("")
        print("Controls:")
        print("  ←/→     Steer left/right")
        print("  ↑/↓     Increase/decrease forward speed")
        print("  1-5     Select maze layout")
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

    def step(self):
        """Perform one simulation step with wall collisions."""
        if self.paused:
            return

        # Apply input decay
        self.turn_input *= self.turn_decay
        if abs(self.turn_input) < 0.01:
            self.turn_input = 0.0

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

    def draw(self):
        """Draw simulation with maze and goal."""
        self.screen.fill((250, 250, 252))  # Slightly off-white background

        # Draw start zone
        if self.show_goal:
            self.draw_start()

        # Draw walls
        self.draw_walls()

        # Draw goal
        if self.show_goal:
            self.draw_goal()

        # Draw particles
        self.draw_particles()

        if self.hide_gui:
            return

        # Draw centroid spine and markers
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

        info_lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Maze: {maze_name}",
            "",
            f"Turn: {self.turn_input:+.2f}",
            f"Speed: {self.forward_speed:+.2f}",
            "",
            "Controls:",
            "←/→: Steer | ↑/↓: Speed",
            "1-4: Maze layout",
            "R: Reset | G: Goal",
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
                    print("Reset positions")

                elif event.key == pygame.K_g:
                    self.show_goal = not self.show_goal
                    print(f"Show goal: {self.show_goal}")

                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info

                elif event.key == pygame.K_h:
                    self.hide_gui = not self.hide_gui

                # Maze selection (1-5)
                elif event.key == pygame.K_1:
                    self.load_maze(1)
                    self._initialize_at_start()
                elif event.key == pygame.K_2:
                    self.load_maze(2)
                    self._initialize_at_start()
                elif event.key == pygame.K_3:
                    self.load_maze(3)
                    self._initialize_at_start()
                elif event.key == pygame.K_4:
                    self.load_maze(4)
                    self._initialize_at_start()
                elif event.key == pygame.K_5:
                    self.load_maze(5)
                    self._initialize_at_start()

                # Forward speed adjustment
                elif event.key == pygame.K_UP:
                    self.update_forward_speed(0.05)
                elif event.key == pygame.K_DOWN:
                    self.update_forward_speed(-0.05)

        # Continuous arrow key input for steering
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
