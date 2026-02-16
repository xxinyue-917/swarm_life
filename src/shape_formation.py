#!/usr/bin/env python3
"""
Multi-Species Manual Control Demo

Demonstrates manual control of N species (2-10) arranged side-by-side.
Arrow keys control swarm movement:
- Left/Right: Turn the swarm
- Up/Down: Adjust speed

Two control modes (toggle with C):
  (A) Manual blend mode: Arrow keys blend translation/rotation K_rot.
  (B) Shape control mode: PD controller tracks desired joint-angle profile.

Controls (both modes):
    ←/→: Turn left/right (manual) / heading bias (control)
    ↑/↓: Speed up/slow down
    +/-: Add/remove species (2-10)
    R: Reset positions
    SPACE: Pause/Resume
    I: Toggle info panel
    Q/ESC: Quit

Control mode extra keys:
    C:     Toggle control mode ON/OFF
    1/2/3/4: Select target profile (STRAIGHT/U/M/HUG)
    [/]:   Decrease/increase phi0 (target curvature)
    K/J:   Increase/decrease kp
    D/F:   Decrease/increase kd
"""

import pygame
import numpy as np
from particle_life import Config, ParticleLife


def generate_translation_matrix(n_species: int, strength: float) -> np.ndarray:
    """
    Generate K_rot matrix for forward translation.

    Only adjacent species pairs are coupled (like joints in a chain).
    Antisymmetric: K[i, i+1] = +strength, K[i+1, i] = -strength
    This creates net forward motion along the chain.
    """
    K = np.zeros((n_species, n_species))

    for i in range(n_species):
        deg = (1 if i > 0 else 0) + (1 if i < n_species - 1 else 0)
        if deg == 0:
            continue

        if i + 1 < n_species:
            K[i, i + 1] = +strength / deg
        if i - 1 >= 0:
            K[i, i - 1] = -strength / deg

    return K


def generate_rotation_matrix(n_species: int, strength: float) -> np.ndarray:
    """
    Generate K_rot matrix for collective rotation.

    Only adjacent species pairs are coupled (like joints in a chain).
    Symmetric: K[i, i+1] = K[i+1, i] = strength
    """
    K = np.zeros((n_species, n_species))

    if n_species <= 1:
        return K

    for i in range(n_species - 1):
        K[i, i + 1] = strength
        K[i + 1, i] = strength

    return K


def generate_position_matrix(n_species: int,
                             self_cohesion: float = 0.6,
                             cross_attraction: float = 0.1) -> np.ndarray:
    """
    Generate K_pos matrix for species cohesion.

    Args:
        n_species: Number of species
        self_cohesion: Diagonal values (attraction within species)
        cross_attraction: Off-diagonal values (attraction between neighbors)
    """
    K = np.zeros((n_species, n_species))
    for i in range(n_species):
        K[i, i] = self_cohesion
        for j in range(n_species):
            if abs(i - j) == 1:  # Only neighbors
                K[i, j] = cross_attraction
    return K


# =============================================================================
# Angle helpers and target profile generators
# =============================================================================

def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_segment_angles(centroids):
    """
    Compute segment angles from species centroids.

    Args:
        centroids: array (S, 2) of species centroid positions

    Returns:
        theta: array (S-1,) segment angles in radians
    """
    d = np.diff(centroids, axis=0)  # (S-1, 2) displacement vectors
    return np.arctan2(d[:, 1], d[:, 0])


def compute_joint_angles(theta):
    """
    Compute joint (turning) angles from segment angles.

    Joint i (for i=0..S-3) is the signed angle change at centroid i+1,
    i.e. how much the chain bends at that interior vertex.

    Args:
        theta: array (S-1,) segment angles

    Returns:
        phi: array (S-2,) joint angles wrapped to [-pi, pi]
    """
    return wrap_to_pi(np.diff(theta))


def sample_points_uniformly(points, n_samples):
    """
    Uniformly sample n_samples points along a polyline by arc length.

    Args:
        points: array (N, 2) of polyline vertices
        n_samples: number of points to sample

    Returns:
        sampled: array (n_samples, 2) of uniformly-spaced points
    """
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cum_length[-1]
    if total < 1e-8:
        return np.tile(points[0], (n_samples, 1))
    targets = np.linspace(0, total, n_samples)
    sampled = np.zeros((n_samples, 2))
    for k, t in enumerate(targets):
        idx = min(np.searchsorted(cum_length, t, side='right') - 1, len(points) - 2)
        seg_len = seg_lengths[idx]
        alpha = (t - cum_length[idx]) / seg_len if seg_len > 1e-8 else 0.0
        sampled[k] = points[idx] + alpha * diffs[idx]
    return sampled


def generate_target_straight(n_joints):
    """All joint angles zero → straight chain."""
    return np.zeros(n_joints)


def generate_target_u_shape(n_joints, phi0):
    """
    U-shape: constant curvature, all joints bend the same direction.

    Creates a circular arc opening upward (∪).
    """
    return np.full(n_joints, -phi0)  # Negative for upward-facing U


def generate_target_m_shape(n_joints, phi0):
    """Symmetric two-peak: +phi0 at ends, -phi0 at center, using cosine profile."""
    target = np.zeros(n_joints)
    for i in range(n_joints):
        # Map i to [0, 2*pi]: cos gives +1 at ends, -1 at center
        t = i / max(1, n_joints - 1) * 2 * np.pi
        target[i] = phi0 * np.cos(t)
    return target


def generate_target_hug(n_joints, phi0):
    """
    Hug/encircle: opposite curvature on each half.

    Left half bends one way, right half bends the other,
    creating a shape that curves inward on both ends.
    """
    target = np.zeros(n_joints)
    if n_joints == 0:
        return target
    mid = (n_joints - 1) / 2.0
    for i in range(n_joints):
        # +phi0 on left half, -phi0 on right half
        if i < mid:
            target[i] = phi0
        elif i > mid:
            target[i] = -phi0
        else:
            target[i] = 0  # center joint is straight
    return target


PATTERN_NAMES = ["STRAIGHT", "U-SHAPE", "M-SHAPE", "HUG", "CUSTOM"]
PATTERN_GENERATORS = [
    generate_target_straight,
    generate_target_u_shape,
    generate_target_m_shape,
    generate_target_hug,
]


class MultiSpeciesDemo(ParticleLife):
    """
    Demo for manual control of N species swarm.

    Species are arranged side-by-side horizontally.
    Arrow keys control turn and speed via K_rot matrix manipulation.
    """

    def __init__(self, n_species: int = 3, n_particles: int = 20):
        # Create config with initial species count
        config = Config(
            n_particles=n_particles,
            n_species=n_species,
            position_matrix=generate_position_matrix(n_species).tolist(),
            orientation_matrix=generate_translation_matrix(n_species, 0.5).tolist(),
        )

        # Initialize base simulation
        super().__init__(config, headless=False)

        # Control state
        self.turn_input = 0.0       # -1 (full left) to +1 (full right)
        self.speed_input = 1.0      # 0 (stop) to 1 (full speed)
        self.base_k_rot = 0.8      # Base rotation matrix strength

        # Input smoothing
        self.turn_decay = 0.95      # How quickly turn input returns to 0
        self.speed_decay = 0.98     # How quickly speed input stabilizes

        # Formation parameters
        self.group_spacing = 1.0    # Meters between species group centers

        # Matrix editing mode
        self.matrix_edit_mode = False
        self.edit_row = 0
        self.edit_col = 0
        self.editing_k_rot = True  # True = K_rot, False = K_pos

        # GUI visibility
        self.hide_gui = False

        # ----- Control mode (shape tracking via PD on joint angles) -----
        self.control_mode = True        # False = manual blend, True = PD shape control
        self.pattern_index = 0          # 0=STRAIGHT, 1=U, 2=M, 3=HUG
        self.phi0 = 0.8                 # Target curvature magnitude (radians, ~46° per joint)
        self.kp = 0.5                   # Proportional gain
        self.kd = 2.0                   # Derivative gain (on raw Δφ, NOT divided by dt)
        self.u_max = 0.15               # Control output clamp
        self.e_max = np.pi / 6          # Max error magnitude (~30°, gentle corrections only)
        self.speed_scale = 0.3          # Multiplier before writing to K_rot
        self.ctrl_alpha = 0.2           # Output smoothing (higher = more responsive)
        self.head_bias = 0.0            # Arrow-key heading offset added to first joint target

        # Mouse-drawn custom shape state
        self.drawing_mode = False
        self.mouse_drawing = False
        self.drawing_points = []
        self.custom_target_angles = None

        # Visualization of extracted shape (screen-space data for overlay)
        self.shape_vis = None          # dict with visualization data, or None
        self.shape_vis_timer = 0.0     # seconds remaining for overlay display
        self.shape_vis_duration = 8.0  # how long to show the overlay (seconds)

        # PD state arrays (sized for current n_species, reset on species change)
        self._init_control_state()

        # Initialize particles in side-by-side formation
        self._initialize_side_by_side()

        # Seed phi_prev from initial chain (avoids derivative spike on first step)
        self._seed_phi_prev()

        # Pre-compute base matrices
        self._update_base_matrices()

        # Override window title
        pygame.display.set_caption("Multi-Species Manual Control Demo")

        print("=" * 60)
        print("Multi-Species Manual Control Demo")
        print("=" * 60)
        print(f"Species: {self.n_species}  Particles: {self.n}")
        print("")
        print("Controls:")
        print("  ←/→     Turn (manual) / heading bias (control)")
        print("  ↑/↓     Speed up/slow down")
        print("  +/-     Add/remove species")
        print("  C       Toggle CONTROL MODE (PD shape tracking)")
        print("  R       Reset positions")
        print("  SPACE   Pause")
        print("  H       Hide/show all GUI")
        print("  I       Toggle info panel")
        print("")
        print("Control mode:")
        print("  1/2/3/4 Pattern: STRAIGHT/U/M/HUG")
        print("  [/]     Decrease/increase phi0")
        print("  K/J     Increase/decrease kp")
        print("  D/F     Decrease/increase kd")
        print("")
        print("Matrix Editing:")
        print("  M       Toggle matrix edit mode")
        print("  TAB     Switch K_rot/K_pos")
        print("  WASD    Navigate cells")
        print("  E/X     Increase/decrease value")
        print("=" * 60)

    def _initialize_side_by_side(self):
        """Arrange species in horizontal line formation."""
        center_x = self.config.sim_width / 2
        center_y = self.config.sim_height / 2

        # Calculate total width of formation
        total_width = (self.n_species - 1) * self.group_spacing
        start_x = center_x - total_width / 2

        particles_per_species = self.n // self.n_species

        for i in range(self.n):
            species_id = min(i // particles_per_species, self.n_species - 1)
            self.species[i] = species_id

            # Group center for this species (meters)
            group_center_x = start_x + species_id * self.group_spacing
            group_center_y = center_y

            # Random offset within group (meters)
            self.positions[i, 0] = group_center_x + self.rng.uniform(-0.2, 0.2)
            self.positions[i, 1] = group_center_y + self.rng.uniform(-0.2, 0.2)

            # Initial velocity (slight forward motion)
            self.velocities[i] = np.array([0.0, -0.1])

    def _update_base_matrices(self):
        """Pre-compute translation base matrix (rotation computed dynamically)."""
        self.K_translation = generate_translation_matrix(self.n_species, 1.0)

    def _init_control_state(self):
        """Initialize / reset PD controller state arrays for current n_species."""
        S = self.n_species
        n_joints = max(0, S - 2)   # number of interior joints (phi has length S-2)
        n_edges = max(0, S - 1)    # number of edges (u_edge has length S-1)
        self.phi_prev = np.zeros(n_joints)
        self.phi_filtered = np.zeros(n_joints)  # low-pass filtered phi for derivative
        self.u_edge_prev = np.zeros(n_edges)
        self.ctrl_mean_error = 0.0  # diagnostic: mean |error| across joints

    def _seed_phi_prev(self):
        """Measure current joint angles and store as phi_prev and phi_filtered.

        This avoids a derivative spike on the first control step
        (phi_prev=0 vs actual phi would give a huge phi_dot).
        """
        if self.n_species < 3:
            return
        centroids = np.array(self.get_species_centroids())
        theta = compute_segment_angles(centroids)
        phi = compute_joint_angles(theta)
        self.phi_prev = phi.copy()
        self.phi_filtered = phi.copy()

    def from_screen(self, screen_pos):
        """Convert screen pixels to simulation coordinates (meters)."""
        return (screen_pos[0] / (self.ppu * self.zoom),
                screen_pos[1] / (self.ppu * self.zoom))

    def _smooth_polyline(self, points, min_spacing=5.0, smooth_passes=3):
        """Smooth a raw mouse polyline to remove jitter.

        Args:
            points: array (N, 2) of raw screen-space points
            min_spacing: minimum pixel distance between kept points
            smooth_passes: number of moving-average passes
        """
        # 1. Downsample: keep points at least min_spacing apart
        filtered = [points[0]]
        for p in points[1:]:
            if np.linalg.norm(p - filtered[-1]) >= min_spacing:
                filtered.append(p)
        if len(filtered) < 2:
            filtered.append(points[-1])
        pts = np.array(filtered, dtype=float)

        # 2. Moving average smoothing (preserve endpoints)
        for _ in range(smooth_passes):
            if len(pts) <= 2:
                break
            smoothed = pts.copy()
            for i in range(1, len(pts) - 1):
                smoothed[i] = 0.25 * pts[i - 1] + 0.5 * pts[i] + 0.25 * pts[i + 1]
            pts = smoothed

        return pts

    def _process_drawn_shape(self):
        """Convert drawn polyline into target joint angles."""
        if len(self.drawing_points) < 2:
            print("Need at least 2 points to define a shape")
            return
        # Smooth raw mouse input to remove jitter
        raw = np.array(self.drawing_points, dtype=float)
        smoothed = self._smooth_polyline(raw)
        # Screen → sim coordinates
        sim_points = np.array([self.from_screen(p) for p in smoothed])
        # Uniformly sample n_species points along arc length
        sampled = sample_points_uniformly(sim_points, self.n_species)
        # Extract joint angles
        theta = compute_segment_angles(sampled)
        phi = compute_joint_angles(theta)

        # Store visualization data (screen-space) before applying
        smoothed_screen = smoothed.tolist()  # already screen coords
        sampled_screen = [self.to_screen(p) for p in sampled]
        self.shape_vis = {
            'smoothed': smoothed_screen,       # polyline (many points)
            'sampled': sampled_screen,          # N species points
            'theta': theta.copy(),              # segment angles
            'phi': phi.copy(),                  # joint angles
            'raw_count': len(raw),
            'smoothed_count': len(smoothed_screen),
        }
        self.shape_vis_timer = self.shape_vis_duration

        # Apply as custom target
        self.custom_target_angles = phi
        self.pattern_index = 4  # CUSTOM
        self.control_mode = True
        self._init_control_state()
        self._seed_phi_prev()
        self.head_bias = 0.0
        print(f"Custom shape applied: {len(phi)} joint angles extracted")

    def _get_target_profile(self):
        """Return the desired joint-angle array phi_star (length S-2)."""
        n_joints = max(0, self.n_species - 2)
        if self.pattern_index == 4:  # CUSTOM
            if self.custom_target_angles is not None and len(self.custom_target_angles) == n_joints:
                phi_star = self.custom_target_angles.copy()
            else:
                phi_star = np.zeros(n_joints)  # fallback to straight
        else:
            gen = PATTERN_GENERATORS[self.pattern_index]
            # STRAIGHT generator ignores phi0; others use it
            if self.pattern_index == 0:
                phi_star = gen(n_joints)
            else:
                phi_star = gen(n_joints, self.phi0)
        # Add head bias: offset the first joint target so arrow keys steer the head
        if n_joints > 0:
            phi_star[0] += self.head_bias
        return phi_star

    def update_matrices_control_mode(self):
        """
        PD shape-tracking controller.

        Each step:
          1. Measure species centroids → segment angles θ → joint angles φ.
          2. Compute error e = wrap(φ* − φ) and φ_dot via finite difference.
          3. PD law → per-joint commands u_joint.
          4. Map to per-edge commands u_edge.
          5. Write u_edge into alignment_matrix as symmetric adjacent entries.

        Only K_rot is modified. K_pos is untouched.
        """
        S = self.n_species
        if S < 3:
            # Need at least 3 species for 1 joint angle
            self.alignment_matrix[:] = 0.0
            return

        n_joints = S - 2  # interior joints: indices 0..S-3
        n_edges = S - 1   # edges: indices 0..S-2

        # --- 1. Compute centroids (S, 2) ---
        centroids_list = self.get_species_centroids()
        centroids = np.array(centroids_list)  # (S, 2)

        # --- 2. Segment and joint angles ---
        theta = compute_segment_angles(centroids)  # (S-1,)
        phi = compute_joint_angles(theta)           # (S-2,) = n_joints

        # --- 3. PD control ---
        phi_star = self._get_target_profile()       # (S-2,)

        # Proportional: error = desired - measured
        # Clamp error magnitude to prevent large K_rot values that cause
        # groups to orbit each other (tangential force → orbiting feedback).
        error = wrap_to_pi(phi_star - phi)          # (S-2,)
        error = np.clip(error, -self.e_max, self.e_max)

        # Low-pass filter phi before differentiation to suppress measurement noise.
        # alpha_filt=0.3 → 70% previous + 30% new (smooth but responsive).
        alpha_filt = 0.3
        phi_filt = (1 - alpha_filt) * self.phi_filtered + alpha_filt * phi

        # Derivative: raw difference of filtered signal (NO division by dt).
        # Absorbing dt into kd avoids 1/dt noise amplification.
        phi_dot = wrap_to_pi(phi_filt - self.phi_filtered)  # (S-2,)

        # PD law: u = kp * e - kd * phi_dot
        u_joint = self.kp * error - self.kd * phi_dot  # (S-2,)

        # Clamp
        u_joint = np.clip(u_joint, -self.u_max, self.u_max)

        # Store for next step's derivative
        self.phi_prev = phi.copy()
        self.phi_filtered = phi_filt.copy()

        # Diagnostic
        self.ctrl_mean_error = float(np.mean(np.abs(error)))

        # --- 4. Map joint commands to edge commands ---
        # Joint j sits at vertex j+1, between edge j (left) and edge j+1 (right).
        # Physics: positive symmetric K_rot on edge i pushes species i UP and
        # species i+1 DOWN (via CCW tangent), which DECREASES theta_i.
        # Since phi_j = theta_{j+1} - theta_j, to INCREASE phi_j:
        #   - edge j (left):  positive K_rot → theta_j decreases → phi_j increases → PLUS
        #   - edge j+1 (right): positive K_rot → theta_{j+1} decreases → phi_j decreases → MINUS
        u_edge = np.zeros(n_edges)
        for j in range(n_joints):
            u_edge[j]     += u_joint[j]   # left edge: positive increases phi_j
            u_edge[j + 1] -= u_joint[j]   # right edge: negative increases phi_j

        # --- Exponential smoothing to avoid jitter ---
        a = self.ctrl_alpha
        u_edge = (1 - a) * self.u_edge_prev + a * u_edge
        self.u_edge_prev = u_edge.copy()

        # --- 5. Write K_rot (alignment_matrix) ---
        # Symmetric adjacent entries: K[i,i+1] = K[i+1,i] = u
        # Symmetric K_rot creates opposite tangential forces on the two species
        # in a pair, producing relative rotation (bending) at that joint.
        # (Antisymmetric would create same-direction forces = translation.)
        self.alignment_matrix[:] = 0.0
        for i in range(n_edges):
            val = u_edge[i] * self.speed_scale
            val = np.clip(val, -1.0, 1.0)
            self.alignment_matrix[i, i + 1] = val
            self.alignment_matrix[i + 1, i] = val

    def update_matrices_from_input(self):
        """
        Update K_rot matrix based on current turn and speed input.

        Blends between translation mode (turn_input ≈ 0) and
        rotation mode (turn_input ≈ ±1).

        Uses differential rotation: outer species in turn arc get more force
        to maintain line formation.
        """
        # Skip auto-update when in matrix edit mode (allow manual edits to persist)
        if self.matrix_edit_mode:
            return

        # Blend factor: 0 = pure translation, 1 = pure rotation
        blend = min(1.0, abs(self.turn_input) * 2)

        # Direction of rotation
        turn_direction = np.sign(self.turn_input) if self.turn_input != 0 else 0

        # Effective strength based on speed
        effective_strength = self.base_k_rot * self.speed_input

        # Compute blended matrix
        K_trans = self.K_translation * effective_strength

        # Generate rotation matrix (direction applied by multiplying by turn_direction)
        K_rotation = generate_rotation_matrix(self.n_species, 1.0)
        K_rot = K_rotation * effective_strength * turn_direction

        # Blend: when not turning, use translation; when turning, add rotation
        K_blended = K_trans * (1 - blend) + K_rot * blend

        # Apply to alignment matrix (only K_rot changes based on input)
        for i in range(self.n_species):
            for j in range(self.n_species):
                self.alignment_matrix[i, j] = np.clip(K_blended[i, j], -1.0, 1.0)

    def draw_shape_extraction_overlay(self):
        """Draw visualization of the shape extraction pipeline.

        Shows: smoothed curve, sampled species points, segment directions,
        and joint angle arcs — all fading out over shape_vis_duration seconds.
        """
        vis = self.shape_vis
        if vis is None or self.shape_vis_timer <= 0:
            return

        # Fade alpha: full opacity for first half, then linear fade
        t_frac = self.shape_vis_timer / self.shape_vis_duration
        alpha = int(255 * min(1.0, t_frac * 2))

        smoothed = vis['smoothed']
        sampled = vis['sampled']
        theta = vis['theta']
        phi = vis['phi']

        # --- 1. Smoothed polyline (template curve) ---
        if len(smoothed) >= 2:
            curve_color = (180, 180, 220, alpha)
            # Draw as dotted: every other segment
            pts = [(int(p[0]), int(p[1])) for p in smoothed]
            for k in range(0, len(pts) - 1, 2):
                pygame.draw.line(self.screen, (180, 180, 220), pts[k], pts[k + 1], 1)

        # --- 2. Segment lines between sampled points ---
        for k in range(len(sampled) - 1):
            p0 = sampled[k]
            p1 = sampled[k + 1]
            seg_color = (100, 100, 255)
            pygame.draw.line(self.screen, seg_color, p0, p1, 2)

        # --- 3. Sampled species points (numbered circles) ---
        for k, (sx, sy) in enumerate(sampled):
            # Outer ring
            pygame.draw.circle(self.screen, (60, 60, 200), (sx, sy), 10, 2)
            # Species index label
            label = self.font.render(str(k), True, (60, 60, 200))
            lrect = label.get_rect(center=(sx, sy - 16))
            self.screen.blit(label, lrect)

        # --- 4. Joint angle arcs at interior vertices ---
        arc_radius = 25
        for j in range(len(phi)):
            # Joint j is at sampled point j+1 (interior vertex)
            cx, cy = sampled[j + 1]
            # Incoming segment angle and outgoing segment angle
            theta_in = theta[j]
            theta_out = theta[j + 1]
            angle_deg = np.degrees(phi[j])

            # Draw arc from incoming to outgoing direction
            # pygame arc angles: 0 = right, CCW positive, in radians
            # We draw the arc on the side of the bend
            start_a = min(theta_in, theta_out)
            end_a = max(theta_in, theta_out)
            # Make sure we draw the shorter arc
            if end_a - start_a > np.pi:
                start_a, end_a = end_a, start_a + 2 * np.pi

            rect = pygame.Rect(cx - arc_radius, cy - arc_radius,
                               arc_radius * 2, arc_radius * 2)
            # pygame.draw.arc uses screen-space angles (y-flipped)
            # Screen y is flipped, so negate angles for display
            pygame.draw.arc(self.screen, (220, 80, 80), rect,
                           -end_a, -start_a, 2)

            # Angle label at the arc midpoint
            mid_a = (theta_in + theta_out) / 2
            lx = int(cx + (arc_radius + 12) * np.cos(mid_a))
            ly = int(cy + (arc_radius + 12) * np.sin(mid_a))
            angle_text = self.font.render(f"{angle_deg:+.0f}\u00b0", True, (220, 80, 80))
            arect = angle_text.get_rect(center=(lx, ly))
            self.screen.blit(angle_text, arect)

        # --- 5. Pipeline info text (top of screen) ---
        info_lines = [
            f"Raw: {vis['raw_count']} pts \u2192 Smoothed: {vis['smoothed_count']} pts "
            f"\u2192 Sampled: {len(sampled)} species pts "
            f"\u2192 {len(theta)} segments \u2192 {len(phi)} joint angles",
        ]
        y = 40
        for line in info_lines:
            text = self.font.render(line, True, (80, 80, 180))
            rect = text.get_rect(center=(self.config.width // 2, y))
            # Background for readability
            bg = rect.inflate(10, 4)
            s = pygame.Surface(bg.size, pygame.SRCALPHA)
            s.fill((255, 255, 255, 200))
            self.screen.blit(s, bg.topleft)
            self.screen.blit(text, rect)
            y += 20

    def step(self):
        """Perform one simulation step with control updates."""
        if self.paused:
            return

        # Tick shape visualization timer
        if self.shape_vis_timer > 0:
            self.shape_vis_timer -= self.config.dt

        if self.control_mode:
            # In control mode: head_bias decays, PD controller updates K_rot
            self.head_bias *= self.turn_decay
            if abs(self.head_bias) < 0.01:
                self.head_bias = 0.0
            self.update_matrices_control_mode()
        else:
            # Manual blend mode: turn_input decays, blend translation/rotation
            self.turn_input *= self.turn_decay
            if abs(self.turn_input) < 0.01:
                self.turn_input = 0.0
            self.update_matrices_from_input()

        # Call parent step for physics
        super().step()

    def draw(self):
        """Draw the simulation with control overlay."""
        self.screen.fill((255, 255, 255))

        # Draw particles
        self.draw_particles()

        # Skip GUI elements if hidden
        if self.hide_gui:
            return

        # Draw centroid spine, markers, and swarm centroid
        if self.show_centroids:
            pts = self.draw_centroid_spine()
            self.draw_centroid_markers(pts)
            self.draw_swarm_centroid()

        # Draw control indicators
        self.draw_control_indicator()

        # Draw mode overlay
        if self.drawing_mode:
            if len(self.drawing_points) >= 2:
                pygame.draw.lines(self.screen, (100, 100, 255), False, self.drawing_points, 2)
            text = self.font.render("DRAW SHAPE (drag mouse)", True, (100, 100, 255))
            rect = text.get_rect(center=(self.config.width // 2, 60))
            self.screen.blit(text, rect)

        # Shape extraction visualization overlay (fades over time)
        self.draw_shape_extraction_overlay()

        # Draw info panel
        if self.show_info:
            self.draw_info_panel()

    def draw_control_indicator(self):
        """Draw visual indicator for current turn/speed input."""
        # Position in bottom center
        cx = self.config.width // 2
        cy = self.config.height - 60

        # Background circle
        pygame.draw.circle(self.screen, (230, 230, 230), (cx, cy), 50)
        pygame.draw.circle(self.screen, (200, 200, 200), (cx, cy), 50, 2)

        # Turn indicator (horizontal bar)
        turn_width = int(self.turn_input * 40)
        if turn_width != 0:
            color = (100, 150, 255) if turn_width < 0 else (255, 150, 100)
            bar_x = cx if turn_width > 0 else cx + turn_width
            pygame.draw.rect(self.screen, color, (bar_x, cy - 5, abs(turn_width), 10))

        # Speed indicator (vertical bar)
        speed_height = int(self.speed_input * 40)
        pygame.draw.rect(self.screen, (100, 200, 100),
                        (cx - 5, cy - speed_height, 10, speed_height))

        # Center dot
        pygame.draw.circle(self.screen, (50, 50, 50), (cx, cy), 5)

        # Labels
        font = self.font
        turn_text = font.render(f"Turn: {self.turn_input:.2f}", True, (100, 100, 100))
        speed_text = font.render(f"Speed: {self.speed_input:.2f}", True, (100, 100, 100))
        self.screen.blit(turn_text, (cx - 40, cy + 55))
        self.screen.blit(speed_text, (cx - 45, cy + 75))

    def draw_info_panel(self):
        """Draw information panel."""
        info_lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Species: {self.n_species}",
            f"Particles: {self.n}",
            "",
        ]

        if self.control_mode:
            info_lines += [
                "CONTROL MODE ON",
                f"Pattern: {PATTERN_NAMES[self.pattern_index]}",
                f"phi0: {self.phi0:.2f}  bias: {self.head_bias:+.2f}",
                f"kp: {self.kp:.1f}  kd: {self.kd:.1f}  u_max: {self.u_max:.1f}",
                f"mean|e|: {self.ctrl_mean_error:.3f}",
                "",
                "C:mode 1-4:pattern [/]:phi0",
                "K/J:kp  D/F:kd  ←/→:bias",
                "G:draw shape  V:centroids",
            ]
        else:
            info_lines += [
                "MANUAL MODE",
                f"Turn: {self.turn_input:+.2f}",
                f"Speed: {self.speed_input:.2f}",
                "",
                "C:control mode  G:draw shape",
                "←/→:turn  ↑/↓:speed",
                "+/-:species  R:reset  V:centroids",
            ]

        y = 10
        for line in info_lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 22

        # Draw K_rot matrix visualization
        self.draw_matrix_viz()

        self.draw_pause_indicator()

    def draw_single_matrix(self, matrix, label_text, x_start, y_start, is_editing=False):
        """Draw a single matrix visualization with values and species colors."""
        cell_size = 35
        color_indicator_size = 12

        # Label color
        if is_editing:
            label_color = (100, 100, 200)
            label_text = label_text + " (EDIT)"
        else:
            label_color = (100, 100, 100)

        label = self.font.render(label_text, True, label_color)
        self.screen.blit(label, (x_start, y_start))
        y_start += 25

        # Draw column color indicators (top)
        for j in range(self.n_species):
            cx = x_start + j * cell_size + cell_size // 2 - 1
            cy = y_start - 10
            pygame.draw.circle(self.screen, self.colors[j], (cx, cy), color_indicator_size // 2)
            pygame.draw.circle(self.screen, (100, 100, 100), (cx, cy), color_indicator_size // 2, 1)

        for i in range(self.n_species):
            # Draw row color indicator (left side)
            rx = x_start - 15
            ry = y_start + i * cell_size + cell_size // 2 - 1
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

                # Always show value text
                val_text = self.font.render(f"{value:.1f}", True, (255, 255, 255))
                text_rect = val_text.get_rect(center=(x + cell_size//2 - 1, y + cell_size//2 - 1))
                self.screen.blit(val_text, text_rect)

        return y_start + self.n_species * cell_size

    def draw_matrix_viz(self):
        """Draw visualization of both K_rot and K_pos matrices."""
        cell_size = 35
        x_start = self.config.width - 30 - self.n_species * cell_size
        y_start = 10

        # Draw K_rot matrix
        is_editing_krot = self.matrix_edit_mode and self.editing_k_rot
        y_after_krot = self.draw_single_matrix(
            self.alignment_matrix, "K_rot:", x_start, y_start, is_editing_krot
        )

        # Draw K_pos matrix below K_rot
        y_pos_start = y_after_krot + 20
        is_editing_kpos = self.matrix_edit_mode and not self.editing_k_rot
        y_after_kpos = self.draw_single_matrix(
            self.matrix, "K_pos:", x_start, y_pos_start, is_editing_kpos
        )

        # Draw edit mode instructions
        if self.matrix_edit_mode:
            instr_y = y_after_kpos + 10
            instr = self.font.render("WASD:move E/X:+/- TAB:switch M:exit", True, (100, 100, 100))
            self.screen.blit(instr, (x_start - 50, instr_y))

    def change_species_count(self, new_count: int):
        """Change the number of species and reinitialize."""
        new_count = max(2, min(10, new_count))
        if new_count == self.n_species:
            return

        print(f"Changing species count: {self.n_species} → {new_count}")

        # Update config
        self.config.n_species = new_count
        self.n_species = new_count
        self.n = self.config.n_particles * self.n_species

        # Regenerate matrices
        self.matrix = generate_position_matrix(new_count)
        self.alignment_matrix = generate_translation_matrix(new_count, self.base_k_rot)

        # Update base matrices
        self._update_base_matrices()

        # Regenerate colors
        self.colors = []
        for i in range(new_count):
            hue = i / new_count
            color = pygame.Color(0)
            color.hsva = (hue * 360, 70, 90, 100)
            self.colors.append((color.r, color.g, color.b))

        # Reinitialize particles (reallocates arrays for new total)
        self.initialize_particles()
        self._initialize_side_by_side()

        # Reset control state
        self.turn_input = 0.0
        self.head_bias = 0.0
        self.edit_row = 0
        self.edit_col = 0
        self._init_control_state()

        # Invalidate custom shape (wrong joint count for new species)
        self.custom_target_angles = None
        if self.pattern_index == 4:
            self.pattern_index = 0

        print(f"Species: {self.n_species} x {self.config.n_particles} = {self.n} particles")

    def _adjust_matrix_value(self, delta: float):
        """Adjust the selected matrix cell value."""
        if self.editing_k_rot:
            current = self.alignment_matrix[self.edit_row, self.edit_col]
            new_val = np.clip(current + delta, -1.0, 1.0)
            self.alignment_matrix[self.edit_row, self.edit_col] = new_val
            print(f"K_rot[{self.edit_row},{self.edit_col}] = {new_val:.2f}")
        else:
            current = self.matrix[self.edit_row, self.edit_col]
            new_val = np.clip(current + delta, -1.0, 1.0)
            self.matrix[self.edit_row, self.edit_col] = new_val
            print(f"K_pos[{self.edit_row},{self.edit_col}] = {new_val:.2f}")

    def handle_events(self) -> bool:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # Mouse events for shape drawing
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.drawing_mode and event.button == 1:
                    self.mouse_drawing = True
                    self.drawing_points = [event.pos]

            elif event.type == pygame.MOUSEMOTION:
                if self.drawing_mode and self.mouse_drawing:
                    self.drawing_points.append(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                if self.drawing_mode and event.button == 1:
                    self.mouse_drawing = False
                    self._process_drawn_shape()
                    self.drawing_mode = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self._initialize_side_by_side()
                    self.turn_input = 0.0
                    self.head_bias = 0.0
                    self._init_control_state()
                    self._seed_phi_prev()
                    print("Reset positions")

                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info

                elif event.key == pygame.K_v:
                    self.show_centroids = not self.show_centroids

                elif event.key == pygame.K_o:
                    self.show_orientations = not self.show_orientations

                elif event.key == pygame.K_h:
                    self.hide_gui = not self.hide_gui

                elif event.key == pygame.K_g:
                    self.drawing_mode = True
                    self.drawing_points = []
                    print("Draw mode: drag mouse to draw shape")

                # Species count adjustment (works in all modes except matrix edit)
                elif not self.matrix_edit_mode and (event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS):
                    self.change_species_count(self.n_species + 1)

                elif not self.matrix_edit_mode and event.key == pygame.K_MINUS:
                    self.change_species_count(self.n_species - 1)

                # Toggle control mode
                elif event.key == pygame.K_c:
                    self.control_mode = not self.control_mode
                    if self.control_mode:
                        self._init_control_state()
                        self._seed_phi_prev()
                        self.head_bias = 0.0
                    print(f"Control mode: {'ON' if self.control_mode else 'OFF'}")

                # Control mode: pattern selection and gain tuning
                elif self.control_mode and not self.matrix_edit_mode:
                    if event.key == pygame.K_1:
                        self.pattern_index = 0
                        print(f"Pattern: {PATTERN_NAMES[0]}")
                    elif event.key == pygame.K_2:
                        self.pattern_index = 1
                        print(f"Pattern: {PATTERN_NAMES[1]}")
                    elif event.key == pygame.K_3:
                        self.pattern_index = 2
                        print(f"Pattern: {PATTERN_NAMES[2]}")
                    elif event.key == pygame.K_4:
                        self.pattern_index = 3
                        print(f"Pattern: {PATTERN_NAMES[3]}")
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
                    elif event.key == pygame.K_f:
                        self.kd = min(5.0, self.kd + 0.1)
                        print(f"kd: {self.kd:.1f}")
                    elif event.key == pygame.K_d:
                        self.kd = max(0.0, self.kd - 0.1)
                        print(f"kd: {self.kd:.1f}")

                # Matrix editing controls
                elif event.key == pygame.K_m:
                    self.matrix_edit_mode = not self.matrix_edit_mode
                    print(f"Matrix edit mode: {'ON' if self.matrix_edit_mode else 'OFF'}")

                elif event.key == pygame.K_TAB and self.matrix_edit_mode:
                    self.editing_k_rot = not self.editing_k_rot
                    print(f"Editing: {'K_rot' if self.editing_k_rot else 'K_pos'}")

                elif self.matrix_edit_mode:
                    # WASD navigation
                    if event.key == pygame.K_w:
                        self.edit_row = max(0, self.edit_row - 1)
                    elif event.key == pygame.K_s:
                        self.edit_row = min(self.n_species - 1, self.edit_row + 1)
                    elif event.key == pygame.K_a:
                        self.edit_col = max(0, self.edit_col - 1)
                    elif event.key == pygame.K_d:
                        self.edit_col = min(self.n_species - 1, self.edit_col + 1)
                    # E/X or +/- to adjust value
                    elif event.key == pygame.K_e or event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self._adjust_matrix_value(0.1)
                    elif event.key == pygame.K_x or event.key == pygame.K_MINUS:
                        self._adjust_matrix_value(-0.1)

        # Handle held keys for continuous control
        keys = pygame.key.get_pressed()

        if self.control_mode:
            # In control mode: Left/Right adjusts heading bias on first joint
            if keys[pygame.K_LEFT]:
                self.head_bias = max(-np.pi, self.head_bias - 0.05)
            if keys[pygame.K_RIGHT]:
                self.head_bias = min(np.pi, self.head_bias + 0.05)
        else:
            # Manual blend mode: Left/Right turns
            if keys[pygame.K_LEFT]:
                self.turn_input = max(-1.0, self.turn_input - 0.05)
            if keys[pygame.K_RIGHT]:
                self.turn_input = min(1.0, self.turn_input + 0.05)

        # Speed control (both modes)
        if keys[pygame.K_UP]:
            self.speed_input = min(1.0, self.speed_input + 0.02)
        if keys[pygame.K_DOWN]:
            self.speed_input = max(0.0, self.speed_input - 0.02)

        return True

    def run(self):
        """Main simulation loop."""
        running = True

        while running:
            running = self.handle_events()
            self.step()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    demo = MultiSpeciesDemo(n_species=4, n_particles=20)
    demo.run()


if __name__ == "__main__":
    main()
