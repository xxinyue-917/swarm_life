#!/usr/bin/env python3
"""
Formation From Image — Shape self-organization via K_pos topology

Particles self-organize into shapes using only K_pos (position matrix).
Sample N points on a shape, compute Gaussian K_pos from pairwise distances,
and let radial forces do the rest — no K_rot, no PID.

All configuration is inside the GUI:

Controls:
    L:     Load image (file dialog)
    1-5:   Built-in shapes (circle, star, square, ring, triangle)
    R:     Reset particles near target positions
    S:     Scatter randomly (test self-organization)
    +/-:   Change species count
    [/]:   Adjust sigma (neighborhood size)
    T:     Toggle target overlay
    V:     Toggle centroids
    I:     Toggle info panel
    SPACE: Pause
    Q/ESC: Quit
"""

import os
import numpy as np
import pygame
import pygame.gfxdraw

from particle_life import Config, ParticleLife


# =============================================================================
# Shape mask generators
# =============================================================================

def generate_circle_mask(size=200):
    y, x = np.ogrid[:size, :size]
    cx, cy, r = size // 2, size // 2, size // 2 - 10
    return (x - cx) ** 2 + (y - cy) ** 2 < r ** 2


def generate_ring_mask(size=200):
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    outer_r = size // 2 - 10
    inner_r = outer_r * 0.55
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    return (d2 < outer_r ** 2) & (d2 > inner_r ** 2)


def generate_square_mask(size=200):
    mask = np.zeros((size, size), dtype=bool)
    m = size // 6
    mask[m:size - m, m:size - m] = True
    return mask


def generate_triangle_mask(size=200):
    mask = np.zeros((size, size), dtype=bool)
    cx = size // 2
    for y in range(size):
        # Triangle: top vertex at (cx, margin), base at bottom
        margin = size // 8
        if y < margin:
            continue
        progress = (y - margin) / (size - 2 * margin)
        half_w = int(progress * (size // 2 - margin))
        x_lo = max(0, cx - half_w)
        x_hi = min(size, cx + half_w)
        mask[y, x_lo:x_hi] = True
    return mask


def generate_star_mask(size=200, n_points=5):
    mask = np.zeros((size, size), dtype=bool)
    cx, cy = size // 2, size // 2
    outer_r = size // 2 - 10
    inner_r = outer_r * 0.38

    # Star polygon vertices
    poly = []
    for i in range(n_points * 2):
        a = -np.pi / 2 + i * np.pi / n_points
        r = outer_r if i % 2 == 0 else inner_r
        poly.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    poly = np.array(poly)

    # Scanline fill
    for y in range(size):
        # Find x intersections
        xs = []
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            yi, yj = poly[i][1], poly[j][1]
            if (yi < y) == (yj < y):
                continue
            xi, xj = poly[i][0], poly[j][0]
            t = (y - yi) / (yj - yi)
            xs.append(xi + t * (xj - xi))
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            x0, x1 = int(xs[k]), int(xs[k + 1])
            mask[y, max(0, x0):min(size, x1)] = True
    return mask


def generate_cross_mask(size=200):
    mask = np.zeros((size, size), dtype=bool)
    m = size // 6
    arm = size // 4
    cx, cy = size // 2, size // 2
    mask[cy - arm:cy + arm, cx - m:cx + m] = True  # vertical
    mask[cy - m:cy + m, cx - arm:cx + arm] = True  # horizontal
    return mask


def generate_crescent_mask(size=200):
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    r_outer = size // 2 - 10
    r_inner = int(r_outer * 0.75)
    offset = int(r_outer * 0.35)
    outer = (x - cx) ** 2 + (y - cy) ** 2 < r_outer ** 2
    inner = (x - cx - offset) ** 2 + (y - cy) ** 2 < r_inner ** 2
    return outer & ~inner


def generate_letter_L_mask(size=200):
    mask = np.zeros((size, size), dtype=bool)
    m = size // 8
    w = size // 4
    # Vertical bar
    mask[m:size - m, m:m + w] = True
    # Horizontal bar
    mask[size - m - w:size - m, m:size - m] = True
    return mask


def generate_arrow_mask(size=200):
    mask = np.zeros((size, size), dtype=bool)
    cx = size // 2
    # Arrow head (triangle pointing right)
    for y in range(size):
        dy = abs(y - cx)
        x_right = size - size // 6
        x_left = cx
        if dy < (size // 3):
            progress = 1.0 - dy / (size // 3)
            x_end = int(x_left + (x_right - x_left) * progress)
            mask[y, x_left:x_end] = True
    # Arrow shaft
    shaft_h = size // 8
    mask[cx - shaft_h:cx + shaft_h, size // 8:cx] = True
    return mask


BUILT_IN_SHAPES = {
    1: ("Circle", generate_circle_mask),
    2: ("Star", generate_star_mask),
    3: ("Square", generate_square_mask),
    4: ("Ring", generate_ring_mask),
    5: ("Triangle", generate_triangle_mask),
    6: ("Cross", generate_cross_mask),
    7: ("Crescent", generate_crescent_mask),
    8: ("Letter L", generate_letter_L_mask),
    9: ("Arrow", generate_arrow_mask),
}


# =============================================================================
# Image loading
# =============================================================================

def load_shape_mask(image_path, threshold=128):
    """Load image and convert to binary mask (True = shape)."""
    from PIL import Image
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    return arr < threshold


def prompt_image_path():
    """Prompt user for image path via console. Returns path or None."""
    print("\nEnter image path (or drag file into terminal):")
    try:
        path = input("  > ").strip().strip("'\"")  # Strip quotes from drag-and-drop
        if path and os.path.isfile(path):
            return path
        elif path:
            print(f"  File not found: {path}")
        return None
    except (EOFError, KeyboardInterrupt):
        return None


# =============================================================================
# Points sampling and K_pos computation
# =============================================================================

def sample_points_in_mask(mask, n_points, sim_w, sim_h, lloyd_iters=20):
    """Sample n_points uniformly within the mask using Lloyd's relaxation.

    1. Random initial placement within the mask
    2. Lloyd's relaxation: iteratively move each point to the centroid
       of its Voronoi region (restricted to the mask)
    This produces very uniform spacing — much better than rejection sampling.
    """
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Empty mask — no shape pixels found")

    margin = 0.5
    scale_x = (sim_w - 2 * margin) / w
    scale_y = (sim_h - 2 * margin) / h

    # Step 1: Random initial placement (in pixel coords)
    indices = np.random.choice(len(xs), size=n_points, replace=False)
    pts_px = np.column_stack([xs[indices], ys[indices]]).astype(float)

    # Step 2: Lloyd's relaxation in pixel space
    # Pre-compute all valid pixel coordinates for Voronoi assignment
    all_pixels = np.column_stack([xs, ys]).astype(float)  # (M, 2)

    for _ in range(lloyd_iters):
        # Assign each mask pixel to nearest point (Voronoi)
        # Use chunked computation to avoid huge memory allocation
        chunk_size = 5000
        labels = np.empty(len(all_pixels), dtype=int)
        for start in range(0, len(all_pixels), chunk_size):
            end = min(start + chunk_size, len(all_pixels))
            chunk = all_pixels[start:end]
            dists = np.linalg.norm(chunk[:, np.newaxis, :] - pts_px[np.newaxis, :, :], axis=2)
            labels[start:end] = np.argmin(dists, axis=1)

        # Move each point to the centroid of its Voronoi region
        for k in range(n_points):
            region = all_pixels[labels == k]
            if len(region) > 0:
                pts_px[k] = region.mean(axis=0)

    # Convert to sim coordinates
    points = np.zeros((n_points, 2))
    points[:, 0] = margin + pts_px[:, 0] * scale_x
    points[:, 1] = margin + pts_px[:, 1] * scale_y

    return points


def compute_kpos_gaussian(points, sigma=None, self_cohesion=0.8, max_attraction=0.3):
    """Compute K_pos using Gaussian kernel on pairwise distances."""
    n = len(points)
    # Vectorized pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)

    if sigma is None:
        nn_dists = []
        for i in range(n):
            row = dists[i]
            row_nonzero = row[row > 0]
            if len(row_nonzero) > 0:
                nn_dists.append(row_nonzero.min())
        sigma = np.median(nn_dists) * 1.5 if nn_dists else 1.0

    K = max_attraction * np.exp(-dists ** 2 / (2 * sigma ** 2))
    K[K < max_attraction * 0.01] = 0.0
    np.fill_diagonal(K, self_cohesion)

    return K, sigma


# =============================================================================
# Demo
# =============================================================================

class FormationFromImage(ParticleLife):
    """Shape formation via K_pos topology derived from an image."""

    def __init__(self, n_species=50, n_particles=5):
        self.n_particles_per = n_particles
        self.sigma_scale = 1.5
        self.shape_name = "Circle"
        self.mask = generate_circle_mask(200)

        sim_w, sim_h = 20.0, 20.0

        self.target_points = sample_points_in_mask(
            self.mask, n_species, sim_w, sim_h
        )
        K_pos, self.sigma = compute_kpos_gaussian(self.target_points)

        config = Config(
            n_particles=n_particles,
            n_species=n_species,
            sim_width=sim_w,
            sim_height=sim_h,
            position_matrix=K_pos.tolist(),
            orientation_matrix=np.zeros((n_species, n_species)).tolist(),
        )

        super().__init__(config, headless=False)

        self.show_targets = False
        self.hide_gui = False
        self.mask_surface = None  # Cached pygame surface for shape preview
        self._update_mask_surface()

        self._initialize_near_targets()

        pygame.display.set_caption("Formation From Image")
        self._print_help()

    def _print_help(self):
        print("=" * 60)
        print("Formation From Image — K_pos Self-Organization")
        print("=" * 60)
        print(f"Shape: {self.shape_name}")
        print(f"Species: {self.n_species}  Sigma: {self.sigma:.2f}")
        print("")
        print("  L       Load image (type path in console)")
        print("  1-5     Circle / Star / Square / Ring / Triangle")
        print("  6-9     Cross / Crescent / Letter L / Arrow")
        print("  R/S     Reset near targets / Scatter randomly")
        print("  +/-     Species count")
        print("  [/]     Adjust sigma")
        print("  T/V/I   Targets / Centroids / Info")
        print("=" * 60)

    def _initialize_near_targets(self):
        particles_per = self.n // self.n_species
        spread = 0.3
        for i in range(self.n):
            sid = min(i // particles_per, self.n_species - 1)
            self.species[i] = sid
            tx, ty = self.target_points[sid]
            self.positions[i, 0] = tx + np.random.uniform(-spread, spread)
            self.positions[i, 1] = ty + np.random.uniform(-spread, spread)
            self.velocities[i] = np.array([0.0, 0.0])

    def _scatter(self):
        m = 0.5
        sw, sh = self.config.sim_width, self.config.sim_height
        particles_per = self.n // self.n_species
        for i in range(self.n):
            sid = min(i // particles_per, self.n_species - 1)
            self.species[i] = sid
            self.positions[i, 0] = np.random.uniform(m, sw - m)
            self.positions[i, 1] = np.random.uniform(m, sh - m)
            self.velocities[i] = np.array([0.0, 0.0])

    def _load_shape(self, mask, name="Custom"):
        """Load a new shape mask, resample, recompute K_pos, reinitialize."""
        # Sample first — only update state if successful
        target_points = sample_points_in_mask(
            mask, self.n_species,
            self.config.sim_width, self.config.sim_height
        )
        self.mask = mask
        self.shape_name = name
        self.target_points = target_points
        K_pos, self.sigma = compute_kpos_gaussian(
            self.target_points, sigma=None
        )
        self.matrix[:] = K_pos
        self.alignment_matrix[:] = 0.0
        self._scatter()
        self._update_mask_surface()
        print(f"Shape: {name}  Sigma: {self.sigma:.2f}  "
              f"K_pos nonzero: {np.count_nonzero(K_pos)}/{self.n_species**2}")

    def _load_image_dialog(self):
        """Prompt for image path in console, load and apply."""
        path = prompt_image_path()
        if path:
            try:
                mask = load_shape_mask(path)
                name = path.split('/')[-1].split('\\')[-1]
                self._load_shape(mask, name)
            except Exception as e:
                print(f"Error loading image: {e}")

    def _rebuild(self, n_species):
        n_species = max(3, min(200, n_species))
        if n_species == self.n_species:
            return

        print(f"Species: {self.n_species} → {n_species}")
        self.config.n_species = n_species
        self.config.n_particles = self.n_particles_per
        self.n_species = n_species
        self.n = self.n_particles_per * n_species

        self.target_points = sample_points_in_mask(
            self.mask, n_species,
            self.config.sim_width, self.config.sim_height
        )
        K_pos, self.sigma = compute_kpos_gaussian(self.target_points)
        self.matrix = K_pos
        self.alignment_matrix = np.zeros((n_species, n_species))

        # Sync parent references
        if hasattr(self.config, '_position_matrix_np'):
            self.config._position_matrix_np = self.matrix
            self.config._orientation_matrix_np = self.alignment_matrix

        self.colors = []
        for i in range(n_species):
            hue = i / n_species
            color = pygame.Color(0)
            color.hsva = (hue * 360, 70, 90, 100)
            self.colors.append((color.r, color.g, color.b))

        self.initialize_particles()
        self._initialize_near_targets()
        print(f"Sigma: {self.sigma:.2f}  "
              f"K_pos nonzero: {np.count_nonzero(K_pos)}/{n_species**2}")

    def _adjust_sigma(self, delta):
        self.sigma_scale = max(0.3, self.sigma_scale + delta)

        nn_dists = []
        n = len(self.target_points)
        for i in range(n):
            d = np.linalg.norm(self.target_points - self.target_points[i], axis=1)
            d_nz = d[d > 0]
            if len(d_nz) > 0:
                nn_dists.append(d_nz.min())
        base_sigma = np.median(nn_dists) if nn_dists else 1.0
        self.sigma = base_sigma * self.sigma_scale

        K_pos, _ = compute_kpos_gaussian(
            self.target_points, sigma=self.sigma
        )
        self.matrix[:] = K_pos
        print(f"Sigma: {self.sigma:.2f} (scale: {self.sigma_scale:.1f})  "
              f"Nonzero: {np.count_nonzero(K_pos)}/{self.n_species**2}")

    def _update_mask_surface(self):
        """Convert current mask to a pygame surface thumbnail for preview."""
        if self.mask is None:
            self.mask_surface = None
            return
        h, w = self.mask.shape
        # Create RGB array: shape pixels = dark gray, background = light gray
        rgb = np.full((h, w, 3), 230, dtype=np.uint8)  # light bg
        rgb[self.mask] = [80, 80, 80]  # dark shape
        # Scale to thumbnail size
        thumb_size = 120
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self.mask_surface = pygame.transform.smoothscale(surf, (thumb_size, thumb_size))

    # ================================================================
    # Drawing
    # ================================================================

    def draw(self):
        self.screen.fill((255, 255, 255))

        if self.show_targets:
            self._draw_targets()

        self.draw_particles()

        if self.hide_gui:
            return

        # Shape preview (top right)
        self._draw_shape_preview()

        if self.show_centroids:
            self.draw_swarm_centroid()

        if self.show_info:
            self._draw_info()

        self.draw_pause_indicator()

    def _draw_shape_preview(self):
        """Draw the target shape thumbnail in the top-right corner."""
        if self.mask_surface is None:
            return
        pad = 10
        x = self.config.width - self.mask_surface.get_width() - pad
        y = pad

        # Border
        rect = pygame.Rect(x - 2, y - 2,
                           self.mask_surface.get_width() + 4,
                           self.mask_surface.get_height() + 4)
        pygame.draw.rect(self.screen, (180, 180, 180), rect, 2)

        self.screen.blit(self.mask_surface, (x, y))

        # Label
        label = self.font.render(self.shape_name, True, (100, 100, 100))
        self.screen.blit(label, (x, y + self.mask_surface.get_height() + 4))

    def _draw_targets(self):
        z = getattr(self, 'zoom', 1.0)
        for i, (tx, ty) in enumerate(self.target_points):
            sx = int(tx * self.ppu * z)
            sy = int(ty * self.ppu * z)
            color = self.colors[i % len(self.colors)]
            faded = (
                min(255, color[0] // 2 + 128),
                min(255, color[1] // 2 + 128),
                min(255, color[2] // 2 + 128),
            )
            pygame.draw.circle(self.screen, faded, (sx, sy), 6, 1)

    def _draw_info(self):
        lines = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Shape: {self.shape_name}",
            f"Species: {self.n_species}  Particles: {self.n}",
            f"Sigma: {self.sigma:.2f} (scale: {self.sigma_scale:.1f})",
            "",
            "L: Load image  1-9: Built-in",
            "R: Reset  S: Scatter",
            "+/-: Species  [/]: Sigma",
            "T: Targets  V: Centroids",
        ]
        y = 10
        for line in lines:
            if line:
                text = self.font.render(line, True, (100, 100, 100))
                self.screen.blit(text, (10, y))
            y += 22

    # ================================================================
    # Events
    # ================================================================

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._initialize_near_targets()
                    print("Reset near targets")
                elif event.key == pygame.K_s:
                    self._scatter()
                    print("Scattered randomly")
                elif event.key == pygame.K_t:
                    self.show_targets = not self.show_targets
                elif event.key == pygame.K_v:
                    self.show_centroids = not self.show_centroids
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_h:
                    self.hide_gui = not self.hide_gui

                # Load image
                elif event.key == pygame.K_l:
                    self._load_image_dialog()

                # Built-in shapes
                elif event.key == pygame.K_1:
                    self._load_shape(generate_circle_mask(200), "Circle")
                elif event.key == pygame.K_2:
                    self._load_shape(generate_star_mask(200), "Star")
                elif event.key == pygame.K_3:
                    self._load_shape(generate_square_mask(200), "Square")
                elif event.key == pygame.K_4:
                    self._load_shape(generate_ring_mask(200), "Ring")
                elif event.key == pygame.K_5:
                    self._load_shape(generate_triangle_mask(200), "Triangle")
                elif event.key == pygame.K_6:
                    self._load_shape(generate_cross_mask(200), "Cross")
                elif event.key == pygame.K_7:
                    self._load_shape(generate_crescent_mask(200), "Crescent")
                elif event.key == pygame.K_8:
                    self._load_shape(generate_letter_L_mask(200), "Letter L")
                elif event.key == pygame.K_9:
                    self._load_shape(generate_arrow_mask(200), "Arrow")

                # Species count
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    step = 10 if self.n_species >= 20 else 5
                    self._rebuild(self.n_species + step)
                elif event.key == pygame.K_MINUS:
                    step = 10 if self.n_species > 20 else 5
                    self._rebuild(self.n_species - step)

                # Sigma
                elif event.key == pygame.K_RIGHTBRACKET:
                    self._adjust_sigma(0.2)
                elif event.key == pygame.K_LEFTBRACKET:
                    self._adjust_sigma(-0.2)

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
    demo = FormationFromImage(n_species=50, n_particles=5)
    demo.run()


if __name__ == "__main__":
    main()
