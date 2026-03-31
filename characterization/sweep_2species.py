#!/usr/bin/env python3
"""
2-Species Characterization Sweep — Video + Screenshot Output

Sweep K_pos and K_rot parameters for a 2-species system.
For each parameter combination, save a video and final screenshot.

Usage:
    python characterization/sweep_2species.py

All configuration is in the CONFIG dict below.
"""

import os
import sys

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import time
from pathlib import Path
from itertools import product

import numpy as np
import pygame
import cv2

from particle_life import Config, ParticleLife


# ============================================================
# CONFIGURATION — Edit these parameters to customize the sweep
# ============================================================

CONFIG = {
    # ----------------------------------------------------------
    # SWEEP TYPE
    # ----------------------------------------------------------
    # Options:
    #   'kpos_offdiag'  — sweep K₁₂ vs K₂₁
    #   'kpos_diag'     — sweep K₁₁ vs K₂₂
    #   'krot_offdiag'  — sweep R₁₂ vs R₂₁
    #   'krot_diag'     — sweep R₁₁ vs R₂₂
    #   'kpos_x_krot'   — sweep K₁₂ vs K₂₁ for each K_rot case (A/B/C/D)
    'sweep_type': 'krot_full',

    # ----------------------------------------------------------
    # GRID RESOLUTION
    # ----------------------------------------------------------
    'grid_points': 5,         # Points per axis (5 → 25 videos, 11 → 121 videos)
    'param_min': -1.0,
    'param_max':  1.0,

    # ----------------------------------------------------------
    # FIXED MATRIX VALUES (when not being swept)
    # ----------------------------------------------------------
    'K11': 0.6,    'K22': 0.6,    'K12': 0.3,    'K21': 0.3,
    'R11': 0.0,    'R22': 0.0,    'R12': 0.0,    'R21': 0.0,

    # ----------------------------------------------------------
    # K_ROT CASES (for sweep_type='kpos_x_krot')
    # ----------------------------------------------------------
    'krot_cases': {
        'A': {'R11': 0, 'R12': 0,  'R21': 0,  'R22': 0},
        'B': {'R11': 0, 'R12': 1,  'R21': 1,  'R22': 0},
        'C': {'R11': 0, 'R12': 1,  'R21': -1, 'R22': 0},
        'D': {'R11': 0, 'R12': 1,  'R21': 0,  'R22': 0},
    },

    # ----------------------------------------------------------
    # SIMULATION PHYSICS
    # ----------------------------------------------------------
    'n_particles': 50,        # Per species
    'sim_width': 10.0,
    'sim_height': 10.0,
    'dt': 0.05,
    'max_speed': 1.0,
    'r_max': 2.0,
    'beta': 0.2,
    'force_scale': 0.5,
    'a_rot': 1.0,
    'far_attraction': 0.1,

    # ----------------------------------------------------------
    # VIDEO / SCREENSHOT SETTINGS
    # ----------------------------------------------------------
    'video_duration': 10,     # Seconds per video
    'fps': 30,                # Frames per second
    'burnin_steps': 200,      # Steps before recording starts
    'window_width': 600,      # Pygame window width (pixels)
    'window_height': 600,     # Pygame window height (pixels)
    'show_matrices': True,    # Overlay matrix values on video

    # ----------------------------------------------------------
    # OUTPUT
    # ----------------------------------------------------------
    'output_dir': 'results',  # Relative to characterization/
}


# ============================================================
# VIDEO RECORDER
# ============================================================

class VideoRecorder:
    """Records pygame surface to video file."""

    def __init__(self, output_path, fps=30, width=600, height=600):
        self.fps = fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not self.out.isOpened():
            raise ValueError(f"Failed to open video writer: {output_path}")

    def add_frame(self, surface):
        frame = pygame.surfarray.array3d(surface)
        frame = np.rot90(frame)
        frame = np.flipud(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.out.write(frame)

    def release(self):
        self.out.release()


# ============================================================
# MATRIX OVERLAY
# ============================================================

def draw_matrix_overlay(surface, sim, font):
    """Draw K_pos and K_rot matrices on the screen."""
    grey = (80, 80, 80)
    x0, y0 = 10, 10
    cell_w = 45

    pos = sim.matrix
    rot = sim.alignment_matrix
    n = sim.n_species
    colors = sim.colors

    # Background
    bg_w = 20 + n * cell_w + 20
    bg_h = (n * 15 + 30) * 2 + 10
    bg = pygame.Surface((bg_w, bg_h))
    bg.set_alpha(220)
    bg.fill((255, 255, 255))
    surface.blit(bg, (x0 - 5, y0 - 5))

    # Position Matrix
    label = font.render("K_pos:", True, grey)
    surface.blit(label, (x0, y0))
    y = y0 + 18

    for j in range(n):
        cx = x0 + 18 + j * cell_w + cell_w // 2
        pygame.draw.circle(surface, colors[j], (cx, y), 4)
    y += 12

    for i in range(n):
        pygame.draw.circle(surface, colors[i], (x0 + 8, y + 6), 4)
        for j in range(n):
            val = pos[i, j]
            txt = font.render(f"{val:+.2f}", True, grey)
            r = txt.get_rect(center=(x0 + 18 + j * cell_w + cell_w // 2, y + 6))
            surface.blit(txt, r)
        y += 15

    y += 8

    # Rotation Matrix
    label = font.render("K_rot:", True, grey)
    surface.blit(label, (x0, y))
    y += 18

    for j in range(n):
        cx = x0 + 18 + j * cell_w + cell_w // 2
        pygame.draw.circle(surface, colors[j], (cx, y), 4)
    y += 12

    for i in range(n):
        pygame.draw.circle(surface, colors[i], (x0 + 8, y + 6), 4)
        for j in range(n):
            val = rot[i, j]
            txt = font.render(f"{val:+.2f}", True, grey)
            r = txt.get_rect(center=(x0 + 18 + j * cell_w + cell_w // 2, y + 6))
            surface.blit(txt, r)
        y += 15


# ============================================================
# PARAMETER LABEL OVERLAY
# ============================================================

def draw_param_label(surface, font, p1_name, p1_val, p2_name, p2_val, krot_case, width):
    """Draw parameter values at bottom of screen."""
    parts = [f"{p1_name}={p1_val:+.2f}", f"{p2_name}={p2_val:+.2f}"]
    if krot_case:
        parts.append(f"Case {krot_case}")
    text = "  |  ".join(parts)
    label = font.render(text, True, (60, 60, 60))
    bg = pygame.Surface((label.get_width() + 16, label.get_height() + 8))
    bg.set_alpha(200)
    bg.fill((255, 255, 255))
    x = (width - bg.get_width()) // 2
    y = 8
    surface.blit(bg, (x, y))
    surface.blit(label, (x + 8, y + 4))


# ============================================================
# SWEEP POINT GENERATION
# ============================================================

def build_matrices(K11, K12, K21, K22, R11, R12, R21, R22):
    K_pos = np.array([[K11, K12], [K21, K22]], dtype=float)
    K_rot = np.array([[R11, R12], [R21, R22]], dtype=float)
    return K_pos, K_rot


def generate_sweep_points(cfg):
    grid = np.linspace(cfg['param_min'], cfg['param_max'], cfg['grid_points'])
    sweep_type = cfg['sweep_type']
    points = []

    if sweep_type == 'kpos_offdiag':
        for k12, k21 in product(grid, grid):
            K_pos, K_rot = build_matrices(
                cfg['K11'], k12, k21, cfg['K22'],
                cfg['R11'], cfg['R12'], cfg['R21'], cfg['R22'])
            points.append(('K12', k12, 'K21', k21, '', K_pos, K_rot))

    elif sweep_type == 'kpos_diag':
        for k11, k22 in product(grid, grid):
            K_pos, K_rot = build_matrices(
                k11, cfg['K12'], cfg['K21'], k22,
                cfg['R11'], cfg['R12'], cfg['R21'], cfg['R22'])
            points.append(('K11', k11, 'K22', k22, '', K_pos, K_rot))

    elif sweep_type == 'krot_offdiag':
        for r12, r21 in product(grid, grid):
            K_pos, K_rot = build_matrices(
                cfg['K11'], cfg['K12'], cfg['K21'], cfg['K22'],
                cfg['R11'], r12, r21, cfg['R22'])
            points.append(('R12', r12, 'R21', r21, '', K_pos, K_rot))

    elif sweep_type == 'krot_diag':
        for r11, r22 in product(grid, grid):
            K_pos, K_rot = build_matrices(
                cfg['K11'], cfg['K12'], cfg['K21'], cfg['K22'],
                r11, cfg['R12'], cfg['R21'], r22)
            points.append(('R11', r11, 'R22', r22, '', K_pos, K_rot))

    elif sweep_type == 'kpos_x_krot':
        for case_name, case_vals in cfg['krot_cases'].items():
            for k12, k21 in product(grid, grid):
                K_pos, K_rot = build_matrices(
                    cfg['K11'], k12, k21, cfg['K22'],
                    case_vals['R11'], case_vals['R12'],
                    case_vals['R21'], case_vals['R22'])
                points.append(('K12', k12, 'K21', k21, case_name, K_pos, K_rot))

    elif sweep_type == 'krot_full':
        # 4D sweep: all 4 K_rot entries
        for r11, r12, r21, r22 in product(grid, grid, grid, grid):
            K_pos, K_rot = build_matrices(
                cfg['K11'], cfg['K12'], cfg['K21'], cfg['K22'],
                r11, r12, r21, r22)
            # Name: "R11_R12_R21_R22"
            name = f"{r11:.1f}_{r12:.1f}_{r21:.1f}_{r22:.1f}"
            points.append(('R12', r12, 'R21', r21, name, K_pos, K_rot))

    else:
        raise ValueError(f"Unknown sweep_type: {sweep_type}")

    return points


# ============================================================
# MAIN SWEEP
# ============================================================

def run_and_record(K_pos, K_rot, cfg, p1_name, p1_val, p2_name, p2_val,
                   krot_case, video_path, screenshot_path, font):
    """Run one simulation, record video, save final screenshot."""
    config = Config(
        n_species=2,
        n_particles=cfg['n_particles'],
        width=cfg['window_width'],
        height=cfg['window_height'],
        sim_width=cfg['sim_width'],
        sim_height=cfg['sim_height'],
        dt=cfg['dt'],
        max_speed=cfg['max_speed'],
        r_max=cfg['r_max'],
        beta=cfg['beta'],
        force_scale=cfg['force_scale'],
        a_rot=cfg['a_rot'],
        far_attraction=cfg['far_attraction'],
        position_matrix=K_pos.tolist(),
        orientation_matrix=K_rot.tolist(),
    )

    sim = ParticleLife(config, headless=False)

    # Scatter particles randomly across the workspace
    m = 0.5
    sim.positions[:, 0] = np.random.uniform(m, cfg['sim_width'] - m, sim.n)
    sim.positions[:, 1] = np.random.uniform(m, cfg['sim_height'] - m, sim.n)
    sim.velocities[:] = 0.0

    recorder = VideoRecorder(video_path, cfg['fps'],
                             cfg['window_width'], cfg['window_height'])

    # Record everything — including self-organization from random start
    total_frames = cfg['video_duration'] * cfg['fps']
    clock = pygame.time.Clock()

    for frame in range(total_frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                recorder.release()
                return

        sim.step()

        # Draw
        sim.screen.fill((255, 255, 255))
        sim.draw_particles()

        if cfg['show_matrices']:
            draw_matrix_overlay(sim.screen, sim, font)

        draw_param_label(sim.screen, font, p1_name, p1_val,
                         p2_name, p2_val, krot_case, cfg['window_width'])

        recorder.add_frame(sim.screen)
        pygame.display.flip()
        clock.tick(cfg['fps'])

    # Save final screenshot
    pygame.image.save(sim.screen, str(screenshot_path))

    recorder.release()
    del sim
    pygame.event.pump()


def main():
    cfg = CONFIG
    sweep_type = cfg['sweep_type']

    # Output directories
    base_dir = Path(__file__).parent / cfg['output_dir'] / sweep_type
    video_dir = base_dir / 'videos'
    screenshot_dir = base_dir / 'screenshots'
    video_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(base_dir / 'config.json', 'w') as f:
        serializable = {k: v for k, v in cfg.items()
                        if not isinstance(v, np.ndarray)}
        json.dump(serializable, f, indent=2, default=str)

    # Generate sweep points
    points = generate_sweep_points(cfg)

    print("=" * 60)
    print(f"2-Species Characterization Sweep")
    print("=" * 60)
    print(f"Sweep type:     {sweep_type}")
    print(f"Grid:           {cfg['grid_points']}×{cfg['grid_points']}")
    print(f"Total videos:   {len(points)}")
    print(f"Video duration: {cfg['video_duration']}s @ {cfg['fps']}fps")
    print(f"Burn-in:        {cfg['burnin_steps']} steps")
    print(f"Output:         {base_dir}")
    print("=" * 60)

    # Init pygame
    pygame.init()
    font = pygame.font.Font(None, 18)

    # Check which already exist (resume support)
    existing = set()
    for f in video_dir.glob('*.mp4'):
        existing.add(f.stem)

    if existing:
        print(f"Resuming: {len(existing)} videos already recorded")

    t_start = time.time()

    for idx, (p1n, p1v, p2n, p2v, kcase, K_pos, K_rot) in enumerate(points):
        # File naming: concise, sortable
        # e.g., "0.5_-0.5" or "0.5_-0.5_B"
        v1 = f"{p1v:.1f}"
        v2 = f"{p2v:.1f}"
        if kcase:
            name = f"{v1}_{v2}_{kcase}"
        else:
            name = f"{v1}_{v2}"

        # Skip if already done
        if name in existing:
            continue

        video_path = video_dir / f"{name}.mp4"
        screenshot_path = screenshot_dir / f"{name}.png"

        print(f"[{idx + 1}/{len(points)}] {name}")

        try:
            run_and_record(K_pos, K_rot, cfg,
                           p1n, p1v, p2n, p2v, kcase,
                           str(video_path), str(screenshot_path), font)
        except Exception as e:
            print(f"  ERROR: {e}")
            pygame.quit()
            pygame.init()
            font = pygame.font.Font(None, 18)
            continue

    pygame.quit()
    elapsed = time.time() - t_start

    print()
    print(f"Done! {len(points)} videos in {elapsed:.0f}s")
    print(f"Videos:      {video_dir}")
    print(f"Screenshots: {screenshot_dir}")


if __name__ == '__main__':
    main()
