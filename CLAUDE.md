# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive 2D particle life simulation with dual interaction matrices (position and orientation) implemented in pygame. The system simulates emergent swarm behaviors through simple local interactions between particles of different species.

## Core Components

### Main Simulation (`src/particle_life.py`)
- **Config**: Dataclass containing all simulation parameters, matrix storage, and JSON serialization
- **ParticleLife**: Main simulation class with pygame visualization
  - Dual matrix system: position matrix (attraction/repulsion) and orientation matrix (alignment/swirling)
  - Interactive matrix editing with real-time parameter adjustment
  - Workspace resizing, fullscreen support, and zoom functionality
  - Equal species distribution when changing particle/species counts

### Video Recording (`src/save_videos.py`)
- **SimulationVideoSaver**: Batch processes presets into MP4 videos
- **VideoRecorder**: OpenCV-based frame capture from pygame surface
- Overlays both interaction matrices on recorded videos for documentation

## Commands

```bash
# Run interactive simulation
python src/particle_life.py
python src/particle_life.py --load presets/3_chase.json

# Record videos from presets
python src/save_videos.py                           # Process all presets
python src/save_videos.py --load presets/3_chase.json  # Single preset

# Install dependencies
pip install pygame numpy
pip install opencv-python  # For video recording
```

## Architecture

### Force Calculation System
The simulation uses a dual-force model:

1. **Position Forces** (radial attraction/repulsion):
   - Attraction term: `k_pos * a_att`
   - Repulsion term: `a_rep / sqrt(r)`
   - Applied along the radial direction between particles

2. **Orientation Forces** (tangential swirling):
   - Swirl term: `-10 * k_rot * (ω_j/ω_max) * (a_rot/r)`
   - Applied along the tangential direction
   - Creates rotational/alignment behaviors

### Interaction Matrices
- **Position Matrix** (`K_pos[i][j]`): Controls attraction/repulsion between species i and j
  - Positive values: net attraction
  - Negative values: net repulsion
  - Range: [-1.0, 1.0]

- **Orientation Matrix** (`K_rot[i][j]`): Controls alignment/swirling between species i and j
  - Influences tangential velocity components
  - Creates collective rotation patterns
  - Range: [-1.0, 1.0]

## Key Controls

### Simulation Controls
- **SPACE**: Pause/Resume
- **R**: Reset particle positions
- **S**: Save current configuration with timestamp
- **I**: Toggle info panel
- **O**: Toggle orientation display
- **F11/F**: Toggle fullscreen with auto-zoom
- **Q/ESC**: Quit

### Parameter Adjustment
- **UP/DOWN**: Change species count (2-10)
- **LEFT/RIGHT**: Change particle count (±50)
- **SHIFT+LEFT/RIGHT**: Change workspace width
- **SHIFT+UP/DOWN**: Change workspace height
- **Mouse drag at edges**: Resize workspace

### Matrix Editing
- **M**: Toggle matrix editor
- **TAB**: Switch between Position/Orientation matrix
- **WASD**: Navigate matrix cells
- **+/-**: Modify selected cell value (±0.1)

## Presets System

Presets are JSON configurations stored in `presets/` containing:
- Workspace dimensions and initialization area
- Particle and species counts
- Physical parameters (dt, max_speed, damping)
- Both interaction matrices (position and orientation)

Example behaviors:
- **2_chase**: Predator-prey dynamics
- **3_rotate_together**: Collective rotation
- **3_encapsulate**: One species surrounds another
- **3_planet**: Orbital dynamics

## Development Notes

### Adding New Behaviors
1. Modify force calculation in `compute_velocities()` (lines 357-418)
2. Adjust matrix interpretation or add new matrix types
3. Create presets demonstrating the behavior

### Performance Optimization
- Currently uses O(n²) neighbor checking
- Cell-linked list optimization possible for large particle counts
- Boundary conditions use reflection with velocity reversal

### Testing Patterns
- Use seed=42 for deterministic testing
- Equal species distribution ensures balanced interactions
- Workspace resizing maintains particle containment

### Common Modifications
- Change force kernels: Edit attraction/repulsion terms in `compute_velocities()`
- Add new matrix types: Extend Config, add computation in force loop
- Modify boundaries: Change reflection logic (lines 456-467)
- Custom presets: Save configurations with 'S' during runtime