# Particle Life - Swarm Dynamics Simulation

An interactive particle life simulation featuring dual interaction matrices for complex emergent swarm behaviors. Watch as particles self-organize into flocks, predator-prey dynamics, rotating clusters, and more!

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Pygame](https://img.shields.io/badge/Pygame-2.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Dual Interaction System**: Separate position (attraction/repulsion) and orientation (alignment/swirling) matrices
- **Real-time Interactive Control**: Modify parameters and matrices while simulation runs
- **11 Preset Behaviors**: From chase dynamics to planetary orbits
- **Video Recording**: Batch convert presets to MP4 videos with matrix overlays
- **Flexible Workspace**: Resizable window with fullscreen and zoom support
- **Visual Orientation**: Particles show heading direction for rotation behaviors

## Quick Start

### Requirements
- Python 3.7+
- pygame >= 2.0
- numpy >= 1.19
- opencv-python >= 4.5 (optional, for video recording)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/swarm_life.git
cd swarm_life

# Install dependencies
pip install pygame numpy

# Optional: For video recording
pip install opencv-python
```

### Running the Simulation

```bash
# Run with default settings
python src/particle_life.py

# Load a specific preset
python src/particle_life.py --load presets/3_chase.json
```

## Controls

### Basic Controls
| Key | Action |
|-----|--------|
| **SPACE** | Pause/Resume simulation |
| **R** | Reset particle positions |
| **S** | Save current configuration |
| **I** | Toggle info panel |
| **O** | Toggle orientation display |
| **F11/F** | Toggle fullscreen |
| **Q/ESC** | Quit |

### Parameter Adjustment
| Key | Action |
|-----|--------|
| **↑/↓** | Change species count (2-10) |
| **←/→** | Change particle count (±50) |
| **Shift+←/→** | Adjust workspace width |
| **Shift+↑/↓** | Adjust workspace height |
| **Mouse drag edges** | Resize window |

### Matrix Editor
| Key | Action |
|-----|--------|
| **M** | Toggle matrix editor |
| **TAB** | Switch Position/Orientation matrix |
| **WASD** | Navigate matrix cells |
| **+/-** | Modify selected value |

## How It Works

### Dual Force System

The simulation computes forces between particles using two interaction matrices:

1. **Position Matrix (K_pos)**: Controls radial forces
   - Positive values → net attraction
   - Negative values → net repulsion
   - Creates clustering and separation behaviors

2. **Orientation Matrix (K_rot)**: Controls tangential forces
   - Influences rotational alignment
   - Creates swirling and orbital patterns
   - Enables collective rotation behaviors

### Force Calculation

For each particle pair (i,j):
```
radial_force = (K_pos[i,j] * a_att - a_rep/√r) * r̂
tangential_force = -10 * K_rot[i,j] * (ω_j/ω_max) * (a_rot/r) * t̂
total_force = radial_force + tangential_force
```

## Preset Behaviors

The `presets/` directory contains 11 example configurations:

| Preset | Description |
|--------|-------------|
| **2_chase** | Predator-prey pursuit dynamics |
| **2_dynamically_rotate** | Dynamic rotation patterns |
| **2_encapsulate** | One species surrounds another |
| **2_move_together** | Cohesive flocking |
| **2_sun_earth** | Orbital dynamics |
| **3_chase** | Three-species cyclic pursuit |
| **3_encapsulate** | Complex containment patterns |
| **3_encapsulate_rotate** | Rotating encapsulation |
| **3_move_together** | Multi-species flocking |
| **3_planet** | Multi-body orbital system |
| **3_rotate_together** | Collective rotation |

## Video Recording

Generate videos from your saved configurations:

```bash
# Process all presets
python src/save_videos.py

# Process specific preset
python src/save_videos.py --load presets/3_chase.json
```

Videos are saved to `videos/` directory with both interaction matrices overlaid.

### Configuration

Edit `src/save_videos.py` to adjust:
- `VIDEO_DURATION`: Length of videos (default: 20 seconds)
- `FPS`: Frame rate (default: 30)
- `OUTPUT_DIR`: Output directory (default: 'videos')

## Creating Custom Behaviors

1. **Run the simulation**: `python src/particle_life.py`
2. **Adjust parameters**: Use arrow keys to set species/particle counts
3. **Edit matrices**: Press 'M' to open matrix editor
4. **Fine-tune**: Use WASD to navigate, +/- to adjust values
5. **Save preset**: Press 'S' to save configuration
6. **Generate video**: Run `python src/save_videos.py`

## Project Structure

```
swarm_life/
├── src/
│   ├── particle_life.py      # Main simulation
│   └── save_videos.py         # Video recording utility
├── presets/                   # Saved configurations
│   ├── 2_chase.json
│   ├── 3_planet.json
│   └── ...
├── videos/                    # Generated videos (created on first use)
├── CLAUDE.md                  # Development guide
├── README.md                  # This file
└── VIDEO_README.md           # Video recording guide
```

## Physics Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.1 | Simulation timestep |
| `max_speed` | 300 | Maximum particle velocity |
| `a_rep` | 5.0 | Repulsion strength |
| `a_att` | 2.0 | Attraction strength |
| `a_rot` | 5.0 | Rotation alignment strength |
| `max_angular_speed` | 20.0 | Maximum angular velocity |
| `init_space_size` | 100 | Initial spawn area size |

## Tips for Interesting Patterns

- **Asymmetric matrices** create more dynamic behaviors
- **Mixed positive/negative values** in position matrix create complex territories
- **Small orientation values** (0.1-0.3) produce subtle collective motion
- **Larger orientation values** (0.5-1.0) create strong vortex patterns
- **Different species counts** reveal different pattern possibilities

## Troubleshooting

**Low FPS**: Reduce particle count with LEFT arrow key

**Particles stuck at edges**: Press 'R' to reset positions

**Matrix changes not visible**: Ensure simulation is not paused (SPACE to toggle)

**Video recording fails**: Install opencv-python: `pip install opencv-python`

## Future Enhancements

Potential areas for extension:
- GPU acceleration for larger particle counts
- Additional force kernels (Lennard-Jones, Morse potential)
- 3D visualization mode
- Network/graph-based interaction topologies
- Goal-directed behaviors and obstacles
- Multi-agent reinforcement learning integration

## License

MIT - See [LICENSE](LICENSE) file for details

## Contributing

Contributions welcome! Feel free to:
- Add new preset behaviors
- Optimize performance
- Enhance visualization
- Extend force models
- Improve documentation

## Acknowledgments

Inspired by:
- Jeffrey Ventrella's Particle Life investigations
- Swarm robotics research
- Complex systems and emergence studies