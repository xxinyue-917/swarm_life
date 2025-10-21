# Particle Life Standalone Simulation

A beautiful, interactive particle life simulation that runs locally on your machine - no browser or server needed!

## Features

- **Real-time particle simulation** with 500 particles and 5 species
- **Interactive controls** to change presets and parameters on the fly
- **Multiple presets**:
  - Chaos - High energy chaotic interactions
  - Symbiosis - Cooperative relationships
  - Predator-Prey - Chase dynamics
  - Random - Random interactions
  - Neutral - No interactions
- **60 FPS smooth animation**
- **Reflective boundaries** - particles bounce off walls

## Requirements

- Python 3.7+
- pygame (install with `pip install pygame`)
- numpy

## Running the Simulation

Simply run:

```bash
cd src
python3 particle_life.py
```

Or from the project root:

```bash
python3 src/particle_life.py
```

## Controls

- **UP/DOWN ARROWS** - Increase/Decrease species count (2-10)
- **SPACE** - Pause/Resume simulation
- **R** - Reset particle positions
- **1-5** - Select different presets:
  - 1: Chaos (default)
  - 2: Symbiosis
  - 3: Predator-Prey
  - 4: Random
  - 5: Neutral
- **I** - Toggle info panel
- **Q/ESC** - Quit

## How It Works

The simulation uses a 3-piece radial force kernel:
- **Close range** (r < 4): Strong repulsion to prevent overlap
- **Medium range** (4 < r < 24): Attraction/repulsion based on species interaction matrix
- **Far range** (24 < r < 36): Weak tail forces

Each species has different interaction strengths with other species, creating complex emergent behaviors from simple rules.

## Customization

You can modify the parameters in the `Config` class:
- `n_particles`: Number of particles (default: 500)
- `n_species`: Number of species (default: 5)
- `dt`: Simulation timestep (default: 0.05)
- `damping`: Velocity damping factor (default: 0.995)
- `max_speed`: Maximum particle speed (default: 300)

Enjoy watching the beautiful emergent patterns!