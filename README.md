# Particle Life Sandbox

A clean, minimal particle life simulation using a smooth 3-piece radial kernel.

## Features

- **3-piece radial kernel**: Smooth interaction forces (repulsion → attraction → zero)
- **N-species support**: Flexible species count (1-10) with editable interaction matrix
- **Real-time editing**: Modify species count, particle count, and interaction matrix on the fly
- **Presets**: 5 built-in scenarios (guards_workers, cyclic, flocking, ecosystem, chaos)
- **Reflective boundaries**: Particles bounce off walls
- **Clean architecture**: Single unified simulation system, easy to modify

## Quick Start

### Requirements
- Python 3.9+
- NumPy, FastAPI, Uvicorn

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy fastapi uvicorn websockets
```

### Run
```bash
uvicorn backend.api:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

## Usage

1. **Adjust species/particles**: Use the input fields
2. **Edit matrix**: Click cells in the interaction matrix table
   - Positive values = repulsion
   - Negative values = attraction
3. **Apply presets**: Select from dropdown and click Apply
4. **Reset**: Click Reset to restart with current settings

## Project Structure

```
swarm_life/
├── backend/
│   ├── simulation.py    # Main simulation class
│   ├── interaction.py   # 3-piece radial kernel
│   ├── config.py        # Configuration dataclass
│   ├── presets.py       # Preset scenarios
│   └── api.py           # FastAPI endpoints + WebSocket
├── frontend/
│   ├── index.html       # UI layout
│   ├── main.js          # WebSocket client + rendering
│   └── styles.css       # Styling
└── tests/
    └── test_kernel.py   # Kernel tests
```

## Modifying Interaction Logic

### 1. Change the Force Function

Edit `backend/interaction.py`:

```python
def radial_kernel(r: float, p: KernelParams) -> float:
    """
    Your custom force function here.
    Positive = repulsion, Negative = attraction
    """
    # Example: simple linear
    if r < p.r_cut:
        return p.a_rep * (1.0 - r / p.r_cut)
    return 0.0
```

### 2. Modify Interaction Matrix

Edit `backend/presets.py`:

```python
MY_PRESET = Preset(
    name="my_preset",
    description="My custom scenario",
    n_species=4,
    matrix=np.array([
        [+0.5, -0.8, +0.3, -0.4],
        [+0.6, +0.2, -0.7, +0.1],
        [-0.3, +0.4, +0.1, -0.6],
        [+0.2, -0.5, +0.8, +0.3],
    ]),
)
```

Then add it to `PRESETS` dict.

### 3. Adjust Kernel Parameters

Modify `backend/interaction.py`:

```python
@dataclass
class KernelParams:
    r_rep: float = 4.0    # Repulsion radius
    r_att: float = 24.0   # Attraction radius
    r_cut: float = 36.0   # Cutoff radius
    a_rep: float = 1.8    # Repulsion strength
    a_att: float = 0.8    # Attraction strength
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend |
| `/config` | GET | Get current config + matrix |
| `/config` | POST | Update config (species, particles, etc.) |
| `/matrix` | POST | Update interaction matrix |
| `/reset` | POST | Reset simulation |
| `/presets` | GET | List available presets |
| `/presets/{name}` | POST | Apply a preset |
| `/ws` | WebSocket | Real-time simulation stream |

## How It Works

### Force Calculation

For each pair of particles i and j:

1. Compute distance `r = ||pos_j - pos_i||`
2. If `r < r_cut`:
   - Get interaction strength: `k = K[species_i][species_j]`
   - Compute kernel: `g = radial_kernel(r, params)`
   - Force magnitude: `f = k * g`
   - Apply force: `force_i += f * direction`

### 3-Piece Kernel

```
g(r) = {
    a_rep * (1 - r/r_rep)                  if r < r_rep
    -a_att * 0.5 * (1 + cos(π(r-r_rep)/(r_att-r_rep)))  if r_rep ≤ r < r_att
    0                                       if r ≥ r_att
}
```

- **r < r_rep**: Linear repulsion (push away)
- **r_rep ≤ r < r_att**: Cosine attraction (pull together)
- **r ≥ r_att**: No force

## Presets

| Name | Species | Description |
|------|---------|-------------|
| `guards_workers` | 2 | Containment: Guards form ring, Workers cluster inside |
| `cyclic` | 3 | Rock-paper-scissors dynamics |
| `flocking` | 3 | Collective motion with mild repulsion |
| `ecosystem` | 4 | Complex multi-species interactions |
| `chaos` | 5 | High-energy chaotic system |

## Tests

```bash
pytest tests/
```

Tests cover:
- Kernel continuity and correctness
- Simulation stability
- Boundary conditions

## License

MIT
