# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python swarm simulation and robotics deployment workspace. Core demos live in `src/`, with `particle_life.py` as the main 2D Pygame simulation plus 3D, formation, snake, and trajectory demos. Behavior presets are JSON files in `presets/`. Parameter sweeps and plotting utilities live in `characterization/`; behavior reproduction scripts live in `behavior_reproduction/`. ROS2/Crazyflie deployment code is under `crazyflie_deployment/`, especially `crazyflie_deployment/src/particle_life/particle_life/`, with launch files, configs, and helper scripts in sibling directories. Generated outputs such as `plots/`, `sweep_out/`, `cache/`, and ROS build/install/log directories are artifacts.

## Build, Test, and Development Commands

Install local Python tooling with:

```bash
python -m pip install -e ".[dev]"
python -m pip install pygame numpy
```

Run the main simulation:

```bash
python src/particle_life.py
python src/particle_life.py --load presets/3_chase.json
```

Run characterization scripts, for example:

```bash
python characterization/sweep_metrics.py
python characterization/plot_heatmaps.py
```

Build the ROS2 package from `crazyflie_deployment/`:

```bash
colcon build --symlink-install
ros2 launch particle_life particle_life.launch.py backend:=sim
```

## Coding Style & Naming Conventions

Use Python 3.9+ for package code; ROS2 deployment targets Ubuntu 24.04 with ROS2 Jazzy. Follow PEP 8, 4-space indentation, and type hints where they clarify simulation or controller interfaces. `pyproject.toml` configures Ruff with 100-character lines and import sorting; run `ruff check .` before broad changes. Use `snake_case` for functions, modules, variables, and preset filenames; use `PascalCase` for classes.

## Testing Guidelines

The project declares `pytest` and `pytest-asyncio` dev dependencies, but there is not yet a dedicated root `tests/` suite. Add tests under `tests/` using `test_*.py` when changing reusable physics, metrics, serialization, or controller code. Prefer deterministic tests with fixed random seeds and small particle counts. Run:

```bash
pytest
```

For ROS2 changes, also build with `colcon build --symlink-install` and run the relevant launch file or script in simulation before hardware tests.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects such as `Add crazyflie's particle life simulation` and `Update the configuration`. Keep the first line concise and action-oriented. Pull requests should describe the behavior change, list commands run, note any generated artifacts, and include screenshots or videos for visualization changes. For Crazyflie or hardware-facing changes, document the tested backend, config files touched, and any safety assumptions.

## Security & Configuration Tips

Do not commit machine-specific ROS build outputs, logs, or secrets. Review `crazyflie_deployment/config/*.yaml` carefully before flight tests; configuration changes can affect real hardware behavior.
