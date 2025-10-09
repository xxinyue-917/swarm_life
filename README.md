# Particle Life Sandbox

FastAPI-powered Particle Life playground with WebSocket streaming and a canvas-based front end.

## Requirements
- Python 3.9+
- No Node toolchain required (static assets are bundled in `frontend/`).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run locally
```bash
uvicorn backend.api:app --reload
```
Open http://127.0.0.1:8000 in your browser. The canvas consumes live particle positions; editing the interaction matrix pushes immediate updates back to the simulation.

## Tests
```bash
pytest
```

## Project layout
- `backend/` – simulation core (`simulation.py`), configuration (`config.py`), presets (`presets.py`), and FastAPI app (`api.py`).
- `frontend/` – HTML, JS, and CSS assets served via FastAPI static files.
- `tests/` – unit tests and WebSocket smoke checks.

## Controls
- Use the control panel to set species and particle counts; applying changes resets the interaction matrix to match the new species list.

## Presets
Three presets (`flocking`, `predator_prey`, `chaos`) are available via the UI dropdown or `POST /presets/{name}`.
