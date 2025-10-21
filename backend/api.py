"""Simple FastAPI backend for particle life simulation."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

from .config import DEFAULT_CONFIG, SimConfig
from .presets import get_preset, list_presets
from .simulation import Simulation

# ============================================================================
# Pydantic Models
# ============================================================================

class ConfigUpdate(BaseModel):
    """Update configuration parameters."""
    n_species: Optional[int] = Field(default=None, ge=1, le=10)
    n_particles: Optional[int] = Field(default=None, ge=1, le=5000)
    dt: Optional[float] = Field(default=None, gt=0.0)
    damping: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    max_speed: Optional[float] = Field(default=None, gt=0.0)
    seed: Optional[int] = None


class MatrixUpdate(BaseModel):
    """Update interaction matrix."""
    matrix: List[List[float]]


# ============================================================================
# FastAPI App with Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    global _simulation_task
    print(">>> STARTUP: Starting background simulation task <<<")
    _simulation_task = asyncio.create_task(_simulation_loop())
    print(f">>> Background task created: {_simulation_task} <<<")

    yield

    # Shutdown
    print(">>> SHUTDOWN: Canceling background simulation task <<<")
    if _simulation_task:
        _simulation_task.cancel()
        try:
            await _simulation_task
        except asyncio.CancelledError:
            pass

app = FastAPI(title="Particle Life Sandbox", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_config = DEFAULT_CONFIG
_simulation = Simulation(_config)
_state_lock = asyncio.Lock()
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
_websocket_clients: set = set()
_simulation_task = None

if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


# ============================================================================
# Background Simulation Task
# ============================================================================

async def _simulation_loop():
    """Background task that steps simulation and broadcasts to all clients."""
    print(">>> Background simulation task STARTED <<<")
    step_count = 0

    while True:
        if step_count % 10 == 0:
            print(f">>> Loop iteration {step_count}, acquiring lock...")

        async with _state_lock:
            if step_count % 10 == 0:
                print(f">>> Lock acquired, calling step()...")

            # Step simulation
            _simulation.step()
            step_count += 1

            if step_count % 10 == 0:
                pos = _simulation.positions[0]
                print(f">>> Step {step_count}, t={_simulation.t:.2f}, p0=({pos[0]:.2f}, {pos[1]:.2f}), clients={len(_websocket_clients)}")

            # Get state snapshot
            state = _simulation.get_state()
            state["matrix"] = _simulation.matrix.tolist()
            state["config"] = {
                "species_count": _simulation.config.n_species,
                "particle_count": _simulation.config.n_particles,
                "interaction_radius": _simulation.kernel_params.r_cut,
                "frame_interval": _simulation.config.frame_interval,
            }

        # Broadcast to all connected clients (outside lock)
        message = {"type": "state", "payload": state}
        dead_clients = set()

        if step_count % 10 == 0:
            print(f">>> Broadcasting to {len(_websocket_clients)} clients...")

        for client in _websocket_clients:
            try:
                # Check if client is still connected before sending
                if client.client_state == WebSocketState.CONNECTED:
                    if step_count % 10 == 0:
                        print(f">>>   Sending to client (state={client.client_state})...")
                    await client.send_json(message)
                    if step_count % 10 == 0:
                        print(f">>>   Send successful!")
                else:
                    dead_clients.add(client)
            except Exception as e:
                print(f">>> ERROR sending to client: {type(e).__name__}: {e}")
                print(f">>>   Client state was: {client.client_state}")
                dead_clients.add(client)

        # Remove dead clients
        if dead_clients:
            print(f">>> Removing {len(dead_clients)} dead clients")
        _websocket_clients.difference_update(dead_clients)

        await asyncio.sleep(_simulation.config.frame_interval)


# Startup/shutdown now handled by lifespan context manager above


# ============================================================================
# Helper Functions
# ============================================================================

def _update_config(config_update: ConfigUpdate) -> SimConfig:
    """Create new config with updates applied."""
    kwargs = {
        "width": _config.width,
        "height": _config.height,
        "n_species": config_update.n_species or _config.n_species,
        "n_particles": config_update.n_particles or _config.n_particles,
        "dt": config_update.dt or _config.dt,
        "damping": config_update.damping or _config.damping,
        "max_speed": config_update.max_speed or _config.max_speed,
        "frame_interval": _config.frame_interval,
        "seed": config_update.seed if config_update.seed is not None else _config.seed,
    }
    return SimConfig(**kwargs)


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/")
async def index() -> FileResponse:
    """Serve frontend."""
    if not _frontend_dir.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(_frontend_dir / "index.html")


@app.get("/debug")
async def debug() -> FileResponse:
    """Serve debug page."""
    if not _frontend_dir.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(_frontend_dir / "debug.html")


@app.get("/test")
async def test() -> FileResponse:
    """Serve test page."""
    if not _frontend_dir.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(_frontend_dir / "test.html")


@app.get("/simple")
async def simple() -> FileResponse:
    """Serve simple test page."""
    if not _frontend_dir.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(_frontend_dir / "simple.html")


@app.get("/fixed")
async def fixed() -> FileResponse:
    """Serve fixed test page."""
    if not _frontend_dir.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(_frontend_dir / "fixed.html")


@app.get("/working")
async def working() -> FileResponse:
    """Serve working index page."""
    if not _frontend_dir.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(_frontend_dir / "index_working.html")


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get current configuration and matrix."""
    async with _state_lock:
        return {
            "config": {
                "width": _simulation.config.width,
                "height": _simulation.config.height,
                "species_count": _simulation.config.n_species,  # Frontend expects species_count
                "particle_count": _simulation.config.n_particles,  # Frontend expects particle_count
                "interaction_radius": _simulation.kernel_params.r_cut,  # For rendering
                "dt": _simulation.config.dt,
                "damping": _simulation.config.damping,
                "max_speed": _simulation.config.max_speed,
                "frame_interval": _simulation.config.frame_interval,
                "seed": _simulation.config.seed,
            },
            "matrix": _simulation.matrix.tolist(),
            "kernel_params": {
                "r_rep": _simulation.kernel_params.r_rep,
                "r_att": _simulation.kernel_params.r_att,
                "r_cut": _simulation.kernel_params.r_cut,
                "a_rep": _simulation.kernel_params.a_rep,
                "a_att": _simulation.kernel_params.a_att,
            }
        }


@app.post("/config")
async def update_config(config_update: ConfigUpdate) -> Dict[str, Any]:
    """Update configuration and restart simulation."""
    global _config, _simulation
    async with _state_lock:
        new_config = _update_config(config_update)
        _config = new_config
        _simulation = Simulation(_config)
        return await get_config()


@app.post("/matrix")
async def update_matrix(matrix_update: MatrixUpdate) -> Dict[str, Any]:
    """Update interaction matrix."""
    async with _state_lock:
        try:
            matrix = np.array(matrix_update.matrix)
            _simulation.set_matrix(matrix)
            return {"matrix": _simulation.matrix.tolist()}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.post("/reset")
async def reset_simulation() -> Dict[str, str]:
    """Reset simulation to initial state."""
    async with _state_lock:
        _simulation.reset()
        return {"status": "reset"}


@app.get("/presets")
async def get_presets() -> List[Dict[str, Any]]:
    """List available presets."""
    presets = list_presets()
    return [
        {
            "name": p.name,
            "description": p.description,
            "n_species": p.n_species,
            "matrix": p.matrix.tolist(),
        }
        for p in presets
    ]


@app.post("/presets/{name}")
async def apply_preset(name: str) -> Dict[str, Any]:
    """Apply a preset scenario."""
    global _config, _simulation
    async with _state_lock:
        try:
            preset = get_preset(name)

            # Create new config with preset's species count
            new_config = SimConfig(
                width=_config.width,
                height=_config.height,
                n_species=preset.n_species,
                n_particles=_config.n_particles,
                dt=_config.dt,
                damping=_config.damping,
                max_speed=_config.max_speed,
                frame_interval=_config.frame_interval,
                seed=_config.seed,
            )

            _config = new_config
            _simulation = Simulation(_config, matrix=preset.matrix, kernel_params=preset.kernel_params)

            return {
                "preset": preset.name,
                "config": {
                    "species_count": _simulation.config.n_species,
                    "particle_count": _simulation.config.n_particles,
                },
                "matrix": _simulation.matrix.tolist(),
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# WebSocket
# ============================================================================

async def _listener(websocket: WebSocket, queue: asyncio.Queue) -> None:
    """Listen for client messages."""
    try:
        while True:
            message = await websocket.receive_json()
            await queue.put(message)
    except WebSocketDisconnect:
        pass


async def _handle_message(message: Dict[str, Any]) -> None:
    """Handle client commands."""
    global _simulation

    msg_type = message.get("type")

    if msg_type == "update_matrix":
        matrix = message.get("matrix")
        if matrix is None:
            raise ValueError("Message missing 'matrix'")
        _simulation.set_matrix(np.array(matrix))

    elif msg_type == "reset":
        _simulation.reset()

    elif msg_type == "use_preset":
        name = message.get("name")
        if name is None:
            raise ValueError("Message missing 'name'")
        preset = get_preset(name)
        _simulation.set_matrix(preset.matrix)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint - clients subscribe to simulation updates."""
    await websocket.accept()
    print(f">>> WebSocket accepted, state={websocket.client_state}, adding to clients...")
    _websocket_clients.add(websocket)
    print(f">>> Client added, total clients: {len(_websocket_clients)}")

    queue: asyncio.Queue = asyncio.Queue()
    listener = asyncio.create_task(_listener(websocket, queue))

    try:
        # Just handle client messages (simulation runs in background)
        while True:
            # Handle pending messages
            while not queue.empty():
                message = await queue.get()
                try:
                    async with _state_lock:
                        await _handle_message(message)
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})

            await asyncio.sleep(0.1)  # Check for messages periodically

    except WebSocketDisconnect as e:
        print(f">>> WebSocket disconnected: {e}")
    except Exception as e:
        print(f">>> Unexpected WebSocket error: {type(e).__name__}: {e}")
    finally:
        print(f">>> Cleaning up client, clients before removal: {len(_websocket_clients)}")
        _websocket_clients.discard(websocket)
        print(f">>> Client removed, remaining clients: {len(_websocket_clients)}")
        listener.cancel()
