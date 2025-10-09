from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.websockets import WebSocketState

from .config import DEFAULT_CONFIG, SimulationConfig
from .presets import get_preset, list_presets
from .simulation import Simulation


class MatrixPayload(BaseModel):
    matrix: List[List[float]]

    @field_validator("matrix")
    @classmethod
    def validate_square(cls, matrix: List[List[float]]) -> List[List[float]]:
        if not matrix:
            raise ValueError("Matrix must not be empty")
        size = len(matrix)
        for row in matrix:
            if len(row) != size:
                raise ValueError("Matrix must be square")
        return matrix


class ConfigPayload(BaseModel):
    width: Optional[float] = None
    height: Optional[float] = None
    species_count: Optional[int] = Field(default=None, ge=1)
    particle_count: Optional[int] = Field(default=None, ge=1)
    interaction_radius: Optional[float] = Field(default=None, gt=0.0)
    time_step: Optional[float] = Field(default=None, gt=0.0)
    velocity_decay: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    max_speed: Optional[float] = Field(default=None, gt=0.0)
    frame_interval: Optional[float] = Field(default=None, gt=0.0)
    acceleration_limit: Optional[float] = Field(default=None, gt=0.0)


class ConfigRequest(BaseModel):
    config: Optional[ConfigPayload] = None
    reset_matrix: bool = False
    matrix: Optional[List[List[float]]] = None
    model_config = ConfigDict(extra="allow")

    def to_payload(self) -> ConfigPayload:
        data: Dict[str, Any] = {}
        if self.config:
            data.update(self.config.model_dump(exclude_none=True))
        if self.model_extra:
            data.update({key: value for key, value in self.model_extra.items() if value is not None})
        return ConfigPayload(**data)


app = FastAPI(title="Particle Life Sandbox")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_config = DEFAULT_CONFIG
_simulation = Simulation(_config)
_state_lock = asyncio.Lock()
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


def _replace_config(current: SimulationConfig, updates: ConfigPayload) -> SimulationConfig:
    data = current.as_dict()
    for key, value in updates.model_dump(exclude_none=True).items():
        data[key] = value
    return SimulationConfig(**data)


def _apply_config_update(
    config_payload: ConfigPayload,
    reset_matrix: bool = False,
    matrix_override: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    global _config, _simulation
    current_config = _simulation.config
    new_config = _replace_config(current_config, config_payload)
    species_changed = new_config.species_count != current_config.species_count

    if matrix_override is not None:
        matrix_to_apply = matrix_override
    elif reset_matrix or species_changed:
        matrix_to_apply = None
    else:
        matrix_to_apply = _simulation.matrix

    _config = new_config
    _simulation.reset(new_config, matrix_to_apply)
    return {"config": _simulation.config.as_dict(), "matrix": _simulation.matrix}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def index() -> FileResponse:
    if not _frontend_dir.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(_frontend_dir / "index.html")


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    async with _state_lock:
        return {"config": _simulation.config.as_dict(), "matrix": _simulation.matrix}


@app.post("/config")
async def update_config(payload: ConfigRequest) -> Dict[str, Any]:
    async with _state_lock:
        try:
            return _apply_config_update(
                payload.to_payload(), reset_matrix=payload.reset_matrix, matrix_override=payload.matrix
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/presets")
async def get_presets() -> List[Dict[str, Any]]:
    return [
        {"name": preset.name, "description": preset.description, "matrix": preset.matrix}
        for preset in list_presets()
    ]


@app.post("/presets/{name}")
async def apply_preset(name: str) -> Dict[str, Any]:
    global _simulation
    async with _state_lock:
        try:
            preset = get_preset(name)
            _simulation.set_matrix(preset.matrix)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"matrix": _simulation.matrix, "preset": preset.name}


@app.post("/matrix")
async def update_matrix(payload: MatrixPayload) -> Dict[str, Any]:
    global _simulation
    async with _state_lock:
        try:
            _simulation.set_matrix(payload.matrix)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"matrix": _simulation.matrix}


async def _listener(websocket: WebSocket, queue: asyncio.Queue) -> None:
    try:
        while True:
            message = await websocket.receive_json()
            await queue.put(message)
    except WebSocketDisconnect:
        pass


async def _handle_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    global _simulation
    message_type = message.get("type")
    if message_type == "update_matrix":
        matrix = message.get("matrix")
        if matrix is None:
            raise ValueError("Message missing 'matrix'")
        _simulation.set_matrix(matrix)
        return {"matrix": _simulation.matrix}
    if message_type == "use_preset":
        name = message.get("name")
        if name is None:
            raise ValueError("Message missing 'name'")
        preset = get_preset(name)
        _simulation.set_matrix(preset.matrix)
        return {"matrix": _simulation.matrix, "preset": preset.name}
    if message_type == "update_config":
        config_payload = ConfigPayload(**message.get("config", {}))
        matrix = message.get("matrix")
        reset_matrix = bool(message.get("reset_matrix"))
        return _apply_config_update(config_payload, reset_matrix=reset_matrix, matrix_override=matrix)
    raise ValueError(f"Unknown message type '{message_type}'")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    listener = asyncio.create_task(_listener(websocket, queue))
    try:
        while True:
            async with _state_lock:
                while not queue.empty():
                    message = await queue.get()
                    try:
                        await _handle_message(message)
                    except ValueError as exc:
                        await websocket.send_json({"type": "error", "detail": str(exc)})
                _simulation.step()
                state = _simulation.get_state()
                state["matrix"] = _simulation.matrix
                state["config"] = _simulation.config.as_dict()
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            try:
                await websocket.send_json({"type": "state", "payload": state})
            except RuntimeError:
                break
            await asyncio.sleep(_simulation.config.frame_interval)
    except WebSocketDisconnect:
        pass
    finally:
        listener.cancel()
