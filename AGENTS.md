# AGENT.md — Particle Life Sandbox (Python backend)

> Build a **clean, minimal, extensible** Particle Life sandbox I can play with, with a **side-panel matrix editor** (editable interaction matrix for all species) and a **bounded workspace** where particles **reflect** on collision with the boundary. Keep architecture simple, avoid over-engineering (e.g., no unnecessary class hierarchies, no blanket `try/except` scaffolding).

---

## 1) Mission & Outcomes

**Mission**
- Implement a 2D Particle Life playground with a Python backend and a lightweight web UI.
- Let me **interactively edit the S×S interaction matrix K** in a side panel and see behavior update live.
- Enforce a **rectangular bounded world** with **reflective boundaries** (specular reflection on walls).
- Keep code **concise, readable, testable**, and **ready to extend for swarm-robotics experiments**.

**Primary Outcomes**
- Smooth real-time simulation (>= 60 FPS client-side rendering target; backend tick decoupled).
- Live editing of:
  - number of particles, number of species S
  - interaction matrix `K[i][j]` (slider or numeric input per cell)
  - force-kernel parameters (short-range repulsion, mid-range attraction, cutoff)
  - damping, timestep, noise level
- Controls: play/pause/reset, seed, speed, save/load presets (JSON).
- Deterministic option via RNG seed.

**Non-Goals (for v1)**
- No heavy dependency frameworks unless clearly justified.
- No persistent DB; presets saved as JSON to disk (or browser download).
- No GPU acceleration (keep CPU-friendly, but structure so it can be added later).

---

## 2) Architecture (minimal, clear layers)

**Backend (Python)**
- Framework: **FastAPI** (simple REST + WebSocket streaming).
- Core modules:
  - `simulation.py`: state structs, neighbor search, force calc, integrator, boundary handling.
  - `config.py`: dataclasses for `SimConfig`, `KernelParams`, and validation.
  - `presets.py`: named presets (dict of configs + matrices).
  - `api.py`: REST endpoints (get/set config, reset), WebSocket for frames.
- Concurrency: a single simulation loop running on an asyncio task; tick at fixed dt; broadcast snapshots to WS subscribers.
- Performance: uniform grid (cell-linked list) neighbor search; avoid per-frame allocations.

**Frontend (web)**
- One static page (`index.html`) with:
  - `<canvas>` for rendering.
  - **Side panel** with an **editable S×S matrix** (grid of inputs/sliders).
  - Controls: play/pause/reset, counts, kernel sliders, speed, seed, noise, presets (save/load).
- Vanilla TypeScript (or ES6) + minimal CSS; no large UI framework required.
- Rendering decoupled from backend tick: draw last received state; use WS for state stream, REST for config/mutations.

**File Tree (target)**

    particle-life/
    ├── backend/
    │   ├── api.py
    │   ├── simulation.py
    │   ├── config.py
    │   ├── presets.py
    │   ├── schemas.py
    │   ├── requirements.txt
    │   └── tests/
    │       ├── test_forces.py
    │       ├── test_boundaries.py
    │       └── test_integrator.py
    ├── frontend/
    │   ├── index.html
    │   ├── app.js # or app.ts compiled to app.js
    │   └── styles.css
    ├── scripts/
    │   └── run_dev.sh
    ├── README.md
    └── AGENT.md

---

## 3) Data & Interfaces

**State Structures**
- `Particles`: arrays (SoA) for `x`, `y`, `vx`, `vy`, `species` (int in `[0..S-1]`).
- `SimConfig`:
  - world: `width`, `height`, `boundary="reflect"`
  - time: `dt`, `damping`, `noise_std`, `seed`
  - counts: `n_particles`, `n_species`
  - kernel: `repel_radius`, `attract_radius`, `cutoff_radius`, `repel_strength`, `attract_strength`
  - matrix: `K` (S×S floats in [-1,1] initially)
- **Kernel**: piecewise radial force:
  - $r < r_{rep}$: strong repulsion
  - $r_{rep} \le r < r_{att}$: attraction scaled by `K[si][sj]`
  - $r \ge r_{cut}$: 0
  - Continuous at the joins; capped at max magnitude to ensure stability.

**Boundary Handling**
- Reflective walls at $x \in [0,W]$, $y \in [0,H]$.
- If next pos crosses boundary, clamp to wall and reflect velocity component (e.g., `vx = -vx * restitution`, restitution ≈ 1.0).
- Ensure no particle gets stuck by backing out to valid region.

**API**
- `GET /config` → current `SimConfig`
- `POST /config` → replace/patch config (validate + apply)
- `POST /reset` → reinitialize state with current config
- `GET /presets` → list preset names
- `GET /presets/{name}` → fetch preset config
- `POST /presets` → upload/save preset JSON
- **WS** `ws://.../state` → stream snapshots (e.g., every N ticks), payload:

        {
          "t": float,
          "particles": { "x": [...], "y": [...], "vx": [...], "vy": [...], "species": [...] },
          "n": int, "S": int
        }

---

## 4) Simulation Loop & Numerics

**Integrator**
- Semi-implicit Euler (symplectic Euler) for stability. The steps are applied each tick:

        v += dt * a
        v *= damping
        x += dt * v

        if noise_std > 0:
            v += gaussian_noise(scale=noise_std)

**Neighbor Search**
- Uniform grid with cell size ≈ `cutoff_radius`.
- For each particle, only inspect its own + 8 neighbor cells.

**Force Accumulation**
- For each pair of particles `(i, j)` within the `cutoff_radius`:

        s_i = species[i]
        s_j = species[j]
        k = K[s_i][s_j]

        r = distance(i, j)
        force_magnitude = k * attract_term(r) - repel_term(r)

        direction_ij = (pos_j - pos_i) / r
        force_on_i = force_magnitude * direction_ij
        total_force[i] += force_on_i
        total_force[j] -= force_on_i

**Stability**
- Cap per-pair force and total acceleration.
- Optional softening `epsilon` to avoid division by zero.

---

## 5) Frontend UX Requirements

**Matrix Editor (Side Panel)**
- Render an S×S grid; each cell editable (slider or numeric).
- Provide buttons: randomize matrix, zero-diagonal toggle, symmetry toggle (`K=0.5*(K+K^T)`), normalize ranges.
- Live-apply: on change, `POST /config` with updated `K`; server restarts forces immediately (no reset required).

**Controls**
- Play/Pause, Reset
- Species count S (rebuild matrix UI)
- Particle count n (on the fly: add/remove with random positions)
- Kernel sliders: radii & strengths
- Damping, noise, speed
- Seed (deterministic runs)
- Presets: dropdown + Save/Load JSON (download/upload)

**Rendering**
- Draw particles as circles; species → distinct hues (computed client-side).
- Option to show boundary box, velocity vectors (toggle).

---

## 6) Extensibility for Swarm Robotics

**Hooks**
- Export **macro metrics** each second: cluster count & size distribution, coverage %, collision rate, neighbor churn.
- Pluggable **task layers**: e.g., goal points, obstacles as external potential $U_{env}(x,y)$ (add to forces).
- Optional **role masks**: special species as “scouts/guards/carriers”.
- **Ground-truth-free controls**: ensure all control laws depend on relative positions only (or RSSI-like range).

**Future Bridges**
- ROS2 adapter (topic for particle states).
- Replace kernel with learned function $F_\theta(r, s_i, s_j)$; training via inverse design (target macro metrics).

---

## 7) Testing & Quality

**Unit Tests (pytest)**
- `test_forces.py`: piecewise kernel continuity, sign conventions, caps.
- `test_boundaries.py`: reflection logic, energy behavior with restitution.
- `test_integrator.py`: stability under simple harmonic potential; invariants within tolerance.
- `test_neighbors.py`: neighbor search correctness vs. brute-force on small sets.

**Validation Demos**
- Preset A (2 species): flocking/segregation.
- Preset B (3 species): cyclic chase (rock–paper–scissors style).
- Preset C (obstacles on): boundary hugging & reflection.

**Style & Simplicity**
- Prefer pure functions + small modules.
- Avoid global singletons; pass config/state explicitly.
- No blanket `try/except`; only narrow, necessary exception handling with clear messages.
- Type hints everywhere; docstrings for public functions.

---

## 8) Initial Preset Example (JSON)

    {
      "name": "tri-cyclic",
      "n_particles": 1500,
      "n_species": 3,
      "seed": 42,
      "dt": 0.02,
      "damping": 0.98,
      "noise_std": 0.0,
      "world": { "width": 800, "height": 600, "boundary": "reflect" },
      "kernel": {
        "repel_radius": 4.0,
        "attract_radius": 24.0,
        "cutoff_radius": 36.0,
        "repel_strength": 1.8,
        "attract_strength": 0.8
      },
      "K": [
        [ -0.2,  0.9, -0.4 ],
        [ -0.4, -0.2,  0.9 ],
        [  0.9, -0.4, -0.2 ]
      ]
    }