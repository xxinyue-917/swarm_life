# Crazyflie Deployment — Particle Life on Physical Drones

## Goal

Deploy the particle life dual-matrix swarm system on 10 Crazyflie 2.1 drones with Vicon motion capture. Demonstrate that the same K_pos/K_rot interaction matrices that produce emergent behaviors in simulation produce recognizable collective behaviors on physical robots.

---

## Progress Checklist

Track your progress here. Check each box as you complete it and note any issues.

### Phase 0 — Environment Setup
- [ ] Ubuntu 24.04 running
- [ ] ROS2 Jazzy installed (`ros2 topic list` shows `/rosout`)
- [ ] `source /opt/ros/jazzy/setup.bash` added to `.bashrc`
- [ ] Crazyswarm2 cloned and built (`ros2 pkg list | grep crazyflie` shows packages)
- [ ] `source ~/ros2_ws/install/setup.bash` added to `.bashrc`
- [ ] `python3 -c "import cflib; print('ok')"` works
- [ ] `cfclient` installed (`pip install cfclient`)

### Phase 1 — Hardware Verification (NO FLYING)
- [ ] Crazyradio PA #1 plugged in, udev rules installed
- [ ] Crazyradio PA #2 plugged in (separate USB host port, not a hub)
- [ ] `cfclient` GUI detects radio(s)
- [ ] Power on 1 Crazyflie (propellers OFF) → cfclient connects, shows IMU data
- [ ] Firmware updated to latest stable via cfclient
- [ ] Repeat firmware check for all 10 drones
- [ ] Vicon Tracker software running on Vicon PC
- [ ] 1 Crazyflie has 4 asymmetric reflective markers attached
- [ ] Vicon Tracker detects and tracks the rigid body (shows position in Tracker GUI)
- [ ] All 10 drones have unique marker configurations, all tracked in Vicon

### Phase 2 — Vicon → ROS2 Pipeline
- [ ] `motion_capture_tracking` node launches without errors
- [ ] `ros2 topic echo /cf1/pose` shows live position data while moving drone by hand
- [ ] Verify coordinate frame: move drone +X in lab → `pose.position.x` increases (not flipped)
- [ ] Verify coordinate frame: move drone +Y in lab → `pose.position.y` increases
- [ ] All 10 drones publishing poses on `/cf1/pose` through `/cf10/pose`
- [ ] Pose update rate confirmed ≥ 100 Hz (`ros2 topic hz /cf1/pose`)

### Phase 3 — M1: Single Drone Hover
- [ ] Safety net / cage around flight area
- [ ] Emergency stop key binding verified (press ESC → lands drone)
- [ ] Propellers installed on 1 drone
- [ ] Drone takes off to z=1.0m via Crazyswarm2 `takeoff()` command
- [ ] Drone hovers stably for 60 seconds
- [ ] Geofence test: gently push drone toward boundary → lands automatically
- [ ] Vicon watchdog test: cover markers briefly → drone holds zero velocity
- [ ] Drone lands cleanly via `land()` command
- [ ] Battery voltage logged during flight (~3.0V/cell = low)

### Phase 4 — M2: Two-Drone K_pos Chase
- [ ] 2 drones hover simultaneously (one per radio is fine)
- [ ] `particle_life_node` scaffold running: subscribes to poses, publishes cmdVelocityWorld
- [ ] Species assigned (drone 1 = species 0, drone 2 = species 1)
- [ ] K_pos-only force computation running at 30 Hz (K_rot = 0)
- [ ] `force_output_scale` tuned (start 0.1, increase until motion visible)
- [ ] `v_max` set to 0.15 m/s
- [ ] Min separation check working (< 30cm → both land)
- [ ] Preset: `2_chase` (K₁₂>0, K₂₁<0) — drone 1 chases drone 2
- [ ] Preset: `2_move_together` — both attracted, move as a pair
- [ ] Preset: `2_encapsulate` — one surrounds the other
- [ ] `v_max` increased to 0.3 m/s after stable runs
- [ ] Rosbag recorded for all presets

### Phase 5 — M3: Ten-Drone Particle Life
- [ ] Warm-up: 2 drones hover ✓ → add 2 more → add 2 more → add 2 more → all 10
- [ ] All 10 drones hover simultaneously, safety layer active
- [ ] 2-species preset loaded (5+5 split), K_pos only, v_max=0.2 m/s
- [ ] Recognizable collective behavior (chase, encapsulate, or cluster)
- [ ] K_rot heading-vector proxy implemented and tested
- [ ] K_rot enabled — visible rotational/orbital component
- [ ] `v_max` increased to 0.5 m/s
- [ ] 3-species preset tested (3+3+4 split, cyclic chase)
- [ ] Digital twin running (pygame visualization fed by Vicon positions)
- [ ] Final recording: rosbag + overhead camera video
- [ ] Side-by-side comparison: simulation replay vs physical flight

### Phase 6 — Analysis and Documentation
- [ ] Trajectory comparison: simulation vs physical for each preset
- [ ] Metrics computed on rosbag data (centroid distance, spacing, speed)
- [ ] Failure modes documented (what worked, what didn't, parameter changes)
- [ ] Video edited for paper/presentation
- [ ] Results written up

### Open TODOs

- [ ] **Custom simulation frontend** — Crazyswarm2's built-in sim GUI is hard to use. Build a lightweight custom visualization that subscribes to drone poses from ROS and renders them (3D scatter, optionally reusing the swarm_life pygame aesthetic). Design doc at `SIM_DESIGN.md`.

### Issues Log

Record issues as you encounter them:

| Date | Phase | Issue | Resolution |
|------|-------|-------|------------|
| | | | |
| | | | |
| | | | |

### Parameter Tuning Log

Track parameter changes during deployment:

| Date | Parameter | Old Value | New Value | Reason |
|------|-----------|-----------|-----------|--------|
| | force_output_scale | 0.1 | | |
| | v_max | 0.2 | | |
| | r_max | 2.0 | | |
| | | | | |

---

## Hardware

| Component | Quantity | Notes |
|-----------|----------|-------|
| Crazyflie 2.1 | 10 | Firmware should be latest stable |
| Crazyradio PA | 2 | 5 drones each, on separate USB host ports (not a hub) |
| Vicon motion capture | 1 system | Minimum 6 cameras for 4×4m arena |
| Reflective markers | 4 per drone | Unique asymmetric configurations per drone |
| Ubuntu 24.04 PC | 1 | Ground station running ROS2 + force computation |
| Spare batteries | 10+ | ~7 min flight time per battery |

## Software Stack

```
Ubuntu 24.04 LTS
├── ROS2 Jazzy Jalisco (Tier 1 on 24.04)
├── Crazyswarm2 (main branch, Jazzy-compatible)
│   ├── crazyflie_server (radio communication)
│   ├── motion_capture_tracking (Vicon → ROS2 poses)
│   └── crazyflie_interfaces (msg/srv definitions)
├── particle_life_node (custom, this project)
│   ├── Force computation (K_pos + optional K_rot)
│   ├── Safety layer (geofence, separation, watchdogs)
│   └── Coordinate transform (Vicon → sim → velocity)
└── Digital twin (optional: pygame visualization fed by Vicon)
```

### Why Crazyswarm2 + ROS2 (not pure cflib)

A multi-agent debate evaluated cflib-only (~150 lines, no ROS) vs Crazyswarm2. Conclusion:

- **cflib wins** for ≤3 drones, experienced developer, rapid prototyping
- **Crazyswarm2 wins** for 10 drones, new-to-hardware user, because:
  - Built-in emergency stop (hardware kill-switch)
  - Synchronized takeoff/landing for all drones
  - `rosbag` recording for post-mortem analysis and reproducibility
  - Vicon integration via `motion_capture_tracking` (no custom bridge)
  - Parameter management via YAML configs (not hardcoded URIs)

### ROS2 Version Note

ROS1 Noetic is EOL and does NOT support Ubuntu 24.04. ROS2 Jazzy is the Tier 1 match. Crazyswarm2's `main` branch targets Jazzy as of mid-2025. Verify binary package availability; if unavailable, build from source (~1 hour):

```bash
sudo apt install ros-jazzy-desktop python3-colcon-common-extensions
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/IMRCLab/crazyswarm2 --recursive
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Ubuntu 24.04 + ROS2 Jazzy                           │
│                                                      │
│  ┌──────────┐     ┌─────────────────────────────┐    │
│  │  Vicon   │────▶│  motion_capture_tracking     │    │
│  │  System  │     │  (VRPN → /cfN/pose at 100Hz) │    │
│  └──────────┘     └─────────────────────────────┘    │
│                              │                        │
│                    10× /cfN/pose topics               │
│                              ▼                        │
│  ┌────────────────────────────────────────────────┐  │
│  │  particle_life_node (single node, 30Hz timer)  │  │
│  │                                                │  │
│  │  ┌─────────────┐   ┌────────────────────┐      │  │
│  │  │  Pose Store │   │  Safety Layer      │      │  │
│  │  │  (dict,     │──▶│  1. Vicon watchdog │      │  │
│  │  │  async      │   │  2. Geofence       │      │  │
│  │  │  callbacks) │   │  3. Min separation │      │  │
│  │  └─────────────┘   │  4. Radio watchdog │      │  │
│  │                     │  5. Speed clamp    │      │  │
│  │                     └────────┬───────────┘      │  │
│  │                              │ (all pass)       │  │
│  │                              ▼                  │  │
│  │                     ┌────────────────────┐      │  │
│  │                     │  Force Computation │      │  │
│  │                     │  (K_pos + K_rot)   │      │  │
│  │                     │  Same NumPy code   │      │  │
│  │                     │  as simulation     │      │  │
│  │                     └────────┬───────────┘      │  │
│  │                              │                  │  │
│  │                              ▼                  │  │
│  │                     ┌────────────────────┐      │  │
│  │                     │  Coord Transform   │      │  │
│  │                     │  sim → m/s world   │      │  │
│  │                     └────────┬───────────┘      │  │
│  │                              │                  │  │
│  │                 10× cmdVelocityWorld             │  │
│  └──────────────────────┬─────────────────────────┘  │
│                         ▼                             │
│  ┌──────────────────────────────────────────────────┐│
│  │  Crazyswarm2 Server                              ││
│  │  ┌─────────────┐  ┌─────────────┐               ││
│  │  │ CrazyradioPA│  │ CrazyradioPA│               ││
│  │  │ (cf1-cf5)   │  │ (cf6-cf10)  │               ││
│  │  └─────────────┘  └─────────────┘               ││
│  └──────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
                         │
           ┌─────────────┼──────────────┐
           ▼             ▼              ▼
        [cf1]  ...    [cf5]  ...    [cf10]
```

## Control Mode: cmdVelocityWorld

Crazyflie accepts velocity setpoints in **world frame** via `cmdVelocityWorld(vx, vy, vz, yaw_rate)`:
- `vx, vy`: horizontal velocity in m/s (maps directly from particle life F_x, F_y)
- `vz`: 0.0 (altitude held by onboard PID at fixed z)
- `yaw_rate`: 0.0 for M1/M2; can carry K_rot heading coupling in M3

Update rate: **30 Hz** (sufficient for particle life timescales; well within radio bandwidth for 2 radios × 5 drones).

## Coordinate Frame Mapping

```
Vicon Lab Frame (mm, Z-up)
    │
    │  ÷ 1000 (handled by vicon_bridge)
    ▼
ROS World Frame (meters, Z-up)
    │
    │  Subtract arena origin + scale
    │  x_sim = (x_ros - origin_x) × ARENA_TO_SIM_SCALE
    │  y_sim = (y_ros - origin_y) × ARENA_TO_SIM_SCALE
    ▼
Particle Life Sim Coords (sim_width × sim_height)
    │
    │  Force computation: v_sim = F(positions, K_pos, K_rot)
    │
    │  Back to world: vx_world = v_sim_x / ARENA_TO_SIM_SCALE × force_output_scale
    ▼
cmdVelocityWorld (m/s, world frame)
```

Example: physical arena = 3m × 3m, simulation = 10 × 10 → `ARENA_TO_SIM_SCALE = 10/3 ≈ 3.33`. A sim velocity of 1.0 unit/s → 0.3 m/s in the real world. Tune `force_output_scale` to keep physical velocities in the 0.1–0.5 m/s range.

**Day-1 verification**: Move a drone by hand, confirm `/cfN/pose.position.x` increases in the expected direction. Fix any axis flips before proceeding.

## Safety Layer (Non-Negotiable)

Checked **every control cycle** (30 Hz), in this order. Any failure skips subsequent checks and triggers the corresponding response.

| # | Check | Threshold | Response |
|---|-------|-----------|----------|
| 1 | Vicon watchdog | Any pose age > 150 ms | Zero velocity to ALL drones (hover) |
| 2 | Geofence | Any drone outside arena − 20 cm margin | Emergency land ALL |
| 3 | Min separation | Any pair < 30 cm | Emergency land ALL |
| 4 | Radio watchdog | link_quality < 60% for any drone | Land that drone |
| 5 | Speed clamp | Per-drone velocity magnitude | Clip to v_max (start 0.2, increase to 0.5 m/s) |
| 6 | Emergency stop | ESC key or hardware button | send_stop_setpoint → land ALL |

**Critical design rule**: the safety layer runs in the main timer callback **before** force computation. If safety fails, force computation is skipped entirely — no stale/phantom forces are ever sent.

**Warm-up protocol**: Never go from 0 to 10 drones. Start with 2 → confirm all 6 safety checks work → add 2 more → repeat. Log every safety trigger for post-mortem review.

## K_rot Mapping (Physical Analog)

### The Problem

In simulation, K_rot tangential forces depend on neighbor angular velocity `ω_j`:
```
F_tangential = K_rot[i,j] × a_rot × (ω_j/ω_max) × (1/r) × t_hat
```
Drones don't have a natural "angular velocity around a neighbor" — they translate, not orbit.

### Solution: Heading-Vector Proxy

Replace `ω_j` with the drone's velocity direction:
```python
# For each neighbor j of drone i:
v_j = velocity_of_drone_j  # from Vicon differentiation or last command
t_hat = perpendicular_to(pos_j - pos_i)  # tangential direction
heading_component = dot(v_j, t_hat)  # how much j moves tangentially
F_tangential = K_rot[i,j] * a_rot * heading_component * (1 - r_norm) * t_hat
```

This preserves the physical intent: "a fast-moving neighbor deflects you tangentially" — without requiring spin.

### Phased Rollout

| Milestone | K_rot Handling |
|-----------|----------------|
| M1 (hover) | N/A |
| M2 (two-drone) | K_rot = 0 (K_pos only: chase, encapsulate, cluster) |
| M3 (ten-drone) | Add heading-vector proxy; compare with K_rot=0 baseline |

## Milestone Plan

### M1 — Single Drone Hover (Days 1–4)

**Goal**: One Crazyflie hovering stably at z=1.0m with full safety layer active.

| Day | Task | Success Criterion |
|-----|------|-------------------|
| 1 | Install ROS2 Jazzy + Crazyswarm2. Flash Crazyflie firmware. Set up udev rules for Crazyradio PA. | `cfclient` GUI connects to drone |
| 2 | Configure Vicon rigid body (4 asymmetric markers). Test `motion_capture_tracking` node. | `ros2 topic echo /cf1/pose` shows live position |
| 3 | Write `particle_life_node` scaffold: pose subscription, 30Hz timer, safety layer, cmdVelocityWorld publisher. Hover = constant zero velocity + altitude hold. | Drone hovers at z=1.0m for 60s |
| 4 | Test all safety triggers: move drone to geofence boundary (lands?), cover Vicon markers (watchdog triggers?), press ESC (emergency stop?). | All 6 safety checks verified |

**Common blockers**: Crazyradio udev rules not set (`sudo cp 99-crazyflie.rules /etc/udev/rules.d/`), Vicon coordinate frame flipped, onboard EKF not receiving external position (enable `motion` deck parameter).

### M2 — Two-Drone K_pos Chase (Days 5–9)

**Goal**: Two drones executing a 2-species particle life preset (e.g., `2_chase`) with K_pos only.

| Day | Task | Success Criterion |
|-----|------|-------------------|
| 5 | Add second drone + second Crazyradio PA. Configure Crazyswarm2 YAML for 2 drones on separate channels. | Both drones hover simultaneously |
| 6 | Port `_compute_velocities_jit` force kernel to the ROS2 node (N=2 version, K_rot=0). Map sim velocities to cmdVelocityWorld. | Force computation runs at 30Hz, velocities published |
| 7 | Load `2_chase` preset (K_pos asymmetric: K₁₂>0, K₂₁<0). Fly with v_max=0.15 m/s. | Drone 1 pursues drone 2; drone 2 flees. Recognizable chase. |
| 8 | Test `2_encapsulate` and `2_move_together` presets. Tune force_output_scale for stable motion. | Visually identifiable behaviors matching simulation |
| 9 | Record rosbag of all 3 presets. Compare trajectories with simulation replay. | Rosbags saved, qualitative match confirmed |

**Key parameter tuning**:
- `force_output_scale`: start at 0.1, increase until motion is visible but not violent
- `v_max`: 0.15 → 0.3 m/s as confidence grows
- If drones oscillate: reduce force_output_scale or increase simulation dt (slower response)

### M3 — Ten-Drone Full Particle Life (Days 10–14)

**Goal**: All 10 drones running a multi-species particle life preset.

| Day | Task | Success Criterion |
|-----|------|-------------------|
| 10 | Scale to 10 drones. Configure 5 per radio. Warm-up: 2 → 4 → 6 → 8 → 10. | All 10 hover simultaneously, safety layer active |
| 11 | Load 2-species preset (5+5 split). Run K_pos-only at v_max=0.2 m/s. | Two recognizable species clusters interacting |
| 12 | Add K_rot heading-vector proxy. Compare with K_rot=0 baseline. | Visible rotational/orbital component in behavior |
| 13 | Test additional presets: 2_sun_earth (orbital), 3_chase (cyclic pursuit if 3+ species). Increase v_max to 0.5 m/s. | Multiple preset behaviors demonstrated |
| 14 | Final recording session. Rosbag + overhead camera video for paper. Run digital twin (pygame) alongside physical swarm. | Publication-quality data and video |

## Species Assignment

With 10 drones, species configurations:

| Config | Species | Drones per species | Best presets |
|--------|---------|-------------------|--------------|
| 2-species | 2 | 5 + 5 | 2_chase, 2_encapsulate, 2_move_together |
| 2-species (asymmetric) | 2 | 3 + 7 | 2_sun_earth (3 = sun, 7 = planet) |
| 3-species | 3 | 3 + 3 + 4 | 3_chase (cyclic pursuit) |
| 5-species | 5 | 2 each | Planet preset (if per-pair beta implemented) |

**Critical warning (from debate)**: N=10 is well below simulation particle counts (50–200). Emergent behaviors that rely on statistical averaging over many particles may not survive at this scale. **Before hardware day**: run every target preset in simulation at N=5-per-species and check whether the behavior is still visually recognizable. Lower expectations or reframe as "proof of concept" if behaviors collapse.

## Digital Twin (Optional but Recommended)

Feed Vicon positions into the existing pygame renderer as a live mirror:

```python
# In the existing particle_life pygame loop:
# Replace self.positions with Vicon-fed positions
positions_from_vicon = get_all_drone_positions()  # from ROS2 subscriber
for i, pos in enumerate(positions_from_vicon):
    self.positions[i] = [pos.x * SCALE, pos.y * SCALE]
```

Zero change to the visualization layer — swap the position source. This gives real-time visual feedback during experiments and allows side-by-side simulation vs reality comparison.

## File Structure

```
crazyflie_deployment/
├── PLAN.md                     # This document
├── config/
│   ├── crazyflies.yaml         # Drone URIs, channels, marker configs
│   ├── arena.yaml              # Geofence bounds, origin, scale factor
│   └── presets/                # K_pos/K_rot matrices for physical deployment
├── src/
│   ├── particle_life_node.py   # Main ROS2 node (force computation + safety)
│   ├── safety.py               # Safety layer (geofence, watchdog, separation)
│   ├── force_computation.py    # Ported from src/particle_life.py
│   ├── coordinate_transform.py # Vicon ↔ sim ↔ world transforms
│   └── digital_twin.py         # Optional pygame visualization from Vicon
├── launch/
│   └── particle_life.launch.py # Launch file for all nodes
└── results/                    # Rosbags, videos (gitignored)
```

## Latency Budget

| Stage | Latency | Notes |
|-------|---------|-------|
| Vicon capture → PC | ~4 ms | VRPN over ethernet |
| ROS2 transport | ~2 ms | DDS serialization |
| Safety + force computation | < 1 ms | N=10, O(N²) = 90 pairs |
| Radio TX (per 5-drone batch) | ~8 ms | Crazyradio PA TDM |
| Crazyflie onboard PID | ~2 ms | 500 Hz inner loop |
| **Total** | **~17 ms** | Well within 33 ms (30 Hz) budget |

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| N=10 too few for legible emergence | High | Pre-validate in simulation at N=5 per species |
| Crazyswarm2 Jazzy build issues | Medium | Fall back to Humble in Docker |
| Vicon occlusion with 10 drones in tight formation | Medium | Increase camera count; add ceiling cameras |
| Battery asymmetry (drones drop out mid-experiment) | Medium | Charge all batteries together; swap as a batch |
| K_rot heading proxy diverges from simulation | Low | Compare sim and hardware trajectories quantitatively |
| Prop wash destabilizes neighbors at close range | Medium | Enforce min separation > 30 cm; reduce K_pos attraction |

## Related Documents

- `CLAUDE.md` — Simulation codebase architecture and physics model
- `behavior_reproduction/PLAN.md` — Behavior reproduction methodology
- `claudedocs/behavioral_universality.md` — Universality argument (simulation → physical robots)
- `characterization/PLAN.md` — Parameter sweep results for choosing deployment presets
