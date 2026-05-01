# Custom Simulation Frontend — Design Document

## Problem

Crazyswarm2's built-in sim (`backend:=sim`) has a clunky pygame GUI that's hard to read and doesn't match this project's visual style. We want a better simulation frontend — just for personal visualization while developing particle-life force controllers.

## Requirements (ranked)

1. **Simple to use.** No week-long learning curve. Stay close to the existing pygame workflow.
2. **Read-only is fine.** The frontend just needs to visualize drone poses coming from ROS (whether Crazyswarm2's sim backend or real Vicon). It doesn't need to control anything.
3. **Matches the existing codebase aesthetic.** The `swarm_life/src/particle_life.py` drawing style (anti-aliased circles, species colors, centroid spine) is already tuned for intuition.
4. **Low lines of code.** Target <100 lines of new code. The smaller the surface area, the less maintenance.
5. **Runs alongside Crazyswarm2.** Subscribes to `/cfN/pose` — works identically in sim and on real hardware.
6. Not needed now, but nice eventually: 3D camera, video recording, force-vector overlays.

## Three options considered

Three agents independently argued for different designs. Summary below; full arguments in the appendix.

### Option A — Reuse `swarm_life/src/particle_life.py`

Instantiate `ParticleLife` headless (without running `step()`), then overwrite `self.positions` from a ROS2 subscriber each frame. All the drawing code (`draw_particles`, `draw_centroid_spine`, matrix overlay) already works because it reads only `self.positions`.

- **New code**: ~60 lines, one file.
- **Dimensionality**: 2D top-down. Arena is 3m × 3m × 1m so altitude barely varies; represent z as a color tint or particle radius if needed.
- **Pros**: Familiar aesthetic. Zero install. `ParticleLife3D` is a drop-in if we later need true 3D.
- **Cons**: Single-process fragility (pygame + rclpy in one thread can stall if ROS hiccups). Frame-mirror gotcha (Vicon y-up vs pygame y-down). Read-only — matrix edits don't propagate to drones.

### Option B — rviz2 + Foxglove (ROS-native)

Publish drone poses as tf2 transforms + `visualization_msgs/MarkerArray`. rviz2 (already installed with Jazzy) renders them with a saved `.rviz` layout.

- **New code**: ~80 lines for the marker publisher + an `.rviz` config file.
- **Pros**: Industry standard. Same visualization for sim and hardware. Record to rosbag / .mcap. Selectable layers (drones, force vectors, waypoints) without custom code.
- **Cons**: ~1 week of learning curve (tf2, QoS, launch quirks). Utilitarian look — not paper-quality. Debugging "why isn't my marker showing?" is harder than `print()` in pygame. **Agent 2's own honest verdict: stay in pygame for research; switch to rviz2 only when flying real drones.**

### Option C — PyVista (VTK-based 3D)

Native desktop window with GPU-accelerated 3D. NumPy-first API matches this codebase's style. Built-in video recording via `plotter.open_movie("run.mp4")`.

- **New code**: ~80 lines + `pip install pyvista` (~150 MB of VTK).
- **Pros**: True 3D with orbital camera, trails, arena box, drone labels. Video recording out of the box. Feels like pygame but 3D.
- **Cons**: 150 MB install. Threading rclpy in background requires care on shutdown. Custom UI widgets (sliders, matrix editor) are clunkier than pygame.

## Decision

**Recommended: Option A (reuse `particle_life.py`) as the default, with Option C (PyVista) as the upgrade path.**

Rationale:
- The user's stated goal is "I just need to use it for myself to visualize it" and "It doesn't have to be very complicated." Option A is the simplest, fastest path to a working viewer — literally tens of lines reusing code already written.
- Agent 2 (the pro-rviz advocate) *agreed* rviz2 isn't the right call for the research phase. So B is deferred until real hardware integration.
- Option C adds real value (3D, video) but costs a dependency and a threading model. Worth adopting *later* if the 2D view proves insufficient for papers/demos. Keep it as a one-liner swap — `particle_life.py` → `pyvista_viewer.py` with the same `/cfN/pose` subscriber.

## Implementation plan (Option A)

Create `crazyflie_deployment/src/particle_life/particle_life/viewer.py`:

```python
# ~60 lines — conceptual, not complete
import rclpy, numpy as np, pygame, threading
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import sys; sys.path.insert(0, '/home/xxinyue/swarm_life/src')
from particle_life import ParticleLife, Config

ARENA_SIZE = 3.0    # meters; matches arena.yaml
N_DRONES = 10

class CFViewer(Node):
    def __init__(self):
        super().__init__('cf_viewer')
        cfg = Config(n_species=2, n_particles=N_DRONES // 2,
                     sim_width=ARENA_SIZE, sim_height=ARENA_SIZE)
        self.sim = ParticleLife(cfg, headless=False)
        self.lock = threading.Lock()
        for i in range(N_DRONES):
            self.create_subscription(
                PoseStamped, f'/cf{i+1}/pose',
                lambda m, idx=i: self._update(idx, m), 10)

    def _update(self, idx, msg):
        # Vicon frame → arena-centered → shift to sim origin
        with self.lock:
            self.sim.positions[idx, 0] = msg.pose.position.x + ARENA_SIZE / 2
            self.sim.positions[idx, 1] = msg.pose.position.y + ARENA_SIZE / 2

def main():
    rclpy.init()
    viewer = CFViewer()
    thread = threading.Thread(target=rclpy.spin, args=(viewer,), daemon=True)
    thread.start()
    clock = pygame.time.Clock()
    while True:
        # skip sim.step() — positions come from ROS
        with viewer.lock:
            viewer.sim.draw()
        pygame.display.flip()
        clock.tick(60)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: return
```

Register in `setup.py` entry points:
```python
'console_scripts': [
    'particle_life_node = particle_life.particle_life_node:main',
    'viewer = particle_life.viewer:main',   # <-- new
],
```

Launch alongside Crazyswarm2's sim:
```bash
# Terminal 1
ros2 launch crazyflie launch.py backend:=sim

# Terminal 2
ros2 run particle_life viewer
```

### Gotchas to remember

- **Frame mirror**: Vicon y-up; pygame y-down. One sign flip or the arena looks mirrored.
- **Thread safety**: `rclpy.spin` in a background thread → writes to `self.sim.positions`; main thread reads it for `draw()`. Use the lock shown above.
- **Shutdown**: Ctrl+C should land drones first (via `particle_life_node`), not kill the viewer first.
- **Species assignment**: hardcode for now; later read from `config/crazyflies.yaml`.

## Scope for v1

In:
- Subscribe to `/cfN/pose` for N drones
- Render in `ParticleLife`'s existing 2D pygame view
- Species colors driven by a local dict
- Arena bounds drawn as a rectangle

Out (deferred):
- 3D view → Option C if/when needed
- Interactive matrix editing that drives real drones → separate concern
- Trails, force arrows, recording → deferred to Option C or bag recording
- rviz2 path → deferred to hardware phase

---

## Appendix — Full agent arguments

(Raw outputs from the 3 sub-agent design debate — kept for reference.)

### Agent 1 — Reuse pygame

> `ParticleLife` is already a plain Python object whose entire world state is three numpy arrays. The physics (`step()`) and the render (`draw()`) read these arrays and nothing else. Instantiate headless, skip `step()`, overwrite `self.positions` from a ROS subscriber. Under 100 lines. 2D is fine; the arena is a shallow slab (3m × 3m × 1m). Main failure mode: pygame + rclpy in one process means a ROS hiccup freezes UI — mitigate with a background thread.

### Agent 2 — rviz2 / Foxglove

> Publish tf2 + MarkerArray. Check an `.rviz` config into git for reproducible layouts. Both tools read the same topics. Honest verdict: if your goal is exploring particle-life dynamics and making figures, stay in pygame. Switch to rviz2/Foxglove only when flying real drones — then the ROS-native path pays for itself.

### Agent 3 — PyVista

> VTK wrapper with NumPy-first API — matches how researchers already think. `plotter.add_mesh(points)` + `actor.points = new_xyz` speak the same language as the existing code. Built-in video recording, orbital camera, trails via `pv.MultipleLines`, drone IDs via `add_point_labels`. 150 MB install, threading with rclpy needs care on shutdown.
