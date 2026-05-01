# Crazyflie Fleet Inventory

Physical drone fleet inherited from the previous Crazyswarm1 project. Each drone has unique radio URI; channel assignments split across 3 channels (likely a bandwidth-management strategy from the old setup).

## Fleet roster

| Drone | URI | Channel | Address | Initial position (m) | Last known state |
|-------|-----|---------|---------|---------------------|------------------|
| cf1 | `radio://0/1/2M/E7E7E7E701` | 1 | `0xE7E7E7E701` | (1.0, 1.0, 0.0) | ✅ alive (2026-05-01) |
| cf2 | `radio://0/1/2M/E7E7E7E702` | 1 | `0xE7E7E7E702` | (1.0, 0.0, 0.0) | ❌ off / dead battery |
| cf3 | `radio://0/1/2M/E7E7E7E703` | 1 | `0xE7E7E7E703` | (1.0, -1.0, 0.0) | ❌ off / dead battery |
| cf4 | `radio://0/1/2M/E7E7E7E704` | 1 | `0xE7E7E7E704` | (0.0, 1.0, 0.0) | ❌ off / dead battery |
| cf5 | `radio://0/5/2M/E7E7E7E705` | 5 | `0xE7E7E7E705` | (0.0, 0.0, 0.0) | ✅ alive (2026-05-01) |
| cf7 | `radio://0/5/2M/E7E7E7E707` | 5 | `0xE7E7E7E707` | (-1.0, 1.0, 0.0) | ✅ alive (2026-05-01) |
| cf9 | `radio://0/6/2M/E7E7E7E709` | 6 | `0xE7E7E7E709` | (0.0, -1.0, 0.0) | ❌ off / dead battery |

**Total: 7 drones** (gaps at id 6, 8, 10 — never existed in the original fleet).

## Quick scan

To see which drones are currently alive on the air:

```bash
source ~/Crazyflie/crazyflie_venv/bin/activate
python3 -c "
import cflib.crtp
cflib.crtp.init_drivers()
fleet = [(1,1),(1,2),(1,3),(1,4),(5,5),(5,7),(6,9)]
for ch, i in fleet:
    uri = f'radio://0/{ch}/2M/E7E7E7E70{i}'
    res = cflib.crtp.scan_interfaces(0xE7E7E7E700+i)
    hit = [u for u,_ in res if f'/{ch}/' in u and f'70{i}' in u]
    print(uri, '<-found' if hit else 'not found')
"
```

## Connecting a single drone in cfclient

In the cfclient address bar, paste the full URI from the table above, e.g.:
```
radio://0/1/2M/E7E7E7E701
```

The `0` at the beginning is the radio dongle index — keep it as `0` even if you have multiple dongles plugged in (cfclient connects to the first dongle).

## Channel split

| Channel | Drones | Notes |
|---------|--------|-------|
| 1 | cf1, cf2, cf3, cf4 | Original sub-swarm A |
| 5 | cf5, cf7 | Original sub-swarm B |
| 6 | cf9 | Original solo / spare |

The Crazyswarm1 setup used 2 Crazyradio PAs to handle 3 channels (one radio can switch between channels, but TDM bandwidth is shared). For Crazyswarm2 it's wise to assign one radio per channel to keep update rates high. With 3 radios you can run all 7 drones simultaneously without bandwidth contention.

## Marker setup (legacy)

Old fleet used `defaultSingleMarker` — one reflective marker per drone, identity by proximity (libobjecttracker). For particle-life behaviors with close interactions, **upgrade to 4-marker rigid bodies** before flying — see `references/dasc_flylab_setup.md` for the failure mode.
