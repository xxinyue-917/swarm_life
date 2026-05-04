#!/usr/bin/env python3
"""
SetpointAdapter — single-codepath position-target streaming for Crazyswarm2.

Crazyswarm2's `sim` backend implements ONLY `cmd_full_state` (cmd_position and
cmd_velocity_world are commented-out dead code in crazyflie_py). To keep one
controller path that works on both sim and hardware:

    * hardware backends ('cflib', 'cpp') → cf.cmdPosition(pos, yaw)
        clean position target; onboard PID at 100+ Hz tracks it.
    * sim backend → cf.cmdFullState(pos, vel=0, acc=0, yaw, omega=0)
        synthesized full-state with zero feedforward — functionally identical
        to a position-only command.

Decision rationale lives in PLAN.md "Control Mode: cmdPosition (via SetpointAdapter)".
"""

from typing import Iterable


class SetpointAdapter:

    _SIM = 'sim'
    _ZERO3 = (0.0, 0.0, 0.0)

    def __init__(self, backend: str = 'cflib'):
        self.backend = backend

    @property
    def is_sim(self) -> bool:
        return self.backend == self._SIM

    def set_target(self, cf, pos: Iterable[float], yaw: float = 0.0) -> None:
        """Stream a position target to one Crazyflie.

        Args:
            cf: a crazyflie_py Crazyflie object (from swarm.allcfs.crazyflies).
            pos: 3-element world-frame position in meters.
            yaw: yaw angle in radians (default 0).
        """
        pos = (float(pos[0]), float(pos[1]), float(pos[2]))
        if self.is_sim:
            cf.cmdFullState(pos, self._ZERO3, self._ZERO3, float(yaw), self._ZERO3)
        else:
            cf.cmdPosition(pos, float(yaw))
