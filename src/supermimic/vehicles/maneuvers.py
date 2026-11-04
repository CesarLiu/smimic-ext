"""Validation manoeuvres, shared by the CommonRoad reference generator and the simulator tests.

Every manoeuvre is driven at the control rate (50 Hz, zero-order hold) through the CommonRoad input
pair (front-wheel steering rate [rad/s], longitudinal acceleration request [m/s^2]). A manoeuvre is a
function of time and of the model's own longitudinal speed and steering angle, so closed-loop speed
holding acts on each model's own state, exactly as a test driver would.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DT = 0.02  # control period [s]
STEER_RATE_MAX = 0.4  # CommonRoad vehicle 2 (BMW 320i) front-wheel steering rate limit [rad/s]


def _hold_speed(v_target: float, vx: float, k: float = 2.0, a_lim: float = 5.0) -> float:
    return max(-a_lim, min(a_lim, k * (v_target - vx)))


def _steer_to(target: float, delta: float) -> float:
    """Steering rate towards ``target`` at the rate limit, without overshoot within one step."""
    return max(-STEER_RATE_MAX, min(STEER_RATE_MAX, (target - delta) / DT))


@dataclass(frozen=True)
class Maneuver:
    name: str
    v0: float  # initial speed [m/s] (straight, steady)
    duration: float  # [s]
    kind: str  # steady | step | sine | accel | brake | brake_in_turn
    delta: float = 0.0  # steady / step / brake_in_turn: steering angle; sine: amplitude [rad]
    accel: float = 0.0  # accel / brake / brake_in_turn: request [m/s^2]
    freq: float = 0.5  # sine [Hz]

    @property
    def steps(self) -> int:
        return round(self.duration / DT)

    def inputs(self, t: float, vx: float, delta: float) -> tuple[float, float]:
        k = self.kind
        if k == "steady":
            return _steer_to(self.delta, delta), _hold_speed(self.v0, vx)
        if k == "step":
            return _steer_to(self.delta if t >= 0.5 else 0.0, delta), _hold_speed(self.v0, vx)
        if k == "sine":
            target = self.delta * math.sin(2 * math.pi * self.freq * (t + DT))
            return _steer_to(target, delta), _hold_speed(self.v0, vx)
        if k in ("accel", "brake"):
            return 0.0, self.accel if t >= 0.5 else 0.0
        if k == "brake_in_turn":
            return _steer_to(self.delta, delta), (self.accel if t >= 4.0 else _hold_speed(self.v0, vx))
        raise ValueError(k)


MANEUVERS = (
    # steady-state cornering: constant steering angle, speed held (linear range up to near the limit)
    *(Maneuver(f"steady_v15_d{d:.3f}", 15.0, 8.0, "steady", delta=d) for d in (0.02, 0.04, 0.06, 0.08)),
    *(Maneuver(f"steady_v25_d{d:.3f}", 25.0, 8.0, "steady", delta=d) for d in (0.01, 0.02, 0.03)),
    # transients
    Maneuver("step_v20_d0.030", 20.0, 4.0, "step", delta=0.03),
    Maneuver("step_v20_d0.060", 20.0, 4.0, "step", delta=0.06),
    Maneuver("sine_v20_a0.040", 20.0, 4.0, "sine", delta=0.04),
    Maneuver("sine_v20_a0.070", 20.0, 4.0, "sine", delta=0.07),
    # longitudinal
    Maneuver("accel_v10_a3", 10.0, 3.5, "accel", accel=3.0),
    Maneuver("brake_v25_a-6", 25.0, 3.0, "brake", accel=-6.0),
    # combined slip: braking in a steady turn
    Maneuver("brake_in_turn_v20_d0.030", 20.0, 6.0, "brake_in_turn", delta=0.03, accel=-4.0),
)
