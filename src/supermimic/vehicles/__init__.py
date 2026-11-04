"""GPU-batched four-wheel vehicle simulator and tracking environment (VEHICLE_PROPOSAL.md, E0).

* ``params``: vehicle and tyre parameters (default: CommonRoad vehicle 2, BMW 320i);
* ``tire``: PAC2002-type magic formula, pure and combined slip (the CommonRoad tyre equations, batched);
* ``dynamics``: double-track model with roll / pitch load transfer, wheel spin, steering and torque
  inputs, optional ABS and ESC logic (deliberately non-differentiable);
* ``maneuvers``: validation manoeuvres shared with the CommonRoad reference generator;
* ``env``: batched tracking environment with actuators, sim context, replay recording and the
  return-based reset sampler.
"""
