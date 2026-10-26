"""Anchor-relative goal observation modes (plan D5 alternative).

The paper's Table A.1 lists proprioception only and leaves the per-frame goal content open. Without any
root-relative feedback the policy drifts in yaw over long clips (EXPERIMENTS, drift diagnostic), so the
goal group can carry the reference anchor pose in the robot anchor frame:

  * ``none``    : no anchor term (paper Table A.1 reading, e2/e6/e7/e8);
  * ``ori``     : rotation only, rot6d (6) = mjlab's ``motion_anchor_ori_b``. Needs the robot's yaw,
                  which an IMU provides, but no global position (supermimic_vibed's choice);
  * ``pos_ori`` : position (3) + rot6d (6) = BeyondMimic's default; the position needs state estimation
                  on hardware (e3/e5).

Configs saved before the ``ori`` mode store a bool: True means ``pos_ori``, False means ``none``.
Kept free of mjlab imports so it can be used from configs and scripts alike.
"""

from __future__ import annotations

ANCHOR_OBS_DIMS = {"none": 0, "ori": 6, "pos_ori": 9}
# Slice of the 9-d [pos(3), rot6d(6)] anchor-relative vector that each mode keeps.
ANCHOR_OBS_SLICES = {"none": slice(0, 0), "ori": slice(3, 9), "pos_ori": slice(0, 9)}


def anchor_obs_mode(value: bool | str | None) -> str:
    """Normalise an ``anchor_obs`` setting (legacy bool or mode name) to a mode name."""
    if value is None or value is False:
        return "none"
    if value is True:
        return "pos_ori"
    if value not in ANCHOR_OBS_DIMS:
        raise ValueError(f"anchor_obs must be one of {tuple(ANCHOR_OBS_DIMS)} (or a legacy bool), got {value!r}")
    return value


def anchor_obs_dim(value: bool | str | None) -> int:
    return ANCHOR_OBS_DIMS[anchor_obs_mode(value)]
