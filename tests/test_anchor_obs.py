from __future__ import annotations

import pytest

from supermimic.anchor_obs import ANCHOR_OBS_SLICES, anchor_obs_dim, anchor_obs_mode


def test_legacy_bools_map_to_modes():
    assert anchor_obs_mode(True) == "pos_ori"
    assert anchor_obs_mode(False) == "none"
    assert anchor_obs_mode(None) == "none"


@pytest.mark.parametrize("mode,dim", [("none", 0), ("ori", 6), ("pos_ori", 9)])
def test_dims_match_slices(mode, dim):
    assert anchor_obs_dim(mode) == dim
    assert len(range(9)[ANCHOR_OBS_SLICES[mode]]) == dim


def test_ori_keeps_the_rotation_part():
    assert ANCHOR_OBS_SLICES["ori"] == slice(3, 9)  # [pos(3), rot6d(6)] -> rot6d


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        anchor_obs_mode("orientation")
