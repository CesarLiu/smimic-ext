"""E1 Jacobian tools: the logged-state step equals the environment, both Jacobian estimators recover a
known Jacobian (including the command acting under a delay), binning, and the check script runs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

from supermimic.vehicles import jacobian as J
from supermimic.vehicles.dynamics import AssistCfg, VehicleDynamics
from supermimic.vehicles.env import ReferenceLibrary, VehicleEnvCfg, VehicleTrackingEnv

ROOT = Path(__file__).resolve().parents[1]
D = 29


@pytest.mark.parametrize("assist", [False, True])
def test_logged_step_is_the_environment_transition(assist):
    cfg = VehicleEnvCfg(num_envs=24, replay_capacity=50, assist=AssistCfg(abs=assist, esc=assist))
    lib = ReferenceLibrary.generate(cfg, num=3, steps=120, seed=0)
    env = VehicleTrackingEnv(cfg, lib, seed=0)
    dyn = VehicleDynamics(env.dyn.p, cfg.dt, cfg.substeps, cfg.assist)
    torch.manual_seed(0)
    for _ in range(25):
        s0, xi = env.state().clone(), env.xi.clone()
        a = torch.randn(24, 2)
        *_, info = env.step(a)
        live = ~info["done"]
        assert torch.equal(J.logged_step(dyn, cfg, s0[live], a[live], xi[live]), env.state()[live])


class _Linear:
    """s' = s + M s + B c, with c the acting command (the action, or the pending one under a delay)."""

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.M = 0.1 * torch.randn(D, D, generator=g, dtype=torch.float64)
        self.B = torch.randn(D, 2, generator=g, dtype=torch.float64)

    def step(self, s, a, xi):
        slot = J.acting_command_slot(xi)
        c = a.clone()
        late = slot >= 0
        if late.any():
            idx = slot[late][:, None] + torch.arange(2)
            c = torch.where(late[:, None], torch.zeros_like(a), a)
            c[late] = torch.gather(s[late], 1, idx)
        return s + s @ self.M.T + c @ self.B.T


def _expected(model, err_scale, xi):
    """Exact Jacobian. Under a delay the acting command is a state component, so its column also carries
    that component's own effect through M."""
    cols = J.STATE_COLS
    n = xi.shape[0]
    E = torch.zeros(n, D, len(cols) + 2, dtype=torch.float64)
    for k, c in enumerate(cols):
        E[:, :, k] = (torch.eye(D, dtype=torch.float64)[:, c] + model.M[:, c])[None] * err_scale[c] / err_scale[None, :]
    slot = J.acting_command_slot(xi)
    for i in range(n):
        for j in range(2):
            col = model.B[:, j] + (model.M[:, slot[i] + j] if slot[i] >= 0 else 0.0)
            E[i, :, len(cols) + j] = col / err_scale
    return J._with_delta(E)


def test_both_estimators_recover_a_known_jacobian_with_delays():
    model = _Linear()
    n = 8
    g = torch.Generator().manual_seed(1)
    s = torch.randn(n, D, generator=g, dtype=torch.float64)
    a = torch.randn(n, 2, generator=g, dtype=torch.float64)
    xi = torch.ones(n, 5, dtype=torch.float64)
    xi[:, 4] = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.float64)
    es = 0.05 + torch.rand(D, generator=g, dtype=torch.float64)
    exp = _expected(model, es, xi)
    rows = J.LEARNED
    fd = J.sim_jacobian(model.step, s, a, xi, es, 1e-3)
    torch.testing.assert_close(fd[:, rows], exp[:, rows], rtol=1e-6, atol=1e-8)
    ad = J.wm_jacobian(model, s, a, xi, es)
    torch.testing.assert_close(ad[:, rows], exp[:, rows], rtol=1e-9, atol=1e-10)
    m = J.compare(ad, fd)
    assert float(m["cos"].min()) > 0.999999 and float((m["mag_ratio"] - 1).abs().max()) < 1e-5


def test_kstep_action_gradient_estimators_agree():
    model = _Linear()
    n, k = 6, 5
    g = torch.Generator().manual_seed(2)
    s = torch.randn(n, D, generator=g, dtype=torch.float64)
    acts = torch.randn(k, n, 2, generator=g, dtype=torch.float64)
    xi = torch.ones(n, 5, dtype=torch.float64)
    xi[:, 4] = 0
    es = torch.ones(D, dtype=torch.float64)
    fd = J.sim_kstep_action_grad(model.step, s, acts, xi, es, 1e-3)
    ad = J.wm_kstep_action_grad(model, s, acts, xi, es)
    rows = J.LEARNED
    torch.testing.assert_close(ad[:, rows], fd[:, rows], rtol=1e-6, atol=1e-8)
    # linear system: d s_k / d a_0 = (I + M)^(k-1) B
    exp = torch.linalg.matrix_power(torch.eye(D, dtype=torch.float64) + model.M, k - 1) @ model.B
    torch.testing.assert_close(ad[0][rows], exp[rows], rtol=1e-9, atol=1e-10)


def test_utilisation_bins():
    s = torch.zeros(5, D)
    s[:, 16] = torch.tensor([0.0, 3.0, 7.0, 9.0, 12.0])  # ay
    xi = torch.ones(5, 5)
    u = J.utilisation(s, xi)
    torch.testing.assert_close(u, s[:, 16] / 9.81)
    assert J.bin_index(u).tolist() == [0, 1, 2, 4, 5]  # 0, 0.31, 0.71, 0.92, 1.22
    assert J.bin_label(5) == ">1.0" and J.bin_label(0) == "0.0-0.3"


def test_check_script_runs(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts" / "vehicle"))
    import jacobian_check

    out = tmp_path / "jac.json"
    code = jacobian_check.main(["--num-envs", "32", "--refs", "4", "--ref-steps", "150", "--collect", "80", "--eval-collect", "80",
                                "--updates", "30", "--batch", "64", "--per-bin", "4", "--out", str(out)])
    rep = json.loads(out.read_text())
    assert code in (0, 1) and rep["gate"]["passed"] == (code == 0)  # too small to have to pass; it must run
    assert len(rep["nse_8step_by_bin"]) == len(J.BINS) - 1 and len(rep["one_step_by_bin"]) == len(J.BINS) - 1
    assert all(math.isfinite(r["nse_8step"]) for r in rep["nse_8step_by_bin"] if r["nse_8step"] is not None)
