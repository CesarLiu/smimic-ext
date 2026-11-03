"""Write ``g1_state_spec_golden.pt``: outputs of the G1 state spec, the state perturbation and one
world-model step on fixed inputs, recorded from the implementation before the StateSpec interface
refactor. ``tests/test_state_spec_interface.py`` checks the current code against it, so the refactor
(and any later change) is proven not to alter G1 numbers.

Run from the repository root:  PYTHONPATH=src python tests/data/make_g1_state_spec_golden.py
Only regenerate it on purpose, when a G1 numeric change is intended.
"""

from pathlib import Path

import torch

from supermimic.eval.perturb import apply_state_delta
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

OUT = Path(__file__).with_name("g1_state_spec_golden.pt")


def inputs(dtype):
    g = torch.Generator().manual_seed(1234)
    n = 16
    s = torch.randn(n, 71, generator=g, dtype=dtype)
    s[:, 3:7] = s[:, 3:7] / s[:, 3:7].norm(dim=-1, keepdim=True)
    s_true = s + 0.05 * torch.randn(n, 71, generator=g, dtype=dtype)
    s_true[:, 3:7] = s_true[:, 3:7] / s_true[:, 3:7].norm(dim=-1, keepdim=True)
    delta = torch.randn(n, 70, generator=g, dtype=dtype)
    err = 0.3 * torch.randn(n, 70, generator=g, dtype=dtype)
    xi = torch.rand(n, 33, generator=g, dtype=dtype)
    action = torch.randn(n, 29, generator=g, dtype=dtype)
    limits = torch.stack([-torch.ones(29, dtype=dtype) * 0.5, torch.ones(29, dtype=dtype) * 0.5], dim=-1)
    return {"s": s, "s_true": s_true, "delta": delta, "err": err, "xi": xi, "action": action, "limits": limits}


def spec_outputs(x):
    out = {}
    for explicit in (True, False):
        spec = StateSpec(StateSpecCfg(explicit_integration=explicit), "cpu", x["s"].dtype)
        out[f"integrate_{explicit}"] = spec.integrate(x["s"], x["delta"])
    spec = StateSpec(StateSpecCfg(), "cpu", x["s"].dtype)
    out["canonical"] = spec.canonical(x["s"])
    out["normalize_xi"] = spec.normalize_xi(x["xi"])
    out["error_vector"] = spec.error_vector(x["s"], x["s_true"])
    out["normalized_sq_error"] = spec.normalized_sq_error(x["s"], x["s_true"])
    out["apply_state_delta"] = apply_state_delta(x["s"], x["err"], spec.err_scale)
    out["apply_state_delta_limits"] = apply_state_delta(x["s"], x["err"], spec.err_scale, x["limits"])
    return out


def world_model_outputs(x):
    torch.manual_seed(0)
    wm = WorldModel(WorldModelCfg(hidden=(64, 64)), heads=None, device="cpu")
    last = wm.dynamics.mlp.net[-1]
    with torch.no_grad():  # the last layer is zero-initialised; make the step non-trivial
        last.weight.normal_(0.0, 0.05)
        last.bias.normal_(0.0, 0.05)
    xf = {k: v.float() for k, v in x.items()}
    with torch.no_grad():
        nxt = wm.step(xf["s"], xf["action"], xf["xi"])
    return {
        "state_dict": {k: v.clone() for k, v in wm.state_dict().items()},
        "step": nxt,
    }


def main():
    x64 = inputs(torch.float64)
    golden = {"inputs": x64, "spec64": spec_outputs(x64), "wm": world_model_outputs(x64)}
    torch.save(golden, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
