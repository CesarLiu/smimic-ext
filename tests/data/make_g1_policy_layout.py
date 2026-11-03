"""Write ``g1_policy_layout.pt``: parameter names and shapes, and the input-scale buffers, of the
full-size G1 policy and critic (goal without / with the 6-d anchor term), recorded before the policy
I/O was parameterised. ``tests/test_policy_io.py`` checks that default construction still gives
exactly this, so existing checkpoints keep loading.

Run from the repository root:  PYTHONPATH=src python tests/data/make_g1_policy_layout.py
"""

from pathlib import Path

import torch

from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg

OUT = Path(__file__).with_name("g1_policy_layout.pt")


def layout(module):
    sd = module.state_dict()
    return {
        "shapes": {k: torch.tensor(v.shape) for k, v in sd.items()},
        "buffers": {k: v.clone() for k, v in module.named_buffers()},
    }


if __name__ == "__main__":
    out = {}
    for extra in (0, 6):
        pc = PolicyCfg(goal_dim=680 + extra)
        out[f"goal{680 + extra}"] = {"policy": layout(Policy(pc)), "critic": layout(Critic(CriticCfg(), pcfg=pc))}
    torch.save(out, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
