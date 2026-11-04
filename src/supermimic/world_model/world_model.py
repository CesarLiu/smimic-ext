"""World model: learned dynamics + learned termination head + analytic observation/reward heads.

Robot-specific parts come from the state spec (``spec_base.py``) and the analytic heads (``heads_base.py``)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from supermimic.world_model.dynamics import MLP, DynamicsMLP
from supermimic.world_model.heads_base import AnalyticHeadsBase, ref_from_tensors, ref_tensors
from supermimic.world_model.spec_base import make_state_spec
from supermimic.world_model.state_spec import StateSpecCfg


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    term_hidden: tuple[int, ...] = (64, 64)
    # Any StateSpecCfgBase works here; the annotation names the G1 config for the tyro CLI.
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    term_loss_weight: float = 1.0
    # G1 termination thresholds (scale of the termination features, see heads.termination_feature_scale).
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8


class TerminationHead(nn.Module):
    """P(terminal | state, reference). Input: analytic termination features and their magnitudes."""

    def __init__(self, scale: tuple[float, ...], hidden=(64, 64)):
        super().__init__()
        self.mlp = MLP(2 * len(scale), tuple(hidden), 1)
        self.register_buffer("scale", torch.tensor([float(x) for x in scale]))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (…, F) (G1: [height error, tilt error]) -> logits (…,)"""
        x = feats * self.scale
        return self.mlp(torch.cat([x, x.abs()], dim=-1)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeadsBase | None, device):
        """``heads=None`` gives a dynamics-only model (``step``, ``rollout_error``); its termination
        head then has the G1 layout."""
        super().__init__()
        self.cfg = cfg
        self.spec = make_state_spec(cfg.state_spec, device)
        sp = self.spec
        self.dynamics = DynamicsMLP(feat_dim=sp.feat_dim, act_dim=sp.action_dim, xi_dim=sp.xi_dim, hidden=cfg.hidden, out_dim=sp.delta_dim)
        term_scale = heads.termination_feature_scale(cfg) if heads is not None else (1.0 / cfg.height_threshold, 1.0 / cfg.tilt_threshold)
        self.term_head = TerminationHead(term_scale, cfg.term_hidden)
        self.heads = heads  # analytic observation / reward / FK (no parameters)
        self.to(device)

    # ------------------------------------------------------------------ one step

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = integrate(s_t, f(s_t, a_t, xi)), with the spec's known parts filled in."""
        feat = self.spec.canonical(s)
        delta = self.dynamics(feat, action, self.spec.normalize_xi(xi))
        return self.spec.step_known(s, self.spec.integrate(s, delta), action, xi)

    def termination_logits(self, s_next: torch.Tensor, ref) -> torch.Tensor:
        return self.term_head(self.heads.state_termination_features(s_next, ref))

    # ------------------------------------------------------------------ Eq. 1

    def _loss_step(self, s, a, xi, s_true, done, *ref_fields):
        """One autoregressive step of Eq. 1: returns (s_next, sq_err (n, err_dim), term_bce (n,), logits (n,)).

        ``ref_fields`` is the reference dataclass flattened by ``ref_tensors`` (tensors only, for compile)."""
        s_next = self.step(s, a, xi)
        e = self.spec.error_vector(s_next, s_true)
        logits = self.termination_logits(s_next, ref_from_tensors(self._ref_cls, ref_fields))
        bce = F.binary_cross_entropy_with_logits(logits, done, reduction="none")
        return s_next, e * e, bce, logits

    def enable_compile(self, mode: str | None = "default") -> None:
        self._loss_step_fn = torch.compile(self._loss_step, mode=None if mode == "default" else mode, dynamic=False) if mode else self._loss_step

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, done: torch.Tensor, refs: list):
        """k-step autoregressive loss.

        states: (k+1, n, state_dim) logged states; a_exec: (k, n, action_dim); xi: (n, xi_dim)
        (G1: 71, 29, 33);
        done: (k, n) termination label of states 1..k; refs[j]: reference of transition j (for the
        termination features of state j+1). Fields the heads' termination features do not read may be None.
        """
        k = a_exec.shape[0]
        s = states[0]
        total_state = 0.0
        total_term = 0.0
        metrics = {}
        err_first = None
        err_last = None
        acc_num = 0.0
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        self._ref_cls = type(refs[0])
        for j in range(k):
            s, sq, bce, logits = step_fn(s, a_exec[j], xi, states[j + 1], done[j], *ref_tensors(refs[j]))
            total_state = total_state + sq.sum(-1).mean()
            total_term = total_term + bce.mean()
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                if j == 0:
                    err_first = nse
                err_last = nse
                acc_num += ((logits > 0).float() == done[j]).float().mean()
        loss = total_state / k + self.cfg.term_loss_weight * total_term / k
        # Metrics stay on-device (detached tensors): a float() here would stall the CPU on the
        # forward pass every minibatch, which in data-parallel runs turns into lockstep waiting.
        metrics.update(
            state_loss=(total_state / k).detach(),
            term_bce=(total_term / k).detach(),
            nse_1step=err_first.detach(),
            nse_kstep=err_last.detach(),
            term_acc=(acc_num / k).detach(),
        )
        return loss, metrics

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Open-loop k-step normalised squared error per step (k,), averaged over the batch."""
        k = a_exec.shape[0]
        s = states[0]
        out = []
        for j in range(k):
            s = self.step(s, a_exec[j], xi)
            out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        return torch.stack(out)
