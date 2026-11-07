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
    # ensemble of dynamics networks (1 = the paper's single model). Members are initialised independently and each
    # is fitted on its own autoregressive rollout of the same windows; imagination uses their mean and their
    # disagreement as an uncertainty estimate (``step_with_uncertainty``).
    ensemble: int = 1
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
        # further ensemble members (none for ensemble = 1, so the single model's parameters keep their names)
        self.ensemble_extra = nn.ModuleList(
            DynamicsMLP(feat_dim=sp.feat_dim, act_dim=sp.action_dim, xi_dim=sp.xi_dim, hidden=cfg.hidden, out_dim=sp.delta_dim) for _ in range(cfg.ensemble - 1))
        term_scale = heads.termination_feature_scale(cfg) if heads is not None else (1.0 / cfg.height_threshold, 1.0 / cfg.tilt_threshold)
        self.term_head = TerminationHead(term_scale, cfg.term_hidden)
        self.heads = heads  # analytic observation / reward / FK (no parameters)
        self.to(device)

    # ------------------------------------------------------------------ one step

    @property
    def ensemble(self) -> int:
        return 1 + len(self.ensemble_extra)

    def member(self, k: int) -> DynamicsMLP:
        return self.dynamics if k == 0 else self.ensemble_extra[k - 1]

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor, member: int | None = None) -> torch.Tensor:
        """s_{t+1} = integrate(s_t, f(s_t, a_t, xi)), with the spec's known parts filled in. With an ensemble, ``f`` is
        the members' mean, or member ``member`` alone."""
        if self.ensemble > 1 and member is None:
            return self.step_with_uncertainty(s, action, xi)[0]
        feat = self.spec.canonical(s)
        net = self.dynamics if member is None else self.member(member)
        delta = net(feat, action, self.spec.normalize_xi(xi))
        return self.spec.step_known(s, self.spec.integrate(s, delta), action, xi)

    def step_with_uncertainty(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor):
        """-> (s_{t+1} from the members' mean increment, disagreement (…,)): the variance across members of the
        (normalised) predicted increments, averaged over its components. Zero for a single model."""
        feat, xin = self.spec.canonical(s), self.spec.normalize_xi(xi)
        deltas = torch.stack([self.member(k)(feat, action, xin) for k in range(self.ensemble)], 0)
        delta = deltas.mean(0)
        u = deltas.var(0, unbiased=False).mean(-1) if self.ensemble > 1 else torch.zeros(delta.shape[:-1], device=delta.device, dtype=delta.dtype)
        return self.spec.step_known(s, self.spec.integrate(s, delta), action, xi), u

    def termination_logits(self, s_next: torch.Tensor, ref) -> torch.Tensor:
        return self.term_head(self.heads.state_termination_features(s_next, ref))

    # ------------------------------------------------------------------ Eq. 1

    def _loss_step(self, s, a, xi, s_true, done, *ref_fields, member=None):
        """One autoregressive step of Eq. 1: returns (s_next, sq_err (n, err_dim), term_bce (n,), logits (n,)).

        ``ref_fields`` is the reference dataclass flattened by ``ref_tensors`` (tensors only, for compile); ``member``:
        the ensemble member to step (None: the single model)."""
        s_next = self.step(s, a, xi, member)
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
        total_state = 0.0
        total_term = 0.0
        metrics = {}
        err_first = None
        err_last = None
        acc_num = 0.0
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        self._ref_cls = type(refs[0])
        members = [None] if self.ensemble == 1 else list(range(self.ensemble))
        for m in members:  # each ensemble member on its own autoregressive rollout (averaged below)
            s = states[0]
            for j in range(k):
                if m is None:
                    s, sq, bce, logits = step_fn(s, a_exec[j], xi, states[j + 1], done[j], *ref_tensors(refs[j]))
                else:
                    s, sq, bce, logits = step_fn(s, a_exec[j], xi, states[j + 1], done[j], *ref_tensors(refs[j]), member=m)
                total_state = total_state + sq.sum(-1).mean()
                total_term = total_term + bce.mean()
                with torch.no_grad():
                    nse = sq.mean(-1).mean()
                    if j == 0:
                        err_first = nse if err_first is None else err_first + nse
                    if j == k - 1:
                        err_last = nse if err_last is None else err_last + nse
                    acc_num += ((logits > 0).float() == done[j]).float().mean()
        nm = len(members)
        total_state, total_term, acc_num = total_state / nm, total_term / nm, acc_num / nm
        err_first, err_last = err_first / nm, err_last / nm
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
