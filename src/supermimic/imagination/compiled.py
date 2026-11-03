"""One imagined step as a single compiled function (plan 3.6: launch-overhead control).

``ImagStep`` bundles world-model dynamics, the analytic heads (reward, next noise-free observation,
auxiliary goal-side output) and the termination logits into one call so ``torch.compile`` can fuse
the hundreds of small kernels of the per-body FK loop. The reference is passed as the flat tuple of
its dataclass fields (already gathered from the motion library) so the compiled graph has static
shapes and sees only tensors.
"""

from __future__ import annotations

import torch

from supermimic.world_model.heads import REWARD_TERM_NAMES  # noqa: F401  (re-exported, G1 order)
from supermimic.world_model.heads_base import ref_from_tensors, ref_tensors
from supermimic.world_model.world_model import WorldModel


class ImagStep:
    def __init__(self, wm: WorldModel, compile_mode: str | None = None):
        self.wm = wm
        self.term_names = wm.heads.reward_term_names
        self._ref_cls = None
        self._fn = self._step
        if compile_mode:
            mode = None if compile_mode == "default" else compile_mode
            self._fn = torch.compile(self._step, mode=mode, dynamic=False)

    def _step(self, s, a, a_prev, xi, *ref_fields):
        ref = ref_from_tensors(self._ref_cls, ref_fields)
        s_next = self.wm.step(s, a, xi)
        out = self.wm.heads.evaluate(s_next, a, a_prev, xi, ref)
        terms = out.reward_terms
        term_vec = torch.stack([terms[k] for k in self.term_names], dim=0)  # (n_terms, n)
        logits = self.wm.termination_logits(s_next, ref)
        return s_next, terms["total"], term_vec, logits, out.obs, out.aux

    def __call__(self, s, a, a_prev, xi, ref):
        """-> s_next, reward, reward-term vector, termination logits, next clean obs, aux (or None)."""
        self._ref_cls = type(ref)
        return self._fn(s, a, a_prev, xi, *ref_tensors(ref))
