"""PPO baseline (VEHICLE_PROPOSAL.md §6, E6 "PPO-Sim"): model-free, on-policy, on the same environment,
observations, rewards, terminations and networks as SuperMimic.

* Actor: SuperMimic's ``Policy`` network as the Gaussian mean, plus a learned state-independent log std.
* Critic: SuperMimic's ``Critic`` with the same privileged inputs (canonical state, goal, normalised sim
  context xi, tracking-error features); asymmetric, as in SuperMimic.
* Clipped surrogate, GAE(lambda), value regression, optional entropy bonus, KL-adaptive learning rate
  (as in rsl_rl). Time-outs are bootstrapped by adding gamma * V(s_t) to the last reward (rsl_rl's
  approximation; the environment resets before the post-time-out state can be evaluated).

Robot-independent: it talks to the environment only through ``step`` and the learner interfaces (``spec``,
``heads``, ``task``); ``critic_inputs`` is the one place that knows where the environment keeps its state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PPOCfg:
    num_steps: int = 32  # rollout length per environment and update
    epochs: int = 5
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    init_log_std: float = -0.5
    desired_kl: float | None = 0.01  # None: fixed learning rate
    lr_bounds: tuple[float, float] = (1e-5, 1e-3)


class PPO:
    def __init__(self, policy: nn.Module, critic: nn.Module, action_dim: int, cfg: PPOCfg, device):
        self.policy, self.critic, self.cfg, self.device = policy, critic, cfg, device
        self.log_std = nn.Parameter(torch.full((action_dim,), cfg.init_log_std, device=device))
        self.params = [*policy.parameters(), self.log_std, *critic.parameters()]
        self.opt = torch.optim.Adam(self.params, lr=cfg.lr)
        self.lr = cfg.lr

    # ------------------------------------------------------------------ acting

    def dist(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.distributions.Normal:
        mean = self.policy(obs, goal)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    @torch.no_grad()
    def act(self, obs, goal):
        d = self.dist(obs, goal)
        a = d.sample()
        return a, d.log_prob(a).sum(-1)

    # ------------------------------------------------------------------ learning

    def gae(self, rewards, values, dones, last_value):
        """rewards, values, dones: (T, n); dones = episode ended after that step. -> advantages, returns."""
        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        last = torch.zeros_like(last_value)
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_v * nonterminal - values[t]
            last = delta + self.cfg.gamma * self.cfg.lam * nonterminal * last
            adv[t] = last
        return adv, adv + values

    def update(self, batch: dict) -> dict:
        """``batch``: flat (N, ...) tensors obs, goal, critic (tuple of inputs), actions, logp, adv, ret."""
        cfg = self.cfg
        N = batch["actions"].shape[0]
        adv = batch["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        mb = max(1, N // cfg.minibatches)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0}
        count = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(N, device=adv.device)
            for k in range(cfg.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                d = self.dist(batch["obs"][idx], batch["goal"][idx])
                logp = d.log_prob(batch["actions"][idx]).sum(-1)
                ratio = torch.exp(logp - batch["logp"][idx])
                a = adv[idx]
                pl = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                v = self.critic(*(x[idx] for x in batch["critic"]))
                vl = (v - batch["ret"][idx]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pl + cfg.value_coef * vl - cfg.entropy_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = (batch["logp"][idx] - logp).mean()  # approx KL(old || new)
                    if cfg.desired_kl is not None:  # adapt the learning rate to the step size (rsl_rl)
                        if kl > 2.0 * cfg.desired_kl:
                            self.lr = max(cfg.lr_bounds[0], self.lr / 1.5)
                        elif 0.0 < kl < cfg.desired_kl / 2.0:
                            self.lr = min(cfg.lr_bounds[1], self.lr * 1.5)
                        for g in self.opt.param_groups:
                            g["lr"] = self.lr
                    stats["policy_loss"] += float(pl)
                    stats["value_loss"] += float(vl)
                    stats["entropy"] += float(ent)
                    stats["kl"] += float(kl)
                    stats["clip_frac"] += float(((ratio - 1).abs() > cfg.clip).float().mean())
                count += 1
        out = {k: v / count for k, v in stats.items()}
        out["lr"] = self.lr
        out["std"] = float(self.log_std.exp().mean())
        return out


def critic_inputs(env, spec, heads, lib) -> tuple[torch.Tensor, ...]:
    """The SuperMimic critic's inputs for the environment's current state (reference at its target frame)."""
    s = env.state()
    ref = lib.ref(env.frame)
    return spec.canonical(s), env.goal(), spec.normalize_xi(env.xi), heads.critic_features(s, ref)


def train_ppo(env, policy, critic, spec, heads, cfg: PPOCfg, history: int, obs_hist_cls, budget_s: float, eval_fn=None,
              eval_every_s: float = 60.0, log=print, device="cpu", max_updates: int | None = None,
              eval_at_steps: tuple[int, ...] = ()) -> tuple[PPO, list]:
    """Collect ``cfg.num_steps`` per environment, update, repeat until ``budget_s`` seconds of training time
    (evaluation excluded) or ``max_updates``. ``eval_fn(policy) -> dict`` runs every ``eval_every_s`` seconds
    of training time and when the simulator-step count first passes each of ``eval_at_steps``; returns the agent and the evaluation curve [{time_s, env_steps, update, **metrics}]."""
    import time

    agent = PPO(policy, critic, int(env.prev_action.shape[-1]), cfg, device)
    n, T = env.num_envs, cfg.num_steps
    obs = env.reset()
    noise_half = heads.obs_noise_half().to(device)
    hist = obs_hist_cls(heads, n, history, device)
    hist.reset(torch.arange(n, device=device), obs["policy"])
    curve, train_s, env_steps, upd, next_eval, passed = [], 0.0, 0, 0, 0.0, set()
    ep_ret, ep_len, finished = torch.zeros(n, device=device), torch.zeros(n, device=device), []
    while True:
        steps_due = [m for m in eval_at_steps if env_steps >= m and m not in passed]
        stopping = train_s >= budget_s or (max_updates is not None and upd >= max_updates)
        if eval_fn is not None and (train_s >= next_eval or steps_due or stopping):
            passed.update(steps_due)
            row = {"time_s": round(train_s, 1), "env_steps": env_steps, "update": upd, **eval_fn(agent.policy)}
            curve.append(row)
            log(row)
            if train_s >= next_eval:
                next_eval += eval_every_s
        if stopping:
            break
        t0 = time.time()
        buf = {k: [] for k in ("obs", "goal", "critic", "actions", "logp", "values", "rewards", "dones")}
        for _ in range(T):
            o, g = hist.flat(), obs["goal"]
            ci = critic_inputs(env, spec, heads, env.lib)
            a, logp = agent.act(o, g)
            with torch.no_grad():
                v = critic(*ci)
            obs, r, term, trunc, info = env.step(a)
            r = r + cfg.gamma * v * trunc.float()  # bootstrap time-outs
            done = info["done"]
            for k, x in (("obs", o), ("goal", g), ("critic", ci), ("actions", a), ("logp", logp), ("values", v), ("rewards", r), ("dones", done.float())):
                buf[k].append(x)
            ep_ret += info["reward_terms"]["total"]
            ep_len += 1
            if done.any():
                ids = done.nonzero().squeeze(-1)
                finished += (ep_ret[ids] / (heads.max_step_reward * ep_len[ids])).tolist()
                ep_ret[ids], ep_len[ids] = 0.0, 0.0
            noisy = obs["policy"] + (torch.rand_like(obs["policy"]) * 2 - 1) * noise_half
            hist.push(noisy)
            if done.any():
                hist.reset(done.nonzero().squeeze(-1), noisy)
        with torch.no_grad():
            last_v = critic(*critic_inputs(env, spec, heads, env.lib))
        values, rewards, dones = (torch.stack(buf[k]) for k in ("values", "rewards", "dones"))
        adv, ret = agent.gae(rewards, values, dones, last_v)
        flat = lambda xs: torch.stack(xs).reshape(T * n, *xs[0].shape[1:])  # noqa: E731
        batch = {"obs": flat(buf["obs"]), "goal": flat(buf["goal"]), "actions": flat(buf["actions"]), "logp": flat(buf["logp"]),
                 "adv": adv.reshape(-1), "ret": ret.reshape(-1),
                 "critic": tuple(torch.stack([c[j] for c in buf["critic"]]).reshape(T * n, -1) for j in range(len(buf["critic"][0])))}
        stats = agent.update(batch)
        upd += 1
        env_steps += T * n
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        train_s += time.time() - t0
        if upd % 20 == 0:
            recent = finished[-500:]
            tr = sum(recent) / max(len(recent), 1)
            log(f"ppo update {upd}  env steps {env_steps}  train return {tr:.3f}  " + "  ".join(f"{k} {v:.4g}" for k, v in stats.items()))
    return agent, curve

