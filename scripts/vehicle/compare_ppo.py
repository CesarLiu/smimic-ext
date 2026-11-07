"""SuperMimic vs PPO vs NMPC on the vehicle: train both on the same library of random curves (friction-matched,
up to the grip limit, as in ``limit_study.py``), then test on held-out curves and on an S-bend near the grip
limit that neither saw in training.

    PYTHONPATH=src python scripts/vehicle/compare_ppo.py --device cuda --budget-min 30 --out-dir logs/compare

* Same environment, observations, rewards, terminations, sim-context randomisation, reset sampler and
  networks (SuperMimic's ``Policy`` / ``Critic``; PPO adds a learned action std). ABS / ESC off.
* Each method gets the same training wall-clock budget (``--budget-min``; evaluation excluded); PPO runs
  many more environments (``--ppo-envs``, default 4096 vs 256), as model-free methods need the samples.
  Both curves are recorded: return against training time and against simulator steps.
* NMPC (``supermimic.baselines.nmpc``, needs ``casadi``): no training; single-track model with tyre curves
  identified from the simulator, IPOPT, 1 s horizon. Two variants: ``nmpc`` reads the true state and
  (``--nmpc-friction known``) the road friction, which the policies do not get (an upper bound);
  ``nmpc_onboard`` gets what the policies get (the 12 on-board measurements, an estimator for sideslip and
  friction, delay compensation) and is the fair opponent. Its solve time per control step is recorded next to
  the policies' inference time.
* ``--eval-mismatch`` puts the plant at the edge of the training ranges for stiffness, mass and CoG (NMPC keeps
  its nominal model).
* ``--eval-noise`` / ``--eval-delay N`` make every evaluation (policies and NMPC alike) use measurement noise as in
  training and a fixed command delay of N control steps; the default is the noise-free, delay-free evaluation
  of earlier runs.
* S-bend: straight, left arc, right arc, straight at ``--scurve-speed`` with the arcs at ``--scurve-ay`` x
  mu g, one per friction level (``ReferenceLibrary.s_curve``).

Writes ``report.json``, ``curves.png`` (learning curves), ``s_curve.png`` (paths and lateral error) and
``policies.pt`` (the trained policies and the NMPC settings, for ``render_scurve.py``) to ``--out-dir``. One seed;
no pass / fail gate.
"""

from __future__ import annotations

import argparse
import copy
import math
import json
from dataclasses import asdict
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_e2e import ObsHistory
from limit_study import MODELS, NEAR_LIMIT, env_cfg, eval_xi, evaluate, libraries, measurement

from supermimic.baselines.nmpc import NMPC, NMPCCfg
from supermimic.baselines.ppo import PPOCfg, train_ppo
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.policy.nets import Critic, CriticCfg, NetIO, Policy, PolicyCfg
from supermimic.vehicles import tracking
from supermimic.vehicles.env import ReferenceLibrary, VehicleTrackingEnv
from supermimic.vehicles.learner import VehicleHeads, VehicleStateSpecCfg, VehicleTask
from supermimic.vehicles.threads import add_threads_arg, limit_threads, usable_cores
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

METHODS = ("supermimic", "ppo", "nmpc", "nmpc_onboard")
COLORS = {"supermimic": "#1f77b4", "ppo": "#d62728", "nmpc": "#2ca02c", "nmpc_onboard": "#9467bd",
          "supermimic_chassis": "#17becf", "ppo_chassis": "#ff7f0e"}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--budget-min", type=float, default=30.0, help="training wall-clock per method [min]")
    ap.add_argument("--eval-every-s", type=float, default=120.0)
    ap.add_argument("--sm-max-iters", type=int, default=None, help="also stop SuperMimic after this many iterations (machine-independent budget)")
    ap.add_argument("--ppo-max-updates", type=int, default=None, help="also stop PPO after this many updates")
    ap.add_argument("--eval-at-steps", type=float, nargs="*", default=[1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8],
                    help="also evaluate when the simulator-step count first passes these (comparison at equal data)")
    # references (as limit_study.py)
    ap.add_argument("--mu-levels", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    ap.add_argument("--ay-frac", type=float, nargs=2, default=[0.4, 0.95])
    ap.add_argument("--refs-per-level", type=int, default=12)
    ap.add_argument("--eval-refs-per-level", type=int, default=6)
    ap.add_argument("--ref-steps", type=int, default=500)
    ap.add_argument("--episode-steps", type=int, default=250)
    ap.add_argument("--history", type=int, default=5)
    ap.add_argument("--scurve-speed", type=float, default=15.0)
    ap.add_argument("--scurve-ay", type=float, default=0.7)
    # SuperMimic (as limit_study.py, large world model)
    ap.add_argument("--wm", default="large", choices=list(MODELS))
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--policy-warmup", type=int, default=300)
    ap.add_argument("--action-noise", type=float, default=0.1)
    ap.add_argument("--wm-ensemble", type=int, default=1, help="dynamics networks in the world model (1 = single model)")
    ap.add_argument("--unc-penalty", type=float, default=0.0, help="imagined reward penalty per unit of ensemble disagreement, "
                    "in units of the largest step reward (needs --wm-ensemble > 1)")
    ap.add_argument("--chassis-explore", type=float, default=0.0, help="SuperMimic data collection: extra slowly varying (Ornstein-"
                    "Uhlenbeck) noise of this std on the chassis actions (2..4), so the world model sees their effects early")
    ap.add_argument("--chassis-explore-tau", type=float, default=0.3, help="correlation time of that noise [s]")
    # PPO
    ap.add_argument("--ppo-envs", type=int, default=4096)
    ap.add_argument("--ppo-steps", type=int, default=32)
    ap.add_argument("--ppo-epochs", type=int, default=5)
    ap.add_argument("--ppo-minibatches", type=int, default=4)
    ap.add_argument("--ppo-lr", type=float, default=3e-4)
    ap.add_argument("--ppo-entropy", type=float, default=0.0)
    # NMPC
    ap.add_argument("--nmpc-friction", default="known", choices=["known", "nominal"], help="known: the true road friction; nominal: assume 1.0")
    ap.add_argument("--nmpc-horizon", type=int, default=25, help="shooting intervals")
    ap.add_argument("--nmpc-interval", type=int, default=2, help="control steps per shooting interval")
    ap.add_argument("--nmpc-assumed-delay", type=int, default=1, help="nmpc_onboard: command delay [control steps] it compensates (not told the true one)")
    # evaluation conditions, for every method
    ap.add_argument("--eval-noise", action="store_true", help="evaluate with the measurement noise of training")
    ap.add_argument("--eval-delay", type=int, default=0, help="evaluate with this fixed command delay [control steps]")
    ap.add_argument("--eval-mismatch", action="store_true", help="evaluate with tyre stiffness x0.85, mass x1.15, CoG 0.1 m forward "
                    "(edge of the training ranges; NMPC's model keeps the nominal values)")
    ap.add_argument("--policies-from", default=None, help="policies.pt of an earlier run: evaluate its supermimic / ppo policies "
                    "(under the evaluation conditions above) instead of training them")
    ap.add_argument("--chassis", action="store_true", help="5-D action: add rear-wheel steering, torque-vectoring yaw moment and the "
                    "roll-stiffness split (VEHICLE_PROPOSAL.md §10); NMPC keeps them neutral")
    ap.add_argument("--chassis-usage-weight", type=float, default=-0.2, help="reward weight of the chassis actuators' usage cost "
                    "(squared squashed commands); 0 switches it off")
    ap.add_argument("--out-dir", default="logs/compare")
    ap.add_argument("--no-plots", action="store_true")
    add_threads_arg(ap)
    return ap.parse_args(argv)


def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def networks(args, env, lib):
    heads = VehicleHeads(env.cfg, env.dyn.p.R_w)
    task = VehicleTask(lib, env.cfg.preview)
    spec = VehicleStateSpecCfg.from_env(env.cfg, env.dyn.p).build(args.device)
    io = NetIO.from_interfaces(spec, heads, task, args.history)
    pcfg = PolicyCfg(encoder_hidden=(128,), decoder_hidden=(256, 128), latent_dim=32, history=args.history, decoder_last_gain=0.1)
    return heads, task, spec, io, pcfg, Policy(pcfg, io).to(args.device), Critic(CriticCfg(hidden=(256, 128)), io=io).to(args.device)


def make_eval(heads, eval_libs, scurves, args):
    """Mean over friction levels: held-out curves (return, near-limit share still on the reference) and S-bends."""

    def run(policy):
        policy.eval()
        h = [evaluate(policy, heads, lib, args) for lib in eval_libs]
        s = [evaluate(policy, heads, lib, args) for lib in scurves]
        policy.train()
        mean = lambda xs, k: sum(x[k] for x in xs) / len(xs)  # noqa: E731
        return {"heldout_return": round(mean(h, "normalized_return"), 4), "heldout_near_limit": round(mean(h, "near_limit_survived_share"), 4),
                "scurve_return": round(mean(s, "normalized_return"), 4), "scurve_completed": round(1 - mean(s, "termination_rate"), 4)}

    return run


def train_supermimic(args, lib, run_eval, log) -> tuple[torch.nn.Module, list, dict]:
    torch.manual_seed(args.seed)
    dev = args.device
    cfg = env_cfg(args, args.num_envs, args.episode_steps + 50)
    env = VehicleTrackingEnv(cfg, lib, device=dev, seed=args.seed)
    heads, task, _spec, io, pcfg, policy, critic = networks(args, env, lib)
    wm = WorldModel(WorldModelCfg(hidden=MODELS[args.wm], state_spec=VehicleStateSpecCfg.from_env(cfg, env.dyn.p), ensemble=args.wm_ensemble), heads, dev)
    wm_tr = WorldModelTrainer(wm, task, WMTrainerCfg(window=8, batches_per_update=2, batch_size=args.batch, lr=args.lr, compile=None))
    pol_tr = PolicyTrainer(wm, policy, critic, task, PolicyTrainerCfg(horizon=args.horizon, batch_size=args.batch, history=args.history, lr=args.lr,
                                                                       critic_lr=args.lr, critic_epochs=2, critic_minibatches=2, compile=None,
                                                                       value_clip=cfg.tracking.max_step_reward / (1 - 0.99), uncertainty_penalty=args.unc_penalty))
    n = args.num_envs
    obs = env.reset()
    hist = ObsHistory(heads, n, args.history, dev)
    hist.reset(torch.arange(n, device=dev), obs["policy"])
    noise_half = heads.obs_noise_half().to(dev)
    curve, train_s, it, next_eval, budget, passed = [], 0.0, 0, 0.0, 60 * args.budget_min, set()
    wm_m, pol_m = {}, {}
    explore = None
    if args.chassis_explore > 0 and cfg.action_dim > 2:  # OU noise on the chassis actions, per environment
        rho = math.exp(-cfg.dt / args.chassis_explore_tau)
        explore = torch.zeros(n, cfg.action_dim - 2, device=dev)
    while True:
        steps_due = [m for m in args.eval_at_steps if it * n >= m and m not in passed]
        done = train_s >= budget or (args.sm_max_iters is not None and it >= args.sm_max_iters)
        if train_s >= next_eval or done or steps_due:
            passed.update(steps_due)
            row = {"time_s": round(train_s, 1), "env_steps": it * n, "iter": it, **run_eval(policy)}
            curve.append(row)
            log(f"supermimic {row}  wm 8-step {float(wm_m.get('nse_kstep', float('nan'))):.3f}  imag return {float(pol_m.get('imag_return_undisc', float('nan'))):.3f}"
                + (f"  disagreement {float(pol_m.get('imag_r/ensemble_disagreement', float('nan'))):.4f}" if args.wm_ensemble > 1 else ""))
            if train_s >= next_eval:
                next_eval += args.eval_every_s
        if done:
            break
        t0 = time.time()
        for _ in range(20 if args.sm_max_iters is None else min(20, args.sm_max_iters - it)):  # timing / evaluation granularity
            with torch.no_grad():
                a = policy(hist.flat(), obs["goal"])
                noise = args.action_noise * torch.randn_like(a)
                if explore is not None:
                    explore = rho * explore + math.sqrt(1 - rho * rho) * args.chassis_explore * torch.randn_like(explore)
                    noise[:, 2:] += explore
                obs, _r, _t, _tr, info = env.step(a, noise)
            noisy = obs["policy"] + (torch.rand_like(obs["policy"]) * 2 - 1) * noise_half
            hist.push(noisy)
            if info["done"].any():
                hist.reset(info["done"].nonzero().squeeze(-1), noisy)
            if it >= args.warmup:
                wm_m = wm_tr.update(env.buffer)
            if it >= args.policy_warmup:
                pol_m = pol_tr.update(env.buffer)
            it += 1
        _sync(dev)
        train_s += time.time() - t0
    return policy, curve, {"iterations": it, "env_steps": it * n, "train_s": round(train_s, 1)}


def train_ppo_method(args, lib, run_eval, log):
    torch.manual_seed(args.seed)
    cfg = env_cfg(args, args.ppo_envs, 2)
    env = VehicleTrackingEnv(cfg, lib, device=args.device, seed=args.seed, record=False)
    heads, _task, spec, _io, _pcfg, policy, critic = networks(args, env, lib)
    pcfg = PPOCfg(num_steps=args.ppo_steps, epochs=args.ppo_epochs, minibatches=args.ppo_minibatches, lr=args.ppo_lr, entropy_coef=args.ppo_entropy)
    agent, curve = train_ppo(env, policy, critic, spec, heads, pcfg, args.history, ObsHistory, 60 * args.budget_min, run_eval,
                             args.eval_every_s, log=lambda r: log(f"ppo {r}" if isinstance(r, dict) else r), device=args.device,
                             eval_at_steps=tuple(int(m) for m in args.eval_at_steps), max_updates=args.ppo_max_updates)
    last = curve[-1]
    return agent.policy, curve, {"updates": last["update"], "env_steps": last["env_steps"], "train_s": last["time_s"]}


@torch.no_grad()
def trajectory(policy, heads, lib: ReferenceLibrary, args, controller=None) -> dict:
    """Deterministic play of a one-reference library from its start; per-step path and errors until it ends.
    ``controller(env, measurements) -> actions`` replaces the policy when given."""
    length = int(lib.lengths[0])
    env = VehicleTrackingEnv(env_cfg(args, 1, 2, train=False), lib, device=args.device, record=False)
    env.forced_xi = eval_xi(args)
    env.forced_frame[:] = lib.layout.seq_start[:1]
    obs = env.reset()
    meas = measurement(obs["policy"], heads, args)
    hist = ObsHistory(heads, 1, args.history, args.device)
    hist.reset(torch.arange(1, device=args.device), meas)
    out = {k: [] for k in ("x", "y", "lat", "beta", "util", "speed")}
    terminated = False
    for _ in range(length - 1):
        ref = lib.ref(env.frame.clone())
        obs, _r, term, _trunc, info = env.step(policy(hist.flat(), obs["goal"]) if controller is None else controller(env, meas))
        if info["done"][0]:
            terminated = bool(term[0])
            break
        s = env.s[0]
        e = tracking.errors(env.s, ref)
        out["x"].append(float(s[0]))
        out["y"].append(float(s[1]))
        out["lat"].append(float(e["lat"][0]))
        out["beta"].append(float(e["beta"][0] + ref.beta[0]))
        out["util"].append(float(torch.sqrt(s[15] ** 2 + s[16] ** 2) / (env.ctx.mu[0] * 9.81)))
        out["speed"].append(float(e["speed"][0]))
        meas = measurement(obs["policy"], heads, args)
        hist.push(meas)
    out["terminated"], out["steps"] = terminated, len(out["x"])
    return out


def load_policy(args, cfg, lib, method: str, blob: dict):
    """The saved policy of ``method`` (architecture from this run's arguments, ``--history`` from the saved run)."""
    run = argparse.Namespace(**{**vars(args), "history": blob["args"]["history"]})
    env = VehicleTrackingEnv(cfg, lib, device=args.device, record=False)
    _h, _t, _s, _io, _pc, policy, _c = networks(run, env, lib)
    policy.load_state_dict(blob["policies"][method])
    return policy


def nmpc_cfg(args, method: str = "nmpc") -> NMPCCfg:
    onboard = method == "nmpc_onboard"
    return NMPCCfg(horizon=args.nmpc_horizon, interval=args.nmpc_interval, friction=args.nmpc_friction,
                   information="onboard" if onboard else "privileged", assumed_delay=args.nmpc_assumed_delay if onboard else 0)


@torch.no_grad()
def policy_latency_ms(policy, heads, args, reps: int = 300) -> dict:
    """One policy evaluation for a single vehicle on the CPU (the deployment case), in milliseconds."""
    import numpy as np

    p = copy.deepcopy(policy).cpu().eval()
    hist = ObsHistory(heads, 1, args.history, "cpu")
    obs, goal = hist.flat(), torch.zeros(1, 5 * len(env_cfg(args, 1, 2).preview))
    ts = []
    for i in range(reps + 20):
        t0 = time.perf_counter()
        p(obs, goal)
        if i >= 20:
            ts.append(1e3 * (time.perf_counter() - t0))
    return {"mean_ms": float(np.mean(ts)), "p95_ms": float(np.percentile(ts, 95))}


def run_nmpc(args, cfg, heads, eval_libs, scurves, log, method: str = "nmpc"):
    """Evaluate an NMPC baseline (no training) on the held-out curves and the S-bends."""
    ctl = NMPC(cfg, cfg=nmpc_cfg(args, method))
    log(f"nmpc: tyre fit {ctl.fit_info}; front {ctl.front}, rear {ctl.rear}")
    final = {f"mu_{mu}": evaluate(None, heads, el, args, controller=ctl) for mu, el in zip(args.mu_levels, eval_libs)}
    sc = {f"mu_{mu}": evaluate(None, heads, s, args, controller=ctl) for mu, s in zip(args.mu_levels, scurves)}
    traj = {f"mu_{mu}": trajectory(None, heads, s, args, controller=ctl) for mu, s in zip(args.mu_levels, scurves)}
    info = {"nmpc": asdict(ctl.cfg), "tyre_fit": ctl.fit_info, "latency": ctl.timing()}
    log(f"nmpc solve time per control step: {info['latency']}")
    return final, sc, traj, info


def plots(report, scurves, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = COLORS
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for m, res in report["methods"].items():
        c = res["curve"]
        if not c:  # no training (NMPC): its final score as a horizontal line
            mean = lambda d: sum(v["normalized_return"] for v in d.values()) / len(d)  # noqa: E731
            for j, key in enumerate(("heldout_final", "scurve_final")):
                for a in ax[j]:
                    a.axhline(mean(res[key]), color=colors[m], ls="--", label=f"{m} (final only)")
            continue
        for j, (key, title) in enumerate((("heldout_return", "held-out curves"), ("scurve_return", "S-bend"))):
            ax[j, 0].plot([r["time_s"] / 60 for r in c], [r[key] for r in c], "-o", ms=3, color=colors[m], label=m)
            ax[j, 1].plot([r["env_steps"] for r in c], [r[key] for r in c], "-o", ms=3, color=colors[m], label=m)
            ax[j, 0].set_ylabel(f"normalised return\n({title})")
    ax[1, 0].set_xlabel("training time [min]")
    ax[1, 1].set_xlabel("simulator steps")
    for a in ax[:, 1]:
        a.set_xscale("symlog", linthresh=1e4)  # keeps the untrained point at 0 steps
    for a in ax.flat:
        a.grid(alpha=0.3)
        a.set_ylim(0, 1)
    ax[0, 0].legend()
    fig.suptitle(" vs ".join(report["methods"]) + " (vehicle, grip limit, ABS/ESC off)")
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=130)
    plt.close(fig)

    levels = list(report["scurve"].keys())
    fig, ax = plt.subplots(2, len(levels), figsize=(5 * len(levels), 7), squeeze=False)
    for j, (lv, lib) in enumerate(zip(levels, scurves)):
        ax[0, j].plot(lib.x.cpu(), lib.y.cpu(), "--", color="0.5", lw=2, label="reference")
        for m, res in report["methods"].items():
            tr = res["scurve_traj"][lv]
            lbl = f"{m} ({'terminated' if tr['terminated'] else 'completed'})"
            ax[0, j].plot(tr["x"], tr["y"], color=colors[m], label=lbl)
            if tr["terminated"] and tr["x"]:
                ax[0, j].plot(tr["x"][-1], tr["y"][-1], "x", color=colors[m], ms=10, mew=2)
            ax[1, j].plot([k * 0.02 for k in range(tr["steps"])], tr["lat"], color=colors[m], label=m)
        ax[0, j].set_title(f"S-bend, {lv.replace('_', ' = ')}")
        ax[0, j].set_xlabel("x [m]")
        ax[0, j].set_ylabel("y [m]")
        ax[0, j].axis("equal")
        ax[0, j].legend(fontsize=8)
        ax[1, j].axhline(0, color="0.5", lw=1)
        ax[1, j].set_xlabel("t [s]")
        ax[1, j].set_ylabel("lateral error [m]")
        ax[1, j].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "s_curve.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse(argv)
    if any(m.startswith("nmpc") for m in args.methods):  # fail before any training, not after it
        try:
            import casadi  # noqa: F401
        except ImportError:
            sys.exit("the nmpc method needs casadi (pip install casadi), or leave it out: --methods supermimic ppo")
    threads = limit_threads(args.threads)
    print(f"cpu threads: {threads} (of {usable_cores()} usable cores), device {args.device}", flush=True)
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)  # noqa: E731
    cfg = env_cfg(args, 2, 2)
    lib = ReferenceLibrary.concat(libraries(args, cfg, args.refs_per_level, args.seed)).to(args.device)
    eval_libs = [x.to(args.device) for x in libraries(args, cfg, args.eval_refs_per_level, args.seed + 1000)]
    scurves = [ReferenceLibrary.s_curve(cfg, speed=args.scurve_speed, ay_frac=args.scurve_ay, mu=mu).to(args.device) for mu in args.mu_levels]
    log(f"references: {lib.layout.num_sequences} train, {sum(x.layout.num_sequences for x in eval_libs)} held-out, S-bends at mu {args.mu_levels} "
        f"({args.scurve_speed} m/s, {args.scurve_ay} mu g, {int(lib.lengths[0])} / {int(scurves[0].lengths[0])} steps; "
        f"S-bend steps over {NEAR_LIMIT} of the friction: {[int((s.util > NEAR_LIMIT).sum()) for s in scurves]})")
    heads = VehicleHeads(cfg, VehicleTrackingEnv(cfg, lib, device=args.device, record=False).dyn.p.R_w)
    run_eval = make_eval(heads, eval_libs, scurves, args)
    report = {"args": vars(args), "methods": {}, "scurve": {f"mu_{mu}": {} for mu in args.mu_levels}}
    policies, controllers = {}, {}
    blob = torch.load(args.policies_from, map_location=args.device) if args.policies_from else None
    for m in args.methods:
        if m.startswith("nmpc"):
            log(f"=== {m}: no training")
            final, sc, traj, info = run_nmpc(args, cfg, heads, eval_libs, scurves, log, m)
            report["methods"][m] = {**info, "curve": [], "heldout_final": final, "scurve_final": sc, "scurve_traj": traj}
            controllers[m] = info["nmpc"]
        else:
            if blob is not None and m in blob["policies"]:
                log(f"=== {m}: loaded from {args.policies_from}, no training")
                policy, curve, info = load_policy(args, cfg, lib, m, blob), [], {"loaded_from": str(args.policies_from)}
            else:
                log(f"=== {m}: {args.budget_min} min training budget")
                policy, curve, info = (train_supermimic if m == "supermimic" else train_ppo_method)(args, lib, run_eval, log)
            policy.eval()
            final = {f"mu_{mu}": evaluate(policy, heads, el, args) for mu, el in zip(args.mu_levels, eval_libs)}
            sc = {f"mu_{mu}": evaluate(policy, heads, s, args) for mu, s in zip(args.mu_levels, scurves)}
            traj = {f"mu_{mu}": trajectory(policy, heads, s, args) for mu, s in zip(args.mu_levels, scurves)}
            info["latency"] = policy_latency_ms(policy, heads, args)
            report["methods"][m] = {**info, "curve": curve, "heldout_final": final, "scurve_final": sc, "scurve_traj": traj}
            policies[m] = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        log(f"{m} final: held-out " + "  ".join(f"{k} {v['normalized_return']:.3f}" for k, v in final.items())
            + " | S-bend " + "  ".join(f"{k} {v['normalized_return']:.3f}{' (completed)' if v['termination_rate'] == 0 else ' (terminated)'}" for k, v in sc.items()))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report["wall_clock_s"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    torch.save({"args": vars(args), "policies": policies, "controllers": controllers}, out / "policies.pt")  # for render_scurve.py
    if not args.no_plots:
        plots(report, scurves, out)
    log(f"report: {out / 'report.json'}" + ("" if args.no_plots else f", plots: {out / 'curves.png'}, {out / 's_curve.png'}"))
    print("\nmethod      " + "  ".join(f"held-out {k}" for k in final) + "  |  " + "  ".join(f"S-bend {k}" for k in sc) + "  |  ms per step (CPU)")
    for m, res in report["methods"].items():
        print(f"{m:11s} " + "  ".join(f"{v['normalized_return']:13.3f}" for v in res["heldout_final"].values()) + "  |  "
              + "  ".join(f"{v['normalized_return']:.3f} {'ok ' if v['termination_rate'] == 0 else 'end'}" for v in res["scurve_final"].values())
              + f"  |  {res['latency']['mean_ms']:.3f} (p95 {res['latency']['p95_ms']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
