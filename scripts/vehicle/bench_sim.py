"""Throughput of the vehicle simulator and environment (environment steps per second).

    PYTHONPATH=src python scripts/vehicle/bench_sim.py --device cuda --num-envs 4096 16384 65536 [--compile]

``dynamics``: ``VehicleDynamics.step`` alone (one 50 Hz control step = ``--substeps`` substeps).
``env``: ``VehicleTrackingEnv.step`` with random actions (actuators, reward, termination, replay recording,
resets and the reset sampler included). The E0 gate is >= 1e6 env steps / s on one GPU.
"""

from __future__ import annotations

import argparse
import time

import torch

from supermimic.vehicles.threads import add_threads_arg, limit_threads
from supermimic.vehicles.dynamics import Context, VehicleDynamics
from supermimic.vehicles.env import ReferenceLibrary, VehicleEnvCfg, VehicleTrackingEnv


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def bench(fn, device, steps, warmup=3):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    _sync(device)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-envs", type=int, nargs="+", default=[1024, 4096, 16384])
    ap.add_argument("--substeps", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--compile", action="store_true", help="torch.compile the dynamics step")
    add_threads_arg(ap)
    args = ap.parse_args()
    threads = limit_threads(args.threads)
    dev = args.device
    cfg0 = VehicleEnvCfg(substeps=args.substeps)
    lib = ReferenceLibrary.generate(cfg0, num=32, steps=600, seed=0, device=dev)
    print(f"device {dev}, substeps {args.substeps}, compile {args.compile}, cpu threads {threads}")
    for n in args.num_envs:
        dyn = VehicleDynamics(dt=0.02, substeps=args.substeps)
        step = torch.compile(dyn.step) if args.compile else dyn.step
        s = dyn.initial_state(torch.full((n,), 20.0, device=dev))
        fl = dyn.new_flags(n, dev)
        ctx = Context.nominal(n, dev)
        sr = 0.1 * torch.randn(n, device=dev)
        ac = torch.randn(n, device=dev)
        state = {"s": s, "f": fl}

        def dstep(step=step, state=state, sr=sr, ac=ac, ctx=ctx):
            state["s"], state["f"] = step(state["s"], state["f"], sr, ac, ctx)

        t_dyn = bench(dstep, dev, args.steps)
        cfg = VehicleEnvCfg(num_envs=n, substeps=args.substeps, replay_capacity=200)
        env = VehicleTrackingEnv(cfg, lib, device=dev)
        if args.compile:
            env.dyn.step = torch.compile(env.dyn.step)
        a = 0.3 * torch.randn(n, 2, device=dev)
        t_env = bench(lambda env=env, a=a: env.step(a), dev, args.steps)
        print(f"n {n:6d}  dynamics {n * args.steps / t_dyn:12,.0f} steps/s   env {n * args.steps / t_env:12,.0f} steps/s")


if __name__ == "__main__":
    main()
