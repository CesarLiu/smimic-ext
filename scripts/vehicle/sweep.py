"""Multi-seed sweeps of ``compare_ppo.py`` on one or more GPUs (VEHICLE_PROPOSAL.md section 11.10).

Each plan is a list of configurations x seeds; every run is one ``compare_ppo.py`` process with its own output
directory (``<out>/<plan>/<config>/seed<k>``). Runs whose ``report.json`` exists are skipped, so an interrupted sweep
resumes where it stopped. ``--per-gpu`` processes share each GPU.

    python scripts/vehicle/sweep.py --plan stability --gpus 0 1 2 3 --per-gpu 3 --out logs/sweep
    python scripts/vehicle/sweep.py --plan ppo_seeds --gpus 0 --out logs/sweep
    python scripts/vehicle/sweep.py --plan stability --dry-run          # print the commands only
    python scripts/vehicle/summarize_runs.py logs/sweep/stability       # table: mean +- sd over seeds

Plans:
* ``stability``: why does SuperMimic's policy training oscillate (section 11.7)? Batch size (the vehicle runs used
  256; the humanoid default is 4096), imagination horizon, learning rate, number of environments, the ensemble with
  uncertainty penalty. 2-D, 5400 iterations each.
* ``ppo_seeds``: PPO with its section 3.11 budget (408 updates of 4096 x 32 steps = 13.4 M steps), for error bars.
* ``chassis_long``: the 5-D chassis setting of section 10.7 (b), with the batch size of ``--chassis-batch``, 10800
  iterations: run it after ``stability`` has picked a batch size.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).with_name("compare_ppo.py")
EVAL_STEPS = ["1e5", "2e5", "5e5", "1e6", "1382400", "2e6", "3e6", "5e6", "1e7", "1.4e7", "2e7"]
SM = ["--methods", "supermimic", "--budget-min", "1000", "--eval-every-s", "1e9", "--eval-at-steps", *EVAL_STEPS]


def plans(args) -> dict[str, tuple[list[tuple[str, list[str]]], list[int]]]:
    it = ["--sm-max-iters", str(args.iters)]
    return {
        "stability": ([
            ("base_b256_h16", [*SM, *it]),  # the CPU runs of section 11.7
            ("b1024_h16", [*SM, *it, "--batch", "1024"]),
            ("b4096_h16", [*SM, *it, "--batch", "4096"]),
            ("b4096_h32", [*SM, *it, "--batch", "4096", "--horizon", "32"]),  # the humanoid defaults
            ("b4096_h16_lr1e-4", [*SM, *it, "--batch", "4096", "--lr", "1e-4"]),
            ("b4096_h16_env1024", [*SM, *it, "--batch", "4096", "--num-envs", "1024"]),
            ("b4096_h16_ens3", [*SM, *it, "--batch", "4096", "--wm-ensemble", "3", "--unc-penalty", "10"]),
        ], list(range(args.seeds))),
        "ppo_seeds": ([
            ("ppo_408", ["--methods", "ppo", "--budget-min", "1000", "--eval-every-s", "1e9", "--eval-at-steps", *EVAL_STEPS,
                         "--ppo-max-updates", "408"]),
        ], list(range(args.seeds))),
        "chassis_long": ([
            (f"chassis_b{args.chassis_batch}", [*SM, "--sm-max-iters", "10800", "--chassis", "--batch", str(args.chassis_batch),
                                                "--wm-ensemble", "3", "--unc-penalty", "10", "--chassis-explore", "0.5",
                                                "--nmpc-data-steps", "3000", "--nmpc-data-envs", "32",
                                                "--nmpc-data-cache", str(Path(args.out) / "nmpc_data_32x3000.pt")]),
            ("ppo_chassis", ["--methods", "ppo", "--budget-min", "1000", "--eval-every-s", "1e9", "--eval-at-steps", *EVAL_STEPS,
                             "--ppo-max-updates", "408", "--chassis"]),
        ], list(range(args.seeds))),
    }


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, choices=["stability", "ppo_seeds", "chassis_long"])
    ap.add_argument("--configs", nargs="*", default=None, help="only these configurations of the plan")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5400, help="SuperMimic iterations (stability plan)")
    ap.add_argument("--chassis-batch", type=int, default=4096)
    ap.add_argument("--gpus", nargs="+", default=["0"], help="CUDA device indices; 'cpu' runs on the CPU")
    ap.add_argument("--per-gpu", type=int, default=2, help="concurrent runs per GPU")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per run")
    ap.add_argument("--out", default="logs/sweep")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    configs, seeds = plans(args)[args.plan]
    if args.configs:
        configs = [c for c in configs if c[0] in args.configs]
    jobs = []
    for name, extra in configs:
        for seed in seeds:
            out = Path(args.out) / args.plan / name / f"seed{seed}"
            if (out / "report.json").exists():
                continue
            device = "cpu" if args.gpus == ["cpu"] else "cuda"
            cmd = [sys.executable, str(SCRIPT), "--device", device, "--seed", str(seed), "--threads", str(args.threads),
                   "--no-plots", "--out-dir", str(out), *extra]
            jobs.append((out, cmd))
    print(f"{args.plan}: {len(jobs)} runs to do ({len(configs)} configurations x {len(seeds)} seeds, finished ones skipped)", flush=True)
    if args.dry_run:
        for out, cmd in jobs:
            print(" ".join(shlex.quote(c) for c in cmd))
        return 0
    slots = [g for g in args.gpus for _ in range(args.per_gpu)]
    running: dict[str, tuple[subprocess.Popen, Path, float, object]] = {}
    failed = []
    t0 = time.time()
    while jobs or running:
        for slot_id, gpu in enumerate(slots):
            key = f"{gpu}:{slot_id}"
            if key in running:
                p, out, ts, log = running[key]
                if p.poll() is None:
                    continue
                log.close()
                ok = p.returncode == 0 and (out / "report.json").exists()
                print(f"[{time.time() - t0:7.0f}s] {'done' if ok else 'FAILED'} {out} ({time.time() - ts:.0f}s)", flush=True)
                if not ok:
                    failed.append(str(out))
                del running[key]
            if jobs and key not in running:
                out, cmd = jobs.pop(0)
                out.mkdir(parents=True, exist_ok=True)
                env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "disable")}
                if gpu != "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = gpu
                log = open(out / "run.log", "w")  # noqa: SIM115  (open for the run's lifetime, closed when it ends)
                running[key] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), out, time.time(), log)
                print(f"[{time.time() - t0:7.0f}s] start {out} on {'cpu' if gpu == 'cpu' else 'gpu ' + gpu}", flush=True)
        time.sleep(5)
    print(f"finished in {time.time() - t0:.0f}s; {len(failed)} failed" + (": " + ", ".join(failed) if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
