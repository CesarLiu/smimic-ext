"""VEHICLE_PROPOSAL.md section 11.16: the hierarchical baseline at 0.80 and 0.75 of the reference speed on the fresh
evaluation set of ``chrono_gas1.py --eval-checkpoint`` (same cases and roads). Argument: the directory that holds
``lift/data.npz`` and ``gas1/run/roads``; writes ``gas1/run5/slow.json`` there."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/vehicle")
import chrono_gas1 as g
import chrono_rollover_control as crc
from chrono_roll_mpc import RollMPC, identify

from supermimic.vehicles.chrono_env import ChronoPool

S = sys.argv[1]
args = g.parse(["--lift-data", f"{S}/lift/data.npz", "--road-dir", f"{S}/gas1/run/roads", "--extra-eval-paths", "8"])
out = {}
with ChronoPool(4) as pool:
    pool.reset([{**g.CFG, "v": 5.0}], ids=[0]); geo = pool.info(ids=[0])[0]
    n0 = args.train_roads + args.eval_paths + args.extra_eval_paths
    roads = g.Roads([*range(args.train_roads), *range(5000, 5000 + args.eval_paths), *range(6000, 6000 + args.extra_eval_paths), *range(7000, 7000 + args.fresh_paths)], args.road_dir, geo)
    rng, cases = np.random.default_rng(args.seed + 4000), []
    for e in range(args.fresh_paths):
        p, _ = crc.lane_change_path(rng); cases += [(p, v, n0 + e) for v in crc.EVAL_SPEEDS]
    ident = identify(args.lift_data)
    for sc in (0.80, 0.75):
        ev, _ = g.evaluate_g(pool, cases, roads, g.Hierarchical(RollMPC(ident), roads, True, sc), 4)
        out[f"hierarchical_slow_{sc:g}"] = ev
        print(f"scale {sc}: off path {ev['off_path']}, rolled {ev['rolled']}, return {ev['mean_return']:.1f}, |lat| {ev['mean_abs_lat']:.3f}, az {ev['rms_az']:.2f}, speed {ev['mean_speed']:.1f}", flush=True)
Path(f"{S}/gas1/run5/slow.json").write_text(json.dumps(out))
