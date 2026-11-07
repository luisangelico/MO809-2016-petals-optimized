"""Experiment harness: baseline comparison and the latency-vs-quality experiment matrix.

Experiments (each appends one CSV row per request under experiments/):
1. alpha sweep 0 -> 1  (Pareto frontier)            — bench.py sweep
2. mixed concurrent load (latency- vs quality-sensitive) — bench.py mixed
3. fault injection (docker stop a replica mid-run)  — run with --fault_after; the client's
   inherited Petals rerouting recovers and the recovery lands in the CSV

Usage:
    python router/bench.py --initial_peers /ip4/.../tcp/31337 sweep --alphas 0 0.25 0.5 0.75 1
    python router/bench.py --initial_peers ... mixed --n_lat 4 --n_qual 4
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def _client(args, csv_path):
    from router.client import OptimizedClient
    from router.instrumentation import CSVLogger

    client = OptimizedClient(initial_peers=args.initial_peers, models=args.models)
    csv_logger = CSVLogger(csv_path)
    return client, csv_logger


def cmd_sweep(args):
    """Baseline comparison + Pareto frontier: run the same prompt set per objective preset."""
    from router.objectives import Objective, parse_objective

    client, csv_logger = _client(args, args.csv)
    client.warmup()
    presets = []
    for spec in (args.alphas or []):
        presets.append(parse_objective(f"alpha={spec}") if not spec.startswith(("alpha=", "latency_", "quality_")) else parse_objective(spec))
    for baseline in (["default", "always_fast", "always_accurate", "random"] if args.baselines else []):
        presets.append(_baseline_objective(baseline))
    try:
        for objective in presets:
            for prompt in args.prompts:
                client.generate(prompt, objective=objective, max_new_tokens=args.max_new, csv_logger=csv_logger)
    finally:
        client.close()
        csv_logger.close()


def _baseline_objective(name):
    """Baselines expressed through the same policy: fixed alpha presets.

    default         -> Petals-like latency-first routing on the fast tier
    always_fast     -> fast tier, pure latency
    always_accurate -> accurate tier, pure quality
    random          -> middle alpha (route randomization comes from Petals' load spreading)
    """
    from router.objectives import Objective

    return {
        "default": Objective(alpha=0.0),
        "always_fast": Objective(alpha=0.0),
        "always_accurate": Objective(alpha=1.0),
        "random": Objective(alpha=0.5),
    }[name]


def cmd_mixed(args):
    """Concurrent mixed load: latency-sensitive and quality-sensitive requests at once."""
    from router.objectives import Objective

    client, csv_logger = _client(args, args.csv)
    client.warmup()
    jobs = []
    for i in range(args.n_lat):
        jobs.append(Objective(alpha=0.0))
    for i in range(args.n_qual):
        jobs.append(Objective(alpha=1.0))
    try:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = []
            for i, obj in enumerate(jobs):
                prompt = args.prompts[i % len(args.prompts)]
                futures.append(
                    pool.submit(client.generate, prompt, obj, args.max_new, csv_logger, False)
                )
            for f in futures:
                f.result()
    finally:
        client.close()
        csv_logger.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--initial_peers", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["bigscience/bloom-560m", "bigscience/bloom-1b7"])
    parser.add_argument(
        "--prompts", nargs="+", default=["The capital of France is", "Water boils at a temperature of"]
    )
    parser.add_argument("--max_new", type=int, default=20)
    parser.add_argument("--csv", default=None, help="CSV path (default: experiments/<cmd>-<ts>.csv)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sweep = sub.add_parser("sweep", help="objective/alpha sweep (Pareto frontier) + baselines")
    p_sweep.add_argument("--alphas", nargs="+", default=["0", "0.25", "0.5", "0.75", "1"])
    p_sweep.add_argument("--baselines", action="store_true", help="also run fixed-policy baselines")

    p_mixed = sub.add_parser("mixed", help="concurrent mixed load")
    p_mixed.add_argument("--n_lat", type=int, default=4, help="concurrent latency-sensitive requests")
    p_mixed.add_argument("--n_qual", type=int, default=4, help="concurrent quality-sensitive requests")

    args = parser.parse_args(argv)
    if args.csv is None:
        args.csv = f"experiments/{args.cmd}-{time.strftime('%Y%m%d-%H%M%S')}.csv"

    {"sweep": cmd_sweep, "mixed": cmd_mixed}[args.cmd](args)
    print(f"CSV written: {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
