"""Experiment harness: baselines and the latency-vs-quality experiment matrix.

Experiments (each appends one CSV row per request under experiments/):
1. `sweep`   — alpha 0 -> 1 (Pareto frontier); `--baselines` also runs the true Petals
               default (vanilla RemoteSequenceManager, single fast tier)
2. `mixed`   — concurrent mixed load (latency-sensitive vs quality-sensitive requests)
3. `lambada` — task accuracy (exact-match on LAMBADA continuations) per objective,
               comparing the optimizer against the vanilla baseline
4. fault injection — `docker stop` a replica mid-run; the client's inherited Petals
               rerouting recovers and the recovery lands in the CSV

Usage:
    python router/bench.py --initial_peers /ip4/.../tcp/31337 sweep --alphas 0 0.25 0.5 0.75 1 --baselines
    python router/bench.py --initial_peers ... mixed --n_lat 4 --n_qual 4
    python router/bench.py --initial_peers ... lambada --n_examples 50 --baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PROMPTS = ["The capital of France is", "Water boils at a temperature of", "The largest planet in the solar system is"]


def _make_client(args, **kwargs):
    from router.client import OptimizedClient

    return OptimizedClient(
        initial_peers=args.initial_peers,
        models=args.models if kwargs.get("baseline") else [m for m in args.models],
        baseline=kwargs.get("baseline", False),
        measure_output_quality=kwargs.get("measure_quality", args.measure_quality),
    )


def _csv(path):
    from router.instrumentation import CSVLogger

    return CSVLogger(path)


def cmd_sweep(args):
    """Pareto frontier: the same prompt set per objective preset; optionally vanilla baseline."""
    from router.objectives import Objective, parse_objective

    csv_logger = _csv(args.csv)
    try:
        client = _make_client(args)
        client.warmup()
        for spec in args.alphas:
            objective = parse_objective(spec if spec.startswith(("alpha=", "latency_", "quality_")) else f"alpha={spec}")
            for prompt in args.prompts:
                client.generate(prompt, objective=objective, max_new_tokens=args.max_new, csv_logger=csv_logger)
        client.close()

        if args.baselines:
            # True Petals default: vanilla RemoteSequenceManager, single fast tier.
            client = _make_client(args, baseline=True)
            client.warmup()
            for prompt in args.prompts * args.n_baseline_repeats:
                client.generate(prompt, max_new_tokens=args.max_new, csv_logger=csv_logger)
            client.close()
    finally:
        csv_logger.close()


def cmd_mixed(args):
    """Concurrent mixed load: latency-sensitive and quality-sensitive requests at once."""
    from router.objectives import Objective

    csv_logger = _csv(args.csv)
    try:
        client = _make_client(args)
        client.warmup()
        jobs = [Objective(alpha=0.0) for _ in range(args.n_lat)] + [Objective(alpha=1.0) for _ in range(args.n_qual)]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = []
            for i, obj in enumerate(jobs):
                prompt = args.prompts[i % len(args.prompts)]
                futures.append(pool.submit(client.generate, prompt, obj, args.max_new, csv_logger, False))
            for f in futures:
                f.result()
        client.close()
    finally:
        csv_logger.close()


def cmd_lambada(args):
    """Task accuracy: exact-match on LAMBADA continuations, per objective, vs baseline."""
    from router.objectives import Objective, parse_objective

    import datasets

    lambada = datasets.load_dataset("lambada", "lambada_openai", split="test")
    examples = [(ex["text"].rsplit(" ", 1)[0], ex["text"].rsplit(" ", 1)[1]) for ex in lambada.select(range(args.n_examples))]

    presets = [(spec if spec.startswith(("alpha=", "latency_", "quality_")) else f"alpha={spec}", parse_objective(spec if spec.startswith(("alpha=", "latency_", "quality_")) else f"alpha={spec}")) for spec in args.alphas]

    csv_logger = _csv(args.csv)
    try:
        client = _make_client(args)
        client.warmup()
        for label, objective in presets:
            hits = total = 0
            for prompt, expected in examples:
                text = client.generate(
                    prompt, objective=objective, max_new_tokens=5, csv_logger=csv_logger, expected=expected
                )
                if not text:
                    continue
                total += 1
                # The expected word should appear at the start of the continuation
                last_word = text[len(prompt):].strip().split()
                if last_word and last_word[0].lower().startswith(expected.lower()):
                    hits += 1
            print(f"objective {label}: task accuracy = {hits}/{total} = {hits/total:.1%}" if total else f"objective {label}: no results")
        client.close()

        if args.baseline:
            client = _make_client(args, baseline=True)
            client.warmup()
            hits = total = 0
            for prompt, expected in examples:
                text = client.generate(prompt, max_new_tokens=5, csv_logger=csv_logger, expected=expected)
                if not text:
                    continue
                total += 1
                last_word = text[len(prompt):].strip().split()
                if last_word and last_word[0].lower().startswith(expected.lower()):
                    hits += 1
            print(f"vanilla baseline: task accuracy = {hits}/{total} = {hits/total:.1%}" if total else "vanilla baseline: no results")
            client.close()
    finally:
        csv_logger.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--initial_peers", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["bigscience/bloom-560m", "bigscience/bloom-1b7"])
    parser.add_argument("--prompts", nargs="+", default=PROMPTS)
    parser.add_argument("--max_new", type=int, default=20)
    parser.add_argument("--measure_quality", action="store_true", help="measure NLL/agreement vs local fp32 reference per request")
    parser.add_argument("--csv", default=None, help="CSV path (default: experiments/<cmd>-<ts>.csv)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sweep = sub.add_parser("sweep", help="objective/alpha sweep (Pareto frontier)")
    p_sweep.add_argument("--alphas", nargs="+", default=["0", "0.25", "0.5", "0.75", "1"])
    p_sweep.add_argument("--baselines", action="store_true", help="also run the true Petals default (vanilla, single tier)")
    p_sweep.add_argument("--n_baseline_repeats", type=int, default=1)

    p_mixed = sub.add_parser("mixed", help="concurrent mixed load")
    p_mixed.add_argument("--n_lat", type=int, default=4, help="concurrent latency-sensitive requests")
    p_mixed.add_argument("--n_qual", type=int, default=4, help="concurrent quality-sensitive requests")

    p_lam = sub.add_parser("lambada", help="task accuracy on LAMBADA per objective vs baseline")
    p_lam.add_argument("--n_examples", type=int, default=50)
    p_lam.add_argument("--alphas", nargs="+", default=["0", "1"])
    p_lam.add_argument("--baseline", action="store_true", help="also run the vanilla baseline")

    args = parser.parse_args(argv)
    if args.csv is None:
        args.csv = f"experiments/{args.cmd}-{time.strftime('%Y%m%d-%H%M%S')}.csv"

    {"sweep": cmd_sweep, "mixed": cmd_mixed, "lambada": cmd_lambada}[args.cmd](args)
    print(f"CSV written: {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
