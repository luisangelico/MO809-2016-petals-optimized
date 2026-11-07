"""CLI: run objective-aware generation against a Petals swarm.

Examples:
    python router/run_client.py --initial_peers /ip4/192.168.2.109/tcp/31337 \
        --objective alpha=0.3 --prompt "The capital of France is" --max_new 20

    python router/run_client.py --initial_peers ... --objective quality_floor=16ppl \
        --models bigscience/bloom-560m bigscience/bloom-1b7 --csv experiments/run1.csv
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--initial_peers", nargs="+", required=True, help="hivemind maddr(s) of swarm peers")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bigscience/bloom-560m", "bigscience/bloom-1b7"],
        help="model tiers to route across (first = fast, last = accurate)",
    )
    parser.add_argument("--objective", default="alpha=0.5", help="alpha=<0..1> | latency_budget=<s> | quality_floor=<ppl>ppl")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max_new", type=int, default=20)
    parser.add_argument("--csv", default=None, help="append one instrumentation row per request to this CSV")
    parser.add_argument("--n_requests", type=int, default=1, help="repeat the request this many times")
    args = parser.parse_args(argv)

    from router.client import OptimizedClient
    from router.instrumentation import CSVLogger
    from router.objectives import parse_objective

    objective = parse_objective(args.objective)
    csv_logger = CSVLogger(args.csv) if args.csv else None

    try:
        client = OptimizedClient(initial_peers=args.initial_peers, models=args.models)
        client.warmup()
        for _ in range(args.n_requests):
            text = client.generate(args.prompt, objective=objective, max_new_tokens=args.max_new, csv_logger=csv_logger)
            print("\n===== RESULT =====")
            print("objective:", objective)
            print("text:", repr(text))
            print("==================")
        client.close()
    except KeyboardInterrupt:
        pass
    finally:
        if csv_logger is not None:
            csv_logger.close()


if __name__ == "__main__":
    sys.exit(main())
