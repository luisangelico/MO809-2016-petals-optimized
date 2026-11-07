"""Aggregate experiment CSVs into per-configuration result tables.

Groups instrumentation rows by (policy, model, objective) and reports mean ± std of
tokens/second, TTFT, generation time, optimizer decision overhead, measured quality
(NLL under the fp32 reference, token agreement) and task accuracy — the numbers for
the report's tables. Optionally plots the latency-vs-quality Pareto frontier.

Usage:
    python router/summarize.py experiments/sweep-20261107.csv
    python router/summarize.py experiments/*.csv --format csv --out results/summary.csv
    python router/summarize.py experiments/sweep-*.csv --plot results/frontier.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
import sys
from collections import defaultdict


def fnum(value: str) -> float:
    try:
        v = float(value)
        return v
    except (TypeError, ValueError):
        return float("nan")


def load_rows(paths) -> list:
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["_source"] = path
                rows.append(row)
    return rows


def group_key(row) -> tuple:
    alpha = row.get("alpha", "")
    alpha = f"{fnum(alpha):.2f}" if alpha else ""
    return (row.get("policy", ""), row.get("model", ""), row.get("objective", "") or f"alpha={alpha}")


def summarize(rows) -> list:
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    summaries = []
    for (policy, model, objective), rs in sorted(groups.items()):
        def col(name):
            values = [fnum(r.get(name)) for r in rs]
            values = [v for v in values if not math.isnan(v)]
            if not values:
                return float("nan"), float("nan"), 0
            return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0), len(values)

        tok_mean, tok_std, _ = col("tok_s")
        ttft_mean, ttft_std, _ = col("ttft_s")
        gen_mean, gen_std, _ = col("gen_time_s")
        dec_mean, dec_std, _ = col("decision_time_s")
        nll_mean, nll_std, nll_n = col("ref_nll_per_token")
        agr_mean, agr_std, _ = col("token_agreement")

        # Task accuracy: fraction of task_match == True
        task_rows = [r for r in rs if r.get("task_match")]
        acc = (sum(1 for r in task_rows if r["task_match"].lower() == "true") / len(task_rows)) if task_rows else float("nan")

        errors = sum(1 for r in rs if r.get("error"))
        summaries.append(
            {
                "policy": policy,
                "model": model,
                "objective": objective,
                "n_requests": len(rs),
                "errors": errors,
                "tok_s_mean": round(tok_mean, 2),
                "tok_s_std": round(tok_std, 2),
                "ttft_s_mean": round(ttft_mean, 3),
                "ttft_s_std": round(ttft_std, 3),
                "gen_time_s_mean": round(gen_mean, 3),
                "gen_time_s_std": round(gen_std, 3),
                "decision_time_s_mean": round(dec_mean, 4),
                "decision_time_s_std": round(dec_std, 4),
                "ref_nll_per_token_mean": round(nll_mean, 3),
                "ref_nll_per_token_std": round(nll_std, 3),
                "ref_nll_n": nll_n,
                "token_agreement_mean": round(agr_mean, 3),
                "task_accuracy": round(acc, 3) if not math.isnan(acc) else "",
            }
        )
    return summaries


def format_markdown(summaries) -> str:
    headers = ["policy", "model", "objective", "n", "tok/s (mean±std)", "TTFT s", "decision ms", "ref NLL/tok", "agreement", "task acc"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for s in summaries:
        row = [
            s["policy"],
            s["model"].split("/")[-1],
            s["objective"],
            str(s["n_requests"]) + (f" ({s['errors']} err)" if s["errors"] else ""),
            f"{s['tok_s_mean']:.1f} ± {s['tok_s_std']:.1f}",
            f"{s['ttft_s_mean']:.3f}",
            f"{s['decision_time_s_mean'] * 1000:.1f}" if not math.isnan(s["decision_time_s_mean"]) else "—",
            f"{s['ref_nll_per_token_mean']:.3f}" if s["ref_nll_n"] else "—",
            f"{s['token_agreement_mean']:.2f}" if s["ref_nll_n"] else "—",
            str(s["task_accuracy"]) if s["task_accuracy"] != "" else "—",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plot_frontier(summaries, out_path):
    """Pareto frontier: mean tokens/s (latency axis, inverted) vs measured quality."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    policies = sorted({s["policy"] for s in summaries})
    colors = dict(zip(policies, ["#4269d0", "#efb118", "#ff725c", "#6cc5b0"]))

    for s in summaries:
        if s["objective"].startswith("alpha=") and s["ref_nll_n"]:
            color = colors.get(s["policy"], "gray")
            marker = "o" if s["policy"] == "optimized" else "s"
            axes[0].errorbar(s["objective"], s["tok_s_mean"], yerr=s["tok_s_std"], fmt=marker, color=color, capsize=3)
            axes[1].errorbar(s["objective"], s["ref_nll_per_token_mean"], yerr=s["ref_nll_per_token_std"], fmt=marker, color=color, capsize=3)

    axes[0].set_xlabel("objective (alpha: 0 = latency, 1 = quality)")
    axes[0].set_ylabel("tokens / second")
    axes[0].set_title("Throughput vs objective")
    axes[1].set_xlabel("objective (alpha: 0 = latency, 1 = quality)")
    axes[1].set_ylabel("NLL per token (lower = closer to FP32)")
    axes[1].set_title("Measured quality vs objective")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Plot written: {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+", help="experiment CSV path(s), globs allowed")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--out", default=None, help="write summary to this file instead of stdout")
    parser.add_argument("--plot", default=None, help="also render the Pareto frontier plot to this PNG")
    args = parser.parse_args(argv)

    paths = []
    for pattern in args.csvs:
        matched = glob.glob(pattern)
        paths.extend(matched or [pattern])

    rows = load_rows(paths)
    if not rows:
        print("No rows found in", paths, file=sys.stderr)
        return 1
    summaries = summarize(rows)

    if args.format == "csv":
        import os

        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out_file = open(args.out, "w", newline="") if args.out else sys.stdout
        writer = csv.DictWriter(out_file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)
        if args.out:
            out_file.close()
            print(f"Summary written: {args.out}")
    else:
        text = format_markdown(summaries)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text)
            print(f"Summary written: {args.out}")
        else:
            print(text)

    if args.plot:
        plot_frontier(summaries, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
