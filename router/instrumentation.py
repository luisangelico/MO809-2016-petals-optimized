"""CSV instrumentation for the route-optimizer experiments.

One row per generation request: objective, chosen model, route composition, decision
time, predicted vs actual latency, TTFT, throughput, quality score, and estimated
activation bytes transferred per token.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
import os
import time
from typing import List, Optional

_FIELDS = [
    "timestamp",
    "request_id",
    "policy",  # "optimized" | "vanilla" (true Petals default routing, baseline)
    "model",
    "objective",
    "alpha",
    "route",
    "n_spans",
    "decision_time_s",
    "predicted_latency_s",
    "ttft_s",
    "gen_time_s",
    "n_tokens",
    "tok_s",
    "quality_ppl",  # predicted route perplexity (quality table)
    "ref_nll_per_token",  # measured: NLL of the generated text under the local fp32 reference
    "token_agreement",  # measured: fraction of tokens matching the fp32 greedy continuation
    "expected",
    "task_match",  # measured: exact-match for prompts with a known continuation
    "activation_bytes_per_token_est",
    "error",
]


@dataclasses.dataclass
class RequestRecord:
    policy: str = "optimized"
    model: str = ""
    objective: str = ""
    alpha: float = float("nan")
    route: str = ""
    n_spans: int = 0
    decision_time_s: float = float("nan")
    predicted_latency_s: float = float("nan")
    ttft_s: float = float("nan")
    gen_time_s: float = float("nan")
    n_tokens: int = 0
    tok_s: float = float("nan")
    quality_ppl: float = float("nan")
    ref_nll_per_token: float = float("nan")
    token_agreement: float = float("nan")
    expected: str = ""
    task_match: str = ""
    activation_bytes_per_token_est: int = 0
    error: str = ""


class CSVLogger:
    """Append-only CSV writer; one file per experiment run."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        self._file = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=_FIELDS)
        if new_file:
            self._writer.writeheader()
            self._file.flush()
        self._ids = itertools.count(1)

    def log(self, record: RequestRecord):
        row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "request_id": next(self._ids), **dataclasses.asdict(record)}
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


def estimate_activation_bytes(hidden_size: int, dtype_bytes: int, n_spans: int) -> int:
    """Estimated activation bytes transferred per generated token.

    Each token's hidden state crosses the network once per route span (plus the exit to
    the client), at `hidden_size` elements x `dtype_bytes` bytes. This is an estimate —
    hidden states are the dominant payload; protocol overhead is ignored.
    """
    hops = n_spans if n_spans > 0 else 1
    return hidden_size * dtype_bytes * (hops + 1)


def route_repr(spans) -> str:
    return " => ".join(f"{s.start}:{s.end}@…{str(s.peer_id)[-6:]}" for s in spans or [])
