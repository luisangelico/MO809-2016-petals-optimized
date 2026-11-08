"""Per-request routing objectives.

An `Objective` declares what one generation request cares about:
- `alpha`: interpolation between pure latency (0) and pure quality (1)
- `latency_budget_s`: hard latency ceiling — spend as much quality as fits under it
- `quality_floor_ppl`: hard quality floor — spend as little latency as possible above it

The objective for the *current* request travels through Petals' call stack via a
`contextvars.ContextVar`, because `InferenceSession._update_sequence()` calls
`make_sequence()` deep inside `generate()` with no objective parameter of its own.
"""

from __future__ import annotations

import dataclasses
from contextvars import ContextVar
from typing import Optional


@dataclasses.dataclass
class Objective:
    """Routing objective for a single generation request."""

    alpha: float = 0.5  # 0 = minimize latency, 1 = maximize quality
    latency_budget_s: Optional[float] = None  # predicted latency must fit under this
    quality_floor_ppl: Optional[float] = None  # predicted perplexity must stay under this

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")

    def __str__(self):
        parts = [f"alpha={self.alpha:.2f}"]
        if self.latency_budget_s is not None:
            parts.append(f"latency_budget={self.latency_budget_s:.2f}s")
        if self.quality_floor_ppl is not None:
            parts.append(f"quality_floor={self.quality_floor_ppl:.1f}ppl")
        return " ".join(parts)


# Set by OptimizedClient.generate() around each request; read inside make_sequence().
current_objective: ContextVar[Optional[Objective]] = ContextVar("current_objective", default=None)

# Per-request route decision, written by RequestAwareSequenceManager.make_sequence() at
# decision time and read back by the same request's thread after generation. Context-local
# on purpose: the manager itself is shared across concurrent requests, so its `last_spans`
# is whatever route was rebuilt last — reading it after generation races under the mixed
# concurrent load (all threads would record the final route). Format:
#   {"spans": [RemoteSpanInfo], "decision_time_s": float, "latency_s": float, "ppl": float}
current_route: ContextVar[Optional[dict]] = ContextVar("current_route", default=None)


def parse_objective(spec: str) -> Objective:
    """Parse a CLI objective spec into an Objective.

    Examples:
        "alpha=0.3"              -> weighted trade-off
        "latency_budget=2.0"     -> max quality that fits under 2 s predicted latency
        "quality_floor=18.0ppl"  -> min latency with predicted ppl <= 18
    """
    spec = spec.strip().lower()
    if spec.startswith("alpha="):
        return Objective(alpha=float(spec[len("alpha="):]))
    if spec.startswith("latency_budget="):
        return Objective(alpha=1.0, latency_budget_s=float(spec[len("latency_budget="):].rstrip("s")))
    if spec.startswith("quality_floor="):
        value = spec[len("quality_floor="):].removesuffix("ppl")
        return Objective(alpha=0.0, quality_floor_ppl=float(value))
    raise ValueError(
        f"Cannot parse objective {spec!r}; expected one of "
        "'alpha=<0..1>', 'latency_budget=<seconds>', 'quality_floor=<ppl>ppl'"
    )
