"""Hierarchical decision, level 1: which model tier should serve this request?

Given the objective and both tiers' predicted latency (from their routing policies) and
quality (from the quality table), pick bloom-560m (fast) or bloom-1b7 (accurate):

- weighted `alpha`: minimize `(1 - alpha) * normalized_latency + alpha * normalized_quality_loss`
- `latency_budget`: best-quality tier whose predicted latency fits the budget (fallback: fastest)
- `quality_floor`: fastest tier whose predicted perplexity meets the floor (fallback: best quality)
"""

from __future__ import annotations

from typing import Dict, Optional

from hivemind import get_logger

from router.objectives import Objective
from router.quality import QualityTable

logger = get_logger(__name__)


class ModelSelector:
    def __init__(self, quality_table: Optional[QualityTable] = None):
        self.quality_table = quality_table or QualityTable.load()

    def choose(
        self,
        objective: Objective,
        candidates: Dict[str, "ModelCandidate"],  # model_name -> predicted latency / quality
    ) -> str:
        """Pick the model tier for this request. `candidates` maps model names to
        objects with `.predicted_latency_s` (float) and `.quality_ppl` (float)."""
        assert candidates, "No model tiers available"

        fastest = min(candidates, key=lambda name: candidates[name].predicted_latency_s)
        best_quality = min(candidates, key=lambda name: candidates[name].quality_ppl)

        if objective.latency_budget_s is not None:
            fits = {n: c for n, c in candidates.items() if c.predicted_latency_s <= objective.latency_budget_s}
            chosen = min(fits, key=lambda n: candidates[n].quality_ppl) if fits else fastest
        elif objective.quality_floor_ppl is not None:
            fits = {n: c for n, c in candidates.items() if c.quality_ppl <= objective.quality_floor_ppl}
            chosen = min(fits, key=lambda n: candidates[n].predicted_latency_s) if fits else best_quality
        else:
            lat_range = self._range(c.predicted_latency_s for c in candidates.values())
            ppl_range = self._range(c.quality_ppl for c in candidates.values())
            chosen = min(
                candidates,
                key=lambda n: (1 - objective.alpha) * _unit(candidates[n].predicted_latency_s, lat_range)
                + objective.alpha * _unit(candidates[n].quality_ppl, ppl_range),
            )

        if chosen != fastest:
            logger.info(
                f"Model selection: {chosen} (objective {objective}); "
                f"latency {candidates[chosen].predicted_latency_s:.2f}s vs fastest "
                f"{candidates[fastest].predicted_latency_s:.2f}s"
            )
        return chosen

    @staticmethod
    def _range(values) -> tuple:
        values = list(values)
        lo, hi = min(values), max(values)
        return (lo, hi) if hi > lo else (lo, lo + 1e-9)


def _unit(value: float, rng: tuple) -> float:
    return (value - rng[0]) / (rng[1] - rng[0])
