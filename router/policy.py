"""Request-aware sequence manager: Petals' routing with a quality term in the cost.

Subclasses `petals.client.routing.RemoteSequenceManager` — the upstream checkout stays
pristine. The two hooks:

- `make_sequence()`: ignores the hard-coded `mode` and routes by the per-request
  `Objective` (read from a ContextVar set by `OptimizedClient.generate()`). For
  latency-budget / quality-floor objectives it iteratively adjusts the latency-vs-quality
  weight until the predicted path satisfies the constraint.
- graph construction: mirrors Petals' `_build_inference_graph` topology (client->server
  RTT edges, forced server-exit edges, per-block inference edges) but keeps **latency**
  and **quality** edge weights separately and combines them as
  `cost = (1 - alpha) * latency + alpha * quality_cost`.

Failure handling (banned peers, rerouting, `on_request_failure`) is inherited unchanged,
so DHT-based fault tolerance keeps working.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple

import dijkstar
from hivemind import get_logger

from petals.client.routing.sequence_manager import MissingBlocksError, RemoteSequenceManager
from petals.data_structures import RemoteSpanInfo

from router.objectives import Objective, current_objective, current_route
from router.quality import QualityTable

logger = get_logger(__name__)


class RequestAwareSequenceManager(RemoteSequenceManager):
    """Routes each request by its declared latency-vs-quality objective.

    :param quality_table: perplexity lookup per (model, dtype); defaults to
      `router/quality_scores.json` (or built-in priors).
    :param quality_scale_s: latency-equivalent (in seconds) of the maximum quality loss
      of a server span; converts quality penalties into the same units as latency.
    """

    def __init__(self, *args, quality_table: Optional[QualityTable] = None, quality_scale_s: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_table = quality_table or load_quality_table()
        self.quality_scale_s = quality_scale_s
        self._ppl_range = self._compute_ppl_range()
        # Observability: last route built and how long the decision took (read by instrumentation).
        self.last_spans: Optional[List[RemoteSpanInfo]] = None
        self.last_decision_time_s: float = 0.0
        self.last_objective: Optional[Objective] = None

    # ------------------------------------------------------------------ hooks

    def make_sequence(
        self,
        start_index: int = 0,
        end_index: Optional[int] = None,
        *,
        mode: Optional[str] = None,  # ignored: the objective decides the trade-off
        cache_tokens_needed: Optional[int] = None,
    ) -> List[RemoteSpanInfo]:
        with self._thread_start_lock:
            if not self.is_alive():
                self._thread.start()
        if not self.ready.is_set():
            self.update(wait=True)

        end_index = end_index if end_index is not None else len(self)
        objective = current_objective.get() or Objective()
        t0 = time.perf_counter()

        span_sequence, latency_s, route_ppl = self._make_sequence_objective_aware(
            start_index, end_index, objective, cache_tokens_needed=cache_tokens_needed
        )

        decision_time_s = time.perf_counter() - t0
        self.last_decision_time_s = decision_time_s
        self.last_objective = objective
        self.last_spans = span_sequence

        route_repr = " => ".join(f"{s.start}:{s.end} via …{str(s.peer_id)[-6:]}" for s in span_sequence)
        logger.info(f"Route found ({objective}): {route_repr}")
        # Also expose the decision context-locally so the requesting thread (not the shared
        # manager state) captures ITS route after generation — the manager is shared across
        # concurrent requests and its last_spans races under the mixed load.
        current_route.set(
            {
                "spans": span_sequence,
                "decision_time_s": decision_time_s,
                "latency_s": latency_s,
                "ppl": route_ppl,
            }
        )
        return span_sequence

    # ------------------------------------------------------- objective-aware routing

    def _make_sequence_objective_aware(
        self,
        start_index: int,
        end_index: int,
        objective: Objective,
        *,
        cache_tokens_needed: Optional[int],
    ) -> Tuple[List[RemoteSpanInfo], float, float]:
        """Route by objective; returns (spans, predicted latency s, route perplexity)."""
        if start_index == end_index:
            return [], 0.0, float("inf")

        # For constrained objectives, start at the preferred end of the trade-off and walk
        # the weight towards the other end until the constraint is satisfied.
        alpha = objective.alpha
        if objective.latency_budget_s is not None:
            alpha = 1.0  # start with the best quality, trade it away until latency fits
        elif objective.quality_floor_ppl is not None:
            alpha = 0.0  # start with the best latency, trade it away until quality fits

        spans: List[RemoteSpanInfo] = []
        latency_s = route_ppl = float("nan")
        for _ in range(7):  # bounded adjustment loop
            spans, latency_s, route_ppl = self._route_with_alpha(
                start_index, end_index, alpha, cache_tokens_needed=cache_tokens_needed
            )
            if not spans:
                return [], latency_s, route_ppl
            if objective.latency_budget_s is not None:
                if latency_s <= objective.latency_budget_s or alpha <= 0.0:
                    break
                alpha = max(0.0, alpha / 2)  # trade quality away until latency fits
            elif objective.quality_floor_ppl is not None:
                if route_ppl <= objective.quality_floor_ppl or alpha >= 1.0:
                    break
                alpha = min(1.0, alpha + (1.0 - alpha) / 2)  # trade latency away until quality fits
            else:
                break
        return spans, latency_s, route_ppl

    def _route_with_alpha(
        self,
        start_index: int,
        end_index: int,
        alpha: float,
        *,
        cache_tokens_needed: Optional[int],
    ) -> Tuple[List[RemoteSpanInfo], float, float]:
        """Build the combined-cost graph, find the best path, and decompose it into
        (spans, predicted latency in seconds, route perplexity)."""
        weights = self._build_weighted_graph(
            start_index, end_index, alpha=alpha, cache_tokens_needed=cache_tokens_needed
        )
        path = dijkstar.find_path(weights.graph, "start", "end")
        if start_index == 0 and end_index == len(self):
            logger.debug(f"Expected latency: {path.total_cost:.2f} s (alpha={alpha:.2f})")

        latency_s = 0.0
        for u, v in zip(path.nodes[:-1], path.nodes[1:]):
            latency_s += weights.edges[(u, v)][0]

        span_sequence = []
        for peer_id, block_idx in path.nodes[1:-1]:
            if not span_sequence or span_sequence[-1].peer_id != peer_id:
                span_sequence.append(RemoteSpanInfo(peer_id, block_idx, block_idx, weights.server_infos[peer_id]))
            else:
                span_sequence[-1].end = block_idx
        span_sequence = [span for span in span_sequence if span.length > 0]

        route_ppl = self._route_perplexity(span_sequence, start_index, end_index)
        return span_sequence, latency_s, route_ppl

    def _route_perplexity(self, span_sequence: List[RemoteSpanInfo], start_index: int, end_index: int) -> float:
        """Length-weighted mean perplexity of the blocks on this route."""
        total_blocks, weighted = 0, 0.0
        for span in span_sequence:
            lo, hi = max(span.start, start_index), min(span.end, end_index)
            if hi <= lo:
                continue
            ppl = self.quality_table.perplexity(self.config.dht_prefix, span.server_info.torch_dtype)
            weighted += ppl * (hi - lo)
            total_blocks += hi - lo
        return weighted / total_blocks if total_blocks else float("inf")

    def predict_min_latency(self, *, cache_tokens_needed: Optional[int] = None) -> Optional[float]:
        """Predicted latency (seconds) of the pure-latency route; None if no route exists."""
        try:
            _, latency_s, _ = self._route_with_alpha(0, len(self), 0.0, cache_tokens_needed=cache_tokens_needed)
            return latency_s
        except MissingBlocksError:
            return None

    # ------------------------------------------------------------- graph construction
    #
    # Same topology as petals.client.routing.RemoteSequenceManager._build_inference_graph,
    # but latency and quality edge weights are tracked separately and combined per alpha:
    #   edge cost = (1 - alpha) * latency + alpha * quality_cost
    # Quality is charged per block: a route's quality cost is its blocks' mean quality
    # loss, scaled to latency-equivalent seconds, so a route through more
    # lower-precision blocks is penalized proportionally.

    def _build_weighted_graph(
        self,
        start_index: int,
        end_index: int,
        *,
        alpha: float,
        cache_tokens_needed: Optional[int],
        overhead_delay: float = 0.018,  # serialization overhead (empirical, from petals)
        default_inference_rps: float = 300,  # if inference RPS unknown
        alloc_delay: float = 10,  # penalty for spans without enough KV cache left
        default_delay: float = 0.15,  # if RTT unknown
        max_delay: float = 5,  # cap for unreachable peers (keep the edge, petals-style)
    ) -> "_WeightedGraph":
        missing_blocks = [
            block_idx
            for block_idx in range(start_index, end_index)
            if not self.state.sequence_info.spans_containing_block[block_idx]
        ]
        if missing_blocks:
            raise MissingBlocksError(missing_blocks)

        client_server_rtts = self.ping_aggregator.to_dict()
        n_blocks = len(self)

        def rtt_delay(rtt: Optional[float]) -> float:
            if rtt is None:
                return default_delay
            return min(rtt / 2, max_delay)

        def alloc_penalty(span: RemoteSpanInfo) -> float:
            if cache_tokens_needed is None or span.server_info.cache_tokens_left is None:
                return 0.0
            if cache_tokens_needed * 2 * span.length > span.server_info.cache_tokens_left:
                return alloc_delay
            return 0.0

        def quality_cost_per_block(span: RemoteSpanInfo) -> float:
            ppl = self.quality_table.perplexity(self.config.dht_prefix, span.server_info.torch_dtype)
            lo, hi = self._ppl_range
            normalized = (ppl - lo) / (hi - lo) if hi > lo else 0.0
            return normalized * self.quality_scale_s / n_blocks

        graph = dijkstar.Graph()
        edges: Dict[Tuple, Tuple[float, float]] = {}
        server_infos = {
            span.peer_id: span.server_info
            for block_idx in range(start_index, end_index)
            for span in self.state.sequence_info.spans_containing_block[block_idx]
        }

        def add_edge(u, v, latency: float, quality: float):
            graph.add_edge(u, v, (1.0 - alpha) * latency + alpha * quality)
            edges[(u, v)] = (latency, quality)

        # Client -> server network delays
        for span in self.state.sequence_info.spans_containing_block[start_index]:
            latency = rtt_delay(client_server_rtts.get(span.peer_id)) + overhead_delay + alloc_penalty(span)
            add_edge("start", (span.peer_id, start_index), latency, 0.0)

        # Server -> client network delays
        for span in self.state.sequence_info.spans_containing_block[end_index - 1]:
            add_edge((span.peer_id, end_index), "end", rtt_delay(client_server_rtts.get(span.peer_id)), 0.0)

        # Server -> server network delays (force going to the end of each server, petals-style)
        for block_idx in range(start_index + 1, end_index):
            for cur_span in self.state.sequence_info.spans_containing_block[block_idx - 1]:
                if cur_span.end != block_idx:
                    continue
                for next_span in self.state.sequence_info.spans_containing_block[block_idx]:
                    rtt = None
                    if cur_span.server_info.next_pings is not None:
                        rtt = cur_span.server_info.next_pings.get(next_span.peer_id.to_base58())
                    latency = rtt_delay(rtt) + overhead_delay + alloc_penalty(next_span)
                    add_edge((cur_span.peer_id, block_idx), (next_span.peer_id, block_idx), latency, 0.0)

        # Per-block edges: inference latency + that block's quality cost
        for span in self.state.sequence_info.spans_by_priority:
            per_block_quality = quality_cost_per_block(span)
            for block_idx in range(max(span.start, start_index), min(span.end, end_index)):
                inference_rps = span.server_info.inference_rps or default_inference_rps
                add_edge(
                    (span.peer_id, block_idx), (span.peer_id, block_idx + 1), 1.0 / inference_rps, per_block_quality
                )

        return _WeightedGraph(graph=graph, edges=edges, server_infos=server_infos)

    def _compute_ppl_range(self) -> Tuple[float, float]:
        all_ppls = [
            p
            for key, model_scores in self.quality_table.scores.items()
            if not key.startswith("_")
            for p in model_scores.values()
        ]
        return (min(all_ppls), max(all_ppls)) if all_ppls else (0.0, 0.0)


@dataclasses.dataclass
class _WeightedGraph:
    graph: dijkstar.Graph
    edges: Dict[Tuple, Tuple[float, float]]  # (u, v) -> (latency_s, quality_cost_s)
    server_infos: Dict
