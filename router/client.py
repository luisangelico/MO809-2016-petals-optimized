"""OptimizedClient: per-request model-tier selection + objective-aware routing.

Holds both model tiers (bloom-560m fast, bloom-1b7 accurate) over one shared DHT, with
each model's default `RemoteSequenceManager` replaced by `RequestAwareSequenceManager`
(the upstream petals checkout stays untouched). `generate()` sets the request's
objective in a ContextVar, lets `ModelSelector` pick the tier, and records one
instrumentation row per request.

Usage:
    client = OptimizedClient(initial_peers=["/ip4/.../tcp/31337"], models=["bloom-560m", "bloom-1b7"])
    client.warmup()
    out = client.generate("The capital of France is", objective=Objective(alpha=0.2))
    client.close()
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence

import torch
from hivemind import DHT, get_logger
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM

from router.instrumentation import CSVLogger, RequestRecord, estimate_activation_bytes, route_repr
from router.objectives import Objective
from router.policy import RequestAwareSequenceManager
from router.quality import QualityTable, normalize_dtype
from router.selector import ModelSelector

logger = get_logger(__name__)


class _ModelCandidate:
    def __init__(self, predicted_latency_s: float, quality_ppl: float):
        self.predicted_latency_s = predicted_latency_s
        self.quality_ppl = quality_ppl


class OptimizedClient:
    """Two-tier Petals client with per-request objective-aware routing."""

    def __init__(
        self,
        initial_peers: Sequence[str],
        models: Sequence[str] = ("bigscience/bloom-560m", "bigscience/bloom-1b7"),
        torch_dtypes: Optional[Dict[str, torch.dtype]] = None,
        quality_table: Optional[QualityTable] = None,
        quality_scale_s: float = 20.0,
    ):
        self.initial_peers = list(initial_peers)
        self.models: Dict[str, _LoadedModel] = {}
        self.quality_table = quality_table or QualityTable.load()
        self.selector = ModelSelector(self.quality_table)
        self.dtypes = torch_dtypes or {name: torch.float32 for name in models}
        self.dht: Optional[DHT] = None

        for i, name in enumerate(models):
            logger.info(f"Loading {name} ...")
            model = AutoDistributedModelForCausalLM.from_pretrained(
                name,
                initial_peers=self.initial_peers,
                torch_dtype=self.dtypes.get(name, torch.float32),
            )
            # Reuse the first model's DHT (created inside its sequence manager) for all tiers.
            if self.dht is None:
                self.dht = model.transformer.h.sequence_manager.dht
            self._install_optimizer(model, name, quality_scale_s)
            tokenizer = AutoTokenizer.from_pretrained(name)
            self.models[name] = _LoadedModel(model=model, tokenizer=tokenizer)
            logger.info(f"Loaded {name} ({len(model.transformer.h)} blocks)")

    def _install_optimizer(self, model, name: str, quality_scale_s: float):
        """Replace the model's default sequence manager with the request-aware subclass."""
        sequential = model.transformer.h
        old_manager = sequential.sequence_manager
        new_manager = RequestAwareSequenceManager(
            model.config,
            old_manager.block_uids,
            dht=self.dht,
            quality_table=self.quality_table,
            quality_scale_s=quality_scale_s,
        )
        if old_manager.is_alive:
            old_manager.shutdown()
        sequential.sequence_manager = new_manager

    # ------------------------------------------------------------------ requests

    def generate(
        self,
        prompt: str,
        objective: Optional[Objective] = None,
        max_new_tokens: int = 20,
        csv_logger: Optional[CSVLogger] = None,
        measure_ttft: bool = True,
    ) -> str:
        """One generation request routed by `objective`; appends an instrumentation row."""
        objective = objective or Objective()
        record = RequestRecord(objective=str(objective), alpha=objective.alpha)

        t0 = time.perf_counter()
        try:
            model_name = self.selector.choose(objective, self._candidates())
            record.model = model_name
            loaded = self.models[model_name]
            policy = loaded.model.transformer.h.sequence_manager

            inputs = loaded.tokenizer(prompt, return_tensors="pt")["input_ids"]

            from router.objectives import current_objective

            token = current_objective.set(objective)
            try:
                if measure_ttft:
                    t_ttft0 = time.perf_counter()
                    loaded.model.generate(inputs, max_new_tokens=1)
                    record.ttft_s = time.perf_counter() - t_ttft0

                t_gen0 = time.perf_counter()
                out = loaded.model.generate(inputs, max_new_tokens=max_new_tokens)
                record.gen_time_s = time.perf_counter() - t_gen0
            finally:
                current_objective.reset(token)

            record.n_tokens = max_new_tokens
            record.tok_s = max_new_tokens / record.gen_time_s if record.gen_time_s > 0 else float("nan")
            record.decision_time_s = policy.last_decision_time_s
            record.predicted_latency_s = _sum_route_latency(policy)
            record.route = route_repr(policy.last_spans)
            record.n_spans = len(policy.last_spans or [])
            record.quality_ppl = policy._route_perplexity(policy.last_spans or [], 0, len(policy))
            dtype_bytes = _dtype_num_bytes(self.dtypes.get(model_name, torch.float32))
            record.activation_bytes_per_token_est = estimate_activation_bytes(
                loaded.model.config.hidden_size, dtype_bytes, record.n_spans
            )
            text = loaded.tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            record.error = repr(e)
            logger.warning(f"Request failed: {e!r}")
            text = ""
        finally:
            if csv_logger is not None:
                csv_logger.log(record)
        if record.error:
            logger.warning(f"Request failed: {record.error}")
        else:
            logger.info(
                f"{record.model}: ttft={record.ttft_s:.2f}s {record.n_tokens} tokens in "
                f"{record.gen_time_s:.2f}s ({record.tok_s:.1f} tok/s), route=[{record.route}]"
            )
        return text

    def _candidates(self) -> Dict[str, _ModelCandidate]:
        candidates = {}
        for name, loaded in self.models.items():
            policy = loaded.model.transformer.h.sequence_manager
            latency = policy.predict_min_latency(cache_tokens_needed=loaded.model.config.max_length or None)
            candidates[name] = _ModelCandidate(
                predicted_latency_s=latency if latency is not None else float("inf"),
                quality_ppl=self.quality_table.model_perplexity(
                    name, [normalize_dtype(str(self.dtypes.get(name, torch.float32)))]
                ),
            )
        return candidates

    def warmup(self):
        """Warm the DHT route and pings with one short request per tier."""
        for name in self.models:
            self.selector.choose(Objective(alpha=0.0), self._candidates())  # refresh candidates
        self.generate("Hello", objective=Objective(alpha=0.0), max_new_tokens=1)

    def close(self):
        for loaded in self.models.values():
            manager = loaded.model.transformer.h.sequence_manager
            if manager.is_alive:
                manager.shutdown()
        if self.dht.is_alive():
            self.dht.shutdown()


class _LoadedModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def _dtype_num_bytes(dtype: torch.dtype) -> int:
    return {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.float64: 8}.get(dtype, 4)


def _sum_route_latency(policy: RequestAwareSequenceManager) -> float:
    """Latency-only sum of the last built route, from the policy's stored decision."""
    spans = policy.last_spans
    if not spans:
        return float("nan")
    # Per-block inference delays + one RTT-ish hop per span; a rough actual-latency proxy.
    total = 0.0
    for span in spans:
        rps = span.server_info.inference_rps or 300
        total += span.length / rps + 0.05  # 0.018 serialization + ~0.03 RTT allowance
    return total
