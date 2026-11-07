"""Quality table: measured generation quality per (model, precision) pair.

Quality is measured offline (see `measure_quality.py`) as validation perplexity on a
WikiText-2 subset. At routing time, the table is a plain dict lookup keyed by
(dht_prefix, torch_dtype) — the two pieces of metadata Petals' `ServerInfo` already
announces to the DHT, so replicas at different precisions are distinguishable by the
client with zero server-side changes.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

# Keyed by (model_key, dtype), where model_key is a dht_prefix fragment
# (e.g. "bloom-560m") and dtype is the normalized ServerInfo.torch_dtype.
# Lower perplexity = higher quality.
DEFAULT_QUALITY_SCORES: Dict[str, Dict[str, float]] = {
    "bloom-560m": {
        "float32": 22.0,  # placeholder priors — regenerate with measure_quality.py
        "bfloat16": 22.3,
    },
    "bloom-1b7": {
        "bfloat16": 14.5,
        "float32": 14.2,
    },
}

Path = str
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "quality_scores.json")


def normalize_dtype(dtype: Optional[str]) -> str:
    """Normalize a ServerInfo.torch_dtype string, e.g. 'torch.bfloat16' -> 'bfloat16'."""
    if not dtype:
        return "float32"  # servers that don't report dtype are treated as fp32
    dtype = str(dtype).removeprefix("torch.").lower()
    return {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}.get(dtype, dtype)


def model_key_for_prefix(dht_prefix: str) -> str:
    """Map a dht_prefix like 'bigscience/bloom-560m-petals' to a table key 'bloom-560m'."""
    name = dht_prefix.rstrip("/").split("/")[-1]
    name = name.removesuffix("-petals")
    for key in DEFAULT_QUALITY_SCORES:
        if key in name:
            return key
    return name


class QualityTable:
    """Lookup of generation quality (perplexity) per (model, dtype)."""

    def __init__(self, scores: Dict[str, Dict[str, float]]):
        self.scores = scores

    def perplexity(self, dht_prefix: str, dtype: Optional[str]) -> float:
        model_key = model_key_for_prefix(dht_prefix)
        dtype = normalize_dtype(dtype)
        try:
            return self.scores[model_key][dtype]
        except KeyError:
            # Fall back to the model's best-known dtype, then to a pessimistic default.
            model_scores = self.scores.get(model_key, {})
            if model_scores:
                return min(model_scores.values())
            return 30.0

    def model_perplexity(self, dht_prefix: str, available_dtypes) -> float:
        """Best (lowest) perplexity the model tier can offer given its serving dtypes."""
        model_key = model_key_for_prefix(dht_prefix)
        model_scores = self.scores.get(model_key, {})
        dtypes = {normalize_dtype(d) for d in available_dtypes} or set(model_scores)
        eligible = [ppl for d, ppl in model_scores.items() if d in dtypes]
        return min(eligible) if eligible else 30.0

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "QualityTable":
        if os.path.exists(path):
            with open(path) as f:
                return cls(json.load(f))
        return cls(DEFAULT_QUALITY_SCORES)
