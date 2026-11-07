"""Measured generation quality per request, against a local fp32 reference.

The quality table predicts route quality offline; this module *measures* the quality of
each actual generation (proposal metric: divergence vs FP32):

- `nll_of_generated`: mean NLL per token of the swarm-generated text under the local
  fp32 reference model — lower = closer to what fp32 would produce.
- `agreement`: fraction of generated tokens identical to the reference greedy
  continuation.
- `task_match`: exact-match check for prompts with a known expected continuation
  (e.g. LAMBADA final word).

The reference model is loaded lazily per (model, fp32) and reused across requests.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class ReferenceEvaluator:
    """Lazy local fp32 reference model measuring output quality of swarm generations."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading local fp32 reference model {self.model_name} ...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float32
            ).to(self.device)
            self._model.eval()

    def continuation(self, prompt: str, max_new_tokens: int) -> str:
        """Greedy fp32 continuation of the prompt."""
        self._load()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self._tokenizer.decode(out[0], skip_special_tokens=True)

    def nll_of_generated(self, prompt: str, generated_text: str) -> float:
        """Mean NLL per generated token under the fp32 reference (NaN if no new tokens)."""
        self._load()
        prompt_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        full_ids = self._tokenizer(generated_text, return_tensors="pt").input_ids.to(self.device)
        n_prompt = min(prompt_ids.size(1), full_ids.size(1) - 1)
        if full_ids.size(1) <= n_prompt + 1:
            return float("nan")

        with torch.no_grad():
            logits = self._model(full_ids).logits
        # Positions n_prompt-1 .. end-1 predict tokens n_prompt .. end
        shift_logits = logits[:, n_prompt - 1 : -1, :]
        shift_labels = full_ids[:, n_prompt:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), reduction="sum"
        )
        return (loss / shift_labels.numel()).item()

    def agreement(self, prompt: str, generated_text: str, max_new_tokens: int) -> float:
        """Fraction of generated tokens matching the greedy fp32 continuation."""
        self._load()
        ref = self.continuation(prompt, max_new_tokens)
        gen_tokens = self._tokenizer.tokenize(generated_text)
        ref_tokens = self._tokenizer.tokenize(ref)
        if not gen_tokens or not ref_tokens:
            return float("nan")
        n = min(len(gen_tokens), len(ref_tokens))
        matches = sum(1 for a, b in zip(gen_tokens[:n], ref_tokens[:n]) if a == b)
        return matches / n

    def task_match(self, generated_text: str, expected: str) -> bool:
        """Exact-match: does the generation contain the expected continuation?"""
        return expected.strip().lower() in generated_text.strip().lower()
