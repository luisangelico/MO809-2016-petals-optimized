"""One-time offline quality measurement: validation perplexity per (model, dtype).

Loads each (model, dtype) pair locally (no swarm needed), computes perplexity on a
WikiText-2 validation subset with the sliding-window strategy, and writes
`router/quality_scores.json`. Run on the Mac with the petals-mo809 conda env:

    export HF_HUB_DISABLE_XET=1
    python router/measure_quality.py --models bigscience/bloom-560m bigscience/bloom-1b7

Requires the `datasets` package (pip install datasets).
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_PATH = os.path.join(os.path.dirname(__file__), "quality_scores.json")


def perplexity(model, tokenizer, texts: list, device: str = "cpu", stride: int = 512) -> float:
    """Sliding-window perplexity over a list of documents (mean of per-doc perplexities)."""
    import math

    log_ppls = []
    for text in texts:
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        n_tokens = input_ids.size(1)
        if n_tokens < 2:
            continue
        nlls, n_seen = [], 0
        prev_end = 0
        for begin in range(0, n_tokens, stride):
            end = min(begin + stride, n_tokens)
            trg_len = end - prev_end
            input_slice = input_ids[:, max(end - stride - 1, 0) : end]
            target = input_ids[:, begin:end]
            with torch.no_grad():
                logits = model(input_slice).logits
            shift_logits = logits[:, -trg_len:, :]
            shift_labels = target
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), reduction="sum"
            )
            nlls.append(loss)
            n_seen += trg_len
            prev_end = end
            if end == n_tokens:
                break
        log_ppls.append((sum(nlls) / n_seen).item())
    return math.exp(sum(log_ppls) / len(log_ppls))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["bigscience/bloom-560m", "bigscience/bloom-1b7"])
    parser.add_argument(
        "--dtypes", nargs="+", default=["float32", "bfloat16"], help="dtypes to evaluate per model"
    )
    parser.add_argument("--n_docs", type=int, default=100, help="number of WikiText-2 validation documents")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args(argv)

    import datasets

    wikitext = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = ["\n".join(t for t in wikitext[i : i + 8]["text"] if t.strip()) for i in range(0, args.n_docs, 8)]
    texts = [t for t in texts if t.strip()][: args.n_docs // 8] or texts[:1]

    scores = {}
    for model_name in args.models:
        scores[model_name.split("/")[-1]] = {}
        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            print(f"\n>> {model_name} @ {dtype_name} ...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(args.device)
            model.eval()
            ppl = perplexity(model, tokenizer, texts, device=args.device)
            scores[model_name.split("/")[-1]][dtype_name] = round(ppl, 2)
            print(f"   perplexity = {ppl:.2f}")
            del model
            torch.cpu.empty_cache()

    out = {"_comment": "Validation perplexity per (model, dtype) on a WikiText-2 subset. Regenerate with: python router/measure_quality.py", **scores}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}: {json.dumps(scores, indent=2)}")


if __name__ == "__main__":
    main()
