import os
import time

import torch
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM

MODEL = "bigscience/bloom-560m"
INITIAL_PEERS = [os.environ["FOUNDER_MADDR"]]
PROMPT = "The capital of France is"
MAX_NEW = 20

print(">> conectando ao swarm e montando a rota pelos blocos 0:24 ...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoDistributedModelForCausalLM.from_pretrained(
    MODEL, initial_peers=INITIAL_PEERS, torch_dtype=torch.float32
)

inputs = tok(PROMPT, return_tensors="pt")["input_ids"]

# time-to-first-token
t0 = time.perf_counter()
_ = model.generate(inputs, max_new_tokens=1)
ttft = time.perf_counter() - t0

# vazao em geracao maior
t1 = time.perf_counter()
out = model.generate(inputs, max_new_tokens=MAX_NEW)
dt = time.perf_counter() - t1

text = tok.decode(out[0], skip_special_tokens=True)
print("\n===== RESULTADO =====")
print("texto:", repr(text))
print(f"time-to-first-token: {ttft:.2f} s")
print(f"{MAX_NEW} tokens em {dt:.2f} s  ->  {MAX_NEW/dt:.2f} tok/s")
print("=====================")
