# MO809-2016 — Optimized Petals: request-aware route optimization for distributed LLM inference

Final project for **MO809A** (Unicamp — Topics in Distributed Computing): a
**request-aware route optimizer** on top of [Petals](https://github.com/bigscience-workshop/petals)
(built on [Hivemind](https://github.com/learning-at-home/hivemind)), evaluated on a
heterogeneous home LAN swarm.

**The problem.** Petals' default routing (`RemoteSequenceManager`) picks servers to
optimize only for throughput/ping — every request is routed identically. But requests
differ: chat/interactive traffic needs **low latency**; summarization/reasoning
**tolerates more latency in exchange for higher quality**.

**The proposal.** Extend the client-side routing to decide, *per request*, which
nodes/replicas best serve a declared objective:

- minimize latency subject to a quality floor, or
- maximize quality within a latency budget,

via a cost function combining measured latency/throughput with a quality score,
parameterized per objective (α weight, latency SLA, or quality floor). The decision is
hierarchical: (1) per-request model selection (architecture/size), (2) per-block replica
selection (precision and node).

The full proposal is in [`apresentacao/`](apresentacao/) and the progress report in
[`relatorio-parcial.txt`](relatorio-parcial.txt) (Portuguese).

## Heterogeneous swarm (3 axes)

| Axis | Variation |
|---|---|
| Quality (coarse) | `bloom-560m` (fast) vs `bloom-1b7` (accurate) |
| Quality (fine) | fp32 vs bf16/fp16 |
| Latency (hardware) | MacBook M4 Pro (MPS, fast) vs Thinkpad T490 (CPU, slow) + stragglers |

Nodes: MacBook M4 Pro (arm64, macOS, GPU via MPS), Lenovo T490 (x86_64, Ubuntu 24.04,
24 GB RAM, no GPU — swarm founder), Lenovo T14 (optional extra). Heterogeneity is an
object of study, not a limitation — it is what makes per-request routing meaningful.

## Software stack (identical on every node, conda Python 3.11)

- `petals` 2.3.0.dev2 — routing policy implemented as an extension of `RemoteSequenceManager`
- `hivemind` 1.2.0.dev0 (pinned commit, required by Petals)
- `transformers` 4.43.1 (pinned, asserted at import)
- `torch` 2.14.0 (CPU on T490; MPS available on the Mac)

Quantization (bitsandbytes 8-bit/NF4) is unavailable here — it requires CUDA.

## Repo layout

```
router/          The request-aware route optimizer (extension package; petals stays pristine)
swarm-run/       Bare-metal 2-node swarm: client test script, server log, peer maddr
docker-swarm/    Dockerized heterogeneous swarm (11 virtual nodes / 3 hosts) — see its README
apresentacao/    Final-project proposal slide deck (generator script)
```

The upstream `hivemind/` and `petals/` checkouts are working copies only — they are
gitignored and not part of this repo.

## How the optimizer works

- **`router/objectives.py`** — a per-request `Objective`: `alpha` (0 = pure latency …
  1 = pure quality), an optional `latency_budget` or `quality_floor` (perplexity ceiling).
- **`router/policy.py`** — `RequestAwareSequenceManager` subclasses Petals'
  `RemoteSequenceManager` (upstream checkout untouched). It mirrors Petals' Dijkstra
  graph topology but keeps **latency** and **quality** edge weights separately and
  combines them as `cost = (1-α)·latency + α·quality_cost`. Constrained objectives
  (budget/floor) walk the weight until the predicted route satisfies the constraint.
  Petals' failure handling (banned peers, DHT rerouting) is inherited unchanged.
- **`router/quality.py`** — perplexity per (model, precision) pair, keyed off
  `ServerInfo.torch_dtype` (already announced to the DHT — fp32/bf16 replicas are
  client-visible with zero server-side changes). Measured offline by
  `router/measure_quality.py` (WikiText-2 sliding-window perplexity).
- **`router/selector.py`** — decision level 1: which model tier (fast 560m vs accurate
  1b7) serves this request.
- **`router/client.py`** — `OptimizedClient`: both tiers over one shared DHT; per
  `generate()` call it selects the tier, routes blocks by objective, and appends a CSV
  instrumentation row (TTFT, tok/s, decision time, route, activation bytes estimate).

## Running it

Against a live swarm (bare-metal or Docker):

```bash
python router/run_client.py --initial_peers /ip4/<FOUNDER_LAN_IP>/tcp/31337 \
  --objective alpha=0.3 --prompt "The capital of France is" --max_new 20 \
  --csv experiments/run1.csv
```

Objectives: `alpha=<0..1>` | `latency_budget=<seconds>` | `quality_floor=<ppl>ppl`.
Baselines and the Pareto sweep:

```bash
python router/bench.py --initial_peers ... sweep --alphas 0 0.25 0.5 0.75 1 --baselines
python router/bench.py --initial_peers ... mixed --n_lat 4 --n_qual 4
```

Regenerate the quality table offline (no swarm needed, uses the conda env):

```bash
export HF_HUB_DISABLE_XET=1
python router/measure_quality.py --models bigscience/bloom-560m bigscience/bloom-1b7
```

## Evaluation

**Metrics:** latency (TTFT, tok/s single-batch and concurrent), quality (perplexity,
divergence vs FP32, LAMBADA/QA accuracy), communication (activation bytes per token,
route composition), and optimizer overhead (route-decision time vs inference time).

**Baselines:** Petals default routing | always-fast | always-accurate | random.

**Experiments:**

1. **Speed vs quality trade-off** — Pareto frontier sweeping the objective (α = 0 … 1),
   vs the baselines.
2. **Adaptation to network/node degradation** — heterogeneity and stragglers via
   `tc netem` (T490) / Network Link Conditioner (Mac).
3. **Mixed concurrent loads** — simultaneous latency-sensitive and quality-sensitive requests.
4. **Fault tolerance mid-generation** — kill a node, measure DHT re-discovery and reroute time.

## Current status

Viability demonstrated: two-node swarm live on the LAN (T490 blocks 0:12 → Mac blocks
12:24), first end-to-end distributed inference with `bloom-560m` — TTFT ≈ 0.51 s,
≈ 14.7 tok/s (CPU, LAN, single batch, warm route; preliminary). Next: implement the
route-optimizer extension, the measurement harness (CSV of tok/s, TTFT, activation
bytes, quality scores), and run the experiment matrix.

## Quick start (bare-metal 2-node swarm)

Node 1 (T490, Ubuntu) — found the swarm and host blocks 0:12:

```bash
conda activate petals-mo809
python -m petals.cli.run_server bigscience/bloom-560m --num_blocks 12 --block_uid 0:12 \
  --device cpu --torch_dtype float32 --new_swarm \
  --host_maddrs /ip4/0.0.0.0/tcp/31337 \
  --announce_maddrs /ip4/<T490_LAN_IP>/tcp/31337
```

Node 2 (Mac) — host blocks 12:24 and join via the T490 as initial peer:

```bash
conda activate petals-mo809
python -m petals.cli.run_server bigscience/bloom-560m --num_blocks 12 --block_uid 12:24 \
  --device cpu --torch_dtype float32 \
  --initial_peers /ip4/<T490_LAN_IP>/tcp/31337 \
  --host_maddrs /ip4/0.0.0.0/tcp/31338 \
  --announce_maddrs /ip4/<MAC_LAN_IP>/tcp/31338
```

`--announce_maddrs` publishes the LAN address to the DHT explicitly — otherwise Docker
bridge / VPN IPs get announced, which the other node cannot route.

Client (`swarm-run/client_test.py`) — builds the route T490 (0:12) → Mac (12:24) from
the DHT and prints time-to-first-token and tok/s for a short generation:

```bash
python swarm-run/client_test.py
```

## Dockerized variant

[`docker-swarm/`](docker-swarm/README.md) scales the same setup to **7 virtual nodes
across the 3 physical hosts**, with per-container `cpus:`/`mem_limit:` tiers (fast /
medium / slow), a local image registry on the T490, and multi-arch builds (amd64 +
arm64) — the block-replicated, heterogeneous substrate the route optimizer studies.

## References

- Borzunov et al., *Petals: Collaborative Inference and Fine-tuning of Large Models*, ACL 2023 (System Demonstrations)
- Borzunov et al., *Distributed Inference and Fine-tuning of Large Language Models Over The Internet*, NeurIPS 2023
- Ryabinin et al., *SWARM Parallelism*, ICML 2023
- Ryabinin & Gusev, *Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts*, NeurIPS 2020
