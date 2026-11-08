# Architecture — Nectar: objective-aware replica selection for distributed LLM inference in Petals

This document describes the hardware testbed and the software architecture developed for
the MO809A final project. The system extends Petals' client-side routing so that
**each request** decides, via a declared objective (latency-sensitive vs
quality-sensitive), which model tier, blocks, replicas, precisions, and nodes to use.

---

## 1. What was developed (overview)

Four contributions on top of upstream Petals/Hivemind (both upstream checkouts stay pristine):

1. **Request-aware route optimizer** (`router/` package, ~1.5 kLOC):
   per-request model selection + per-block replica/precision/node selection with a
   pluggable objective (weighted α trade-off, latency budget, or quality floor).
2. **Quality model**: per-(model, dtype) perplexity table (measured on WikiText-2)
   consumed by the routing cost function, plus online quality evaluation
   (NLL-of-generated vs FP32, token agreement, task accuracy) during experiments.
3. **Dockerized heterogeneous swarm**: 12 virtual nodes across 3 physical hosts with
   controlled placement (tiers, precisions, CPU/memory caps) and a private registry.
4. **Experiment harness + instrumentation**: per-request CSV telemetry (TTFT, tok/s,
   decision overhead, route composition, activation bytes, quality), a Pareto sweep,
   mixed concurrent loads, LAMBADA accuracy, and fault injection.

---

## 2. Hardware architecture

### 2.1 Physical testbed (3 heterogeneous hosts on the home LAN)

| Host | Hardware | OS / runtime | Role in the swarm |
|---|---|---|---|
| **Mac-DJ3V7M49DX** | Apple M4 Pro, 24 GB RAM (10-core CPU) | macOS, Docker Desktop (Linux VM, 18 GB) | **Fast tier** — hosts fp32 and bf16 replicas of both models, the bloom-1b7 tier, and the benchmark client |
| **Lenovo T490** | Intel CPU | Linux (WSL2/native), Docker | **Bootstrap node** — DHT founder (port 31337) + first blocks of bloom-560m + 1b7 tier; also runs the private Docker registry (port 5001) |
| **Lenovo T14 (BYC-3DLSaxkXMSw)** | Intel CPU, 32 GB RAM | Windows 11 + WSL2, apt-installed Docker | **Corporate-network weak tier** — fp32 and bf16 bloom-560m shards; the straggler/remote-network case in the experiments |

All three hosts run **CPU-only inference** (fp32/bfloat16; no GPU in any container).

### 2.2 Network topology

```
        home LAN 192.168.2.0/24
  ┌─────────────┐   ┌─────────────────┐   ┌──────────────────────┐
  │ Mac         │   │ T490            │   │ T14 (corporate host) │
  │ .163        │   │ .109            │   │ .164                 │
  │ 5 containers│   │ 4 containers    │   │ 3 containers (WSL2)  │
  └──────┬──────┘   └───────┬─────────┘   └──────────┬───────────┘
         │  DHT (libp2p TCP)│                        │
         └────────┬─────────┴────────────────────────┘
                  ▼
        bootstrap DHT founder  /ip4/192.168.2.109/tcp/31337
                  +
        private Docker registry  192.168.2.109:5001
        (multi-arch: linux/arm64 + linux/amd64)
```

- All peers join a **Hivemind DHT** (libp2p) rooted at the T490 founder; every server
  announces the block ranges it serves (`bloom-560m-petals.*`, `bloom-1b7-petals.*`).
- The registry is marked *insecure* in every host's Docker daemon (LAN-only).
- **WSL2 specifics on the T14**: WSL2's NAT does not forward inbound LAN connections,
  so the T14 publishes its containers through Windows-side
  `netsh interface portproxy` rules (31342–31344 → WSL IP) plus a Windows firewall
  inbound-allow rule; re-applied after reboots (WSL IP is DHCP-assigned).

### 2.3 Virtual-node topology (12 containers)

Each bloom model is split into **24 blocks**; containers are the placement unit.

| Host | Container | Blocks (of 24) | Precision | CPU/RAM caps | Purpose |
|---|---|---|---|---|---|
| T490 | founder | 4 (0–3) | fp32 | 2 / 2 GB | DHT bootstrap + blocks |
| T490 | slow | 1 | fp32 | 1 / 1 GB | straggler node |
| T490 | medium | 4 | bf16 | 2 / 2 GB | fast replica |
| T490 | 1b7 | 24 | bf16 | 4 / 6 GB | accurate tier |
| Mac | fast | 9 | fp32 | 4 / 4 GB | main fp32 replica |
| Mac | medium | 4 | fp32 | 2 / 2 GB | fp32 replica |
| Mac | slow | 1 | fp32 | 1 / 1 GB | straggler node |
| Mac | fast-bf16 | 9 | bf16 | 4 / 3 GB | low-latency replica |
| Mac | 1b7 | 24 | bf16 | 4 / 6 GB | accurate tier |
| T14 | medium | 4 | fp32 | 2 / 2 GB | corporate-network shard |
| T14 | slow | 1 | fp32 | 1 / 1 GB | straggler node |
| T14 | bf16 | 2 | bf16 | 2 / 2 GB | fast replica |

- **Replicas**: several blocks of bloom-560m exist in *both* fp32 and bf16 on
  different hosts — this is what makes precision/node selection meaningful.
- **Model tiers**: bloom-560m (fast) and bloom-1b7 (accurate) are announced under
  separate DHT prefixes, so per-request *model selection* is a routing decision.
- The swarm's own block-rebalancing (Petals `block_selection`) is left enabled; hosts
  periodically migrate blocks to improve swarm balance, and migrations are announced
  to the DHT (declare offline → declare joining).

---

## 3. Software architecture

### 3.1 Stack layers

```
┌────────────────────────────────────────────────────────────────┐
│ experiments: bench.py (sweep | mixed | lambada), summarize.py  │
├────────────────────────────────────────────────────────────────┤
│ router/  (developed)                                           │
│   OptimizedClient · ModelSelector · RequestAwareSequenceManager │
│   QualityTable · CSVLogger/RequestRecord · objectives          │
├────────────────────────────────────────────────────────────────┤
│ Petals 2.3.0.dev2 (upstream, unmodified)                       │
│   AutoDistributedModelForCausalLM · InferenceSession           │
│   RemoteSequenceManager (subclassed, not forked)               │
├────────────────────────────────────────────────────────────────┤
│ Hivemind (upstream, unmodified)                                │
│   DHT (Kademlia over libp2p) · RPC · compression               │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 The `router/` package

| Module | LOC | Responsibility |
|---|---|---|
| `objectives.py` | 64 | `Objective` dataclass — `alpha`, `latency_budget_s`, `quality_floor_ppl`; travels through Petals' call stack via a `contextvars.ContextVar` (because `InferenceSession._update_sequence()` has no objective parameter) |
| `selector.py` | 70 | **Hierarchical decision, level 1**: `ModelSelector` picks bloom-560m vs bloom-1b7 per request — weighted-α normalization, or constraint-satisfaction for budget/floor objectives |
| `policy.py` | 289 | **Hierarchical decision, level 2**: `RequestAwareSequenceManager(RemoteSequenceManager)` — routes per-block by objective; mirrors Petals' inference-graph topology but keeps **latency and quality edge weights separately**, combining them as `cost = (1 − α)·latency + α·quality_cost`; iteratively adjusts α until budget/floor constraints are met. Failure handling (banned peers, rerouting, `on_request_failure`) is inherited unchanged |
| `client.py` | 245 | `OptimizedClient` — holds both tiers over one shared DHT; `generate()` sets the request's objective, picks the tier, records one telemetry row per request; `warmup()`, `close()` |
| `quality.py` / `quality_scores.json` | 83 | `QualityTable` — measured perplexity per (model, dtype): bloom-560m fp32 22.0 / bf16 22.3; bloom-1b7 fp32 14.2 / bf16 14.5 (WikiText-2, `measure_quality.py`) |
| `quality_eval.py` | 87 | Online quality: `ReferenceEvaluator` runs a local fp32 bloom-560m reference to compute NLL-of-generated, token agreement vs reference, and task match |
| `instrumentation.py` | 100 | Per-request CSV record; `estimate_activation_bytes(hidden, dtype, n_spans) = hidden · dtype_bytes · (hops+1)` |
| `bench.py` | 173 | Experiment driver: `sweep` (α 0→1 + vanilla baseline), `mixed` (concurrent α=0 and α=1 request streams via `ThreadPoolExecutor`), `lambada` (accuracy on the LAMBADA openai test split) |
| `run_client.py` / `summarize.py` | 68 / 198 | Interactive client; stdlib-only report aggregation (per-objective tables, Pareto frontier plot) |

### 3.3 Decision flow (one request)

```
generate(prompt, objective)
  1. objective → ContextVar                (client.py)
  2. level 1: ModelSelector.choose()       (selector.py)
       predicts latency/quality of each tier → picks model
  3. level 2: RequestAwareSequenceManager.make_sequence()   (policy.py)
       DHT lookup → candidate spans (node, blocks, dtype)
       inference graph with latency + quality edge weights
       cost = (1−α)·latency + α·quality_cost  → Dijkstra shortest path
       constraint objectives: bisect α until predicted path fits budget/floor
  4. generation via Petals InferenceSession (unchanged)
       failures → inherited rerouting (ban peer, re-query DHT, rebuild route)
  5. one CSV row: policy, model, objective, route, n_spans,
     decision_time_s, predicted_latency_s, ttft_s, gen_time_s, n_tokens,
     tok_s, quality_ppl, ref_nll, token_agreement, expected, task_match,
     activation_bytes_per_token_est, error
```

### 3.4 Docker images and deployment

- **Image** (`docker-swarm/Dockerfile`): petals + hivemind + `router/` baked in;
  built per-architecture (`linux/arm64` on the Mac, `linux/amd64` for the Lenovos)
  and pushed to the private registry as `192.168.2.109:5001/petals-swarm:{arm64,amd64,latest}`.
- **Compose files** (`docker-compose.{mac,t490,t14}.yml`): one service per virtual
  node with pinned block counts, dtype, CPU/memory caps, and `host_maddrs` ports;
  `.env` carries the founder maddr and LAN IP.
- **Bring-up order**: registry → T490 founder (its peer ID is recorded into `.env`)
  → remaining servers → clients. Detailed runbook in `docker-swarm/README.md`.

---

## 4. Metrics and evaluation design

| Metric | Source |
|---|---|
| TTFT, tok/s, generation time | client-side timing per request (CSV) |
| Perplexity (predicted per route) | `QualityTable` (offline WikiText-2 measurements) |
| Divergence vs FP32 | `ReferenceEvaluator`: NLL of generated text + token agreement under a local fp32 reference |
| Task accuracy | LAMBADA openai exact-match (`bench.py lambada`), per objective vs baseline |
| Activation bytes / token | `hidden_size · dtype_bytes · (hops+1)` per route |
| Router decision overhead | `decision_time_s` vs total latency per request |
| Route composition | `route` field: ordered spans (node, block range, dtype) |
| Pareto frontier | `sweep` over α ∈ {0, 0.25, 0.5, 0.75, 1} vs baselines (Petals default, always-fast, always-accurate, random) |

**Experiments**: (1) α sweep → Pareto frontier; (2) network degradation via
`tc netem` on T14/T490 containers (delay/loss injection; `NET_ADMIN` cap) and
Network Link Conditioner on the Mac; (3) mixed concurrent loads (`bench.py mixed`);
(4) fault injection mid-generation (`docker stop` a route member → observe DHT
re-routing; boundary case: kill a node whose blocks have no replica).

**Fault-injection result (measured, `experiments/fault-injection-final.log`)**:
killing a route member mid-generation produced a four-phase timeline —
(1) **detection is instant**: `ConnectionResetError` caught in the same second as
the `docker stop`; (2) **stale-DHT window (~5 min)**: the DHT kept returning the
dead node's announcement for its span, so route rebuilds re-picked it and retry
backoff escalated 1→2→4→8→16→32→60 s; (3) **re-route succeeds at the swarm level**:
once the stale record expired, the DHT rebuilt the lost span onto live replacement
replicas automatically; (4) **the in-flight session cannot migrate**: the KV cache
lives host-local (server-side `MemoryCache`), so continuing at position N on a
fresh span fails a length assertion (`AssertionError('0 and N')`) and the session
retries indefinitely — new requests recover, the interrupted one does not.

---

## 5. Operational lessons (relevant to reproducibility)

- **DHT staleness**: ungraceful server kills leave stale announcements; the client
  may route to a block the server no longer hosts ("does not serve .N"). Graceful
  restarts (`docker restart -t 30`) declare blocks offline before exit and clear it.
- **VM memory sizing**: servers (~9.6 GB of weights on the Mac) + a client that loads
  an fp32 reference model need headroom; the Docker Desktop VM was raised to 18 GB
  to avoid SIGKILL (exit 137) during `--measure_quality` runs.
- **WSL2 inbound networking**: see §2.2 — portproxy + firewall rules are required
  for LAN peers to dial into the T14.
- **Retries**: Petals' exponential retry backoff (up to 60 s) means one unreachable
  peer on the chosen route can stall a request; keep the swarm fully reachable
  before benchmarking.
- **Mid-generation failover gap**: Petals' stateful `InferenceSession` stores the KV
  cache server-side, so a session cannot transparently migrate to a new span after
  its holder dies (cold-cache length assertion). Combined with the DHT's stale-record
  window, an interrupted request is effectively lost even though the swarm re-covers
  the block. A prefix re-send on failover is the natural extension for the
  request-aware router.
