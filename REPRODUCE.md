# Reproducibility guide — request-aware route optimizer on a Docker-simulated heterogeneous Petals swarm

This document is the per-PC protocol for reproducing the MO809A experiment end to end.
Every number in the report comes from CSV rows produced by `router/` instrumentation and
aggregated by `router/summarize.py` — nothing is transcribed by hand.

**Testbed:** 3 physical hosts; Docker containers simulate the heterogeneous computers:

| Host | Containers (service) | Role |
|---|---|---|
| Lenovo T490 (Ubuntu 24.04, x86, 24 GB) | `t490-founder`, `t490-slow`, `t490-bf16`, `t490-1b7` | swarm founder + slow/medium tiers |
| MacBook M4 Pro (arm64, macOS) | `mac-fast`, `mac-medium`, `mac-slow`, `mac-fast-bf16`, `mac-1b7` | fast tier + client runs |
| Lenovo T14 (corporate, best-effort) | `t14-medium`, `t14-slow` | optional bonus; drop if blocked |

Quality axes the optimizer routes over: model tier (`bloom-560m` fast vs `bloom-1b7`
accurate) and precision (fp32 vs bf16 replicas), distinguishable by the client via
`ServerInfo.torch_dtype` in the DHT.

---

## 0. Common prerequisites (every PC)

- Clone this repo. The upstream `petals/` and `hivemind/` clones are copied into the
  Docker image at build time (`docker-swarm/Dockerfile`), so they must exist in the
  working tree before building.
- Export on **every** HF-fetching process and shell:
  ```bash
  export HF_HUB_DISABLE_XET=1   # corporate proxy 403s the HF Xet CDN
  ```
- Models used: `bigscience/bloom-560m`, `bigscience/bloom-1b7` (CPU, fp32/bf16 only —
  no CUDA on any node, so bitsandbytes quantization is unavailable).
- Freeze the software environment **once per host** for the record:
  ```bash
  pip freeze > env-<hostname>.txt   # snapshot committed with the final results
  ```

## 1. Lenovo T490 (swarm founder, image builder)

No corporate proxy on this host — standard pip paths work.

1. **Local Docker registry** (one-time, keeps running):
   ```bash
   docker run -d -p 5000:5000 --restart=always --name registry registry:2
   ```
2. **Build + push the amd64 image** (from the repo root; repeat after any code change):
   ```bash
   docker buildx build --platform linux/amd64 -f docker-swarm/Dockerfile \
     -t 192.168.2.109:5000/petals-swarm:amd64 . --push
   ```
3. **Bring up the T490 services** (order in §4):
   ```bash
   cp docker-swarm/.env.t490.example docker-swarm/.env   # fill HOST_LAN_IP; FOUNDER_MADDR later
   docker compose -f docker-swarm/docker-compose.t490.yml up -d t490-founder
   docker compose -f docker-swarm/docker-compose.t490.yml logs t490-founder   # copy the announced maddr
   # paste it as FOUNDER_MADDR into .env, then:
   docker compose -f docker-swarm/docker-compose.t490.yml up -d
   ```
4. **Network-conditioning experiments** (experiment 2): inject latency/loss on a
   container's interface with `tc netem`, e.g.
   ```bash
   tc qdisc add dev eth0 root netem delay 50ms 10ms loss 1%    # inside the container
   tc qdisc del dev eth0 root                                   # restore
   ```

## 2. MacBook M4 Pro (fast tier + measurement client)

Behind the Bayer MITM proxy — conda/pip/git/Docker-build need the corporate CA bundle:

```bash
export B=/Users/luishenriqueangelico/proxy-fix-bundle/allCAbundle-with-bayer.pem
export REQUESTS_CA_BUNDLE=$B SSL_CERT_FILE=$B PIP_CERT=$B GIT_SSL_CAINFO=$B
```

1. **Conda env** (bare-metal runs: quality measurement, local reference model):
   ```bash
   conda create -n petals-mo809 python=3.11 -y && conda activate petals-mo809
   python -m pip install "setuptools<81" wheel "grpcio-tools>=1.33.2,<1.68"
   python -m pip install --no-build-isolation \
     "git+https://github.com/learning-at-home/hivemind.git@213bff98a62accb91f254e2afdccbf1d69ebdea9"
   python -m pip install --no-build-isolation ./petals ./router 2>/dev/null || true
   python -m pip install datasets matplotlib   # for measure_quality.py / summarize.py --plot
   ```
2. **Build + push the arm64 image** (CA as a BuildKit secret):
   ```bash
   docker buildx build --platform linux/arm64 --secret id=cacert,src=$B \
     -f docker-swarm/Dockerfile -t 192.168.2.109:5000/petals-swarm:arm64 . --push
   ```
3. **Merge the multi-arch manifest** (either host):
   ```bash
   docker manifest create 192.168.2.109:5000/petals-swarm:latest \
     192.168.2.109:5000/petals-swarm:amd64 192.168.2.109:5000/petals-swarm:arm64
   docker manifest push 192.168.2.109:5000/petals-swarm:latest
   ```
4. **Trust the insecure registry** (one-time): Docker Desktop → Settings → Docker
   Engine → `"insecure-registries": ["192.168.2.109:5000"]` → Apply & Restart.
5. **Bring up the Mac services** (after `FOUNDER_MADDR` is known; §4).
6. **Network Link Conditioner** (macOS) for the Mac-side degradation runs.

## 3. Lenovo T14 (optional, best-effort)

Corporate VPN/firewall may block inbound libp2p TCP. If a container's blocks never
leave "joining", drop this host — the T490+Mac swarm stands alone.

```bash
# /etc/docker/daemon.json → {"insecure-registries": ["192.168.2.109:5000"]}; restart docker
docker compose -f docker-swarm/docker-compose.t14.yml up -d
```

## 4. Swarm bring-up order (any restart)

1. `t490-founder` first (`--new_swarm`) → read its announced
   `/ip4/<LAN_IP>/tcp/31337/p2p/<peer-id>` maddr from the logs.
2. Set `FOUNDER_MADDR` in every host's `.env`, then bring up everything else:
   T490 → Mac → T14 (if usable).
3. Verify: every container logs its block range **online** (never stuck "joining").
   A server stuck joining usually means the Xet CDN got hit — check
   `HF_HUB_DISABLE_XET=1` is set in the container env (it is, in the image).

## 5. Measurement protocol (scientific procedure)

Run these **in this order**; each writes CSV rows under `experiments/`. All commands
run from the repo root (bare-metal on the Mac, or via the compose profiles). Fixed
prompt set = `router/bench.py:PROMPTS` + LAMBADA examples; fix the number of repeats
before starting and do not change it between configurations.

1. **Quality table** (one-time, no swarm needed, on the Mac):
   ```bash
   conda activate petals-mo809 && export HF_HUB_DISABLE_XET=1
   python router/measure_quality.py --models bigscience/bloom-560m bigscience/bloom-1b7
   ```
   This regenerates `router/quality_scores.json` (validation perplexity per model×dtype)
   — the routing decisions depend on it, so it must precede every run.
2. **Pareto sweep + true Petals baseline** (experiment 1 + baseline):
   ```bash
   python router/bench.py --initial_peers $FOUNDER_MADDR sweep \
     --alphas 0 0.25 0.5 0.75 1 --baselines --measure_quality
   ```
   `- -baselines` runs the vanilla `RemoteSequenceManager` (true Petals default, single
   fast tier) alongside the optimizer presets; `--measure_quality` adds NLL-under-fp32
   and token agreement per request. Repeats: run the sweep 3× (or `--n_baseline_repeats 3`).
3. **Task accuracy** (LAMBADA exact-match per objective vs baseline):
   ```bash
   python router/bench.py --initial_peers $FOUNDER_MADDR lambada --n_examples 50 --baseline
   ```
4. **Mixed concurrent load** (latency-sensitive + quality-sensitive at once):
   ```bash
   python router/bench.py --initial_peers $FOUNDER_MADDR mixed --n_lat 4 --n_qual 4
   ```
5. **Network degradation** (experiment 2): apply `tc netem` (T490 containers) / Network
   Link Conditioner (Mac host), re-run step 2 for each condition (e.g. 25/50/100 ms),
   then remove the conditioning.
6. **Fault tolerance** (experiment 4): start a generation, `docker stop` one replica
   container mid-run; the client's inherited Petals rerouting recovers — recovery time
   is captured in the same CSV (`error` column + subsequent successful rows).
7. **Aggregation** (on the Mac, after collecting the CSVs from all hosts):
   ```bash
   python router/summarize.py experiments/sweep-*.csv --out results/sweep-summary.md
   python router/summarize.py experiments/lambada-*.csv --out results/lambada-summary.md
   python router/summarize.py experiments/sweep-*.csv --plot results/frontier.png
   ```

## 6. What each metric is

| Metric | CSV column | Meaning |
|---|---|---|
| Throughput | `tok_s` | generated tokens / s (the report's latency axis, inverted) |
| Time-to-first-token | `ttft_s` | includes route discovery |
| Optimizer overhead | `decision_time_s` | wall-clock inside `make_sequence` (NaN for the vanilla baseline — its decision cost is inside `gen_time_s`) |
| Quality (predicted) | `quality_ppl` | route perplexity from the quality table — what the optimizer optimized |
| Quality (measured) | `ref_nll_per_token`, `token_agreement` | divergence of the actual generation vs the local fp32 reference |
| Task accuracy | `task_match` | LAMBADA exact-match |
| Communication | `activation_bytes_per_token_est`, `route`, `n_spans` | per-token activation payload + route composition |
| Faults | `error` | failed requests (fault-injection windows) |

## 7. Reproducibility checklist

- [ ] `env-<hostname>.txt` (pip freeze) snapshot from every host, committed with results
- [ ] `router/quality_scores.json` regenerated (not the placeholder priors)
- [ ] Fixed prompt set and repeats count, stated in the report
- [ ] All CSVs under `experiments/` kept with their timestamps; summaries + plots under `results/`
- [ ] Swarm topology recorded (which containers online, block ranges) at run time
- [ ] Network-conditioning parameters recorded for each degraded run
- [ ] Docker image digest (`docker images --digests`) recorded per build
