# Dockerized heterogeneous Petals swarm (Mac + T490 + T14)

11 virtual nodes across 3 physical hosts:

- **7 × `bloom-560m` fp32** (the original layout) across fast/medium/slow resource tiers
  (`cpus:`/`mem_limit:` per container) — the default/always-fast baseline substrate.
- **2 × `bloom-560m` bf16 replicas** (`mac-fast-bf16`, `t490-bf16`) overlapping the fp32
  ranges — the optimizer's per-block precision choice (distinguishable via
  `ServerInfo.torch_dtype` in the DHT, no server-side changes).
- **2 × `bloom-1b7` bf16** (`mac-1b7`, `t490-1b7`, full 24-block model, `mem_limit: 6g`) —
  the "accurate" model tier. Block ranges converge via Petals' least-loaded balancing
  (`block_selection.py`), so fp32 and bf16 replicas naturally overlap.

The route optimizer (`router/`, copied into the image) reads each request's objective and
selects per request the model tier and the fp32/bf16 replicas. See the top-level README.

All commands below must be run from each machine's own terminal (the LAN is not
reachable from a sandboxed tool shell).

## 0. One-time: local registry on T490

```bash
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```

## 1. Build + push the image (once, and again after any Dockerfile/petals change)

On **T490** (from the repo root, `/Users/luishenriqueangelico/MO809A`):
```bash
docker buildx build --platform linux/amd64 -f docker-swarm/Dockerfile \
  -t 192.168.2.109:5000/petals-swarm:amd64 . --push
```

On **Mac** (needs the corporate proxy CA as a BuildKit secret):
```bash
export B=/Users/luishenriqueangelico/proxy-fix-bundle/allCAbundle-with-bayer.pem
docker buildx build --platform linux/arm64 --secret id=cacert,src=$B \
  -f docker-swarm/Dockerfile -t 192.168.2.109:5000/petals-swarm:arm64 . --push
```

Merge into one multi-arch tag (from T490 or Mac):
```bash
docker manifest create 192.168.2.109:5000/petals-swarm:latest \
  192.168.2.109:5000/petals-swarm:amd64 192.168.2.109:5000/petals-swarm:arm64
docker manifest push 192.168.2.109:5000/petals-swarm:latest
```

T14 never builds — it only pulls `192.168.2.109:5000/petals-swarm:latest` (resolves to amd64).

## 2. One-time: trust the insecure registry on Mac + T14

- **Mac**: Docker Desktop → Settings → Docker Engine → add to the JSON:
  `"insecure-registries": ["192.168.2.109:5000"]`, then Apply & Restart.
- **T14** (Linux): `/etc/docker/daemon.json` →
  `{"insecure-registries": ["192.168.2.109:5000"]}`, then `sudo systemctl restart docker`.

## 3. Bring-up order (founder peer ID is only known after its first boot)

1. Copy `.env.t490.example` → `.env` next to `docker-compose.t490.yml`, fill `HOST_LAN_IP`.
   Leave `FOUNDER_MADDR` empty for now.
2. `docker compose -f docker-compose.t490.yml up -d t490-founder`
3. `docker compose -f docker-compose.t490.yml logs t490-founder` — copy the announced
   `/ip4/.../tcp/31337/p2p/<peer-id>` maddr.
4. Paste that maddr as `FOUNDER_MADDR` into `.env` (T490) and into the `.env` files you
   copy from `.env.mac.example` / `.env.t14.example` on the Mac and T14 (fill `HOST_LAN_IP`
   on each too).
5. `docker compose -f docker-compose.t490.yml up -d` (brings up `t490-slow` now that
   `FOUNDER_MADDR` is set).
6. On the Mac: `docker compose -f docker-compose.mac.yml up -d`.
7. On T14: `docker compose -f docker-compose.t14.yml up -d`. If its blocks never leave
   "joining" (corporate VPN/firewall blocking inbound TCP), drop it — the T490+Mac
   5-node swarm still stands on its own.
8. Confirm every container logs its own block range as online (not stuck "joining").

## 4. End-to-end test

```bash
docker compose -f docker-compose.mac.yml --profile client run --rm client
```
Prints time-to-first-token and tok/s for a short generation routed across whichever
nodes are online. Compare against the bare-metal 2-node baseline (~14.7 tok/s,
`swarm-run/`) to separate "more/smaller nodes" overhead from "containerization" overhead.

## 5. Route-optimizer client

```bash
docker compose -f docker-compose.mac.yml --profile router-client run --rm router-client
```
Runs `router/run_client.py` inside the swarm network: objective-aware generation with
CSV instrumentation (experiments CSV lands inside the container's `/app/experiments` —
add `--csv` mounted out if you want it on the host). Change the objective in the
compose command or run ad hoc:

```bash
docker compose -f docker-compose.mac.yml --profile router-client run --rm router-client \
  python router/run_client.py --initial_peers $FOUNDER_MADDR \
  --objective quality_floor=16ppl --max_new 30 --csv - 
```

## 6. Fault-tolerance smoke test

Stop any `*-slow` or `*-medium` container mid-generation (`docker stop <container>`) and
re-run the client — confirms DHT rerouting around the missing node.
