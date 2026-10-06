# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace layout

This is a **research workspace containing two separate, upstream Python projects** (not a monorepo — each has its own build config, git origin, and test suite):

- `hivemind/` — [Learning@home Hivemind](https://github.com/learning-at-home/hivemind): a PyTorch library for decentralized deep learning (DHT, decentralized averaging, Mixture-of-Experts, fault-tolerant optimizers, libp2p networking).
- `petals/` — [BigScience Petals](https://github.com/bigscience-workshop/petals): distributed inference and fine-tuning of 100B+ LLMs (Llama, Mixtral, Falcon, BLOOM), built **on top of hivemind**.

**Petals depends on a pinned hivemind git commit** (see `install_requires` in `petals/setup.cfg`), not on the local `hivemind/` checkout. Changes made in `hivemind/` are *not* automatically picked up by petals unless you `pip install -e ../hivemind` into the same environment. Keep this in mind when tracing behavior across the two.

Always run build/test/lint commands from inside the relevant project directory (`hivemind/` or `petals/`), not from the workspace root.

## hivemind

### Commands
- Dev install: `pip install -e .[dev]` (compiles `.proto` files and downloads a precompiled `p2pd` libp2p daemon binary at build time).
- Build the Go daemon from source instead of downloading: `HIVEMIND_BUILDGO=1 pip install .` (needs Go >= 1.20).
- Run all tests: `cd tests && export HIVEMIND_MEMORY_SHARING_STRATEGY=file_descriptor && pytest`
- Run one test file / test: `pytest tests/test_dht.py` / `pytest tests/test_dht.py::test_name`
- Run only p2p tests: `pytest -k "p2p" -v`
- Lint (must pass CI): `ruff check` and `ruff format --check --diff` — **ruff is pinned to `==0.11.2`** (`pyproject.toml`), line length 119. Also `codespell --skip=".git"`.
- Benchmarks (`benchmarks/`): `benchmark_dht.py`, `benchmark_averaging.py`, `benchmark_optimizer.py`, `benchmark_throughput.py`, `benchmark_tensor_compression.py`.
- CLI entrypoints (installed as console scripts): `hivemind-dht` and `hivemind-server`.

### Architecture
Layered, roughly bottom-up:
- `hivemind/proto/` — protobuf definitions; compiled to `*_pb2.py` at build time (do not edit generated files; edit `.proto` and rebuild).
- `hivemind/p2p/` — libp2p networking. `p2p_daemon_bindings/` wraps the external Go `p2pd` binary; `servicer.py` provides an RPC-over-P2P abstraction used by higher layers.
- `hivemind/dht/` — the Distributed Hash Table: `node.py` (Kademlia-style routing), `protocol.py`, `routing.py`, `storage.py`, plus record `validation.py`/`schema.py`/`crypto.py`. `dht.py` is the user-facing async-in-subprocess wrapper.
- `hivemind/averaging/` — decentralized parameter/gradient averaging via all-reduce. `matchmaking.py` forms groups over the DHT, `allreduce.py` does the reduction, `averager.py` is the entrypoint.
- `hivemind/moe/` — Decentralized Mixture-of-Experts, split into `server/` (hosts expert layers, `runtime.py` + `task_pool.py` batch requests) and `client/` (`RemoteExpert`, `RemoteMixtureOfExperts`, beam search over the DHT).
- `hivemind/optim/` — fault-tolerant training: `optimizer.py` (the main `hivemind.Optimizer`), gradient/state averagers, `progress_tracker.py`.
- `hivemind/compression/` — tensor (de)serialization and quantization applied during network transfer.

The public API is re-exported from `hivemind/__init__.py` (`DHT`, `DecentralizedAverager`, `Optimizer`, MoE classes, etc.).

## petals

### Commands
- Dev install: `pip install .[dev]` (pulls the pinned hivemind commit from git).
- Style (must pass CI): `black --check --diff` (**pinned `==22.3.0`**, line length 120) and `isort` (**pinned `==5.10.1`**, black profile, line length 120). Config in `pyproject.toml` / `setup.cfg`.
- Run a server (host model blocks): `python -m petals.cli.run_server <MODEL_NAME>` (e.g. `--num_blocks 5 --device cpu --torch_dtype float32 --initial_peers <MADDR>`).
- Run a bootstrap DHT node: `python -m petals.cli.run_dht --identity_path <id> --host_maddrs <maddr>`.
- Benchmarks (`benchmarks/`): `benchmark_inference.py`, `benchmark_forward.py`, `benchmark_training.py`.

### Testing (important — tests are integration tests against a live swarm)
`pytest tests` does **not** run standalone. The tests expect environment variables (`MODEL_NAME`, `REF_NAME`, `INITIAL_PEERS`, optionally `ADAPTER_NAME`) and a running swarm: a bootstrap DHT node plus several `run_server` processes serving overlapping block ranges. See `petals/.github/workflows/run-tests.yaml` for the exact multi-process setup (bootstrap → servers → `pytest tests`). Relevant knobs: `PETALS_MAX_RETRIES` (cap retries to surface tracebacks), `TRANSFORMERS_CACHE`. CI uses tiny models like `Maykeye/TinyLLama-v0` and `bigscience/bloom-560m` on CPU.

### Architecture
A Petals swarm splits a transformer into blocks hosted across many machines; clients route activations through a chain of servers.
- `src/petals/models/{bloom,falcon,llama,mixtral}/` — per-architecture support. Each has `config.py`, `block.py` (a single transformer block runnable in isolation), and `model.py` (the `AutoDistributedModelForCausalLM`-style wrapper). Adding a new model architecture means adding a directory here and registering it in `models/__init__.py`.
- `src/petals/server/` — the block host. `server.py` orchestrates; `block_functions.py`/`backend.py` run forward/backward on hosted blocks; `handler.py` serves RPCs; `memory_cache.py` holds the attention KV cache; `block_selection.py`/`throughput.py` decide which blocks to serve and rebalance the swarm.
- `src/petals/client/` — the inference/fine-tuning client. `routing/sequence_manager.py` discovers which peers hold which blocks (via the DHT) and builds a route; `sequential_autograd.py` implements distributed autograd across remote blocks; `inference_session.py` manages a stateful multi-step generation session; `remote_generation.py`/`lm_head.py`/`ptune.py` add generation and prompt-tuning on top.
- `src/petals/cli/` — `run_server.py`, `run_dht.py` entrypoints.
- `transformers` is **pinned to `==4.43.1`** and asserted at import in `__init__.py` — bumping it requires updating both the pin and the version assert.

## GitNexus (code intelligence)

A GitNexus MCP server and its skills (`gitnexus-exploring`, `gitnexus-debugging`, `gitnexus-impact-analysis`, `gitnexus-refactoring`, etc.) are available for querying an indexed knowledge graph of this codebase. Prefer these for cross-file "how does X work / what calls this / what breaks if I change this" questions over manual grepping. Use the `gitnexus-cli` skill to (re)index a repo after significant changes.

## Local environment (this M4 Pro Mac — verified working install)

Context: this is a course workspace (MO809A final project — reproducing Petals distributed inference at home). The Petals stack was installed and smoke-tested successfully on this machine (Apple Silicon arm64, macOS). Reuse the conda env `petals-mo809` (Python 3.11) — the base Python is 3.13, which is too new for the pinned `transformers==4.43.1` / `bitsandbytes==0.41.1`.

**Corporate proxy (critical):** the machine sits behind a Bayer MITM proxy. `curl` works (system keychain) but conda/pip/git use their own CA store and fail with `CERTIFICATE_VERIFY_FAILED`. Export the corporate CA bundle before *any* conda/pip/git network command:

```bash
export B=/Users/luishenriqueangelico/proxy-fix-bundle/allCAbundle-with-bayer.pem
export REQUESTS_CA_BUNDLE=$B SSL_CERT_FILE=$B PIP_CERT=$B CONDA_SSL_VERIFY=$B GIT_SSL_CAINFO=$B
```

**Gotchas learned during install:**
- Use `python -m pip`, **not** bare `pip` — the env has no `pip` shim on PATH, so `pip` falls through to the system 3.13 framework pip and installs into the wrong place.
- The pinned hivemind commit fails to build under `setuptools>=81` (`No module named pkg_resources`). Fix: `python -m pip install "setuptools<81" wheel "grpcio-tools>=1.33.2,<1.68"`, then install hivemind **and** petals with `--no-build-isolation`. Install the pinned hivemind first, then `./petals`. (The pinned commit ships a darwin-arm64 p2pd binary, so no Go build is needed.)
- `bitsandbytes==0.41.1` imports fine but has no CUDA backend on this hardware — run models in **float32/bfloat16 only**, no quantization. `torch.backends.mps.is_available()` is `True`, so `--device mps` is an option alongside `--device cpu`.
- Server logs vanish if you pipe a long-running `run_server`/`run_dht` through `tail` (pipe buffering never flushes). Redirect to a file instead: `python -u -m petals.cli.run_server ... > run.log 2>&1 &`, then inspect the file. Put such temp logs under `$CLAUDE_JOB_DIR/tmp`.
- Verified smoke test: `python -m petals.cli.run_server bigscience/bloom-560m --num_blocks 2 --device cpu --torch_dtype float32 --new_swarm --host_maddrs /ip4/127.0.0.1/tcp/0` boots and announces its blocks to the DHT.
