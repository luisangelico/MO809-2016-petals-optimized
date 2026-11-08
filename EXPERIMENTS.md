# The Experiments — explained in plain terms

Our project splits a language model across several computers in a network, like a
conveyor belt where each machine does part of the work. A router decides, for every
question asked, *which machines* will do the work. But there is always a choice to
make: **fast answers with small models, or better answers with bigger models**.
These experiments measure that choice from every angle.

---

## The setup in one paragraph

We run two versions of the BLOOM language model, split into 24 pieces ("blocks")
each, spread across 12 containers on 3 computers:

- **bloom-560m** — the small, fast model (like a quick junior assistant)
- **bloom-1b7** — the bigger, smarter model (about 3× the size — like a senior specialist)

Some blocks exist in **two qualities**: full-precision copies (more accurate, heavier)
and half-precision copies (lighter and faster, slightly blurrier answers). The router
picks a path through the machines for *each request*, and we can tell it how to
decide — prioritize **speed**, prioritize **quality**, or anywhere in between
(a dial called **α**, from 0 = "speed only" to 1 = "quality only").

---

## Data used for testing

| Data | What it is | Where it's used |
|---|---|---|
| **3 everyday prompts** | Short fill-in-the-blank sentences: "The capital of France is", "Water boils at a temperature of", "The largest planet in the solar system is" | Main benchmarks (experiments 1 and 3) — every prompt is sent 20 tokens deep, timed and logged |
| **WikiText-2** | A public collection of Wikipedia articles | Offline quality scoring — we measure how *perplexed* each model/precision copy is when reading natural text (lower = better). These scores live in a lookup table the router consults |
| **LAMBADA** | A public dataset of passages where you must guess the **last word** — requires reading the whole paragraph, not just pattern-matching | Accuracy test (experiment 5): did the model produce the correct final word? |
| **A local reference model** | A pristine full-precision copy of bloom-560m kept on the client machine | The "answer key" — every generated answer is compared against it to measure how far a fast/blurry route drifted from the ideal |

---

## Experiment 1 — The speed vs. quality dial (α sweep)

**Question:** when we move the dial from "fast" (α=0) to "quality" (α=1), what do we
gain and what do we pay?

**How it works:** we send the same prompts five times, each time asking the router
to care a little more about quality: α = 0, 0.25, 0.5, 0.75, 1. For each level we
record how long the first word took to appear, the words-per-second rate, and how
good the answer was (perplexity + agreement with the reference model).

**What we measure:** the **route cost across α** — the weighted combination of
measured latency and the quality *proxy* (predicted perplexity). Whether proxy
differences yield observable output-quality differences is assessed separately
(experiment 5); the router's curve is compared against four naive strategies:
Petals' default routing, always the fastest option, always the most accurate one,
and random picking. If our router's curve sits *above/left* of the naive ones,
it wins.

## Experiment 2 — When the network gets worse (degradation)

**Question:** what happens to the smart routes when the pipes slow down?

**How it works:** we deliberately degrade the network on one machine — using the
standard Linux traffic-shaping tool (`tc netem`) on the Lenovo computers, and the
Network Link Conditioner on the Mac — adding artificial delays (e.g., +150 ms) and
lost packets (e.g., 2%) to simulate a bad Wi-Fi day. Then we rerun the same requests
and compare: did the router reroute around the degraded machine? How much did
latency and route composition change?

**What we found (measured, `netem-before.csv` vs `netem-after.csv`, +150 ms ±50 ms
and 2% loss injected on the remote machine's medium node):** the impact splits
exactly along the objective. The **latency route (α=0) does not touch the degraded
node and was completely unaffected** — TTFT 0.58 → 0.31 s, 4.43 → 4.70 tok/s, same
5-span composition. The **quality routes (α=0.5, α=1) went straight through it** —
because the blocks it holds have single holders in this topology, there is no
alternative full-precision path, and TTFT went 0.51 → 1.63 s (α=0.5, +220%) and
0.43 → 1.56 s (α=1), throughput dropping ~50% (3.42 → 1.49 and 2.93 → 1.60 tok/s).
The route composition stayed identical in all six runs. The **predicted-latency
model barely reacted** (0.91 → 0.94 s) while actual TTFT tripled: the DHT
ping-based RTT table refreshes too slowly to capture a sudden injected delay.
Bottom line: differentiated routes absorb network damage when the degraded node
isn't needed, cannot avoid it when it's the sole holder of required blocks, and
the cost model underestimates fast-changing latency — pointing at the natural
extension: on-demand per-request RTT probes (or event-driven DHT updates) feeding
the route decision.

## Experiment 3 — Everyone asks at once (mixed concurrent loads)

**Question:** does the router still make good decisions when it must serve
impatient and patient users **simultaneously**?

**How it works:** one test sends **8 requests at the same time** — 4 marked
"speed only" (α=0) and 4 marked "quality only" (α=1). A well-designed router should
send the impatient ones down the fast half-precision path and the patient ones up
the accurate path, *at the same time*, without the two groups destroying each
other's performance. We log every request and check each group got the treatment
it asked for.

## Experiment 4 — A machine dies mid-sentence (fault tolerance)

**Question:** if one of the computers crashes while it's generating an answer, does
the whole thing fall apart?

**How it works:** we start generating an answer and, halfway through, **shut one of
the containers down** (`docker stop`). The system should notice the failure, ask the
directory (DHT) which other machines hold the same block, rebuild the route, and
continue — the user only sees a hiccup. We measure the hiccup: extra time added,
and whether the answer completed. We also test the boundary: kill a machine holding
a block that *no one else has* — there the answer must fail, and we verify the
system recovers cleanly once the machine comes back.

**What we found (measured, `fault-injection-final.log`):** the story splits into
four phases. (1) The failure is **detected instantly** — the connection error lands
in the same second as the shutdown. (2) For about **5 minutes** the directory (DHT)
keeps advertising the dead machine, so route rebuilds keep picking it and the client
backs off progressively (1 → 2 → 4 → … → 60 s). (3) Once the stale record expires,
the swarm **re-covers the lost block automatically** — the DHT hands out live
replacement replicas. (4) But the **interrupted answer cannot migrate**: the
conversation memory (KV cache) lives on the machine that died, so continuing on a
fresh machine fails a length check and the request never completes — while every
*new* request works fine on the repaired topology. Bottom line: **the swarm
survives, the in-flight request does not** — which points straight at the natural
extension: re-sending the prefix to the new holder on failover.

## Experiment 5 — Is the smart route actually smarter? (LAMBADA accuracy)

**Question:** do the quality-oriented routes actually produce more *correct* answers,
not just better-looking ones?

**How it works:** 50 LAMBADA passages. The model reads the passage with the last
word hidden and must produce it; the answer is a simple right/wrong. We run the
same 50 questions through the "speed" setting and the "quality" setting (plus the
default Petals router as a control) and compare accuracy. Meanwhile, the reference
model also scores *how far* each answer drifted in probability terms, even when
the exact word was right.

**What we found (measured, `lambada-20261108-023638.csv`, 150 requests, 0 errors):**
routing changes **latency, not exact-match accuracy** on this task. The
quality-oriented route (α=1) detours through the two fp32 spans hosted on the
remote Windows machine (7 spans instead of 5) and pays **+40% TTFT** (1.35 s vs
0.96 s) and **−24% throughput** (2.02 vs 2.65 tok/s) — but accuracy is identical
across all three settings (38/50 = 76% by the per-request `task_match` metric;
17/50 = 34% by the stricter stdout prefix-match, same for all). That is the
expected outcome for a fixed model: every route serves the same weights (greedy
decoding is deterministic), so the quality dimension of the trade-off shows up in
the **route cost and latency**, not in right/wrong — the α=1 route buys *potential*
quality (full-precision blocks) at a measured latency premium. The router's
decision overhead stayed negligible throughout (0.37–0.44 ms per request).

---

## How it's all recorded

Every single request writes one row to a spreadsheet (CSV): when it started, which
route it took (which machines, which blocks, which precision), how long the router
thought about it, first-word latency, speed, quality scores, estimated bytes moved
through the network, and any errors. A summary script then turns those rows into
the report tables and the Pareto plot. This is the raw evidence behind every
conclusion in the report.
