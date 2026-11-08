# GLOSSARY / GLOSSÁRIO — Nectar (MO809A)

Glossário bilíngue dos termos usados no projeto. / Bilingual glossary of the terms used in this project.

---

## Qualidade do modelo / Model quality

### Perplexidade / Perplexity
**PT.** O quanto o modelo fica "surpreso" ao ler texto — exponencial da NLL média por token.
Perplexidade **baixa** = o modelo prevê texto natural melhor = qualidade maior. Neste projeto é
medida **offline**: cada combinação (modelo, precisão) lê documentos de validação do WikiText-2.
Exemplo (`router/quality_scores.json`): bloom-560m fica em 22.0 em fp32 e 22.3 em bf16; o
bloom-1b7 maior fica em 14.2 — mais parâmetros, menos surpresa. É o **score de qualidade**
que alimenta o custo da rota (um *proxy*: mede a qualidade potencial das réplicas, não a
diferença nas saídas — todas as rotas servem o mesmo modelo).

**EN.** How "surprised" the model is when reading text — the exponential of mean NLL per token.
**Low** perplexity = the model predicts natural text better = higher quality. Here it is measured
**offline**: each (model, precision) combination reads WikiText-2 validation documents. Example
(`router/quality_scores.json`): bloom-560m scores 22.0 in fp32 and 22.3 in bf16; the larger
bloom-1b7 scores 14.2 — more parameters, less surprise. It is the **quality score** behind the
route cost (a *proxy*: it measures the potential quality of the replicas, not differences in the
outputs — every route serves the same model).

### NLL por token / NLL per token
**PT.** Negative Log-Likelihood: quanto menor, mais próximo o texto gerado está da distribuição da
referência fp32. Medido por requisição comparando o texto gerado com um modelo de referência local
(bloom-560m fp32 íntegro no cliente). Perplexidade = exp(NLL média).
**EN.** Negative Log-Likelihood: the lower, the closer the generated text is to the FP32
reference's distribution. Measured per request against a local reference model (a pristine
bloom-560m fp32 on the client). Perplexity = exp(mean NLL).

### Proxy de qualidade / Quality proxy
**PT.** A perplexidade offline atua como substituta da qualidade: o roteador *cobre* a qualidade
potencial de cada réplica, mas não mede diferenças nas saídas (mesmo modelo em todas as rotas →
acurácia idêntica, como o experimento 5 mostra).
**EN.** Offline perplexity acts as a stand-in for quality: the router *charges* the potential
quality of each replica but does not measure output differences (same model on every route →
identical accuracy, as experiment 5 shows).

## Latência e desempenho / Latency and performance

### TTFT (Time To First Token)
**PT.** Tempo até o primeiro token aparecer — a métrica de latência que o usuário percebe
primeiro. Inclui descoberta de rota + forward pelos blocos.
**EN.** Time to the first token appearing — the latency metric the user notices first. Includes
route discovery + forward through the blocks.

### Tokens/s (throughput)
**PT.** Velocidade de geração depois do primeiro token.
**EN.** Generation speed after the first token.

## Roteamento / Routing

### Bloco / Block
**PT.** Fatia do transformer (uma camada ou mais) hospedada num servidor. O bloom-560m tem 24
blocos; gerar um token exige percorrer todos em ordem.
**EN.** A slice of the transformer (one or more layers) hosted on a server. bloom-560m has 24
blocks; generating a token requires traversing all of them in order.

### Span
**PT.** Sequência de blocos consecutivos no mesmo par (ex.: `8:17@…RXK8cG` = blocos 8–16 num
servidor). Uma rota é uma cadeia de spans.
**EN.** A run of consecutive blocks on the same peer (e.g. `8:17@…RXK8cG` = blocks 8–16 on one
server). A route is a chain of spans.

### Réplica / Replica
**PT.** Cópia de um bloco (ou intervalo) hospedada por um servidor, com uma precisão (fp32/bf16)
anunciada no DHT.
**EN.** A copy of a block (or block range) hosted by a server, with a precision (fp32/bf16)
announced to the DHT.

### α (peso do objetivo) / α (objective weight)
**PT.** O dial latência↔qualidade: α=0 roteia só pela latência; α=1 só pela qualidade; valores
intermediários combinam os dois. Viaja por requisição num `ContextVar`.
**EN.** The latency↔quality dial: α=0 routes purely by latency; α=1 purely by quality; values in
between combine them. Travels per request in a `ContextVar`.

### Escalarização / Scalarization
**PT.** Converter dois objetivos (latência, qualidade) em um número só:
`custo = (1 − α)·latência + α·custo_qualidade`. Limitação conhecida: regiões não convexas da
fronteira ficam inalcançáveis — por isso os objetivos com restrição (orçamento/piso) existem.
**EN.** Turning two objectives (latency, quality) into a single number:
`cost = (1 − α)·latency + α·quality_cost`. Known limitation: non-convex frontier regions are
unreachable — which is why the constrained objectives (budget/floor) exist.

### Orçamento de latência / Latency budget (SLA)
**PT.** Objetivo com restrição: "a melhor qualidade que ainda responde dentro de X segundos". O
roteador começa em α=1 e divide α por 2 até a latência prevista caber.
**EN.** A constrained objective: "the best quality that still answers within X seconds". The
router starts at α=1 and halves α until the predicted latency fits.

### Piso de qualidade / Quality floor
**PT.** O inverso: "a rota mais rápida cuja perplexidade prevista não passa de Y". Começa em α=0 e
caminha α para cima.
**EN.** The reverse: "the fastest route whose predicted perplexity stays under Y". Starts at α=0
and walks α up.

### Fronteira de Pareto / Pareto frontier
**PT.** Neste projeto: o conjunto de rotas em que melhorar latência só é possível piorando a
qualidade (e vice-versa). O α percorre essa fronteira. Medida aqui como trade-off
custo-de-rota × latência, não saídas.
**EN.** Here: the set of routes where improving latency is only possible by worsening quality
(and vice versa). α walks this frontier. Measured as a route-cost × latency trade-off, not
outputs.

## Infraestrutura / Infrastructure

### DHT (Distributed Hash Table)
**PT.** Tabela hash distribuída (Kademlia, XOR métrica) do Hivemind: cada servidor anuncia quais
blocos hospeda, com que precisão, throughput e estado. É o "quadro de avisos" que o roteador lê.
**EN.** Hivemind's distributed hash table (Kademlia, XOR metric): each server announces which
blocks it hosts, with which precision, throughput and state. It is the "bulletin board" the
router reads.

### Waggle dance (metáfora)
**PT.** Os anúncios do DHT — como a dança das abelhas que aponta onde está o néctar. Nectar (o
roteador) lê o waggle para escolher a rota por requisição.
**EN.** The DHT announcements — like the bees' dance that points to where the nectar is. Nectar
(the router) reads the waggle to pick the per-request route.

### Nectar
**PT.** Nome do roteador objetivo-consciente deste projeto: lê o *waggle* (DHT) e escolhe, por
requisição, a flor mais próxima (latência) ou o néctar mais rico (qualidade).
**EN.** This project's objective-aware router: reads the *waggle* (the DHT) to pick, per request,
the nearest bloom (latency) or the richest nectar (quality).

### KV cache
**PT.** Memória de atenção da geração, viva **no servidor**: por isso uma requisição em andamento
não migra se o nó da rota cai (experimento 4 — `AssertionError`).
**EN.** The generation's attention memory, alive **on the server**: which is why an in-flight
request cannot migrate when a route node dies (experiment 4 — `AssertionError`).

### Registro DHT obsoleto / Stale DHT record
**PT.** Depois que um servidor morre, o DHT continua anunciando-o por ~5 min (TTL). Rotas
reconstruídas nesse intervalo continuam escolhendo o nó morto.
**EN.** After a server dies, the DHT keeps advertising it for ~5 min (TTL). Routes rebuilt in that
window keep picking the dead node.

### Backoff
**PT.** Recuo progressivo do cliente entre tentativas (1 → 2 → 4 → … → 60 s) quando o nó escolhido
falha.
**EN.** The client's progressive pause between retries (1 → 2 → 4 → … → 60 s) when the chosen
node fails.
