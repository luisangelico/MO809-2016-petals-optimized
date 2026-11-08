# Preparação para avaliação — MO809A (Prof. Allan Mariano de Souza)

Q&A de preparação, organizado pelo ângulo mais provável de pergunta. As respostas usam
apenas os números reais do projeto (ver `RELATORIO.md` / `REPORT.md`).

**Ângulos-chave do avaliador (do perfil Lattes + página da disciplina):**
- Tese premiada em **re-rotação de tráfego veicular multi-objetivo** → vai apertar a parte
  de multi-objetivo / fronteira de Pareto / escalarização.
- Pesquisador de **aprendizado distribuído / aprendizagem federada** → vai perguntar onde
  está o "aprendizado" num projeto de inferência.
- Disciplina cobriu: treinamento paralelo, split learning, FL centralizada e descentralizada,
  **fine-tuning federado de LLMs**, agentes de LLM → use esse vocabulário.

---

## 1. Enquadramento no curso (aprendizado distribuído)

**Q1. Seu projeto é inferência — onde entra "aprendizado distribuído"?**
O Petals foi construído sobre o Hivemind exatamente para aprender de forma distribuída:
além da inferência, ele suporta **fine-tuning distribuído** via autograd sequencial através
dos mesmos servidores. Nossa extensão vive no roteador, que é compartilhado pelos dois
caminhos — inferência e treino — então o roteamento objetivo-consciente se aplica ao
fine-tuning distribuído sem mudança. E a disciplina cobriu *fine-tuning federado de LLMs*:
o Petals é a infraestrutura desse cenário — a diferença é que o particionamento dele é
**pipeline** (camadas espalhadas entre nós), não **paralelismo de dados** como no FL clássico.

**Q2. Compare seu sistema com aprendizagem federada (centralizada e descentralizada).**
No FL os clientes têm o **modelo inteiro** e trocam gradientes/pesos (all-reduce); no Petals
cada nó tem **um fatia do modelo** e trocam **ativações** (forward/backward em cadeia). O FL
do Hivemind é descentralizado (sem servidor de parâmetros, via DHT) — a mesma base de
descoberta que nosso roteador usa. Nossos nós são heterogêneos como clientes FL, mas a
heterogeneidade aparece como *replicas em precisão/nó diferentes*, que é justamente o que
o roteamento explora.

**Q3. Split learning e pipeline parallelism — qual a diferença no seu sistema?**
Split learning corta o modelo em **um ponto** cliente–servidor (o servidor guarda os labels).
O Petals corta em **múltiplos pontos**: uma cadeia de servidores, cada um com um intervalo de
blocos — pipeline paralelismo puro. Nosso trabalho não muda o corte: decide **qual réplica de
cada intervalo** usar em cada requisição.

**Q4. A troca de ativações não é um gargalo de comunicação enorme?**
É — por isso medimos bytes de ativação estimados por requisição e por isso a rota importa:
cada salto adiciona RTT + serialização ao caminho por token. O roteador minimiza essa soma
(latência) ou troca por qualidade, por requisição.

## 2. Multi-objetivo e fronteira de Pareto

**Q5. Por que soma ponderada (α) e não ε-restrição ou otimização lexicográfica?**
Soma ponderada é a escalarização mais simples e interpretável como dial: uma única chamada
`custo = (1−α)·latência + α·custo_qualidade`. Limitação conhecida — e declarada no relatório:
soma ponderada não alcança regiões **não convexas** da fronteira. Por isso também
implementamos os objetivos **restritos** (`latency_budget`, `quality_floor`), que caminham no
peso até a rota prevista satisfazer a restrição — uma forma prática de ε-restrição sobre o
mesmo grafo.

**Q6. Vocês validaram a fronteira de Pareto?**
Com honestidade: validamos a fronteira de **custo de rota × latência**. A dimensão qualidade é
um *proxy* (perplexidade offline por (modelo, precisão) em WikiText-2); como todas as rotas
do testebed servem o mesmo modelo com decodificação gulosa determinística, as saídas são
equivalentes — a diferença aparece no escore de roteamento (ppl prevista 22.075 → 22.025),
não no texto. Validar a fronteira **end-to-end** exige alternativas de execução com diferença
observável de qualidade (réplicas de modelos distintos por requisição) — está declarado como
trabalho futuro no relatório.

**Q7. Só 5 pontos de α?**
Sim, com 3 prompts compartilhados e vanilla como linha de base (18 requisições). É varredura
de *sanidade* da mecânica, não caracterização fina da fronteira — mais amostras e grade
densa de α são trabalho futuro declarado.

**Q8. Como se escolhe α na prática?**
O nível 1 do roteador aceita três formas de objetivo declarado: α direto, `latency_budget`
(SLA de TTFT) ou `quality_floor`. Na prática, o tipo de requisição define: chat/autocomplete
→ α≈0 ou SLA; sumarização/raciocínio → α alto. É a mesma lógica de personalização de um
sistema de re-rotação: o objetivo pertence ao cliente, não ao sistema.

**Q9. Qual a analogia com re-rotação de tráfego?** *(se puxarem pelo campo dele)*
A mesma estrutura: rotas alternativas em rede dinâmica, objetivo por usuário
(multi-objetivo), e telemetria de rede **defasada** — nosso DHT atualiza RTT por ping com
janela de minutos, o equivalente à informação de congestionamento velha num sistema veicular.
O experimento 2 mostrou exatamente o custo da defasagem: latência prevista quase imóvel
(0.91 → 0.94 s) enquanto o TTFT real triplicou.

## 3. Sistemas distribuídos (fundamentos)

**Q10. Por que DHT (Kademlia) e não um tracker central?**
Sem ponto único de falha nem gargalo de descoberta; o DHT propaga anúncios de blocos e
métricas (`ServerInfo`: dtype, RPS, cache, pings) e envelhece registros — o que vimos no
experimento 4: janela de ~5 min com nó morto ainda anunciado, com backoff progressivo do
cliente (1 → 2 → 4 → … → 60 s), depois autorrecuperação do enxame.

**Q11. O que acontece com a requisição interrompida na falha?**
Não migra: o KV cache da sessão vive **no servidor** que caiu. Continuar na réplica nova
falha na checagem de comprimento (`AssertionError('0 and N')`). A extensão natural — e o
fechamento do experimento 4 — é **reenviar o prefixo** ao novo dono e reconstruir o KV,
custo O(comprimento do prefixo).

**Q12. Onde o roteamento roda? Custa quanto?**
100% no cliente, sem tocar o upstream: subclasse de `RemoteSequenceManager` espelhando o
Dijkstra do Petals com pesos separados (latência, qualidade). Overhead medido por requisição:
**0.2–1.5 ms** — desprezível contra TTFT da ordem de 1 s.

**Q13. Rotas diferentes para objetivos diferentes convivem sob carga?**
Sim — experimento 3: 8 requisições simultâneas (4 α=0 + 4 α=1), rotas diferenciadas por grupo
no mesmo instante, 0 erros. A instrumentação por requisição (ContextVar `current_route`)
elimina a corrida de ler o estado compartilhado do gerenciador.

## 4. Metodologia e avaliação

**Q14. Como mediram o overhead de decisão?**
Tempo entre receber o objetivo e devolver a rota, registrado por requisição no CSV de
instrumentação. Faixa 0.2–0.6 ms na varredura de α, 0.2–1.5 ms sob concorrência.

**Q15. O que o netem provou?**
Partição do impacto ao longo do objetivo: a rota de latência (α=0) não toca o nó degradado e
fica imune; as rotas de qualidade (α=0.5/1) atravessam e **não podem desviar** — não por
falha do roteador, mas porque os blocos têm **dono único** nesta topologia. Resiliência
adaptativa exige réplicas sobrepostas com telemetria atual — restrição declarada.

**Q16. Por que a acurácia LAMBADA é idêntica (76%) nos três cenários?**
Mesmo modelo, decodificação gulosa determinística: a rota não altera os pesos, então o texto
é o mesmo. É o resultado esperado e ele **confirma o isolamento do mecanismo**: o roteamento
muda latência (+40% TTFT no α=1) e custo de rota, não a função computada.

**Q17. Por que BLOOM-560m/1b7 e não um Llama 70B?**
Restrição de hardware declarada: sem CUDA (fp32/bf16 em CPU/MPS), 24 blocos manejáveis, e
controle científico — com um modelo pequeno e determinístico, qualquer divergência de saída
seria atribuível ao mecanismo, não a estocastidade do modelo.

**Q18. Validade estatística?**
Amostras pequenas e uma topologia — limitações declaradas. O conjunto de métricas
(TTFT, tok/s, tempo de decisão, composição de rota, NLL, acurácia) cobre custo e efeito;
intervalos de confiança e mais amostras são trabalho futuro.

### Se a acurácia não melhora, qual a vantagem de usar α=1?

Resposta em três camadas:

1. **Neste testbed (honesto):** nenhuma vantagem *observável nas saídas* — todas as rotas servem
o mesmo modelo, então a acurácia é idêntica (76% nos três cenários). O que α=1 compra aqui é a
**composição da rota**: 7 spans pelos spans fp32 do T14 em vez da rota larga bf16 — a
demonstração de que a decisão por requisição funciona. O **preço** é medido: +40% de TTFT. É a
conclusão honesta: num enxame de pesos idênticos, o dial muda latência, não qualidade.

2. **No mecanismo geral (onde a vantagem aparece):** o dial vale quando o enxame oferece
*alternativas reais de qualidade*:
   - **Réplicas quantizadas** (int8/int4): as saídas divergem de verdade — α=1 protege a
     perplexidade; a heterogeneidade de dtype já está anunciada no DHT, o roteador a explora
     sem mudança no servidor.
   - **Múltiplos modelos/tiers**: o nível 1 prefere bloom-1b7 (perplexidade 14.2 vs 22.0) —
     diferença de qualidade real e observável.
   - **Combinado com restrição**: α=1 é o extremo "zero compromisso de qualidade", que o
     orçamento de latência então equilibra.

3. **O valor é a *opção*, não a resposta:** a contribuição é o cliente poder *declarar* o
objetivo por requisição — a mesma lógica de personalização do re-roteamento veicular. Numa
topologia sem alternativas de qualidade, a declaração se torna preço medido (TTFT +40%) em vez
de ganho; o sistema expõe isso honestamente no custo da rota. A validação end-to-end da
fronteira (qualidade observável) fica como trabalho futuro com réplicas quantizadas.

## 5. Limitações (pergunta obrigatória do roteiro da disciplina)

**Q19. Qual a limitação mais importante?**
Que a dimensão qualidade é *proxy*: não demonstramos diferença mensurável de qualidade de
saída — todas as rotas serviam o mesmo modelo. O projeto valida o **mecanismo de roteamento**
(overhead, diferenciação, adaptação de latência, partição do impacto no netem, fronteira de
tolerância a falhas), e declara a validação da fronteira latência–qualidade end-to-end como
trabalho futuro com réplicas de qualidade observavelmente distinta.

**Q20. Trabalhos futuros, em ordem de impacto?**
1. Failover de sessão (reenvio de prefixo + reconstrução de KV) — fecha a lacuna do exp. 4;
2. Sondas de RTT por requisição / atualizações de DHT dirigidas a eventos — fecha a defasagem
   do exp. 2;
3. Réplicas com donos múltiplos — permite desvio real sob degradação;
4. Variantes de modelo/precisão por requisição — permite validar a fronteira de qualidade;
5. Quantização (8-bit/NF4) — exige GPU CUDA.

---

## Roteiro rápido — respostas em uma linha

| Pergunta | Resposta em uma linha |
|---|---|
| Problema? | Roteador do Petals ignora o objetivo da requisição; roteia tudo igual |
| Trabalhos relacionados — força/limite? | Petals/Hivemind dão a infraestrutura; ninguém faz seleção por requisição de configuração de execução sob objetivo explícito |
| Objetivo? | Roteamento objetivo-consciente por requisição no cliente, upstream intacto |
| Solução? | Subclasse do gerenciador de sequência com grafo Dijkstra de dois pesos (latência, qualidade) + tabela de qualidade offline + ContextVar |
| Processo? | Pacote `router/` separado, 5 experimentos em enxame real de 3 hosts heterogêneo |
| Avaliação/métricas? | TTFT, tok/s, overhead de decisão, composição de rota, NLL vs fp32, acurácia LAMBADA |
| Resultados quantitativos? | Overhead 0.2–1.5 ms; rotas diferenciadas sob concorrência (0 erros); +220% TTFT no α=0.5 sob netem com α=0 imune; LAMBADA 76% igual nos 3 cenários |
| Qualitativos? | Impacto se parte ao longo do objetivo; defasagem da telemetria do DHT é o limite de reatividade |
| Limitações? | Qualidade é proxy (mesmo modelo em todas as rotas); donos únicos impedem desvio; amostras pequenas |
| Próximos passos? | Failover de sessão, sondas de RTT, réplicas múltiplas, validação end-to-end da fronteira |
