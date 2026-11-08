# -*- coding: utf-8 -*-
"""Gera a apresentacao final (MO809A) — estrutura SBC, portugues.

Estende o estilo de gen_slides.py (python-pptx, 16:9). Resultados de cada
experimento ficam no bloco RESULTADOS abaixo — atualizar ali e rodar de novo:
    python apresentacao/gen_slides_final.py
"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

AZUL = RGBColor(0x1F, 0x3A, 0x5F)
AZUL_CLARO = RGBColor(0xCF, 0xDA, 0xE8)
CINZA = RGBColor(0x44, 0x44, 0x44)
CLARO = RGBColor(0xF2, 0xF4, 0xF7)
BRANCO = RGBColor(0xFF, 0xFF, 0xFF)
VERDE = RGBColor(0x2E, 0x7D, 0x32)
VERMELHO = RGBColor(0xC6, 0x28, 0x28)

# ---------------------------------------------------------------- RESULTADOS
# Atualizar apos rodar os experimentos restantes (2: netem, 3: mixed, 5: LAMBADA)
R_EXP1 = [
    # (alpha, n_spans, tok/s, TTFT s, ppl prevista, bytes/token)
    ("0.00", "5", "4,2 ± 0,2", "0,74", "22,075", "24 576"),
    ("0.25", "5", "4,2 ± 0,3", "0,79", "22,075", "24 576"),
    ("0.50", "5", "4,2 ± 0,2", "0,95", "22,075", "24 576"),
    ("0.75", "5", "4,3 ± 0,4", "1,07", "22,075", "24 576"),
    ("1.00", "6", "3,9 ± 0,2", "0,75", "22,025", "28 672"),
]
R_BASELINE = "~4,2 tok/s · TTFT ~0,50 s · rota padrão do Petals (uma república, sem cota de qualidade)"  # TODO exp1 vanilla
R_EXP2 = ("Netem +150 ms ±50 ms, 2% de perda no nó médio do T14 (antes/depois, 3α cada): rota de latência "
          "(α=0) não toca o nó degradado — inalterada (TTFT 0,58→0,31 s; 4,4→4,7 tok/s). Rotas de qualidade "
          "(α=0,5/1) passam por ele e não podem desviar (blocos com dono único): TTFT 0,51→1,63 s (+220%) e "
          "0,43→1,56 s; tok/s cai ~50%. Latência prevista quase não reagiu (0,91→0,94 s): RTT do DHT atualiza "
          "lento demais — extensão natural: sondas por request.")  # netem-before/after.csv
R_EXP3 = ("Carga mista concorrente (4α=0 + 4α=1 simultâneas, CSV limpo): rotas diferenciadas por grupo "
          "(α=0 → span largo bf16; α=1 → passa pelo span fp32 do T14), TTFT 0,6–1,6 s por request, "
          "overhead de decisão 0,2–1,5 ms por request (desprezível), 0 erros.")
R_EXP5 = ("Acurácia LAMBADA (50 exemplos, 0 erros): idêntica nos três cenários — 38/50 (76%) para α=0, "
          "α=1 e vanilla. Rota de qualidade (α=1, 7 spans via T14 fp32) paga +40% TTFT (1,35 s vs 0,96 s) "
          "e −24% tok/s; com modelo fixo e decodificação gulosa, o trade-off aparece em latência, não em "
          "acerto. Overhead de decisão 0,37–0,44 ms.")  # lambada-20261108-023638.csv
# ---------------------------------------------------------------------------

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def faixa_titulo(slide, texto):
    barra = slide.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.15))
    barra.fill.solid()
    barra.fill.fore_color.rgb = AZUL
    barra.line.fill.background()
    barra.shadow.inherit = False
    tf = barra.text_frame
    tf.margin_left = Inches(0.5)
    tf.margin_right = Inches(0.5)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = texto
    r.font.size = Pt(28)
    r.font.bold = True
    r.font.color.rgb = BRANCO


def slide_conteudo(titulo, itens):
    """itens: lista de (nivel, texto) ou (nivel, texto, cor)."""
    s = prs.slides.add_slide(BLANK)
    faixa_titulo(s, titulo)
    cx = s.shapes.add_textbox(Inches(0.7), Inches(1.5), Inches(12.0), Inches(5.6))
    tf = cx.text_frame
    tf.word_wrap = True
    primeiro = True
    for item in itens:
        nivel, texto = item[0], item[1]
        cor = item[2] if len(item) > 2 else (CINZA if nivel else AZUL)
        p = tf.paragraphs[0] if primeiro else tf.add_paragraph()
        primeiro = False
        p.level = nivel
        p.space_after = Pt(8)
        marcador = "–" if nivel else "•"
        r = p.add_run()
        r.text = f"{marcador} {texto}" if texto else marcador
        r.font.size = Pt(19) if nivel == 0 else Pt(15)
        r.font.color.rgb = cor
        if nivel == 0:
            r.font.bold = True
    return s


def slide_tabela(titulo, cab, linhas, nota=None, fonte_size=13):
    s = prs.slides.add_slide(BLANK)
    faixa_titulo(s, titulo)
    n_lin = len(linhas) + 1
    tabela = s.shapes.add_table(n_lin, len(cab), Inches(0.7), Inches(1.5), Inches(12.0), Inches(0.4) * n_lin).table
    for j, texto in enumerate(cab):
        cel = tabela.cell(0, j)
        cel.text = texto
        cel.fill.solid()
        cel.fill.fore_color.rgb = AZUL
        p = cel.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.size = Pt(fonte_size)
            r.font.bold = True
            r.font.color.rgb = BRANCO
    for i, linha in enumerate(linhas, start=1):
        for j, texto in enumerate(linha):
            cel = tabela.cell(i, j)
            cel.text = str(texto)
            cel.fill.solid()
            cel.fill.fore_color.rgb = CLARO if i % 2 else BRANCO
            p = cel.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(fonte_size)
                r.font.color.rgb = CINZA
    if nota:
        cx = s.shapes.add_textbox(Inches(0.7), Inches(1.6) + Inches(0.42) * n_lin, Inches(12.0), Inches(0.8))
        tf = cx.text_frame
        tf.word_wrap = True
        r = tf.paragraphs[0].add_run()
        r.text = nota
        r.font.size = Pt(13)
        r.font.italic = True
        r.font.color.rgb = CINZA
    return s


# ------------------------------------------------------------------ Título
s = prs.slides.add_slide(BLANK)
fundo = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
fundo.fill.solid()
fundo.fill.fore_color.rgb = AZUL
fundo.line.fill.background()
fundo.shadow.inherit = False
cx = s.shapes.add_textbox(Inches(0.9), Inches(2.0), Inches(11.5), Inches(3.5))
tf = cx.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run()
r.text = "Nectar: Seleção Objetivo-Consciente de Réplicas no Petals"
r.font.size = Pt(36)
r.font.bold = True
r.font.color.rgb = BRANCO
p2 = tf.add_paragraph()
p2.space_before = Pt(14)
r2 = p2.add_run()
r2.text = "Inferência distribuída de LLMs — extensão do Petals em rede doméstica heterogênea"
r2.font.size = Pt(22)
r2.font.color.rgb = AZUL_CLARO
p3 = tf.add_paragraph()
p3.space_before = Pt(28)
r3 = p3.add_run()
r3.text = "MO809A — Tópicos em Sistemas de Computação · 2026"
r3.font.size = Pt(16)
r3.font.color.rgb = AZUL_CLARO

# -------------------------------------------------------------- Introdução
slide_conteudo(
    "Introdução",
    [
        (0, "Tema"),
        (1, "Inferência distribuída de LLMs: um modelo grande é fatiado em blocos hospedados por máquinas diferentes, conectadas em rede (Petals)."),
        (0, "Problema"),
        (1, "O roteamento original do Petals escolhe réplicas por throughput declarado, ignorando o que cada requisição quer: latência baixa ou qualidade alta."),
        (1, "Nodos heterogêneos (potência de computação, precisão fp32/bf16, posição na rede) tornam a escolha de rota uma decisão de verdade."),
        (0, "Motivação"),
        (1, "Executar LLMs em hardware doméstico é viável (estudo de caso real: Mac + 2 notebooks em LAN), mas exige decidir bem para onde mandar cada ativação."),
        (0, "Objetivo"),
        (1, "Estender o cliente do Petals com um roteador ciente da requisição: cada pedido declara um objetivo (α: latência ↔ qualidade) e a rota é otimizada para ele."),
        (1, "Avaliar a seleção de rota por requisição (latência e um proxy de qualidade) contra o roteamento padrão e linhas de base."),
    ],
)

# ---------------------------------------------------------------- Background
slide_conteudo(
    "Background — Conceitos Teóricos",
    [
        (0, "Inferência distribuída de transformers"),
        (1, "Camadas do modelo distribuídas entre servidores; ativações trafegam pela rede em cadeia (pipeline paralelo)."),
        (1, "KV cache fica no servidor — sessões de geração são stateful."),
        (0, "Petals + Hivemind"),
        (1, "DHT (Kademlia sobre libp2p): cada servidor anuncia os blocos que hospeda; clientes descobrem a rota consultando a tabela distribuída."),
        (1, "RemoteSequenceManager monta a rota e a reconstrói em caso de falha (banir peer, reconsultar DHT)."),
        (0, "Trade-off latência × qualidade"),
        (1, "Precisão (fp32 vs bf16), tamanho do modelo (560M vs 1.7B) e posição na rede (LAN local vs rede corporativa) afetam ambos os eixos."),
        (1, "Fronteira de Pareto: conjunto de rotas não dominadas — melhorar um eixo sem piorar o outro."),
    ],
)

# ---------------------------------------------------- Trabalhos Relacionados
slide_conteudo(
    "Trabalhos Relacionados",
    [
        (0, "Inferência distribuída colaborativa"),
        (1, "Petals (Borzunov et al., 2023) — inferência e fine-tuning de BLOOM/Llama sobre swarm voluntário; base deste trabalho."),
        (1, "Learning@home (Ryabinin & Gusev, 2020) e Hivemind (Ryabinin et al., 2023) — DHT e all-reduce descentralizado; petalista de rede do Petals."),
        (0, "Servindo LLMs em cluster"),
        (1, "vLLM (Kwon et al., 2023), FlexGen (Sheng et al., 2023), DeepSpeed-Inference — otimização de memória/throughput, mas com topologia fixa e centralizada."),
        (0, "Roteamento em sistemas distribuídos"),
        (1, "Kademlia (Maymounkov & Mazières, 2002) — DHT por proximidade de ID; Petals reusa a ideia para localizar blocos."),
        (0, "Diferencial deste trabalho"),
        (1, "Decisão por requisição (não por sistema), objetivo declarado pelo cliente, e avaliação da seleção de rota (latência e proxy de qualidade) com hardware doméstico real."),
    ],
)

# ------------------------------------------------------------------ Proposta
slide_conteudo(
    "Proposta",
    [
        (0, "Roteador ciente da requisição (router/), sobre o Petals sem modificar o upstream"),
        (0, "Decisão hierárquica em 2 níveis"),
        (1, "Nível 1 — seleção de modelo: bloom-560m (rápido) vs bloom-1b7 (preciso), por α normalizado ou por restrição (orçamento de latência / piso de qualidade)."),
        (1, "Nível 2 — seleção de blocos/réplicas: grafo de inferência com arestas (nó, blocos, precisão) e pesos de latência e qualidade separados; custo = (1−α)·latência + α·qualidade; caminho mínimo (Dijkstra)."),
        (1, "Escalarização multi-objetivo: soma ponderada é um dial; limitação conhecida — não alcança regiões não-convexas da fronteira. Objetivos restritos (orçamento de latência / piso de qualidade) atuam como ε-restrição sobre o mesmo grafo."),
        (0, "Objetivo via ContextVar — atravessa a pilha do Petals (InferenceSession não tem parâmetro de objetivo)."),
        (0, "Modelo de qualidade"),
        (1, "Tabela de perplexidade por (modelo, precisão), medida em WikiText-2; avaliação online: NLL do texto gerado vs referência fp32 local, concordância de tokens, acurácia LAMBADA."),
        (0, "Falhas herdadas do Petals sem alteração — o roteador só muda a escolha da rota."),
    ],
)

# --------------------------------------------------------------- Metodologia
slide_conteudo(
    "Metodologia",
    [
        (0, "Testbed: 3 hosts heterogêneos em LAN, 12 nodos virtuais (Docker)"),
        (1, "Mac M4 Pro (24 GB) — camada rápida; Lenovo T490 — bootstrap DHT + registry; Lenovo T14 (rede corporativa, via WSL2) — camada fraca/distante."),
        (1, "bloom-560m em 24 blocos, réplicas em fp32 e bf16 em hosts diferentes; bloom-1b7 como camada precisa."),
        (0, "Métricas por requisição (CSV): TTFT, tok/s, overhead de decisão, rota composta, bytes de ativação/token, NLL vs fp32, acurácia."),
        (0, "Experimentos"),
        (1, "1. Varredura de α (0→1) — custo de rota vs linhas de base (padrão Petals, sempre-rápido, sempre-preciso, aleatório)."),
        (1, "2. Degradação de rede (tc netem: atraso + perda) — sensibilidade da decisão."),
        (1, "3. Carga mista concorrente (requisições de latência e de qualidade ao mesmo tempo)."),
        (1, "4. Injeção de falha em plena geração (derrubar nó da rota) — re-roteamento pela DHT."),
        (1, "5. Acurácia em tarefa real (LAMBADA) por objetivo, vs linha de base."),
    ],
)

# ---------------------------------------------------------------- Resultados 1
slide_tabela(
    "Resultados 1/3 — Varredura de α (custo de rota)",
    ["α", "spans", "tok/s", "TTFT (s)", "ppl prevista", "bytes/token"],
    R_EXP1,
    nota="α=0 (só latência): rota de 5 spans, tudo em hosts locais rápidos. α=1 (só qualidade): roteador inclui span fp32 do T14 — perplexidade cai (22,075 → 22,025) ao custo de tok/s menor e mais bytes de ativação. Overhead de decisão: 0,2–0,6 ms (desprezível).",
)

# ---------------------------------------------------------------- Resultados 2
slide_conteudo(
    "Resultados 2/3 — Rede, carga mista e tarefa",
    [
        (0, "Degradação de rede (netem no T14)"),
        (1, R_EXP2),
        (0, "Carga mista concorrente"),
        (1, R_EXP3),
        (0, "Acurácia em tarefa (LAMBADA, última palavra, 50 exemplos)"),
        (1, R_EXP5),
    ],
)

# ---------------------------------------------------------------- Resultados 3
slide_conteudo(
    "Resultados 3/3 — Injeção de falha em plena geração",
    [
        (0, "Protocolo: derrubar (docker stop) um nó da rota no meio da geração"),
        (0, "Linha do tempo medida (fault-injection-final.log)"),
        (1, "Detecção instantânea: ConnectionResetError capturado no mesmo segundo da queda."),
        (1, "Janela de registro obsoleto (~5 min): DHT continua anunciando o nó morto; backoff de repetição 1→2→4→8→16→32→60 s.", CINZA),
        (1, "Re-roteamento pela DHT funciona no nível do sistema: o span perdido volta a ser coberto por réplicas ativas."),
        (1, "Limitação: a sessão interrompida não migra — o KV cache fica no servidor e morre com o nó (asserção de comprimento em cache frio); requisições novas se recuperam.", VERMELHO),
        (0, "Conclusão parcial"),
        (1, "O swarm sobrevive; a requisição em voo não. Reenviar o prefixo ao falhar é a extensão natural do roteador ciente da requisição.", VERDE),
    ],
)

# ------------------------------------------------------------------ Conclusão
slide_conteudo(
    "Conclusão",
    [
        (0, "Reproduzimos o Petals em um swarm doméstico real (3 hosts, 12 nodos virtuais) e estendemos o roteamento do cliente."),
        (0, "Contribuições"),
        (1, "Roteador ciente da requisição em 2 níveis (modelo, depois blocos/réplicas/precisão/nó) com objetivo declarado por requisição."),
        (1, "Modelo de qualidade medido (perplexidade por modelo/precisão) integrado ao custo de roteamento."),
        (1, "Instrumentação por requisição + 5 experimentos executados em hardware heterogêneo."),
        (1, "Alinhamento com aprendizado distribuído: o mesmo roteador atende inferência e fine-tuning distribuído do Petals (autograd sequencial); no FL trocam-se gradientes (paralelismo de dados), no Petals trocam-se ativações em cadeia (pipeline)."),
        (0, "Resultados principais"),
        (1, "A rota muda com α: α=1 inclui o span fp32 do T14 (ppl prevista 22,075 → 22,025) ao custo de tok/s; qualidade é proxy — saídas equivalentes (mesmo modelo); overhead ~0,5 ms."),
        (1, "Falhas: detecção instantânea e re-cobertura automática do span, mas sessões em voo não migram (KV cache no servidor)."),
        (0, "Trabalho futuro"),
        (1, "Reenvio de prefixo no failover; medição de qualidade online no roteador; testbed com GPU."),
    ],
)

# ---------------------------------------------------------------- Referências
slide_conteudo(
    "Referências (formato SBC)",
    [
        (1, "BORZUNOV, A.; MAXIMOV, V.; RYABININ, M. et al. Petals: Collaborative Inference and Fine-tuning of Large Models. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023), 2023."),
        (1, "RYABININ, M.; BORZUNOV, A.; GORBUNOV, A. et al. Distributed Deep Learning in Open Collaborations. In: Advances in Neural Information Processing Systems (NeurIPS 2023), 2023."),
        (1, "RYABININ, M.; GUSEV, A. Learning@home: Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts. In: NeurIPS Workshop on Distributed Machine Learning, 2020."),
        (1, "MAYMOUNKOV, P.; MAZIÈRES, D. Kademlia: A Peer-to-peer Information System Based on the XOR Metric. In: IPTPS, 2002."),
        (1, "Vaswani, A. et al. Attention Is All You Need. In: NeurIPS, 2017."),
        (1, "LE SCAO, T. et al. BLOOM: A 176B-Parameter Open-Access Multilingual Language Model. arXiv:2211.05100, 2022."),
        (1, "KWON, W. et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. In: SOSP, 2023."),
        (1, "SHENG, Y. et al. FlexGen: High-Throughput Generative Inference on Single GPU. In: ICML, 2023."),
        (1, "Papamartzivanos, D. et al. LAMBADA (Linguistic Analysis of Multi-word Ambiguities in Discourse) — Evaluation of Language Models on Broad-Context Reasoning. ACL 2016."),
    ],
)

out = os.path.join(os.path.dirname(__file__) or ".", "apresentacao_final.pptx")
prs.save(out)
print(f"OK: {out}")
