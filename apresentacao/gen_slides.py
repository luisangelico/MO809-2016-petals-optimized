# -*- coding: utf-8 -*-
"""Gera a apresentacao de proposta (MO809A) — 1 slide por pergunta."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

AZUL = RGBColor(0x1F, 0x3A, 0x5F)
CINZA = RGBColor(0x44, 0x44, 0x44)
CLARO = RGBColor(0xF2, 0xF4, 0xF7)
BRANCO = RGBColor(0xFF, 0xFF, 0xFF)

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
    s = prs.slides.add_slide(BLANK)
    faixa_titulo(s, titulo)
    cx = s.shapes.add_textbox(Inches(0.7), Inches(1.5), Inches(12.0), Inches(5.6))
    tf = cx.text_frame
    tf.word_wrap = True
    primeiro = True
    for nivel, texto in itens:
        p = tf.paragraphs[0] if primeiro else tf.add_paragraph()
        primeiro = False
        p.level = nivel
        p.space_after = Pt(8)
        marcador = "-" if nivel == 0 else "."
        r = p.add_run()
        r.text = f"{marcador} {texto}"
        r.font.size = Pt(20) if nivel == 0 else Pt(16)
        r.font.color.rgb = CINZA if nivel else AZUL
        if nivel == 0:
            r.font.bold = True
    return s


# ---------- Slide de titulo ----------
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
r.text = "Otimizador de Rotas Ciente da Requisicao para Inferencia Distribuida de LLMs"
r.font.size = Pt(36)
r.font.bold = True
r.font.color.rgb = BRANCO
p2 = tf.add_paragraph()
p2.space_before = Pt(14)
r2 = p2.add_run()
r2.text = "Reproducao do Petals em rede domestica heterogenea"
r2.font.size = Pt(22)
r2.font.color.rgb = RGBColor(0xCF, 0xDA, 0xE8)
cx2 = s.shapes.add_textbox(Inches(0.9), Inches(5.6), Inches(11.5), Inches(1.4))
tf2 = cx2.text_frame
for linha, tam in [
    ("MO809A - Sistemas Distribuidos | Proposta de Projeto Final", 18),
    ("Luis Henrique Angelico | Outubro de 2026", 16),
]:
    pp = tf2.paragraphs[0] if linha.startswith("MO809A") else tf2.add_paragraph()
    rr = pp.add_run()
    rr.text = linha
    rr.font.size = Pt(tam)
    rr.font.color.rgb = BRANCO

# ---------- 1. Motivacao e contextualizacao ----------
slide_conteudo("Qual a motivacao e contextualizacao do trabalho?", [
    (0, "LLMs com bilhoes de parametros nao cabem na memoria de uma maquina comum."),
    (0, "Petals distribui os blocos do transformer entre nos voluntarios heterogeneos (estilo BitTorrent), sobre a biblioteca Hivemind."),
    (0, "Num swarm real, os nos diferem em velocidade, precisao e confiabilidade."),
    (0, "As requisicoes tambem diferem em necessidade:"),
    (1, "chat/interativo exige baixa latencia;"),
    (1, "sumarizacao/raciocinio tolera mais latencia em troca de mais qualidade."),
    (0, "O roteamento do Petals escolhe servidores apenas por vazao/ping: otimiza latencia e trata toda requisicao igual."),
    (0, "Oportunidade: um escalonamento de rota ciente da requisicao."),
])

# ---------- 2. Problema investigado ----------
slide_conteudo("Qual o problema investigado?", [
    (0, "Dado um swarm heterogeneo com blocos replicados em nos de perfis distintos (velocidade e qualidade)..."),
    (0, "...e um fluxo de requisicoes com objetivos diferentes (sensiveis a latencia vs. sensiveis a qualidade):"),
    (1, "Como selecionar, por requisicao, o conjunto de nos/replicas que melhor atende ao objetivo?"),
    (1, "Minimizar latencia respeitando um piso de qualidade, ou"),
    (1, "Maximizar qualidade dentro de um orcamento de latencia."),
    (0, "O Petals padrao nao expoe nem otimiza esse compromisso (roteamento so por latencia, agnostico a requisicao)."),
])

# ---------- 3. Objetivo ----------
slide_conteudo("Qual o objetivo do projeto?", [
    (0, "Projetar e implementar um otimizador de rota ciente da requisicao sobre o Petals."),
    (0, "Decidir, por requisicao, quais nos/replicas usar conforme um objetivo declarado (velocidade vs. acuracia)."),
    (0, "Caracterizar empiricamente o compromisso latencia x qualidade (fronteira de Pareto), comparando com o roteamento padrao do Petals."),
    (0, "Avaliar o sistema sob a otica de sistemas distribuidos: heterogeneidade, stragglers, sobrecarga de comunicacao e tolerancia a falhas."),
])

# ---------- 4. O que sera desenvolvido ----------
slide_conteudo("O que sera desenvolvido?", [
    (0, "Um swarm heterogeneo com replicas em tres eixos:"),
    (1, "Qualidade (eixo forte): camadas de modelo - bloom-560m (rapido) vs bloom-1b7 (acurado)."),
    (1, "Qualidade (eixo fino): precisao numerica - fp32 vs bf16/fp16."),
    (1, "Latencia: nos fisicos - Mac M4 Pro (MPS, rapido) vs T490 (CPU, lento) + stragglers induzidos."),
    (0, "Politica de roteamento customizada estendendo o RemoteSequenceManager do Petals:"),
    (1, "funcao de custo combinando latencia/vazao medidas e um escore de qualidade;"),
    (1, "parametrizada por objetivo (peso alfa ou restricao: SLA de latencia / piso de qualidade)."),
    (0, "Decisao hierarquica: (1) camada de modelo por requisicao; (2) replica por bloco (precisao/no)."),
    (0, "Instrumentacao: rota escolhida, latencia (TTFT, tok/s), bytes de ativacao e qualidade."),
])

# ---------- 5. Como sera avaliado ----------
slide_conteudo("Como sera avaliado? Quais metricas?", [
    (0, "Metricas:"),
    (1, "Latencia: tempo ate o primeiro token (TTFT) e tokens/s (lote unico e concorrente);"),
    (1, "Qualidade: perplexidade, divergencia da saida vs. referencia em fp32, acuracia em tarefa pequena (ex.: LAMBADA/QA);"),
    (1, "Comunicacao: bytes de ativacao por token; composicao da rota."),
    (0, "Baselines: roteamento padrao do Petals; sempre-rapido; sempre-acurado; aleatorio."),
    (0, "Experimentos:"),
    (1, "fronteira de Pareto variando o objetivo (alfa);"),
    (1, "heterogeneidade e stragglers via tc netem (T490) / Network Link Conditioner (Mac);"),
    (1, "cargas mistas de requisicoes; injecao de falha (derrubar no -> re-roteamento pela DHT)."),
    (0, "Infra: swarm de 2 nos na LAN domestica; stack Petals/Hivemind fixada e reproduzivel."),
])

# ---------- 6. Resultados esperados ----------
slide_conteudo("Quais os resultados esperados?", [
    (0, "O otimizador tracando uma fronteira de Pareto que domina as politicas ingenuas:"),
    (1, "requisicoes sensiveis a latencia: perto do sempre-rapido, com perda de qualidade limitada;"),
    (1, "requisicoes sensiveis a qualidade: perto do sempre-acurado, com latencia menor que a ingenua."),
    (0, "Adaptacao demonstravel a heterogeneidade e a stragglers (evita nos lentos quando a latencia importa)."),
    (0, "Overhead baixo do proprio otimizador; quantificacao do compromisso atingivel em escala domestica."),
    (0, "Resultado preliminar ja obtido (viabilidade):"),
    (1, "swarm de 2 nos no ar; primeira inferencia distribuida ponta a ponta com bloom-560m;"),
    (1, "forward atravessando T490 (blocos 0:12) -> Mac (blocos 12:24); ~14,7 tokens/s em CPU/LAN."),
])

OUT = "/Users/luishenriqueangelico/MO809A/apresentacao/MO809A-apresentacao-proposta.pptx"
prs.save(OUT)
print("salvo:", OUT, "| slides:", len(prs.slides._sldIdLst))
