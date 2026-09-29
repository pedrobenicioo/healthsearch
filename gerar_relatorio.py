# Autor: PEDRO HENRIQUE BENICIO DE OLIVEIRA | RGM: 33602697
"""Gera RELATORIO.pdf (máx. 2 páginas) do HealthSearch, com gráfico de comparação de ranks.
Uso: python gerar_relatorio.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors

import healthsearch_app as h

AQUI = os.path.dirname(os.path.abspath(__file__))
CONSULTAS = ["ataque cardíaco", "CÓD-ECG-12D", "AAS 100mg"]


def grafico():
    fig, axes = plt.subplots(1, len(CONSULTAS), figsize=(11, 3.2), sharey=True)
    ids = [d["id"] for d in h.CORPUS]
    x = np.arange(len(ids))
    for ax, q in zip(axes, CONSULTAS):
        df, _ = h.montar_tabela(q, 1.2, 0.75, 0.5, True)
        for i, (col, cor) in enumerate([("Rank BM25", "#d95f02"), ("Rank Semântico", "#1b9e77"), ("Rank RRF", "#3b5bdb")]):
            ax.bar(x + (i - 1) * 0.27, df[col], 0.27, label=col, color=cor)
        ax.set_xticks(x, [i.replace("Doc ", "D") for i in ids])
        ax.set_title(f'"{q}"', fontsize=10)
        ax.invert_yaxis()
    axes[0].set_ylabel("Posição (1 = melhor)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    caminho = os.path.join(AQUI, "comparacao_ranks.png")
    fig.savefig(caminho, dpi=160)
    return caminho


def main():
    img = grafico()
    ss = getSampleStyleSheet()
    corpo = ParagraphStyle("c", parent=ss["BodyText"], fontSize=9.5, leading=12.5)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=6, spaceAfter=3)
    doc = SimpleDocTemplate(os.path.join(AQUI, "RELATORIO.pdf"), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm, topMargin=1.6*cm, bottomMargin=1.6*cm)
    P = lambda t, s=corpo: Paragraph(t, s)
    el = [
        P("HealthSearch — Motor de Busca Híbrido (BM25 + Semântico + RRF)", ss["Title"]),
        P("<b>Autor: PEDRO HENRIQUE BENICIO DE OLIVEIRA — RGM: 33602697</b>", corpo),
        P("Relatório técnico • UNIPÊ — Tendências em Ciência da Computação", corpo),
        P("1. Arquitetura da solução", h2),
        P("O aplicativo Streamlit (<b>healthsearch_app.py</b>) segue quatro fases. <b>(1) Pré-processamento:</b> "
          "minúsculas, remoção de acentos e caracteres especiais, stopwords em português; códigos como "
          "<i>CÓD-ECG-12D</i> são mantidos como token único e também divididos em partes. "
          "<b>(2) BM25:</b> Okapi BM25 (rank_bm25) recalculado a cada movimento dos sliders "
          "k₁ (0–3, padrão 1,2) e b (0–1, padrão 0,75). <b>(3) Semântico:</b> embeddings "
          "<i>paraphrase-multilingual-MiniLM-L12-v2</i> normalizados; similaridade de cosseno entre consulta e "
          "documentos, com simulação vetorial documentada (bag-of-concepts) como contingência offline. "
          "<b>(4) RRF:</b> Score = α·1/(60+rank_BM25) + (1−α)·1/(60+rank_sem), com α ajustável. "
          "A interface tem abas Léxico, Semântico, Híbrido RRF e Matriz Comparativa; o bônus aplica "
          "Cross-Encoder (ms-marco-MiniLM-L-6-v2) sobre o Top-3 híbrido, exibindo a variação de posição."),
        P("2. Comparação de ranks", h2),
        Image(img, width=17*cm, height=17*cm*3.2/11),
        P("A busca por <i>ataque cardíaco</i> não compartilha tokens com o corpus (que usa \"infarto\" e "
          "\"síndrome coronariana\"): todos os scores BM25 são 0 e a ordem é mero desempate, enquanto o semântico "
          "já ordena por proximidade de significado. Para <i>CÓD-ECG-12D</i> o BM25 recupera exatamente os Docs 1 e 6 "
          "(scores ≈ 2,4 e 2,3; demais 0), ao passo que o semântico distribui a similaridade por documentos "
          "cardíacos genéricos. Documentos com BM25 = 0 não contribuem com a parcela léxica do RRF, evitando que "
          "um desempate arbitrário influencie a fusão. Em <i>AAS 100mg</i> a sigla não aparece no corpus (que usa "
          "\"ácido acetilsalicílico\"): o BM25 é cego e o modelo semântico é limitado, o que ilustra a necessidade "
          "de expansão de sinônimos ou de um modelo clínico em português. O RRF combina os sinais sem normalizar scores."),
        P("3. Observações", h2),
        P("• k₁ alto valoriza repetição de termos; b alto penaliza documentos longos — com apenas 6 documentos "
          "curtos o efeito é pequeno, mas visível nos scores. • α=1 reproduz o BM25 e α=0 o semântico, "
          "servindo de diagnóstico. • O RRF usa apenas posições, portanto é robusto à diferença de escala entre "
          "scores BM25 (não limitado) e cosseno (−1 a 1)."),
        P("4. Divisão de tarefas da equipe", h2),
    ]
    tabela = Table([
        ["Integrante", "Responsabilidade"],
        ["PEDRO HENRIQUE BENICIO DE OLIVEIRA (RGM 33602697)", "Fase 1 e 2: pré-processamento, BM25 e sliders"],
        ["Aluno 2", "Fase 3: embeddings, cosseno e bônus Cross-Encoder"],
        ["Aluno 3", "Fase 4: RRF, interface em abas e relatório"],
    ], colWidths=[3.5*cm, 13.5*cm])
    tabela.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    el += [tabela, Spacer(1, 4), P("<i>Substitua \"Aluno 1/2/3\" pelos nomes da equipe.</i>", corpo)]
    doc.build(el)
    print("RELATORIO.pdf gerado")


if __name__ == "__main__":
    main()



