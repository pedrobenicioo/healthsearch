# Autor: PEDRO HENRIQUE BENICIO DE OLIVEIRA | RGM: 33602697
"""
HealthSearch — Motor de Busca Híbrido (BM25 + Semântico) com Reciprocal Rank Fusion.

Execução:  streamlit run healthsearch_app.py

Fases:
  1. Ingestão do corpus e pré-processamento (minúsculas, remoção de especiais, stopwords PT)
  2. Motor léxico Okapi BM25 com k1 e b ajustáveis
  3. Motor semântico vetorial (sentence-transformers; fallback documentado offline)
  4. Fusão RRF:  Score = a/(k+rank_bm25) + (1-a)/(k+rank_sem),  k = 60
Bônus: re-ranking com Cross-Encoder sobre o Top-3 híbrido.
"""
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

K_RRF = 60
MODELO_EMB = "paraphrase-multilingual-MiniLM-L12-v2"
MODELO_CROSS = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ----------------------------------------------------------------------------
# Corpus (hardcode obrigatório)
# ----------------------------------------------------------------------------
CORPUS = [
    {"id": "Doc 1", "titulo": "Protocolo Emergência ECG",
     "texto": "Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos."},
    {"id": "Doc 2", "titulo": "Guia de Farmacologia Cardíaca",
     "texto": "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto agudo do miocárdio."},
    {"id": "Doc 3", "titulo": "Diretriz de Hipertensão Arterial",
     "texto": "A crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial na UTI."},
    {"id": "Doc 4", "titulo": "Manual de AVC Isquêmico",
     "texto": "O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas."},
    {"id": "Doc 5", "titulo": "Protocolo de Reanimação RCR",
     "texto": "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul."},
    {"id": "Doc 6", "titulo": "Procedimentos de UTI Geral",
     "texto": "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria."},
]

# Sinônimos médicos usados APENAS na simulação vetorial (fallback sem modelo).
# Cada grupo é mapeado para um mesmo "conceito" (dimensão) do espaço simulado.
CONCEITOS_SIMULADOS = {
    "cardiaco": ["infarto", "ataque", "cardiaco", "coracao", "coronariana", "miocardio", "isquemia",
                 "precordial", "acetilsalicilico", "aas", "antiagregantes", "arritmias", "cardiaca"],
    "ecg": ["eletrocardiograma", "ecg", "exame", "telemetria", "monitorizacao"],
    "pressao": ["hipertensiva", "hipertensao", "pressao", "arterial", "anti", "hipertensivos"],
    "avc": ["avc", "cerebral", "vascular", "acidente", "isquemico", "trombolíticos", "trombioliticos", "derrame"],
    "parada": ["parada", "cardiorrespiratoria", "reanimacao", "rcr", "compressoes", "desfibrilacao", "azul"],
    "uti": ["uti", "intensiva", "continuo", "monitoramento", "monitorizacao"],
    "urgencia": ["agudo", "aguda", "imediato", "emergencia", "severa", "precoce"],
}

STOPWORDS_PT = set("""
a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre entre até ate e ou mas que se
ao aos à às pelo pela pelos pelas seu sua seus suas ele ela eles elas isso isto esse essa esses essas este esta estes
estas como mais menos muito muita já ja não nao são sao é foi ser está estao estão há ha ter tem têm the
""".split())


# ----------------------------------------------------------------------------
# Fase 1 — pré-processamento
# ----------------------------------------------------------------------------
def remover_acentos(txt: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", txt) if unicodedata.category(c) != "Mn")


def preprocessar(texto: str) -> list[str]:
    """Minúsculas -> remove acentos/especiais -> tokeniza -> remove stopwords.
    Códigos como 'CÓD-ECG-12D' são preservados como termo único ('cod-ecg-12d') e também
    divididos em partes, para casar buscas por 'ECG-12D' ou 'ecg'."""
    txt = remover_acentos(texto.lower())
    txt = re.sub(r"[^a-z0-9\-\s]", " ", txt)
    tokens = []
    for tok in txt.split():
        tok = tok.strip("-")
        if not tok or tok in STOPWORDS_PT or remover_acentos(tok) in STOPWORDS_PT:
            continue
        tokens.append(tok)
        if "-" in tok:  # código composto: adiciona também as partes
            tokens.extend(p for p in tok.split("-") if p and p not in STOPWORDS_PT)
    return tokens


# ----------------------------------------------------------------------------
# Fase 2 — BM25
# ----------------------------------------------------------------------------
def ranking_bm25(consulta: str, docs_tok: list[list[str]], k1: float, b: float) -> np.ndarray:
    bm25 = BM25Okapi(docs_tok, k1=k1, b=b)
    return np.array(bm25.get_scores(preprocessar(consulta)))


# ----------------------------------------------------------------------------
# Fase 3 — Semântico
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def carregar_modelo():
    """Tenta carregar o sentence-transformers; se indisponível (offline), retorna None
    e o app usa a simulação vetorial documentada (CONCEITOS_SIMULADOS)."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODELO_EMB)
    except Exception:
        return None


@st.cache_resource(show_spinner="Carregando Cross-Encoder...")
def carregar_cross():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(MODELO_CROSS)
    except Exception:
        return None


def embed_simulado(textos: list[str]) -> np.ndarray:
    """Simulação vetorial: bag-of-concepts. Cada dimensão = um grupo de sinônimos médicos."""
    conceitos = {k: {remover_acentos(w) for w in v} for k, v in CONCEITOS_SIMULADOS.items()}
    mat = np.zeros((len(textos), len(conceitos)))
    for i, t in enumerate(textos):
        toks = set(re.findall(r"[a-z0-9]+", remover_acentos(t.lower())))
        for j, grupo in enumerate(conceitos.values()):
            mat[i, j] = len(toks & grupo)
    return mat


@st.cache_data(show_spinner=False)
def embeddings_corpus(usar_modelo: bool) -> np.ndarray:
    textos = [d["texto"] for d in CORPUS]
    modelo = carregar_modelo() if usar_modelo else None
    if modelo is not None:
        return modelo.encode(textos, normalize_embeddings=True)
    return embed_simulado(textos)


def ranking_semantico(consulta: str, usar_modelo: bool) -> tuple[np.ndarray, str]:
    docs = embeddings_corpus(usar_modelo)
    modelo = carregar_modelo() if usar_modelo else None
    if modelo is not None:
        q = modelo.encode([consulta], normalize_embeddings=True)[0]
        return docs @ q, f"sentence-transformers ({MODELO_EMB})"
    q = embed_simulado([consulta])[0]
    denom = np.linalg.norm(docs, axis=1) * (np.linalg.norm(q) or 1)
    denom[denom == 0] = 1
    return (docs @ q) / denom, "simulação vetorial (bag-of-concepts)"


# ----------------------------------------------------------------------------
# Fase 4 — RRF
# ----------------------------------------------------------------------------
def posicoes(scores: np.ndarray) -> np.ndarray:
    """Rank 1 = melhor score. Empates recebem posições distintas por ordem estável."""
    ordem = np.argsort(-scores, kind="stable")
    rank = np.empty(len(scores), dtype=int)
    rank[ordem] = np.arange(1, len(scores) + 1)
    return rank


def rrf(rank_bm25: np.ndarray, rank_sem: np.ndarray, alpha: float, k: int = K_RRF,
        mask_bm25: np.ndarray | None = None) -> np.ndarray:
    """Score_RRF = a/(k+rank_bm25) + (1-a)/(k+rank_sem).
    mask_bm25 (opcional): documentos com BM25 = 0 não casaram nenhum termo; seu 'rank' é apenas
    desempate arbitrário, então não contribuem com a parcela léxica."""
    parcela_lex = 1.0 / (k + rank_bm25)
    if mask_bm25 is not None:
        parcela_lex = np.where(mask_bm25, parcela_lex, 0.0)
    return alpha * parcela_lex + (1 - alpha) * (1.0 / (k + rank_sem))


def montar_tabela(consulta, k1, b, alpha, usar_modelo):
    docs_tok = [preprocessar(d["texto"] + " " + d["titulo"]) for d in CORPUS]
    s_bm25 = ranking_bm25(consulta, docs_tok, k1, b)
    s_sem, motor = ranking_semantico(consulta, usar_modelo)
    r_bm25, r_sem = posicoes(s_bm25), posicoes(s_sem)
    s_rrf = rrf(r_bm25, r_sem, alpha, mask_bm25=s_bm25 > 0)
    df = pd.DataFrame({
        "ID": [d["id"] for d in CORPUS],
        "Título": [d["titulo"] for d in CORPUS],
        "Score BM25": s_bm25, "Rank BM25": r_bm25,
        "Score Semântico": s_sem, "Rank Semântico": r_sem,
        "Score RRF": s_rrf,
    })
    df["Rank RRF"] = posicoes(df["Score RRF"].values)
    return df, motor


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="HealthSearch", page_icon="🩺", layout="wide")
    st.title("🩺 HealthSearch — Busca Híbrida BM25 + Semântica (RRF)")
    st.caption("HealthTech Solutions • protocolos de triagem e emergência")

    with st.sidebar:
        st.header("⚙️ Calibração")
        k1 = st.slider("k₁ — saturação de frequência", 0.0, 3.0, 1.2, 0.05)
        b = st.slider("b — normalização por tamanho", 0.0, 1.0, 0.75, 0.05)
        alpha = st.slider("α — peso BM25 na fusão RRF", 0.0, 1.0, 0.5, 0.05,
                          help="α=1 só BM25; α=0 só semântico")
        top_n = st.slider("Resultados exibidos", 1, 6, 6)
        usar_modelo = st.checkbox("Usar modelo sentence-transformers", value=True,
                                  help="Desmarque para usar a simulação vetorial offline.")
        cross_on = st.checkbox("🎯 Bônus: re-ranking Cross-Encoder (Top-3 RRF)")
        st.markdown(f"**k_RRF** = {K_RRF} (fixo)")
        st.divider()
        st.markdown("Exemplos:")
        exemplos = ["ataque cardíaco", "CÓD-ECG-12D", "AAS 100mg", "derrame cerebral", "parada cardíaca"]
        for ex in exemplos:
            if st.button(ex, use_container_width=True):
                st.session_state["consulta"] = ex

    consulta = st.text_input("Consulta clínica", key="consulta", placeholder="Ex.: infarto, ECG-12D, AAS 100mg")
    if not consulta.strip():
        st.info("Digite uma consulta ou escolha um exemplo na barra lateral.")
        st.subheader("Corpus carregado")
        st.dataframe(pd.DataFrame(CORPUS), use_container_width=True, hide_index=True)
        return

    df, motor = montar_tabela(consulta, k1, b, alpha, usar_modelo)
    st.caption(f"Motor semântico: **{motor}** • tokens da consulta: `{preprocessar(consulta)}`")

    tab_lex, tab_sem, tab_hib, tab_mat = st.tabs(
        ["📝 Léxico (BM25)", "🧠 Semântico", "🔀 Híbrido RRF", "📊 Matriz Comparativa"])
    corpus_txt = {d["id"]: d["texto"] for d in CORPUS}

    def listar(sub, col_score, col_rank):
        for _, r in sub.head(top_n).iterrows():
            with st.container(border=True):
                st.markdown(f"**#{int(r[col_rank])} — {r['ID']} · {r['Título']}** — score `{r[col_score]:.4f}`")
                st.write(corpus_txt[r["ID"]])

    with tab_lex:
        st.markdown("Ranking por **Okapi BM25** (casamento exato de termos).")
        listar(df.sort_values("Rank BM25"), "Score BM25", "Rank BM25")
    with tab_sem:
        st.markdown("Ranking por **similaridade de cosseno** entre embeddings.")
        listar(df.sort_values("Rank Semântico"), "Score Semântico", "Rank Semântico")
    with tab_hib:
        st.markdown(r"$Score_{RRF}(D)=\alpha\cdot\frac{1}{60+Rank_{BM25}}+(1-\alpha)\cdot\frac{1}{60+Rank_{Sem}}$")
        hib = df.sort_values("Rank RRF").reset_index(drop=True)
        if cross_on:
            cross = carregar_cross()
            if cross is None:
                st.warning("Cross-Encoder indisponível (sem rede/modelo). Bônus não aplicado.")
            else:
                top3 = hib.head(3).copy()
                pares = [(consulta, corpus_txt[i]) for i in top3["ID"]]
                top3["Nota Cross-Encoder"] = cross.predict(pares)
                top3["Rank pós Cross"] = posicoes(top3["Nota Cross-Encoder"].values)
                top3["Variação"] = top3["Rank RRF"].rank(method="first").astype(int) - top3["Rank pós Cross"]
                st.subheader("Re-ranking Top-3")
                st.dataframe(top3[["ID", "Título", "Score RRF", "Rank RRF", "Nota Cross-Encoder",
                                   "Rank pós Cross", "Variação"]], hide_index=True, use_container_width=True)
        listar(hib, "Score RRF", "Rank RRF")
    with tab_mat:
        st.markdown("Comparação dos três rankings (posição 1 = melhor).")
        st.dataframe(df.sort_values("Rank RRF"), hide_index=True, use_container_width=True)
        graf = df.set_index("ID")[["Rank BM25", "Rank Semântico", "Rank RRF"]]
        st.markdown("**Gráfico de comparação de ranks** (barras menores = melhor posição)")
        st.bar_chart(graf)
        melhor = {c: df.loc[df[c].idxmin(), "ID"] for c in ["Rank BM25", "Rank Semântico", "Rank RRF"]}
        c1, c2, c3 = st.columns(3)
        c1.metric("Top-1 BM25", melhor["Rank BM25"])
        c2.metric("Top-1 Semântico", melhor["Rank Semântico"])
        c3.metric("Top-1 Híbrido", melhor["Rank RRF"])


if __name__ == "__main__":
    main()

