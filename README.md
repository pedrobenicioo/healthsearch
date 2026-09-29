# 🩺 HealthSearch — Busca Híbrida BM25 + Semântica (RRF)

Motor de busca para protocolos médicos que combina **Okapi BM25** (precisão em termos/códigos exatos) e **busca semântica vetorial** (sinônimos médicos) por meio de **Reciprocal Rank Fusion**, em um app Streamlit.

```
Score_RRF(D) = α · 1/(60 + Rank_BM25) + (1 − α) · 1/(60 + Rank_Semântico)
```

## Executar

```bash
pip install -r requirements.txt
streamlit run healthsearch_app.py
```

Na primeira execução o modelo `paraphrase-multilingual-MiniLM-L12-v2` é baixado. Sem internet, desmarque *Usar modelo sentence-transformers* para usar a simulação vetorial offline.

## Recursos

- Pré-processamento: minúsculas, remoção de acentos/especiais e stopwords em português; códigos como `CÓD-ECG-12D` são preservados.
- Sliders: **k₁** (0–3), **b** (0–1) e **α** (0–1).
- Abas: Léxico (BM25), Semântico, Híbrido RRF e Matriz Comparativa.
- Bônus: re-ranking com Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) no Top-3 híbrido.

## Consultas sugeridas

| Consulta | Demonstra |
|---|---|
| `CÓD-ECG-12D` | BM25 acerta o código exato (Docs 1 e 6) |
| `ataque cardíaco` | BM25 cego (sem termos em comum); semântico resolve |
| `derrame cerebral` | sinônimo de AVC |
| `AAS 100mg` | limitação: o corpus usa "ácido acetilsalicílico" |

Teste também α=0 (só semântico) e α=1 (só BM25).

## Relatório

`RELATORIO.pdf` é gerado por `python gerar_relatorio.py`.

---
Autor: **PEDRO HENRIQUE BENICIO DE OLIVEIRA** — RGM: 33602697

